"""
Live Trading con NinjaTrader 8.
Utiliza modelos entrenados para ejecutar operaciones en tiempo real con NinjaTrader 8
comunicándose mediante la estrategia NT8StrategyServer.
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
import logging
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import threading

# Añadir directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar componentes del proyecto
from agents.ppo_agent import PPOAgent
from environment.trading_env import TradingEnv
from config.config import get_config
from data.data_loader import DataLoader
from utils.logger import setup_logger

# Importar el cliente TCP para NinjaTrader
from connector_nt8.ninjatrader_client import NT8Client

# Añadir clase auxiliar para cargar modelos PPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class PPOAgentInference:
    """Versión simplificada de PPOAgent para inferencia."""
    
    def __init__(self, model_path, force_cpu=False):
        """
        Carga un modelo PPO pre-entrenado para inferencia.
        
        Args:
            model_path (str): Ruta al modelo entrenado
            force_cpu (bool): Si es True, fuerza el uso de CPU en lugar de GPU
        """
        self.model_path = model_path
        
        if force_cpu:
            # Forzar CPU para evitar advertencias y problemas con GPU
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            device = "cpu"
            logger.info("Forzando uso de CPU para el modelo")
        else:
            # Debe ser una cadena, no None
            device = "auto"  # Esto usará CUDA si está disponible, de lo contrario CPU
            logger.info("Usando detección automática de dispositivo (GPU/CPU)")
            
        logger.info(f"Cargando modelo desde {model_path}")
        # Proporcionar device como cadena (auto, cpu, cuda, etc.)
        self.model = PPO.load(model_path, device=device)
        logger.info(f"Modelo cargado correctamente en dispositivo: {self.model.device}")
    
    def predict(self, observation, deterministic=True):
        """
        Realiza una predicción usando el modelo cargado.
        
        Args:
            observation: Observación del entorno
            deterministic (bool): Si se debe usar un comportamiento determinista
            
        Returns:
            Tuple: (acción, estado)
        """
        return self.model.predict(observation, deterministic=deterministic)

# Configurar logger
logger = logging.getLogger(__name__)

class NT8Trader:
    """
    Live Trader para NinjaTrader 8.
    Implementa el trading en tiempo real utilizando modelos entrenados con PPO
    y comunicándose con la estrategia NT8StrategyServer.
    """
    
    def __init__(self, 
                 model_path: str, 
                 instrument: str = None,
                 host: str = "localhost",
                 port: int = 5555,
                 quantity: int = 1,
                 config: Optional[Dict[str, Any]] = None,
                 data_path: Optional[str] = None,
                 update_interval: int = 30,
                 min_bars_required: int = 1,
                 cooldown_period: int = 15):
        """
        Inicializa el trader para NinjaTrader 8.
        
        Args:
            model_path (str): Ruta al modelo entrenado (.zip)
            instrument (str, optional): Instrumento a operar (si es None, se detectará automáticamente)
            host (str): Dirección IP o hostname donde se ejecuta NinjaTrader
            port (int): Puerto TCP de la estrategia NT8StrategyServer
            quantity (int): Cantidad de contratos a operar
            config (Dict[str, Any]): Configuración (si es None, se carga la predeterminada)
            data_path (str): Ruta a datos históricos para inicializar el entorno
            update_interval (int): Intervalo de actualización en segundos
            min_bars_required (int): Mínimo número de barras requeridas para empezar a operar
            cooldown_period (int): Período de enfriamiento entre operaciones (segundos)
        """
        self.model_path = model_path
        self.primary_instrument = instrument  # Ahora es opcional
        self.host = host
        self.port = port
        self.quantity = quantity
        self.update_interval = update_interval
        self.min_bars_required = min_bars_required
        self.cooldown_period = cooldown_period
        
        self.running = False
        self.positions = {}  # Diccionario de posiciones por instrumento
        self.last_action_time = {}  # Seguimiento de última acción por instrumento
        self.detected_instruments = set()  # Conjunto de instrumentos detectados
        
        # Cargar configuración si no se proporciona
        self.config = config if config is not None else get_config()
        
        # Crear cliente de NinjaTrader
        self.nt8_client = NT8Client(host=host, port=port)
        
        # Configurar callbacks para eventos del cliente
        self.nt8_client.set_on_bar_data(self._on_bar_data)
        self.nt8_client.set_on_position_update(self._on_position_update)
        self.nt8_client.set_on_order_update(self._on_order_update)
        self.nt8_client.set_on_error(self._on_error)
        
        # Para seguimiento del estado del trading
        self.lock = threading.Lock()
        self.ready_to_trade = False
        self.available_instruments = {}  # Almacena info por instrumento
        
        # Cargar datos iniciales si se proporciona una ruta
        self.data_loader = DataLoader(config=self.config)
        
        if data_path and os.path.exists(data_path):
            self.initial_data = self.data_loader.load_data(data_path)
            logger.info(f"Datos iniciales cargados de {data_path}")
        else:
            # Intentar encontrar datos en ubicaciones comunes
            possible_paths = [
                "./data/processed/training_data.csv",
                "../data/processed/training_data.csv",
                "./data/processed/test_data.csv",
                "../data/processed/test_data.csv",
                "./data/dataset/processed_data.csv",
                "../data/dataset/processed_data.csv"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.initial_data = self.data_loader.load_data(path)
                    logger.info(f"Datos iniciales cargados de {path}")
                    break
            else:
                logger.warning("No se encontraron datos históricos para inicializar. Se usarán sólo datos en vivo.")
                self.initial_data = None
        
        # Comprobar que el modelo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
        
        # Comprobar el archivo de normalización
        self.vec_normalize_path = model_path + "_vecnormalize.pkl"
        if not os.path.exists(self.vec_normalize_path):
            logger.warning(f"No se encontró el archivo de normalización en {self.vec_normalize_path}")
            self.vec_normalize_path = None
        
        # Configurar entorno y modelo
        self._setup_environment_and_model()
    
    def _setup_environment_and_model(self):
        """Configura el entorno de trading y carga el modelo."""
        try:
            logger.info("Configurando entorno de trading")
            
            # Crear DataFrame vacío con las columnas necesarias
            columns = ["timestamp", "open", "high", "low", "close", "volume"]
            if "feature_columns" in self.config:
                for col in self.config["feature_columns"]:
                    if col not in columns:
                        columns.append(col)
            empty_df = pd.DataFrame(columns=columns)
            
            # Crear entorno de trading
            self.env = TradingEnv(
                data=empty_df,
                config=self.config,
                mode="inference"
            )
            
            # Detectar si se debe forzar CPU basado en variable de entorno
            force_cpu = os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
            
            # Cargar el modelo entrenado usando la clase auxiliar
            logger.info(f"Cargando modelo desde {self.model_path}")
            self.agent = PPOAgentInference(self.model_path, force_cpu=force_cpu)
            
            logger.info("Entorno y modelo configurados correctamente")
        except Exception as e:
            logger.error(f"Error al configurar entorno y modelo: {e}")
            import traceback
            logger.error(f"Traceback completo: {traceback.format_exc()}")
            raise
    
    def start(self):
        """Inicia el trading en tiempo real."""
        if self.running:
            logger.warning("El trader ya está ejecutándose")
            return
        
        self.running = True
        self.ready_to_trade = False
        logger.info(f"Iniciando trading en vivo con NinjaTrader 8 en {self.host}:{self.port}")
        
        try:
            # Conectar con el servidor NT8StrategyServer
            if not self.nt8_client.connect():
                logger.error("No se pudo conectar con el servidor NT8StrategyServer")
                self.running = False
                return
            
            logger.info("Conexión establecida correctamente. Esperando recibir datos de los instrumentos...")
            
            # Esperar a tener suficientes datos para operar
            while self.running and not self.ready_to_trade:
                try:
                    # Verificar si tenemos suficientes barras de algún instrumento
                    with self.lock:
                        if len(self.detected_instruments) > 0:
                            for instrument in self.detected_instruments:
                                bar_history = self.nt8_client.get_bar_history(instrument)
                                if len(bar_history) >= self.min_bars_required:
                                    self.ready_to_trade = True
                                    
                                    # Si no se especificó instrumento, usar el primero detectado con suficientes barras
                                    if self.primary_instrument is None:
                                        self.primary_instrument = instrument
                                        logger.info(f"Instrumento principal seleccionado automáticamente: {self.primary_instrument}")
                                    
                                    logger.info(f"Suficientes datos recibidos para {instrument} ({len(bar_history)} barras). Listo para operar.")
                                    
                                    # Actualizar el entorno con los datos históricos
                                    self._update_environment_with_history(instrument)
                                    break
                                else:
                                    logger.info(f"Esperando más datos para {instrument}. Actualmente {len(bar_history)} barras, necesito {self.min_bars_required}.")
                        else:
                            logger.info("Esperando detectar instrumentos disponibles...")
                
                except Exception as e:
                    logger.error(f"Error al verificar datos iniciales: {e}")
                
                # Esperar antes de verificar nuevamente
                time.sleep(5)
            
            # Bucle principal de trading
            last_update_time = datetime.now() - timedelta(seconds=self.update_interval)
            
            while self.running:
                try:
                    # Verificar si es momento de actualizar
                    current_time = datetime.now()
                    if (current_time - last_update_time).total_seconds() >= self.update_interval:
                        # Mostrar estado actual
                        logger.info(f"Instrumentos disponibles: {list(self.detected_instruments)}")
                        
                        # Evaluar modelo para cada instrumento detectado
                        for instrument in self.detected_instruments:
                            if self.primary_instrument is None or instrument == self.primary_instrument:
                                self._evaluate_and_trade(instrument)
                        
                        last_update_time = current_time
                    
                    # Esperar un poco para no sobrecargar la CPU
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error en ciclo de trading: {e}")
                    import traceback
                    logger.error(f"Traceback completo: {traceback.format_exc()}")
                    time.sleep(5)  # Esperar un poco antes de reintentar
            
        except KeyboardInterrupt:
            logger.info("Trading detenido por el usuario")
        finally:
            self.stop()
    
    def stop(self):
        """Detiene el trading y limpia recursos."""
        if not self.running:
            return
        
        self.running = False
        logger.info("Deteniendo trading...")
        
        # Cerrar posiciones antes de salir
        try:
            for instrument in self.detected_instruments:
                try:
                    self.nt8_client.close_position(instrument)
                    logger.info(f"Posición en {instrument} cerrada")
                except Exception as e:
                    logger.error(f"Error al cerrar posición en {instrument}: {e}")
        except Exception as e:
            logger.error(f"Error al cerrar posiciones: {e}")
        
        # Desconectar del servidor
        self.nt8_client.disconnect()
        
        logger.info("Trading detenido correctamente")
    
    def _update_environment_with_history(self, instrument: str) -> Optional[np.ndarray]:
        """
        Actualiza el entorno con datos históricos del instrumento.
        
        Args:
            instrument (str): Instrumento a actualizar
            
        Returns:
            Optional[np.ndarray]: Observación inicial si se actualizó con éxito
        """
        try:
            # Obtener historial de barras para el instrumento
            bar_history = self.nt8_client.get_bar_history(instrument)
            if bar_history.empty:
                logger.warning(f"No hay datos históricos para {instrument}")
                return None
            
            logger.info(f"Obtenido historial para {instrument}: {len(bar_history)} barras")
            
            try:
                # Convertir a DataFrame numérico limpio
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                numeric_data = bar_history[numeric_columns].copy()
                
                # Asegurar que los datos son floats
                for col in numeric_columns:
                    numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
                
                # Establecer índice de tiempo si está disponible
                if 'timestamp' in bar_history.columns:
                    numeric_data.index = bar_history["timestamp"]
                
                logger.info(f"Creado DataFrame numérico limpio para modelo: {numeric_data.shape}")
                logger.debug(f"Columnas originales: {numeric_data.columns.tolist()}")
                logger.debug(f"Tipos de datos del DataFrame numérico: {numeric_data.dtypes}")
                
                # Convertir los datos a la forma que espera el modelo
                # El modelo espera (60, 25) pero recibimos (n, 5) donde n puede ser menor que 60
                # Añadir indicadores técnicos si es necesario para llegar a 25 columnas
                expanded_data = self._expand_data_for_model(numeric_data)
                logger.debug(f"Después de expandir: shape={expanded_data.shape}, columnas={expanded_data.columns.tolist()}")
                
                # **AQUÍ ESTÁ EL ERROR CRÍTICO**: Verificación explícita del número de columnas
                if len(expanded_data.columns) != 25:
                    logger.error(f"ERROR CRÍTICO: El DataFrame expandido tiene {len(expanded_data.columns)} columnas en lugar de 25")
                    # Forzar truncado a 25 columnas como último recurso
                    if len(expanded_data.columns) > 25:
                        logger.warning("Truncando columnas a 25 como último recurso")
                        expanded_data = expanded_data.iloc[:, :25]
                    else:
                        # Añadir columnas faltantes con ceros
                        logger.warning("Añadiendo columnas faltantes con ceros como último recurso")
                        for i in range(len(expanded_data.columns), 25):
                            expanded_data[f'feature_{i}'] = 0.0
                
                # Verificación final obligatoria
                assert len(expanded_data.columns) == 25, f"Error crítico persistente: {len(expanded_data.columns)} columnas en lugar de 25"
                
                # Llamar a la función de actualización del entorno con los datos expandidos
                logger.info(f"Llamando a env.update_data con DataFrame expandido de forma {expanded_data.shape}")
                obs = self.env.update_data(expanded_data)
                logger.info(f"Entorno actualizado con {len(expanded_data)} barras históricas")
                return obs
            except Exception as e:
                logger.error(f"Error al actualizar entorno con datos numéricos: {e}")
                import traceback
                logger.error(f"Traceback completo: {traceback.format_exc()}")
                return None
                
        except Exception as e:
            logger.error(f"Error al actualizar entorno con historial: {e}")
            import traceback
            logger.error(f"Traceback completo: {traceback.format_exc()}")
            return None
    
    def _expand_data_for_model(self, df):
        """Expande los datos para que sean compatibles con el modelo."""
        try:
            if df is None or df.empty:
                logger.warning("DataFrame vacío o None en _expand_data_for_model")
                # Creamos un DataFrame con 25 columnas de ceros con nombres adecuados
                empty_df = pd.DataFrame()
                for i in range(25):
                    if i < 5:
                        cols = ['open', 'high', 'low', 'close', 'volume']
                        empty_df[cols[i]] = [0.0]
                    else:
                        empty_df[f'feature_{i-4}'] = [0.0]
                return empty_df

            logger.debug(f"Expandiendo datos para el modelo, shape inicial: {df.shape}, columnas: {df.columns.tolist()}")
            
            # Aseguramos que tengamos exactamente 25 columnas
            if len(df.columns) < 25:
                logger.warning(f"Insuficientes columnas: {len(df.columns)}, añadiendo columnas hasta llegar a 25")
                # Primero aseguramos que tenemos las columnas OHLCV
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = 0.0  # Añadir columna faltante con ceros
                
                # Luego añadimos las columnas faltantes hasta llegar a 25
                existing_cols = len(df.columns)
                for i in range(existing_cols, 25):
                    df[f'feature_{i-5+1}'] = 0.0  # Añadir columnas técnicas
            
            elif len(df.columns) > 25:
                logger.warning(f"Demasiadas columnas: {len(df.columns)}, truncando a 25 columnas")
                # Priorizamos las columnas OHLCV y los indicadores técnicos más importantes
                priority_cols = ['open', 'high', 'low', 'close', 'volume', 
                                'sma_5', 'sma_10', 'sma_20', 'rsi_14', 
                                'macd', 'macd_signal', 'macd_hist', 'bollinger_upper',
                                'bollinger_middle', 'bollinger_lower']
                
                # Creamos una lista con las columnas prioritarias que existen en el DataFrame
                final_cols = []
                for col in priority_cols:
                    if col in df.columns and len(final_cols) < 25:
                        final_cols.append(col)
                
                # Si todavía necesitamos más columnas, tomamos las primeras disponibles
                remaining_cols = [col for col in df.columns if col not in final_cols]
                while len(final_cols) < 25 and remaining_cols:
                    final_cols.append(remaining_cols.pop(0))
                
                # Nos quedamos con las 25 primeras columnas
                df = df[final_cols[:25]]
            
            # Verificación final
            assert len(df.columns) == 25, f"Error crítico: El DataFrame debería tener 25 columnas, pero tiene {len(df.columns)}"
            logger.debug(f"DataFrame expandido correctamente: shape={df.shape}, columnas={df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Error en _expand_data_for_model: {str(e)}")
            logger.exception("Detalles del error:")
            raise
    
    def _evaluate_and_trade(self, instrument: str):
        """
        Evalúa el modelo y ejecuta operaciones de trading para un instrumento específico.
        
        Args:
            instrument (str): Instrumento a evaluar
        """
        if not self.ready_to_trade:
            return
        
        try:
            # Verificar período de enfriamiento
            current_time = datetime.now()
            if instrument in self.last_action_time:
                time_since_last_action = (current_time - self.last_action_time[instrument]).total_seconds()
                if time_since_last_action < self.cooldown_period:
                    logger.debug(f"Período de enfriamiento activo para {instrument}. "
                                f"Faltan {self.cooldown_period - time_since_last_action:.1f}s")
                    return
            
            # Obtener últimos datos
            with self.lock:
                bar_history = self.nt8_client.get_bar_history(instrument)
            
            if bar_history.empty:
                logger.warning(f"No hay datos disponibles para evaluar {instrument}")
                return
            
            # Verificar si hay barras nuevas desde la última actualización
            if instrument in self.available_instruments:
                last_df = self.available_instruments[instrument]
                if len(bar_history) <= len(last_df):
                    newest_bar_time = bar_history["timestamp"].max()
                    last_df_newest_time = last_df["timestamp"].max()
                    
                    if newest_bar_time <= last_df_newest_time:
                        logger.debug(f"No hay barras nuevas para {instrument} desde la última evaluación")
                        return
            
            # Actualizar entorno con los nuevos datos
            obs = self._update_environment_with_history(instrument)
            
            if obs is None:
                logger.warning(f"No se pudo obtener observación del entorno para {instrument}")
                return
            
            # Obtener predicción del modelo
            action, _ = self.agent.predict(obs, deterministic=True)
            
            # Determinar la acción basada en la salida del modelo
            if isinstance(action, np.ndarray) and action.size > 1:
                # Si es un array multi-dimensional, determinar la acción discreta
                action_value = action[0]  # Asumir que el primer valor determina la dirección
                
                # Normalizar valores a rango -1 a 1 si es necesario
                if -1.0 <= action_value <= 1.0:
                    # Convertir valores continuos (-1 a 1) a discretos (0, 1, 2)
                    # Umbrales reducidos para hacer el modelo más propenso a abrir operaciones
                    if action_value < -0.1:  # Vender (umbral reducido de -0.3 a -0.1)
                        discrete_action = 2
                    elif action_value > 0.1:  # Comprar (umbral reducido de 0.3 a 0.1)
                        discrete_action = 1
                    else:  # Mantener
                        discrete_action = 0
                else:
                    # Si el modelo ya devuelve acciones discretas
                    # Usar módulo para convertir cualquier valor mayor a 2 a un valor válido (0, 1, 2)
                    discrete_action = int(action_value) % 3
            else:
                # Si es un escalar, usarlo directamente, asegurando que esté en el rango válido
                discrete_action = int(action) % 3
            
            logger.info(f"Predicción para {instrument}: acción={discrete_action} (original: {action})")
            
            # Ejecutar la acción
            self._execute_action(instrument, discrete_action)
            
        except Exception as e:
            logger.error(f"Error al evaluar y operar con {instrument}: {e}")
            import traceback
            logger.error(f"Traceback completo: {traceback.format_exc()}")
    
    def _execute_action(self, instrument: str, action: int):
        """
        Ejecuta una acción de trading para un instrumento específico.
        
        Args:
            instrument (str): Instrumento a operar
            action (int): Acción del modelo (0 = mantener, 1 = comprar, 2 = vender)
        """
        try:
            # Obtener posición actual
            position = self.nt8_client.get_position(instrument)
            
            # Actualizar el estado de posición
            current_position = 0  # Sin posición
            if position:
                if position["market_position"] == "Long":
                    current_position = 1
                elif position["market_position"] == "Short":
                    current_position = -1
            
            self.positions[instrument] = current_position
            
            # Interpretar y ejecutar la acción
            if action == 1 and current_position <= 0:  # Comprar
                logger.info(f"Señal de COMPRA detectada para {instrument}")
                
                # Si hay posición corta, cerrarla primero
                if current_position < 0:
                    self.nt8_client.close_position(instrument)
                    time.sleep(1)  # Pequeña pausa para asegurar que la orden se procesa
                
                # Abrir posición larga
                success = self.nt8_client.market_buy(
                    instrument, 
                    quantity=self.quantity
                )
                
                if success:
                    self.positions[instrument] = 1
                    self.last_action_time[instrument] = datetime.now()
                    logger.info(f"Orden de COMPRA enviada para {self.quantity} contratos de {instrument}")
                else:
                    logger.error(f"Error al enviar orden de COMPRA para {instrument}")
            
            elif action == 2 and current_position >= 0:  # Vender
                logger.info(f"Señal de VENTA detectada para {instrument}")
                
                # Si hay posición larga, cerrarla primero
                if current_position > 0:
                    self.nt8_client.close_position(instrument)
                    time.sleep(1)  # Pequeña pausa para asegurar que la orden se procesa
                
                # Abrir posición corta
                success = self.nt8_client.market_sell(
                    instrument, 
                    quantity=self.quantity
                )
                
                if success:
                    self.positions[instrument] = -1
                    self.last_action_time[instrument] = datetime.now()
                    logger.info(f"Orden de VENTA enviada para {self.quantity} contratos de {instrument}")
                else:
                    logger.error(f"Error al enviar orden de VENTA para {instrument}")
            
            elif action == 0 and current_position != 0:  # Cerrar posición
                logger.info(f"Señal de CIERRE detectada para {instrument}")
                
                success = self.nt8_client.close_position(instrument)
                
                if success:
                    self.positions[instrument] = 0
                    self.last_action_time[instrument] = datetime.now()
                    logger.info(f"Posición cerrada para {instrument}")
                else:
                    logger.error(f"Error al cerrar posición para {instrument}")
            
            else:
                logger.debug(f"Acción {action} ignorada en posición actual {current_position} para {instrument}")
                    
        except Exception as e:
            logger.error(f"Error al ejecutar acción {action} para {instrument}: {e}")
            import traceback
            logger.error(f"Traceback completo: {traceback.format_exc()}")
    
    def _on_bar_data(self, bar_data):
        """
        Callback para recibir nuevos datos de barras.
        
        Args:
            bar_data (Dict): Datos de la barra recibida
        """
        # Almacenar el instrumento detectado
        instrument = bar_data['instrument']
        self.detected_instruments.add(instrument)
        
        logger.debug(f"Nueva barra recibida: {instrument} - {bar_data['timestamp']}")
        logger.debug(f"OHLCV: {bar_data['open']}, {bar_data['high']}, {bar_data['low']}, {bar_data['close']}, {bar_data['volume']}")
        
        # Si estamos esperando suficientes datos, verificar si ya los tenemos
        if not self.ready_to_trade:
            with self.lock:
                bar_history = self.nt8_client.get_bar_history(instrument)
                if len(bar_history) >= self.min_bars_required:
                    self.ready_to_trade = True
                    
                    # Si no se especificó instrumento, usar el primero detectado
                    if self.primary_instrument is None:
                        self.primary_instrument = instrument
                        logger.info(f"Instrumento principal seleccionado automáticamente: {self.primary_instrument}")
                    
                    logger.info(f"Suficientes datos recibidos ({len(bar_history)} barras) para {instrument}. Listo para operar.")
                    
                    # Actualizar el entorno con los datos históricos
                    self._update_environment_with_history(instrument)
    
    def _on_position_update(self, position_data):
        """
        Callback para recibir actualizaciones de posiciones.
        
        Args:
            position_data (Dict): Datos de la posición actualizada
        """
        instrument = position_data.get('instrument')
        if instrument:
            self.detected_instruments.add(instrument)
            
        logger.info(f"Actualización de posición: {position_data}")
        
        # Actualizar estado interno de posiciones
        if instrument:
            if position_data["market_position"] == "Long":
                self.positions[instrument] = 1
            elif position_data["market_position"] == "Short":
                self.positions[instrument] = -1
            else:
                self.positions[instrument] = 0
    
    def _on_order_update(self, order_data):
        """
        Callback para recibir actualizaciones de órdenes.
        
        Args:
            order_data (Dict): Datos de la orden actualizada
        """
        instrument = order_data.get('instrument')
        if instrument:
            self.detected_instruments.add(instrument)
            
        logger.info(f"Actualización de orden: {order_data}")
    
    def _on_error(self, error_msg):
        """
        Callback para recibir errores del servidor.
        
        Args:
            error_msg (str): Mensaje de error
        """
        logger.error(f"Error desde servidor: {error_msg}")


def main():
    """Función principal para ejecutar el trader desde línea de comandos."""
    parser = argparse.ArgumentParser(description='NinjaTrader 8 Live Trader con modelo PPO')
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo entrenado')
    parser.add_argument('--instrument', type=str, help='Instrumento a operar (opcional, se detectará automáticamente si no se especifica)')
    parser.add_argument('--host', type=str, default='localhost', help='Dirección IP del servidor NinjaTrader')
    parser.add_argument('--port', type=int, default=5555, help='Puerto TCP del servidor')
    parser.add_argument('--quantity', type=int, default=1, help='Cantidad de contratos')
    parser.add_argument('--data', type=str, help='Ruta a datos históricos para inicializar')
    parser.add_argument('--interval', type=int, default=30, help='Intervalo de actualización en segundos')
    parser.add_argument('--min-bars', type=int, default=1, help='Mínimo de barras requeridas para empezar a operar')
    parser.add_argument('--cooldown', type=int, default=15, help='Período de enfriamiento entre operaciones (segundos)')
    parser.add_argument('--force-cpu', action='store_true', help='Forzar el uso de CPU en lugar de GPU')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', 
                       help='Nivel de detalle de los logs')
    
    args = parser.parse_args()
    
    # Configurar logger
    setup_logger(
        name="NT8Trader", 
        log_file="./logs/nt8_trader.log", 
        level=args.log_level
    )
    
    try:
        # Si se especifica --force-cpu, configurar CUDA_VISIBLE_DEVICES a cadena vacía
        if args.force_cpu:
            logger.info("Forzando el uso de CPU (--force-cpu)")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Crear y ejecutar trader
        trader = NT8Trader(
            model_path=args.model,
            instrument=args.instrument,
            host=args.host,
            port=args.port,
            quantity=args.quantity,
            data_path=args.data,
            update_interval=args.interval,
            min_bars_required=args.min_bars,
            cooldown_period=args.cooldown
        )
        
        trader.start()
        
    except KeyboardInterrupt:
        logger.info("Programa interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
