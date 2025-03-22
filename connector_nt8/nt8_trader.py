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
                 instrument: str,
                 host: str = "localhost",
                 port: int = 5555,
                 quantity: int = 1,
                 config: Optional[Dict[str, Any]] = None,
                 data_path: Optional[str] = None,
                 update_interval: int = 60,
                 min_bars_required: int = 20):
        """
        Inicializa el trader para NinjaTrader 8.
        
        Args:
            model_path (str): Ruta al modelo entrenado (.zip)
            instrument (str): Instrumento a operar (ej. "NQ 06-23")
            host (str): Dirección IP o hostname donde se ejecuta NinjaTrader
            port (int): Puerto TCP de la estrategia NT8StrategyServer
            quantity (int): Cantidad de contratos a operar
            config (Dict[str, Any]): Configuración (si es None, se carga la predeterminada)
            data_path (str): Ruta a datos históricos para inicializar el entorno
            update_interval (int): Intervalo de actualización en segundos
            min_bars_required (int): Mínimo número de barras requeridas para empezar a operar
        """
        self.model_path = model_path
        self.instrument = instrument
        self.host = host
        self.port = port
        self.quantity = quantity
        self.update_interval = update_interval
        self.min_bars_required = min_bars_required
        
        self.running = False
        self.current_position = 0  # 0 = sin posición, 1 = long, -1 = short
        self.last_action_time = datetime.now() - timedelta(minutes=5)  # Evitar acciones inmediatas
        self.min_action_interval = 10  # Mínimo intervalo entre acciones (segundos)
        
        # Cargar configuración si no se proporciona
        self.config = config if config is not None else get_config()
        
        # Crear cliente de NinjaTrader
        self.nt_client = NT8Client(host=host, port=port)
        
        # Configurar callbacks para eventos del cliente
        self.nt_client.set_on_bar_data(self._on_bar_data)
        self.nt_client.set_on_position_update(self._on_position_update)
        self.nt_client.set_on_error(self._on_error)
        
        # Cargar datos históricos para inicializar el entorno
        self.data_loader = DataLoader()
        
        # Para seguimiento del estado del trading
        self.lock = threading.Lock()
        self.ready_to_trade = False
        self.last_df = None
        
        # Cargar datos iniciales si se proporciona una ruta
        if data_path and os.path.exists(data_path):
            self.initial_data = self.data_loader.load_data(data_path)
        else:
            # Intentar encontrar datos en ubicaciones comunes
            possible_paths = [
                "./data/processed/training_data.csv",
                "../data/processed/training_data.csv",
                "./data/processed/test_data.csv",
                "../data/processed/test_data.csv"
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
        
        # Crear entorno y cargar modelo
        self._setup_environment_and_model()
    
    def _setup_environment_and_model(self):
        """Configura el entorno y carga el modelo."""
        try:
            # Crear entorno de trading en modo live
            if self.initial_data is not None:
                self.env = TradingEnv(self.initial_data, self.config, mode="live")
            else:
                # Si no hay datos iniciales, creamos un DataFrame vacío con las columnas necesarias
                columns = ["timestamp", "open", "high", "low", "close", "volume"]
                if "feature_columns" in self.config:
                    for col in self.config["feature_columns"]:
                        if col not in columns:
                            columns.append(col)
                empty_df = pd.DataFrame(columns=columns)
                self.env = TradingEnv(empty_df, self.config, mode="live")
            
            # Crear agente PPO
            self.agent = PPOAgent(self.env, self.config)
            
            # Cargar modelo entrenado
            self.agent.load(self.model_path, env=self.env)
            
            logger.info(f"Modelo cargado correctamente desde {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error al configurar entorno o cargar modelo: {e}")
            raise
    
    def start(self):
        """Inicia el trading en tiempo real."""
        if self.running:
            logger.warning("El trader ya está ejecutándose")
            return
        
        self.running = True
        self.ready_to_trade = False
        logger.info(f"Iniciando trading en vivo con {self.instrument}")
        
        try:
            # Conectar con el servidor NT8StrategyServer
            if not self.nt_client.connect():
                logger.error("No se pudo conectar con el servidor NT8StrategyServer")
                self.running = False
                return
            
            # Esperar a tener suficientes datos para operar
            while self.running and not self.ready_to_trade:
                try:
                    # Verificar si tenemos suficientes barras
                    with self.lock:
                        bar_history = self.nt_client.get_bar_history(self.instrument)
                        if len(bar_history) >= self.min_bars_required:
                            self.ready_to_trade = True
                            logger.info(f"Suficientes datos recibidos ({len(bar_history)} barras). Listo para operar.")
                            
                            # Actualizar el entorno con los datos históricos
                            self._update_environment_with_history(bar_history)
                        else:
                            logger.info(f"Esperando más datos. Actualmente {len(bar_history)} barras, necesito {self.min_bars_required}.")
                
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
                        # Evaluar modelo y ejecutar acciones
                        self._evaluate_and_trade()
                        last_update_time = current_time
                    
                    # Esperar un poco para no sobrecargar la CPU
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error en ciclo de trading: {e}")
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
            self.nt_client.close_position(self.instrument)
            logger.info(f"Todas las posiciones en {self.instrument} cerradas")
        except Exception as e:
            logger.error(f"Error al cerrar posiciones: {e}")
        
        # Desconectar del servidor
        self.nt_client.disconnect()
        
        logger.info("Trading detenido correctamente")
    
    def _update_environment_with_history(self, bar_history: pd.DataFrame):
        """
        Actualiza el entorno con los datos históricos recibidos.
        
        Args:
            bar_history (pd.DataFrame): Historial de barras
        """
        if bar_history.empty:
            return
        
        try:
            # Convertir al formato esperado por el entorno
            bar_history = bar_history.sort_values("timestamp")
            
            # Guardar una copia del último DataFrame actualizado
            self.last_df = bar_history.copy()
            
            # Actualizar el entorno
            obs = self.env.update_data(bar_history)
            logger.info(f"Entorno actualizado con {len(bar_history)} barras históricas")
            
            return obs
        except Exception as e:
            logger.error(f"Error al actualizar entorno con historial: {e}")
            return None
    
    def _evaluate_and_trade(self):
        """Evalúa el modelo y ejecuta operaciones de trading."""
        if not self.ready_to_trade:
            return
        
        try:
            # Obtener últimos datos
            with self.lock:
                bar_history = self.nt_client.get_bar_history(self.instrument)
            
            if bar_history.empty:
                logger.warning("No hay datos disponibles para evaluar")
                return
            
            # Verificar si hay barras nuevas desde la última actualización
            if self.last_df is not None:
                if len(bar_history) <= len(self.last_df):
                    newest_bar_time = bar_history["timestamp"].max()
                    last_df_newest_time = self.last_df["timestamp"].max()
                    
                    if newest_bar_time <= last_df_newest_time:
                        logger.debug("No hay barras nuevas desde la última evaluación")
                        return
            
            # Actualizar entorno con los nuevos datos
            obs = self._update_environment_with_history(bar_history)
            
            if obs is None:
                logger.warning("No se pudo obtener observación del entorno")
                return
            
            # Obtener predicción del modelo
            action, _ = self.agent.predict(obs, deterministic=True)
            
            # Ejecutar la acción
            self._execute_action(action)
            
        except Exception as e:
            logger.error(f"Error al evaluar y operar: {e}")
    
    def _execute_action(self, action):
        """
        Ejecuta una acción de trading.
        
        Args:
            action: Acción del modelo (0 = mantener, 1 = comprar, 2 = vender)
        """
        # Verificar intervalo mínimo entre acciones
        current_time = datetime.now()
        time_since_last_action = (current_time - self.last_action_time).total_seconds()
        
        if time_since_last_action < self.min_action_interval:
            logger.debug(f"Ignorando acción: intervalo mínimo no cumplido ({time_since_last_action:.1f}s < {self.min_action_interval}s)")
            return
        
        try:
            # Obtener posición actual
            position = self.nt_client.get_position(self.instrument)
            
            if position:
                if position["market_position"] == "Long":
                    self.current_position = 1
                elif position["market_position"] == "Short":
                    self.current_position = -1
                else:
                    self.current_position = 0
            else:
                self.current_position = 0
            
            # Interpretar acción según la configuración del entorno
            # Esto debe adaptarse según cómo hayas definido las acciones en tu entorno
            if action == 1 and self.current_position <= 0:  # Comprar
                logger.info(f"Señal de COMPRA detectada")
                
                # Si hay posición corta, cerrarla primero
                if self.current_position < 0:
                    self.nt_client.close_position(self.instrument)
                    time.sleep(1)  # Pequeña pausa para asegurar que la orden se procesa
                
                # Abrir posición larga
                success = self.nt_client.market_buy(
                    self.instrument, 
                    quantity=self.quantity
                )
                
                if success:
                    self.current_position = 1
                    self.last_action_time = current_time
                    logger.info(f"Orden de COMPRA enviada para {self.quantity} contratos de {self.instrument}")
                else:
                    logger.error("Error al enviar orden de COMPRA")
            
            elif action == 2 and self.current_position >= 0:  # Vender
                logger.info(f"Señal de VENTA detectada")
                
                # Si hay posición larga, cerrarla primero
                if self.current_position > 0:
                    self.nt_client.close_position(self.instrument)
                    time.sleep(1)  # Pequeña pausa para asegurar que la orden se procesa
                
                # Abrir posición corta
                success = self.nt_client.market_sell(
                    self.instrument, 
                    quantity=self.quantity
                )
                
                if success:
                    self.current_position = -1
                    self.last_action_time = current_time
                    logger.info(f"Orden de VENTA enviada para {self.quantity} contratos de {self.instrument}")
                else:
                    logger.error("Error al enviar orden de VENTA")
            
            elif action == 0 and self.current_position != 0:  # Cerrar posición
                logger.info(f"Señal de CIERRE detectada")
                
                success = self.nt_client.close_position(self.instrument)
                
                if success:
                    self.current_position = 0
                    self.last_action_time = current_time
                    logger.info(f"Posición cerrada para {self.instrument}")
                else:
                    logger.error("Error al cerrar posición")
            
            else:
                logger.debug(f"Acción {action} ignorada en posición actual {self.current_position}")
                    
        except Exception as e:
            logger.error(f"Error al ejecutar acción {action}: {e}")
    
    def _on_bar_data(self, bar_data):
        """
        Callback para recibir nuevos datos de barras.
        
        Args:
            bar_data (Dict): Datos de la barra recibida
        """
        logger.debug(f"Nueva barra recibida: {bar_data['instrument']} - {bar_data['timestamp']}")
        
        # Si estamos esperando suficientes datos, verificar si ya los tenemos
        if not self.ready_to_trade:
            with self.lock:
                bar_history = self.nt_client.get_bar_history(self.instrument)
                if len(bar_history) >= self.min_bars_required:
                    self.ready_to_trade = True
                    logger.info(f"Suficientes datos recibidos ({len(bar_history)} barras). Listo para operar.")
                    
                    # Actualizar el entorno con los datos históricos
                    self._update_environment_with_history(bar_history)
    
    def _on_position_update(self, position_data):
        """
        Callback para recibir actualizaciones de posiciones.
        
        Args:
            position_data (Dict): Datos de la posición actualizada
        """
        logger.info(f"Actualización de posición: {position_data['instrument']} - {position_data['market_position']} - {position_data['quantity']}")
        
        # Actualizar estado interno
        if position_data["instrument"] == self.instrument:
            if position_data["market_position"] == "Long":
                self.current_position = 1
            elif position_data["market_position"] == "Short":
                self.current_position = -1
            else:
                self.current_position = 0
    
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
    parser.add_argument('--instrument', type=str, required=True, help='Instrumento a operar (ej. "NQ 06-23")')
    parser.add_argument('--host', type=str, default='localhost', help='Dirección IP del servidor NinjaTrader')
    parser.add_argument('--port', type=int, default=5555, help='Puerto TCP del servidor')
    parser.add_argument('--quantity', type=int, default=1, help='Cantidad de contratos')
    parser.add_argument('--data', type=str, help='Ruta a datos históricos para inicializar')
    parser.add_argument('--interval', type=int, default=60, help='Intervalo de actualización en segundos')
    parser.add_argument('--min-bars', type=int, default=20, help='Mínimo de barras requeridas para empezar a operar')
    
    args = parser.parse_args()
    
    # Configurar logger
    setup_logger(log_level=logging.INFO)
    
    try:
        # Crear y ejecutar trader
        trader = NT8Trader(
            model_path=args.model,
            instrument=args.instrument,
            host=args.host,
            port=args.port,
            quantity=args.quantity,
            data_path=args.data,
            update_interval=args.interval,
            min_bars_required=args.min_bars
        )
        
        trader.start()
        
    except KeyboardInterrupt:
        logger.info("Programa interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())