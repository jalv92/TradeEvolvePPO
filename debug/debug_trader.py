"""
Script para depurar la conexión con NinjaTrader 8 y el trading con el modelo.
Este script configura un sistema de logging detallado y ejecuta el trader.
"""

import os
import sys
import logging
import time
from datetime import datetime

# Configurar logging extremadamente detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_trader.log', mode='w')
    ]
)

# Configurar logging para bibliotecas externas
logging.getLogger('stable_baselines3').setLevel(logging.INFO)
logging.getLogger('gymnasium').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logger = logging.getLogger("DebugTrader")

# Añadir directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_with_detailed_logs():
    """Ejecuta el trader con logs detallados."""
    logger.info("=" * 80)
    logger.info("INICIANDO DEPURACIÓN DEL TRADER")
    logger.info("=" * 80)
    
    # Buscar ubicación del modelo
    model_paths = []
    for root, dirs, files in os.walk('results'):
        for file in files:
            if file.endswith('.zip'):
                model_paths.append(os.path.join(root, file))
    
    if not model_paths:
        logger.error("No se encontraron modelos entrenados (.zip) en la carpeta 'results'")
        return
    
    # Usar el modelo "best_model.zip" si existe, o el primero encontrado
    best_model = None
    for path in model_paths:
        if 'best_model.zip' in path:
            best_model = path
            break
    
    model_path = best_model if best_model else model_paths[0]
    logger.info(f"Usando modelo: {model_path}")
    
    # Importar el trader después de configurar logging
    from connector_nt8.ninjatrader_client import NT8Client
    
    # Probar primero la conexión básica
    logger.info("Probando conexión básica con NinjaTrader 8...")
    client = NT8Client(host="localhost", port=5555)
    
    # Almacenar datos recibidos para análisis
    received_bars = []
    
    def on_bar_data(bar_data):
        received_bars.append(bar_data)
        instrument = bar_data.get('instrument', 'unknown')
        timestamp = bar_data.get('timestamp', 'unknown')
        logger.info(f"[CALLBACK] Nueva barra recibida: {instrument} - {timestamp}")
        logger.debug(f"[CALLBACK] Datos completos: {bar_data}")
    
    def on_position_update(position_data):
        logger.info(f"[CALLBACK] Actualización de posición: {position_data}")
    
    def on_order_update(order_data):
        logger.info(f"[CALLBACK] Actualización de orden: {order_data}")
    
    def on_error(error_msg):
        logger.error(f"[CALLBACK] Error desde servidor: {error_msg}")
    
    client.set_on_bar_data(on_bar_data)
    client.set_on_position_update(on_position_update)
    client.set_on_order_update(on_order_update)
    client.set_on_error(on_error)
    
    if not client.connect():
        logger.error("No se pudo conectar con NinjaTrader 8. Abortando.")
        return
    
    logger.info("Conexión establecida correctamente. Esperando datos iniciales por 10 segundos...")
    
    # Esperar 10 segundos para recibir datos iniciales
    for i in range(10):
        time.sleep(1)
        logger.info(f"Esperando datos... {i+1}/10")
    
    # Verificar qué instrumentos hemos detectado realmente
    logger.info("Verificando instrumentos detectados...")
    real_instruments = []
    
    for instrument_name in client.bar_history.keys():
        try:
            bar_history = client.get_bar_history(instrument_name)
            # Verificar si este es un instrumento real o solo una columna
            if isinstance(bar_history, object) and hasattr(bar_history, 'shape') and bar_history.shape[0] > 0:
                real_instruments.append(instrument_name)
                logger.info(f"Instrumento real detectado: {instrument_name} - {len(bar_history)} barras")
                if not bar_history.empty:
                    logger.info(f"Primera barra: {bar_history.iloc[0].to_dict()}")
                    logger.info(f"Última barra: {bar_history.iloc[-1].to_dict()}")
        except Exception as e:
            logger.error(f"Error al procesar el instrumento {instrument_name}: {e}")
    
    if not real_instruments:
        # Si no encontramos instrumentos reales, vamos a verificar los datos recibidos directamente
        logger.warning("No se detectaron instrumentos reales después de 10 segundos.")
        
        # Analizar datos recibidos directamente
        actual_instruments = set()
        for bar in received_bars:
            if 'instrument' in bar:
                actual_instruments.add(bar['instrument'])
        
        if actual_instruments:
            logger.info(f"Instrumentos detectados en callbacks: {list(actual_instruments)}")
            real_instruments = list(actual_instruments)
        else:
            logger.error("No se detectaron instrumentos ni en el historial ni en callbacks. Abortando.")
            client.disconnect()
            return
    
    logger.info(f"Se detectaron {len(real_instruments)} instrumentos reales: {real_instruments}")
    
    # Si llegamos aquí, la conexión básica funciona. Ahora cargar el modelo
    logger.info("Conexión básica exitosa. Ahora intentando cargar el modelo...")
    
    try:
        # Importar clases necesarias para cargar el modelo
        from connector_nt8.nt8_trader import NT8Trader
    
        # Crear una instancia del trader
        trader = NT8Trader(
            model_path=model_path,
            instrument=real_instruments[0] if real_instruments else None,  # Usar el primer instrumento detectado
            host="localhost",
            port=5555,
            quantity=1,
            update_interval=30,
            min_bars_required=1,  # Reducido para entornos de prueba
            cooldown_period=15
        )
        
        logger.info(f"Trader creado con éxito. Iniciando trading con instrumento: {real_instruments[0] if real_instruments else 'auto-detectado'}")
        logger.info("LOGS DETALLADOS DISPONIBLES EN debug_trader.log")
        
        # Iniciar el trader
        trader.start()
        
    except Exception as e:
        logger.exception(f"Error al cargar el modelo o iniciar el trading: {e}")
    finally:
        logger.info("Finalizando depuración")
        if 'client' in locals() and client:
            client.disconnect()

if __name__ == "__main__":
    run_with_detailed_logs() 