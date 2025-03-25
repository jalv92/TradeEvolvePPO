"""
Script de prueba para verificar la conexión con NinjaTrader 8.
Este script se conecta a la estrategia NT8StrategyServer que debe estar ejecutándose en NinjaTrader 8.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NT8ConnectionTest")

# Añadir directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar el cliente para NinjaTrader 8
from connector_nt8.ninjatrader_client import NT8Client

# Variable global para almacenar los instrumentos detectados
detected_instruments = set()

def on_bar_data(bar_data):
    """Callback para recibir nuevos datos de barras."""
    global detected_instruments
    
    instrument = bar_data['instrument']
    detected_instruments.add(instrument)
    
    logger.info(f"Nueva barra recibida: {instrument} - {bar_data['timestamp']}")
    logger.info(f"OHLCV: {bar_data['open']}, {bar_data['high']}, {bar_data['low']}, {bar_data['close']}, {bar_data['volume']}")

def on_position_update(position_data):
    """Callback para recibir actualizaciones de posiciones."""
    global detected_instruments
    
    instrument = position_data.get('instrument')
    if instrument:
        detected_instruments.add(instrument)
    
    logger.info(f"Actualización de posición: {position_data}")

def on_order_update(order_data):
    """Callback para recibir actualizaciones de órdenes."""
    global detected_instruments
    
    instrument = order_data.get('instrument')
    if instrument:
        detected_instruments.add(instrument)
    
    logger.info(f"Actualización de orden: {order_data}")

def on_error(error_msg):
    """Callback para recibir errores del servidor."""
    logger.error(f"Error desde servidor: {error_msg}")

def main():
    # Parámetros de conexión
    host = "localhost"
    port = 5555  # Puerto por defecto del NT8StrategyServer
    
    logger.info(f"Iniciando prueba de conexión con NinjaTrader 8 en {host}:{port}")
    
    # Crear cliente
    client = NT8Client(host=host, port=port)
    
    # Configurar callbacks
    client.set_on_bar_data(on_bar_data)
    client.set_on_position_update(on_position_update)
    client.set_on_order_update(on_order_update)
    client.set_on_error(on_error)
    
    # Conectar
    if not client.connect():
        logger.error("No se pudo conectar con NT8StrategyServer")
        return 1
    
    logger.info("Conexión establecida correctamente. Esperando datos...")
    
    try:
        # Dejar correr un tiempo para recibir datos
        connection_time = 60  # segundos para mantener la conexión
        start_time = time.time()
        
        while time.time() - start_time < connection_time:
            # Cada 10 segundos mostrar información actual
            if int(time.time() - start_time) % 10 == 0 and int(time.time() - start_time) > 0:
                logger.info(f"Instrumentos detectados: {list(detected_instruments)}")
                
                # Mostrar información para cada instrumento detectado
                for instrument in detected_instruments:
                    # Obtener historial de barras
                    bar_history = client.get_bar_history(instrument)
                    if not bar_history.empty:
                        logger.info(f"Instrumento {instrument}: {len(bar_history)} barras históricas")
                        logger.info(f"Última barra de {instrument}: {bar_history.iloc[-1].to_dict() if len(bar_history) > 0 else 'No hay barras'}")
                    else:
                        logger.warning(f"No hay barras históricas para {instrument}")
                    
                    # Obtener posición actual
                    position = client.get_position(instrument)
                    logger.info(f"Posición actual para {instrument}: {position}")
                
                # Pequeña pausa para no hacer log constantemente
                time.sleep(1)
            
            # Pequeña pausa para no consumir CPU
            time.sleep(0.1)
            
        logger.info(f"Prueba completada. Conexión mantenida durante {connection_time} segundos.")
        logger.info(f"Instrumentos detectados durante la prueba: {list(detected_instruments)}")
        
    except KeyboardInterrupt:
        logger.info("Prueba interrumpida por el usuario")
    except Exception as e:
        logger.error(f"Error durante la prueba: {e}")
    finally:
        # Desconectar
        client.disconnect()
        logger.info("Desconectado del servidor")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 