"""
Cliente TCP para NinjaTrader 8.
Proporciona una interfaz para comunicarse con la estrategia NT8StrategyServer en NinjaTrader 8
y enviar/recibir datos y órdenes.
"""

import socket
import threading
import time
import logging
import queue
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class NT8Client:
    """
    Cliente para comunicarse con la estrategia NT8StrategyServer en NinjaTrader 8.
    Permite recibir datos de mercado y enviar órdenes de trading.
    """
    
    def __init__(self, host: str = "localhost", port: int = 5555):
        """
        Inicializa el cliente NinjaTrader.
        
        Args:
            host (str): Dirección IP o hostname del servidor (NinjaTrader)
            port (int): Puerto TCP del servidor
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.running = False
        self.receive_thread = None
        self.lock = threading.Lock()
        
        # Colas para almacenar datos recibidos
        self.bar_data_queue = queue.Queue()
        self.order_data_queue = queue.Queue()
        self.position_data_queue = queue.Queue()
        self.error_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Callbacks para procesar datos
        self.on_bar_data = None
        self.on_order_update = None
        self.on_position_update = None
        self.on_error = None
        
        # Almacenamiento de datos más recientes
        self.last_bar_data = None
        self.orders = {}
        self.positions = {}
        
        # Historial de barras para análisis
        self.bar_history = pd.DataFrame()
        self.max_history_size = 1000  # Número máximo de barras a mantener
    
    def connect(self) -> bool:
        """
        Conecta con el servidor NT8StrategyServer.
        
        Returns:
            bool: True si la conexión fue exitosa, False en caso contrario
        """
        if self.connected:
            return True
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.running = True
            
            # Iniciar hilo para recibir datos
            self.receive_thread = threading.Thread(target=self._receive_data)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            logger.info(f"Conectado a NT8StrategyServer en {self.host}:{self.port}")
            
            # Solicitar datos iniciales
            self.request_bar_data()
            self.request_orders()
            self.request_positions()
            
            return True
        
        except Exception as e:
            logger.error(f"Error al conectar con NT8StrategyServer: {e}")
            self.connected = False
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
            return False
    
    def disconnect(self) -> None:
        """
        Desconecta del servidor NT8StrategyServer.
        """
        self.running = False
        
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                logger.error(f"Error al cerrar el socket: {e}")
            finally:
                self.socket = None
        
        self.connected = False
        logger.info("Desconectado de NT8StrategyServer")
    
    def _receive_data(self) -> None:
        """
        Hilo para recibir datos del servidor continuamente.
        """
        buffer = ""
        
        while self.running and self.connected:
            try:
                # Leer datos del socket
                data = self.socket.recv(4096)
                
                if not data:
                    logger.warning("Conexión cerrada por el servidor")
                    self.connected = False
                    break
                
                # Decodificar y procesar los datos
                buffer += data.decode('utf-8')
                
                # Dividir por líneas y procesar cada mensaje completo
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    self._process_message(line.strip())
                
            except Exception as e:
                logger.error(f"Error al recibir datos: {e}")
                self.connected = False
                break
        
        # Limpiar después de salir del bucle
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        self.connected = False
        logger.info("Hilo de recepción finalizado")
    
    def _process_message(self, message: str) -> None:
        """
        Procesa un mensaje recibido del servidor.
        
        Args:
            message (str): Mensaje recibido
        """
        if not message:
            return
        
        try:
            # Dividir el mensaje en partes
            parts = message.split(';')
            
            if len(parts) < 1:
                return
            
            # Determinar el tipo de mensaje
            msg_type = parts[0].upper()
            
            if msg_type == "BARDATA":
                # Formato: BARDATA;instrumento;timestamp;open;high;low;close;volume
                if len(parts) >= 7:
                    bar_data = {
                        "instrument": parts[1],
                        "timestamp": datetime.strptime(parts[2], "%Y-%m-%d %H:%M:%S"),
                        "open": float(parts[3]),
                        "high": float(parts[4]),
                        "low": float(parts[5]),
                        "close": float(parts[6]),
                        "volume": float(parts[7]) if len(parts) > 7 else 0
                    }
                    
                    self.last_bar_data = bar_data
                    self.bar_data_queue.put(bar_data)
                    self._update_bar_history(bar_data)
                    
                    # Ejecutar callback si existe
                    if self.on_bar_data:
                        self.on_bar_data(bar_data)
            
            elif msg_type == "ORDER":
                # Formato: ORDER;id;instrumento;acción;tipo;cantidad;limitPrice;stopPrice;estado
                if len(parts) >= 8:
                    order_data = {
                        "id": parts[1],
                        "instrument": parts[2],
                        "action": parts[3],
                        "type": parts[4],
                        "quantity": int(parts[5]),
                        "limit_price": float(parts[6]) if parts[6] else 0,
                        "stop_price": float(parts[7]) if parts[7] else 0,
                        "state": parts[8] if len(parts) > 8 else ""
                    }
                    
                    self.orders[order_data["id"]] = order_data
                    self.order_data_queue.put(order_data)
                    
                    # Ejecutar callback si existe
                    if self.on_order_update:
                        self.on_order_update(order_data)
            
            elif msg_type == "POSITION":
                # Formato: POSITION;instrumento;marketPosition;cantidad;precioPromedio;pnlMoneda;pnlPorciento
                if len(parts) >= 6:
                    position_data = {
                        "instrument": parts[1],
                        "market_position": parts[2],
                        "quantity": int(parts[3]),
                        "average_price": float(parts[4]),
                        "pnl_currency": float(parts[5]),
                        "pnl_percent": float(parts[6]) if len(parts) > 6 else 0
                    }
                    
                    self.positions[position_data["instrument"]] = position_data
                    self.position_data_queue.put(position_data)
                    
                    # Ejecutar callback si existe
                    if self.on_position_update:
                        self.on_position_update(position_data)
            
            elif msg_type == "ERROR":
                # Formato: ERROR;mensaje
                error_msg = parts[1] if len(parts) > 1 else "Error desconocido"
                
                logger.error(f"Error desde NT8StrategyServer: {error_msg}")
                self.error_queue.put(error_msg)
                
                # Ejecutar callback si existe
                if self.on_error:
                    self.on_error(error_msg)
            
            elif msg_type in ["ORDERPLACED", "POSITIONCLOSED", "ORDERSCANCELED", "PONG"]:
                # Respuestas a comandos
                self.response_queue.put(message)
            
            else:
                logger.warning(f"Mensaje desconocido: {message}")
        
        except Exception as e:
            logger.error(f"Error al procesar mensaje '{message}': {e}")
    
    def _update_bar_history(self, bar_data: Dict[str, Any]) -> None:
        """
        Actualiza el historial de barras con nuevos datos.
        
        Args:
            bar_data (Dict[str, Any]): Datos de la nueva barra
        """
        try:
            # Verificar los datos antes de procesar
            required_fields = ["timestamp", "open", "high", "low", "close", "volume", "instrument"]
            for field in required_fields:
                if field not in bar_data:
                    logger.error(f"Error: campo '{field}' faltante en bar_data: {bar_data}")
                    return
            
            # Asegurar que timestamp es un objeto datetime
            if not isinstance(bar_data["timestamp"], datetime):
                try:
                    if isinstance(bar_data["timestamp"], str):
                        bar_data["timestamp"] = datetime.strptime(bar_data["timestamp"], "%Y-%m-%d %H:%M:%S")
                    else:
                        logger.error(f"Tipo de timestamp no soportado: {type(bar_data['timestamp'])}")
                        return
                except Exception as e:
                    logger.error(f"Error al convertir timestamp: {e}")
                    return
            
            # Convertir a DataFrame
            new_row = {
                "timestamp": [bar_data["timestamp"]],
                "open": [float(bar_data["open"])],
                "high": [float(bar_data["high"])],
                "low": [float(bar_data["low"])],
                "close": [float(bar_data["close"])],
                "volume": [float(bar_data["volume"])],
                "instrument": [bar_data["instrument"]]
            }
            
            df_row = pd.DataFrame(new_row)
            
            # Añadir a historial
            with self.lock:
                if self.bar_history.empty:
                    self.bar_history = df_row
                    logger.debug(f"Iniciado historial de barras con primera barra: {bar_data['instrument']} - {bar_data['timestamp']}")
                else:
                    # Verificar si ya existe esta barra (mismo timestamp e instrumento)
                    mask = (self.bar_history["timestamp"] == bar_data["timestamp"]) & \
                           (self.bar_history["instrument"] == bar_data["instrument"])
                    
                    if mask.any():
                        # Actualizar la barra existente
                        idx = mask.idxmax()
                        for col in df_row.columns:
                            self.bar_history.loc[idx, col] = df_row.iloc[0][col]
                        logger.debug(f"Actualizada barra existente: {bar_data['instrument']} - {bar_data['timestamp']}")
                    else:
                        # Agregar nueva barra - usando merge de dict para evitar errores de longitud desigual
                        self.bar_history = pd.concat([self.bar_history, df_row], ignore_index=True)
                        logger.debug(f"Añadida nueva barra: {bar_data['instrument']} - {bar_data['timestamp']}")
                
                # Limitar el tamaño del historial
                if len(self.bar_history) > self.max_history_size:
                    self.bar_history = self.bar_history.iloc[-self.max_history_size:]
                    logger.debug(f"Limitado historial a {self.max_history_size} barras")
                
                # Asegurar que todas las columnas tienen el tipo correcto
                self.bar_history["timestamp"] = pd.to_datetime(self.bar_history["timestamp"])
                for col in ["open", "high", "low", "close", "volume"]:
                    self.bar_history[col] = self.bar_history[col].astype(float)
                self.bar_history["instrument"] = self.bar_history["instrument"].astype(str)
        
        except Exception as e:
            import traceback
            logger.error(f"Error al actualizar historial de barras: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Datos que causaron el error: {bar_data}")
    
    def get_bar_history(self, instrument: Optional[str] = None, n_bars: Optional[int] = None) -> pd.DataFrame:
        """
        Obtiene el historial de barras, opcionalmente filtrado por instrumento.
        
        Args:
            instrument (str, optional): Instrumento a filtrar
            n_bars (int, optional): Número de barras más recientes a devolver
        
        Returns:
            pd.DataFrame: DataFrame con el historial de barras
        """
        with self.lock:
            if self.bar_history.empty:
                return pd.DataFrame()
            
            # Filtrar por instrumento si se especifica
            df = self.bar_history
            if instrument:
                df = df[df["instrument"] == instrument]
            
            # Ordenar por timestamp
            df = df.sort_values("timestamp")
            
            # Limitar número de barras si se especifica
            if n_bars and n_bars > 0:
                df = df.iloc[-n_bars:]
            
            return df.reset_index(drop=True)
    
    def get_last_bar(self, instrument: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Obtiene la última barra recibida, opcionalmente filtrada por instrumento.
        
        Args:
            instrument (str, optional): Instrumento a filtrar
        
        Returns:
            Dict[str, Any] o None: Datos de la última barra o None si no hay datos
        """
        with self.lock:
            if self.bar_history.empty:
                return None
            
            # Filtrar por instrumento si se especifica
            df = self.bar_history
            if instrument:
                df = df[df["instrument"] == instrument]
            
            if df.empty:
                return None
            
            # Obtener la barra más reciente
            last_bar = df.iloc[-1].to_dict()
            return last_bar
    
    def _send_command(self, command: str) -> bool:
        """
        Envía un comando al servidor.
        
        Args:
            command (str): Comando a enviar
        
        Returns:
            bool: True si el comando se envió correctamente, False en caso contrario
        """
        if not self.connected or not self.socket:
            logger.error("No conectado al servidor")
            return False
        
        try:
            self.socket.sendall((command + "\n").encode('utf-8'))
            return True
        except Exception as e:
            logger.error(f"Error al enviar comando: {e}")
            self.connected = False
            return False
    
    def request_bar_data(self) -> bool:
        """
        Solicita datos de barras al servidor.
        
        Returns:
            bool: True si la solicitud se envió correctamente, False en caso contrario
        """
        return self._send_command("GETDATA")
    
    def request_orders(self) -> bool:
        """
        Solicita datos de órdenes al servidor.
        
        Returns:
            bool: True si la solicitud se envió correctamente, False en caso contrario
        """
        return self._send_command("GETORDERS")
    
    def request_positions(self) -> bool:
        """
        Solicita datos de posiciones al servidor.
        
        Returns:
            bool: True si la solicitud se envió correctamente, False en caso contrario
        """
        return self._send_command("GETPOSITIONS")
    
    def ping(self) -> bool:
        """
        Envía un comando PING para verificar la conexión.
        
        Returns:
            bool: True si la solicitud se envió correctamente, False en caso contrario
        """
        return self._send_command("PING")
    
    def market_buy(self, instrument: str, quantity: int = 1) -> bool:
        """
        Envía una orden de compra a mercado.
        
        Args:
            instrument (str): Instrumento a operar
            quantity (int): Cantidad de contratos
        
        Returns:
            bool: True si la orden se envió correctamente, False en caso contrario
        """
        command = f"BUY;{instrument};{quantity};MARKET"
        return self._send_command(command)
    
    def market_sell(self, instrument: str, quantity: int = 1) -> bool:
        """
        Envía una orden de venta a mercado.
        
        Args:
            instrument (str): Instrumento a operar
            quantity (int): Cantidad de contratos
        
        Returns:
            bool: True si la orden se envió correctamente, False en caso contrario
        """
        command = f"SELL;{instrument};{quantity};MARKET"
        return self._send_command(command)
    
    def limit_buy(self, instrument: str, quantity: int = 1, price: float = 0) -> bool:
        """
        Envía una orden de compra limitada.
        
        Args:
            instrument (str): Instrumento a operar
            quantity (int): Cantidad de contratos
            price (float): Precio límite
        
        Returns:
            bool: True si la orden se envió correctamente, False en caso contrario
        """
        command = f"BUY;{instrument};{quantity};LIMIT;{price}"
        return self._send_command(command)
    
    def limit_sell(self, instrument: str, quantity: int = 1, price: float = 0) -> bool:
        """
        Envía una orden de venta limitada.
        
        Args:
            instrument (str): Instrumento a operar
            quantity (int): Cantidad de contratos
            price (float): Precio límite
        
        Returns:
            bool: True si la orden se envió correctamente, False en caso contrario
        """
        command = f"SELL;{instrument};{quantity};LIMIT;{price}"
        return self._send_command(command)
    
    def stop_buy(self, instrument: str, quantity: int = 1, price: float = 0) -> bool:
        """
        Envía una orden de compra con stop.
        
        Args:
            instrument (str): Instrumento a operar
            quantity (int): Cantidad de contratos
            price (float): Precio del stop
        
        Returns:
            bool: True si la orden se envió correctamente, False en caso contrario
        """
        command = f"BUY;{instrument};{quantity};STOP;{price}"
        return self._send_command(command)
    
    def stop_sell(self, instrument: str, quantity: int = 1, price: float = 0) -> bool:
        """
        Envía una orden de venta con stop.
        
        Args:
            instrument (str): Instrumento a operar
            quantity (int): Cantidad de contratos
            price (float): Precio del stop
        
        Returns:
            bool: True si la orden se envió correctamente, False en caso contrario
        """
        command = f"SELL;{instrument};{quantity};STOP;{price}"
        return self._send_command(command)
    
    def close_position(self, instrument: str) -> bool:
        """
        Cierra todas las posiciones en un instrumento.
        
        Args:
            instrument (str): Instrumento a operar
        
        Returns:
            bool: True si la orden se envió correctamente, False en caso contrario
        """
        command = f"CLOSEPOSITION;{instrument}"
        return self._send_command(command)
    
    def cancel_orders(self, instrument: str) -> bool:
        """
        Cancela todas las órdenes pendientes para un instrumento.
        
        Args:
            instrument (str): Instrumento a operar
        
        Returns:
            bool: True si la solicitud se envió correctamente, False en caso contrario
        """
        command = f"CANCELORDERS;{instrument}"
        return self._send_command(command)
    
    def get_position(self, instrument: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene los datos de la posición para un instrumento.
        
        Args:
            instrument (str): Instrumento a consultar
        
        Returns:
            Dict[str, Any] o None: Datos de la posición o None si no hay posición
        """
        with self.lock:
            if instrument in self.positions:
                return self.positions[instrument]
            return None
    
    def get_orders(self, instrument: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtiene las órdenes pendientes, opcionalmente filtradas por instrumento.
        
        Args:
            instrument (str, optional): Instrumento a filtrar
        
        Returns:
            List[Dict[str, Any]]: Lista de órdenes
        """
        with self.lock:
            if instrument:
                return [order for order in self.orders.values() if order["instrument"] == instrument]
            else:
                return list(self.orders.values())
    
    def wait_for_response(self, timeout: float = 5.0) -> Optional[str]:
        """
        Espera por una respuesta del servidor.
        
        Args:
            timeout (float): Tiempo máximo de espera en segundos
        
        Returns:
            str o None: Respuesta recibida o None si se agotó el tiempo
        """
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def set_on_bar_data(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Establece la función de callback para datos de barras.
        
        Args:
            callback (Callable): Función a llamar cuando se reciben nuevos datos
        """
        self.on_bar_data = callback
    
    def set_on_order_update(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Establece la función de callback para actualizaciones de órdenes.
        
        Args:
            callback (Callable): Función a llamar cuando se reciben nuevos datos
        """
        self.on_order_update = callback
    
    def set_on_position_update(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Establece la función de callback para actualizaciones de posiciones.
        
        Args:
            callback (Callable): Función a llamar cuando se reciben nuevos datos
        """
        self.on_position_update = callback
    
    def set_on_error(self, callback: Callable[[str], None]) -> None:
        """
        Establece la función de callback para errores.
        
        Args:
            callback (Callable): Función a llamar cuando se reciben errores
        """
        self.on_error = callback
    
    def __del__(self):
        """Destructor para asegurar que la conexión se cierra correctamente."""
        self.disconnect() 