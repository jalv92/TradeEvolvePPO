"""
Conector para NinjaTrader 8.
Proporciona una interfaz para enviar órdenes a NinjaTrader 8 utilizando la Advanced Trade Interface (ATI).
"""

import os
import uuid
import socket
import time
import winreg
import logging

logger = logging.getLogger(__name__)

class NinjaTraderConnector:
    """
    Conector para NinjaTrader 8 utilizando la Advanced Trade Interface (ATI).
    Permite enviar órdenes de compra/venta y gestionar posiciones.
    """
    
    def __init__(self, account="Sim101", ati_port=36973):
        """
        Inicializa el conector de NinjaTrader.
        
        Args:
            account (str): Cuenta de NinjaTrader a utilizar (por defecto "Sim101" para simulación)
            ati_port (int): Puerto del servidor ATI (por defecto 36973)
        """
        self.account = account
        self.ati_port = ati_port
        self.personal_root = self._get_personal_root_from_registry()
        self.is_connected = False
        self.socket = None
        
        if not self.personal_root:
            logger.warning("No se pudo encontrar la ruta de NinjaTrader en el registro. " 
                          "El modo de escritura de archivos podría no funcionar.")
    
    def _get_personal_root_from_registry(self):
        """
        Obtiene la ruta de instalación de NinjaTrader desde el registro de Windows.
        
        Returns:
            str: Ruta al directorio personal de NinjaTrader, o None si no se encuentra
        """
        base_reg_path = r'SOFTWARE\NinjaTrader, LLC'
        versions = ['NinjaTrader 8', 'NinjaTrader 7']
        
        for version in versions:
            reg_path = os.path.join(base_reg_path, version)
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path) as key:
                    i = 0
                    while True:
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            if 'cmp' in subkey_name:
                                with winreg.OpenKey(key, subkey_name) as subkey:
                                    personal_root, _ = winreg.QueryValueEx(subkey, 'PERSONAL_ROOT')
                                    logger.info(f"Encontrada ruta de NinjaTrader: {personal_root}")
                                    return personal_root
                        except OSError:
                            break
                        i += 1
            except OSError:
                continue
        
        logger.error("No se pudo encontrar la ruta de NinjaTrader en el registro")
        return None
    
    def connect_ati(self):
        """
        Conecta con NinjaTrader a través de la interfaz ATI mediante sockets.
        
        Returns:
            bool: True si la conexión fue exitosa, False en caso contrario
        """
        if self.is_connected:
            return True
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect(('localhost', self.ati_port))
            self.is_connected = True
            logger.info(f"Conectado a NinjaTrader ATI en puerto {self.ati_port}")
            return True
        except Exception as e:
            logger.error(f"Error al conectar con NinjaTrader ATI: {e}")
            self.is_connected = False
            self.socket = None
            return False
    
    def disconnect_ati(self):
        """
        Desconecta de la interfaz ATI.
        """
        if self.socket and self.is_connected:
            try:
                self.socket.close()
            except Exception as e:
                logger.error(f"Error al cerrar socket ATI: {e}")
            finally:
                self.socket = None
                self.is_connected = False
                logger.info("Desconectado de NinjaTrader ATI")
    
    def send_ati_command(self, command):
        """
        Envía un comando a NinjaTrader a través de la interfaz ATI.
        
        Args:
            command (str): Comando a enviar en formato ATI
            
        Returns:
            str: Respuesta de NinjaTrader, o None si hubo un error
        """
        if not self.is_connected:
            if not self.connect_ati():
                return None
        
        try:
            self.socket.sendall(f"{command}\r\n".encode())
            response = self.socket.recv(4096).decode().strip()
            logger.debug(f"Comando ATI enviado: {command}")
            logger.debug(f"Respuesta ATI: {response}")
            return response
        except Exception as e:
            logger.error(f"Error al enviar comando ATI: {e}")
            self.is_connected = False
            self.socket = None
            return None
    
    def execute_command(self, command):
        """
        Ejecuta un comando en NinjaTrader 8 a través de archivos de orden.
        Este método es una alternativa a la comunicación por sockets.
        
        Args:
            command (str): Comando en formato ATI
            
        Returns:
            bool: True si el comando se envió correctamente, False en caso contrario
        """
        if not self.personal_root:
            logger.error("No se pudo encontrar la ruta de NinjaTrader en el registro")
            return False
        
        incoming_dir = os.path.join(self.personal_root, 'incoming')
        if not os.path.exists(incoming_dir):
            try:
                os.makedirs(incoming_dir)
            except Exception as e:
                logger.error(f"Error al crear directorio incoming: {e}")
                return False
        
        file_name = os.path.join(incoming_dir, f'oif{uuid.uuid4()}.txt')
        try:
            with open(file_name, 'w') as f:
                f.write(command)
            logger.info(f"Comando enviado via archivo: {file_name}")
            logger.debug(f"Contenido del comando: {command}")
            return True
        except Exception as e:
            logger.error(f"Error al escribir archivo de orden: {e}")
            return False
    
    def market_buy(self, instrument, quantity=1, use_socket=True):
        """
        Envía una orden de compra a mercado.
        
        Args:
            instrument (str): Instrumento a operar (ej. "NQ 06-23")
            quantity (int): Cantidad de contratos
            use_socket (bool): Si es True, usa la conexión por socket, sino usa archivos
            
        Returns:
            bool: True si la orden se envió correctamente, False en caso contrario
        """
        command = f"PLACE;{self.account};{instrument};BUY;{quantity};MARKET;;;DAY;;;;"
        
        if use_socket:
            response = self.send_ati_command(command)
            return response is not None and "ERROR" not in response
        else:
            return self.execute_command(command)
    
    def market_sell(self, instrument, quantity=1, use_socket=True):
        """
        Envía una orden de venta a mercado.
        
        Args:
            instrument (str): Instrumento a operar (ej. "NQ 06-23")
            quantity (int): Cantidad de contratos
            use_socket (bool): Si es True, usa la conexión por socket, sino usa archivos
            
        Returns:
            bool: True si la orden se envió correctamente, False en caso contrario
        """
        command = f"PLACE;{self.account};{instrument};SELL;{quantity};MARKET;;;DAY;;;;"
        
        if use_socket:
            response = self.send_ati_command(command)
            return response is not None and "ERROR" not in response
        else:
            return self.execute_command(command)
    
    def limit_buy(self, instrument, quantity=1, price=0, use_socket=True):
        """
        Envía una orden de compra limitada.
        
        Args:
            instrument (str): Instrumento a operar (ej. "NQ 06-23")
            quantity (int): Cantidad de contratos
            price (float): Precio límite
            use_socket (bool): Si es True, usa la conexión por socket, sino usa archivos
            
        Returns:
            bool: True si la orden se envió correctamente, False en caso contrario
        """
        command = f"PLACE;{self.account};{instrument};BUY;{quantity};LIMIT;{price};;DAY;;;;"
        
        if use_socket:
            response = self.send_ati_command(command)
            return response is not None and "ERROR" not in response
        else:
            return self.execute_command(command)
    
    def limit_sell(self, instrument, quantity=1, price=0, use_socket=True):
        """
        Envía una orden de venta limitada.
        
        Args:
            instrument (str): Instrumento a operar (ej. "NQ 06-23")
            quantity (int): Cantidad de contratos
            price (float): Precio límite
            use_socket (bool): Si es True, usa la conexión por socket, sino usa archivos
            
        Returns:
            bool: True si la orden se envió correctamente, False en caso contrario
        """
        command = f"PLACE;{self.account};{instrument};SELL;{quantity};LIMIT;{price};;DAY;;;;"
        
        if use_socket:
            response = self.send_ati_command(command)
            return response is not None and "ERROR" not in response
        else:
            return self.execute_command(command)
    
    def stop_buy(self, instrument, quantity=1, price=0, use_socket=True):
        """
        Envía una orden de compra con stop.
        
        Args:
            instrument (str): Instrumento a operar (ej. "NQ 06-23")
            quantity (int): Cantidad de contratos
            price (float): Precio del stop
            use_socket (bool): Si es True, usa la conexión por socket, sino usa archivos
            
        Returns:
            bool: True si la orden se envió correctamente, False en caso contrario
        """
        command = f"PLACE;{self.account};{instrument};BUY;{quantity};STOP;{price};;DAY;;;;"
        
        if use_socket:
            response = self.send_ati_command(command)
            return response is not None and "ERROR" not in response
        else:
            return self.execute_command(command)
    
    def stop_sell(self, instrument, quantity=1, price=0, use_socket=True):
        """
        Envía una orden de venta con stop.
        
        Args:
            instrument (str): Instrumento a operar (ej. "NQ 06-23")
            quantity (int): Cantidad de contratos
            price (float): Precio del stop
            use_socket (bool): Si es True, usa la conexión por socket, sino usa archivos
            
        Returns:
            bool: True si la orden se envió correctamente, False en caso contrario
        """
        command = f"PLACE;{self.account};{instrument};SELL;{quantity};STOP;{price};;DAY;;;;"
        
        if use_socket:
            response = self.send_ati_command(command)
            return response is not None and "ERROR" not in response
        else:
            return self.execute_command(command)
    
    def close_position(self, instrument, use_socket=True):
        """
        Cierra todas las posiciones en un instrumento.
        
        Args:
            instrument (str): Instrumento a operar (ej. "NQ 06-23")
            use_socket (bool): Si es True, usa la conexión por socket, sino usa archivos
            
        Returns:
            bool: True si la orden se envió correctamente, False en caso contrario
        """
        command = f"CLOSEPOSITION;{self.account};{instrument};;;;;;;;;;"
        
        if use_socket:
            response = self.send_ati_command(command)
            return response is not None and "ERROR" not in response
        else:
            return self.execute_command(command)
    
    def cancel_orders(self, instrument, use_socket=True):
        """
        Cancela todas las órdenes pendientes para un instrumento.
        
        Args:
            instrument (str): Instrumento a operar (ej. "NQ 06-23")
            use_socket (bool): Si es True, usa la conexión por socket, sino usa archivos
            
        Returns:
            bool: True si la orden se envió correctamente, False en caso contrario
        """
        command = f"CANCELORDERS;{self.account};{instrument};;;;;;;;;;"
        
        if use_socket:
            response = self.send_ati_command(command)
            return response is not None and "ERROR" not in response
        else:
            return self.execute_command(command)
    
    def get_account_info(self):
        """
        Obtiene información de la cuenta.
        
        Returns:
            dict: Información de la cuenta o None si hay error
        """
        if not self.is_connected:
            if not self.connect_ati():
                return None
        
        command = f"GETACCOUNTDATA;{self.account};;;;;;;;;"
        response = self.send_ati_command(command)
        
        if response and "ERROR" not in response:
            parts = response.split(';')
            if len(parts) >= 6:
                account_info = {
                    'account': parts[0],
                    'cash_value': float(parts[1]) if parts[1] else 0,
                    'net_liquidation': float(parts[2]) if parts[2] else 0,
                    'buying_power': float(parts[3]) if parts[3] else 0,
                    'account_value': float(parts[4]) if parts[4] else 0,
                }
                return account_info
        
        logger.error(f"Error al obtener información de cuenta: {response}")
        return None
    
    def get_positions(self):
        """
        Obtiene las posiciones actuales.
        
        Returns:
            list: Lista de posiciones o None si hay error
        """
        if not self.is_connected:
            if not self.connect_ati():
                return None
        
        command = f"GETPOSITIONS;{self.account};;;;;;;;;"
        response = self.send_ati_command(command)
        
        if response and response != "null" and "ERROR" not in response:
            positions = []
            for line in response.split('\r\n'):
                if not line:
                    continue
                parts = line.split(';')
                if len(parts) >= 5:
                    position = {
                        'instrument': parts[0],
                        'position': int(parts[1]) if parts[1] else 0,
                        'avg_price': float(parts[2]) if parts[2] else 0,
                        'unrealized_pnl': float(parts[3]) if parts[3] else 0,
                        'realized_pnl': float(parts[4]) if parts[4] else 0,
                    }
                    positions.append(position)
            return positions
        
        logger.error(f"Error al obtener posiciones: {response}")
        return None
        
    def __del__(self):
        """Destructor para asegurar que la conexión se cierra correctamente."""
        self.disconnect_ati() 