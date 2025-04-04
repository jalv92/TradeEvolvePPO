"""
Custom trading environment for reinforcement learning.
Implements a Gymnasium environment for trading NQ futures.
"""

import os
import sys
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional, Union, Any

import logging
import warnings
from config.config import BASE_CONFIG, ENV_CONFIG
from evaluation.metrics import TradingMetrics
import traceback
import random
import matplotlib.pyplot as plt
import datetime
import math
from collections import deque

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """
    Custom Environment for trading NQ futures with risk management.
    Implements the Gymnasium interface.
    """
    
    metadata = {'render.modes': ['console']}
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 config: Dict[str, Any], 
                 initial_balance: float = 100000.0,
                 window_size: int = 60,
                 mode: str = 'train'):
        """
        Initialize the TradingEnv environment.
        
        Args:
            data (pd.DataFrame): Pandas DataFrame with OHLCV data
            config (Dict[str, Any]): Configuration parameters
            initial_balance (float, optional): Starting balance. Defaults to 100000.0.
            window_size (int, optional): Number of candles to include in observation. Defaults to 60.
            mode (str, optional): Mode, either 'train', 'validation', 'eval' or 'inference'. Defaults to 'train'.
        """
        super(TradingEnv, self).__init__()
        
        # Validar el modo
        valid_modes = ['train', 'validation', 'eval', 'inference', 'test']
        if mode not in valid_modes:
            raise ValueError(f"Mode '{mode}' not recognized. Valid modes: {valid_modes}")
            
        self.data = data
        self.config = config
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.mode = mode
        
        # Ajustes específicos para modo de evaluación
        if self.mode == 'eval':
            # Modificar config para modo eval para forzar más operaciones
            self.config = config.copy()
            self.config['force_action_prob'] = 0.7  # Probabilidad muy alta de forzar acciones
            self.config['position_change_threshold'] = 0.1  # Umbral muy bajo para cambios de posición
            self.config['min_hold_steps'] = 1  # Mínimo período de mantener posición
            self.config['position_cooldown'] = 1  # Mínimo cooldown entre posiciones
            self.config['force_min_hold'] = False  # Desactivar forzado de hold mínimo
            self.config['initial_exploration_steps'] = 100  # Más exploración inicial
            print(f"⚠️ Modo EVAL: Configuración modificada para forzar más operaciones")
            
        # Setup trading environment
        self.prices = data['close'].values
        
        # Manejar diferentes tipos de índices
        if isinstance(data.index, pd.DatetimeIndex):
            self.dates = data.index
        else:
            # Si el índice no es DatetimeIndex, usarlo directamente sin convertir
            # Esto es útil para pruebas con datos sintéticos que tienen índices numéricos
            self.dates = data.index
        
        # Validation
        if len(self.data) < self.window_size + 1:
            raise ValueError(f"El conjunto de datos debe tener al menos {self.window_size + 1} filas, pero tiene {len(self.data)}")
            
        # Define action space: Continuous between -1 and 1
        # -1 = full short, 0 = neutral, 1 = full long
        # If using SL/TP management, the second dimension controls trailing stop and breakeven
        if self.config.get('enable_sl_tp_management', False):
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0]),  # [position, sl_tp_management]
                high=np.array([1.0, 1.0]),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        
        # Define observation space
        # Let's define a custom observation space based on window size and features
        n_features = self._calculate_num_features()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.current_step = 0
        self.current_position = 0
        self.position_size = 0
        self.entry_price = 0.0
        self.entry_time = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.balance = self.initial_balance
        self.done = False
        self.trades = []
        self.trade_history = []
        self.position_history = []
        self.performance_history = []
        self.trade_active = False
        self.unrealized_pnl = 0.0
        
        # Initialize reward-related variables
        self.current_pnl = 0.0
        self.prev_pnl = 0.0
        self.prev_position = 0
        self.max_drawdown = 0.0
        self.trade_completed = False
        self.last_trade_pnl = 0.0
        self.direction_changed = False
        self.max_balance = self.initial_balance
        self.position_duration = 0
        self.inactive_steps = 0
        
        # Initialize metrics tracker
        self.metrics = TradingMetrics(initial_balance=self.initial_balance)
        
        # Initialize tracking for reward calculation and performance metrics
        self.profitable_steps = 0
        self.prev_unrealized_pnl = 0.0
        self.cooldown_counter = 0
        self._just_opened_position = False
        self._just_closed_position = False
        self._last_trade_pnl = 0.0
        
        # For SL/TP management
        self.trailing_stop_active = False
        self.entry_step = 0
        self.max_price_reached = 0.0
        self.min_price_reached = float('inf')
        
        # For tracking ticks
        self.positive_ticks = 0
        self.negative_ticks = 0
        self.positive_ticks_total = 0
        self.negative_ticks_total = 0
        
        # For tracking balance
        self.equity_curve = [self.initial_balance]
        
        # Store parameters
        self.log_rewards = self.config.get('log_reward_components', False)
        
        # Inicializar reward_config (extraído de config)
        self.reward_config = self.config.get('reward_config', {})
        
        # Asegurarnos de que config tiene window_size y features
        if 'window_size' not in self.config:
            self.config['window_size'] = self.window_size
            logger.info(f"Añadido window_size={self.window_size} a la configuración")

        if 'features' not in self.config:
            self.config['features'] = 25  # Valor predeterminado para features
            logger.info(f"Añadido features=25 a la configuración")
        
        # NUEVO: Recompensas retrasadas para mejor aprendizaje a largo plazo
        self.pending_rewards = []
        self.reward_delay_steps = self.config.get('reward_delay_steps', 5)  # Retraso predeterminado de 5 pasos
        
        # NUEVO: Memoria de posiciones rentables para tracking
        self.prev_unrealized_pnl = 0.0  # PnL anterior para comparación
        
        # NUEVO: Tiempo de enfriamiento entre operaciones
        self.min_hold_steps = self.config.get('min_hold_steps', 30)  # Duración mínima recomendada (aumentada a 30)
        self.position_cooldown = self.config.get('position_cooldown', 40)  # Tiempo de enfriamiento (aumentado a 40)
        
        # NUEVO: Flag para forzar mantener posiciones durante el tiempo mínimo
        self.force_min_hold = self.config.get('force_min_hold', True)  # Siempre forzar duración mínima
        
        # NUEVO: Penalización severa por operaciones cortas
        self.short_trade_penalty_factor = self.config.get('short_trade_penalty_factor', 25.0)  # Aumentado a 25.0
        
        # NUEVO: Factor de escala para recompensas por duración
        self.duration_scaling_factor = self.config.get('duration_scaling_factor', 8.0)  # Factor de escala aumentado a 8.0
        
        # NUEVO: Configuración para tamaño mínimo de operaciones
        self.min_sl_ticks = self.config.get('min_sl_ticks', 50)  # Stop loss mínimo en ticks (aumentado a 50)
        self.min_tp_ticks = self.config.get('min_tp_ticks', 50)  # Take profit mínimo en ticks (aumentado a 50)
        self.tick_size = self.config.get('tick_size', 0.25)  # Tamaño del tick (0.25 para NQ)
        self.enforce_min_trade_size = self.config.get('enforce_min_trade_size', True)  # Forzar tamaño mínimo
        self.reward_larger_trades = self.config.get('reward_larger_trades', True)  # Recompensar operaciones grandes
        
        # Definir columnas a normalizar (predeterminado: todas numéricas)
        self.normalize_columns = self.config.get('normalize_columns', ['open', 'high', 'low', 'close'])
        
        # Inicializar lista de columnas para features (sin timestamp)
        self.feature_columns = [col for col in self.data.columns if col != 'timestamp']
        
        # Set up price scaler for normalization
        if len(self.data) > 0 and 'close' in self.data.columns:
            close_prices = self.data['close'].values
            self.price_min = close_prices.min()
            self.price_max = close_prices.max()
            self.price_range = self.price_max - self.price_min
        else:
            # Valores predeterminados para DataFrame vacío
            logger.warning("DataFrame vacío o sin columna 'close', usando valores predeterminados para normalización")
            self.price_min = 0.0
            self.price_max = 1.0
            self.price_range = 1.0
        
        # Store data properties
        self.num_features = len(self.data.columns)
        self.feature_names = list(self.data.columns)
        self.dates = self.data.index.tolist() if hasattr(self.data.index, 'tolist') else list(range(len(self.data)))
        
        # Initialize position and state variables
        self.position_size = 0
        self.current_position = 0  # Alias para position_size
        self.current_step = 0
        self.done = False
        self.trades = []
        self.trade_history = []
        self.position_history = []
        self.performance_history = []
        self.trade_active = False
        self.unrealized_pnl = 0.0
        
        # Variables para cálculo de recompensa
        self.current_pnl = 0.0
        self.prev_pnl = 0.0
        self.prev_position = 0
        self.max_drawdown = 0.0
        self.trade_completed = False
        self.last_trade_pnl = 0.0
        self.direction_changed = False
        self.max_balance = initial_balance
        self.position_duration = 0
        self.min_account_balance = initial_balance * 0.5  # 50% del balance inicial
        
        # Añadir variable current_price para evitar AttributeError
        self.current_price = 0.0
        
        # Risk management parameters
        self.stop_loss_pct = config.get('stop_loss_pct', 2.0)
        self.take_profit_pct = config.get('take_profit_pct', 4.0)
        self.max_position = config.get('max_position', 5)
        self.max_drawdown_pct = config.get('max_drawdown_pct', 20.0)
        self.contract_size = config.get('contract_size', 1)
        self.commission_rate = config.get('commission_rate', 0.0)
        self.slippage = config.get('slippage', 0.0)
        
        # Current trade parameters
        self.entry_price = 0.0
        self.entry_time = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        
        # Nueva: Variables para gestión de stops avanzada
        self.original_stop_loss = 0.0     # Stop loss original para detectar ajustes a break-even
        self.trailing_stop_active = False  # Si el trailing stop está activo
        self.trailing_stop_distance = 0.0  # Distancia del trailing stop
        self.profit_streak = 0             # Contador de operaciones rentables consecutivas
        self.risk_reward_ratio = 0.0       # Ratio riesgo/beneficio de la operación actual
        
        # Performance tracking
        self.initial_net_worth = self.initial_balance
        self.net_worth = self.initial_net_worth
        self.max_net_worth = self.initial_net_worth
        self.returns = []
        self.drawdowns = []
        self.equity_curve = [self.initial_balance]
        
        # Variables para tracking de posiciones y trades
        self._just_opened_position = False
        self._just_closed_position = False
        self._last_trade_pnl = 0.0
        self._last_trade_action_initiated = False

        # Trading session state
        self.position_steps = 0
        self.inactive_steps = 0
        
        # Stop loss and take profit
        self.trailing_stop = None
        
        # Logger setup
        self.logger = logging.getLogger('trading_env')
        if self.logger.handlers == []:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Setup trades_logger
        self.trades_logger = logging.getLogger('trades')
        if not self.trades_logger.handlers:
            self.trades_logger.setLevel(logging.INFO)
            try:
                os.makedirs('logs', exist_ok=True)
                handler = logging.FileHandler('logs/trades.log')
                formatter = logging.Formatter('%(asctime)s - %(message)s')
                handler.setFormatter(formatter)
                self.trades_logger.addHandler(handler)
            except Exception as e:
                self.logger.error(f"Error al configurar trades_logger: {e}")
                # Usar el logger principal como fallback
                self.trades_logger = self.logger
        
        # Current prices at this step (will be updated in step())
        self.open_price = 0.0
        self.high_price = 0.0
        self.low_price = 0.0

        # Check that data has enough records - skip check in inference mode
        if mode != 'inference' and len(data) < self.window_size + 10:
            raise ValueError(f"Data has only {len(data)} records, need at least {self.window_size + 10} for training/validation/test")

        # Add detailed trade logging
        self.current_trade = None
        self.log_trades = self.config.get('log_trades', True)  # Flag para habilitar logging de trades

        # Variables to track reward components
        self._reward_components = {
            'pnl': 0.0,
            'risk': 0.0,
            'activity': 0.0,
            'size': 0.0,
            'win_rate': 0.0,
            'opportunity_cost': 0.0,
            'exploration': 0.0
        }

    def _calculate_num_features(self):
        """
        Calcula el número de características en la observación.
        
        Returns:
            int: Número total de características
        """
        # Características básicas del mercado (OHLCV)
        base_features = 5  # Open, High, Low, Close, Volume
        
        # Características técnicas adicionales que podemos calcular
        technical_features = self.config.get('technical_indicators', 0)
        
        # Estado actual de trading
        trading_state_features = 4  # Posición, Balance, PnL no realizado, Duration
        
        # Información de gestión de riesgo (SL/TP)
        risk_features = 0
        if self.config.get('enable_sl_tp_management', False):
            risk_features = 4  # SL actual, TP actual, trailing activo, distancia al SL/TP
            
        # Ventana de tiempo
        window_multiplier = self.window_size
        
        # Total de características
        if self.config.get('use_full_observation_space', True):
            # Cada barra tiene todas las características
            return (base_features + technical_features) * window_multiplier + trading_state_features + risk_features
        else:
            # Solo usamos el cierre de cada barra más el estado actual
            return window_multiplier + trading_state_features + risk_features

    def _open_position(self, direction, position_size=1):
        """
        Abre una posición en la dirección especificada.
        
        Args:
            direction (int): 1 para posición larga, -1 para posición corta
            position_size (int): Tamaño de la posición a abrir
            
        Returns:
            bool: True si la posición se abrió con éxito, False en caso contrario
        """
        if self.current_position != 0:
            logger.debug(f"Ya hay una posición abierta: {self.current_position}")
            return False
            
        # Obtener precio actual
        current_price = self.data.iloc[self.current_step]['close']
        self.current_price = current_price  # Guardar para fácil acceso
        
        # Validaciones de tamaño de posición
        if position_size <= 0:
            logger.warning(f"Tamaño de posición inválido: {position_size}")
            # Ajustar a 1 unidad como mínimo
            position_size = 1
            
        # Verificar posibles restricciones de margen
        margin_requirement = self.config.get('margin_requirement', 0.1)  # Valor predeterminado de 10%
        margin_required = current_price * position_size * margin_requirement
        
        if margin_required > self.balance:
            logger.warning(f"Margen insuficiente: requerido={margin_required:.2f}, balance={self.balance:.2f}")
            # CORRECCIÓN: En lugar de retornar, ajustar el tamaño de posición al máximo posible
            position_size = int(self.balance / (current_price * margin_requirement))
            if position_size <= 0:
                logger.error("No hay suficiente balance para abrir ninguna posición")
                return False
            logger.info(f"Ajustando tamaño de posición a {position_size} unidades")
        
        # Registrar la entrada
        self.entry_price = current_price
        self.current_position = direction * position_size
        
        # SISTEMA BASADO EN TICKS: Calcular niveles de stop loss y take profit usando ticks
        # Usar el número mínimo de ticks configurado 
        sl_ticks = self.min_sl_ticks
        tp_ticks = self.min_tp_ticks
        
        # Calcular las distancias en precio
        sl_distance = sl_ticks * self.tick_size
        tp_distance = tp_ticks * self.tick_size
        
        # Calcular precios de SL y TP basados en ticks
        if direction > 0:  # Posición larga
            self.stop_loss = self.entry_price - sl_distance
            self.take_profit = self.entry_price + tp_distance
        else:  # Posición corta
            self.stop_loss = self.entry_price + sl_distance
            self.take_profit = self.entry_price - tp_distance
            
        # Guardar referencia al stop loss original (para detectar ajustes a break-even)
        self.original_stop_loss = self.stop_loss
        
        # Reiniciar el estado del trailing stop
        self.trailing_stop_active = False
                
        # Registrar la operación incluyendo los valores de ticks
        trade_info = {
            'entry_time': self.current_step,
            'entry_step': self.current_step,
            'entry_price': self.entry_price,
            'position': self.current_position,
            'direction': 'long' if direction > 0 else 'short',
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'sl_ticks': sl_ticks,
            'tp_ticks': tp_ticks,
            'original_stop_loss': self.original_stop_loss,  # Guardar el SL original para comparaciones
            'trailing_activated': False,  # Flag para saber si se activó el trailing
            'breakeven_activated': False,  # Flag para saber si se movió el SL a break-even
            'active': True
        }
        
        # Actualizar listas de operaciones
        self.trades.append(trade_info)
        self.trade_history.append(trade_info.copy())
        self.trade_active = True
        
        # Registrar detalles de la posición para análisis
        if self.log_trades:
            self.trades_logger.info(
                f"OPEN {'LONG' if direction > 0 else 'SHORT'} @ {self.entry_price:.2f} | "
                f"Size: {position_size} | SL: {self.stop_loss:.2f} ({sl_ticks} ticks) | "
                f"TP: {self.take_profit:.2f} ({tp_ticks} ticks)"
            )
        
        # Reiniciar contadores relacionados con la operación
        self.entry_time = self.current_step
        self.position_duration = 0
        
        # Marcar como éxito
        return True

    def _close_position(self, close_price: Optional[float] = None):
        """
        Cierra la posición actual y actualiza el balance.

        Args:
            close_price (Optional[float]): El precio al que se cierra la posición.
                                           Si es None, usa el precio de cierre de la barra actual.
        """
        if self.current_position == 0:
            logger.warning("Intentando cerrar posición cuando no hay ninguna abierta")
            return False

        # Usar el precio de cierre proporcionado o el cierre de la barra actual
        effective_close_price = close_price if close_price is not None else self.data.iloc[self.current_step]['close']
        log_price_source = "SL/TP" if close_price is not None else "Close"

        logger.info(f"Cerrando posición {self.current_position} entrada a {self.entry_price:.2f}, precio de cierre ({log_price_source}): {effective_close_price:.2f}")

        # Calcular PnL
        if self.current_position > 0:  # Posición larga
            pnl = (effective_close_price - self.entry_price) * abs(self.current_position)
        else:  # Posición corta
            pnl = (self.entry_price - effective_close_price) * abs(self.current_position)

        # Deducir comisión
        commission_rate = self.config.get('commission_rate', 0.001)  # Valor predeterminado de 0.1%
        commission = abs(self.current_position) * effective_close_price * commission_rate
        net_pnl = pnl - commission

        # Calcular duración de la posición
        position_duration = 0
        if self.trades and 'entry_time' in self.trades[-1]:
            position_duration = self.current_step - self.trades[-1]['entry_time']
        
        # NUEVO: Factor de duración altamente mejorado para escalar el PnL
        duration_factor = 1.0
        
        # MEJORADO: Aplicar factor de duración más agresivo y solo a operaciones rentables
        if position_duration >= self.min_hold_steps and pnl > 0:
            # Escalar el PnL positivo según la duración (hasta 5x para posiciones largas)
            # Utilizar el factor configurado duration_scaling_factor
            max_duration_factor = self.duration_scaling_factor  # Por defecto 5.0
            
            # Escala exponencial para premiar posiciones más largas
            if position_duration > self.min_hold_steps * 2:
                # Crecimiento exponencial para posiciones muy largas (rentables)
                duration_factor = min(max_duration_factor, 
                                     1.0 + ((position_duration - self.min_hold_steps) / 10) ** 1.5)
            else:
                # Crecimiento lineal para posiciones de duración normal
                duration_factor = min(max_duration_factor, 
                                     1.0 + (position_duration - self.min_hold_steps) / 10)
                
            # NUEVO: Aplicar un bonus adicional si la posición ha sido consistentemente rentable
            if hasattr(self, 'profitable_steps') and self.profitable_steps > self.min_hold_steps:
                profitable_bonus = min(2.0, self.profitable_steps / self.min_hold_steps)
                duration_factor *= profitable_bonus
                logger.info(f"Bonus adicional por posición consistentemente rentable: {profitable_bonus:.2f}x")
                
            # Aplicar factor de duración
            pnl = pnl * duration_factor
            net_pnl = pnl - commission
                
            logger.info(f"PnL aumentado por duración: factor={duration_factor:.2f}, duración={position_duration}, PnL ajustado={net_pnl:.2f}")
        
        # Actualizar balance
        self.balance += net_pnl
        
        # Actualizar el historial de trades
        if self.trades:
            last_trade = self.trades[-1]
            last_trade.update({
                'exit_time': self.current_step,
                'exit_price': self.current_price,
                'pnl': pnl,
                'net_pnl': net_pnl,
                'commission': commission,
                'duration': position_duration,  # Registrar duración de la posición
                'duration_factor': duration_factor,  # Registrar factor de duración aplicado
                'profitable_steps': getattr(self, 'profitable_steps', 0)  # NUEVO: Registrar pasos rentables
            })
            
            # Guardar el PnL para cálculo de recompensa
            self._last_trade_pnl = net_pnl
            
            # Marcar que acabamos de cerrar posición para el cálculo de la recompensa
            self._just_closed_position = True
            
            # Actualizar trade completo flag
            self.trade_completed = True
        
        logger.info(f"Posición cerrada: PnL={pnl:.2f}, neto={net_pnl:.2f}, comisión={commission:.2f}, duración={position_duration}, nuevo balance={self.balance:.2f}")
        
        # Resetear variables de posición
        self.current_position = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        
        # NUEVO: Resetear contador de pasos rentables para la próxima operación
        self.profitable_steps = 0
        self.prev_unrealized_pnl = 0.0
        
        return True

    def _calculate_unrealized_pnl(self) -> float:
        """
        Calculate the unrealized PnL for the current position.
        
        Returns:
            float: Unrealized PnL
        """
        if self.position_size == 0:
            return 0.0
        
        price = self.data.iloc[self.current_step]['close']
        unrealized_pnl = (price - self.entry_price) * self.position_size * self.contract_size
        return unrealized_pnl
        
    def _get_observation(self):
        """
        Obtiene la observación actual del entorno.
        
        Returns:
            np.ndarray: Observación
        """
        # Asegurar que estamos dentro de los límites del dataset
        if self.current_step >= len(self.data) - 1:
            self.current_step = len(self.data) - 1
            
        # Obtener ventana de datos anteriores
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        
        # Extracción de características base (OHLCV)
        ohlcv_data = []
        for i in range(start_idx, end_idx):
            row = self.data.iloc[i]
            features = [
                row['open'], 
                row['high'], 
                row['low'], 
                row['close'], 
                row['volume'] if 'volume' in row else 0.0
            ]
            ohlcv_data.append(features)
            
        # Asegurar que tenemos la longitud correcta (padding si es necesario)
        if len(ohlcv_data) < self.window_size:
            # Padding con ceros al principio si faltan datos
            padding = [[0.0, 0.0, 0.0, 0.0, 0.0]] * (self.window_size - len(ohlcv_data))
            ohlcv_data = padding + ohlcv_data
            
        # Convertir a array de numpy
        ohlcv_array = np.array(ohlcv_data, dtype=np.float32)
        
        # Características del estado actual de trading
        position_normalized = self.current_position / self.config.get('max_position_size', 1)
        balance_normalized = self.balance / self.initial_balance - 1.0  # Cambio porcentual
        unrealized_pnl = self._calculate_unrealized_pnl() / self.balance if self.balance > 0 else 0
        duration_normalized = self.position_duration / self.config.get('max_trade_duration', 100) if self.current_position != 0 else 0
        
        trading_state = np.array([
            position_normalized,
            balance_normalized,
            unrealized_pnl,
            duration_normalized
        ], dtype=np.float32)
        
        # Características de gestión de riesgo (si están habilitadas)
        risk_features = np.array([], dtype=np.float32)
        if self.config.get('enable_sl_tp_management', False):
            # Calcular características relativas a SL/TP
            if self.current_position != 0:
                # Normalizar distancias
                current_price = self.data.iloc[self.current_step]['close']
                sl_distance = abs(current_price - self.stop_loss) / current_price if self.stop_loss > 0 else 0
                tp_distance = abs(self.take_profit - current_price) / current_price if self.take_profit > 0 else 0
                
                risk_features = np.array([
                    sl_distance,
                    tp_distance,
                    1.0 if self.trailing_stop_active else 0.0,
                    sl_distance / (tp_distance + 1e-6)  # Ratio SL/TP 
                ], dtype=np.float32)
            else:
                risk_features = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Combinar todas las características en una observación 1D
        if self.config.get('use_full_observation_space', True):
            # Aplanar OHLCV
            ohlcv_flat = ohlcv_array.flatten()
            # Concatenar todo
            observation = np.concatenate([ohlcv_flat, trading_state, risk_features])
        else:
            # Solo usar cierre de cada barra y estado actual
            close_prices = ohlcv_array[:, 3]  # Índice 3 = cierre
            observation = np.concatenate([close_prices, trading_state, risk_features])
            
        return observation
    
    def _calculate_reward(self, action=None, processed_action=None, is_inactive=False):
        """
        Calcula la recompensa para el paso actual basada en PnL, gestión de riesgo y actividad.
        
        Args:
            action: Acción original
            processed_action: Acción procesada
            is_inactive: Si el agente ha estado inactivo
            
        Returns:
            float: Recompensa calculada
        """
        # Inicializar componentes de recompensa
        reward_components = {
            'pnl': 0.0,
            'risk': 0.0,
            'activity': 0.0,
            'size': 0.0,
            'win_rate': 0.0,
            'opportunity_cost': 0.0,
            'exploration': 0.0
        }
        
        # Base para la recompensa
        base_reward = self.config.get('base_reward', 0.0)
        reward = base_reward
        
        # 1. Componente principal: PnL (realizado + no realizado)
        unrealized_pnl = self._calculate_unrealized_pnl() if self.current_position != 0 else 0
        realized_pnl = self.balance - self.initial_balance
        total_pnl = realized_pnl + unrealized_pnl
        
        # 2. Recompensa por PnL normalizado por el saldo inicial
        pnl_scale = self.config.get('pnl_scale', 1.0)
        pnl_reward = (total_pnl / self.initial_balance) * pnl_scale
        reward_components['pnl'] = pnl_reward
        reward += pnl_reward
        
        # 3. Penalización por inactividad prolongada
        if is_inactive:
            inactivity_penalty = self.config.get('inactivity_penalty', -0.1)
            reward_components['activity'] = inactivity_penalty
            reward += inactivity_penalty
        
        # 4. Penalización por drawdown extremo
        max_drawdown_pct = self.config.get('max_drawdown_pct', 0.05)
        current_drawdown = (self.max_balance - self.balance) / self.max_balance if self.max_balance > 0 else 0
        if current_drawdown > max_drawdown_pct:
            drawdown_penalty = -0.5 * current_drawdown / max_drawdown_pct
            reward_components['risk'] = drawdown_penalty
            reward += drawdown_penalty
        
        # 5. Penalización por sobretrading (muchas operaciones en poco tiempo)
        if hasattr(self, 'trades') and len(self.trades) > 50:
            recent_trades = sum(1 for t in self.trades[-50:] if t.get('exit_step', 0) > self.current_step - 50)
            if recent_trades > 25:  # Más de 25 operaciones en los últimos 50 pasos
                overtrade_penalty = self.config.get('overtrade_penalty', -0.5) * (recent_trades / 25)
                reward_components['opportunity_cost'] = overtrade_penalty
                reward += overtrade_penalty
        
        # Guardar componentes de recompensa para análisis
        self.reward_components = reward_components
        
        return reward
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional options for environment reset
            
        Returns:
            Tuple[np.ndarray, dict]: Observation and info dictionary
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset state variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.position_size = 0
        self.current_position = 0
        self.entry_price = 0.0
        self.entry_time = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.done = False
        self.trades = []
        self.trade_history = []
        self.position_history = []
        self.performance_history = []
        self.trade_active = False
        self.unrealized_pnl = 0.0
        
        # Reset reward-related variables
        self.current_pnl = 0.0
        self.prev_pnl = 0.0
        self.prev_position = 0
        self.max_drawdown = 0.0
        self.trade_completed = False
        self.last_trade_pnl = 0.0
        self.direction_changed = False
        self.max_balance = self.initial_balance
        self.position_duration = 0
        self.inactive_steps = 0
        
        # Reset metrics tracker
        self.metrics = TradingMetrics(initial_balance=self.initial_balance)
        
        # Reset counters for reward calculation
        self.profitable_steps = 0
        self.prev_unrealized_pnl = 0.0
        self.cooldown_counter = 0
        self._just_opened_position = False
        self._just_closed_position = False
        self._last_trade_pnl = 0.0
        
        # Get initial observation
        observation = self._get_observation()
        
        # Create info dictionary
        info = {
            'balance': self.balance,
            'position': self.position_size,
            'trades': len(self.trades),
            'step': self.current_step
        }
        
        return observation, info
        
    def step(self, action):
        """
        Avanzar un paso en el entorno.
        
        Args:
            action: Acción a realizar
            
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        info = {}
        done = False
        truncated = False
        
        # Guardar step actual
        self.current_step += 1
        
        # Verificar si es el último step
        if self.current_step >= len(self.data) - 1:
            done = True
            
        # Procesar la acción
        action_dict = self._process_action(action)
        action_type = action_dict['action_type']  # Usando action_type como está en el diccionario
        sl_tp_action = action_dict['sl_tp_action']
        
        # Verificar inactividad
        is_inactive = False
        if self.current_position == 0:
            self.inactive_steps += 1
            inactivity_threshold = self.config.get('inactivity_threshold', 100)
            if self.inactive_steps > inactivity_threshold:
                is_inactive = True
        else:
            self.inactive_steps = 0
            
        # Verificar cooldown para no abrir posiciones demasiado seguido
        in_cooldown = self.cooldown_counter > 0
        if in_cooldown:
            self.cooldown_counter -= 1
            
        # Verificar si estamos dentro del período mínimo de mantener posición
        in_min_hold_period = False
        if self.current_position != 0 and self.config.get('force_min_hold', False):
            min_hold_steps = self.config.get('min_hold_steps', 10)
            in_min_hold_period = self.position_duration < min_hold_steps
            
        # Verificar si el entorno permite exploración forzada
        force_action_prob = self.config.get('force_action_prob', 0.0)
        
        # Si hay cooldown activo, no permitir abrir nuevas posiciones
        if in_cooldown and action_type != 'hold':
            action_type = 'hold'  # Forzar mantener durante el cooldown
            logger.info(f"Bloqueando acción durante cooldown ({self.cooldown_counter} pasos restantes)")
            
        # NUEVO: Si estamos en período de hold mínimo, forzar mantener posición
        if in_min_hold_period:
            if (self.current_position > 0 and action_type != 'open_long') or \
               (self.current_position < 0 and action_type != 'open_short'):
                logger.info(f"Forzando mantener posición por período mínimo de hold ({self.position_duration}/{self.config.get('min_hold_steps')})")
                action_type = 'hold'
            
        # A veces, forzar acción aleatoria para exploración
        if self.mode == 'train' and self.current_position == 0 and np.random.random() < force_action_prob:
            # En lugar de usar 1 y 2, usar 'open_long' y 'open_short'
            forced_action = np.random.choice(['open_long', 'open_short'])
            logger.info(f"FORZANDO acción aleatoria: {forced_action} (probabilidad: {force_action_prob:.2f})")
            action_type = forced_action
            
        # Ejecutar la acción procesada
        if action_type == 'open_long':  # Comprar
            if self.current_position <= 0:  # Si no hay posición o hay corta
                # Cerrar posición corta si existe
                if self.current_position < 0:
                    reward_close = self._close_position(close_reason='manual')
                else:
                    reward_close = 0
                    
                # Abrir posición larga
                self._open_position(1)
                
        elif action_type == 'open_short':  # Vender
            if self.current_position >= 0:  # Si no hay posición o hay larga
                # Cerrar posición larga si existe
                if self.current_position > 0:
                    reward_close = self._close_position(close_reason='manual')
                else:
                    reward_close = 0
                    
                # Abrir posición corta
                self._open_position(-1)
                
        # AMPLIADO: Procesamiento de la gestión de SL/TP
        if self.current_position != 0 and self.config.get('enable_sl_tp_management', True):
            # Verificar si hay PnL positivo suficiente para permitir ciertas acciones
            unrealized_pnl = self._calculate_unrealized_pnl()
            pnl_pct = abs(unrealized_pnl) / (self.entry_price * abs(self.current_position))
            
            breakeven_threshold = self.config.get('breakeven_activation_threshold', 0.15) / 100.0
            trailing_threshold = self.config.get('trailing_activation_threshold', 0.3) / 100.0
            
            # Verificar si la operación tiene suficiente beneficio para cada acción
            can_activate_breakeven = pnl_pct >= breakeven_threshold and unrealized_pnl > 0
            can_activate_trailing = pnl_pct >= trailing_threshold and unrealized_pnl > 0
            
            # Decidir qué acción de gestión tomar
            if sl_tp_action > 0.66 and can_activate_trailing:  # Activar trailing stop
                if not self.trailing_stop_active:
                    self.trailing_stop_active = True
                    logger.info(f"Activando trailing stop en la posición (PnL: {unrealized_pnl:.2f}, {pnl_pct:.2%})")
                    # Actualizar el estado de la operación
                    if len(self.trades) > 0 and self.trades[-1].get('active', False):
                        self.trades[-1]['trailing_activated'] = True
                
            elif sl_tp_action > 0.33 and can_activate_breakeven:  # Mover SL a break-even
                if self.current_position > 0 and self.stop_loss < self.entry_price:
                    logger.info(f"Moviendo stop loss a break-even: {self.stop_loss:.2f} -> {self.entry_price:.2f}")
                    self.stop_loss = self.entry_price
                    # Actualizar el estado de la operación
                    if len(self.trades) > 0 and self.trades[-1].get('active', False):
                        self.trades[-1]['breakeven_activated'] = True
                        self.trades[-1]['stop_loss'] = self.entry_price
                        
                elif self.current_position < 0 and self.stop_loss > self.entry_price:
                    logger.info(f"Moviendo stop loss a break-even: {self.stop_loss:.2f} -> {self.entry_price:.2f}")
                    self.stop_loss = self.entry_price
                    # Actualizar el estado de la operación
                    if len(self.trades) > 0 and self.trades[-1].get('active', False):
                        self.trades[-1]['breakeven_activated'] = True
                        self.trades[-1]['stop_loss'] = self.entry_price
        
        # Actualizar precios
        self._update_price()
        
        # Si hay una posición abierta, verificar SL/TP y actualizar trailing stop si está activo
        reward_sl_tp = 0
        if self.current_position != 0:
            # Incrementar duración de la posición
            self.position_duration += 1
            
            # Actualizar high y low de la operación
            if self.current_position > 0:  # Long
                if self.current_price > self.high_price:
                    self.high_price = self.current_price
            else:  # Short
                if self.current_price < self.low_price:
                    self.low_price = self.current_price
                    
            # Activar trailing stop si el precio ha alcanzado suficiente beneficio y trailing está activo
            if self.trailing_stop_active:
                self._update_trailing_stop()
                
            # Verificar si se ha alcanzado SL o TP
            reward_sl_tp = self._check_sl_tp()
        
        # Obtener observación
        observation = self._get_observation()
        
        # Calcular recompensa
        reward = self._calculate_reward(action=action, processed_action=action_type, is_inactive=is_inactive)
        
        # Actualizar información adicional
        info.update({
            'step': self.current_step,
            'current_price': self.current_price,
            'current_position': self.current_position,
            'balance': self.balance,
            'unrealized_pnl': self._calculate_unrealized_pnl() if self.current_position != 0 else 0,
            'equity': self.balance + (self._calculate_unrealized_pnl() if self.current_position != 0 else 0),
            'trade_active': self.current_position != 0,
            'position_duration': self.position_duration if self.current_position != 0 else 0,
            'is_inactive': is_inactive,
            'entry_price': self.entry_price if self.current_position != 0 else 0,
            'stop_loss': self.stop_loss if self.current_position != 0 else 0,
            'take_profit': self.take_profit if self.current_position != 0 else 0,
            'processed_action': action_type,
            'trailing_active': self.trailing_stop_active
        })
        
        # Calcular métricas para el historial
        if self.current_position != 0:
            current_equity = self.balance + self._calculate_unrealized_pnl()
        else:
            current_equity = self.balance
        
        self.equity_curve.append(current_equity)
        
        return observation, reward, done, truncated, info
    
    def _update_trailing_stop(self):
        """
        Actualiza el trailing stop si está activo, utilizando los parámetros configurados.
        Ahora utiliza distancia en ticks en lugar de porcentaje para mayor precisión.
        """
        if not self.trailing_stop_active or self.current_position == 0:
            return
        
        current_price = self.data.iloc[self.current_step]['close']
        
        # Calcular distancia de trailing en precio basada en ticks configurados
        trail_distance_ticks = self.config.get('trailing_stop_distance_ticks', 20)
        trail_distance = trail_distance_ticks * self.tick_size
        
        # Para posiciones largas, mover el stop loss hacia arriba si el precio sube
        if self.current_position > 0:
            new_stop = current_price - trail_distance
            
            # Solo actualizar si el nuevo stop es mayor al actual
            if new_stop > self.stop_loss:
                old_stop = self.stop_loss
                self.stop_loss = new_stop
                
                # Calcular la distancia en ticks para el log
                distance_moved_ticks = round((new_stop - old_stop) / self.tick_size)
                
                # Registrar la actividad
                logger.info(f"Trailing stop ajustado: {old_stop:.2f} -> {new_stop:.2f} (+{distance_moved_ticks} ticks)")
                logger.info(f"Distancia al precio actual: {trail_distance_ticks} ticks ({(current_price - new_stop):.2f} puntos)")
                
                # Actualizar información de la operación activa con el nuevo stop loss
                if len(self.trades) > 0 and 'active' in self.trades[-1] and self.trades[-1]['active']:
                    self.trades[-1]['stop_loss'] = new_stop
                    self.trades[-1]['trailing_activated'] = True
                
        # Para posiciones cortas, mover el stop loss hacia abajo si el precio baja
        elif self.current_position < 0:
            new_stop = current_price + trail_distance
            
            # Solo actualizar si el nuevo stop es menor al actual
            if new_stop < self.stop_loss:
                old_stop = self.stop_loss
                self.stop_loss = new_stop
                
                # Calcular la distancia en ticks para el log
                distance_moved_ticks = round((old_stop - new_stop) / self.tick_size)
                
                # Registrar la actividad
                logger.info(f"Trailing stop ajustado: {old_stop:.2f} -> {new_stop:.2f} (-{distance_moved_ticks} ticks)")
                logger.info(f"Distancia al precio actual: {trail_distance_ticks} ticks ({(new_stop - current_price):.2f} puntos)")
                
                # Actualizar información de la operación activa con el nuevo stop loss
                if len(self.trades) > 0 and 'active' in self.trades[-1] and self.trades[-1]['active']:
                    self.trades[-1]['stop_loss'] = new_stop
                    self.trades[-1]['trailing_activated'] = True
    
    def _check_sl_tp(self):
        """
        Verifica si el precio ha llegado a los niveles de stop loss o take profit.
        Implementa un período mínimo obligatorio y un buffer para evitar cierres prematuros.
        """
        if self.current_position == 0:
            return 0.0
        
        # Obtener configuración
        min_hold_steps = self.config.get('min_hold_steps', 10)  # Mínimo 10 barras por defecto
        force_min_hold = self.config.get('force_min_hold', True)  # Forzar período mínimo
        sl_buffer_ticks = self.config.get('sl_buffer_ticks', 5)  # Buffer para SL (en ticks)
        tp_buffer_ticks = self.config.get('tp_buffer_ticks', 5)  # Buffer para TP (en ticks)
        
        # Convertir buffer de ticks a precio
        sl_buffer = sl_buffer_ticks * self.tick_size
        tp_buffer = tp_buffer_ticks * self.tick_size
        
        # Verificar si estamos en período mínimo obligatorio
        in_min_hold_period = self.position_duration < min_hold_steps
        
        # Si estamos en período mínimo y está activado force_min_hold, no verificar SL/TP
        if in_min_hold_period and force_min_hold:
            if self.position_duration == 0:  # Solo loggear una vez al inicio
                logger.info(f"En período mínimo obligatorio ({min_hold_steps} barras). SL/TP no activos.")
            return 0.0
        
        current_high = self.data.iloc[self.current_step]['high']
        current_low = self.data.iloc[self.current_step]['low']
        
        # Para posiciones largas
        if self.current_position > 0:
            # Verificar si el precio bajo ha tocado o cruzado el stop loss (con buffer)
            effective_sl = self.stop_loss - sl_buffer  # Dar un margen adicional
            if current_low <= effective_sl:
                logger.info(f"Stop Loss alcanzado para posición larga: {self.stop_loss:.2f} (precio bajo: {current_low:.2f})")
                # Registrar diagnóstico
                logger.info(f"Diagnóstico SL: duración={self.position_duration}, " +
                           f"distancia_original={self.entry_price-self.stop_loss:.2f}, " +
                           f"distancia_efectiva={self.entry_price-effective_sl:.2f}")
                # Cerrar a precio de stop loss original (no el efectivo con buffer)
                return self._close_position(close_price=self.stop_loss)

            # Verificar si el precio alto ha tocado o cruzado el take profit (con buffer)
            effective_tp = self.take_profit + tp_buffer  # Dar un margen adicional
            if current_high >= effective_tp:
                logger.info(f"Take Profit alcanzado para posición larga: {self.take_profit:.2f} (precio alto: {current_high:.2f})")
                # Registrar diagnóstico
                logger.info(f"Diagnóstico TP: duración={self.position_duration}, " +
                           f"distancia_original={self.take_profit-self.entry_price:.2f}, " +
                           f"distancia_efectiva={effective_tp-self.entry_price:.2f}")
                # Cerrar a precio de take profit original (no el efectivo con buffer)
                return self._close_position(close_price=self.take_profit)

        # Para posiciones cortas
        elif self.current_position < 0:
            # Verificar si el precio alto ha tocado o cruzado el stop loss (con buffer)
            effective_sl = self.stop_loss + sl_buffer  # Dar un margen adicional
            if current_high >= effective_sl:
                logger.info(f"Stop Loss alcanzado para posición corta: {self.stop_loss:.2f} (precio alto: {current_high:.2f})")
                # Registrar diagnóstico
                logger.info(f"Diagnóstico SL: duración={self.position_duration}, " +
                           f"distancia_original={self.stop_loss-self.entry_price:.2f}, " +
                           f"distancia_efectiva={effective_sl-self.entry_price:.2f}")
                # Cerrar a precio de stop loss original (no el efectivo con buffer)
                return self._close_position(close_price=self.stop_loss)

            # Verificar si el precio bajo ha tocado o cruzado el take profit (con buffer)
            effective_tp = self.take_profit - tp_buffer  # Dar un margen adicional
            if current_low <= effective_tp:
                logger.info(f"Take Profit alcanzado para posición corta: {self.take_profit:.2f} (precio bajo: {current_low:.2f})")
                # Registrar diagnóstico
                logger.info(f"Diagnóstico TP: duración={self.position_duration}, " +
                           f"distancia_original={self.entry_price-self.take_profit:.2f}, " +
                           f"distancia_efectiva={self.entry_price-effective_tp:.2f}")
                # Cerrar a precio de take profit original (no el efectivo con buffer)
                return self._close_position(close_price=self.take_profit)
        
        return 0.0
        
    def get_performance_summary(self):
        """
        Obtiene un resumen del rendimiento del entorno durante la sesión de trading.
        
        Returns:
            Dict: Diccionario con métricas de rendimiento.
        """
        # Contabilizar operaciones ganadoras y perdedoras
        total_trades = len([t for t in self.trades if 'exit_price' in t])
        winning_trades = len([t for t in self.trades if 'net_pnl' in t and t['net_pnl'] > 0])
        losing_trades = len([t for t in self.trades if 'net_pnl' in t and t['net_pnl'] <= 0])
        
        # Calcular win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calcular profit factor (ganancia total / pérdida total)
        total_profit = sum([t['net_pnl'] for t in self.trades if 'net_pnl' in t and t['net_pnl'] > 0])
        total_loss = sum([abs(t['net_pnl']) for t in self.trades if 'net_pnl' in t and t['net_pnl'] < 0])
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0.0
        
        # Calcular promedio de ganancia/pérdida
        avg_profit = total_profit / winning_trades if winning_trades > 0 else 0.0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0.0
        
        # Calcular duración media de las operaciones
        avg_duration = sum([t.get('duration', 0) for t in self.trades if 'duration' in t]) / total_trades if total_trades > 0 else 0.0
        
        # Calcular expectativa (expectancy)
        expectancy = (win_rate * avg_profit) - ((1 - win_rate) * avg_loss) if (winning_trades > 0 or losing_trades > 0) else 0.0
        
        # Obtener otros valores importantes
        balance_final = self.balance
        return_pct = (balance_final - self.initial_balance) / self.initial_balance * 100 if self.initial_balance > 0 else 0.0
        max_drawdown = self.max_drawdown * 100  # Convertir a porcentaje
        
        # Calcular métricas adicionales - CORREGIDO: calcular la recompensa media real, no el balance promedio
        # Calculamos la diferencia de patrimonio neto respecto al inicial para representar mejor la recompensa acumulada
        if self.equity_curve:
            equity_changes = [self.equity_curve[i] - self.equity_curve[i-1] for i in range(1, len(self.equity_curve))]
            avg_reward = sum(equity_changes) / len(equity_changes) if equity_changes else 0.0
        else:
            avg_reward = 0.0
        
        # Devolver resumen completo
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'avg_duration': avg_duration,
            'expectancy': expectancy,
            'final_balance': balance_final,
            'return_pct': return_pct,
            'max_drawdown': max_drawdown,
            'mean_reward': avg_reward,
            'equity_curve': self.equity_curve
        }

    def _process_action(self, action):
        """
        Procesa la acción recibida y realiza los cambios correspondientes en el entorno.
        
        Args:
            action: Acción a realizar
            
        Returns:
            Dict: Información sobre la acción procesada
        """
        action_dict = {
            'raw_action': action.copy() if isinstance(action, np.ndarray) else action,
            'processed_action': None,
            'action_type': None,
            'position_size': 0,
            'sl_tp_action': 0
        }
        
        # Para modo de inferencia o evaluación, el agente debe ser más proactivo
        if self.mode in ['inference', 'eval']:
            force_action = np.random.rand() < 0.8  # 80% de probabilidad en evaluación
        else:
            # Durante entrenamiento, a veces forzamos acciones para exploración
            force_action = np.random.rand() < self.config.get('force_action_prob', 0.5)  # Aumentado a 0.5
        
        # Primera dimensión: Tamaño y dirección de la posición, segunda dimensión: gestión SL/TP
        if isinstance(action, np.ndarray) and len(action) == 2:
            position_action = action[0]  # Valor entre -1.0 y 1.0
            sl_tp_action = action[1] if len(action) > 1 else 0.0  # Valor entre 0.0 y 1.0
        else:
            position_action = action
            sl_tp_action = 0.0
            
        action_dict['sl_tp_action'] = sl_tp_action
        
        # Si estamos en fase de exploración inicial, tomar acciones aleatorias con mayor frecuencia
        if self.current_step < self.config.get('initial_exploration_steps', 500) and self.mode == 'train':
            # Aumentar probabilidad de actuar durante exploración inicial a 80%
            if np.random.rand() < 0.8:  # Aumentado de 0.4 a 0.8
                position_action = np.random.uniform(-1.0, 1.0)
                force_action = True
                
        # PRUEBA: Intercalar forzado de acciones aleatorias para exploración
        elif self.mode == 'train' and self.current_step % 10 == 0:  # Cada 10 pasos
            if np.random.rand() < 0.7:  # 70% de probabilidad
                position_action = np.random.uniform(-1.0, 1.0)
                force_action = True
        
        # NUEVO: En evaluación, amplificar la acción para hacerla más decisiva
        if self.mode in ['eval', 'inference']:
            # Amplificar la acción para hacerla más extrema (más cerca de -1 o 1)
            position_action = np.sign(position_action) * np.power(abs(position_action), 0.5)
            
            # Cada 5 pasos en evaluación, considerar una acción aleatoria
            if self.current_step % 5 == 0 and np.random.rand() < 0.5:
                position_action = np.random.uniform(-1.0, 1.0)
                force_action = True
        
        # Determinar tipo de acción y cambio de posición
        action_processed = position_action  # Valor normalizado entre -1.0 y 1.0
        
        # Cuando la acción es cercana a cero y no estamos forzando acción,
        # interpretamos como "mantener posición actual"
        threshold = self.config.get('position_change_threshold', 0.2)  # Reducido de 0.4 a 0.2
        
        # Reducir el umbral en evaluación para facilitar toma de acciones
        if self.mode in ['eval', 'inference']:
            threshold = 0.1  # Umbral más bajo en evaluación
        
        # Si estamos forzando acción o el valor absoluto supera el umbral, realizamos cambio
        if force_action or abs(action_processed) > threshold:
            # Si posición actual es 0 (sin posición), abrir nueva
            if self.current_position == 0:
                # Valor positivo = posición larga, valor negativo = posición corta
                if action_processed > threshold:
                    action_dict['action_type'] = 'open_long'
                    action_dict['position_size'] = self.position_size
                elif action_processed < -threshold:
                    action_dict['action_type'] = 'open_short'
                    action_dict['position_size'] = -self.position_size
                else:
                    action_dict['action_type'] = 'hold'
                    action_dict['position_size'] = 0
            # Si tenemos posición larga abierta
            elif self.current_position > 0:
                # Si la acción sugiere cambiar a corto o cerrar
                if action_processed < -threshold:
                    action_dict['action_type'] = 'close'
                    action_dict['position_size'] = 0
                else:
                    action_dict['action_type'] = 'hold'
                    action_dict['position_size'] = self.current_position
            # Si tenemos posición corta abierta
            elif self.current_position < 0:
                # Si la acción sugiere cambiar a largo o cerrar
                if action_processed > threshold:
                    action_dict['action_type'] = 'close'
                    action_dict['position_size'] = 0
                else:
                    action_dict['action_type'] = 'hold'
                    action_dict['position_size'] = self.current_position
        else:
            # Mantener posición actual
            action_dict['action_type'] = 'hold'
            action_dict['position_size'] = self.current_position
            
        action_dict['processed_action'] = action_processed
        return action_dict

    def _update_price(self):
        """Actualiza los precios actuales desde los datos"""
        self.current_price = self.data.iloc[self.current_step]['close']
        self.open_price = self.data.iloc[self.current_step]['open']
        self.high_price = self.data.iloc[self.current_step]['high']
        self.low_price = self.data.iloc[self.current_step]['low']
        
        # Actualizar balance máximo para cálculos de drawdown
        current_equity = self.balance
        if self.current_position != 0:
            current_equity += self._calculate_unrealized_pnl()
            
        self.max_balance = max(self.max_balance, current_equity)
