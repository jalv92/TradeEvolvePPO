"""
Custom trading environment for reinforcement learning.
Implements a Gymnasium environment for trading NQ futures.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional, Union, Any

import logging
import warnings
from config.config import BASE_CONFIG, ENV_CONFIG
from evaluation.metrics import TradingMetrics

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
        Initialize the trading environment.
        
        Args:
            data (pd.DataFrame): DataFrame with financial data
            config (Dict[str, Any]): Environment configuration
            initial_balance (float, optional): Initial account balance. Defaults to 100000.0.
            window_size (int, optional): Size of the observation window. Defaults to 60.
            mode (str, optional): Environment mode ('train', 'validation', 'test'). Defaults to 'train'.
        """
        super(TradingEnv, self).__init__()
        
        # Store parameters
        self.data = data.copy()
        self.config = config
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.window_size = config.get('window_size', window_size)  # Usar el valor de config si existe, o el parámetro como respaldo
        self.mode = mode
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
        
        # Current prices at this step (will be updated in step())
        self.open_price = 0.0
        self.high_price = 0.0
        self.low_price = 0.0

        # Check that data has enough records - skip check in inference mode
        if mode != 'inference' and len(data) < self.window_size + 10:
            raise ValueError(f"Data has only {len(data)} records, need at least {self.window_size + 10} for training/validation/test")

        # Define action and observation space
        # Espacio de acción de 2 dimensiones: [posición, gestión_sl_tp]
        # Dimension 0: Continuous value from -1.0 to 1.0 representing position size and direction
        # Dimension 1: Continuous value from 0.0 to 1.0 for SL/TP management
        # - 0.0 to 0.33: maintain current SL/TP
        # - 0.33 to 0.66: move SL to break-even
        # - 0.66 to 1.0: activate trailing stop
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Observation space: Market data for window_size steps
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size, self.config['features']),
            dtype=np.float32
        )

        # Initialize metrics tracker
        self.metrics = TradingMetrics(initial_balance=self.initial_balance)
        
        # Add detailed trade logging
        self.current_trade = None

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

    def _get_observation(self) -> np.ndarray:
        """
        Genera la observación actual para el agente.
        La observación consiste en una ventana de datos históricos
        junto con información sobre la posición actual.
        
        Returns:
            np.ndarray: Observación en forma de matriz 2D (window_size, features)
        """
        try:
            # Verificar si hay datos disponibles
            if self.data.empty:
                logger.warning("DataFrame vacío, generando observación con ceros")
                # Generar exactamente el número correcto de características (25)
                return np.zeros((self.config["window_size"], self.config["features"]), dtype=np.float32)
                
            # Verificar si current_step está dentro de los límites
            if self.current_step < 0 or self.current_step >= len(self.data):
                logger.warning(f"current_step ({self.current_step}) fuera de los límites del DataFrame (0-{len(self.data)-1 if len(self.data) > 0 else 0})")
                # Ajustar current_step al último índice válido si está fuera de rango
                if len(self.data) > 0:
                    self.current_step = len(self.data) - 1
                else:
                    return np.zeros((self.config["window_size"], self.config["features"]), dtype=np.float32)
            
            if self.mode == "inference":
                # Para inferencia, usar window_data
                # Obtener las columnas que están disponibles
                available_cols = self.data.columns.tolist()
                logger.debug(f"Usando columnas disponibles: {available_cols}")
                
                # Verificar si tenemos el número correcto de columnas
                if len(available_cols) > self.config["features"]:
                    # Si hay más, tomar solo las primeras config["features"]
                    logger.warning(f"Demasiadas columnas: {len(available_cols)}, truncando a {self.config['features']}")
                    available_cols = available_cols[:self.config["features"]]
                    data_subset = self.data[available_cols]
                elif len(available_cols) < self.config["features"]:
                    # Si hay menos, añadir columnas de ceros
                    logger.warning(f"Columnas insuficientes: {len(available_cols)}, añadiendo columnas de ceros hasta llegar a {self.config['features']}")
                    data_subset = self.data.copy()
                    for i in range(len(available_cols), self.config["features"]):
                        data_subset[f'feature_{i}'] = 0.0
                else:
                    data_subset = self.data
                
                # Verificar que tenemos exactamente el número requerido de columnas
                assert len(data_subset.columns) == self.config["features"], f"Error: {len(data_subset.columns)} columnas en lugar de {self.config['features']}"
                
                # Extraer una ventana de datos
                data_window = data_subset.iloc[-self.config["window_size"]:] if len(data_subset) >= self.config["window_size"] else data_subset
                
                # Si la ventana es más pequeña que window_size, rellenar con ceros
                if len(data_window) < self.config["window_size"]:
                    logger.debug(f"Rellenando datos: ventana de tamaño {len(data_window)} < window_size {self.config['window_size']}")
                    # Crear un array de ceros
                    padded_window = np.zeros((self.config["window_size"], self.config["features"]), dtype=np.float32)
                    # Llenar la parte final con los datos disponibles
                    padded_window[-len(data_window):] = data_window.values
                    return padded_window
                
                # Verificación final antes de devolver
                final_observation = data_window.values.astype(np.float32)
                if final_observation.shape[1] != self.config["features"]:
                    logger.error(f"Error en forma de observación: {final_observation.shape}")
                    # Corrección forzada si necesario
                    if final_observation.shape[1] > self.config["features"]:
                        final_observation = final_observation[:, :self.config["features"]]
                    else:
                        padded_obs = np.zeros((final_observation.shape[0], self.config["features"]), dtype=np.float32)
                        padded_obs[:, :final_observation.shape[1]] = final_observation
                        final_observation = padded_obs
                
                return final_observation
            else:
                # Para otros modos, usar índice actual
                start_idx = max(0, self.current_step - self.config["window_size"] + 1)
                end_idx = self.current_step + 1
                
                # Asegurar que tenemos suficientes datos
                if start_idx == 0 and end_idx - start_idx < self.config["window_size"]:
                    # Rellenar con ceros al inicio si es necesario
                    padding = self.config["window_size"] - (end_idx - start_idx)
                    normalized_window = np.zeros((self.config["window_size"], self.config["features"]), dtype=np.float32)
                    
                    # Si hay datos disponibles, copiarlos
                    if end_idx > start_idx:
                        data_slice = self.data.iloc[start_idx:end_idx]
                        
                        # Asegurar que tiene el número correcto de columnas
                        if len(data_slice.columns) > self.config["features"]:
                            data_slice = data_slice.iloc[:, :self.config["features"]]
                        elif len(data_slice.columns) < self.config["features"]:
                            # Asegurar que tenemos exactamente 25 columnas de características (requerido para el espacio de observación)
                            num_columns = data_slice.shape[1]
                            if num_columns < 25:
                                # Añadir columnas adicionales con ceros si hacen falta dimensiones
                                for i in range(num_columns, 25):
                                    # Usar .loc para evitar SettingWithCopyWarning
                                    data_slice.loc[:, f'feature_{i}'] = 0.0
                        
                        # Copiar los datos disponibles a la parte final
                        normalized_window[padding:] = data_slice.values
                    
                    return normalized_window
                else:
                    # Obtener slice 
                    data_slice = self.data.iloc[start_idx:end_idx]
                    
                    # Asegurar que tiene el número correcto de columnas
                    if len(data_slice.columns) > self.config["features"]:
                        data_slice = data_slice.iloc[:, :self.config["features"]]
                    elif len(data_slice.columns) < self.config["features"]:
                        # Asegurar que tenemos exactamente 25 columnas de características (requerido para el espacio de observación)
                        num_columns = data_slice.shape[1]
                        if num_columns < 25:
                            # Añadir columnas adicionales con ceros si hacen falta dimensiones
                            for i in range(num_columns, 25):
                                # Usar .loc para evitar SettingWithCopyWarning
                                data_slice.loc[:, f'feature_{i}'] = 0.0
                    
                    # Verificación final
                    final_observation = data_slice.values.astype(np.float32)
                    if final_observation.shape[1] != self.config["features"]:
                        logger.error(f"Error en forma de observación: {final_observation.shape}")
                        # Corrección forzada si necesario
                        if final_observation.shape[1] > self.config["features"]:
                            final_observation = final_observation[:, :self.config["features"]]
                        else:
                            padded_obs = np.zeros((final_observation.shape[0], self.config["features"]), dtype=np.float32)
                            padded_obs[:, :final_observation.shape[1]] = final_observation
                            final_observation = padded_obs
                    
                    return final_observation
                    
        except Exception as e:
            logger.error(f"Error en _get_observation: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # En caso de error, devolver una matriz de ceros con la forma correcta
            return np.zeros((self.config["window_size"], self.config["features"]), dtype=np.float32)
    
    def _calculate_reward(self):
        """Calculate the reward for the current step."""
        # Inicializar con la penalización base por cada paso
        reward = self.reward_config.get('base_reward', -0.05)
        
        # 1. Recompensa por PnL (normalizado)
        pnl_change = self.current_pnl - self.prev_pnl
        normalized_pnl = pnl_change / (self.initial_balance * 0.01) if self.initial_balance > 0 else 0
        pnl_reward = normalized_pnl * self.reward_config.get('pnl_weight', 3.0)
        reward += pnl_reward
        
        # 2. Penalización por drawdown
        if self.max_drawdown > 0:
            drawdown_penalty = -self.max_drawdown * self.reward_config.get('drawdown_weight', 0.05)
            reward += drawdown_penalty
        
        # 3. Bonus por completar operaciones
        if self.trade_completed:
            trade_bonus = self.reward_config.get('trade_completion_bonus', 5.0)
            reward += trade_bonus
            
            # Bonus adicional si es una operación ganadora
            if self.last_trade_pnl > 0:
                reward += trade_bonus * 0.5  # 50% adicional para trades ganadores
        
        # 4. Bonus por cambio de dirección (diversificación en estrategia)
        if self.direction_changed:
            direction_bonus = self.reward_config.get('direction_change_bonus', 0.2)
            reward += direction_bonus
        
        # 5. Factor de crecimiento de capital
        if self.balance > self.max_balance:
            capital_growth_reward = (self.balance - self.max_balance) / self.initial_balance
            reward += capital_growth_reward
            # Actualizar max_balance
            self.max_balance = self.balance
        
        # 6. Penalización por inactividad
        if self.inactive_steps > 0:
            inactivity_penalty = -self.inactive_steps * self.reward_config.get('inactivity_weight', 2.0) / 100
            reward += inactivity_penalty
        
        # 7. Penalización por mantener posiciones demasiado tiempo
        if self.position_duration > 20 and self.current_position != 0:
            holding_penalty = -0.01 * (self.position_duration - 20)
            reward += holding_penalty
        
        # 8. Factor de escala final
        scale_factor = self.reward_config.get('scale_factor', 10.0)
        reward *= scale_factor
        
        return reward

    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action (int): 0 = close position, 1 = buy/long, 2 = sell/short
        
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        # ---- Gestión de acciones forzadas ----
        exploration_steps = self.reward_config.get('exploration_steps', 1000000)
        exploration_prob = self.reward_config.get('exploration_prob', 0.4)
        inactivity_threshold = self.reward_config.get('inactivity_threshold', 50)
        
        force_action = False
        forced_action = action
        
        # Forzar exploración durante los primeros pasos o después de inactividad
        if (self.current_step < exploration_steps and np.random.random() < exploration_prob) or \
           (self.inactive_steps > inactivity_threshold and np.random.random() < 0.8):
            force_action = True
            if self.current_position == 0:  # Si no tenemos posición
                # 80% probabilidad de abrir posición (larga o corta aleatoriamente)
                if np.random.random() < 0.8:
                    forced_action = np.random.choice([1, 2])  # 1=long, 2=short
            else:  # Si ya tenemos posición
                # Forzar cierre si la posición se ha mantenido demasiado tiempo
                if self.position_duration > 30 and np.random.random() < 0.7:
                    forced_action = 0  # Cerrar posición
                # De lo contrario, mantener o cambiar de dirección
                elif np.random.random() < 0.3:
                    # Cambiar de dirección (de long a short o viceversa)
                    forced_action = 2 if self.current_position > 0 else 1
        
        # Usar acción forzada si corresponde
        action_taken = forced_action if force_action else action
        
        # Extraer valor escalar de la acción si es un array
        if isinstance(action_taken, (np.ndarray, list)):
            if len(action_taken) > 0:
                action_value = action_taken[0]
            else:
                action_value = 0
        else:
            action_value = action_taken
        
        # ---- Ejecución de la acción ----
        self.prev_pnl = self.current_pnl
        self.prev_position = self.current_position
        
        # Establecer valores predeterminados
        self.trade_completed = False
        self.direction_changed = False
        self.last_trade_pnl = 0
        
        # Actualizar contadores
        self.current_step += 1
        if self.current_position != 0:
            self.position_duration += 1
        
        # Si no hay operación activa y no abrimos una, incrementar inactividad
        if self.current_position == 0 and action_value == 0:
            self.inactive_steps += 1
        else:
            self.inactive_steps = 0  # Reiniciar contador si hay actividad
        
        # Ejecutar acción en el mercado simulado
        if action_value == 0:  # Cerrar posición
            if self.current_position != 0:
                # Calcular PnL al cerrar
                self.last_trade_pnl = self._close_position()
                self.trade_completed = True
                self.position_duration = 0
        elif action_value == 1:  # Comprar/Long
            if self.current_position <= 0:  # Si estamos en corto o sin posición
                if self.current_position < 0:
                    # Cerrar posición corta primero
                    self.last_trade_pnl = self._close_position()
                    self.trade_completed = True
                # Obtener el precio actual
                price = self.data.iloc[self.current_step]['close']
                # Abrir posición larga
                self._open_position(1, price)
                self.position_duration = 0
                if self.prev_position < 0:
                    self.direction_changed = True
        elif action_value == 2:  # Vender/Short
            if self.current_position >= 0:  # Si estamos en largo o sin posición
                if self.current_position > 0:
                    # Cerrar posición larga primero
                    self.last_trade_pnl = self._close_position()
                    self.trade_completed = True
                # Obtener el precio actual
                price = self.data.iloc[self.current_step]['close']
                # Abrir posición corta
                self._open_position(-1, price)
                self.position_duration = 0
                if self.prev_position > 0:
                    self.direction_changed = True
        
        # Actualizar estado del mercado (avanzar al siguiente tick)
        reached_end = not self._update_market_state()
        
        # Calcular recompensa
        reward = self._calculate_reward()
        
        # Actualizar observación
        self._update_observation()
        
        # Verificar condiciones de finalización
        done = reached_end or self.balance <= self.min_account_balance
        
        # Crear info adicional para logging y debugging
        info = {
            'current_step': self.current_step,
            'balance': self.balance,
            'position': self.current_position,
            'pnl': self.current_pnl,
            'trades': len(self.trades),
            'action_taken': action_taken,
            'forced_action': force_action,
            'inactive_steps': self.inactive_steps
        }
        
        # Retornar en formato estándar de Gymnasium
        return self.observation, reward, done, False, info
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Reset random generator if seed provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset position and state variables
        self.position_size = 0
        self.current_step = self.window_size - 1  # Set to start after window_size
        self.done = False
        self.trades = []
        self.trade_history = []
        self.position_history = []
        self.performance_history = []
        self.trade_active = False
        self.unrealized_pnl = 0.0
        
        # Reset balance
        self.balance = self.initial_balance
        
        # Reset trade parameters
        self.entry_price = 0.0
        self.entry_time = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        
        # Reset advanced risk management variables
        self.original_stop_loss = 0.0
        self.trailing_stop_active = False
        self.trailing_stop_distance = 0.0
        self.profit_streak = 0
        self.risk_reward_ratio = 0.0
        
        # Reset performance tracking
        self.initial_net_worth = self.initial_balance
        self.net_worth = self.initial_net_worth
        self.max_net_worth = self.initial_net_worth
        self.returns = []
        self.drawdowns = []
        self.equity_curve = [self.initial_balance]
        
        # Reset trade flags
        self._just_opened_position = False
        self._just_closed_position = False
        self._last_trade_pnl = 0.0
        self._last_trade_action_initiated = False
        
        # Reset step counters
        self.position_steps = 0
        self.inactive_steps = 0
        
        # In non-training modes, use random starting point to avoid overfitting
        if self.mode != 'train' and options and options.get('random_start', False):
            max_start = len(self.data) - self.window_size - 100  # Ensure space for an episode
            if max_start > self.window_size:
                self.current_step = np.random.randint(self.window_size, max_start)
        
        # Get initial observation
        observation = self._get_observation()
        
        # Initial performance update
        # Actualizar métricas de rendimiento
        self._calculate_unrealized_pnl()
        
        # Return initial observation and info
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'net_worth': self.net_worth,
            'position': self.position_size
        }
        
        return observation, info
    
    def render(self, mode='console') -> None:
        """
        Render the environment.
        
        Args:
            mode (str, optional): Rendering mode. Defaults to 'console'.
        """
        if mode != 'console':
            raise NotImplementedError(f"Rendering mode {mode} not implemented, only 'console' is available")
        
        # Get current data point
        date = self.dates[self.current_step] if self.current_step < len(self.dates) else None
        price = self.data.iloc[self.current_step]['close'] if self.current_step < len(self.data) else None
        
        # Display information
        print(f"Step: {self.current_step}")
        print(f"Date: {date}")
        print(f"Price: {price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Net Worth: ${self.net_worth:.2f}")
        print(f"Position: {self.position_size}")
        
        if self.position_size != 0:
            print(f"Entry Price: {self.entry_price:.2f}")
            print(f"Stop Loss: {self.stop_loss:.2f}")
            print(f"Take Profit: {self.take_profit:.2f}")
        
        print(f"Total Trades: {len(self.trade_history)}")
        print(f"Max Drawdown: {max(self.drawdowns) * 100:.2f}%" if self.drawdowns else "Max Drawdown: 0.00%")
        print("---------------------------")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the environment's performance.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        if not self.performance_history:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
            }
        
        # Calculate returns
        total_return = self.returns[-1] if self.returns else 0.0
        daily_returns = np.diff(self.returns) if len(self.returns) > 1 else np.array([0.0])
        
        # Calculate Sharpe ratio
        risk_free_rate = self.config.get('risk_free_rate', 0.02) / 252  # Daily risk-free rate
        sharpe_ratio = 0.0
        if len(daily_returns) > 1 and np.std(daily_returns) != 0:
            sharpe_ratio = (np.mean(daily_returns) - risk_free_rate) / np.std(daily_returns) * np.sqrt(252)
        
        # Calculate max drawdown
        max_drawdown = max(self.drawdowns) if self.drawdowns else 0.0
        
        # Calculate win rate and profit factor
        win_count = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        total_trades = len(self.trade_history)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit factor
        gross_profit = sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'final_net_worth': self.net_worth,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
        }

    def get_trade_history_df(self) -> pd.DataFrame:
        """
        Get trade history as a DataFrame for analysis.
        
        Returns:
            pd.DataFrame: Trade history with metrics
        """
        return self.metrics.get_debug_dataframe()

    def plot_metrics(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate and optionally save metrics plots.
        
        Args:
            save_path (str, optional): Directory to save plots
            
        Returns:
            Dict[str, Any]: Dictionary of plot figures
        """
        return self.metrics.plot_metrics(save_path)
        
    def update_data(self, new_data: pd.DataFrame) -> np.ndarray:
        """
        Actualiza los datos en modo inferencia y devuelve una nueva observación.
        
        Args:
            new_data (pd.DataFrame): Nuevos datos a añadir
            
        Returns:
            np.ndarray: Nueva observación
        """
        try:
            if self.mode != "inference":
                logger.warning("update_data() solo debe usarse en modo inferencia")
            
            if new_data.empty:
                logger.warning("Los nuevos datos están vacíos, no se actualiza nada")
                return None
            
            # Actualizar parámetros de normalización si es necesario
            prev_min = self.price_min
            prev_max = self.price_max
            
            if 'close' in new_data.columns:
                data_min = new_data['close'].min()
                data_max = new_data['close'].max()
                
                # Actualizar min/max si es necesario
                if self.price_min is None or data_min < self.price_min:
                    self.price_min = data_min
                
                if self.price_max is None or data_max > self.price_max:
                    self.price_max = data_max
                
                # Actualizar rango de precios
                self.price_range = self.price_max - self.price_min
                if self.price_range == 0:
                    self.price_range = 1.0
                
                # Informar si hubo cambios en la normalización
                if prev_min != self.price_min or prev_max != self.price_max:
                    logger.info(f"Precios actualizados: min={self.price_min}, max={self.price_max}, range={self.price_range}")
            
            # Actualizar datos de ventana con los nuevos datos
            self.data = new_data.copy()
            
            # Verificar columnas en los datos de entrada
            logger.debug(f"Datos recibidos en update_data: shape={new_data.shape}, columnas={new_data.columns.tolist()}")
            
            # VERIFICACIÓN CRÍTICA: Asegurar que el número de columnas sea exactamente el esperado
            if len(new_data.columns) != self.config["features"]:
                logger.warning(f"Número incorrecto de columnas: {len(new_data.columns)}, se esperan {self.config['features']}")
                
                # Corregir el DataFrame si es necesario
                if len(new_data.columns) > self.config["features"]:
                    logger.warning(f"Truncando DataFrame de {len(new_data.columns)} a {self.config['features']} columnas")
                    # Priorizar columnas OHLCV y técnicas principales
                    priority_cols = ['open', 'high', 'low', 'close', 'volume', 
                                   'sma_5', 'sma_10', 'sma_20', 'rsi_14', 
                                   'macd', 'macd_signal']
                    
                    # Filtrar para columnas que existen
                    priority_cols = [col for col in priority_cols if col in new_data.columns]
                    
                    # Completar con otras columnas hasta llegar a 25
                    other_cols = [col for col in new_data.columns if col not in priority_cols]
                    selected_cols = priority_cols + other_cols
                    
                    # Truncar a 25 columnas
                    self.data = new_data[selected_cols[:self.config["features"]]].copy()
                else:
                    logger.warning(f"Añadiendo columnas al DataFrame para llegar a {self.config['features']} columnas")
                    self.data = new_data.copy()
                    for i in range(len(new_data.columns), self.config["features"]):
                        self.data[f'feature_{i}'] = 0.0
            
            # Verificar después de la corrección
            logger.debug(f"DataFrame corregido: shape={self.data.shape}, columnas={self.data.columns.tolist()}")
            
            if len(self.data) < self.config["window_size"]:
                logger.warning(f"Datos insuficientes: {len(self.data)} filas, se requieren > {self.config['window_size']}")
            
            logger.info("Datos actualizados correctamente, generando nueva observación")
            observation = self._get_observation()
            
            # Verificar shape antes de devolver
            logger.info(f"Observación generada con shape: {observation.shape} (esperada: ({self.config['window_size']}, {self.config['features']}))")
            if observation.shape != (self.config["window_size"], self.config["features"]):
                logger.error(f"ERROR CRÍTICO: Shape de observación generada {observation.shape} no coincide con la esperada ({self.config['window_size']}, {self.config['features']})")
                
                # Intentar corregir el tamaño de la observación
                if observation.shape[1] > self.config["features"]:
                    # Si hay más columnas de las esperadas, truncar
                    logger.warning(f"Truncando observación de {observation.shape[1]} a {self.config['features']} columnas")
                    observation = observation[:, :self.config["features"]]
                elif observation.shape[1] < self.config["features"]:
                    # Si hay menos columnas de las esperadas, rellenar con ceros
                    logger.warning(f"Rellenando observación de {observation.shape[1]} a {self.config['features']} columnas")
                    padded_obs = np.zeros((self.config["window_size"], self.config["features"]), dtype=np.float32)
                    padded_obs[:, :observation.shape[1]] = observation
                    observation = padded_obs
                
                # Corregir filas si es necesario
                if observation.shape[0] != self.config["window_size"]:
                    logger.warning(f"Corrigiendo filas de {observation.shape[0]} a {self.config['window_size']}")
                    if observation.shape[0] > self.config["window_size"]:
                        # Si hay más filas, tomar las últimas window_size filas
                        observation = observation[-self.config["window_size"]:]
                    else:
                        # Si hay menos filas, rellenar con ceros
                        padded_obs = np.zeros((self.config["window_size"], observation.shape[1]), dtype=np.float32)
                        padded_obs[-observation.shape[0]:] = observation
                        observation = padded_obs
            
            # Verificación final
            logger.info(f"Observación final con shape: {observation.shape}")
            assert observation.shape == (self.config["window_size"], self.config["features"]), f"Forma de observación incorrecta: {observation.shape}"
            return observation
        except Exception as e:
            logger.error(f"Error en update_data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # En caso de error, devolver una matriz de ceros con la forma correcta
            return np.zeros((self.config["window_size"], self.config["features"]), dtype=np.float32)

    def _open_position(self, action: float, price: float) -> None:
        """
        Open a new position.
        
        Args:
            action (float): Action value representing position size and direction
            price (float): Price at which to open position
        """
        # Si ya hay una posición abierta, cerrarla primero
        if self.position_size != 0:
            self._close_position(price, "new_position")
        
        # Set position and entry parameters
        self.position_size = action
        self.entry_price = price
        self.entry_time = self.current_step
        
        # Determinar tamaño y dirección
        size = abs(self.position_size)
        direction = 1 if self.position_size > 0 else -1
        
        # Calcular capital a arriesgar (como un porcentaje del balance)
        risk_pct = self.config.get('risk_per_trade_limit', 0.02)  # Límite de riesgo por operación (2%)
        capital_at_risk = self.balance * risk_pct
        
        # Configurar stop loss basado en porcentaje del balance
        stop_loss_pct = self.config.get('stop_loss_pct', 0.02)
        take_profit_ratio = self.config.get('take_profit_ratio', 1.5)
        
        # Calcular stop loss en puntos de precio
        stop_dist_pct = stop_loss_pct / size
        stop_dist = price * stop_dist_pct
        
        # Set stop loss and take profit levels
        if direction > 0:  # Long position
            self.stop_loss = price - stop_dist
            self.take_profit = price + (stop_dist * take_profit_ratio)
        else:  # Short position
            self.stop_loss = price + stop_dist
            self.take_profit = price - (stop_dist * take_profit_ratio)
        
        # Almacenar el stop loss original para detectar ajustes posteriores
        self.original_stop_loss = self.stop_loss
        
        # Inicializar opciones de trailing stop
        self.trailing_stop_active = False
        self.trailing_stop_distance = stop_dist
        
        # Calcular y almacenar ratio riesgo/beneficio
        risk = abs(price - self.stop_loss)
        reward = abs(price - self.take_profit)
        self.risk_reward_ratio = reward / risk if risk > 0 else 0.0
        
        # Apply commission
        commission = price * abs(self.position_size) * self.commission_rate
        self.balance -= commission
        
        # Record trade
        trade = {
            'entry_step': self.current_step,
            'entry_price': price,
            'size': self.position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'exit_step': None,
            'exit_price': None,
            'pnl': None,
            'reason': None
        }
        self.trades.append(trade)
        
        # Set flags
        self.trade_active = True
        self._just_opened_position = True
        self.position_steps = 0

    def update_stop_loss(self, action_type: str, new_level: Optional[float] = None) -> bool:
        """
        Actualiza el stop loss según la acción solicitada.
        
        Args:
            action_type (str): Tipo de actualización ('breakeven', 'trailing', 'fixed')
            new_level (float, optional): Nuevo nivel para stop loss fijo
            
        Returns:
            bool: True si se aplicó el cambio, False en caso contrario
        """
        if not self.trade_active:
            return False
        
        price = self.data.iloc[self.current_step]['close']
        direction = 1 if self.position_size > 0 else -1
        
        # Verificar que la posición está en ganancia para ciertas acciones
        is_profit = (direction > 0 and price > self.entry_price) or (direction < 0 and price < self.entry_price)
        
        if action_type == 'breakeven' and is_profit:
            # Mover stop loss a punto de entrada (break-even)
            self.stop_loss = self.entry_price
            self.logger.debug(f"Stop loss movido a break-even: {self.stop_loss:.2f}")
            return True
            
        elif action_type == 'trailing' and is_profit:
            # Activar trailing stop si no está activo
            if not self.trailing_stop_active:
                self.trailing_stop_active = True
                self.logger.debug("Trailing stop activado")
            
            # Actualizar nivel de trailing stop
            if direction > 0:  # Long position
                new_stop = price - self.trailing_stop_distance
                if new_stop > self.stop_loss:
                    self.stop_loss = new_stop
                    self.logger.debug(f"Trailing stop actualizado: {self.stop_loss:.2f}")
                    return True
            else:  # Short position
                new_stop = price + self.trailing_stop_distance
                if new_stop < self.stop_loss:
                    self.stop_loss = new_stop
                    self.logger.debug(f"Trailing stop actualizado: {self.stop_loss:.2f}")
                    return True
            
        elif action_type == 'fixed' and new_level is not None:
            # Validar que el nuevo stop loss es aceptable
            if direction > 0:  # Long position
                if new_level < self.entry_price and new_level < price:
                    self.stop_loss = new_level
                    self.logger.debug(f"Stop loss fijo actualizado: {self.stop_loss:.2f}")
                    return True
            else:  # Short position
                if new_level > self.entry_price and new_level > price:
                    self.stop_loss = new_level
                    self.logger.debug(f"Stop loss fijo actualizado: {self.stop_loss:.2f}")
                    return True
                    
        return False

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
        
    def _update_observation(self) -> None:
        """
        Actualiza la observación actual después de cada paso.
        """
        self.observation = self._get_observation()
        
    def _update_market_state(self) -> bool:
        """
        Actualiza el estado del mercado y comprueba si hemos llegado al final de los datos.
        
        Returns:
            bool: True si hay más datos disponibles, False si hemos llegado al final
        """
        # Comprobar si hemos llegado al final de los datos
        if self.current_step >= len(self.data) - 1:
            return False
            
        # Actualizar precios actuales
        if 'open' in self.data.columns:
            self.open_price = self.data.iloc[self.current_step]['open']
        if 'high' in self.data.columns:
            self.high_price = self.data.iloc[self.current_step]['high']
        if 'low' in self.data.columns:
            self.low_price = self.data.iloc[self.current_step]['low']
        if 'close' in self.data.columns:
            self.current_price = self.data.iloc[self.current_step]['close']
            
        # Actualizar el valor no realizado de la posición actual
        self.unrealized_pnl = self._calculate_unrealized_pnl()
        self.current_pnl = self.unrealized_pnl
        
        # Actualizar net worth
        self.net_worth = self.balance + self.unrealized_pnl
        
        # Actualizar drawdown
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        
        drawdown = 0.0
        if self.max_net_worth > 0:
            drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        
        self.drawdowns.append(drawdown)
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            
        # Retorno de que hay más datos disponibles
        return True
        
    def _close_position(self, price=None, reason="manual") -> float:
        """
        Cierra la posición actual y calcula el PnL resultante.
        
        Args:
            price (float, optional): Precio al que cerrar la posición. Si es None, se usa el precio actual.
            reason (str, optional): Razón del cierre de la posición.
            
        Returns:
            float: PnL de la operación cerrada
        """
        if self.position_size == 0:
            return 0.0
            
        # Determinar precio de cierre
        if price is None:
            price = self.data.iloc[self.current_step]['close']
            
        # Calcular PnL
        pnl = (price - self.entry_price) * self.position_size * self.contract_size
        
        # Aplicar comisión
        commission = price * abs(self.position_size) * self.commission_rate
        pnl -= commission
        
        # Actualizar saldo
        self.balance += pnl
        
        # Actualizar historial de trades
        if self.trades:
            last_trade = self.trades[-1]
            last_trade['exit_step'] = self.current_step
            last_trade['exit_price'] = price
            last_trade['pnl'] = pnl
            last_trade['reason'] = reason
            
            # Añadir al historial completo
            self.trade_history.append(last_trade.copy())
            
        # Resetear variables de posición
        self.position_size = 0
        self.current_position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.unrealized_pnl = 0.0
        self.trade_active = False
        
        # Actualizar flags
        self._just_closed_position = True
        self._last_trade_pnl = pnl
        
        return pnl
