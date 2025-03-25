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
        
        # NUEVO: Recompensas retrasadas para mejor aprendizaje a largo plazo
        self.pending_rewards = []
        self.reward_delay_steps = self.config.get('reward_delay_steps', 5)  # Retraso predeterminado de 5 pasos
        
        # NUEVO: Memoria de posiciones rentables para tracking
        self.profitable_steps = 0  # Contador de pasos rentables consecutivos
        self.prev_unrealized_pnl = 0.0  # PnL anterior para comparación
        
        # NUEVO: Tiempo de enfriamiento entre operaciones
        self.cooldown_counter = 0  # Contador de tiempo de enfriamiento
        self.min_hold_steps = self.config.get('min_hold_steps', 5)  # Duración mínima recomendada de una operación
        self.position_cooldown = self.config.get('position_cooldown', 10)  # Tiempo de enfriamiento entre operaciones
        
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

    def _get_observation(self):
        """
        Obtiene la observación actual del entorno.
        Returns:
            numpy.ndarray: La observación actual con forma (window_size, 25).
        """
        try:
            # Seleccionar la ventana de observación
            current_step = self.current_step
            start_idx = max(0, current_step - self.window_size + 1)
            end_idx = current_step + 1
            
            # Obtener slice de datos
            if start_idx >= end_idx:
                logger.warning(f"Índices inválidos para slice: start={start_idx}, end={end_idx}")
                # Devolver observación vacía con el tamaño correcto (window_size, 25)
                return np.zeros((self.window_size, 25))
            
            data_slice = self.data.iloc[start_idx:end_idx].copy()
            
            # Seleccionar solo columnas numéricas para evitar problemas con timestamps
            numeric_cols = data_slice.select_dtypes(include=[np.number]).columns
            data_slice = data_slice[numeric_cols]
            
            # Asegurarse de que la observación tiene la forma correcta
            if len(data_slice) < self.window_size:
                # Rellenar con ceros si no hay suficientes datos
                padding = np.zeros((self.window_size - len(data_slice), len(numeric_cols)))
                obs = np.vstack([padding, data_slice.values])
            else:
                obs = data_slice.values
            
            # Normalizar los datos si está configurado
            normalize_observation = self.config.get('normalize_observation', False)
            if normalize_observation:
                # Normalizar cada columna independientemente
                if not hasattr(self, 'scaler') or self.scaler is None:
                    from sklearn.preprocessing import StandardScaler
                    self.scaler = StandardScaler()
                    # Fittear el scaler solo con los datos de entrenamiento
                    train_data = self.data.iloc[:int(0.8 * len(self.data))][numeric_cols]
                    self.scaler.fit(train_data)
                
                # Aplanar, normalizar y restaurar la forma
                orig_shape = obs.shape
                obs_flat = obs.reshape(-1, obs.shape[-1])
                obs_norm = self.scaler.transform(obs_flat)
                obs = obs_norm.reshape(orig_shape)
            
            # SOLUCIÓN: Garantizar que la observación tenga exactamente 25 características
            # Si hay menos de 25 columnas, rellenar con ceros adicionales
            if obs.shape[1] < 25:
                padding_cols = np.zeros((obs.shape[0], 25 - obs.shape[1]))
                obs = np.hstack([obs, padding_cols])
            # Si hay más de 25 columnas, recortar
            elif obs.shape[1] > 25:
                obs = obs[:, :25]
                
            # Verificar la forma final
            if obs.shape != (self.window_size, 25):
                logger.warning(f"Forma de observación incorrecta: {obs.shape}, ajustando a {(self.window_size, 25)}")
                # Forzar la forma correcta como último recurso
                tmp_obs = np.zeros((self.window_size, 25))
                rows_to_copy = min(obs.shape[0], self.window_size)
                cols_to_copy = min(obs.shape[1], 25)
                tmp_obs[:rows_to_copy, :cols_to_copy] = obs[:rows_to_copy, :cols_to_copy]
                obs = tmp_obs
            
            return obs
            
        except Exception as e:
            logger.error(f"Error en _get_observation: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Devolver una observación vacía con el tamaño correcto (window_size, 25)
            return np.zeros((self.window_size, 25))
    
    def _calculate_reward(self):
        """Calculate the reward for the current step."""
        # MEJORADO: Sistema de recompensas más agresivo para incentivar operaciones
        
        # Inicializar componentes de recompensa
        reward_components = {
            'base': 0.0,
            'pnl': 0.0,
            'trade_bonus': 0.0,
            'inactivity': 0.0,
            'exploration': 0.0,
            'drawdown': 0.0,
            'holding': 0.0,
            'profitable_hold': 0.0,  # Recompensa específica por mantener posiciones rentables
            'overtrade': 0.0,        # Penalización por sobretrading
            'short_trade_penalty': 0.0,  # NUEVO: Penalización por operaciones de corta duración
            'duration_bonus': 0.0,   # NUEVO: Bonus por mantener posiciones por más tiempo
        }
        
        # 1. Recompensa base (siempre presente)
        base_reward = self.reward_config.get('base_reward', -0.1)  # Penalización base
        reward_components['base'] = base_reward
        
        # 2. Recompensa por PnL (normalizado)
        pnl_change = self.current_pnl - self.prev_pnl
        normalized_pnl = pnl_change / (self.initial_balance * 0.01) if self.initial_balance > 0 else 0
        pnl_weight = self.reward_config.get('pnl_weight', 1.5)  # Aumentado de 1.0 a 1.5
        reward_components['pnl'] = normalized_pnl * pnl_weight
        
        # 3. Bonus por abrir posición
        if self._just_opened_position:
            open_position_bonus = 2.0  # Bonus significativo por abrir posición
            reward_components['exploration'] = open_position_bonus
            self._just_opened_position = False  # Resetear la flag
            
            # Penalización por sobretrading (abrir posición poco después de cerrar otra)
            steps_since_last_trade = 0
            if len(self.trades) >= 2:
                current_entry_time = self.trades[-1]['entry_time']
                prev_exit_time = self.trades[-2].get('exit_time', 0)
                steps_since_last_trade = current_entry_time - prev_exit_time
                
            if steps_since_last_trade < 5:  # Si abrió posición menos de 5 pasos después de cerrar la anterior
                overtrade_penalty = self.config.get('overtrade_penalty', -3.0)
                reward_components['overtrade'] = overtrade_penalty * (1.0 - steps_since_last_trade / 5)
            
        # 4. Bonus por completar operaciones
        if self.trade_completed:
            trade_bonus = self.reward_config.get('trade_completion_bonus', 12.0)  # Aumentado de 10.0 a 12.0
            reward_components['trade_bonus'] = trade_bonus
            
            # Bonus adicional si es una operación ganadora
            if self._last_trade_pnl > 0:
                reward_components['trade_bonus'] += trade_bonus * 0.8  # 80% adicional para trades ganadores
            
            # NUEVO: Penalización severa por operaciones de corta duración
            if len(self.trades) > 0:
                last_trade = self.trades[-1]
                if 'entry_time' in last_trade and 'exit_time' in last_trade:
                    position_duration = last_trade['exit_time'] - last_trade['entry_time']
                    
                    # Penalizar fuertemente operaciones que duran menos del mínimo recomendado
                    if position_duration < self.min_hold_steps:
                        short_trade_penalty = -8.0 * (self.min_hold_steps - position_duration)
                        reward_components['short_trade_penalty'] = short_trade_penalty
                        logger.debug(f"Penalización por operación corta: {short_trade_penalty} (duración: {position_duration})")
                    
                    # NUEVO: Bonus por operaciones de larga duración
                    elif position_duration > self.min_hold_steps:
                        # Bonus que crece con la duración, pero se satura
                        duration_factor = min(3.0, 1.0 + (position_duration - self.min_hold_steps) / 10)
                        duration_bonus = 2.0 * duration_factor
                        reward_components['duration_bonus'] = duration_bonus
                        logger.debug(f"Bonus por operación larga: {duration_bonus} (duración: {position_duration})")
            
            self.trade_completed = False  # Resetear la flag
        
        # 5. Penalización por inactividad (más severa)
        if self.inactive_steps > 0:
            inactivity_penalty = -self.inactive_steps * self.reward_config.get('inactivity_weight', 3.0) / 10  # Aumentado de 2.0 a 3.0
            reward_components['inactivity'] = inactivity_penalty
        
        # 6. Penalización por drawdown
        if self.max_drawdown > 0:
            drawdown_penalty = -self.max_drawdown * self.reward_config.get('drawdown_weight', 0.05)
            reward_components['drawdown'] = drawdown_penalty
        
        # 7. Penalización por mantener posiciones demasiado tiempo
        if self.position_duration > 15 and self.current_position != 0:  # Reducido de 20 a 15
            # NUEVO: Solo penalizar si la posición no es rentable
            unrealized_pnl = self._calculate_unrealized_pnl()
            if unrealized_pnl <= 0:
                holding_penalty = -0.02 * (self.position_duration - 15)  # Penalización más severa
                reward_components['holding'] = holding_penalty
        
        # 8. Recompensa por mantener posiciones rentables
        if self.current_position != 0:
            unrealized_pnl = self._calculate_unrealized_pnl()
            
            if unrealized_pnl > 0:
                # Actualizar contador de pasos rentables
                if unrealized_pnl >= self.prev_unrealized_pnl:
                    self.profitable_steps += 1
                else:
                    self.profitable_steps = max(0, self.profitable_steps - 1)
                
                # MEJORADO: Factor de tiempo más agresivo (mayor recompensa por tiempo)
                # Mayor recompensa cuanto más tiempo mantiene posición rentable
                time_factor = min(2.0, 1.0 + self.profitable_steps / 15)  # Aumentado de 1.0 a 2.0, y reducido de 30 a 15
                
                # Factor de beneficio: mayor recompensa cuanto más rentable es la posición
                profit_factor = min(2.0, 1.0 + unrealized_pnl / (self.initial_balance * 0.0005))  # Aumentado de 1.0 a 2.0
                
                # Obtener factor de recompensa por mantener posiciones rentables
                hold_reward_factor = self.config.get('hold_reward', 0.0)
                
                # MEJORADO: Aumentar significativamente el multiplicador para recompensa por mantener
                # Combinar factores para recompensa
                hold_reward = time_factor * profit_factor * hold_reward_factor * 10.0  # Aumentado de 5.0 a 10.0
                reward_components['profitable_hold'] = hold_reward
                
                logger.debug(f"Recompensa por mantener posición rentable: {hold_reward:.2f} (time={time_factor:.2f}, profit={profit_factor:.2f})")
            
            # Actualizar PnL anterior para la próxima comparación
            self.prev_unrealized_pnl = unrealized_pnl
            
        # Calcular recompensa total
        reward = sum(reward_components.values())
        
        # Factor de escala final (aumentado)
        scale_factor = self.reward_config.get('scale_factor', 12.0)  # Aumentado de 10.0 a 12.0
        reward *= scale_factor
        
        # Normalizar la recompensa para evitar valores extremos
        if abs(reward) > 1000:
            reward = np.sign(reward) * (1000 + np.log(abs(reward) - 999))
        
        # Verificar rangos razonables
        reward = np.clip(reward, -10000, 10000)
        
        # Guardar componentes para logging
        self.reward_components = reward_components
        
        return reward

    def step(self, action):
        """
        Ejecuta un paso en el entorno con la acción dada.
        
        Args:
            action: Acción a ejecutar (0: mantener, 1: comprar, 2: vender)
            
        Returns:
            Tuple: (observación, recompensa, terminado, truncado, info)
        """
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Registrar el estado antes de ejecutar la acción
        logger.info(f"Paso {self.current_step}: Recibida acción {action}, posición actual: {self.current_position}")
        
        # NUEVO: Aplicar tiempo de enfriamiento entre operaciones
        in_cooldown = False
        if hasattr(self, 'cooldown_counter') and self.cooldown_counter > 0:
            in_cooldown = True
            self.cooldown_counter -= 1
            logger.info(f"En período de enfriamiento: {self.cooldown_counter} pasos restantes")
        
        # Procesar la acción
        if isinstance(action, np.ndarray) and len(action) > 0:
            # Para espacio de acción continuo
            action_value = action[0]
            logger.info(f"Acción continua recibida: {action_value}")
            
            # MODIFICADO: Umbral mucho más alto para cambiar de posición (0.4 -> 0.7)
            # Esto hace que el agente necesite estar mucho más "seguro" para cambiar de posición
            if action_value > 0.7 and self.current_position <= 0:
                processed_action = 1  # Comprar
            elif action_value < -0.7 and self.current_position >= 0:
                processed_action = 2  # Vender
            else:
                processed_action = 0  # Mantener
                
            # NUEVO: Si estamos en período de enfriamiento, no permitir abrir nuevas posiciones
            if in_cooldown and processed_action != 0 and self.current_position == 0:
                logger.info(f"Acción {processed_action} ignorada debido a período de enfriamiento")
                processed_action = 0  # Forzar a mantener durante enfriamiento
        else:
            # Para espacio de acción discreto
            processed_action = action
            
            # NUEVO: Si estamos en período de enfriamiento, no permitir abrir nuevas posiciones
            if in_cooldown and processed_action != 0 and self.current_position == 0:
                logger.info(f"Acción {processed_action} ignorada debido a período de enfriamiento")
                processed_action = 0  # Forzar a mantener durante enfriamiento
        
        # CORRECCIÓN: Aplicar probabilidad de forzar acción (para exploración)
        force_action_prob = self.config.get('force_action_prob', 0.0)
        if np.random.random() < force_action_prob and not in_cooldown:  # No forzar acción durante enfriamiento
            original_action = processed_action
            # Forzar una acción diferente a la actual
            if self.current_position == 0:  # Si no hay posición, forzar compra o venta
                processed_action = np.random.choice([1, 2])
            elif self.current_position > 0:  # Si está comprado, forzar venta
                processed_action = 2
            elif self.current_position < 0:  # Si está vendido, forzar compra
                processed_action = 1
            logger.info(f"Forzando acción de {original_action} a {processed_action}")
        
        # CORRECCIÓN: Ejecutar la acción procesada
        if processed_action == 1:  # Comprar
            # Si estamos vendidos, cerrar posición primero
            if self.current_position < 0:
                logger.info("Cerrando posición vendida antes de comprar")
                self._close_position()
            # Abrir posición comprada si no estamos ya comprados
            if self.current_position == 0:
                logger.info("Abriendo posición comprada")
                self._open_position(direction=1)
        elif processed_action == 2:  # Vender
            # Si estamos comprados, cerrar posición primero
            if self.current_position > 0:
                logger.info("Cerrando posición comprada antes de vender")
                self._close_position()
            # Abrir posición vendida si no estamos ya vendidos
            if self.current_position == 0:
                logger.info("Abriendo posición vendida")
                self._open_position(direction=-1)
        elif processed_action == 0 and self.current_position != 0:
            # Cerrar posición actual si la acción es 0 (mantener) pero tenemos posición
            logger.info("Cerrando posición existente")
            self._close_position()
            
            # NUEVO: Activar período de enfriamiento cuando se cierra una posición
            # para evitar que se abra inmediatamente una nueva
            self.cooldown_counter = self.position_cooldown
            logger.info(f"Iniciando período de enfriamiento: {self.cooldown_counter} pasos")
        # Para acción 0 (mantener) y sin posición no hacemos nada
        
        # Calcular recompensa para este paso
        current_reward = self._calculate_reward()
        
        # NUEVO: Sistema de recompensas retrasadas para mejor aprendizaje a largo plazo
        # Añadir recompensa actual a la cola de pendientes
        self.pending_rewards.append(current_reward)
        
        # Obtener recompensa retrasada si hay suficientes
        if len(self.pending_rewards) > self.reward_delay_steps:
            reward = self.pending_rewards.pop(0)
        else:
            # Durante los primeros pasos, dar una recompensa más pequeña
            reward = current_reward * 0.1
        
        # Si es el último paso, dar todas las recompensas pendientes
        if done and self.pending_rewards:
            remaining_rewards = sum(self.pending_rewards)
            reward += remaining_rewards
            self.pending_rewards = []
        
        # Obtener la siguiente observación
        observation = self._get_observation()
        
        # Información adicional
        info = {
            'position': self.current_position,
            'balance': self.balance,
            'action_taken': processed_action,
            'current_reward': current_reward,  # Añadir recompensa actual para debugging
            'delayed_reward': reward,          # Añadir recompensa retrasada para debugging
            'in_cooldown': in_cooldown,        # NUEVO: Indicar si estamos en período de enfriamiento
        }
        
        # Verificar si log_reward_components está en la configuración
        log_reward_components = self.config.get('log_reward_components', False)
        if log_reward_components and hasattr(self, 'reward_components'):
            info.update(self.reward_components)
        
        logger.info(f"Después de la acción: posición={self.current_position}, balance={self.balance:.2f}, recompensa={reward:.2f}")
        
        return observation, reward, done, False, info
    
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
        Obtener un resumen del rendimiento del entorno.
        
        Returns:
            Dict[str, Any]: Métricas de rendimiento
        """
        logger.debug(f"Calculando resumen de rendimiento. Trades totales: {len(self.trades)}")
        
        # Si no hay operaciones, retornar estadísticas básicas
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_pnl': 0.0,
                'balance': self.balance
            }
        
        # Filtrar trades completados (aquellos que tienen exit_time)
        completed_trades = [trade for trade in self.trades if 'exit_time' in trade]
        
        # Calcular estadísticas básicas
        total_trades = len(completed_trades)
        
        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_pnl': 0.0,
                'balance': self.balance
            }
        
        # Calcular ganancias y pérdidas
        winning_trades = [trade for trade in completed_trades if trade.get('pnl', 0) > 0]
        losing_trades = [trade for trade in completed_trades if trade.get('pnl', 0) <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        total_profit = sum(trade.get('pnl', 0) for trade in winning_trades)
        total_loss = sum(abs(trade.get('pnl', 0)) for trade in losing_trades)
        
        # Evitar división por cero
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calcular totales
        total_pnl = sum(trade.get('pnl', 0) for trade in completed_trades)
        avg_profit = total_profit / len(winning_trades) if winning_trades else 0.0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0.0
        
        logger.info(f"Resumen de rendimiento: trades={total_trades}, win_rate={win_rate:.2f}, profit_factor={profit_factor:.2f}")
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'balance': self.balance
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
            logger.error(f"Traceback: {traceback.format_exc()}")
            # En caso de error, devolver una matriz de ceros con la forma correcta
            return np.zeros((self.config["window_size"], self.config["features"]), dtype=np.float32)

    def _open_position(self, direction):
        """
        Abre una posición en la dirección dada.
        
        Args:
            direction: Dirección de la posición (1: compra, -1: venta)
        """
        # CORRECCIÓN: Mejor validación y registro de intentos
        current_price = self.data.iloc[self.current_step]['close']
        
        logger.info(f"Intentando abrir posición {direction} a precio {current_price}")
        
        # Validar que no hay posición abierta en la dirección contraria
        if (direction > 0 and self.current_position < 0) or (direction < 0 and self.current_position > 0):
            logger.warning(f"No se puede abrir posición {direction} porque ya hay posición contraria {self.current_position}")
            return False
        
        # Validar que no hay posición abierta en la misma dirección
        if (direction > 0 and self.current_position > 0) or (direction < 0 and self.current_position < 0):
            logger.warning(f"Ya hay una posición abierta en dirección {direction}")
            return False
        
        # CORRECCIÓN: Verificar que hay suficiente margen para abrir posición
        position_size = self.max_position
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
        
        # Calcular niveles de stop loss y take profit
        stop_loss_pct = self.config.get('stop_loss_pct', 0.01)  # Valor predeterminado de 1%
        take_profit_pct = self.config.get('take_profit_pct', 0.02)  # Valor predeterminado de 2%
        
        self.stop_loss = self.entry_price * (1 - direction * stop_loss_pct)
        self.take_profit = self.entry_price * (1 + direction * take_profit_pct)
        
        # CORRECCIÓN: Registrar la operación incluyendo el campo 'direction'
        trade_info = {
            'entry_time': self.current_step,
            'entry_price': self.entry_price,
            'position': self.current_position,
            'direction': 'long' if direction > 0 else 'short',  # Añadir campo direction
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }
        self.trades.append(trade_info)
        
        # CORRECCIÓN: Siempre deducir la comisión al abrir
        commission_rate = self.config.get('commission_rate', 0.001)  # Valor predeterminado de 0.1%
        commission = abs(self.current_position) * current_price * commission_rate
        self.balance -= commission
        
        # Marcar que acabamos de abrir posición para el cálculo de la recompensa
        self._just_opened_position = True
        
        logger.info(f"Posición abierta: {self.current_position} ({trade_info['direction']}) a {self.entry_price:.2f}, SL={self.stop_loss:.2f}, TP={self.take_profit:.2f}, comisión={commission:.2f}")
        return True
    
    def _close_position(self):
        """
        Cierra la posición actual y actualiza el balance.
        """
        if self.current_position == 0:
            logger.warning("Intentando cerrar posición cuando no hay ninguna abierta")
            return False
        
        current_price = self.data.iloc[self.current_step]['close']
        logger.info(f"Cerrando posición {self.current_position} entrada a {self.entry_price:.2f}, precio actual {current_price:.2f}")
        
        # Calcular PnL
        if self.current_position > 0:  # Posición larga
            pnl = (current_price - self.entry_price) * abs(self.current_position)
        else:  # Posición corta
            pnl = (self.entry_price - current_price) * abs(self.current_position)
        
        # Deducir comisión
        commission_rate = self.config.get('commission_rate', 0.001)  # Valor predeterminado de 0.1%
        commission = abs(self.current_position) * current_price * commission_rate
        net_pnl = pnl - commission
        
        # NUEVO: Calcular duración de la posición
        position_duration = 0
        if self.trades and 'entry_time' in self.trades[-1]:
            position_duration = self.current_step - self.trades[-1]['entry_time']
        
        # NUEVO: Factor de duración para escalar el PnL
        duration_factor = 1.0
        if position_duration >= self.min_hold_steps:
            # Escalar el PnL positivo según la duración (hasta 2x para posiciones largas)
            if pnl > 0:
                max_duration_factor = 2.0
                duration_factor = min(max_duration_factor, 
                                     1.0 + (position_duration - self.min_hold_steps) / 20)
                
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
                'exit_price': current_price,
                'pnl': pnl,
                'net_pnl': net_pnl,
                'commission': commission,
                'duration': position_duration,  # NUEVO: Registrar duración de la posición
                'duration_factor': duration_factor  # NUEVO: Registrar factor de duración aplicado
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
