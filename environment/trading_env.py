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
        self.window_size = window_size
        self.mode = mode
        self.log_rewards = self.config.get('log_reward_components', False)
        
        # Set up price scaler for normalization
        close_prices = self.data['close'].values
        self.price_min = close_prices.min()
        self.price_max = close_prices.max()
        self.price_range = self.price_max - self.price_min
        
        # Store data properties
        self.num_features = len(self.data.columns)
        self.feature_names = list(self.data.columns)
        self.dates = self.data.index.tolist() if hasattr(self.data.index, 'tolist') else list(range(len(self.data)))
        
        # Initialize position and state variables
        self.position = 0
        self.current_step = 0
        self.done = False
        self.trades = []
        self.trade_history = []
        self.position_history = []
        self.performance_history = []
        self.trade_active = False
        self.unrealized_pnl = 0.0
        
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
        
        # Performance tracking
        self.initial_net_worth = self.initial_balance
        self.net_worth = self.initial_net_worth
        self.max_net_worth = self.initial_net_worth
        self.returns = []
        self.drawdowns = []
        self.equity_curve = [self.initial_balance]
        
        # Check that data has enough records
        if len(self.data) <= self.window_size:
            raise ValueError(f"Data length ({len(self.data)}) must be greater than window size ({self.window_size})")
        
        # Define action space
        # [buy/sell/hold, position_size, stop_loss_pct, take_profit_pct]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0.5, 1.0]), 
            high=np.array([3, 1, 5.0, 10.0]),
            dtype=np.float32
        )
        
        # Define observation space
        # Will include price data, indicators, and account/position info
        self.state_dim = (self.window_size, self.num_features + 4)  # +4 for position info
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=self.state_dim,
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

        # Variables para tracking de posiciones y trades
        self._just_opened_position = False
        self._just_closed_position = False
        self._last_trade_pnl = 0.0
        self._last_trade_action_initiated = False

        # Trading session state
        self.steps_with_position = 0
        self.steps_since_last_trade = 0
        
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

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation (state).
        
        Returns:
            np.ndarray: Current state observation
        """
        # Get the last window_size rows of data
        start = max(0, self.current_step - self.window_size + 1)
        end = self.current_step + 1
        
        # Extract window data
        window_data = self.data.iloc[start:end].values
        
        # Pad with zeros if necessary (at the beginning of the episode)
        if len(window_data) < self.window_size:
            padding = np.zeros((self.window_size - len(window_data), self.num_features))
            window_data = np.vstack((padding, window_data))
        
        # Add position information to each time step in the window
        # [position, balance_pct, unrealized_pnl_pct, drawdown_pct]
        position_info = np.zeros((self.window_size, 4))
        
        # Fill the last row with current position info
        position_info[-1, 0] = self.position / self.max_position  # Normalized position
        position_info[-1, 1] = self.balance / self.initial_balance - 1.0  # Balance percent change
        
        # Calculate unrealized PnL if there's an open position
        price = self.data.iloc[self.current_step]['close']
        if self.position != 0:
            unrealized_pnl = (price - self.entry_price) * self.position * self.contract_size
            position_info[-1, 2] = unrealized_pnl / self.initial_balance
        
        # Calculate drawdown
        if self.max_net_worth > 0:
            drawdown_pct = (self.max_net_worth - self.net_worth) / self.max_net_worth
            position_info[-1, 3] = drawdown_pct
        
        # Combine the data window with the position information
        observation = np.hstack((window_data, position_info))
        
        return observation.astype(np.float32)
    
    def _calculate_reward(self) -> float:
        """
        Calcula la recompensa para el agente con un sistema balanceado que promueve
        exploración pero evita valores extremos que pueden causar inestabilidad.
        
        Returns:
            float: Recompensa total para el paso actual
        """
        # Inicializar los componentes de recompensa
        reward = 0.0
        self._reward_components = {}
        
        # RECOMPENSA MODERADA POR TENER POSICIÓN ACTIVA (+0.05 por cada paso)
        if self.trade_active:
            active_reward = 0.05
            reward += active_reward
            self._reward_components['active_reward'] = active_reward
        else:
            self._reward_components['active_reward'] = 0.0
        
        # RECOMPENSA POR ABRIR POSICIÓN (+0.3)
        if self._just_opened_position:
            open_reward = 0.3
            reward += open_reward
            self._reward_components['open_reward'] = open_reward
        else:
            self._reward_components['open_reward'] = 0.0
        
        # RECOMPENSA POR PNL (BALANCEADA)
        if self._just_closed_position:
            # Normalizar PnL por el tamaño de la posición * precio de entrada
            entry_price = self.trades[-1]['entry_price'] if self.trades else 1.0
            position_size = abs(self.trades[-1]['size']) if self.trades else 1.0
            
            # Evitar división por cero
            if entry_price * position_size > 0:
                # Escalar PnL y limitar para evitar valores extremos
                pnl_reward = np.clip(
                    self._last_trade_pnl / (entry_price * position_size) * 5.0,
                    -1.0, 1.0
                )
            else:
                pnl_reward = np.clip(self._last_trade_pnl * 5.0, -1.0, 1.0)
            
            reward += pnl_reward
            self._reward_components['pnl_reward'] = pnl_reward
            
            # RECOMPENSA ESPECÍFICA POR OPERACIÓN GANADORA
            if self._last_trade_pnl > 0:
                win_reward = 0.5
                reward += win_reward
                self._reward_components['win_reward'] = win_reward
            else:
                self._reward_components['win_reward'] = 0.0
        else:
            self._reward_components['pnl_reward'] = 0.0
            self._reward_components['win_reward'] = 0.0
        
        # PENALIZACIÓN POR INACTIVIDAD (MODERADA)
        inactivity_threshold = 50
        
        if not self.trade_active and self.steps_since_last_trade > inactivity_threshold:
            # Penalización que aumenta gradualmente, pero con límite
            inactivity_factor = min(1.0, (self.steps_since_last_trade - inactivity_threshold) / 100.0)
            inactivity_penalty = -0.05 * inactivity_factor
            reward += inactivity_penalty
            self._reward_components['inactivity_penalty'] = inactivity_penalty
        else:
            self._reward_components['inactivity_penalty'] = 0.0
        
        # RECOMPENSA POR PNL POSITIVO NO REALIZADO (MODERADA)
        if self.trade_active and self.unrealized_pnl > 0:
            # Normalizar y limitar
            position_value = abs(self.position) * self.entry_price
            if position_value > 0:
                unrealized_reward = min(0.1, self.unrealized_pnl / position_value * 0.2)
                reward += unrealized_reward
                self._reward_components['unrealized_reward'] = unrealized_reward
            else:
                self._reward_components['unrealized_reward'] = 0.0
        else:
            self._reward_components['unrealized_reward'] = 0.0
        
        # PENALIZACIÓN POR MANTENER POSICIÓN PERDEDORA
        if self.trade_active and self.unrealized_pnl < 0 and self.steps_with_position > 20:
            # Penalización moderada que aumenta con el tiempo
            losing_position_factor = min(1.0, (self.steps_with_position - 20) / 50.0)
            losing_position_penalty = -0.05 * losing_position_factor
            reward += losing_position_penalty
            self._reward_components['losing_position_penalty'] = losing_position_penalty
        else:
            self._reward_components['losing_position_penalty'] = 0.0
        
        # Registrar los componentes de recompensa si está habilitado
        if self.log_rewards and (self.current_step % 100 == 0 or self._just_opened_position or self._just_closed_position):
            components_str = " | ".join([f"{k}: {v:.2f}" for k, v in self._reward_components.items() if v != 0.0])
            self.logger.info(f"Step {self.current_step} Reward: {reward:.2f} Components: {components_str}")
        
        return reward
    
    def _process_stop_limits(self) -> Tuple[bool, float]:
        """
        Process stop loss and take profit limits.
        
        Returns:
            Tuple[bool, float]: Tuple of (was_triggered, pnl)
        """
        if self.position == 0:
            return False, 0.0
        
        price = self.data.iloc[self.current_step]['close']
        
        # Check for stop loss trigger
        if self.position > 0 and price <= self.stop_loss:
            return self._close_position(self.stop_loss, "stop_loss")
        elif self.position < 0 and price >= self.stop_loss:
            return self._close_position(self.stop_loss, "stop_loss")
        
        # Check for take profit trigger
        if self.position > 0 and price >= self.take_profit:
            return self._close_position(self.take_profit, "take_profit")
        elif self.position < 0 and price <= self.take_profit:
            return self._close_position(self.take_profit, "take_profit")
        
        return False, 0.0
    
    def _close_position(self, price: float, reason: str) -> Tuple[bool, float]:
        """
        Close the current position.
        
        Args:
            price (float): Price at which to close position
            reason (str): Reason for closing position
            
        Returns:
            Tuple[bool, float]: Tuple of (was_closed, pnl)
        """
        if self.position == 0:
            return False, 0.0
        
        # Calculate PnL
        pnl = (price - self.entry_price) * self.position * self.contract_size
        
        # Apply commission
        commission = price * abs(self.position) * self.commission_rate
        pnl -= commission
        
        # Update balance
        self.balance += pnl
        
        # Actualizar la última operación en self.trades con la información de cierre
        if self.trades:
            self.trades[-1]['exit_step'] = self.current_step
            self.trades[-1]['exit_price'] = price
            self.trades[-1]['pnl'] = pnl
            self.trades[-1]['reason'] = reason
        
        # Record trade en trade_history
        trade = {
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'position': self.position,
            'exit_step': self.current_step,
            'exit_price': price,
            'pnl': pnl,
            'reason': reason
        }
        self.trade_history.append(trade)
        
        # Set flags para tracking
        self._just_closed_position = True
        self._last_trade_pnl = pnl
        self._last_trade_action_initiated = (reason == "agent_action")
        
        # Reset position
        self.position = 0
        self.entry_price = 0
        self.entry_time = 0
        self.stop_loss = None
        self.take_profit = None
        self.trade_active = False
        self.steps_with_position = 0
        
        return True, pnl
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics for current step."""
        # Calculate net worth
        price = self.data.iloc[self.current_step]['close']
        unrealized_pnl = 0.0
        
        if self.position != 0:
            unrealized_pnl = (price - self.entry_price) * self.position * self.contract_size
        
        self.net_worth = self.balance + unrealized_pnl
        
        # Update max net worth
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        
        # Calculate return
        returns = self.net_worth / self.initial_net_worth - 1
        self.returns.append(returns)
        
        # Calculate drawdown
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0
        self.drawdowns.append(drawdown)
        
        # Record performance metrics
        performance = {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'net_worth': self.net_worth,
            'return': returns,
            'drawdown': drawdown
        }
        self.performance_history.append(performance)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step in the environment.
        
        Args:
            action: Agent's action (continuous)
            
        Returns:
            observation: Agent's observation
            reward: Reward from the action
            terminated: Whether episode is terminated (e.g., reached goal, exceeded limits)
            truncated: Whether episode was truncated (e.g., time limit reached)
            info: Additional information
        """
        assert self.action_space.contains(action), f"Action {action} is invalid"
        assert not self.done, "Episode already done. Call reset() first."
        
        # Reset flags for reward calculation
        self._just_opened_position = False
        self._just_closed_position = False
        self._last_trade_pnl = 0.0
        self._last_trade_action_initiated = False
        
        # Inicializar trade_closed para evitar el error de variable no definida
        trade_closed = False
        
        # Get current prices at this step
        self.open_price = self.data.iloc[self.current_step]['open']
        self.high_price = self.data.iloc[self.current_step]['high']
        self.low_price = self.data.iloc[self.current_step]['low']
        self.current_price = self.data.iloc[self.current_step]['close']
        
        # ENTRENAMIENTO FORZADO EXTREMADAMENTE AGRESIVO
        # Forzar operaciones aleatorias para asegurar exploración
        if self.mode == 'train':
            # Modificar la acción directamente para forzar operaciones
            action = action.copy()  # Crear copia para no modificar la original
            
            # Fase 1: Forzar operaciones en los primeros 500 pasos con 95% de probabilidad
            if self.current_step < 500 and np.random.random() < 0.95:
                # Si no hay posición abierta, abrir una aleatoria
                if self.position == 0:
                    direction = np.random.choice([-1, 1])  # -1: short, 1: long
                    size = np.random.uniform(0.05, 0.3)  # Tamaño significativo
                    action[0] = direction * size
                    self.logger.info(f"[FORZADO INICIAL] Abriendo posición {direction} con tamaño {size:.3f}")
                else:
                    # 30% de probabilidad de cerrar posición existente
                    if np.random.random() < 0.3:
                        action[0] = 0
                        self.logger.info(f"[FORZADO INICIAL] Cerrando posición")
            
            # Fase 2: Mantener forzado agresivo hasta el paso 2000, pero con menor probabilidad
            elif self.current_step < 2000 and np.random.random() < 0.7:
                if self.position == 0:
                    direction = np.random.choice([-1, 1])
                    size = np.random.uniform(0.03, 0.2)
                    action[0] = direction * size
                    self.logger.info(f"[FORZADO INTERMEDIO] Abriendo posición {direction} con tamaño {size:.3f}")
                else:
                    # 20% de probabilidad de cerrar posición existente
                    if np.random.random() < 0.2:
                        action[0] = 0
                        self.logger.info(f"[FORZADO INTERMEDIO] Cerrando posición")
            
            # Fase 3: Forzar operaciones si hay inactividad prolongada
            elif self.position == 0 and self.steps_since_last_trade > 20:
                # Probabilidad aumenta con la inactividad
                force_prob = min(0.5 + (self.steps_since_last_trade - 20) * 0.02, 0.95)
                
                if np.random.random() < force_prob:
                    direction = np.random.choice([-1, 1])
                    size = np.random.uniform(0.03, 0.15)
                    action[0] = direction * size
                    self.logger.info(f"[FORZADO POR INACTIVIDAD] Después de {self.steps_since_last_trade} pasos")
            
            # Fase 4: Forzar cierre si mantiene posición perdedora demasiado tiempo
            elif self.trade_active and self.steps_with_position > 30:
                # Si la posición está en pérdida
                if (self.position > 0 and self.unrealized_pnl < 0) or (self.position < 0 and self.unrealized_pnl < 0):
                    # Aumentar probabilidad de cierre con el tiempo
                    close_prob = min(0.4 + (self.steps_with_position - 30) * 0.03, 0.9)
                    
                    if np.random.random() < close_prob:
                        action[0] = 0
                        self.logger.info(f"[CIERRE FORZADO] Posición perdedora después de {self.steps_with_position} pasos")
        
        # Process the action - direction is the first dimension
        new_position = action[0]
        
        # Normalize position to allowed range
        new_position = np.clip(new_position, -self.max_position, self.max_position)
        
        # If no change in position, update steps counter and skip the rest
        if abs(new_position - self.position) < 1e-5:
            # Actualizar contadores
            if self.position != 0:
                self.steps_with_position += 1
            else:
                self.steps_since_last_trade += 1
                
            # Apply stop-loss and take-profit if position is open
            if self.trade_active:
                triggered, reward_adjustment = self._process_stop_limits()
                
                # Skip further actions if a stop/limit was triggered
                if triggered:
                    trade_closed = True  # Asegurarnos de actualizar trade_closed aquí
                    reward = self._calculate_reward() + reward_adjustment
                    
                    # Get next observation
                    observation = self._get_observation()
                    
                    # Check if done
                    terminated = False
                    truncated = False
                    
                    # Episode is done if we've reached the end of data minus offset
                    if self.current_step >= len(self.data) - 1:
                        terminated = True
                        
                        # Close any open position at the end
                        if self.trade_active:
                            final_price = self.data.iloc[-1]['close']
                            trade_closed, final_reward = self._close_position(final_price, "end_of_episode")
                            reward += final_reward
                    
                    # Implement early stopping if we've lost too much money - UMBRAL REDUCIDO
                    max_drawdown = self.config.get('max_drawdown', 0.15)  # Reducido de 0.25 a 0.15
                    if self.drawdowns and self.drawdowns[-1] > max_drawdown:
                        terminated = True
                        reward -= self.config.get('bankruptcy_penalty', 2.0)  # Penalización aumentada
                    
                    # Truncate if we've reached the maximum steps per episode
                    max_steps = self.config.get('max_steps', 10000)
                    if self.current_step - self.window_size >= max_steps:
                        truncated = True
                    
                    # Create info dictionary
                    info = {
                        'step': self.current_step,
                        'price': self.current_price,
                        'balance': self.balance,
                        'position': self.position,
                        'equity': self.net_worth,
                        'reward_components': self._reward_components,
                        'trade_closed': trade_closed
                    }
                    
                    # Update current step
                    self.current_step += 1
                    
                    self.done = terminated or truncated
                    return observation, reward, terminated, truncated, info
        
        # Handle the case of position change
        reward = 0
        
        # Close existing position if changing from long/short to nothing or opposite
        if self.position != 0 and (new_position == 0 or new_position * self.position < 0):
            trade_closed, close_reward = self._close_position(self.current_price, "agent_action")
            reward += close_reward
        
        # Open new position if changing from nothing to long/short or flip position
        if new_position != 0 and (self.position == 0 or self.position * new_position < 0):
            # Determine direction
            direction = 1 if new_position > 0 else -1
            
            # Determine size (absolute value)
            size = abs(new_position)
            
            # Calculate capital to risk (as a portion of balance)
            risk_factor = size  # Direct mapping from position size to risk
            capital_at_risk = self.balance * risk_factor
            
            # Set stop loss based on % of balance (config or fixed)
            stop_loss_pct = self.config.get('stop_loss_pct', 0.02)  # Default 2% loss
            take_profit_ratio = self.config.get('take_profit_ratio', 1.5)  # Default TP is 1.5x SL
            
            # Calculate stop distance in price points
            risk_per_point = self.contract_size  # For simplicity, assume 1 contract = $1 per point
            stop_distance = capital_at_risk * stop_loss_pct / (risk_per_point * size)
            
            # Set stop loss and take profit levels
            if direction > 0:  # Long position
                self.stop_loss = self.current_price - stop_distance
                self.take_profit = self.current_price + (stop_distance * take_profit_ratio)
            else:  # Short position
                self.stop_loss = self.current_price + stop_distance
                self.take_profit = self.current_price - (stop_distance * take_profit_ratio)
            
            # Open the position
            self.position = direction * size
            self.entry_price = self.current_price
            self.entry_time = self.current_step
            self.trade_active = True
            self.steps_with_position = 1  # Reset counter for steps with position
            self.steps_since_last_trade = 0  # Reset inactivity counter
            
            # Calculate commission for opening position
            commission = self.current_price * abs(self.position) * self.commission_rate
            self.balance -= commission
            
            # Record trade entry
            trade = {
                'entry_time': self.current_step,
                'entry_price': self.current_price,
                'position': self.position,
                'size': self.position,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'commission': commission,
                'balance_at_entry': self.balance
            }
            self.trades.append(trade)
            
            # Set flag for reward calculation
            self._just_opened_position = True
            self._last_trade_action_initiated = True
            
            # Add to position history
            self.position_history.append((self.current_step, self.position))
        
        # Calculate unrealized PnL for current position
        if self.position != 0:
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.position * self.contract_size
        else:
            self.unrealized_pnl = 0.0
        
        # Update performance metrics (equity, returns, drawdowns)
        self._update_performance_metrics()
        
        # Calculate reward for this step
        reward += self._calculate_reward()
        
        # Get observation for next step
        observation = self._get_observation()
        
        # Check if this episode is done
        terminated = False
        truncated = False
        
        # Episode is done if we've reached the end of data
        if self.current_step >= len(self.data) - 1:
            terminated = True
            
            # Close any open position at the end of the episode
            if self.trade_active:
                final_price = self.data.iloc[-1]['close']
                trade_closed, final_reward = self._close_position(final_price, "end_of_episode")
                reward += final_reward
        
        # Implement early stopping if we've lost too much money
        max_drawdown = self.config.get('max_drawdown', 0.15)
        if self.drawdowns and self.drawdowns[-1] > max_drawdown:
            terminated = True
            reward -= self.config.get('bankruptcy_penalty', 2.0)
        
        # Truncate if we've reached the maximum steps per episode
        max_steps = self.config.get('max_steps', 10000)
        if self.current_step - self.window_size >= max_steps:
            truncated = True
        
        # Update done flag
        self.done = terminated or truncated
        
        # Create info dictionary with relevant information
        info = {
            'step': self.current_step,
            'price': self.current_price,
            'balance': self.balance,
            'position': self.position,
            'equity': self.net_worth,
            'reward_components': self._reward_components,
            'trade_closed': trade_closed
        }
        
        # Move to next step
        self.current_step += 1
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset step counter
        self.current_step = self.window_size  # Start after window_size to have enough history
        
        # Reset account
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.entry_time = 0
        self.unrealized_pnl = 0.0
        
        # Reset trading state
        self.trade_active = False
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.steps_with_position = 0
        self.steps_since_last_trade = 0
        
        # Reset performance tracking
        self.trades = []
        self.trade_history = []
        self.returns = [0]
        self.drawdowns = [0]
        self.equity_curve = [self.initial_balance]
        self.position_history = []
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        # Reset reward tracking
        self._reward_components = {}
        self._just_opened_position = False
        self._just_closed_position = False
        self._last_trade_pnl = 0.0
        self._last_trade_action_initiated = False
        
        # Reset done flag
        self.done = False
        
        # Get initial observation
        observation = self._get_observation()
        
        # Initial info
        info = {
            'step': self.current_step,
            'price': self.data.iloc[self.current_step]['close'],
            'balance': self.balance,
            'position': self.position,
            'equity': self.net_worth
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
        print(f"Position: {self.position}")
        
        if self.position != 0:
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