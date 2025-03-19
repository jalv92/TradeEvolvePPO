"""
Custom trading environment for reinforcement learning.
Implements a Gymnasium environment for trading NQ futures.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional, Union, Any


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
        self.performance_history = []
        
        # Risk management parameters
        self.stop_loss_pct = config.get('stop_loss_pct', 2.0)
        self.take_profit_pct = config.get('take_profit_pct', 4.0)
        self.max_position = config.get('max_position', 5)
        self.max_drawdown_pct = config.get('max_drawdown_pct', 20.0)
        self.contract_size = config.get('contract_size', 1)
        self.transaction_fee = config.get('transaction_fee', 0.0)
        self.slippage = config.get('slippage', 0.0)
        
        # Current trade parameters
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        
        # Performance tracking
        self.initial_net_worth = self.initial_balance
        self.net_worth = self.initial_net_worth
        self.max_net_worth = self.initial_net_worth
        self.returns = []
        self.drawdowns = []
        
        # Check that data has enough records
        if len(self.data) <= self.window_size:
            raise ValueError(f"Data length ({len(self.data)}) must be greater than window size ({self.window_size})")
        
        # Define action space
        # [buy/sell/hold, position_size, stop_loss_pct, take_profit_pct]
        self.action_space = spaces.Box(
            low=np.array([-1, 0, 0.5, 1.0]), 
            high=np.array([1, 1, 5.0, 10.0]),
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
        Calculate the reward for the current step.
        
        Returns:
            float: Reward value
        """
        reward_config = self.config.get('reward_config', {})
        price = self.data.iloc[self.current_step]['close']
        
        # Initialize reward components
        pnl_reward = 0.0
        risk_reward = 0.0
        
        # Calculate PnL reward
        if len(self.trade_history) > 0 and self.trade_history[-1]['exit_step'] == self.current_step:
            # Realized profit
            last_trade = self.trade_history[-1]
            pnl = last_trade['pnl']
            pnl_pct = pnl / self.initial_balance
            
            # Scale reward based on PnL
            pnl_weight = reward_config.get('pnl_weight', 1.0)
            pnl_reward = pnl_weight * pnl_pct
        
        # Risk management reward/penalty
        risk_weight = reward_config.get('risk_weight', 0.5)
        
        # Penalize excessive position sizes
        if abs(self.position) > 0.8 * self.max_position:
            exposure_penalty = reward_config.get('exposure_penalty', 0.01)
            risk_reward -= exposure_penalty
        
        # Penalize large drawdowns
        current_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0
        if current_drawdown > 0.1:  # 10% drawdown
            drawdown_penalty = reward_config.get('drawdown_penalty', 0.1)
            risk_reward -= drawdown_penalty * current_drawdown
        
        # Reward for good stop loss and take profit setup
        if self.position != 0:
            sl_tp_ratio = abs(self.stop_loss - self.entry_price) / abs(self.take_profit - self.entry_price)
            if 0.3 <= sl_tp_ratio <= 0.5:  # Good risk/reward ratio
                consistency_bonus = reward_config.get('consistency_bonus', 0.05)
                risk_reward += consistency_bonus
        
        # Small penalty for any trade to discourage excessive trading
        trade_penalty = reward_config.get('trade_penalty', 0.001)
        if self.trades and self.trades[-1]['step'] == self.current_step:
            risk_reward -= trade_penalty
        
        # Combine reward components
        total_reward = pnl_reward + risk_reward
        
        # Scale reward
        reward_scaling = self.config.get('reward_scaling', 1.0)
        total_reward *= reward_scaling
        
        return float(total_reward)
    
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
        
        # Apply transaction fee
        pnl -= abs(self.position) * self.transaction_fee
        
        # Update balance
        self.balance += pnl
        
        # Record trade
        trade = {
            'entry_step': self.trades[-1]['step'],
            'entry_price': self.entry_price,
            'position': self.position,
            'exit_step': self.current_step,
            'exit_price': price,
            'pnl': pnl,
            'reason': reason
        }
        self.trade_history.append(trade)
        
        # Reset position
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        
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
        Execute one step in the environment.
        
        Args:
            action (np.ndarray): Action to take
            
        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: 
                (observation, reward, terminated, truncated, info)
        """
        # Make sure we have a valid action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Extract action components
        action_type = action[0]  # -1 to 1 (negative=sell, positive=buy, near zero=hold)
        position_size = action[1]  # 0 to 1 (scaled by max_position)
        sl_pct = action[2]  # 0.5 to 5.0 (percentage for stop loss)
        tp_pct = action[3]  # 1.0 to 10.0 (percentage for take profit)
        
        # Apply slippage to current price (random factor)
        current_price = self.data.iloc[self.current_step]['close']
        slippage_factor = 1.0 + np.random.uniform(-self.slippage, self.slippage)
        executed_price = current_price * slippage_factor
        
        # Check if we need to close the current position due to stop/limit orders
        stop_triggered, stop_pnl = self._process_stop_limits()
        
        # Process the action
        old_position = self.position
        trade_executed = False
        
        # Only execute a new trade if no stop/limit was triggered
        if not stop_triggered:
            # Determine the action type
            # Action type between -0.33 and 0.33 means HOLD
            # Below -0.33 means SELL, above 0.33 means BUY
            if action_type > 0.33:  # BUY
                # Close any existing short position
                if self.position < 0:
                    self._close_position(executed_price, "action")
                
                # Calculate new position size
                target_position = int(position_size * self.max_position)
                new_position = max(0, min(target_position, self.max_position))
                
                # Only execute if position would change
                if new_position > self.position:
                    # Calculate additional contracts to buy
                    additional_contracts = new_position - self.position
                    
                    # Check if we have enough balance
                    cost = additional_contracts * executed_price * self.contract_size
                    if cost <= self.balance:
                        # Update position and record trade
                        self.position = new_position
                        
                        # Calculate weighted average entry price if adding to position
                        if old_position > 0:
                            self.entry_price = ((old_position * self.entry_price) + (additional_contracts * executed_price)) / self.position
                        else:
                            self.entry_price = executed_price
                        
                        # Apply transaction fee
                        self.balance -= additional_contracts * self.transaction_fee
                        
                        # Set stop loss and take profit levels
                        self.stop_loss = self.entry_price * (1.0 - sl_pct / 100.0)
                        self.take_profit = self.entry_price * (1.0 + tp_pct / 100.0)
                        
                        # Record trade
                        trade = {
                            'step': self.current_step,
                            'price': executed_price,
                            'position_change': additional_contracts,
                            'position': self.position,
                            'type': 'buy',
                            'stop_loss': self.stop_loss,
                            'take_profit': self.take_profit
                        }
                        self.trades.append(trade)
                        trade_executed = True
            
            elif action_type < -0.33:  # SELL
                # Close any existing long position
                if self.position > 0:
                    self._close_position(executed_price, "action")
                
                # Calculate new position size (negative for short)
                target_position = -int(position_size * self.max_position)
                new_position = min(0, max(target_position, -self.max_position))
                
                # Only execute if position would change
                if new_position < self.position:
                    # Calculate additional contracts to sell
                    additional_contracts = abs(new_position - self.position)
                    
                    # Check if we have enough balance
                    margin_required = additional_contracts * executed_price * self.contract_size * 0.5  # 50% margin
                    if margin_required <= self.balance:
                        # Update position and record trade
                        self.position = new_position
                        
                        # Calculate weighted average entry price if adding to position
                        if old_position < 0:
                            self.entry_price = ((abs(old_position) * self.entry_price) + (additional_contracts * executed_price)) / abs(self.position)
                        else:
                            self.entry_price = executed_price
                        
                        # Apply transaction fee
                        self.balance -= additional_contracts * self.transaction_fee
                        
                        # Set stop loss and take profit levels
                        self.stop_loss = self.entry_price * (1.0 + sl_pct / 100.0)
                        self.take_profit = self.entry_price * (1.0 - tp_pct / 100.0)
                        
                        # Record trade
                        trade = {
                            'step': self.current_step,
                            'price': executed_price,
                            'position_change': -additional_contracts,
                            'position': self.position,
                            'type': 'sell',
                            'stop_loss': self.stop_loss,
                            'take_profit': self.take_profit
                        }
                        self.trades.append(trade)
                        trade_executed = True
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Check for bankruptcy or max drawdown
        terminated = False
        if self.balance <= 0:
            terminated = True
        
        if self.config.get('done_on_max_drawdown', True):
            current_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0
            if current_drawdown >= self.max_drawdown_pct / 100.0:
                terminated = True
        
        # Truncated if we've reached the end of the data
        truncated = self.current_step >= len(self.data) - 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Move to the next step
        self.current_step += 1
        done = terminated or truncated
        
        # Get the next observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'net_worth': self.net_worth,
            'position': self.position,
            'entry_price': self.entry_price,
            'current_price': current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trade_executed': trade_executed,
            'stop_triggered': stop_triggered,
            'max_drawdown': max(self.drawdowns) if self.drawdowns else 0.0,
            'total_trades': len(self.trade_history),
            'terminated': terminated,
            'truncated': truncated,
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to the initial state.
        
        Args:
            seed (Optional[int], optional): Random seed. Defaults to None.
            options (Optional[Dict[str, Any]], optional): Additional options. Defaults to None.
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset variables
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.current_step = 0
        self.trades = []
        self.trade_history = []
        self.performance_history = []
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.returns = []
        self.drawdowns = []
        
        # Get initial observation
        observation = self._get_observation()
        
        # Info dictionary
        info = {
            'initial_balance': self.initial_balance,
            'date': self.dates[self.current_step] if self.current_step < len(self.dates) else None,
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