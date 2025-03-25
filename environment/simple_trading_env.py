#!/usr/bin/env python
"""
Entorno de trading simplificado para diagnóstico y pruebas.
Este entorno elimina todas las validaciones y complejidades innecesarias
para facilitar el diagnóstico de problemas fundamentales.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simple_env.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("simple_env")

class SimpleTradingEnv(gym.Env):
    """
    Entorno de trading simplificado para diagnóstico y pruebas.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, data, config=None, window_size=60, mode='train'):
        """
        Inicializar el entorno simplificado.
        
        Args:
            data: DataFrame con datos OHLCV
            config: Configuración del entorno
            window_size: Tamaño de la ventana de observación
            mode: Modo de operación ('train', 'validation', 'test')
        """
        super(SimpleTradingEnv, self).__init__()
        
        # Configuración básica
        self.data = data.copy() if data is not None else None
        self.config = config or {}
        self.window_size = window_size
        self.mode = mode
        self.logger = logger
        
        # Uso de listas para simplicidad
        self.trades = []
        self.position_history = []
        
        # Parámetros de entorno
        self.initial_balance = self.config.get('initial_balance', 10000.0)
        self.commission_rate = self.config.get('commission_rate', 0.001)
        self.max_position = self.config.get('max_position', 1)
        self.action_space_type = self.config.get('action_space_type', 'discrete')
        
        # Variables de estado
        self.balance = self.initial_balance
        self.current_position = 0
        self.entry_price = 0.0
        self.current_step = 0
        self.trade_active = False
        self.inactive_steps = 0
        self.position_duration = 0
        self.current_pnl = 0.0
        self.trade_completed = False
        self.last_trade_pnl = 0.0
        
        # Definir espacios de acción y observación
        if self.action_space_type == 'discrete':
            self.action_space = spaces.Discrete(3)  # 0: hold/close, 1: buy, 2: sell
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observación: ventana de datos de precio + estado de la cuenta
        num_features = self.data.shape[1] if data is not None else 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, num_features), 
            dtype=np.float32
        )
        
        self.observation = None
        self.reset()
        
        logger.info(f"SimpleTradingEnv inicializado: balance={self.initial_balance}, modo={mode}")
    
    def reset(self, seed=None, options=None):
        """
        Reiniciar el entorno.
        
        Args:
            seed: Semilla para reproducibilidad
            options: Opciones adicionales
            
        Returns:
            observation: Observación inicial
            info: Información adicional
        """
        # Reiniciar semilla si se proporciona
        if seed is not None:
            np.random.seed(seed)
        
        # Reiniciar variables de estado
        self.balance = self.initial_balance
        self.current_position = 0
        self.entry_price = 0.0
        self.current_step = self.window_size - 1  # Empezar después de la ventana inicial
        self.inactive_steps = 0
        self.position_duration = 0
        self.current_pnl = 0.0
        self.trade_active = False
        self.trades = []
        self.position_history = []
        self.trade_completed = False
        self.last_trade_pnl = 0.0
        
        # Obtener observación inicial
        self.observation = self._get_observation()
        
        info = {
            'balance': self.balance,
            'position': self.current_position
        }
        
        return self.observation, info
    
    def step(self, action):
        """
        Ejecutar un paso en el entorno.
        
        Args:
            action: Acción a ejecutar (0: hold/close, 1: buy, 2: sell, o valor continuo)
            
        Returns:
            observation: Nueva observación
            reward: Recompensa obtenida
            done: Indicador de fin de episodio
            truncated: Indicador de truncamiento
            info: Información adicional
        """
        # Procesar acción para formato estandarizado
        action_value = self._process_action(action)
        
        logger.debug(f"Paso {self.current_step}: Acción={action}, Procesada={action_value}")
        
        # Guardar estado anterior
        prev_balance = self.balance
        prev_position = self.current_position
        
        # Reiniciar flags
        self.trade_completed = False
        
        # Actualizar contadores
        self.current_step += 1
        if self.current_position != 0:
            self.position_duration += 1
        
        # Gestión de inactividad
        if self.current_position == 0 and action_value == 0:
            self.inactive_steps += 1
            logger.debug(f"Inactividad: {self.inactive_steps} pasos")
        else:
            self.inactive_steps = 0
        
        # Ejecutar acción
        if action_value == 0:  # Cerrar/Hold
            if self.current_position != 0:
                logger.debug("Cerrando posición...")
                self._close_position()
        elif action_value == 1:  # Comprar/Long
            if self.current_position <= 0:
                if self.current_position < 0:
                    self._close_position()
                logger.debug("Abriendo posición larga...")
                self._open_position(1)
        elif action_value == 2:  # Vender/Short
            if self.current_position >= 0:
                if self.current_position > 0:
                    self._close_position()
                logger.debug("Abriendo posición corta...")
                self._open_position(-1)
        
        # Verificar fin de datos
        done = self.current_step >= len(self.data) - 1
        
        # Actualizar observación
        self.observation = self._get_observation()
        
        # Calcular recompensa
        reward = self._calculate_reward(prev_balance)
        
        # Información adicional
        info = {
            'balance': self.balance,
            'position': self.current_position,
            'action_taken': action_value,
            'inactive_steps': self.inactive_steps,
            'position_changed': prev_position != self.current_position,
            'trade_completed': self.trade_completed,
            'num_trades': len(self.trades)
        }
        
        return self.observation, reward, done, False, info
    
    def _process_action(self, action):
        """
        Procesar acción a formato estandarizado (0, 1, 2).
        
        Args:
            action: Acción original
            
        Returns:
            int: Acción procesada
        """
        if isinstance(action, (np.ndarray, list)):
            if len(action) > 0:
                action_value = action[0]
                # Usar umbrales extremadamente permisivos
                if action_value > 0.1:  # Reducido de 0.3
                    return 1  # Comprar
                elif action_value < -0.1:  # Reducido de -0.3
                    return 2  # Vender
                return 0  # Mantener
            else:
                return 0
        return action  # Para acciones discretas
    
    def _get_observation(self):
        """
        Obtener observación del entorno.
        
        Returns:
            np.ndarray: Observación
        """
        if self.current_step < self.window_size:
            # Si estamos al inicio, usar los primeros window_size datos
            data_window = self.data.iloc[:self.window_size].values
        else:
            # Obtener ventana de datos
            start_idx = self.current_step - self.window_size + 1
            end_idx = self.current_step + 1
            data_window = self.data.iloc[start_idx:end_idx].values
        
        # Asegurar forma correcta
        if data_window.shape[0] != self.window_size:
            # Rellenar con ceros si es necesario
            padding = np.zeros((self.window_size - data_window.shape[0], data_window.shape[1]))
            data_window = np.vstack([padding, data_window])
        
        return data_window.astype(np.float32)
    
    def _open_position(self, direction):
        """
        Abrir posición de trading.
        
        Args:
            direction: Dirección (1: long, -1: short)
            
        Returns:
            bool: Éxito de la operación
        """
        # SIMPLIFICACIÓN: Abrir siempre sin validaciones excesivas
        
        # Obtener precio actual
        price = self.data.iloc[self.current_step]['close']
        
        # Establecer dirección y tamaño
        position_size = direction * self.max_position
        
        # Registrar posición
        self.current_position = position_size
        self.entry_price = price
        self.trade_active = True
        
        # Registrar en historial
        self.position_history.append((self.current_step, position_size, price))
        
        logger.debug(f"Posición abierta: dirección={direction}, precio={price}")
        
        return True
    
    def _close_position(self):
        """
        Cerrar posición actual.
        
        Returns:
            float: PnL de la operación
        """
        # Si no hay posición abierta, no hacer nada
        if self.current_position == 0:
            return 0.0
        
        # Obtener precio actual
        price = self.data.iloc[self.current_step]['close']
        
        # Calcular PnL
        direction = 1 if self.current_position > 0 else -1
        pnl_points = (price - self.entry_price) * direction
        pnl = pnl_points * abs(self.current_position)
        
        # Aplicar comisión
        commission = abs(self.current_position) * price * self.commission_rate
        pnl -= commission
        
        # Actualizar balance
        self.balance += pnl
        
        # Registrar operación
        trade_info = {
            'entry_step': self.current_step - self.position_duration,
            'exit_step': self.current_step,
            'direction': direction,
            'entry_price': self.entry_price,
            'exit_price': price,
            'pnl': pnl,
            'commission': commission
        }
        self.trades.append(trade_info)
        
        # Actualizar métricas
        self.current_pnl += pnl
        self.last_trade_pnl = pnl
        self.trade_completed = True
        
        # Registrar en historial
        self.position_history.append((self.current_step, 0, price))
        
        # Reiniciar posición
        self.current_position = 0
        self.entry_price = 0.0
        self.trade_active = False
        self.position_duration = 0
        
        logger.debug(f"Posición cerrada: PnL={pnl}, precio={price}")
        
        return pnl
    
    def _calculate_reward(self, prev_balance):
        """
        Calcular recompensa del paso actual.
        
        Args:
            prev_balance: Balance anterior
            
        Returns:
            float: Recompensa
        """
        # Recompensa base MÍNIMA por paso
        reward = -0.001
        
        # SIMPLIFICACIÓN: Recompensa masiva por operaciones
        if self.trade_completed:
            # Enorme bonus por completar operaciones
            reward += 20.0
            
            # Bonus adicional por operaciones ganadoras
            if self.last_trade_pnl > 0:
                reward += 30.0
        
        # Penalización mayor por inactividad prolongada
        if self.inactive_steps > 10:
            reward -= self.inactive_steps * 0.5
        
        return reward
    
    def get_performance_summary(self):
        """
        Obtener resumen de rendimiento.
        
        Returns:
            dict: Métricas de rendimiento
        """
        # Si no hay operaciones, retornar estadísticas básicas
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_pnl': 0.0,
                'balance': self.balance
            }
        
        # Calcular estadísticas
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
        win_rate = winning_trades / total_trades
        
        total_profit = sum(trade['pnl'] for trade in self.trades if trade['pnl'] > 0)
        total_loss = sum(abs(trade['pnl']) for trade in self.trades if trade['pnl'] < 0)
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        logger.debug(f"Performance: trades={total_trades}, win_rate={win_rate}, profit_factor={profit_factor}")
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': sum(trade['pnl'] for trade in self.trades),
            'avg_trade': sum(trade['pnl'] for trade in self.trades) / total_trades,
            'balance': self.balance
        }
    
    def render(self, mode='human'):
        """
        Renderizar el estado actual del entorno.
        
        Args:
            mode: Modo de renderizado
            
        Returns:
            None
        """
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Date: {self.data.index[self.current_step] if hasattr(self.data, 'index') else self.current_step}")
            print(f"Price: {self.data.iloc[self.current_step]['close']}")
            print(f"Balance: {self.balance}")
            print(f"Position: {self.current_position}")
            if self.current_position != 0:
                print(f"Entry Price: {self.entry_price}")
            print(f"Trades: {len(self.trades)}")
            print("-" * 50)
        return None 