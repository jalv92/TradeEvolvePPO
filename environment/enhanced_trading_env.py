"""
Entorno de trading mejorado con características avanzadas de gestión de riesgo.
Extiende el entorno base TradingEnv con mejoras para evitar hipertrading.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import random

from environment.trading_env import TradingEnv

logger = logging.getLogger(__name__)

class EnhancedTradingEnv(TradingEnv):
    """
    Entorno de trading mejorado con características avanzadas de gestión de riesgo.
    Implementa mejoras para evitar hipertrading y optimizar la gestión de posiciones.
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 config: Dict[str, Any], 
                 initial_balance: float = 100000.0,
                 window_size: int = 60,
                 mode: str = 'train'):
        """
        Inicializa el entorno de trading mejorado.
        
        Args:
            data (pd.DataFrame): DataFrame con datos OHLCV
            config (Dict[str, Any]): Parámetros de configuración
            initial_balance (float, optional): Balance inicial. Por defecto 100000.0.
            window_size (int, optional): Número de velas en la observación. Por defecto 60.
            mode (str, optional): Modo, puede ser 'train', 'validation', 'eval' o 'inference'. Por defecto 'train'.
        """
        # Configuración mejorada para evitar hipertrading
        enhanced_config = config.copy()
        
        # Valores predeterminados mejorados
        enhanced_defaults = {
            # Parámetros de gestión de posiciones
            'min_hold_steps': 30,  # Duración mínima recomendada (aumentada a 30)
            'position_cooldown': 40,  # Tiempo de enfriamiento (aumentado a 40)
            'force_min_hold': True,  # Siempre forzar duración mínima
            
            # Parámetros de tamaño de operaciones
            'min_sl_ticks': 50,  # Stop loss mínimo en ticks (aumentado a 50)
            'min_tp_ticks': 50,  # Take profit mínimo en ticks (aumentado a 50)
            'tick_size': 0.25,  # Tamaño del tick (0.25 para NQ)
            'enforce_min_trade_size': True,  # Forzar tamaño mínimo
            
            # Parámetros de gestión de SL/TP
            'sl_buffer_ticks': 5,  # Buffer para SL (en ticks)
            'tp_buffer_ticks': 5,  # Buffer para TP (en ticks)
            'trailing_stop_distance_ticks': 20,  # Distancia del trailing stop (en ticks)
            'breakeven_activation_threshold': 0.15,  # Umbral para activar break-even (%)
            'trailing_activation_threshold': 0.3,  # Umbral para activar trailing stop (%)
            
            # Parámetros de recompensa
            'duration_scaling_factor': 8.0,  # Factor de escala para recompensas por duración
            'short_trade_penalty_factor': 25.0,  # Penalización por operaciones cortas
            
            # Parámetros de exploración
            'force_action_prob': 0.2,  # Probabilidad de forzar acción (reducida para menos operaciones)
            'position_change_threshold': 0.3,  # Umbral para cambios de posición (aumentado para menos cambios)
            
            # Logging
            'log_trades': True,  # Activar logging de trades
            'log_ticks': True,  # Activar logging de ticks
        }
        
        # Actualizar configuración con valores predeterminados mejorados
        for key, value in enhanced_defaults.items():
            if key not in enhanced_config:
                enhanced_config[key] = value
        
        # Inicializar entorno base
        super(EnhancedTradingEnv, self).__init__(
            data=data,
            config=enhanced_config,
            initial_balance=initial_balance,
            window_size=window_size,
            mode=mode
        )
        
        # Variables adicionales para tracking de ticks
        self.positive_ticks = 0
        self.negative_ticks = 0
        self.positive_ticks_total = 0
        self.negative_ticks_total = 0
        
        # Variables para diagnóstico
        self.premature_closures = 0
        self.total_closures = 0
        
        logger.info("Entorno de trading mejorado inicializado con configuración anti-hipertrading")
    
    def _check_sl_tp(self):
        """
        Versión mejorada de la verificación de stop loss y take profit.
        Implementa un período mínimo obligatorio más estricto y un buffer para evitar cierres prematuros.
        
        Returns:
            float: Recompensa por cierre de posición
        """
        if self.current_position == 0:
            return 0.0
        
        # Obtener configuración
        min_hold_steps = self.config.get('min_hold_steps', 30)  # Mínimo 30 barras por defecto (aumentado)
        force_min_hold = self.config.get('force_min_hold', True)  # Forzar período mínimo
        sl_buffer_ticks = self.config.get('sl_buffer_ticks', 5)  # Buffer para SL (en ticks)
        tp_buffer_ticks = self.config.get('tp_buffer_ticks', 5)  # Buffer para TP (en ticks)
        
        # Convertir buffer de ticks a precio
        sl_buffer = sl_buffer_ticks * self.tick_size
        tp_buffer = tp_buffer_ticks * self.tick_size
        
        # Verificar si estamos en período mínimo obligatorio
        in_min_hold_period = self.position_duration < min_hold_steps
        
        # MEJORADO: Si estamos en período mínimo y está activado force_min_hold, no verificar SL/TP
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
                
                # MEJORADO: Registrar tipo de cierre
                if hasattr(self, 'trades') and self.trades:
                    self.trades[-1]['close_type'] = 'STOP LOSS'
                
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
                
                # MEJORADO: Registrar tipo de cierre
                if hasattr(self, 'trades') and self.trades:
                    self.trades[-1]['close_type'] = 'TAKE PROFIT'
                
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
                
                # MEJORADO: Registrar tipo de cierre
                if hasattr(self, 'trades') and self.trades:
                    self.trades[-1]['close_type'] = 'STOP LOSS'
                
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
                
                # MEJORADO: Registrar tipo de cierre
                if hasattr(self, 'trades') and self.trades:
                    self.trades[-1]['close_type'] = 'TAKE PROFIT'
                
                # Cerrar a precio de take profit original (no el efectivo con buffer)
                return self._close_position(close_price=self.take_profit)
        
        return 0.0
    
    def _close_position(self, close_price: Optional[float] = None, close_reason: str = 'unknown'):
        """
        Versión mejorada del cierre de posición con tracking adicional.
        
        Args:
            close_price (Optional[float]): Precio de cierre. Si es None, usa el precio actual.
            close_reason (str): Razón del cierre de posición.
            
        Returns:
            bool: True si la posición se cerró con éxito, False en caso contrario.
        """
        if self.current_position == 0:
            logger.warning("Intentando cerrar posición cuando no hay ninguna abierta")
            return False
        
        # Registrar cierre en estadísticas
        self.total_closures += 1
        
        # Verificar si es un cierre prematuro
        if self.position_duration < self.min_hold_steps:
            self.premature_closures += 1
            logger.warning(f"Cierre prematuro: duración={self.position_duration}, mínimo recomendado={self.min_hold_steps}")
        
        # Registrar tipo de cierre si no está ya registrado
        if hasattr(self, 'trades') and self.trades and 'close_type' not in self.trades[-1]:
            self.trades[-1]['close_type'] = close_reason
        
        # Llamar al método base para cerrar la posición
        result = super(EnhancedTradingEnv, self)._close_position(close_price=close_price)
        
        # Activar cooldown después de cerrar posición
        if result:
            self.cooldown_counter = self.position_cooldown
            logger.info(f"Activando cooldown por {self.position_cooldown} pasos")
        
        return result
    
    def _open_position(self, direction, position_size=1):
        """
        Versión mejorada de apertura de posición con validaciones adicionales.
        
        Args:
            direction (int): 1 para posición larga, -1 para posición corta
            position_size (int): Tamaño de la posición
            
        Returns:
            bool: True si la posición se abrió con éxito, False en caso contrario
        """
        # Verificar cooldown
        if self.cooldown_counter > 0:
            logger.info(f"No se puede abrir posición durante cooldown ({self.cooldown_counter} pasos restantes)")
            return False
        
        # Llamar al método base para abrir la posición
        result = super(EnhancedTradingEnv, self)._open_position(direction, position_size)
        
        # Inicializar contadores de ticks
        if result and self.config.get('log_ticks', False):
            self.positive_ticks = 0
            self.negative_ticks = 0
        
        return result
    
    def step(self, action):
        """
        Versión mejorada del paso del entorno con tracking adicional.
        
        Args:
            action: Acción a realizar
            
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        # Llamar al método base para ejecutar el paso
        observation, reward, done, truncated, info = super(EnhancedTradingEnv, self).step(action)
        
        # Tracking adicional de ticks si hay posición abierta
        if self.current_position != 0 and self.config.get('log_ticks', False):
            # Calcular PnL no realizado
            unrealized_pnl = self._calculate_unrealized_pnl()
            
            # Comparar con PnL anterior
            if unrealized_pnl > self.prev_unrealized_pnl:
                self.positive_ticks += 1
                self.positive_ticks_total += 1
            elif unrealized_pnl < self.prev_unrealized_pnl:
                self.negative_ticks += 1
                self.negative_ticks_total += 1
            
            # Actualizar PnL anterior
            self.prev_unrealized_pnl = unrealized_pnl
            
            # Actualizar información de ticks en la operación actual
            if self.trades:
                self.trades[-1]['positive_ticks'] = self.positive_ticks
                self.trades[-1]['negative_ticks'] = self.negative_ticks
            
            # Actualizar información adicional
            info.update({
                'positive_ticks': self.positive_ticks,
                'negative_ticks': self.negative_ticks,
                'tick_ratio': self.positive_ticks / max(1, self.negative_ticks)
            })
        
        return observation, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        """
        Versión mejorada del reset del entorno con inicialización adicional.
        
        Args:
            seed (int, optional): Semilla aleatoria
            options (dict, optional): Opciones adicionales
            
        Returns:
            Tuple[np.ndarray, dict]: Observación e información
        """
        # Llamar al método base para resetear el entorno
        observation, info = super(EnhancedTradingEnv, self).reset(seed=seed, options=options)
        
        # Resetear variables adicionales
        self.positive_ticks = 0
        self.negative_ticks = 0
        self.positive_ticks_total = 0
        self.negative_ticks_total = 0
        self.premature_closures = 0
        self.total_closures = 0
        
        return observation, info
    
    def get_performance_summary(self):
        """
        Versión mejorada del resumen de rendimiento con métricas adicionales.
        
        Returns:
            Dict: Diccionario con métricas de rendimiento
        """
        # Obtener resumen base
        summary = super(EnhancedTradingEnv, self).get_performance_summary()
        
        # Añadir métricas adicionales
        summary.update({
            'positive_ticks_total': self.positive_ticks_total,
            'negative_ticks_total': self.negative_ticks_total,
            'tick_ratio_total': self.positive_ticks_total / max(1, self.negative_ticks_total),
            'premature_closures': self.premature_closures,
            'total_closures': self.total_closures,
            'premature_closure_pct': 100 * self.premature_closures / max(1, self.total_closures)
        })
        
        return summary
