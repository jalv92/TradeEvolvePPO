"""
Configuración global para el proyecto TradeEvolvePPO.
Define los valores por defecto para el entorno, agente y entrenamiento.
"""

import torch

# Configuración base
BASE_CONFIG = {
    'symbol': 'NQ_06-25_combined_20250320_225417',
    'timeframe': '5min',
    'start_date': '2022-01-01',
    'end_date': '2022-12-31',
    'seed': 42
}

# Configuración de datos
DATA_CONFIG = {
    # Características técnicas/indicadores que se espera recibir de NinjaTrader 8
    'features': [
        'open', 'high', 'low', 'close', 'volume',
        'sma_20', 'sma_50', 'sma_200',
        'ema_9', 'ema_21',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'atr_14', 'bollinger_upper', 'bollinger_lower',
        'adx_14', 'stoch_k', 'stoch_d',
        'vwap', 'momentum_14'
    ],
    # Características derivadas calculadas internamente
    'derived_features': [
        'returns_1', 'returns_5', 'returns_10',
        'high_low_ratio', 'close_open_ratio'
    ],
    # Número de barras históricas a mantener como contexto
    'lookback_window': 20,
    # Características que se normalizarán
    'normalize_features': True,
    # Método de normalización ('standard', 'minmax', 'robust')
    'normalization_method': 'robust',
    # Qué hacer con valores NaN
    'fill_na_method': 'bfill',
    # Punto decimal de los precios
    'price_precision': 2
}

# Configuración del entorno - AJUSTES MÁS ESTRICTOS ANTI-HIPERTRADING
ENV_CONFIG = {
    # Parámetros generales
    'initial_balance': 50000.0,
    'commission_rate': 0.0001,
    'contract_size': 1,
    'max_position': 1,
    'episode_length': 1000,
    'reward_scaling': 1.5,

    # Gestión de riesgo
    'max_loss_pct': 0.05,
    'max_drawdown_pct': 0.10,
    'risk_per_trade_pct': 0.005,

    # Nuevos parámetros para el sistema de recompensas
    'pnl_scale': 3.0,
    'position_hold_threshold': 80,
    'take_profit_ratio': 1.0,
    'atr_base': 20.0,
    'initial_exploration_steps': 300,
    'force_action_prob': 0.25,
    'log_reward_components': True,

    # Parámetros de inactividad
    'inactivity_threshold': 50,
    'trivial_trade_threshold': 10.0,

    # Factores de normalización
    'position_normalization': 1.0,
    'price_normalization': 10000.0,
    'balance_normalization': 50000.0,

    # Parámetros de observación
    'use_market_features': True,
    'use_account_features': True,
    'use_position_features': True,
    'continuous_position_size': True,

    # Parámetros de configuración técnicos
    'observation_type': 'tensor',
    'action_space_type': 'continuous',

    # Nuevos parámetros para compatibilidad con TradingEnv
    'window_size': 60,
    'features': 28,

    # Parámetros para tamaño mínimo y TP/SL - ACTUALIZADOS PARA USAR TICKS EN LUGAR DE PORCENTAJES
    'min_sl_ticks': 50,          # Establecido a 50 ticks mínimo para stop loss
    'min_tp_ticks': 50,          # Establecido a 50 ticks mínimo para take profit
    'tick_size': 0.25,           # Tamaño del tick para NQ (0.25 puntos por tick)
    'enforce_min_trade_size': True,  # Activado: obligatorio respetar tamaños mínimos de SL/TP
    'reward_larger_trades': True,    # Activado: premiar operaciones con mayor tamaño/riesgo

    # NUEVO: Parámetros para trailing stop dinámico
    'enable_dynamic_trailing': True,  # Permite que el agente decida activar trailing stops
    'trailing_stop_distance_ticks': 20,  # Distancia en ticks para trailing stop (más cerca que el SL)
    'trailing_activation_threshold': 0.3,  # Umbral de ganancias para permitir trailing (% del precio)
    'breakeven_activation_threshold': 0.15,  # Umbral de ganancias para mover a break-even (% del precio)
    'reward_for_good_sl_adjustment': 0.5,  # Recompensa por ajustes acertados de SL
    'enable_sl_tp_management': True,  # Habilitar segunda dimensión del espacio de acción

    # Parámetros para equilibrio frecuencia/duración - VALORES MÁS PERMISIVOS
    'min_hold_steps': 3,         # Reducido drásticamente para pruebas
    'position_cooldown': 5,      # Reducido drásticamente para pruebas
    'force_min_hold': False,     # Desactivado para permitir más flexibilidad
    'short_trade_penalty_factor': 1.0,   # Reducido para no penalizar operaciones cortas
    'duration_scaling_factor': 0.5,    # Reducido significativamente
    'position_change_threshold': 0.2,  # Muy permisivo para facilitar cambios

    # Parámetros para equilibrar ticks
    'log_ticks': True,
    'positive_ticks_reward': 0.010, 
    'negative_ticks_penalty': 0.015, # Reducido de 0.020 a 0.015
    'asymmetric_reward_ratio': 1.5, # Reducido de 2.0 a 1.5
    'tp_ratio_reward_factor': 1.5, # Reducido de 2.0 a 1.5
    'sl_ratio_reward_factor': 0.5,

    # Parámetros para distancia
    'tp_distance_factor': 1.0,
    'sl_distance_factor': 1.0,

    # Anti-sobretrading moderado
    'overtrade_penalty': -0.5,       # Reducido aún más para pruebas
    'hold_reward': 0.05,             # Reducido para no incentivar tanto la inactividad
    'reward_delay_steps': 1,         # Mínimo retraso posible
}

# Configuración de recompensa - SIMPLIFICADA
REWARD_CONFIG = {
    'base_reward': -0.001,
    'pnl_weight': 2.0,                
    'risk_weight': 0.5,               # Reducido de 1.0 a 0.5
    'drawdown_weight': 0.3,           # Reducido de 0.6 a 0.3
    'profit_factor_weight': 0.5,      # Reducido de 0.8 a 0.5
    'win_rate_weight': 0.5,           # Reducido de 0.8 a 0.5
    'inactivity_weight': 0.1,         # Reducido de 0.2 a 0.1
    'trade_completion_bonus': 0.5,    
    'direction_change_bonus': 0.0,    
    'diversification_weight': 0.0,    
    'trade_frequency_target': 0.2,    # Aumentado de 0.1 a 0.2
    'scale_factor': 2.0,              # Reducido de 5.0 a 2.0
}

# Configuración del agente PPO (sin cambios respecto a v1.3.8)
PPO_CONFIG = {
    'learning_rate': 0.0001,
    'n_steps': 4096,
    'batch_size': 128,
    'n_epochs': 4,
    'gamma': 0.98,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'clip_range_vf': 0.2,
    'normalize_advantage': True,
    'ent_coef': 0.25,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'target_kl': None,
    'activation_fn': 'tanh',
    'net_arch': [
        {
            'pi': [256, 128, 64],
            'vf': [256, 128, 64]
        }
    ],
    'device': 'auto',
    'exploration_config': {
        'exploration_steps': 300000,
        'exploration_prob': 0.2,
        'exploration_decay': 0.9995,
        'inactivity_threshold': 50,
    }
}

# Alias para compatibilidad
AGENT_CONFIG = PPO_CONFIG

# Configuración de entrenamiento (sin cambios respecto a v1.3.8)
TRAINING_CONFIG = {
    # Parámetros generales
    'total_timesteps': 2000000,
    'log_freq': 10000,
    'eval_freq': 10000,
    'n_eval_episodes': 5,
    'checkpoint_freq': 25000,
    'deterministic_eval': False,
    'early_stopping_patience': 25,

    # Curriculum learning
    'use_curriculum': True,
    'progressive_steps': [250000, 750000, 1500000],
    'curriculum_parameters': {
        'inactivity_threshold': [80, 50, 30],
        'risk_aversion': [0.1, 0.3, 0.5],
        'penalty_factor': [0.5, 1.0, 1.5],
    },

    # Validación cruzada
    'use_cross_validation': True,
    'cv_segments': 5,
    'train_test_split': 0.8,

    # Callbacks y opciones adicionales
    'use_early_stopping': True,
    'save_replay_buffer': False,
    'prioritized_replay': False,
    'verbose': 1
}

# Configuración de visualización (sin cambios)
VISUALIZATION_CONFIG = {
    # Opciones de gráficos
    'plot_learning_curve': True,
    'plot_reward_components': True,
    'plot_equity_curve': True,
    'plot_position_history': True,
    'plot_drawdown': True,

    # Opciones para gráficos de trading
    'chart_width': 1200,
    'chart_height': 800,
    'chart_theme': 'dark',
    'save_charts': True,
    'interactive_charts': False,

    # Métricas a visualizar
    'show_metrics': [
        'win_rate', 'profit_factor', 'sharpe_ratio', 'sortino_ratio',
        'max_drawdown', 'avg_trade_pnl', 'total_trades',
        'market_exposure_pct', 'avg_position_duration'
    ]
}

# Configuración de logging (sin cambios)
LOGGING_CONFIG = {
    # Niveles de detalle
    'level': 'INFO',
    'log_level': 'INFO',
    'console_level': 'INFO',
    'file_level': 'INFO',
    'log_to_file': True,
    'log_to_console': True,
    'log_trades': True,
    'log_portfolio': True,
    'log_hyperparams': True,

    # Métricas detalladas para diagnóstico
    'log_reward_components': True,
    'log_network_weights': False,
    'log_gradients': False,

    # Opciones de rendimiento
    'log_system_stats': True,
    'log_frequency': 10000,
    'tensorboard': True,
    'wandb': False
}

# Combine all configurations
CONFIG = {
    'base': BASE_CONFIG,
    'data': DATA_CONFIG,
    'environment': ENV_CONFIG,
    'reward': REWARD_CONFIG,
    'ppo': PPO_CONFIG,
    'training': TRAINING_CONFIG,
    'visualization': VISUALIZATION_CONFIG,
    'logging': LOGGING_CONFIG
}

def get_config():
    """
    Get the complete configuration.

    Returns:
        dict: Complete configuration
    """
    return CONFIG

def update_config(config_updates):
    """
    Update the configuration with new values.

    Args:
        config_updates (dict): Dictionary with configuration updates

    Returns:
        dict: Updated configuration
    """
    for section, updates in config_updates.items():
        if section in CONFIG:
            CONFIG[section].update(updates)

    return CONFIG
