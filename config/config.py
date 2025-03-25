"""
Configuración global para el proyecto TradeEvolvePPO.
Define los valores por defecto para el entorno, agente y entrenamiento.
"""

import torch

# Configuración base
BASE_CONFIG = {
    'symbol': 'NQ',
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

# Configuración del entorno
ENV_CONFIG = {
    # Parámetros generales
    'initial_balance': 50000.0,  # Saldo inicial en USD
    'commission_rate': 0.0001,   # Comisión por operación (0.01%)
    'contract_size': 1,          # Tamaño del contrato (siempre 1 para simplificar)
    'max_position': 1,           # Posición máxima (long: +1, short: -1)
    'episode_length': 1000,      # Duración máxima del episodio
    'reward_scaling': 1.5,       # Factor de escala para la recompensa
    
    # Gestión de riesgo
    'max_loss_pct': 0.05,        # Pérdida máxima permitida (% del balance)
    'max_drawdown_pct': 0.10,    # Drawdown máximo permitido (% del balance)
    'risk_per_trade_pct': 0.005, # Riesgo por operación (% del balance) (reducido de 0.01)
    
    # Nuevos parámetros para el sistema de recompensas
    'pnl_scale': 5.0,            # Escala para la recompensa de PnL (reducido de 10.0)
    'position_hold_threshold': 50, # Umbral de pasos para penalizar posiciones mantenidas demasiado tiempo
    'stop_loss_pct': 0.01,       # Porcentaje de stop loss (reducido de 0.02)
    'take_profit_ratio': 2.0,    # Ratio take profit en relación al stop loss (aumentado de 1.5)
    'atr_base': 20.0,            # Valor base para normalizar ATR
    'initial_exploration_steps': 300, # Pasos iniciales con alta probabilidad de exploración
    'force_action_prob': 0.95,   # Aumentado de 0.9 a 0.95
    'log_reward_components': True, # Registrar componentes de recompensa para análisis
    
    # Parámetros de inactividad
    'inactivity_threshold': 20,  # Reducido de 100 a 20
    'trivial_trade_threshold': 5.0, # Reducido de 10.0 a 5.0
    
    # Factores de normalización
    'position_normalization': 1.0,
    'price_normalization': 10000.0,
    'balance_normalization': 50000.0,
    
    # Parámetros de observación
    'use_market_features': True,
    'use_account_features': True,
    'use_position_features': True,
    'continuous_position_size': True, # True = posición como valor continuo, False = one-hot
    
    # Parámetros de configuración técnicos
    'observation_type': 'tensor',  # 'tensor' o 'dict'
    'action_space_type': 'continuous',  # 'discrete' o 'continuous'
    
    # Nuevos parámetros para compatibilidad con TradingEnv
    'window_size': 60,           # Tamaño de la ventana de observación
    'features': 25               # Número exacto de características
}

# Configuración de recompensa - COMPLETAMENTE REEMPLAZADO POR EL NUEVO SISTEMA
REWARD_CONFIG = {
    'base_reward': -0.002,           # Reducido de -0.01 a -0.002
    'pnl_weight': 2.5,               # Aumentado de 1.0 a 2.5
    'risk_weight': 0.1,              # Reducido de 0.2 a 0.1
    'drawdown_weight': 0.01,         # Significativamente reducido de 0.05 a 0.01
    'profit_factor_weight': 0.35,    # Sin cambios
    'win_rate_weight': 0.5,          # Sin cambios
    'inactivity_weight': 2.0,        # Aumentado de 0.5 a 2.0
    'trade_completion_bonus': 8.0,   # Significativamente aumentado de 3.0 a 8.0
    'direction_change_bonus': 0.5,   # Aumentado de 0.2 a 0.5
    'diversification_weight': 0.4,   # Sin cambios
    'trade_frequency_target': 0.65,  # Sin cambios
    'scale_factor': 5.0,             # Aumentado de 2.0 a 5.0
}

# Configuración del agente PPO - OPTIMIZADA PARA TRADING
AGENT_CONFIG = {
    'learning_rate': 0.0003,          # Aumentado de 0.0001 a 0.0003
    'n_steps': 8192,                  # Sin cambios
    'batch_size': 1024,               # Sin cambios
    'n_epochs': 10,                   # Sin cambios
    'gamma': 0.95,                    # Reducido de 0.99 a 0.95 para priorizar recompensas a corto plazo
    'gae_lambda': 0.95,               # Sin cambios
    'clip_range': 0.2,                # Sin cambios
    'clip_range_vf': 0.2,             # Sin cambios
    'normalize_advantage': True,      # Sin cambios
    'ent_coef': 0.3,                  # Significativamente aumentado de 0.1 a 0.3
    'vf_coef': 0.5,                   # Sin cambios
    'max_grad_norm': 0.5,             # Sin cambios
    'target_kl': 0.015,               # Sin cambios
    'activation_fn': 'tanh',          # Sin cambios
    'net_arch': [
        {
            'pi': [512, 256, 128],    # Arquitectura de política
            'vf': [512, 256, 128]     # Arquitectura de función de valor
        }
    ],
    'features_extractor': 'AttentionFeaturesExtractor',  # Extractor de características
    'features_extractor_kwargs': {
        'features_dim': 128,          # Dimensión de características
        'num_attention_heads': 4,     # Número de cabezas de atención
        'attention_dropout': 0.1,     # Dropout de atención
    },
    'exploration_config': {
        'exploration_steps': 1000000,  # Aumentado de 500000 a 1000000
        'exploration_prob': 0.5,      # Aumentado de 0.3 a 0.5
        'inactivity_threshold': 20,   # Reducido de 50 a 20
    }
}

# Alias para compatibilidad
PPO_CONFIG = AGENT_CONFIG

# Configuración de entrenamiento
TRAINING_CONFIG = {
    # Parámetros generales
    'total_timesteps': 2000000,        # AUMENTADO DE 800K A 2M DE PASOS
    'log_freq': 5000,                 # Frecuencia de logging
    'eval_freq': 20000,               # Frecuencia de evaluación
    'n_eval_episodes': 5,             # Episodios de evaluación
    'checkpoint_freq': 40000,         # Frecuencia de guardado
    'deterministic_eval': True,       # Evaluación determinista
    'early_stopping_patience': 30,    # AUMENTADO DE 20 A 30
    
    # Curriculum learning (aprendizaje progresivo) - MÁS AGRESIVO (REQUERIMIENTO #4-4)
    'use_curriculum': True,           # Usar curriculum learning
    'progressive_steps': [200000, 600000, 1200000, 1800000], # AJUSTADOS PARA 2M PASOS
    'curriculum_parameters': {        # Parámetros que cambiarán progresivamente
        'inactivity_threshold': [50, 40, 30, 20],  # REDUCIDOS (antes [100, 80, 60, 40])
        'risk_aversion': [0.1, 0.2, 0.3, 0.4],      # Comenzar con mucha menos aversión
        'trivial_trade_threshold': [2, 5, 10, 15],  # Valores más bajos para incentivar operaciones
        'penalty_factor': [0.3, 0.4, 0.6, 0.8],     # Menor penalización inicial
        'max_drawdown': [0.40, 0.35, 0.25, 0.20]    # Permitir mayor drawdown inicial
    },
    
    # Validación cruzada
    'use_cross_validation': True,     # Usar validación cruzada
    'cv_segments': 5,                 # Número de segmentos para validación
    'train_test_split': 0.8,          # Proporción de datos para entrenamiento
    
    # Callbacks y opciones adicionales
    'use_early_stopping': True,       # Usar early stopping
    'save_replay_buffer': False,      # Guardar buffer de replay
    'prioritized_replay': False,      # Usar replay priorizado
    'verbose': 1                      # Nivel de detalle
}

# Configuración de visualización
VISUALIZATION_CONFIG = {
    # Opciones de gráficos
    'plot_learning_curve': True,     # Graficar curva de aprendizaje
    'plot_reward_components': True,  # Graficar componentes de recompensa
    'plot_equity_curve': True,       # Graficar curva de equity
    'plot_position_history': True,   # Graficar historia de posiciones
    'plot_drawdown': True,           # Graficar drawdown
    
    # Opciones para gráficos de trading
    'chart_width': 1200,             # Ancho del gráfico
    'chart_height': 800,             # Altura del gráfico
    'chart_theme': 'dark',           # Tema del gráfico ('light', 'dark')
    'save_charts': True,             # Guardar gráficos
    'interactive_charts': False,     # Gráficos interactivos
    
    # Métricas a visualizar
    'show_metrics': [
        'win_rate', 'profit_factor', 'sharpe_ratio', 'sortino_ratio',
        'max_drawdown', 'avg_trade_pnl', 'total_trades',
        'market_exposure_pct', 'avg_position_duration'
    ]
}

# Configuración de logging
LOGGING_CONFIG = {
    # Niveles de detalle
    'level': 'WARNING',               # Nivel general de logging cambiado a WARNING
    'log_level': 'WARNING',           # Nivel de log para compatibilidad con main.py
    'console_level': 'ERROR',         # Nivel para consola cambiado a ERROR (solo errores críticos)
    'file_level': 'INFO',             # Mantener nivel INFO para archivos (para diagnóstico posterior)
    'log_to_file': True,              # Guardar logs en archivo
    'log_to_console': False,          # Desactivar logs en consola para reducir ruido
    'log_trades': True,               # Mantener registro de operaciones
    'log_portfolio': True,            # Mantener cambios en cartera
    'log_hyperparams': False,         # Desactivar log de hiperparámetros
    
    # Métricas detalladas para diagnóstico
    'log_reward_components': False,   # Desactivar registro detallado de componentes de recompensa
    'log_network_weights': False,     # Desactivar registro de pesos de la red
    'log_gradients': False,           # Desactivar registro de gradientes
    
    # Opciones de rendimiento
    'log_system_stats': False,        # Desactivar estadísticas del sistema
    'log_frequency': 20000,           # Aumentar frecuencia de logging a cada 20000 pasos
    'tensorboard': True,              # Mantener TensorBoard
    'wandb': False                    # Mantener Weights & Biases desactivado
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
