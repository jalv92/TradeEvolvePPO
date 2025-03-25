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
    'inactivity_threshold': 100, # Umbral de pasos para penalizar inactividad
    'position_hold_threshold': 50, # Umbral de pasos para penalizar posiciones mantenidas demasiado tiempo
    'stop_loss_pct': 0.01,       # Porcentaje de stop loss (reducido de 0.02)
    'take_profit_ratio': 2.0,    # Ratio take profit en relación al stop loss (aumentado de 1.5)
    'atr_base': 20.0,            # Valor base para normalizar ATR
    'initial_exploration_steps': 300, # Pasos iniciales con alta probabilidad de exploración
    'force_action_prob': 0.9,    # Probabilidad de forzar acciones durante exploración
    'log_reward_components': True, # Registrar componentes de recompensa para análisis
    
    # Parámetros de inactividad
    'inactivity_threshold': 100,  # Pasos de inactividad antes de penalización (actualizado de 30 a 100)
    'trivial_trade_threshold': 10.0, # Umbral para considerar una operación como trivial ($)
    
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
    'action_space_type': 'continuous',  # 'discrete' o 'continuous' - ACTUALIZADO A CONTINUOUS
    
    # Nuevos parámetros para compatibilidad con TradingEnv
    'window_size': 60,           # Tamaño de la ventana de observación
    'features': 25               # Número exacto de características
}

# Configuración de recompensa - COMPLETAMENTE REEMPLAZADO POR EL NUEVO SISTEMA
REWARD_CONFIG = {
    'base_reward': -0.01,             # Penalización base por cada paso (reducida de -0.05)
    'pnl_weight': 1.0,                # Peso para PnL (reducido de 3.0)
    'risk_weight': 0.2,               # Peso para riesgo
    'drawdown_weight': 0.05,          # Peso para drawdown
    'profit_factor_weight': 0.35,     # Peso para factor de beneficio
    'win_rate_weight': 0.5,           # Peso para tasa de victorias
    'inactivity_weight': 0.5,         # Peso para penalización por inactividad (reducido de 2.0)
    'trade_completion_bonus': 3.0,    # Bonus por completar una operación (reducido de 5.0)
    'direction_change_bonus': 0.2,    # Bonus por cambiar dirección de trading
    'diversification_weight': 0.4,    # Peso para diversificación de instrumentos
    'trade_frequency_target': 0.65,   # Objetivo de frecuencia de trading (% del tiempo en mercado)
    'scale_factor': 2.0,              # Factor de escala para la recompensa final (reducido de 10.0)
}

# Configuración del agente PPO - OPTIMIZADA PARA TRADING
AGENT_CONFIG = {
    'learning_rate': 0.0001,          # Tasa de aprendizaje del modelo (reducida de 0.001)
    'n_steps': 8192,                  # Pasos por actualización (reducido de 16384)
    'batch_size': 1024,               # Tamaño del lote (reducido de 2048)
    'n_epochs': 10,                   # Épocas por actualización
    'gamma': 0.99,                    # Factor de descuento (aumentado de 0.90)
    'gae_lambda': 0.95,               # Lambda para GAE
    'clip_range': 0.2,                # Rango de recorte para PPO
    'clip_range_vf': 0.2,             # Rango de recorte para función de valor
    'normalize_advantage': True,      # Normalizar ventaja
    'ent_coef': 0.1,                  # Coeficiente de entropía (reducido de 0.5)
    'vf_coef': 0.5,                   # Coeficiente de función de valor
    'max_grad_norm': 0.5,             # Norma de gradiente máxima
    'target_kl': 0.015,               # KL divergencia objetivo
    'activation_fn': 'tanh',          # Función de activación
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
        'exploration_steps': 500000,  # Pasos de exploración forzada (reducido de 1000000)
        'exploration_prob': 0.3,      # Probabilidad de exploración (reducida de 0.4)
        'inactivity_threshold': 50,   # Umbral de inactividad
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
