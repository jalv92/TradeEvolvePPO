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
    'risk_per_trade_pct': 0.01,  # Riesgo por operación (% del balance)
    
    # Nuevos parámetros para el sistema de recompensas
    'pnl_scale': 10.0,           # Escala para la recompensa de PnL
    'inactivity_threshold': 100, # Umbral de pasos para penalizar inactividad
    'position_hold_threshold': 50, # Umbral de pasos para penalizar posiciones mantenidas demasiado tiempo
    'stop_loss_pct': 0.02,       # Porcentaje de stop loss (2% del balance)
    'take_profit_ratio': 1.5,    # Ratio take profit en relación al stop loss
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
    'action_space_type': 'continuous'  # 'discrete' o 'continuous' - ACTUALIZADO A CONTINUOUS
}

# Configuración de recompensa - COMPLETAMENTE REEMPLAZADO POR EL NUEVO SISTEMA
REWARD_CONFIG = {
    # Pesos para componentes de recompensa
    'pnl_weight': 1.0,              # Peso para recompensa basada en PnL
    'action_reward': 0.1,           # Recompensa fija por tomar acción (+0.1)
    'risk_management_reward': 0.5,  # Recompensa por cerrar con stop-loss/take-profit (+0.5)
    'inactivity_penalty': -0.05,    # Penalización por inactividad prolongada (-0.05)
    'excessive_hold_penalty': -0.01, # Factor para penalización por riesgo excesivo (-0.01 * pasos)
    
    # Umbrales
    'inactivity_threshold': 100,    # Umbral de pasos para penalizar inactividad 
    'position_hold_threshold': 50,  # Umbral de pasos para penalizar posiciones mantenidas
    
    # Configuración de volatilidad
    'atr_base': 20.0,               # Valor base para normalizar ATR
    'volatility_max_factor': 2.0,   # Factor máximo de amplificación por volatilidad
    
    # Configuración de entrenamiento forzado
    'initial_exploration_steps': 300, # Pasos iniciales con acciones forzadas
    'force_action_prob': 0.9,        # Probabilidad de forzar acciones en exploración inicial
    
    # Opciones generales
    'log_reward_components': True   # Registrar componentes de la recompensa
}

# Configuración del agente PPO - ACTUALIZADA PARA MEJOR EXPLORACIÓN
AGENT_CONFIG = {
    # RL Algorithm Parameters
    'algorithm': 'PPO',
    'policy': 'MlpPolicy',  # 'MlpPolicy' o 'LstmPolicy'
    'learning_rate': 0.0005,  # Reducido para prevenir explosión de gradientes
    'gamma': 0.95,  # Reducido para priorizar recompensas inmediatas
    'gae_lambda': 0.9,  # Reducido
    'clip_range': 0.2,  # Valor más conservador
    'ent_coef': 0.1,  # Reducido pero aún alto para exploración
    'vf_coef': 0.5,  # Aumentado para mejor estimación de valores
    'max_grad_norm': 0.5,  # Clip más agresivo para prevenir explosión
    
    # Training Parameters
    'n_steps': 256,  # Aumentado ligeramente
    'batch_size': 64,  # Aumentado para mejor estabilidad
    'n_epochs': 10,  # Valor moderado
    
    # Neural Network Parameters
    'net_arch': {
        'pi': [128, 128],  # Red más robusta
        'vf': [128, 128]
    },
    'activation_fn': 'tanh',  # Cambio a tanh para prevenir explosión
    
    # LSTM Parameters (if using LstmPolicy)
    'lstm_hidden_size': 128,
    'num_lstm_layers': 1,
    'bidirectional': False,
    
    # SDE (Exploratory noise)
    'use_sde': True,  # Mantener para exploración
    'sde_sample_freq': 8,  # Menos frecuente
    'log_std_init': 0.0,  # Valor mucho más conservador
    
    # Estabilidad y prevención de NaN
    'normalize_advantage': True,  # Normalizar ventajas
    'target_kl': 0.015,  # Limitar divergencia de KL
    'use_state_normalization': True,  # Normalizar estados
    
    # Otros
    'buffer_size': 10000,
    'verbose': 1
}

# Alias para compatibilidad
PPO_CONFIG = AGENT_CONFIG

# Configuración de entrenamiento
TRAINING_CONFIG = {
    # Parámetros generales
    'total_timesteps': 800000,        # Aumentado de 400k a 800k pasos
    'log_freq': 5000,                 # Frecuencia de logging
    'eval_freq': 20000,               # Frecuencia de evaluación
    'n_eval_episodes': 5,             # Episodios de evaluación
    'checkpoint_freq': 40000,         # Frecuencia de guardado
    'deterministic_eval': True,       # Evaluación determinista
    'early_stopping_patience': 20,    # Aumentado de 10 a 20
    
    # Curriculum learning (aprendizaje progresivo) - MÁS AGRESIVO (REQUERIMIENTO #4-4)
    'use_curriculum': True,           # Usar curriculum learning
    'progressive_steps': [80000, 160000, 240000, 320000], # Ajustados para 400k pasos
    'curriculum_parameters': {        # Parámetros que cambiarán progresivamente
        'inactivity_threshold': [100, 80, 60, 40],  # Valores más altos inicialmente (antes [50, 40, 30, 25])
        'risk_aversion': [0.1, 0.2, 0.3, 0.4],      # Comenzar con mucha menos aversión (antes [0.3, 0.4, 0.5, 0.6])
        'trivial_trade_threshold': [2, 5, 10, 15],  # Valores más bajos para incentivar operaciones (antes [5, 10, 15, 20])
        'penalty_factor': [0.3, 0.4, 0.6, 0.8],     # Menor penalización inicial (antes [0.5, 0.6, 0.7, 0.8])
        'max_drawdown': [0.40, 0.35, 0.25, 0.20]    # Permitir mayor drawdown inicial (antes [0.30, 0.25, 0.20, 0.15])
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
    'level': 'INFO',               # Nivel general de logging
    'log_level': 'INFO',           # Nivel de log para compatibilidad con main.py
    'console_level': 'WARNING',    # Nivel para consola (solo mensajes importantes)
    'file_level': 'INFO',          # Nivel para archivos (información detallada)
    'log_to_file': True,           # Guardar logs en archivo
    'log_to_console': True,        # Mostrar logs en consola
    'log_trades': True,            # Registrar operaciones en detalle
    'log_portfolio': True,         # Registrar cambios en cartera
    'log_hyperparams': True,       # Registrar hiperparámetros
    
    # Métricas detalladas para diagnóstico
    'log_reward_components': True,  # Registrar componentes de recompensa
    'log_network_weights': False,   # Registrar pesos de la red
    'log_gradients': False,         # Registrar gradientes
    
    # Opciones de rendimiento
    'log_system_stats': True,       # Registrar estadísticas del sistema
    'log_frequency': 5000,          # Frecuencia de logging en pasos
    'tensorboard': True,            # Usar TensorBoard
    'wandb': False                  # Usar Weights & Biases
}

# Combine all configurations
CONFIG = {
    'base': BASE_CONFIG,
    'data': DATA_CONFIG,
    'env': ENV_CONFIG,
    'agent': AGENT_CONFIG,
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