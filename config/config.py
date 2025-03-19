"""
Configuration module for TradeEvolvePPO.
"""

# Base configuration
BASE_CONFIG = {
    'version': '0.1.0',
    'seed': 42,
    'log_level': 'INFO',
    'save_dir': 'models',
    'results_dir': 'results',
    'tensorboard_dir': 'logs',
}

# Data configuration
DATA_CONFIG = {
    'date_column': ['datetime'],
    'index_column': 0,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'handle_missing': True,
    'missing_strategy': 'ffill',
    'normalize': True,
    'normalize_columns': [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd_line', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower',
        'ema_fast', 'ema_slow', 'atr'
    ],
    # Lista de indicadores que se esperan recibir de NinjaTrader 8
    'indicators': [
        'rsi',                  # RSI (Relative Strength Index)
        'macd_line',            # MACD (Moving Average Convergence Divergence) - Línea principal
        'macd_signal',          # MACD - Línea de señal
        'macd_histogram',       # MACD - Histograma
        'bb_upper',             # Bandas de Bollinger - Superior
        'bb_middle',            # Bandas de Bollinger - Media
        'bb_lower',             # Bandas de Bollinger - Inferior
        'ema_fast',             # EMA (Exponential Moving Average) rápida
        'ema_slow',             # EMA lenta
        'atr',                  # ATR (Average True Range)
        # Indicadores opcionales que podrían incluirse
        'stoch_k',              # Estocástico %K
        'stoch_d',              # Estocástico %D
        'adx',                  # ADX (Average Directional Index)
        'pos_di',               # +DI (Positive Directional Indicator)
        'neg_di',               # -DI (Negative Directional Indicator)
        'vwap',                 # VWAP (Volume Weighted Average Price)
        'obv'                   # OBV (On-Balance Volume)
    ]
}

# Environment configuration
ENV_CONFIG = {
    'id': 'TradingEnv-v0',
    'window_size': 60,
    'max_steps': 20000,
    'commission': 0.001,
    'initial_balance': 10000,
    'reward_scaling': 1.0,
    'position_penalty': 0.0001,
    'position_reward': 0.0001,
    'trade_reward': 0.0,
    'done_on_bankruptcy': True,
    'bankruptcy_threshold': 0.2,
    'features': [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd_line', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower',
        'ema_fast', 'ema_slow', 'atr'
    ],
    'actions': ['hold', 'buy', 'sell'],
    'position_history': 5
}

# Agent configuration (PPO)
AGENT_CONFIG = {
    'policy': 'MlpPolicy',
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'clip_range_vf': None,
    'normalize_advantage': True,
    'ent_coef': 0.0,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'use_sde': False,
    'sde_sample_freq': -1,
    'target_kl': None,
    'verbose': 1
}

# Training configuration
TRAINING_CONFIG = {
    'total_timesteps': 1000000,
    'eval_freq': 50000,
    'save_freq': 100000,
    'log_freq': 10000,
    'n_eval_episodes': 10,
    'early_stopping_patience': 5,
    'early_stopping_threshold': 0.01,
    'model_checkpoint': True,
    'best_model_save': True
}

# Evaluation configuration
EVAL_CONFIG = {
    'n_episodes': 10,
    'render': False,
    'deterministic': True,
    'metrics': [
        'total_return', 'sharpe_ratio', 'max_drawdown',
        'win_rate', 'profit_factor', 'expectancy'
    ]
}

# Combine all configurations
CONFIG = {
    'base': BASE_CONFIG,
    'data': DATA_CONFIG,
    'env': ENV_CONFIG,
    'agent': AGENT_CONFIG,
    'training': TRAINING_CONFIG,
    'eval': EVAL_CONFIG
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