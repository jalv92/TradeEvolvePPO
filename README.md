# TradeEvolvePPO

Sistema de trading algorítmico basado en aprendizaje por refuerzo utilizando el algoritmo PPO (Proximal Policy Optimization).

## Estructura del Proyecto

```
TradeEvolvePPO/
├── README.md                 # Project documentation
├── CHANGELOG.md              # Version history
├── requirements.txt          # Project dependencies
├── main.py                   # Entry point for training and evaluation
├── config/
│   └── config.py             # Configuration parameters
├── data/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading and preprocessing
│   └── indicators.py         # Technical indicators implementation
├── environment/
│   ├── __init__.py
│   └── trading_env.py        # Custom Gym environment
├── agents/
│   ├── __init__.py
│   └── ppo_agent.py          # PPO agent implementation
├── training/
│   ├── __init__.py
│   ├── trainer.py            # Training pipeline
│   └── callback.py           # Custom callbacks for training
├── evaluation/
│   ├── __init__.py
│   ├── backtest.py           # Backtesting functionality
│   └── metrics.py            # Trading performance metrics
├── visualization/
│   ├── __init__.py
│   └── visualizer.py         # Performance visualization
└── utils/
    ├── __init__.py
    ├── logger.py             # Logging functionality
    └── helpers.py            # Helper functions
```

## Versión Actual
v0.1.2
