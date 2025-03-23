# TradeEvolvePPO 📈🤖

![Version](https://img.shields.io/badge/Version-0.1.56-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

**TradeEvolvePPO** is an advanced algorithmic trading system powered by reinforcement learning, specifically using Proximal Policy Optimization (PPO). The system evolves trading strategies through experience, optimizing for profitability while managing risk.

<p align="center">
  <img src="./Cerebro2.jpg"
  alt="TradeEvolvePPO Architecture" width="80%">
</p>

## 🌟 Features

- **AI-Powered Trading**: Leverages reinforcement learning with PPO to develop adaptive trading strategies
- **Comprehensive Backtesting**: Robust evaluation with detailed performance metrics and visualization
- **NinjaTrader 8 Integration**: Real-time trading capabilities through direct connection to NinjaTrader 8
- **Risk Management**: Built-in risk controls including stop-loss, take-profit, and maximum drawdown limits
- **Performance Analytics**: Extensive metrics calculation including Sharpe ratio, Sortino ratio, win rate, and more
- **Visualization Tools**: Graphical representation of equity curves, drawdowns, and trade distributions

## 📋 Table of Contents

- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
  - [Training a Model](#training-a-model)
  - [Testing/Backtesting](#testingbacktesting)
  - [Live Trading](#live-trading)
- [NinjaTrader 8 Integration](#-ninjatrader-8-integration)
- [Configuration](#-configuration)
- [Example Results](#-example-results)
- [Contact](#-contact)

## 💻 System Requirements

- Python 3.7 or higher
- NinjaTrader 8 (for live trading)
- CUDA-compatible GPU (recommended for faster training)
- Windows operating system (for NinjaTrader integration)

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/TradeEvolvePPO.git
   cd TradeEvolvePPO
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup NinjaTrader 8 integration** (for live trading):
   - Copy `connector_nt8/NT8StrategyServer.cs` to your NinjaTrader 8 strategies folder
   - In NinjaTrader 8, import the strategy via Tools > Import > NinjaScript Add-On
   - Add the strategy to a chart of the instrument you want to trade

## 📂 Project Structure

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
│   ├── ppo_agent.py          # PPO agent implementation
│   └── lstm_policy.py        # LSTM architecture for PPO
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
├── connector_nt8/
│   ├── __init__.py
│   ├── ninjatrader_client.py # Python client for NinjaTrader
│   ├── nt8_trader.py         # Live trading implementation
│   └── NT8StrategyServer.cs  # NinjaTrader strategy (server)
└── utils/
    ├── __init__.py
    ├── logger.py             # Logging functionality
    └── helpers.py            # Helper functions
```

## 🚀 Usage

### Training a Model

To train a new model on historical data:

```bash
python main.py train --data path/to/price_data.csv --output ./results/my_model
```

Additional options:
- `--config path/to/custom_config.py`: Use a custom configuration
- `--timesteps 500000`: Set the number of training timesteps
- `--device cuda`: Specify computation device (cpu or cuda)

### Testing/Backtesting

To backtest a trained model on historical data:

```bash
python main.py backtest --model ./results/my_model/models/final_model.zip --data path/to/test_data.csv --output ./results/backtest_results
```

For a quick test on validation data:

```bash
python main.py test --model ./results/my_model/models/final_model.zip --data path/to/validation_data.csv --output ./results/test_results
```

### Live Trading

To use a trained model for live trading with NinjaTrader 8:

```bash
python -m connector_nt8.nt8_trader --model ./results/my_model/models/final_model.zip --instrument "NQ 06-25" --quantity 1
```

Parameters:
- `--instrument`: The instrument to trade (format must match NinjaTrader)
- `--host`: IP address of NinjaTrader 8 (default: localhost)
- `--port`: TCP port for NT8StrategyServer (default: 5555)
- `--quantity`: Contract size for trades (default: 1)
- `--interval`: Update interval in seconds (default: 60)
- `--min-bars`: Minimum bars required before trading (default: 20)

## 🔌 NinjaTrader 8 Integration

TradeEvolvePPO connects to NinjaTrader 8 through a custom strategy that acts as a TCP server:

1. **Server**: NT8StrategyServer runs in NinjaTrader 8 and provides:
   - Real-time price data streaming
   - Order execution capabilities
   - Position and order updates

2. **Client**: NT8Client in Python connects to the server to:
   - Receive market data
   - Send trading signals from the AI model
   - Monitor positions and orders

### NT8StrategyServer Configuration Parameters:

- **TCP Port**: Port for communication (default: 5555)
- **Send bar data**: Enable/disable OHLCV data streaming
- **Send order updates**: Send notifications about order changes
- **Send position updates**: Send notifications about position changes
- **Allow remote orders**: Permit Python client to send orders
- **Update interval**: How frequently to send data updates

## ⚙️ Configuration

TradeEvolvePPO uses a modular configuration system in `config/config.py` with these key sections:

```python
# Key configuration sections:
BASE_CONFIG      # Basic settings (symbol, timeframe, etc.)
DATA_CONFIG      # Data preprocessing settings
ENV_CONFIG       # Trading environment parameters
AGENT_CONFIG     # PPO model architecture and hyperparameters
TRAINING_CONFIG  # Training process settings
REWARD_CONFIG    # Reward function configuration
VISUALIZATION_CONFIG  # Plotting and visualization options
LOGGING_CONFIG   # Logging settings
```

### Main Environment Parameters:

```python
# Example environment configuration
ENV_CONFIG = {
    'initial_balance': 50000.0,  # Starting capital
    'commission_rate': 0.0001,   # Trading commission (0.01%)
    'max_position': 1,           # Maximum position size
    'stop_loss_pct': 0.02,       # Stop loss percentage (2%)
    'take_profit_ratio': 1.5,    # TP ratio (1.5x the stop distance)
    'max_drawdown_pct': 0.15,    # Maximum allowed drawdown
}
```

### Agent Configuration:

```python
# Example agent configuration
AGENT_CONFIG = {
    'policy': 'LSTMPolicy',          # MlpPolicy or LSTMPolicy
    'learning_rate': 0.0005,         # Learning rate for optimizer
    'n_steps': 256,                  # Steps to collect before updating
    'batch_size': 64,                # Minibatch size for updates
    'n_epochs': 10,                  # Training epochs per update
    'gamma': 0.95,                   # Discount factor
    'gae_lambda': 0.9,               # GAE parameter
    'clip_range': 0.2,               # PPO clipping parameter
    'ent_coef': 0.1,                 # Entropy coefficient
    'vf_coef': 0.5,                  # Value function coefficient
    'lstm_hidden_size': 256,         # LSTM hidden layer size
    'num_lstm_layers': 2,            # Number of LSTM layers
    'lstm_bidirectional': False,     # Bidirectional LSTM
}
```

## 📊 Example Results

Here are some example backtest results from the system:

<p align="center">
  <img src="https://raw.githubusercontent.com/yourusername/TradeEvolvePPO/main/assets/equity_curve.png" alt="Equity Curve" width="80%">
</p>

Sample performance metrics:
- Total Return: 32.45%
- Sharpe Ratio: 1.87
- Sortino Ratio: 2.43
- Max Drawdown: 8.32%
- Win Rate: 62.8%
- Profit Factor: 1.94

## 📝 Data Format

The system expects price data in CSV format with at least the following columns:
- `timestamp` or `date`: Date and time of the bar
- `open`: Opening price
- `high`: Highest price in the period
- `low`: Lowest price in the period
- `close`: Closing price
- `volume`: Volume traded

Additional indicator columns can be included and specified in the configuration.

## 🔧 Advanced Usage

### Progressive Training

You can enable progressive training with curriculum learning:

```python
# In config.py
TRAINING_CONFIG = {
    'use_curriculum': True,
    'progressive_steps': [80000, 160000, 240000, 320000],
    'curriculum_parameters': {
        'inactivity_threshold': [100, 80, 60, 40],
        'risk_aversion': [0.1, 0.2, 0.3, 0.4],
        'trivial_trade_threshold': [2, 5, 10, 15],
        'penalty_factor': [0.3, 0.4, 0.6, 0.8],
        'max_drawdown': [0.40, 0.35, 0.25, 0.20]
    }
}
```

### Custom Reward Functions

Customize the reward function to focus on specific trading aspects:

```python
# In config.py
REWARD_CONFIG = {
    'pnl_weight': 1.0,              # Weight for PnL-based reward
    'action_reward': 0.1,           # Fixed reward for taking actions
    'risk_management_reward': 0.5,  # Reward for proper risk management
    'inactivity_penalty': -0.05,    # Penalty for prolonged inactivity
    'excessive_hold_penalty': -0.01  # Penalty for excessive risk
}
```

## 📞 Contact

If you need help, have questions, or would like to contribute to this project, please contact:

jvlora@hublai.com

---

<p align="center">
  <b>TradeEvolvePPO</b> - Evolving Trading Intelligence Through Reinforcement Learning
</p>