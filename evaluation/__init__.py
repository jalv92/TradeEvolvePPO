"""
Evaluation module initialization file.
Contains backtesting and metrics functionality.
"""

from .backtest import Backtester
from .metrics import calculate_metrics

__all__ = ['Backtester', 'calculate_metrics']