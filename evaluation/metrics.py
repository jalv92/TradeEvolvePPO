"""
Metrics module for TradeEvolvePPO.
Calculates trading performance metrics from backtest results.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union


def calculate_sharpe_ratio(returns: Union[List[float], np.ndarray], 
                          risk_free_rate: float = 0.0,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio of returns.
    
    Args:
        returns (Union[List[float], np.ndarray]): Daily returns
        risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.
        periods_per_year (int, optional): Trading periods per year. Defaults to 252.
        
    Returns:
        float: Sharpe ratio
    """
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if len(excess_returns) < 2:
        return 0.0
    
    std = np.std(excess_returns, ddof=1)
    if std == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / std * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_sortino_ratio(returns: Union[List[float], np.ndarray],
                           risk_free_rate: float = 0.0,
                           periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio of returns.
    
    Args:
        returns (Union[List[float], np.ndarray]): Daily returns
        risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.
        periods_per_year (int, optional): Trading periods per year. Defaults to 252.
        
    Returns:
        float: Sortino ratio
    """
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if len(excess_returns) < 2:
        return 0.0
    
    # Calculate downside deviation (standard deviation of negative returns only)
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0
    
    downside_std = np.std(negative_returns, ddof=1)
    if downside_std == 0:
        return 0.0
    
    sortino = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
    return float(sortino)


def calculate_max_drawdown(equity_curve: Union[List[float], np.ndarray]) -> float:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve (Union[List[float], np.ndarray]): Equity curve
        
    Returns:
        float: Maximum drawdown as a decimal (not percentage)
    """
    equity_curve = np.array(equity_curve)
    
    if len(equity_curve) < 2:
        return 0.0
    
    # Calculate the running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown
    drawdowns = (running_max - equity_curve) / running_max
    
    # Return the maximum drawdown
    return float(np.max(drawdowns))


def calculate_win_rate(trades_df: pd.DataFrame) -> float:
    """
    Calculate win rate from trades DataFrame.
    
    Args:
        trades_df (pd.DataFrame): DataFrame with trade information
        
    Returns:
        float: Win rate as a decimal (not percentage)
    """
    if len(trades_df) == 0:
        return 0.0
    
    # Count winning trades
    winning_trades = trades_df[trades_df['pnl'] > 0]
    win_rate = len(winning_trades) / len(trades_df)
    
    return float(win_rate)


def calculate_profit_factor(trades_df: pd.DataFrame) -> float:
    """
    Calculate profit factor from trades DataFrame.
    
    Args:
        trades_df (pd.DataFrame): DataFrame with trade information
        
    Returns:
        float: Profit factor
    """
    if len(trades_df) == 0:
        return 0.0
    
    # Calculate gross profit and gross loss
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    profit_factor = gross_profit / gross_loss
    
    return float(profit_factor)


def calculate_average_trade(trades_df: pd.DataFrame) -> float:
    """
    Calculate average trade P&L from trades DataFrame.
    
    Args:
        trades_df (pd.DataFrame): DataFrame with trade information
        
    Returns:
        float: Average trade P&L
    """
    if len(trades_df) == 0:
        return 0.0
    
    avg_trade = trades_df['pnl'].mean()
    
    return float(avg_trade)


def calculate_annual_return(performance_df: pd.DataFrame, 
                           periods_per_year: int = 252) -> float:
    """
    Calculate annualized return from performance DataFrame.
    
    Args:
        performance_df (pd.DataFrame): DataFrame with performance information
        periods_per_year (int, optional): Trading periods per year. Defaults to 252.
        
    Returns:
        float: Annualized return as a decimal (not percentage)
    """
    if len(performance_df) < 2:
        return 0.0
    
    # Get initial and final equity
    initial_equity = performance_df['net_worth'].iloc[0]
    final_equity = performance_df['net_worth'].iloc[-1]
    
    # Calculate total return
    total_return = final_equity / initial_equity - 1
    
    # Calculate number of years
    n_periods = len(performance_df)
    n_years = n_periods / periods_per_year
    
    # Calculate annualized return
    annual_return = (1 + total_return) ** (1 / n_years) - 1
    
    return float(annual_return)


def calculate_calmar_ratio(performance_df: pd.DataFrame, 
                          periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).
    
    Args:
        performance_df (pd.DataFrame): DataFrame with performance information
        periods_per_year (int, optional): Trading periods per year. Defaults to 252.
        
    Returns:
        float: Calmar ratio
    """
    if len(performance_df) < 2:
        return 0.0
    
    # Calculate annual return
    annual_return = calculate_annual_return(performance_df, periods_per_year)
    
    # Calculate max drawdown
    max_dd = calculate_max_drawdown(performance_df['net_worth'].values)
    
    if max_dd == 0:
        return float('inf') if annual_return > 0 else 0.0
    
    # Calculate Calmar ratio
    calmar = annual_return / max_dd
    
    return float(calmar)


def calculate_metrics(trades_df: pd.DataFrame, 
                     performance_df: pd.DataFrame,
                     config: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate trading performance metrics.
    
    Args:
        trades_df (pd.DataFrame): DataFrame with trade information
        performance_df (pd.DataFrame): DataFrame with performance information
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Dict[str, float]: Dictionary with calculated metrics
    """
    # Extract configuration
    risk_free_rate = config.get('env_config', {}).get('risk_free_rate', 0.02)
    periods_per_year = 252  # Default for daily trading
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Return 0 for all metrics if no trades or performance data
    if len(trades_df) == 0 or len(performance_df) == 0:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_trade_pnl': 0.0,
            'calmar_ratio': 0.0,
            'total_trades': 0
        }
    
    # Calculate daily returns from performance data if available
    if 'return' in performance_df.columns:
        daily_returns = performance_df['return'].diff().dropna().values
    else:
        daily_returns = np.diff(performance_df['net_worth'].values) / performance_df['net_worth'].values[:-1]
    
    # Calculate total return
    initial_equity = performance_df['net_worth'].iloc[0]
    final_equity = performance_df['net_worth'].iloc[-1]
    total_return = final_equity / initial_equity - 1
    
    # Calculate metrics
    metrics['total_return'] = float(total_return)
    metrics['annual_return'] = calculate_annual_return(performance_df, periods_per_year)
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(daily_returns, risk_free_rate, periods_per_year)
    metrics['sortino_ratio'] = calculate_sortino_ratio(daily_returns, risk_free_rate, periods_per_year)
    metrics['max_drawdown'] = calculate_max_drawdown(performance_df['net_worth'].values)
    metrics['win_rate'] = calculate_win_rate(trades_df)
    metrics['profit_factor'] = calculate_profit_factor(trades_df)
    metrics['avg_trade_pnl'] = calculate_average_trade(trades_df)
    metrics['calmar_ratio'] = calculate_calmar_ratio(performance_df, periods_per_year)
    metrics['total_trades'] = len(trades_df)
    
    # Add drawdown percentage for clarity
    metrics['max_drawdown_pct'] = metrics['max_drawdown'] * 100
    
    # Calculate average trade duration if entry and exit steps are available
    if 'entry_step' in trades_df.columns and 'exit_step' in trades_df.columns:
        avg_duration = (trades_df['exit_step'] - trades_df['entry_step']).mean()
        metrics['avg_trade_duration'] = float(avg_duration)
    
    # Calculate win/loss ratio
    win_trades = trades_df[trades_df['pnl'] > 0]
    loss_trades = trades_df[trades_df['pnl'] < 0]
    
    if len(win_trades) > 0 and len(loss_trades) > 0:
        avg_win = win_trades['pnl'].mean()
        avg_loss = abs(loss_trades['pnl'].mean())
        
        if avg_loss != 0:
            metrics['win_loss_ratio'] = float(avg_win / avg_loss)
        else:
            metrics['win_loss_ratio'] = float('inf')
    else:
        metrics['win_loss_ratio'] = 0.0
    
    return metrics