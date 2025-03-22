"""
Metrics module for TradeEvolvePPO.
Calculates trading performance metrics from backtest results.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import matplotlib.pyplot as plt
from collections import defaultdict


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


class TradingMetrics:
    """Calcular y trackear métricas de diagnóstico y evaluación para trading."""
    
    def __init__(self, initial_balance: float = 50000.0):
        """
        Inicializar calculador de métricas.
        
        Args:
            initial_balance (float): Balance inicial para cálculos porcentuales
        """
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        """Resetear todas las métricas de seguimiento."""
        # Métricas generales
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.returns = []
        self.drawdowns = []
        
        # Métricas de diagnóstico
        self.market_exposure = []  # % tiempo en mercado
        self.position_durations = []  # Duración de posiciones
        self.trade_sizes = []  # Tamaño de operaciones (PnL absoluto)
        self.trade_sizes_pct = []  # Tamaño de operaciones (% del capital)
        self.small_trades_count = 0  # Número de operaciones pequeñas
        self.normal_trades_count = 0  # Número de operaciones normales
        self.direction_bias = 0.0  # Sesgo direccional (positivo = largo, negativo = corto)
        
        # Metricas de gestión de riesgo
        self.sl_distances = []  # Distancia de stop loss en ticks
        self.tp_distances = []  # Distancia de take profit en ticks
        self.rr_ratios = []  # Ratios riesgo/recompensa
        self.realized_rr_ratios = []  # Ratios R:R realizados
    
    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Registrar una operación de trading para análisis.
        
        Args:
            trade_data (Dict): Datos de la operación
                Debe contener: entry_price, exit_price, position, pnl,
                               entry_step, exit_step, reason, etc.
        """
        self.trades.append(trade_data)
        
        # Calcular y actualizar métricas
        pnl = trade_data.get('pnl', 0.0)
        pnl_pct = pnl / self.initial_balance * 100
        
        # Duración de la posición
        duration = trade_data.get('exit_step', 0) - trade_data.get('entry_step', 0)
        self.position_durations.append(duration)
        
        # Tamaño de la operación
        self.trade_sizes.append(abs(pnl))
        self.trade_sizes_pct.append(abs(pnl_pct))
        
        # Clasificar tamaño de operación
        if abs(pnl) < 10.0:  # Operación trivial/pequeña
            self.small_trades_count += 1
        else:
            self.normal_trades_count += 1
        
        # Actualizar sesgo direccional
        position = trade_data.get('position', 0)
        self.direction_bias += (1 if position > 0 else -1)
        
        # Gestión de riesgo
        if 'stop_loss' in trade_data and 'take_profit' in trade_data:
            entry_price = trade_data.get('entry_price', 0)
            sl_price = trade_data.get('stop_loss', 0)
            tp_price = trade_data.get('take_profit', 0)
            
            # Calcular distancias en ticks (asumiendo 0.25 por tick en NQ)
            tick_size = 0.25  # Para NQ
            sl_distance = abs(entry_price - sl_price) / tick_size
            tp_distance = abs(entry_price - tp_price) / tick_size
            
            self.sl_distances.append(sl_distance)
            self.tp_distances.append(tp_distance)
            
            # Calcular ratio R:R
            if sl_distance > 0:
                rr_ratio = tp_distance / sl_distance
                self.rr_ratios.append(rr_ratio)
            
            # Ratio R:R realizado
            exit_price = trade_data.get('exit_price', 0)
            actual_move = abs(exit_price - entry_price) / tick_size
            if sl_distance > 0:
                realized_rr = actual_move / sl_distance if pnl > 0 else -actual_move / sl_distance
                self.realized_rr_ratios.append(realized_rr)
    
    def log_step(self, step_data: Dict[str, Any]) -> None:
        """
        Registrar datos de un paso para seguimiento.
        
        Args:
            step_data (Dict): Datos del paso
                Debe contener: balance, position, etc.
        """
        # Actualizar equity curve
        equity = step_data.get('equity', self.equity_curve[-1] if self.equity_curve else self.initial_balance)
        self.equity_curve.append(equity)
        
        # Calcular retorno
        if len(self.equity_curve) > 1:
            returns = self.equity_curve[-1] / self.equity_curve[-2] - 1
            self.returns.append(returns)
        
        # Calcular drawdown
        peak = max(self.equity_curve)
        current = self.equity_curve[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0
        self.drawdowns.append(drawdown)
        
        # Actualizar exposición a mercado
        position = step_data.get('position', 0)
        self.market_exposure.append(1 if position != 0 else 0)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas resumidas de todas las métricas.
        
        Returns:
            Dict: Métricas resumidas
        """
        stats = {}
        
        # Métricas de trading básicas
        stats['total_trades'] = len(self.trades)
        
        if self.trades:
            # Win rate
            winners = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
            stats['win_rate'] = (winners / len(self.trades)) * 100
            
            # Profit factor
            gross_profit = sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0)
            gross_loss = abs(sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0))
            stats['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Avg trade PnL
            stats['avg_trade_pnl'] = sum(t.get('pnl', 0) for t in self.trades) / len(self.trades)
        else:
            stats['win_rate'] = 0
            stats['profit_factor'] = 0
            stats['avg_trade_pnl'] = 0
        
        # Métricas de retorno/riesgo
        if self.equity_curve:
            stats['total_return'] = (self.equity_curve[-1] / self.initial_balance - 1) * 100
            stats['max_drawdown'] = max(self.drawdowns) * 100 if self.drawdowns else 0
        else:
            stats['total_return'] = 0
            stats['max_drawdown'] = 0
        
        # Calcular ratios riesgo/retorno
        if len(self.returns) > 1:
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns) + 1e-9  # Evitar división por cero
            
            # Sharpe ratio
            stats['sharpe_ratio'] = mean_return / std_return * np.sqrt(252)  # Anualizado
            
            # Sortino ratio (solo downside risk)
            downside_returns = [r for r in self.returns if r < 0]
            downside_std = np.std(downside_returns) if downside_returns else 1e-9
            stats['sortino_ratio'] = mean_return / downside_std * np.sqrt(252)
        else:
            stats['sharpe_ratio'] = 0
            stats['sortino_ratio'] = 0
        
        # Métricas de diagnóstico avanzadas
        
        # % Tiempo en mercado
        stats['market_exposure_pct'] = np.mean(self.market_exposure) * 100 if self.market_exposure else 0
        
        # Duración media de posiciones
        stats['avg_position_duration'] = np.mean(self.position_durations) if self.position_durations else 0
        
        # Ratio trades pequeños vs normales
        stats['small_trades_pct'] = (self.small_trades_count / len(self.trades) * 100) if self.trades else 0
        
        # Sesgo direccional (normalizado entre -1 y 1)
        total_trades = len(self.trades)
        stats['direction_bias'] = self.direction_bias / total_trades if total_trades > 0 else 0
        
        # Gestión de riesgo
        stats['avg_sl_distance'] = np.mean(self.sl_distances) if self.sl_distances else 0
        stats['avg_tp_distance'] = np.mean(self.tp_distances) if self.tp_distances else 0
        stats['avg_rr_ratio'] = np.mean(self.rr_ratios) if self.rr_ratios else 0
        stats['avg_realized_rr'] = np.mean(self.realized_rr_ratios) if self.realized_rr_ratios else 0
        
        return stats
    
    def plot_metrics(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generar y guardar gráficos de las métricas principales.
        
        Args:
            save_path (str, optional): Ruta para guardar los gráficos. Si None, solo devuelve figuras.
            
        Returns:
            Dict: Diccionario de figuras matplotlib
        """
        figures = {}
        
        # Crear figura para equity curve
        if len(self.equity_curve) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.equity_curve, label='Equity Curve')
            ax.set_title('Equity Curve')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Equity ($)')
            ax.grid(True)
            ax.legend()
            
            if save_path:
                fig.savefig(f"{save_path}/equity_curve.png", dpi=300)
            
            figures['equity_curve'] = fig
        
        # Crear figura para drawdowns
        if self.drawdowns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.drawdowns, label='Drawdown', color='red')
            ax.set_title('Drawdown')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Drawdown (%)')
            ax.grid(True)
            ax.legend()
            
            if save_path:
                fig.savefig(f"{save_path}/drawdown.png", dpi=300)
            
            figures['drawdown'] = fig
        
        # Distribución de tamaños de operaciones
        if self.trade_sizes:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(self.trade_sizes, bins=20, alpha=0.7, label='Trade Size ($)')
            ax.set_title('Trade Size Distribution')
            ax.set_xlabel('Trade Size ($)')
            ax.set_ylabel('Frequency')
            ax.grid(True)
            ax.legend()
            
            if save_path:
                fig.savefig(f"{save_path}/trade_size_dist.png", dpi=300)
            
            figures['trade_size_dist'] = fig
        
        # Distribución de R:R ratios
        if self.rr_ratios:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(self.rr_ratios, bins=20, alpha=0.7, label='Risk:Reward Ratio')
            ax.set_title('Risk:Reward Ratio Distribution')
            ax.set_xlabel('R:R Ratio')
            ax.set_ylabel('Frequency')
            ax.axvline(x=1.0, color='red', linestyle='--', label='1:1 R:R')
            ax.axvline(x=2.0, color='green', linestyle='--', label='2:1 R:R')
            ax.grid(True)
            ax.legend()
            
            if save_path:
                fig.savefig(f"{save_path}/rr_ratio_dist.png", dpi=300)
            
            figures['rr_ratio_dist'] = fig
        
        # Evolución de distancias SL/TP
        if self.sl_distances and self.tp_distances:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = range(len(self.sl_distances))
            ax.plot(x, self.sl_distances, label='Stop Loss (ticks)', color='red')
            ax.plot(x, self.tp_distances, label='Take Profit (ticks)', color='green')
            ax.set_title('SL/TP Distance Evolution')
            ax.set_xlabel('Trade #')
            ax.set_ylabel('Distance (ticks)')
            ax.grid(True)
            ax.legend()
            
            if save_path:
                fig.savefig(f"{save_path}/sltp_evolution.png", dpi=300)
            
            figures['sltp_evolution'] = fig
        
        return figures
    
    def get_debug_dataframe(self) -> pd.DataFrame:
        """
        Convertir todos los trades a DataFrame para depuración y análisis.
        
        Returns:
            pd.DataFrame: DataFrame con todas las operaciones y métricas
        """
        if not self.trades:
            return pd.DataFrame()
        
        # Convertir lista de trades a DataFrame
        df = pd.DataFrame(self.trades)
        
        # Añadir métricas adicionales
        if 'entry_step' in df.columns and 'exit_step' in df.columns:
            df['duration'] = df['exit_step'] - df['entry_step']
        
        if 'entry_price' in df.columns and 'stop_loss' in df.columns and 'take_profit' in df.columns:
            tick_size = 0.25  # Para NQ
            
            # Calcular distancias y ratios
            df['sl_distance'] = abs(df['entry_price'] - df['stop_loss']) / tick_size
            df['tp_distance'] = abs(df['entry_price'] - df['take_profit']) / tick_size
            
            # Evitar división por cero
            df['rr_ratio'] = df.apply(
                lambda row: row['tp_distance'] / row['sl_distance'] if row['sl_distance'] > 0 else 0, 
                axis=1
            )
            
            # Calcular ratio R:R realizado
            df['realized_move'] = abs(df['exit_price'] - df['entry_price']) / tick_size
            df['realized_rr'] = df.apply(
                lambda row: (row['realized_move'] / row['sl_distance'] if row['pnl'] > 0 else -row['realized_move'] / row['sl_distance']) 
                           if row['sl_distance'] > 0 else 0,
                axis=1
            )
        
        return df


def calculate_trade_metrics(trade_history: List[Dict], initial_balance: float = 50000.0) -> Dict[str, Any]:
    """
    Calcular métricas de trading a partir del historial de operaciones.
    
    Args:
        trade_history (List[Dict]): Lista de operaciones realizadas
        initial_balance (float): Balance inicial
        
    Returns:
        Dict[str, Any]: Diccionario con métricas calculadas
    """
    metrics = {}
    
    # Métricas básicas
    total_trades = len(trade_history)
    metrics['total_trades'] = total_trades
    
    if total_trades == 0:
        # No hay trades, devolver valores por defecto
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_trade_pnl': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'market_exposure_pct': 0
        }
    
    # Win rate y profit factor
    winning_trades = [t for t in trade_history if t.get('pnl', 0) > 0]
    losing_trades = [t for t in trade_history if t.get('pnl', 0) < 0]
    
    metrics['win_rate'] = (len(winning_trades) / total_trades) * 100
    
    gross_profit = sum(t.get('pnl', 0) for t in winning_trades)
    gross_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
    
    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    metrics['avg_trade_pnl'] = sum(t.get('pnl', 0) for t in trade_history) / total_trades
    
    # Calcular equity curve y retornos
    equity_curve = [initial_balance]
    for trade in sorted(trade_history, key=lambda x: x.get('exit_step', 0)):
        equity_curve.append(equity_curve[-1] + trade.get('pnl', 0))
    
    # Calcular retorno total
    metrics['total_return'] = (equity_curve[-1] / initial_balance - 1) * 100
    
    # Calcular drawdown máximo
    peak = initial_balance
    drawdowns = []
    
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        
        drawdown = (peak - equity) / peak if peak > 0 else 0
        drawdowns.append(drawdown)
    
    metrics['max_drawdown'] = max(drawdowns) * 100
    
    # Calcular exposición al mercado
    if trade_history and 'entry_step' in trade_history[0] and 'exit_step' in trade_history[0]:
        total_steps = max(t.get('exit_step', 0) for t in trade_history)
        in_market_steps = 0
        
        # Contar pasos en mercado
        market_exposure = np.zeros(total_steps + 1)
        
        for trade in trade_history:
            entry = trade.get('entry_step', 0)
            exit = trade.get('exit_step', 0)
            
            if entry < exit:
                market_exposure[entry:exit] = 1
        
        in_market_steps = np.sum(market_exposure)
        metrics['market_exposure_pct'] = (in_market_steps / total_steps) * 100
    else:
        metrics['market_exposure_pct'] = 0
    
    # Análisis de gestión de riesgos
    if 'stop_loss' in trade_history[0] and 'take_profit' in trade_history[0]:
        tick_size = 0.25  # Para NQ
        
        sl_distances = []
        tp_distances = []
        rr_ratios = []
        
        for trade in trade_history:
            entry = trade.get('entry_price', 0)
            sl = trade.get('stop_loss', 0)
            tp = trade.get('take_profit', 0)
            
            sl_distance = abs(entry - sl) / tick_size
            tp_distance = abs(entry - tp) / tick_size
            
            sl_distances.append(sl_distance)
            tp_distances.append(tp_distance)
            
            if sl_distance > 0:
                rr_ratios.append(tp_distance / sl_distance)
        
        metrics['avg_sl_distance'] = np.mean(sl_distances)
        metrics['avg_tp_distance'] = np.mean(tp_distances)
        metrics['avg_rr_ratio'] = np.mean(rr_ratios) if rr_ratios else 0
    
    return metrics