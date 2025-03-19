"""
Visualization module for TradeEvolvePPO.
Provides functions for visualizing trading results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union
from matplotlib.figure import Figure


class Visualizer:
    """
    Visualizer class for creating and saving trading performance plots.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Visualizer.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.visualization_config = config.get('visualization_config', {})
        
        # Set up plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # Store created figures
        self.figures = {}
    
    def plot_equity_curve(self, performance_df: pd.DataFrame) -> Figure:
        """
        Plot equity curve from performance data.
        
        Args:
            performance_df (pd.DataFrame): DataFrame with performance information
            
        Returns:
            Figure: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot net worth
        ax.plot(performance_df['step'], performance_df['net_worth'], label='Net Worth', linewidth=2)
        
        # Add initial balance line for reference
        initial_balance = performance_df['net_worth'].iloc[0]
        ax.axhline(y=initial_balance, color='r', linestyle='--', label='Initial Balance')
        
        # Configure plot
        ax.set_title('Equity Curve', fontsize=16)
        ax.set_xlabel('Steps', fontsize=14)
        ax.set_ylabel('Account Value ($)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.2f}"))
        
        # Add final return as annotation
        final_return = (performance_df['net_worth'].iloc[-1] / initial_balance - 1) * 100
        ax.annotate(f'Return: {final_return:.2f}%', 
                    xy=(0.02, 0.95), 
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Store figure
        self.figures['equity_curve'] = fig
        
        return fig
    
    def plot_drawdown(self, performance_df: pd.DataFrame) -> Figure:
        """
        Plot drawdown from performance data.
        
        Args:
            performance_df (pd.DataFrame): DataFrame with performance information
            
        Returns:
            Figure: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate running maximum
        running_max = performance_df['net_worth'].cummax()
        
        # Calculate drawdown
        drawdown = (running_max - performance_df['net_worth']) / running_max * 100
        
        # Plot drawdown
        ax.fill_between(performance_df['step'], 0, drawdown, color='crimson', alpha=0.3, label='Drawdown')
        ax.plot(performance_df['step'], drawdown, color='crimson', linewidth=1)
        
        # Configure plot
        ax.set_title('Drawdown', fontsize=16)
        ax.set_xlabel('Steps', fontsize=14)
        ax.set_ylabel('Drawdown (%)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Invert y-axis for better visualization
        ax.invert_yaxis()
        
        # Add max drawdown as annotation
        max_drawdown = drawdown.max()
        ax.annotate(f'Max Drawdown: {max_drawdown:.2f}%', 
                    xy=(0.02, 0.05), 
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Store figure
        self.figures['drawdown'] = fig
        
        return fig
    
    def plot_trade_distribution(self, trades_df: pd.DataFrame) -> Figure:
        """
        Plot trade distribution from trades data.
        
        Args:
            trades_df (pd.DataFrame): DataFrame with trade information
            
        Returns:
            Figure: Matplotlib figure object
        """
        if len(trades_df) == 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center', fontsize=14)
            self.figures['trade_distribution'] = fig
            return fig
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot histogram of trade PnL
        sns.histplot(trades_df['pnl'], bins=20, kde=True, ax=ax1, color='skyblue')
        ax1.axvline(x=0, color='r', linestyle='--', label='Break Even')
        ax1.set_title('Trade P&L Distribution', fontsize=16)
        ax1.set_xlabel('P&L ($)', fontsize=14)
        ax1.set_ylabel('Count', fontsize=14)
        ax1.legend()
        
        # Calculate and plot winning vs. losing trades
        win_trades = len(trades_df[trades_df['pnl'] > 0])
        loss_trades = len(trades_df[trades_df['pnl'] < 0])
        even_trades = len(trades_df[trades_df['pnl'] == 0])
        
        labels = ['Winning', 'Losing', 'Break Even']
        counts = [win_trades, loss_trades, even_trades]
        colors = ['green', 'red', 'gray']
        
        ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
                wedgeprops={'edgecolor': 'w', 'linewidth': 1.5})
        ax2.set_title('Trade Outcome Distribution', fontsize=16)
        
        # Add text with trade counts
        textstr = f"Total Trades: {len(trades_df)}\n"
        textstr += f"Winning Trades: {win_trades}\n"
        textstr += f"Losing Trades: {loss_trades}\n"
        textstr += f"Break Even Trades: {even_trades}"
        
        ax2.text(1.05, 0.5, textstr, transform=ax2.transAxes, fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        
        # Store figure
        self.figures['trade_distribution'] = fig
        
        return fig
    
    def plot_returns_distribution(self, performance_df: pd.DataFrame) -> Figure:
        """
        Plot distribution of returns from performance data.
        
        Args:
            performance_df (pd.DataFrame): DataFrame with performance information
            
        Returns:
            Figure: Matplotlib figure object
        """
        # Calculate daily returns
        if 'return' in performance_df.columns:
            returns = performance_df['return'].diff().dropna()
        else:
            returns = performance_df['net_worth'].pct_change().dropna()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot returns distribution
        sns.histplot(returns * 100, bins=50, kde=True, ax=ax, color='skyblue')
        ax.axvline(x=0, color='r', linestyle='--', label='Zero Return')
        
        # Configure plot
        ax.set_title('Returns Distribution', fontsize=16)
        ax.set_xlabel('Return (%)', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics as annotation
        stats_text = (
            f"Mean: {returns.mean() * 100:.2f}%\n"
            f"Std Dev: {returns.std() * 100:.2f}%\n"
            f"Min: {returns.min() * 100:.2f}%\n"
            f"Max: {returns.max() * 100:.2f}%"
        )
        
        ax.annotate(stats_text, 
                    xy=(0.02, 0.95), 
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    verticalalignment='top')
        
        # Store figure
        self.figures['returns_distribution'] = fig
        
        return fig
    
    def plot_position_over_time(self, performance_df: pd.DataFrame) -> Figure:
        """
        Plot position size over time from performance data.
        
        Args:
            performance_df (pd.DataFrame): DataFrame with performance information
            
        Returns:
            Figure: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot position
        ax.plot(performance_df['step'], performance_df['position'], label='Position Size', color='purple')
        
        # Configure plot
        ax.set_title('Position Size Over Time', fontsize=16)
        ax.set_xlabel('Steps', fontsize=14)
        ax.set_ylabel('Position Size', fontsize=14)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Store figure
        self.figures['position_over_time'] = fig
        
        return fig
    
    def plot_trade_metrics(self, trades_df: pd.DataFrame) -> Figure:
        """
        Plot various trade metrics from trades data.
        
        Args:
            trades_df (pd.DataFrame): DataFrame with trade information
            
        Returns:
            Figure: Matplotlib figure object
        """
        if len(trades_df) == 0:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center', fontsize=14)
            self.figures['trade_metrics'] = fig
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate cumulative P&L
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        
        # Plot 1: Cumulative P&L over trades
        axes[0, 0].plot(range(len(trades_df)), trades_df['cumulative_pnl'], color='green', linewidth=2)
        axes[0, 0].set_title('Cumulative P&L Over Trades', fontsize=14)
        axes[0, 0].set_xlabel('Trade Number', fontsize=12)
        axes[0, 0].set_ylabel('Cumulative P&L ($)', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Average P&L by exit reason
        if 'reason' in trades_df.columns:
            avg_pnl_by_reason = trades_df.groupby('reason')['pnl'].mean().reset_index()
            sns.barplot(x='reason', y='pnl', data=avg_pnl_by_reason, ax=axes[0, 1], palette='viridis')
            axes[0, 1].set_title('Average P&L by Exit Reason', fontsize=14)
            axes[0, 1].set_xlabel('Exit Reason', fontsize=12)
            axes[0, 1].set_ylabel('Average P&L ($)', fontsize=12)
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, 'No exit reason data available', ha='center', va='center', fontsize=14)
        
        # Plot 3: Trade P&L by type (buy/sell)
        if 'type' in trades_df.columns:
            sns.boxplot(x='type', y='pnl', data=trades_df, ax=axes[1, 0], palette='pastel')
            axes[1, 0].set_title('Trade P&L by Type', fontsize=14)
            axes[1, 0].set_xlabel('Trade Type', fontsize=12)
            axes[1, 0].set_ylabel('P&L ($)', fontsize=12)
        else:
            axes[1, 0].text(0.5, 0.5, 'No trade type data available', ha='center', va='center', fontsize=14)
        
        # Plot 4: Trade duration histogram
        if 'entry_step' in trades_df.columns and 'exit_step' in trades_df.columns:
            trades_df['duration'] = trades_df['exit_step'] - trades_df['entry_step']
            sns.histplot(trades_df['duration'], ax=axes[1, 1], kde=True, color='skyblue')
            axes[1, 1].set_title('Trade Duration Distribution', fontsize=14)
            axes[1, 1].set_xlabel('Duration (steps)', fontsize=12)
            axes[1, 1].set_ylabel('Frequency', fontsize=12)
        else:
            axes[1, 1].text(0.5, 0.5, 'No trade duration data available', ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        
        # Store figure
        self.figures['trade_metrics'] = fig
        
        return fig
    
    def plot_reward_curve(self, rewards: List[float]) -> Figure:
        """
        Plot reward curve from training.
        
        Args:
            rewards (List[float]): List of rewards
            
        Returns:
            Figure: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot rewards
        x = range(len(rewards))
        ax.plot(x, rewards, color='orange', linewidth=1, alpha=0.6)
        
        # Add moving average
        window = min(100, max(10, len(rewards) // 10))
        if window > 0 and len(rewards) > window:
            rewards_series = pd.Series(rewards)
            ma = rewards_series.rolling(window=window).mean()
            ax.plot(x, ma, color='crimson', linewidth=2, label=f'{window}-Step Moving Average')
        
        # Configure plot
        ax.set_title('Training Rewards', fontsize=16)
        ax.set_xlabel('Steps', fontsize=14)
        ax.set_ylabel('Reward', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Store figure
        self.figures['reward_curve'] = fig
        
        return fig
    
    def create_performance_dashboard(self, trades_df: pd.DataFrame, performance_df: pd.DataFrame, metrics: Dict[str, float]) -> Figure:
        """
        Create comprehensive performance dashboard.
        
        Args:
            trades_df (pd.DataFrame): DataFrame with trade information
            performance_df (pd.DataFrame): DataFrame with performance information
            metrics (Dict[str, float]): Dictionary with calculated metrics
            
        Returns:
            Figure: Matplotlib figure object
        """
        # Create a 2x2 grid of plots
        fig = plt.figure(figsize=(20, 16))
        
        # Plot 1: Equity Curve
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax1.plot(performance_df['step'], performance_df['net_worth'], label='Net Worth', linewidth=2)
        initial_balance = performance_df['net_worth'].iloc[0]
        ax1.axhline(y=initial_balance, color='r', linestyle='--', label='Initial Balance')
        ax1.set_title('Equity Curve', fontsize=16)
        ax1.set_xlabel('Steps', fontsize=14)
        ax1.set_ylabel('Account Value ($)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        running_max = performance_df['net_worth'].cummax()
        drawdown = (running_max - performance_df['net_worth']) / running_max * 100
        ax2.fill_between(performance_df['step'], 0, drawdown, color='crimson', alpha=0.3, label='Drawdown')
        ax2.set_title('Drawdown', fontsize=16)
        ax2.set_xlabel('Steps', fontsize=14)
        ax2.set_ylabel('Drawdown (%)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        # Plot 3: Trade Outcomes
        ax3 = plt.subplot2grid((2, 2), (1, 0))
        
        if len(trades_df) > 0:
            win_trades = len(trades_df[trades_df['pnl'] > 0])
            loss_trades = len(trades_df[trades_df['pnl'] < 0])
            even_trades = len(trades_df[trades_df['pnl'] == 0])
            
            labels = ['Winning', 'Losing', 'Break Even']
            counts = [win_trades, loss_trades, even_trades]
            colors = ['green', 'red', 'gray']
            
            ax3.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
                    wedgeprops={'edgecolor': 'w', 'linewidth': 1.5})
            ax3.set_title('Trade Outcome Distribution', fontsize=16)
        else:
            ax3.text(0.5, 0.5, 'No trades to display', ha='center', va='center', fontsize=14)
        
        # Plot 4: Metrics Table
        ax4 = plt.subplot2grid((2, 2), (1, 1))
        ax4.axis('tight')
        ax4.axis('off')
        
        metrics_to_show = [
            ('Total Return', f"{metrics.get('total_return', 0) * 100:.2f}%"),
            ('Annual Return', f"{metrics.get('annual_return', 0) * 100:.2f}%"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
            ('Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"),
            ('Max Drawdown', f"{metrics.get('max_drawdown', 0) * 100:.2f}%"),
            ('Win Rate', f"{metrics.get('win_rate', 0) * 100:.2f}%"),
            ('Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"),
            ('Total Trades', f"{metrics.get('total_trades', 0)}"),
            ('Avg Trade P&L', f"${metrics.get('avg_trade_pnl', 0):.2f}"),
            ('Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.2f}")
        ]
        
        # Create the table
        table = ax4.table(
            cellText=[[v] for _, v in metrics_to_show],
            rowLabels=[k for k, _ in metrics_to_show],
            colWidths=[0.6],
            loc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 2)
        
        ax4.set_title('Performance Metrics', fontsize=16)
        
        plt.tight_layout()
        
        # Store figure
        self.figures['performance_dashboard'] = fig
        
        return fig
    
    def save_plots(self, path: str) -> None:
        """
        Save all created plots to disk.
        
        Args:
            path (str): Directory to save plots
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save each figure
        for name, fig in self.figures.items():
            fig_path = os.path.join(path, f"{name}.png")
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved {name} plot to {fig_path}")