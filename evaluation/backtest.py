"""
Backtesting module for TradeEvolvePPO.
Evaluates trained agents on historical data.
"""

import os
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple, Union

from agents.ppo_agent import PPOAgent
from evaluation.metrics import calculate_metrics
from visualization.visualizer import Visualizer


class Backtester:
    """
    Backtester for evaluating trading agents.
    Runs a trained agent on historical data and calculates performance metrics.
    """
    
    def __init__(self, env: gym.Env, agent: PPOAgent, config: Dict[str, Any]):
        """
        Initialize the Backtester.
        
        Args:
            env (gym.Env): Trading environment for backtesting
            agent (PPOAgent): Trained agent to evaluate
            config (Dict[str, Any]): Configuration dictionary
        """
        self.env = env
        self.agent = agent
        self.config = config
        
        # Extract configuration
        self.eval_config = config.get('eval_config', {})
        self.visualization_config = config.get('visualization_config', {})
        
        # Results storage
        self.results = None
        self.trades = None
        self.metrics = None
    
    def run(self, n_episodes: int = 1, deterministic: bool = True) -> Dict[str, Any]:
        """
        Run backtest for the specified number of episodes.
        
        Args:
            n_episodes (int, optional): Number of episodes to run. Defaults to 1.
            deterministic (bool, optional): Whether to use deterministic actions. Defaults to True.
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        # Initialize results storage
        all_trades = []
        all_performance = []
        episode_returns = []
        episode_lengths = []
        episode_drawdowns = []
        
        # Run episodes
        for i in range(n_episodes):
            print(f"Running backtest episode {i+1}/{n_episodes}")
            
            # Reset environment
            obs, info = self.env.reset()
            done = False
            truncated = False
            
            # Track episode data
            episode_rewards = []
            steps = 0
            
            # Run episode
            while not done and not truncated:
                # Get action from agent
                action, _ = self.agent.predict(obs, deterministic=deterministic)
                
                # Take step in environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Update tracking
                episode_rewards.append(reward)
                steps += 1
                
                # Update observation
                obs = next_obs
                done = terminated
            
            # Get episode results
            trades = self.env.trade_history
            performance = self.env.performance_history
            
            # Get environment metrics
            env_metrics = self.env.get_performance_summary()
            
            # Record episode stats
            episode_returns.append(sum(episode_rewards))
            episode_lengths.append(steps)
            episode_drawdowns.append(env_metrics['max_drawdown'])
            
            # Add to overall results
            all_trades.extend(trades)
            all_performance.extend(performance)
        
        # Convert to DataFrame
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
        else:
            trades_df = pd.DataFrame(columns=['entry_step', 'entry_price', 'position', 
                                             'exit_step', 'exit_price', 'pnl', 'reason'])
        
        if all_performance:
            performance_df = pd.DataFrame(all_performance)
        else:
            performance_df = pd.DataFrame(columns=['step', 'balance', 'position', 
                                                  'net_worth', 'return', 'drawdown'])
        
        # Calculate overall metrics
        metrics = calculate_metrics(trades_df, performance_df, self.config)
        
        # Store results
        self.results = {
            'trades': trades_df,
            'performance': performance_df,
            'metrics': metrics,
            'episode_returns': episode_returns,
            'episode_lengths': episode_lengths,
            'episode_drawdowns': episode_drawdowns,
            'n_episodes': n_episodes
        }
        
        self.trades = trades_df
        self.metrics = metrics
        
        return self.results
    
    def evaluate(self, n_episodes: int = 1, deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluate agent and generate performance report.
        
        Args:
            n_episodes (int, optional): Number of episodes to run. Defaults to 1.
            deterministic (bool, optional): Whether to use deterministic actions. Defaults to True.
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        # Run backtest
        results = self.run(n_episodes, deterministic)
        
        # Generate visualization if configured
        if self.visualization_config.get('plot_metrics', True):
            try:
                visualizer = Visualizer(self.config)
                
                # Create plots
                if results['trades'].shape[0] > 0:  # Only create plots if there are trades
                    visualizer.plot_equity_curve(results['performance'])
                    visualizer.plot_drawdown(results['performance'])
                    visualizer.plot_trade_distribution(results['trades'])
                    visualizer.plot_returns_distribution(results['performance'])
                    
                    # Save plots if configured
                    if self.visualization_config.get('save_plots', True):
                        plot_path = self.visualization_config.get('plot_path', './plots/')
                        os.makedirs(plot_path, exist_ok=True)
                        visualizer.save_plots(plot_path)
            except Exception as e:
                print(f"Error generating visualizations: {e}")
        
        # Generate performance report
        report = self._generate_report(results)
        
        return report
    
    def _generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate performance report from backtest results.
        
        Args:
            results (Dict[str, Any]): Backtest results
            
        Returns:
            Dict[str, Any]: Performance report
        """
        metrics = results['metrics']
        trades = results['trades']
        
        # Calculate trade statistics
        n_trades = len(trades)
        win_rate = metrics.get('win_rate', 0) * 100
        profit_factor = metrics.get('profit_factor', 0)
        avg_trade = metrics.get('avg_trade_pnl', 0)
        max_drawdown = metrics.get('max_drawdown', 0) * 100
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        sortino_ratio = metrics.get('sortino_ratio', 0)
        total_return = metrics.get('total_return', 0) * 100
        annual_return = metrics.get('annual_return', 0) * 100
        
        # Generate report dictionary
        report = {
            'total_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_pnl': avg_trade,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'total_return': total_return,
            'annual_return': annual_return,
            'episode_avg_return': np.mean(results['episode_returns']),
            'episode_avg_length': np.mean(results['episode_lengths']),
            'episode_avg_drawdown': np.mean(results['episode_drawdowns']) * 100,
        }
        
        # Print report to console
        print("\n===== BACKTEST REPORT =====")
        print(f"Total Trades: {n_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Trade P&L: ${avg_trade:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annual Return: {annual_return:.2f}%")
        print("===========================\n")
        
        return report
    
    def save_results(self, path: str) -> None:
        """
        Save backtest results to files.
        
        Args:
            path (str): Directory to save results
        """
        if self.results is None:
            print("No results to save. Run a backtest first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save trades and performance to CSV
        trades_path = os.path.join(path, 'trades.csv')
        performance_path = os.path.join(path, 'performance.csv')
        metrics_path = os.path.join(path, 'metrics.json')
        
        self.results['trades'].to_csv(trades_path, index=False)
        self.results['performance'].to_csv(performance_path, index=False)
        
        # Save metrics to JSON
        import json
        with open(metrics_path, 'w') as f:
            json.dump(self.results['metrics'], f, indent=4)
        
        print(f"Results saved to {path}")