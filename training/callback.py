"""
Custom callbacks for PPO training.
Implements monitoring and logging callbacks.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
import gymnasium as gym
from typing import Dict, Any, Optional, Union, List, Tuple

from evaluation.metrics import calculate_metrics


class TradeCallback(BaseCallback):
    """
    Custom callback for trading agent training.
    Handles logging, model saving, periodic evaluation, and reward tracking.
    """
    
    def __init__(
        self,
        log_dir: str,
        save_path: str,
        save_interval: int = 10000,
        eval_interval: int = 5000,
        eval_env: Optional[Union[gym.Env, VecEnv]] = None,
        n_eval_episodes: int = 5,
        verbose: int = 1,
        reward_scaling: Optional[float] = None,
        report_progress_interval: int = 1000,
        reward_progress_window: int = 100
    ):
        """
        Initialize the callback.
        
        Args:
            log_dir (str): Directory for logs
            save_path (str): Directory to save models
            save_interval (int, optional): Steps between model savings. Defaults to 10000.
            eval_interval (int, optional): Steps between evaluations. Defaults to 5000.
            eval_env (Optional[Union[gym.Env, VecEnv]], optional): Evaluation environment. Defaults to None.
            n_eval_episodes (int, optional): Number of episodes for evaluation. Defaults to 5.
            verbose (int, optional): Verbosity level. Defaults to 1.
            reward_scaling (Optional[float], optional): Scaling factor for rewards. Defaults to None.
            report_progress_interval (int, optional): Steps between progress reports. Defaults to 1000.
            reward_progress_window (int, optional): Window size for reward smoothing. Defaults to 100.
        """
        super(TradeCallback, self).__init__(verbose)
        
        self.log_dir = log_dir
        self.save_path = save_path
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.reward_scaling = reward_scaling
        self.report_progress_interval = report_progress_interval
        self.reward_progress_window = reward_progress_window
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
        
        # Initialize metrics
        self.best_mean_reward = -np.inf
        self.last_eval_step = 0
        self.start_time = time.time()
        self.training_start_time = None
        
        # Reward tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.reward_history = []
        self.steps_history = []
        self.recent_rewards = []
        
        # Trade metrics tracking
        self.trade_count_history = []
        self.win_rate_history = []
        self.drawdown_history = []
        self.balance_history = []
        
        # Initialize eval callback if eval environment is provided
        if eval_env is not None:
            self.eval_callback = EvalCallback(
                eval_env=eval_env,
                callback_on_new_best=None,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_interval,
                log_path=log_dir,
                best_model_save_path=save_path,
                deterministic=True,
                render=False,
                verbose=verbose
            )
        else:
            self.eval_callback = None
        
        # CSV headers
        self.log_headers = [
            "timestep", "episode", "reward", "reward_mean", "reward_std", 
            "length_mean", "length_std", "wins", "losses", "win_rate",
            "drawdown", "balance", "elapsed_time"
        ]
    
    def _init_callback(self) -> None:
        """Initialize the callback."""
        if self.eval_callback is not None:
            self.eval_callback._init_callback()
        
        # Record start time
        self.training_start_time = time.time()
        
        # Create log files
        self.log_file = open(os.path.join(self.log_dir, 'training_log.csv'), 'w')
        self.log_file.write(','.join(self.log_headers) + '\n')
        
        # Create reward log file
        self.reward_log_file = open(os.path.join(self.log_dir, 'reward_log.csv'), 'w')
        self.reward_log_file.write("timestep,reward\n")
        
        # Create episode log file
        self.episode_log_file = open(os.path.join(self.log_dir, 'episode_log.csv'), 'w')
        self.episode_log_file.write("episode,timestep,reward,length,time\n")
        
        # Initialize episode info
        self.current_episode = 0
        self.episode_start_time = time.time()
        self.episode_start_step = 0
        
        if self.verbose > 0:
            print("Training started. Logs will be saved to:", self.log_dir)
    
    def _on_step(self) -> bool:
        """
        Callback called at each step.
        
        Returns:
            bool: Whether to continue training
        """
        # Save model at specified intervals
        if self.n_calls % self.save_interval == 0:
            self._save_model()
        
        # Track reward for each step
        if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
            if hasattr(self.model.rollout_buffer, 'rewards') and len(self.model.rollout_buffer.rewards) > 0:
                # Get last reward
                last_reward = self.model.rollout_buffer.rewards[-1]
                if isinstance(last_reward, np.ndarray) and len(last_reward) > 0:
                    last_reward = last_reward[-1]
                
                # Store reward
                self.reward_history.append(float(last_reward))
                self.steps_history.append(self.n_calls)
                
                # Log reward
                self.reward_log_file.write(f"{self.n_calls},{last_reward}\n")
                
                # Keep track of recent rewards for smoothing
                self.recent_rewards.append(float(last_reward))
                if len(self.recent_rewards) > self.reward_progress_window:
                    self.recent_rewards.pop(0)
        
        # Log episode information if an episode has ended
        if self._check_episode_end():
            self._log_episode_info()
        
        # Report progress at intervals
        if self.n_calls % self.report_progress_interval == 0:
            self._report_progress()
        
        # Perform evaluation at specified intervals if evaluation environment is provided
        if self.eval_callback is not None and self.n_calls - self.last_eval_step >= self.eval_interval:
            self._perform_evaluation()
        
        # Plot training data periodically
        if self.n_calls % (self.save_interval * 2) == 0:
            self._plot_training_data()
        
        return True
    
    def _on_rollout_end(self) -> None:
        """
        Callback called after a rollout is completed.
        This is where we can update the reward scaling or other parameters.
        """
        # Update any parameters based on training progress
        pass
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Calculate total training time
        total_time = time.time() - self.training_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Print training summary
        if self.verbose > 0:
            print(f"\nTraining completed after {self.n_calls} steps.")
            print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            print(f"Total episodes: {self.current_episode}")
            if len(self.episode_rewards) > 0:
                print(f"Final average reward (last 100 episodes): {np.mean(self.episode_rewards[-100:]):.2f}")
        
        # Close log files
        if hasattr(self, 'log_file'):
            self.log_file.close()
        
        if hasattr(self, 'reward_log_file'):
            self.reward_log_file.close()
        
        if hasattr(self, 'episode_log_file'):
            self.episode_log_file.close()
        
        # Save final model
        final_model_path = os.path.join(self.save_path, 'final_model.zip')
        self.model.save(final_model_path)
        
        # Generate and save final plots
        self._plot_training_data(final=True)
        
        if self.verbose > 0:
            print(f"Final model saved to {final_model_path}")
            print(f"Training logs and plots saved to {self.log_dir}")
    
    def _check_episode_end(self) -> bool:
        """
        Check if an episode has ended.
        
        Returns:
            bool: True if an episode has ended, False otherwise
        """
        if len(self.model.ep_info_buffer) > 0:
            # Check if we have a new episode (buffer size increased)
            if len(self.model.ep_info_buffer) > self.current_episode:
                return True
        return False
    
    def _log_episode_info(self) -> None:
        """Log information about the completed episode."""
        # Get the latest episode info
        ep_idx = min(self.current_episode, len(self.model.ep_info_buffer) - 1)
        
        if ep_idx >= 0 and ep_idx < len(self.model.ep_info_buffer):
            ep_info = self.model.ep_info_buffer[ep_idx]
            
            # Extract episode data
            ep_reward = ep_info.get('r', 0)
            ep_length = ep_info.get('l', 0)
            
            # Calculate episode time
            episode_time = time.time() - self.episode_start_time
            
            # Update episode tracking
            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)
            self.episode_times.append(episode_time)
            
            # Update episode counter
            self.current_episode += 1
            
            # Log to episode file
            self.episode_log_file.write(
                f"{self.current_episode},{self.n_calls},{ep_reward},{ep_length},{episode_time}\n"
            )
            self.episode_log_file.flush()
            
            # Extract trading metrics if available in the info
            if 'win_rate' in ep_info:
                self.win_rate_history.append(ep_info['win_rate'])
            if 'drawdown' in ep_info:
                self.drawdown_history.append(ep_info['drawdown'])
            if 'balance' in ep_info:
                self.balance_history.append(ep_info['balance'])
            if 'total_trades' in ep_info:
                self.trade_count_history.append(ep_info['total_trades'])
            
            # Reset episode start time and step
            self.episode_start_time = time.time()
            self.episode_start_step = self.n_calls
    
    def _report_progress(self) -> None:
        """Report training progress to the console and log file."""
        if len(self.episode_rewards) == 0:
            return
        
        # Calculate statistics
        reward_mean = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
        reward_std = np.std(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.std(self.episode_rewards)
        length_mean = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths)
        length_std = np.std(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.std(self.episode_lengths)
        
        # Calculate win rate and drawdown if available
        win_rate = np.mean(self.win_rate_history[-100:]) if len(self.win_rate_history) >= 100 else (
            np.mean(self.win_rate_history) if len(self.win_rate_history) > 0 else 0
        )
        
        drawdown = np.mean(self.drawdown_history[-100:]) if len(self.drawdown_history) >= 100 else (
            np.mean(self.drawdown_history) if len(self.drawdown_history) > 0 else 0
        )
        
        balance = self.balance_history[-1] if len(self.balance_history) > 0 else 0
        
        # Count wins and losses
        wins = sum(1 for r in self.episode_rewards[-100:] if r > 0) if len(self.episode_rewards) >= 100 else sum(1 for r in self.episode_rewards if r > 0)
        losses = sum(1 for r in self.episode_rewards[-100:] if r <= 0) if len(self.episode_rewards) >= 100 else sum(1 for r in self.episode_rewards if r <= 0)
        
        # Calculate elapsed time
        elapsed_time = time.time() - self.training_start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Print progress
        if self.verbose > 0:
            smoothed_reward = np.mean(self.recent_rewards) if len(self.recent_rewards) > 0 else 0
            
            print(f"\nStep: {self.n_calls} | Episodes: {self.current_episode}")
            print(f"Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            print(f"Reward: {smoothed_reward:.2f} (recent) | {reward_mean:.2f} ± {reward_std:.2f} (100 ep mean ± std)")
            print(f"Episode Length: {length_mean:.1f} ± {length_std:.1f} (100 ep mean ± std)")
            print(f"Win Rate: {win_rate:.2%} | Drawdown: {drawdown:.2%} | Balance: {balance:.2f}")
        
        # Log to CSV
        self.log_file.write(
            f"{self.n_calls},{self.current_episode},{smoothed_reward},{reward_mean},{reward_std},"
            f"{length_mean},{length_std},{wins},{losses},{win_rate},"
            f"{drawdown},{balance},{elapsed_time}\n"
        )
        self.log_file.flush()
    
    def _perform_evaluation(self) -> None:
        """Perform evaluation using the evaluation environment."""
        self.last_eval_step = self.n_calls
        
        # Sync normalization statistics if using VecNormalize
        if hasattr(self.model, 'get_vec_normalize_env') and self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_callback.eval_env)
            except:
                if self.verbose > 0:
                    print("Warning: Could not synchronize environment normalizations")
        
        # Run evaluation
        self.eval_callback._on_step()
        
        # Check if this is the best model so far
        if hasattr(self.eval_callback, 'best_mean_reward'):
            if self.eval_callback.best_mean_reward > self.best_mean_reward:
                self.best_mean_reward = self.eval_callback.best_mean_reward
                
                # Save the best model
                best_model_path = os.path.join(self.save_path, 'best_model.zip')
                self.model.save(best_model_path)
                
                if self.verbose > 0:
                    print(f"New best model saved with mean reward: {self.best_mean_reward:.2f}")
    
    def _save_model(self) -> None:
        """Save the current model checkpoint."""
        checkpoint_path = os.path.join(self.save_path, f'model_{self.n_calls}_steps.zip')
        self.model.save(checkpoint_path)
        
        # Also save current training state to JSON
        training_state = {
            'steps': self.n_calls,
            'episodes': self.current_episode,
            'best_mean_reward': float(self.best_mean_reward),
            'elapsed_time': time.time() - self.training_start_time,
            'reward_mean': float(np.mean(self.episode_rewards[-100:])) if len(self.episode_rewards) >= 100 else float(np.mean(self.episode_rewards)) if len(self.episode_rewards) > 0 else 0,
            'reward_std': float(np.std(self.episode_rewards[-100:])) if len(self.episode_rewards) >= 100 else float(np.std(self.episode_rewards)) if len(self.episode_rewards) > 0 else 0,
            'win_rate': float(np.mean(self.win_rate_history[-100:])) if len(self.win_rate_history) >= 100 else float(np.mean(self.win_rate_history)) if len(self.win_rate_history) > 0 else 0,
        }
        
        state_path = os.path.join(self.save_path, f'training_state_{self.n_calls}_steps.json')
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=4)
        
        if self.verbose > 0:
            print(f"Model checkpoint saved to {checkpoint_path}")
    
    def _plot_training_data(self, final: bool = False) -> None:
        """
        Plot training data and save to file.
        
        Args:
            final (bool, optional): Whether this is the final plot. Defaults to False.
        """
        # Only plot if we have data
        if len(self.episode_rewards) == 0:
            return
        
        # Define plot directory
        plot_dir = os.path.join(self.log_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot reward history
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        
        # Add moving average
        window = min(100, max(5, len(self.episode_rewards) // 10))
        if window > 0:
            rewards_series = pd.Series(self.episode_rewards)
            ma = rewards_series.rolling(window=window).mean()
            plt.plot(range(len(ma)), ma, color='red', linewidth=2, label=f'{window}-Episode Moving Average')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        suffix = '_final' if final else f'_{self.n_calls}'
        plt.savefig(os.path.join(plot_dir, f'reward_history{suffix}.png'), dpi=200, bbox_inches='tight')
        plt.close()
        
        # Plot win rate history if available
        if len(self.win_rate_history) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(self.win_rate_history)), [rate * 100 for rate in self.win_rate_history])
            
            # Add moving average
            window = min(100, max(5, len(self.win_rate_history) // 10))
            if window > 0:
                win_rate_series = pd.Series(self.win_rate_history)
                ma = win_rate_series.rolling(window=window).mean() * 100
                plt.plot(range(len(ma)), ma, color='green', linewidth=2, label=f'{window}-Episode Moving Average')
            
            plt.xlabel('Episode')
            plt.ylabel('Win Rate (%)')
            plt.title('Win Rate During Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plt.savefig(os.path.join(plot_dir, f'win_rate_history{suffix}.png'), dpi=200, bbox_inches='tight')
            plt.close()
        
        # Plot drawdown history if available
        if len(self.drawdown_history) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(self.drawdown_history)), [dd * 100 for dd in self.drawdown_history])
            
            # Add moving average
            window = min(100, max(5, len(self.drawdown_history) // 10))
            if window > 0:
                drawdown_series = pd.Series(self.drawdown_history)
                ma = drawdown_series.rolling(window=window).mean() * 100
                plt.plot(range(len(ma)), ma, color='red', linewidth=2, label=f'{window}-Episode Moving Average')
            
            plt.xlabel('Episode')
            plt.ylabel('Drawdown (%)')
            plt.title('Max Drawdown During Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plt.savefig(os.path.join(plot_dir, f'drawdown_history{suffix}.png'), dpi=200, bbox_inches='tight')
            plt.close()
        
        # Plot balance history if available
        if len(self.balance_history) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(self.balance_history)), self.balance_history)
            
            # Add moving average
            window = min(100, max(5, len(self.balance_history) // 10))
            if window > 0:
                balance_series = pd.Series(self.balance_history)
                ma = balance_series.rolling(window=window).mean()
                plt.plot(range(len(ma)), ma, color='green', linewidth=2, label=f'{window}-Episode Moving Average')
            
            plt.xlabel('Episode')
            plt.ylabel('Balance ($)')
            plt.title('Account Balance During Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plt.savefig(os.path.join(plot_dir, f'balance_history{suffix}.png'), dpi=200, bbox_inches='tight')
            plt.close()
        
        # Create a summary dashboard for the final plot
        if final:
            self._create_summary_dashboard(plot_dir)
    
    def _create_summary_dashboard(self, plot_dir: str) -> None:
        """
        Create a summary dashboard with multiple plots.
        
        Args:
            plot_dir (str): Directory to save the plot
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: Reward history
        axes[0, 0].plot(range(len(self.episode_rewards)), self.episode_rewards, alpha=0.6)
        
        # Add moving average
        window = min(100, max(5, len(self.episode_rewards) // 10))
        if window > 0:
            rewards_series = pd.Series(self.episode_rewards)
            ma = rewards_series.rolling(window=window).mean()
            axes[0, 0].plot(range(len(ma)), ma, color='red', linewidth=2, label=f'{window}-Episode Moving Average')
        
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Win rate history if available
        if len(self.win_rate_history) > 0:
            axes[0, 1].plot(range(len(self.win_rate_history)), [rate * 100 for rate in self.win_rate_history], alpha=0.6)
            
            # Add moving average
            window = min(100, max(5, len(self.win_rate_history) // 10))
            if window > 0:
                win_rate_series = pd.Series(self.win_rate_history)
                ma = win_rate_series.rolling(window=window).mean() * 100
                axes[0, 1].plot(range(len(ma)), ma, color='green', linewidth=2, label=f'{window}-Episode Moving Average')
            
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Win Rate (%)')
            axes[0, 1].set_title('Win Rate')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No win rate data available', ha='center', va='center', fontsize=14)
        
        # Plot 3: Drawdown history if available
        if len(self.drawdown_history) > 0:
            axes[1, 0].plot(range(len(self.drawdown_history)), [dd * 100 for dd in self.drawdown_history], alpha=0.6)
            
            # Add moving average
            window = min(100, max(5, len(self.drawdown_history) // 10))
            if window > 0:
                drawdown_series = pd.Series(self.drawdown_history)
                ma = drawdown_series.rolling(window=window).mean() * 100
                axes[1, 0].plot(range(len(ma)), ma, color='red', linewidth=2, label=f'{window}-Episode Moving Average')
            
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Drawdown (%)')
            axes[1, 0].set_title('Max Drawdown')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No drawdown data available', ha='center', va='center', fontsize=14)
        
        # Plot 4: Balance history if available
        if len(self.balance_history) > 0:
            axes[1, 1].plot(range(len(self.balance_history)), self.balance_history, alpha=0.6)
            
            # Add moving average
            window = min(100, max(5, len(self.balance_history) // 10))
            if window > 0:
                balance_series = pd.Series(self.balance_history)
                ma = balance_series.rolling(window=window).mean()
                axes[1, 1].plot(range(len(ma)), ma, color='green', linewidth=2, label=f'{window}-Episode Moving Average')
            
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Balance ($)')
            axes[1, 1].set_title('Account Balance')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No balance data available', ha='center', va='center', fontsize=14)
        
        # Add summary statistics
        fig.suptitle(f'Training Summary (Episodes: {self.current_episode}, Steps: {self.n_calls})', fontsize=20)
        
        # Calculate elapsed time
        elapsed_time = time.time() - self.training_start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Calculate final statistics
        final_stats = []
        final_stats.append(f"Total Episodes: {self.current_episode}")
        final_stats.append(f"Total Steps: {self.n_calls}")
        final_stats.append(f"Training Time: {time_str}")
        
        if len(self.episode_rewards) > 0:
            final_stats.append(f"Final Reward (100-ep avg): {np.mean(self.episode_rewards[-100:]):.2f}")
        
        if len(self.win_rate_history) > 0:
            final_stats.append(f"Final Win Rate (100-ep avg): {np.mean(self.win_rate_history[-100:]) * 100:.2f}%")
        
        if len(self.balance_history) > 0:
            final_stats.append(f"Final Balance: {self.balance_history[-1]:.2f}")
        
        if len(self.drawdown_history) > 0:
            final_stats.append(f"Avg Drawdown (100-ep): {np.mean(self.drawdown_history[-100:]) * 100:.2f}%")
        
        # Add text box with stats
        stats_text = '\n'.join(final_stats)
        fig.text(0.5, 0.02, stats_text, ha='center', va='bottom', fontsize=14,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        
        # Save the dashboard
        plt.savefig(os.path.join(plot_dir, 'training_summary.png'), dpi=200, bbox_inches='tight')
        plt.close()


class ProgressiveRewardCallback(TradeCallback):
    """
    Callback for progressive reward scaling during training.
    Extends TradeCallback with reward scaling functionality.
    """
    
    def __init__(
        self,
        log_dir: str,
        save_path: str,
        progressive_stages: List[Tuple[int, Dict[str, float]]],
        **kwargs
    ):
        """
        Initialize the progressive reward callback.
        
        Args:
            log_dir (str): Directory for logs
            save_path (str): Directory to save models
            progressive_stages (List[Tuple[int, Dict[str, float]]]): List of tuples with (step, reward_config)
            **kwargs: Additional arguments for TradeCallback
        """
        super(ProgressiveRewardCallback, self).__init__(log_dir, save_path, **kwargs)
        
        self.progressive_stages = progressive_stages
        self.current_stage = 0
        self.next_stage_step = progressive_stages[0][0] if progressive_stages else float('inf')
        
        # Create stage transition log
        self.stage_log_file = None
    
    def _init_callback(self) -> None:
        """Initialize the callback."""
        super()._init_callback()
        
        # Create stage transition log
        self.stage_log_file = open(os.path.join(self.log_dir, 'stage_transitions.csv'), 'w')
        self.stage_log_file.write("stage,timestep,reward_config\n")
        
        # Log initial stage
        if self.progressive_stages:
            initial_config = self.progressive_stages[0][1]
            self.stage_log_file.write(f"0,0,{json.dumps(initial_config)}\n")
            self.stage_log_file.flush()
    
    def _on_step(self) -> bool:
        """
        Callback called at each step.
        Handles stage transitions for progressive rewards.
        
        Returns:
            bool: Whether to continue training
        """
        # Check if we need to transition to the next stage
        if self.current_stage < len(self.progressive_stages) - 1 and self.n_calls >= self.next_stage_step:
            self._transition_to_next_stage()
        
        return super()._on_step()
    
    def _transition_to_next_stage(self) -> None:
        """Transition to the next reward stage."""
        self.current_stage += 1
        
        # Get new reward configuration
        stage_step, reward_config = self.progressive_stages[self.current_stage]
        
        # Set next stage step
        if self.current_stage < len(self.progressive_stages) - 1:
            self.next_stage_step = self.progressive_stages[self.current_stage + 1][0]
        else:
            self.next_stage_step = float('inf')
        
        # Log stage transition
        if self.verbose > 0:
            print(f"\n===== Transitioning to Stage {self.current_stage + 1} at step {self.n_calls} =====")
            print(f"New reward configuration: {reward_config}")
        
        # Log to stage transition file
        self.stage_log_file.write(f"{self.current_stage + 1},{self.n_calls},{json.dumps(reward_config)}\n")
        self.stage_log_file.flush()
        
        # Save checkpoint at stage transition
        stage_model_path = os.path.join(self.save_path, f'model_stage{self.current_stage}_transition.zip')
        self.model.save(stage_model_path)
        
        if self.verbose > 0:
            print(f"Stage transition checkpoint saved to {stage_model_path}")
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Close stage log file
        if hasattr(self, 'stage_log_file') and self.stage_log_file is not None:
            self.stage_log_file.close()
        
        super()._on_training_end()