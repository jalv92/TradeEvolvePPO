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
from stable_baselines3.common.evaluation import evaluate_policy

import logging
logger = logging.getLogger(__name__)


class TradeCallback(BaseCallback):
    """
    Custom callback for PPO trading agent.
    Handles evaluation and checkpointing during training.
    """
    
    def __init__(
            self,
            log_dir: str,
            save_path: str,
            save_interval: int = 10000,
            eval_interval: int = 5000,
            eval_env = None,
            verbose: int = 1
        ):
        """
        Initialize the callback.
        
        Args:
            log_dir (str): Directory for logs
            save_path (str): Directory to save models
            save_interval (int, optional): Save model every this many timesteps. Defaults to 10000.
            eval_interval (int, optional): Evaluate every this many timesteps. Defaults to 5000.
            eval_env (Optional, optional): Environment for evaluation. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        super(TradeCallback, self).__init__(verbose)
        
        # Setup directories
        self.log_dir = log_dir
        self.save_path = save_path
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        
        # Setup intervals
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        
        # Evaluation environment
        self.eval_env = eval_env
        
        # Initialize metrics
        self.best_return = -np.inf
        self.best_model_path = None
        self.eval_results = []
        self.eval_timesteps = []
        
        # Initialize logs
        self.log_file = os.path.join(log_dir, "training.log")
        with open(self.log_file, "w") as f:
            f.write("Timestep,Reward,Success\n")
    
    def _init_callback(self) -> None:
        """
        Initialize callback attributes.
        """
        # Initialize parent
        super()._init_callback()
    
    def _on_step(self) -> bool:
        """
        Called at each step during training.
        
        Returns:
            bool: Whether to continue training
        """
        # Evaluate policy periodically
        if self.eval_env is not None and self.n_calls % self.eval_interval == 0:
            self._evaluate_policy()
        
        # Save model periodically
        if self.n_calls % self.save_interval == 0:
            self._save_model()
        
        return True
    
    def _evaluate_policy(self) -> None:
        """
        Evaluate the current policy.
        """
        if self.verbose > 0:
            # Reducimos los mensajes a un formato más conciso
            print(f"\n=== Evaluación en {self.n_calls} pasos ===")
        
        try:
            # Run evaluation
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=5, 
                deterministic=True
            )
            
            # Store results
            self.eval_results.append(mean_reward)
            self.eval_timesteps.append(self.n_calls)
            
            # Obtener métricas detalladas de trading
            try:
                # Intentar extraer métricas detalladas del entorno de evaluación
                metrics = self.eval_env.metrics.get_summary_stats() if hasattr(self.eval_env, 'metrics') else {}
                
                # Obtener y mostrar las métricas más importantes en la consola con formato mejorado
                total_trades = metrics.get('total_trades', 0)
                win_rate = metrics.get('win_rate', 0)
                profit_factor = metrics.get('profit_factor', 0)
                avg_trade_pnl = metrics.get('avg_trade_pnl', 0)
                max_drawdown = metrics.get('max_drawdown', 0)
                
                if self.verbose > 0:
                    print(f"Reward: {mean_reward:.2f} ± {std_reward:.2f}")
                    print(f"Trades: {total_trades} | Win Rate: {win_rate:.1f}% | PF: {profit_factor:.2f}")
                    print(f"Avg PnL: {avg_trade_pnl:.2f} | Max DD: {max_drawdown:.2f}%")
                    
                    # Si hay operaciones, mostrar distribución de dirección
                    if total_trades > 0:
                        long_trades_pct = metrics.get('long_trades_pct', 0)
                        print(f"Dirección: {long_trades_pct:.1f}% Largo | {100-long_trades_pct:.1f}% Corto")
                
                # Log detallado a archivo
                self._log_info(f"Evaluation at {self.n_calls} timesteps: {mean_reward:.2f} ± {std_reward:.2f}")
                self._log_info(f"Trading metrics: Trades={total_trades}, WinRate={win_rate:.1f}%, PF={profit_factor:.2f}, AvgPnL={avg_trade_pnl:.2f}")
            
            except Exception as e:
                # Si hay error al obtener métricas detalladas, seguimos usando el formato básico
                if self.verbose > 0:
                    print(f"Reward: {mean_reward:.2f} ± {std_reward:.2f}")
                self._log_info(f"Evaluation at {self.n_calls} timesteps: {mean_reward:.2f} ± {std_reward:.2f}")
                self._log_info(f"Error getting detailed metrics: {e}")
            
            # Save best model
            if mean_reward > self.best_return:
                self.best_return = mean_reward
                best_model_path = os.path.join(self.save_path, "best_model")
                self.model.save(best_model_path)
                self.best_model_path = best_model_path
                if self.verbose > 0:
                    print(f"✓ Nuevo mejor modelo guardado: {mean_reward:.2f}")
                self._log_info(f"New best model saved with reward {mean_reward:.2f}", True)
        
        except Exception as e:
            if self.verbose > 0:
                print(f"Error durante evaluación: {e}")
            self._log_info(f"Error during evaluation: {e}", True)
        
        if self.verbose > 0:
            print("="*40)
    
    def _save_model(self) -> None:
        """
        Save the current model.
        """
        # Reducimos la verbosidad al guardar modelos intermedios
        if self.verbose > 1:  # Solo mostrar si verbose > 1
            print(f"Guardando modelo en paso {self.n_calls}")
        
        model_path = os.path.join(self.save_path, f"model_{self.n_calls}_steps")
        self.model.save(model_path)
        
        # Log the save (solo en archivo)
        self._log_info(f"Model saved to {model_path}", True)
    
    def _log_info(self, msg: str, print_to_console: bool = False) -> None:
        """
        Log information during training.
        
        Args:
            msg (str): Message to log
            print_to_console (bool): Whether to print to console regardless of verbosity
        """
        # Solo imprimir en consola si se solicita explícitamente o si verbose > 1
        if print_to_console or self.verbose > 1:
            print(msg)
        
        # Siempre log a archivo
        with open(self.log_file, "a") as f:
            f.write(f"{msg}\n")
    
    def _on_training_end(self) -> None:
        """
        Called at the end of training.
        """
        # Save final model
        final_model_path = os.path.join(self.save_path, "final_model")
        self.model.save(final_model_path)
        
        # Log the save
        self._log_info(f"Final model saved to {final_model_path}", True)
        
        # Save evaluation results
        self._save_evaluation_results()
    
    def _save_evaluation_results(self) -> None:
        """
        Save evaluation results to file.
        """
        # Save eval results if we have any
        if self.eval_results:
            results_path = os.path.join(self.log_dir, "eval_results.json")
            
            # Prepare data
            eval_data = {
                "timesteps": self.eval_timesteps,
                "results": self.eval_results,
            }
            
            # Save to file
            with open(results_path, "w") as f:
                json.dump(eval_data, f, indent=4)
            
            # Log the save
            self._log_info(f"Evaluation results saved to {results_path}", True)
            
            # Plot evaluation results
            self._plot_evaluation_results(eval_data)
    
    def _plot_evaluation_results(self, eval_data: Dict[str, List]) -> None:
        """
        Plot evaluation results.
        
        Args:
            eval_data (Dict[str, List]): Evaluation data
        """
        if not eval_data["timesteps"]:
            return
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(eval_data["timesteps"], eval_data["results"])
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Reward")
        plt.title("Evaluation Results During Training")
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.log_dir, "eval_results.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Log the save
        self._log_info(f"Evaluation plot saved to {plot_path}", True)


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