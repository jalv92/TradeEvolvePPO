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
            eval_episodes: int = 5,
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
            eval_episodes (int, optional): Number of episodes for evaluation. Defaults to 5.
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
        self.eval_episodes = eval_episodes
        
        # Initialize metrics
        self.best_mean_reward = -np.inf
        self.best_model_path = None
        self.eval_results = {'timesteps': [], 'mean_reward': [], 'std_reward': []}
        
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
        Evaluate the current policy and potentially save a new best model.
        """
        try:
            # Skip evaluation if no eval env is provided
            if self.eval_env is None:
                return

            # Evaluate current policy
            from stable_baselines3.common.evaluation import evaluate_policy
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=self.eval_episodes
            )
            
            # Obtener métricas de trading del entorno de evaluación
            if hasattr(self.eval_env, 'get_performance_summary'):
                # Si es un entorno envuelto, acceder al entorno original
                if hasattr(self.eval_env, 'env'):
                    env = self.eval_env.env
                    while hasattr(env, 'env'):
                        env = env.env
                    metrics = env.get_performance_summary()
                else:
                    metrics = self.eval_env.get_performance_summary()
                
                # Extraer métricas clave para mostrar
                total_trades = metrics.get('total_trades', 0)
                win_rate = metrics.get('win_rate', 0.0) * 100
                profit_factor = metrics.get('profit_factor', 0.0)
                
                # Mostrar solo métricas importantes si verbose > 0
                if self.verbose > 0:
                    print(f"Evaluación en {self.n_calls} pasos: {mean_reward:.2f} +/- {std_reward:.2f}")
                    print(f"Métricas: Trades={total_trades}, WinRate={win_rate:.1f}%, PF={profit_factor:.2f}")
            else:
                # Si no hay métricas de trading disponibles
                if self.verbose > 0:
                    print(f"Evaluación en {self.n_calls} pasos: {mean_reward:.2f} +/- {std_reward:.2f}")
            
            # Log metrics to file (sin caracteres especiales para evitar problemas de codificación)
            self._log_info(f"Timestep,Reward,Success")
            self._log_info(f"Evaluation at {self.n_calls} timesteps: {mean_reward:.2f} +/- {std_reward:.2f}")
            if hasattr(self.eval_env, 'get_performance_summary'):
                self._log_info(f"Trading metrics: Trades={total_trades}, WinRate={win_rate:.1f}%, PF={profit_factor:.2f}, AvgPnL={metrics.get('average_pnl', 0.0):.2f}")
            
            # Save details in eval_results for tracking
            self.eval_results['timesteps'].append(self.n_calls)
            self.eval_results['mean_reward'].append(mean_reward)
            self.eval_results['std_reward'].append(std_reward)
            
            if hasattr(self.eval_env, 'get_performance_summary'):
                for key, value in metrics.items():
                    if key not in self.eval_results:
                        self.eval_results[key] = []
                    self.eval_results[key].append(value)
            
            # Save model if it's the best so far
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                
                # Save best model
                best_model_path = os.path.join(self.save_path, "best_model")
                self.model.save(best_model_path)
                
                # Informar solo de modelos nuevos importantes
                if self.verbose > 0:
                    print(f"Nuevo mejor modelo guardado con recompensa {mean_reward:.2f}")
                self._log_info(f"New best model saved with reward {mean_reward:.2f}", True)
        
        except Exception as e:
            if self.verbose > 0:
                print(f"Error durante evaluación: {e}")
            self._log_info(f"Error during evaluation: {e}", True)
    
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
        if len(self.eval_results['timesteps']) > 0:
            results_path = os.path.join(self.log_dir, "eval_results.json")
            
            # Prepare data for JSON serialization
            eval_data = {}
            for key, values in self.eval_results.items():
                eval_data[key] = values if isinstance(values, list) else values.tolist()
            
            # Save to file
            try:
                with open(results_path, "w") as f:
                    json.dump(eval_data, f, indent=4)
                
                # Log the save
                self._log_info(f"Evaluation results saved to {results_path}", True)
                
                # Plot evaluation results
                self._plot_evaluation_results(eval_data)
            except Exception as e:
                self._log_info(f"Error saving evaluation results: {e}", True)
    
    def _plot_evaluation_results(self, eval_data: Dict[str, List]) -> None:
        """
        Plot evaluation results.
        
        Args:
            eval_data (Dict[str, List]): Evaluation data
        """
        if not eval_data.get("timesteps") or len(eval_data["timesteps"]) == 0:
            return
        
        try:
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(eval_data["timesteps"], eval_data["mean_reward"])
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
        except Exception as e:
            self._log_info(f"Error plotting evaluation results: {e}", True)


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