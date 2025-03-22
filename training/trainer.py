"""
Training module for TradeEvolvePPO.
Implements the training pipeline for PPO agents.
"""

import os
import time
import numpy as np
import pandas as pd
import gymnasium as gym
from typing import Dict, Any, Optional, List, Tuple

from data.data_loader import DataLoader
from environment.trading_env import TradingEnv
from agents.ppo_agent import PPOAgent
from training.callback import TradeCallback
from evaluation.backtest import Backtester
from utils.logger import setup_logger


class Trainer:
    """
    Trainer class for PPO trading agents.
    Handles the entire training pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Trainer.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.data_config = config.get('data_config', {})
        self.env_config = config.get('env_config', {})
        self.training_config = config.get('training_config', {})
        
        # Setup logger
        self.log_path = self.training_config.get('log_path', './logs/')
        os.makedirs(self.log_path, exist_ok=True)
        self.logger = setup_logger(
            name='trainer',
            log_file=os.path.join(self.log_path, 'trainer.log'),
            level=self.config.get('logging_config', {}).get('log_level', 'INFO'),
            console_level=self.config.get('logging_config', {}).get('console_level', 'WARNING'),
            file_level=self.config.get('logging_config', {}).get('file_level', 'INFO')
        )
        
        # Setup data loader
        self.data_loader = DataLoader(self.data_config)
        
        # Initialize environments and agent
        self.train_env = None
        self.val_env = None
        self.test_env = None
        self.agent = None
        
        # Training statistics
        self.train_stats = {}
        self.best_val_metrics = None
        
        self.logger.info("Trainer initialized")
    
    def setup(self, data_path: Optional[str] = None) -> None:
        """
        Set up the training environment and agent.
        
        Args:
            data_path (Optional[str], optional): Path to the data file. Defaults to None.
        """
        self.logger.info("Setting up training pipeline")
        
        # Load and prepare data
        try:
            train_data, val_data, test_data = self.data_loader.prepare_data(data_path)
            self.logger.info(f"Data loaded and prepared: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
            
            # Print some data stats
            data_stats = self.data_loader.get_data_stats()
            self.logger.info(f"Data statistics: {data_stats}")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
        
        # Create environments
        window_size = self.data_config.get('sequence_length', 60)
        
        try:
            # Training environment
            self.train_env = TradingEnv(
                data=train_data,
                config=self.env_config,
                initial_balance=self.env_config.get('initial_balance', 100000.0),
                window_size=window_size,
                mode='train'
            )
            
            # Validation environment
            self.val_env = TradingEnv(
                data=val_data,
                config=self.env_config,
                initial_balance=self.env_config.get('initial_balance', 100000.0),
                window_size=window_size,
                mode='validation'
            )
            
            # Test environment
            self.test_env = TradingEnv(
                data=test_data,
                config=self.env_config,
                initial_balance=self.env_config.get('initial_balance', 100000.0),
                window_size=window_size,
                mode='test'
            )
            
            self.logger.info("Environments created")
        except Exception as e:
            self.logger.error(f"Error creating environments: {e}")
            raise
        
        # Create agent
        try:
            self.agent = PPOAgent(
                env=self.train_env,
                config=self.config
            )
            self.logger.info("Agent created")
        except Exception as e:
            self.logger.error(f"Error creating agent: {e}")
            raise
        
        self.logger.info("Setup complete")
    
    def train(self) -> Dict[str, Any]:
        """
        Train the agent.
        
        Returns:
            Dict[str, Any]: Training statistics
        """
        if self.agent is None or self.train_env is None:
            self.logger.error("Agent or environment not initialized. Call setup() first.")
            raise ValueError("Agent or environment not initialized. Call setup() first.")
        
        self.logger.info("Starting training")
        start_time = time.time()
        
        # Create custom callback
        save_path = self.training_config.get('save_path', './models/')
        os.makedirs(save_path, exist_ok=True)
        
        callback = TradeCallback(
            log_dir=self.log_path,
            save_path=save_path,
            save_interval=self.training_config.get('save_interval', 10000),
            eval_interval=self.training_config.get('eval_interval', 5000),
            eval_env=self.val_env,
            verbose=1
        )
        
        # Train the agent
        try:
            self.train_stats = self.agent.train(callback=callback)
            
            # Calculate training time
            training_time = time.time() - start_time
            self.train_stats['training_time'] = training_time
            
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            self.logger.info(f"Training statistics: {self.train_stats}")
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
        
        # Save the final model
        try:
            final_model_path = os.path.join(save_path, 'final_model.zip')
            self.agent.save(final_model_path)
            self.logger.info(f"Final model saved to {final_model_path}")
        except Exception as e:
            self.logger.error(f"Error saving final model: {e}")
        
        return self.train_stats
    
    def evaluate(self, model_path: Optional[str] = None, env: Optional[gym.Env] = None) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model_path (Optional[str], optional): Path to the model to evaluate. Defaults to None.
            env (Optional[gym.Env], optional): Environment to evaluate on. Defaults to None.
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        # Use test environment if not specified
        if env is None:
            env = self.test_env
        
        if env is None:
            self.logger.error("Environment not initialized. Call setup() first.")
            raise ValueError("Environment not initialized. Call setup() first.")
        
        # Load model if path provided
        if model_path is not None:
            if self.agent is None:
                self.logger.error("Agent not initialized. Call setup() first.")
                raise ValueError("Agent not initialized. Call setup() first.")
            
            try:
                self.agent.load(model_path, env=env)
                self.logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                raise
        
        if self.agent is None:
            self.logger.error("Agent not initialized. Call setup() or provide model_path.")
            raise ValueError("Agent not initialized. Call setup() or provide model_path.")
        
        # Create backtester
        backtester = Backtester(env, self.agent, self.config)
        
        # Run evaluation
        try:
            self.logger.info("Starting evaluation")
            eval_metrics = backtester.evaluate()
            self.logger.info(f"Evaluation completed: {eval_metrics}")
            
            return eval_metrics
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise
    
    def run_progressive_training(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run progressive training with evolving reward functions.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Training results for each stage
        """
        if not self.config.get('training_config', {}).get('progressive_rewards', False):
            self.logger.info("Progressive rewards not enabled. Running normal training.")
            return {'stages': [self.train()]}
        
        self.logger.info("Starting progressive training")
        
        # Get progressive stages
        progressive_steps = self.training_config.get('progressive_steps', [200000, 600000, 1200000])
        total_timesteps = self.training_config.get('total_timesteps', 2000000)
        
        # Validate progressive steps
        if not progressive_steps or not isinstance(progressive_steps, list) or progressive_steps[-1] >= total_timesteps:
            self.logger.warning("Invalid progressive steps. Running normal training.")
            return {'stages': [self.train()]}
        
        # Log progressive training plan
        self.logger.info(f"Progressive training plan: {progressive_steps} steps, total {total_timesteps} steps")
        
        # Initialize stage results
        all_results = {'stages': []}
        cumulative_steps = 0
        
        # Etapa 1: Incentivo de exploración máximo (aprendizaje básico de operaciones)
        self.logger.info("=== Stage 1: Exploration Phase ===")
        stage1_steps = progressive_steps[0]
        
        # Configuración para Etapa 1: Exploración máxima, sin penalizaciones severas
        reward_config_stage1 = {
            'profit_weight': 0.7,
            'drawdown_weight': 0.2,
            'volatility_weight': 0.1,
            'trade_penalty': 0.0,  # Sin penalización por operaciones
            'position_penalty': 0.0,  # Sin penalización por posiciones
            'profit_scaling': 10.0,
            'use_differential_sharpe': False,
            'activity_bonus': 0.01,  # Bonus alto por actividad
            'exploration_bonus': 0.02  # Bonus máximo por exploración
        }
        
        # Aplicar configuración de Etapa 1
        orig_reward_config = self.config.get('reward_config', {}).copy()
        self.config['reward_config'] = reward_config_stage1
        
        # Entrenar Etapa 1
        self.logger.info(f"Stage 1 training: {stage1_steps} steps")
        self.agent.model.learning_rate = self.config.get('ppo_config', {}).get('learning_rate', 5e-4)
        self.agent.model.ent_coef = 0.05  # Alto coeficiente de entropía para máxima exploración
        stage1_results = self.train_stage(stage1_steps)
        all_results['stages'].append(stage1_results)
        
        cumulative_steps += stage1_steps
        self.logger.info(f"Stage 1 completed. Cumulative steps: {cumulative_steps}")
        
        # Etapa 2: Balance entre exploración y explotación
        self.logger.info("=== Stage 2: Balanced Learning Phase ===")
        stage2_steps = progressive_steps[1] - progressive_steps[0]
        
        # Configuración para Etapa 2: Balance entre exploración y explotación
        reward_config_stage2 = {
            'profit_weight': 1.0,
            'drawdown_weight': 0.3,
            'volatility_weight': 0.2,
            'trade_penalty': 0.00001,  # Pequeña penalización por operaciones
            'position_penalty': 0.00001,
            'profit_scaling': 15.0,
            'use_differential_sharpe': False,
            'activity_bonus': 0.005,
            'exploration_bonus': 0.01
        }
        
        # Aplicar configuración de Etapa 2
        self.config['reward_config'] = reward_config_stage2
        
        # Entrenar Etapa 2
        self.logger.info(f"Stage 2 training: {stage2_steps} steps")
        self.agent.model.learning_rate = self.config.get('ppo_config', {}).get('learning_rate', 5e-4) * 0.8  # Reducir learning rate
        self.agent.model.ent_coef = 0.02  # Reducir entropía para balancear exploración/explotación
        stage2_results = self.train_stage(stage2_steps)
        all_results['stages'].append(stage2_results)
        
        cumulative_steps += stage2_steps
        self.logger.info(f"Stage 2 completed. Cumulative steps: {cumulative_steps}")
        
        # Etapa 3: Refinamiento y optimización
        self.logger.info("=== Stage 3: Optimization Phase ===")
        stage3_steps = progressive_steps[2] - progressive_steps[1]
        
        # Configuración para Etapa 3: Enfoque en refinamiento y rentabilidad
        reward_config_stage3 = {
            'profit_weight': 1.2,
            'drawdown_weight': 0.4,
            'volatility_weight': 0.3,
            'trade_penalty': 0.0001,
            'position_penalty': 0.0001,
            'profit_scaling': 15.0,
            'sharpe_weight': 0.5,
            'consistency_weight': 0.4,
            'use_differential_sharpe': True,
            'activity_bonus': 0.001,
            'exploration_bonus': 0.003
        }
        
        # Aplicar configuración de Etapa 3
        self.config['reward_config'] = reward_config_stage3
        
        # Entrenar Etapa 3
        self.logger.info(f"Stage 3 training: {stage3_steps} steps")
        self.agent.model.learning_rate = self.config.get('ppo_config', {}).get('learning_rate', 5e-4) * 0.5  # Reducir learning rate
        self.agent.model.ent_coef = 0.01  # Reducir entropía para favorecer explotación
        stage3_results = self.train_stage(stage3_steps)
        all_results['stages'].append(stage3_results)
        
        cumulative_steps += stage3_steps
        self.logger.info(f"Stage 3 completed. Cumulative steps: {cumulative_steps}")
        
        # Etapa 4: Final - Optimización de rendimiento
        self.logger.info("=== Stage 4: Performance Optimization Phase ===")
        stage4_steps = total_timesteps - cumulative_steps
        
        # Configuración para Etapa 4: Enfoque en rendimiento óptimo
        reward_config_stage4 = {
            'profit_weight': 1.5,
            'drawdown_weight': 0.5,
            'volatility_weight': 0.4,
            'trade_penalty': 0.0002,  # Mayor penalización para evitar sobretradeo
            'position_penalty': 0.0002,
            'profit_scaling': 25.0,
            'use_differential_sharpe': True,
            'activity_bonus': 0.002,
            'exploration_bonus': 0.002
        }
        
        # Aplicar configuración de Etapa 4
        self.config['reward_config'] = reward_config_stage4
        
        # Entrenar Etapa 4
        self.logger.info(f"Stage 4 training: {stage4_steps} steps")
        self.agent.model.learning_rate = self.config.get('ppo_config', {}).get('learning_rate', 5e-4) * 0.2  # Learning rate mínimo
        self.agent.model.ent_coef = 0.005  # Mínima entropía para maximizar explotación
        stage4_results = self.train_stage(stage4_steps)
        all_results['stages'].append(stage4_results)
        
        # Restaurar configuración original
        self.config['reward_config'] = orig_reward_config
        
        self.logger.info("Progressive training completed")
        return all_results
    
    def train_stage(self, timesteps: int) -> Dict[str, Any]:
        """
        Train the agent for a specific number of timesteps.
        
        Args:
            timesteps (int): Number of timesteps to train for
            
        Returns:
            Dict[str, Any]: Training statistics
        """
        if self.agent is None or self.train_env is None:
            self.logger.error("Agent or environment not initialized. Call setup() first.")
            raise ValueError("Agent or environment not initialized. Call setup() first.")
        
        # Create custom callback
        save_path = self.training_config.get('save_path', './models/')
        os.makedirs(save_path, exist_ok=True)
        
        callback = TradeCallback(
            log_dir=self.log_path,
            save_path=save_path,
            save_interval=self.training_config.get('save_interval', 10000),
            eval_interval=self.training_config.get('eval_interval', 5000),
            eval_env=self.val_env,
            verbose=1
        )
        
        # Train the agent
        try:
            self.train_stats = self.agent.train(
                total_timesteps=timesteps,
                callback=callback
            )
            
            self.logger.info(f"Stage training completed: {timesteps} steps")
            self.logger.info(f"Training statistics: {self.train_stats}")
        except Exception as e:
            self.logger.error(f"Error during stage training: {e}")
            raise
        
        return self.train_stats
    
    def save_training_results(self, results: Dict[str, Any], path: Optional[str] = None) -> None:
        """
        Save training results to a file.
        
        Args:
            results (Dict[str, Any]): Training results
            path (Optional[str], optional): Save path. Defaults to None.
        """
        if path is None:
            path = os.path.join(self.log_path, 'training_results.json')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save to JSON
        try:
            import json
            with open(path, 'w') as f:
                json.dump(results, f, indent=4)
            
            self.logger.info(f"Training results saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving training results: {e}")