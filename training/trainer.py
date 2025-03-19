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
            level=self.config.get('logging_config', {}).get('log_level', 'INFO')
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
        progressive_steps = self.training_config.get('progressive_steps', [100000, 300000, 600000])
        total_timesteps = self.training_config.get('total_timesteps', 1000000)
        
        # Make sure steps are in ascending order and less than total steps
        progressive_steps = sorted([step for step in progressive_steps if step < total_timesteps])
        
        # Add total timesteps as the final step
        progressive_steps.append(total_timesteps)
        
        # Define reward weights for each stage
        reward_config = self.config.get('reward_config', {})
        
        # Stage 1: Focus on basic PnL
        stage1_weights = {
            'pnl_weight': 1.0,
            'risk_weight': 0.2,
            'trade_penalty': 0.0005,
            'exposure_penalty': 0.005,
            'drawdown_penalty': 0.05,
            'consistency_bonus': 0.02,
            'sharpe_weight': 0.1,
            'exit_bonus': 0.01,
            'profit_factor_bonus': 0.02
        }
        
        # Stage 2: Increase risk management importance
        stage2_weights = {
            'pnl_weight': 1.0,
            'risk_weight': 0.5,
            'trade_penalty': 0.001,
            'exposure_penalty': 0.01,
            'drawdown_penalty': 0.1,
            'consistency_bonus': 0.05,
            'sharpe_weight': 0.2,
            'exit_bonus': 0.02,
            'profit_factor_bonus': 0.05
        }
        
        # Stage 3: Emphasize consistency and sharpe ratio
        stage3_weights = {
            'pnl_weight': 1.0,
            'risk_weight': 0.7,
            'trade_penalty': 0.002,
            'exposure_penalty': 0.02,
            'drawdown_penalty': 0.15,
            'consistency_bonus': 0.1,
            'sharpe_weight': 0.3,
            'exit_bonus': 0.03,
            'profit_factor_bonus': 0.1
        }
        
        # Store the original weights
        original_weights = reward_config.copy()
        
        # Initialize results
        results = {'stages': []}
        
        # Run each stage
        for i, timesteps in enumerate(progressive_steps):
            stage_num = i + 1
            self.logger.info(f"Starting stage {stage_num} with {timesteps} timesteps")
            
            # Update reward weights based on stage
            if stage_num == 1:
                reward_weights = stage1_weights
            elif stage_num == 2:
                reward_weights = stage2_weights
            else:
                reward_weights = stage3_weights
            
            # Update config with new weights
            self.config['reward_config'] = reward_weights
            self.env_config['reward_config'] = reward_weights
            
            # Update training timesteps
            self.config['training_config']['total_timesteps'] = timesteps
            
            # If not first stage, load the best model from previous stage
            if stage_num > 1:
                best_model_path = os.path.join(
                    self.training_config.get('save_path', './models/'),
                    f'best_model_stage_{stage_num-1}.zip'
                )
                
                if os.path.exists(best_model_path):
                    self.agent.load(best_model_path, env=self.train_env)
                    self.logger.info(f"Loaded best model from stage {stage_num-1}")
            
            # Train the agent
            stage_stats = self.train()
            
            # Rename and save the best model for this stage
            best_model_path = os.path.join(
                self.training_config.get('save_path', './models/'),
                'best_model.zip'
            )
            
            if os.path.exists(best_model_path):
                stage_best_path = os.path.join(
                    self.training_config.get('save_path', './models/'),
                    f'best_model_stage_{stage_num}.zip'
                )
                
                try:
                    # Copy the best model
                    import shutil
                    shutil.copy(best_model_path, stage_best_path)
                    self.logger.info(f"Best model from stage {stage_num} saved to {stage_best_path}")
                except Exception as e:
                    self.logger.error(f"Error saving stage {stage_num} best model: {e}")
            
            # Evaluate on validation set
            val_metrics = self.evaluate(env=self.val_env)
            
            # Store results
            stage_results = {
                'stage': stage_num,
                'timesteps': timesteps,
                'reward_weights': reward_weights,
                'training_stats': stage_stats,
                'validation_metrics': val_metrics
            }
            
            results['stages'].append(stage_results)
            self.logger.info(f"Stage {stage_num} completed: {val_metrics}")
        
        # Restore original weights
        self.config['reward_config'] = original_weights
        self.env_config['reward_config'] = original_weights
        
        self.logger.info("Progressive training completed")
        return results
    
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