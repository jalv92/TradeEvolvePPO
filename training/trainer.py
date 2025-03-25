"""
Training module for TradeEvolvePPO.
Implements the training pipeline for PPO agents.
"""

import os
import time
import sys
import glob
import json
import numpy as np
import torch
import pandas as pd
import traceback
from tqdm import tqdm
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from agents.ppo_agent import PPOAgent
import logging
from utils.logger import setup_logger, configure_logging
from evaluation.metrics import calculate_metrics
from environment.trading_env import TradingEnv
from utils.helpers import save_model, load_model
import matplotlib.pyplot as plt
import gc
import csv
import shutil
import psutil
import collections

from data.data_loader import DataLoader
from training.callback import TradeCallback
from evaluation.backtest import Backtester
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO


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
    
    def train(self, show_progress=True) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            show_progress (bool): Whether to show progress updates during training
        
        Returns:
            Dict[str, Any]: Training results
        """
        try:
            self.logger.info("Iniciando entrenamiento")
            start_time = time.time()
            
            # Resetear métricas y contadores si el método existe
            try:
                if hasattr(self.agent, 'reset_metrics') and callable(getattr(self.agent, 'reset_metrics')):
                    self.agent.reset_metrics()
            except Exception as e:
                self.logger.warning(f"No se pudieron resetear las métricas: {e}")
                # Continuar con el entrenamiento de todos modos
            
            # Preparar callbacks
            callbacks = self.create_callback()
            
            # Añadir callback para monitorear distribución de acciones
            action_distribution_cb = ActionDistributionCallback(verbose=1)
            callbacks.append(action_distribution_cb)
            
            # Training parameters
            total_timesteps = self.training_config.get('total_timesteps', 2000000)
            
            # Crear barra de progreso con tqdm
            with tqdm(total=total_timesteps, desc="Entrenando") as pbar:
                # Crear callback para actualizar la barra de progreso
                class TqdmCallback(BaseCallback):
                    def __init__(self, pbar, eval_env, eval_freq=10000, n_eval_episodes=3, verbose=0):
                        super(TqdmCallback, self).__init__(verbose)
                        self.pbar = pbar
                        self.eval_env = eval_env
                        self.eval_freq = eval_freq
                        self.n_eval_episodes = n_eval_episodes
                        self.last_timestep = 0
                        self.best_metrics = {
                            'mean_reward': float('-inf'),
                            'win_rate': 0.0,
                            'profit_factor': 0.0,
                            'total_trades': 0
                        }
                        
                    def _init_callback(self):
                        self.last_timestep = 0
                    
                    def _on_step(self):
                        # Actualizar barra de progreso
                        step_diff = self.model.num_timesteps - self.last_timestep
                        self.pbar.update(step_diff)
                        self.last_timestep = self.model.num_timesteps
                        
                        # Mostrar métricas en la barra de progreso cada N pasos
                        if self.model.num_timesteps % self.eval_freq == 0:
                            try:
                                # Evaluar modelo
                                mean_reward, std_reward = evaluate_policy(
                                    self.model, 
                                    self.eval_env, 
                                    n_eval_episodes=self.n_eval_episodes
                                )
                                
                                # Obtener métricas de trading si disponibles
                                metrics = {}
                                if hasattr(self.eval_env, 'get_performance_summary'):
                                    metrics = self.eval_env.get_performance_summary()
                                
                                # Extraer métricas importantes
                                win_rate = metrics.get('win_rate', 0.0) * 100
                                profit_factor = metrics.get('profit_factor', 0.0)
                                total_trades = metrics.get('total_trades', 0)
                                
                                # Actualizar mejores métricas
                                if mean_reward > self.best_metrics['mean_reward']:
                                    self.best_metrics['mean_reward'] = mean_reward
                                if win_rate > self.best_metrics['win_rate']:
                                    self.best_metrics['win_rate'] = win_rate
                                if profit_factor > self.best_metrics['profit_factor']:
                                    self.best_metrics['profit_factor'] = profit_factor
                                if total_trades > self.best_metrics['total_trades']:
                                    self.best_metrics['total_trades'] = total_trades
                                
                                # Actualizar descripción de la barra de progreso
                                self.pbar.set_description(
                                    f"R:{mean_reward:.0f} | T:{total_trades} | WR:{win_rate:.1f}% | PF:{profit_factor:.2f}"
                                )
                                
                                # Cada 100k pasos, mostrar las mejores métricas hasta ahora
                                if self.model.num_timesteps % 100000 == 0:
                                    print(f"\nMejores métricas ({self.model.num_timesteps} pasos):")
                                    print(f"  Recompensa: {self.best_metrics['mean_reward']:.1f}")
                                    print(f"  Win Rate: {self.best_metrics['win_rate']:.1f}%")
                                    print(f"  Profit Factor: {self.best_metrics['profit_factor']:.2f}")
                                    print(f"  Total Trades: {self.best_metrics['total_trades']}")
                            
                            except Exception as e:
                                if self.verbose > 0:
                                    print(f"Error en evaluación: {e}")
                        
                        return True
                
                # Añadir el callback de tqdm a la lista
                tqdm_callback = TqdmCallback(
                    pbar=pbar, 
                    eval_env=self.val_env,
                    eval_freq=self.training_config.get('eval_freq', 20000),
                    n_eval_episodes=self.training_config.get('n_eval_episodes', 3),
                    verbose=1
                )
                callbacks.append(tqdm_callback)
                
                # Iniciar entrenamiento
                self.agent.model.learn(
                    total_timesteps=total_timesteps,
                    callback=callbacks
                )
            
            # Guardar modelo entrenado
            if self.training_config.get('save_path', None):
                model_path = os.path.join(self.training_config['save_path'], "final_model")
                self.agent.save(model_path)
                self.logger.info(f"Modelo guardado en {model_path}")
            
            # Calcular estadísticas finales de entrenamiento
            elapsed_time = time.time() - start_time
            steps_per_second = total_timesteps / elapsed_time if elapsed_time > 0 else 0
            
            self.logger.info(f"Entrenamiento completado en {elapsed_time:.2f} segundos")
            self.logger.info(f"Velocidad promedio: {steps_per_second:.1f} pasos/segundo")
            
            # Mostrar distribución final de acciones
            action_distribution_cb._report_action_distribution()
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error durante el entrenamiento: {e}")
            self.logger.error(traceback.format_exc())
            
            # Intentar guardar el modelo incluso si hubo un error
            if self.training_config.get('save_path', None) and hasattr(self, 'agent') and self.agent.model is not None:
                model_path = os.path.join(self.training_config['save_path'], "error_recovery_model")
                try:
                    self.agent.save(model_path)
                    self.logger.info(f"Modelo de recuperación guardado en {model_path}")
                except Exception as save_error:
                    self.logger.error(f"No se pudo guardar el modelo de recuperación: {save_error}")
            
            return False
    
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

    def create_callback(self) -> TradeCallback:
        """
        Create and return a custom callback for training.
        
        Returns:
            TradeCallback: Configured callback for training monitoring
        """
        # Crear directorios si no existen
        save_path = self.training_config.get('save_path', './models/')
        os.makedirs(save_path, exist_ok=True)
        
        # Configurar parámetros del callback
        save_interval = self.training_config.get('save_interval', 10000)
        eval_interval = self.training_config.get('eval_interval', 5000)
        
        # Crear y devolver el callback con verbosidad reducida
        callback = TradeCallback(
            log_dir=self.log_path,
            save_path=save_path,
            save_interval=save_interval,
            eval_interval=eval_interval,
            eval_env=self.val_env,
            eval_episodes=self.training_config.get('n_eval_episodes', 5),
            verbose=0  # Configurar a 0 para mínima verbosidad
        )
        
        return callback

class ActionDistributionCallback(BaseCallback):
    """
    Monitorea la distribución de acciones tomadas por el agente durante el entrenamiento.
    """
    
    def __init__(self, verbose=0):
        super(ActionDistributionCallback, self).__init__(verbose)
        # Contadores para cada tipo de acción
        self.action_counts = {'long': 0, 'short': 0, 'close': 0}
        # Valores medios de cada acción
        self.action_values = {'long': [], 'short': [], 'close': []}
        # Total de acciones
        self.total_actions = 0
        # Últimas acciones para análisis
        self.last_actions = collections.deque(maxlen=100)
    
    def _on_step(self) -> bool:
        """Procesa la última acción tomada por el modelo"""
        # Obtener la última acción tomada
        if self.locals.get('actions') is not None:
            action = self.locals['actions'][0]  # Tomamos la primera acción (en caso de múltiples entornos)
            
            # Imprimir el tipo y valor de la acción para debugging
            print(f"Action type: {type(action)}, value: {action}")
            
            # Categorizar la acción
            if action == 0:  # Mantener
                action_category = 'close'
            elif action == 1:  # Comprar/Long
                action_category = 'long'
            elif action == 2:  # Vender/Short
                action_category = 'short'
            else:
                print(f"Acción no reconocida: {action} de tipo {type(action)}")
                return True  # Si la acción no está en el rango esperado, ignoramos
            
            # Actualizar contadores
            self.action_counts[action_category] += 1
            self.total_actions += 1
            self.last_actions.append(action_category)
            
            # Reportar distribución cada 1000 pasos
            if self.n_calls % 1000 == 0 and self.verbose > 0:
                self._report_action_distribution()
        
        return True
    
    def _report_action_distribution(self):
        """Imprime la distribución de acciones tomadas"""
        if self.total_actions == 0:
            return
        
        print("\n===== Distribución de Acciones =====")
        for action_type, count in self.action_counts.items():
            percentage = (count / self.total_actions) * 100
            print(f"{action_type}: {count} ({percentage:.2f}%)")
        
        # Análisis de sesgo en últimas acciones
        last_actions_count = collections.Counter(self.last_actions)
        print("\n===== Últimas 100 Acciones =====")
        for action_type, count in last_actions_count.items():
            percentage = (count / len(self.last_actions)) * 100
            print(f"{action_type}: {count} ({percentage:.2f}%)")
        
        # Detección de sesgo extremo
        max_action = max(last_actions_count.items(), key=lambda x: x[1], default=(None, 0))
        if max_action[1] > 80:  # Si más del 80% son del mismo tipo
            print(f"\n⚠️ ALERTA: Posible sesgo extremo hacia {max_action[0]} ({max_action[1]}%)")
        
        print("=====================================\n")
