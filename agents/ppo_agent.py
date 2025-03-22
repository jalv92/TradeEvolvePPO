"""
PPO Agent implementation for TradeEvolvePPO.
Implements a PPO-based agent for trading.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Dict, Any, Optional, Union, Tuple, List, Type
import logging

# Importar nuestra política LSTM personalizada
from agents.lstm_policy import LSTMPolicy

# Configurar logger
logger = logging.getLogger(__name__)

# NaN Prevention: Clase para detectar y prevenir valores NaN
class NaNDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.nan_detected = False
        self.nan_count = 0
    
    def detect_nan(self, tensor, tensor_name=""):
        if torch.isnan(tensor).any():
            logger.warning(f"NaN detected in {tensor_name}")
            self.nan_detected = True
            self.nan_count += 1
            return True
        return False

# NaN Prevention: Clase para limitar la desviación estándar
class StdLimiterCallback:
    def __init__(self, max_std=5.0, check_freq=100):
        self.max_std = max_std
        self.check_freq = check_freq
        self.step_count = 0
    
    def __call__(self, locals, globals):
        self.step_count += 1
        if self.step_count % self.check_freq == 0:
            model = locals['self']
            if hasattr(model.policy, 'log_std'):
                # Limitar log_std para prevenir valores extremos
                with torch.no_grad():
                    model.policy.log_std.clamp_(min=-5.0, max=np.log(self.max_std))
            
            # Verificar si hay NaN en la política
            for name, param in model.policy.named_parameters():
                if torch.isnan(param).any():
                    logger.warning(f"NaN detected in policy parameter {name}. Resetting parameter.")
                    # Reiniciar el parámetro
                    if 'log_std' in name:
                        param.data.fill_(np.log(1.0))
                    elif 'weight' in name:
                        nn.init.orthogonal_(param, gain=0.01)
                    else:
                        param.data.fill_(0.0)
        
        return True

# Modified PPO Policy to handle NaN
class StablePPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nan_detector = NaNDetector()
    
    def forward(self, obs, deterministic=False):
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        
        # NaN Prevention: Detectar NaN en los latentes
        if self.nan_detector.detect_nan(latent_pi, "latent_pi"):
            # Si detectamos NaN, reiniciar a valores seguros
            logger.warning("Resetting latent_pi due to NaN values")
            latent_pi = torch.zeros_like(latent_pi)
        
        if self.nan_detector.detect_nan(latent_vf, "latent_vf"):
            logger.warning("Resetting latent_vf due to NaN values")
            latent_vf = torch.zeros_like(latent_vf)
        
        distribution, values = self._get_action_dist_from_latent(latent_pi, latent_vf)
        
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        # NaN Prevention: Verificar acciones y probabilidades
        if self.nan_detector.detect_nan(actions, "actions"):
            logger.warning("NaN in actions, replacing with zeros")
            actions = torch.zeros_like(actions)
        
        if self.nan_detector.detect_nan(log_prob, "log_prob"):
            logger.warning("NaN in log_prob, replacing with zeros")
            log_prob = torch.zeros_like(log_prob)
        
        return actions, values, log_prob

class PPOAgent:
    """
    PPO Agent for trading using Stable-Baselines3.
    """
    
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        """
        Initialize the PPO agent.
        
        Args:
            env (gym.Env): Trading environment
            config (Dict[str, Any]): Agent configuration
        """
        self.env = env
        self.config = config
        self.logger = logging.getLogger('ppo_agent')
        
        # Mostrar configuración completa
        self._log_config()
        
        # Extract PPO configuration
        self.ppo_config = config.get('ppo_config', {})
        
        # Extract training configuration
        self.training_config = config.get('training_config', {})
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create vectorized environment
        self.vec_env = DummyVecEnv([lambda: env])
        
        # Normalize the environment (optional)
        self.vec_env = VecNormalize(
            self.vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=self.ppo_config.get('gamma', 0.99),
            epsilon=1e-8
        )
        
        # Create the PPO model
        self.model = None
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the PPO model."""
        # Extract PPO parameters
        policy_type = self.ppo_config.get('policy_type', 'MlpPolicy')
        learning_rate = self.ppo_config.get('learning_rate', 3e-4)
        n_steps = self.ppo_config.get('n_steps', 2048)
        batch_size = self.ppo_config.get('batch_size', 64)
        n_epochs = self.ppo_config.get('n_epochs', 10)
        gamma = self.ppo_config.get('gamma', 0.99)
        gae_lambda = self.ppo_config.get('gae_lambda', 0.95)
        clip_range = self.ppo_config.get('clip_range', 0.2)
        clip_range_vf = self.ppo_config.get('clip_range_vf', None)
        ent_coef = self.ppo_config.get('ent_coef', 0.01)
        vf_coef = self.ppo_config.get('vf_coef', 0.5)
        max_grad_norm = self.ppo_config.get('max_grad_norm', 0.5)
        use_sde = self.ppo_config.get('use_sde', False)
        sde_sample_freq = self.ppo_config.get('sde_sample_freq', -1)
        target_kl = self.ppo_config.get('target_kl', None)
        tensorboard_log = self.ppo_config.get('tensorboard_log', None)
        verbose = self.ppo_config.get('verbose', 0)
        
        # Obtener argumentos específicos para LSTMPolicy
        policy_kwargs = self.ppo_config.get('policy_kwargs', {})
        
        # Handle custom policy type
        if policy_type == 'LSTMPolicy':
            policy_cls = LSTMPolicy
            
            # Asegurar que los parámetros LSTM estén en policy_kwargs
            lstm_params = {
                'lstm_hidden_size': self.ppo_config.get('lstm_hidden_size', 256),
                'num_lstm_layers': self.ppo_config.get('num_lstm_layers', 2),
                'lstm_bidirectional': self.ppo_config.get('lstm_bidirectional', False)
            }
            
            # Actualizar policy_kwargs con parámetros LSTM si no están presentes
            for key, value in lstm_params.items():
                if key not in policy_kwargs:
                    policy_kwargs[key] = value
        else:
            policy_cls = policy_type
        
        # Create the model
        self.model = PPO(
            policy=policy_cls,
            env=self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=self.device
        )
    
    def _log_config(self):
        """
        Registra la configuración completa para referencia.
        """
        self.logger.info("Initializing PPO agent with config:")
        
        for key, value in self.config.items():
            self.logger.info(f"  {key}: {value}")
    
    def train(self, total_timesteps: Optional[int] = None, callback: Optional[BaseCallback] = None) -> Dict[str, Any]:
        """
        Train the PPO agent.
        
        Args:
            total_timesteps (Optional[int], optional): Number of timesteps to train for. If None, 
                                                       uses the value from config. Defaults to None.
            callback (Optional[BaseCallback], optional): Training callback. Defaults to None.
            
        Returns:
            Dict[str, Any]: Training statistics
        """
        # Extract training parameters
        if total_timesteps is None:
            total_timesteps = self.training_config.get('total_timesteps', 1000000)
        
        log_interval = self.training_config.get('log_interval', 1)
        tb_log_name = "ppo_trading"
        reset_num_timesteps = True
        
        # Create callback if not provided
        if callback is None:
            log_dir = self.training_config.get('log_path', './logs/')
            save_path = self.training_config.get('save_path', './models/')
            save_interval = self.training_config.get('save_interval', 10000)
            eval_interval = self.training_config.get('eval_interval', 5000)
            
            # Create a custom callback
            callback = TradeCallback(
                log_dir=log_dir,
                save_path=save_path,
                save_interval=save_interval,
                eval_interval=eval_interval,
                verbose=1
            )
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps
        )
        
        # Get training statistics
        training_stats = {
            'total_timesteps': total_timesteps,
            'final_reward': self.model.ep_info_buffer[-1]['r'] if len(self.model.ep_info_buffer) > 0 else 0,
            'mean_reward': np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer]) if len(self.model.ep_info_buffer) > 0 else 0,
            'episodes': len(self.model.ep_info_buffer)
        }
        
        return training_stats
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict action using the trained model.
        
        Args:
            observation (np.ndarray): Environment observation
            deterministic (bool, optional): Whether to use deterministic actions. Defaults to True.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (action, state)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train() or load() first.")
        
        # Normalize observation if VecNormalize is used
        obs = self.vec_env.normalize_obs(observation)
        
        # Get action from the model
        action, state = self.model.predict(obs, deterministic=deterministic)
        
        return action, state
    
    def save(self, path: str) -> None:
        """
        Save the trained model.
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        self.model.save(path)
        
        # Save the VecNormalize statistics
        if isinstance(self.vec_env, VecNormalize):
            vec_normalize_path = path + "_vecnormalize.pkl"
            self.vec_env.save(vec_normalize_path)
            print(f"VecNormalize statistics saved to {vec_normalize_path}")
        
        print(f"Model saved to {path}")
    
    def load(self, path: str, env: Optional[gym.Env] = None) -> None:
        """
        Load a trained model.
        
        Args:
            path (str): Path to the saved model
            env (Optional[gym.Env], optional): New environment to use. Defaults to None.
        """
        # Update environment if provided
        if env is not None:
            self.env = env
            self.vec_env = DummyVecEnv([lambda: env])
            self.vec_env = VecNormalize(
                self.vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=self.ppo_config.get('gamma', 0.99),
                epsilon=1e-8
            )
        
        # Load the model
        self.model = PPO.load(path, env=self.vec_env)
        
        # Load VecNormalize statistics if they exist
        vec_normalize_path = path + "_vecnormalize.pkl"
        if os.path.exists(vec_normalize_path) and isinstance(self.vec_env, VecNormalize):
            self.vec_env = VecNormalize.load(vec_normalize_path, self.vec_env)
            # Don't update the normalization statistics during testing
            self.vec_env.training = False
            # Don't normalize rewards when testing
            self.vec_env.norm_reward = False
            
            print(f"VecNormalize statistics loaded from {vec_normalize_path}")
        
        print(f"Model loaded from {path}")