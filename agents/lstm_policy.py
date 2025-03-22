"""
Implementación de políticas LSTM para agentes PPO en trading.
Proporciona una arquitectura recurrente para capturar dependencias temporales.
"""

import gym
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Type, Union, Optional

from stable_baselines3.common.policies import ActorCriticPolicy


class LSTMPolicy(ActorCriticPolicy):
    """
    Política Actor-Crítica puramente LSTM para agentes PPO.
    Reemplaza completamente la arquitectura MLP con LSTM para procesamiento de series temporales.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: callable,
        lstm_hidden_size: int = 256,
        num_lstm_layers: int = 2,
        lstm_bidirectional: bool = False,
        activation_fn: Type[nn.Module] = nn.Tanh,
        use_sde: bool = False,
        log_std_init: float = -0.5,  # Nuevo parámetro para inicialización de log_std
        *args,
        **kwargs
    ):
        # Guardar configuración para LSTM antes de llamar a super().__init__
        self._lstm_hidden_size = lstm_hidden_size
        self._num_lstm_layers = num_lstm_layers
        self._lstm_bidirectional = lstm_bidirectional
        self._lstm_directions = 2 if lstm_bidirectional else 1
        self._observation_shape = observation_space.shape
        self._log_std_init = log_std_init
        
        # Extraer dimensiones del espacio de observación
        self._seq_len = self._observation_shape[0]  # window_size (ej: 60)
        self._input_features = self._observation_shape[1]  # num_features+4 (ej: 25)
        
        # Contador para reducir logs - aumentado para reducir frecuencia
        self._log_counter = 0
        self._log_frequency = 500  # Solo mostrar logs cada 500 llamadas (antes era 100)
        
        print(f"Observation space shape: {self._observation_shape}")
        print(f"Using seq_len={self._seq_len}, input_features={self._input_features}")
        
        # Inicializar la política base con arquitectura vacía
        # Estamos siendo explícitos en los parámetros para evitar problemas
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=[],  # Arquitectura vacía - la sobrescribiremos
            activation_fn=activation_fn,
            use_sde=use_sde,
            *args,
            **kwargs
        )
        
        # Ahora que super().__init__ ha sido llamado, podemos crear módulos nn
        # Calcular tamaño de salida del LSTM
        lstm_output_size = self._lstm_hidden_size * self._lstm_directions
        
        # Crear red LSTM
        self.lstm = nn.LSTM(
            input_size=self._input_features,
            hidden_size=self._lstm_hidden_size,
            num_layers=self._num_lstm_layers,
            batch_first=True,
            bidirectional=self._lstm_bidirectional
        )
        
        # Definir las redes de política y valor directamente
        # (ignorando el MLP extractor predeterminado)
        self.custom_policy_net = nn.Linear(lstm_output_size, action_space.shape[0])
        self.custom_value_net = nn.Linear(lstm_output_size, 1)
        
        # Inicialización adaptativa de log_std según el espacio de acción
        # Esto permite mejor exploración inicial adaptada al rango de acciones
        action_std_init = max(
            np.exp(self._log_std_init),  # Valor base
            0.3 * (action_space.high - action_space.low).mean().item() / 2  # 30% del rango medio
        )
        
        # Convertir de nuevo a log_std
        log_std_init = np.log(action_std_init)
        
        # Parámetro para los log_std (necesario para DiagGaussianDistribution)
        self.log_std = nn.Parameter(torch.full((action_space.shape[0],), log_std_init))
        
        print(f"Initialized LSTM Policy with log_std: {log_std_init:.4f} (std: {action_std_init:.4f})")
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass en la política LSTM.
        
        Args:
            obs: Tensor de observaciones
            deterministic: Si se debe usar un comportamiento determinista
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: acciones, valores, log_probs
        """
        # Convertir observaciones al formato adecuado para LSTM
        features = self._preprocess_lstm(obs)
        
        # Obtener valores
        values = self.custom_value_net(features)
        
        # Obtener logits de acción (que son en realidad las medias para DiagGaussianDistribution)
        mean_actions = self.custom_policy_net(features)
        
        # Para DiagGaussianDistribution, necesitamos proporcionar mean y log_std
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        
        # Obtener acciones
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
        
        # Calcular log probabilidades
        log_probs = distribution.log_prob(actions)
        
        return actions, values, log_probs
    
    def _preprocess_lstm(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Preprocesa las observaciones para el LSTM.
        
        Args:
            obs: Tensor de observaciones (batch_size, seq_len*features) o (batch_size, seq_len, features)
            
        Returns:
            Tensor: Características procesadas por el LSTM
        """
        # Print para depuración solo cada N llamadas (reducido para mejorar rendimiento)
        self._log_counter += 1
        should_log = self._log_counter % self._log_frequency == 0
        
        if should_log:
            print(f"LSTM input shape: {obs.shape}")
        
        # En stable_baselines3, las observaciones vienen como (batch_size, *obs_shape)
        batch_size = obs.shape[0]
        
        # Reshape si es necesario
        if len(obs.shape) == 3:
            # Ya está en formato (batch_size, seq_len, features)
            x = obs
        else:
            # Si está en formato (batch_size, seq_len*features) - lo más probable
            x = obs.reshape(batch_size, self._seq_len, self._input_features)
        
        if should_log:
            print(f"Reshaped tensor for LSTM: {x.shape}")
        
        # Procesar con LSTM
        lstm_out, _ = self.lstm(x)
        
        # Usar solo la salida del último paso temporal
        lstm_features = lstm_out[:, -1, :]
        
        return lstm_features
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluar el valor de log-likelihood y entropía para acciones dadas.
        
        Args:
            obs: Tensor de observaciones
            actions: Tensor de acciones
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: valores, log_probs, entropía
        """
        # Obtener características usando LSTM
        features = self._preprocess_lstm(obs)
        
        # Obtener valores
        values = self.custom_value_net(features).flatten()
        
        # Obtener medias de acciones
        mean_actions = self.custom_policy_net(features)
        
        # Crear distribución de acción - usando mean y log_std
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        
        # Calcular log probabilidades y entropía
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, log_probs, entropy
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Obtener acción a partir de observación.
        
        Args:
            observation: Observación
            deterministic: Si se debe usar un comportamiento determinista
            
        Returns:
            torch.Tensor: Acción
        """
        features = self._preprocess_lstm(observation)
        mean_actions = self.custom_policy_net(features)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        
        if deterministic:
            return distribution.mode()
        return distribution.sample()
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predecir valores a partir de observaciones.
        
        Args:
            obs: Tensor de observaciones
            
        Returns:
            torch.Tensor: Valores predichos
        """
        features = self._preprocess_lstm(obs)
        return self.custom_value_net(features) 