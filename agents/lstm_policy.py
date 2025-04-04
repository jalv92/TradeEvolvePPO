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
    Con regularización mejorada para prevenir sobreajuste.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: callable,
        lstm_hidden_size: int = 256,  # Reducido de 512 a 256 para evitar sobreajuste
        num_lstm_layers: int = 2,
        lstm_bidirectional: bool = False,
        dropout: float = 0.2,  # NUEVO: Dropout para regularización
        weight_decay: float = 1e-5,  # NUEVO: Regularización L2
        activation_fn: Type[nn.Module] = nn.Tanh,
        use_sde: bool = False,
        log_std_init: float = -0.5,
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
        
        # MODIFICADO: Manejar tanto espacios de observación 1D como 2D
        if len(self._observation_shape) == 1:
            # Si es un vector 1D, necesitamos inferir seq_len y features
            # Calculamos input_features dividiendo el tamaño total por seq_len
            self._seq_len = 60  # Mantenemos window_size=60 como valor por defecto
            self._input_features = self._observation_shape[0] // self._seq_len
            
            # Si hay un resto, incrementamos input_features para asegurarnos de que cubre toda la entrada
            if self._observation_shape[0] % self._seq_len != 0:
                self._input_features += 1
                
            print(f"Observación 1D detectada con forma {self._observation_shape}")
            print(f"Calculado automáticamente: seq_len={self._seq_len}, input_features={self._input_features}")
        elif len(self._observation_shape) == 2:
            # Extracción estándar para formato 2D
            self._seq_len = self._observation_shape[0]  # window_size (ej: 60)
            self._input_features = self._observation_shape[1]  # num_features+4 (ej: 25)
            print(f"Observación 2D detectada con forma {self._observation_shape}")
            print(f"Using seq_len={self._seq_len}, input_features={self._input_features}")
        else:
            # Caso de error, utilizar valores por defecto
            self._seq_len = 60
            self._input_features = 10  # Aumentamos el valor por defecto para manejar vectores más grandes
            print(f"ADVERTENCIA: Forma de observación no reconocida: {self._observation_shape}")
            print(f"Usando valores por defecto: seq_len={self._seq_len}, input_features={self._input_features}")
        
        # Contador para reducir logs - reducir drásticamente para mejorar rendimiento
        self._log_counter = 0
        self._log_frequency = 5000  # Solo mostrar logs cada 5000 llamadas (antes eran 500)
        
        # Control para ajuste dinámico del LSTM
        self._adjusted_lstm = False
        
        # MODIFICADO: Eliminar net_arch de kwargs si existe para evitar duplicación
        if 'net_arch' in kwargs:
            del kwargs['net_arch']
        
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
        
        # Guardar parámetros de regularización
        self.dropout_rate = dropout
        self.weight_decay = weight_decay
        
        # Crear red LSTM con dropout
        self.lstm = nn.LSTM(
            input_size=self._input_features,
            hidden_size=self._lstm_hidden_size,
            num_layers=self._num_lstm_layers,
            batch_first=True,
            bidirectional=self._lstm_bidirectional,
            dropout=dropout if self._num_lstm_layers > 1 else 0  # Aplicar dropout entre capas LSTM
        )
        
        # Capa de dropout adicional después de LSTM
        self.dropout = nn.Dropout(dropout)
        
        # Definir las redes de política y valor con regularización
        self.custom_policy_net = nn.Sequential(
            nn.Linear(lstm_output_size, 128),  # Capa intermedia más pequeña
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(128, action_space.shape[0])
        )
        
        self.custom_value_net = nn.Sequential(
            nn.Linear(lstm_output_size, 128),  # Capa intermedia más pequeña
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # Aplicar inicialización de pesos con regularización
        for module in [self.custom_policy_net, self.custom_value_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    # Inicialización Xavier para evitar la vanishing/exploding gradient
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
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
        Preprocesa las observaciones para el LSTM de manera optimizada para GPU.
        
        Args:
            obs: Tensor de observaciones (batch_size, *obs_shape)
            
        Returns:
            Tensor: Características procesadas por el LSTM
        """
        # Minimizar logs para mejorar rendimiento
        self._log_counter += 1
        should_log = self._log_counter % self._log_frequency == 0
        
        # Verificar tipos para evitar transferencias CPU-GPU innecesarias
        if obs.dtype != torch.float32:
            # Convertir a float32 directamente para evitar problemas de tipos
            obs = obs.to(dtype=torch.float32)
            
        if should_log:
            print(f"LSTM batch size: {obs.shape[0]}, obs shape: {obs.shape}, device: {obs.device}")
        
        batch_size = obs.shape[0]
        
        # Manejar correctamente las dimensiones de entrada
        # En entornos de trading, generalmente tenemos un vector plano que necesitamos reshape
        if len(obs.shape) == 2:  # Formato (batch_size, features)
            # Este es el caso habitual con stable_baselines3 - un vector plano
            if should_log:
                print(f"Vector plano detectado de tamaño {obs.shape}")
                
            # Aquí necesitamos una solución importante: la entrada LSTM se ajustará al 
            # tamaño actual del vector en lugar de intentar forzar un reshape específico
            
            # Crear mini-secuencias para procesado LSTM
            # Vamos a dividir el vector en bloques de input_features
            total_features = obs.shape[1]
            
            # Determinar el número de características por paso temporal
            # Calculamos cuántos timesteps podemos obtener del total de características
            n_timesteps = min(self._seq_len, total_features)
            
            # Características por paso temporal
            features_per_timestep = total_features // n_timesteps
            
            if should_log:
                print(f"Creando secuencia: {n_timesteps} pasos con {features_per_timestep} características por paso")
            
            # Ajustar la red LSTM para trabajar con estas dimensiones
            # Solo hacemos esto la primera vez que se llama
            if not hasattr(self, '_adjusted_lstm') or not self._adjusted_lstm:
                self._input_features = features_per_timestep
                # Recrear el LSTM con las dimensiones correctas
                self.lstm = nn.LSTM(
                    input_size=features_per_timestep,
                    hidden_size=self._lstm_hidden_size,
                    num_layers=self._num_lstm_layers,
                    batch_first=True,
                    bidirectional=self._lstm_bidirectional,
                    dropout=self.dropout_rate if self._num_lstm_layers > 1 else 0
                )
                
                # Mover el LSTM al mismo dispositivo que los tensores de entrada
                self.lstm = self.lstm.to(obs.device)
                
                self._adjusted_lstm = True
                if should_log:
                    print(f"LSTM ajustado para trabajar con input_size={features_per_timestep} en dispositivo {obs.device}")
            
            # Reshape de la entrada para que sea (batch, timesteps, features)
            # Si el tamaño no divide exactamente, recortamos las características extras al final
            usable_features = n_timesteps * features_per_timestep
            if usable_features < total_features:
                if should_log:
                    print(f"Recortando entrada: usando {usable_features}/{total_features} características")
                obs = obs[:, :usable_features]
            
            # Reshape a formato (batch, timesteps, features)
            x = obs.reshape(batch_size, n_timesteps, features_per_timestep)
            
        elif len(obs.shape) == 3:  # Formato (batch_size, seq_len, features)
            # Ya está en el formato correcto para LSTM
            x = obs
            
            # Asegurarnos de que la red LSTM tenga las dimensiones correctas
            if not hasattr(self, '_adjusted_lstm') or not self._adjusted_lstm:
                self._input_features = obs.shape[2]
                # Recrear el LSTM con las dimensiones correctas
                self.lstm = nn.LSTM(
                    input_size=obs.shape[2],
                    hidden_size=self._lstm_hidden_size,
                    num_layers=self._num_lstm_layers,
                    batch_first=True,
                    bidirectional=self._lstm_bidirectional,
                    dropout=self.dropout_rate if self._num_lstm_layers > 1 else 0
                )
                
                # Mover el LSTM al mismo dispositivo que los tensores de entrada
                self.lstm = self.lstm.to(obs.device)
                
                self._adjusted_lstm = True
                if should_log:
                    print(f"LSTM ajustado para trabajar con input_size={obs.shape[2]} en dispositivo {obs.device}")
        else:
            # Caso de error: dimensiones inesperadas
            raise ValueError(f"Formato de observación no soportado: {obs.shape}")
        
        # Procesar con LSTM - manejar todo el batch a la vez
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
        Predecir valores para observaciones dadas.
        
        Args:
            obs: Tensor de observaciones
            
        Returns:
            torch.Tensor: Valores predichos
        """
        features = self._preprocess_lstm(obs)
        return self.custom_value_net(features).flatten()
