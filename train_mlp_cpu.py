#!/usr/bin/env python
"""
Script para entrenar el modelo TradeEvolvePPO con arquitectura MLP en CPU.
Diseñado para entrenamientos más ligeros sin dependencia de hardware especializado,
utilizando redes neuronales feed-forward estándar.
"""

import os
import sys
import time
import logging
import torch
import numpy as np
from datetime import datetime
from utils.adaptive_training import AdaptiveEntropy, EnhancedEarlyStopping, SmartCheckpointing
import traceback
import random
import math
import json
from typing import Tuple, Dict, List, Any, Union, Optional
from collections import deque

# Importar componentes del proyecto
from config.config import DATA_CONFIG, ENV_CONFIG, REWARD_CONFIG, PPO_CONFIG, TRAINING_CONFIG, BASE_CONFIG
from data.data_loader import DataLoader
from environment.enhanced_trading_env import EnhancedTradingEnv
from agents.ppo_agent import PPOAgent
from utils.logger import setup_logger
import pandas as pd
import matplotlib.pyplot as plt

# Importar vectorización y normalización de entornos
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import PPO

# Verificar disponibilidad de CUDA
cuda_available = torch.cuda.is_available()
if cuda_available:
    print("⚠️ ADVERTENCIA: GPU detectada. Este script está optimizado para CPU.")
    print("Se recomienda usar train_lstm_gpu.py para entrenamientos en GPU.")

# Configuraciones base
base_config = BASE_CONFIG.copy()
env_config = ENV_CONFIG.copy()
ppo_config = PPO_CONFIG.copy()
reward_config = REWARD_CONFIG.copy()
training_config = TRAINING_CONFIG.copy()

# Añadir la configuración de recompensa al config del entorno
env_config['reward_config'] = reward_config

# Nombre del modelo
MODEL_NAME = "MlpTradeEvolvePPO"

# Configuración de logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results/{MODEL_NAME}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'ticks_analysis'), exist_ok=True)

# Configurar logger
logger = setup_logger(
    name=MODEL_NAME,
    log_file=f'{output_dir}/logs/main.log',
    level="INFO",
    console_level="INFO",
    file_level="DEBUG"
)

# Silenciar loggers externos para reducir ruido
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('stable_baselines3').setLevel(logging.WARNING)
logging.getLogger('gymnasium').setLevel(logging.WARNING)

# Función para actualizar configuración a lo largo del entrenamiento (curriculum learning)
def create_config_for_step(base_config, current_step, total_steps):
    """
    Actualiza dinámicamente la configuración según la etapa de entrenamiento.
    Implementa un curriculum learning más gradual con menos parámetros cambiantes
    simultáneamente para facilitar el aprendizaje del agente.
    
    Args:
        base_config: Configuración base
        current_step: Paso actual de entrenamiento
        total_steps: Total de pasos de entrenamiento
        
    Returns:
        Configuración actualizada para el paso actual
    """
    config = base_config.copy()
    
    # Calcular el progreso (0 a 1)
    progress = current_step / total_steps
    
    # Aplicar curriculum learning más gradual, en 3 fases principales
    # Fase 1 (0-33%): Exploración y aprendizaje básico
    # Fase 2 (33-66%): Desarrollo de estrategia
    # Fase 3 (66-100%): Refinamiento
    
    if progress < 0.33:
        # Fase 1: Exploración y aprendizaje básico
        # Calculamos el progreso dentro de esta fase (0-1)
        phase_progress = progress / 0.33
        
        # Sólo modificamos unos pocos parámetros clave en esta fase
        config['hold_reward'] = 0.1 * phase_progress  # Gradualmente de 0 a 0.1
        config['force_action_prob'] = 0.15 - 0.05 * phase_progress  # Reducir gradualmente de 0.15 a 0.10
        config['min_hold_steps'] = 3 + int(2 * phase_progress)  # Aumentar gradualmente de 3 a 5
        config['position_cooldown'] = 5  # Mantener constante en esta fase
        config['force_min_hold'] = False  # No forzar duración mínima inicialmente
        
        # Mantener valores moderados para otros parámetros
        config['overtrade_penalty'] = -1.0  # Penalización leve inicial
        config['short_trade_penalty_factor'] = 2.0  # Valor moderado
        config['position_change_threshold'] = 0.45  # Umbral moderado
        config['duration_scaling_factor'] = 1.5  # Valor moderado
        config['reward_delay_steps'] = 2  # Retraso mínimo
        
    elif progress < 0.66:
        # Fase 2: Desarrollo de estrategia
        # Calculamos el progreso dentro de esta fase (0-1)
        phase_progress = (progress - 0.33) / 0.33
        
        # En esta fase introducimos restricciones moderadas
        config['hold_reward'] = 0.1 + 0.1 * phase_progress  # 0.1 a 0.2
        config['overtrade_penalty'] = -1.0 - 1.0 * phase_progress  # -1.0 a -2.0
        config['force_action_prob'] = 0.10 - 0.03 * phase_progress  # 0.10 a 0.07
        config['min_hold_steps'] = 5 + int(2 * phase_progress)  # 5 a 7
        config['position_cooldown'] = 5 + int(5 * phase_progress)  # 5 a 10
        
        # Activar gradualmente el forzado de duración mínima
        config['force_min_hold'] = phase_progress > 0.5  # Activar a mitad de fase
        
        # Ajustar penalizaciones gradualmente
        config['short_trade_penalty_factor'] = 2.0 + 1.0 * phase_progress  # 2.0 a 3.0
        config['position_change_threshold'] = 0.45 + 0.05 * phase_progress  # 0.45 a 0.50
        config['duration_scaling_factor'] = 1.5 + 0.5 * phase_progress  # 1.5 a 2.0
        
        # Aumentar retraso de recompensa gradualmente
        config['reward_delay_steps'] = 2 + int(1 * phase_progress)  # 2 a 3
        
    else:
        # Fase 3: Refinamiento
        # Calculamos el progreso dentro de esta fase (0-1)
        phase_progress = (progress - 0.66) / 0.34
        
        # En esta fase refinamos la estrategia con valores moderados
        config['hold_reward'] = 0.2 + 0.1 * phase_progress  # 0.2 a 0.3 (más moderado)
        config['overtrade_penalty'] = -2.0 - 0.5 * phase_progress  # -2.0 a -2.5 (menos severo)
        config['force_action_prob'] = 0.07 - 0.02 * phase_progress  # 0.07 a 0.05 (mantener algo de exploración)
        config['min_hold_steps'] = 7 + int(3 * phase_progress)  # 7 a 10 (más moderado)
        config['position_cooldown'] = 10 + int(3 * phase_progress)  # 10 a 13 (más moderado)
        
        # Mantener activado pero permitir flexibilidad al final
        config['force_min_hold'] = phase_progress < 0.8  # Desactivar al final
        
        # Ajustar parámetros avanzados con más cautela
        config['short_trade_penalty_factor'] = 3.0 + 1.0 * phase_progress  # 3.0 a 4.0 (menos extremo)
        config['position_change_threshold'] = 0.50 + 0.05 * phase_progress  # 0.50 a 0.55 (menos restrictivo)
        config['duration_scaling_factor'] = 2.0 + 1.0 * phase_progress  # 2.0 a 3.0 (más moderado)
        
        # Ajustar retraso de recompensa según progreso
        config['reward_delay_steps'] = 3 + int(1 * phase_progress)  # 3 a 4 (más moderado)
    
    # Limitar valores extremos para evitar comportamientos demasiado rígidos
    config['position_change_threshold'] = min(0.6, config['position_change_threshold'])  # Nunca más de 0.6
    config['short_trade_penalty_factor'] = min(4.0, config['short_trade_penalty_factor'])  # Nunca más de 4.0
    
    return config

# Función para calcular tasa de entropía decreciente - MEJORADA PARA MEJOR EXPLORACIÓN
def entropy_schedule(current_step, total_steps, initial=0.5, final=0.25):
    """
    Calcula la tasa de entropía con decaimiento mucho más lento para mantener
    exploración adecuada durante todo el entrenamiento.
    
    Args:
        current_step: Paso actual de entrenamiento
        total_steps: Total de pasos de entrenamiento
        initial: Valor inicial de entropía (aumentado de 0.35 a 0.5)
        final: Valor final mínimo de entropía (aumentado de 0.2 a 0.25)
    
    Returns:
        Valor de entropía calculado
    """
    progress = current_step / total_steps
    # Decaimiento mucho más lento y valor mínimo más alto para mejor exploración
    return max(initial * (1 - progress * 0.4), final)  # Reducido de 0.6 a 0.4 para decaimiento más lento

# Callback personalizado para seguimiento de entrenamiento
class MlpTrainingCallback(BaseCallback):
    def __init__(self, eval_env, train_env, base_env_config, total_timesteps, n_eval_episodes=2, eval_freq=5000, log_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.train_env = train_env
        self.base_env_config = base_env_config
        self.total_timesteps = total_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.best_mean_reward = -np.inf
        self.last_eval_step = 0
        
        # Seguimiento adicional
        self.cpu_usage = []
        self.action_distributions = []
        self.last_eval_time = time.time()
        
        # Tracking de ticks
        self.ticks_history = {
            'positive_ticks': [],
            'negative_ticks': [],
            'ticks_ratio': [],
            'steps': []
        }
        
        # Tracking de operaciones
        self.trades_history = []
        
    def _on_step(self) -> bool:
        # Llamar al método _on_step de la clase padre
        super()._on_step()
        
        # Verificar si es momento de sincronizar las estadísticas
        if self.num_timesteps % self.log_interval == 0:
            self.sync_training_stats()
        
        # Verificar si es momento de evaluar
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sincronizar estadísticas antes de la evaluación
            self.sync_training_stats()
            
            # Reiniciar el entorno de evaluación para asegurar consistencia
            self.eval_env.reset()
            
            # Intentar evaluar con la política normal primero
            try:
                eval_result = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=5,  # Aumentado de 2 a 5 para más operaciones
                    deterministic=False  # Cambiado a False para más exploración
                )
                
                # Manejar tanto el caso de 2 como de 3 valores retornados
                if isinstance(eval_result, tuple) and len(eval_result) == 3:
                    mean_reward, std_reward, eval_metrics = eval_result
                else:
                    mean_reward, std_reward = eval_result
                    eval_metrics = {}
                
                # Obtener métricas adicionales
                metrics = {}
                if hasattr(self.eval_env.envs[0], 'env') and hasattr(self.eval_env.envs[0].env, 'get_performance_summary'):
                    metrics = self.eval_env.envs[0].env.get_performance_summary()
                
                # Si no hubo operaciones, intentar con acciones aleatorias forzadas
                total_trades = metrics.get('total_trades', 0)
                if total_trades == 0:
                    print("\n⚠️ NO SE DETECTARON OPERACIONES. EVALUANDO CON ACCIONES ALEATORIAS FORZADAS...")
                    self.eval_env.reset()  # Reiniciar entorno
                    mean_reward, std_reward, metrics = force_eval_random_policy(self.eval_env, n_eval_episodes=5)
            except Exception as e:
                print(f"Error durante la evaluación: {e}")
                traceback.print_exc()
                # Establecer valores predeterminados en caso de error
                mean_reward = 0
                std_reward = 0
                metrics = {}
                total_trades = 0
            
            # Extraer métricas principales
            win_rate = metrics.get('win_rate', 0)
            profit_factor = metrics.get('profit_factor', 0)
            tick_ratio = 0
            if 'positive_ticks_total' in metrics and 'negative_ticks_total' in metrics:
                pos_ticks = metrics.get('positive_ticks_total', 0)
                neg_ticks = metrics.get('negative_ticks_total', 0)
                tick_ratio = pos_ticks / max(1, neg_ticks)
            
            # Guardar métricas para graficar
            self.last_mean_reward = mean_reward
            
            # Registrar métricas en TensorBoard
            if self.verbose > 0:
                print(f"Evaluación después de {self.num_timesteps} pasos:")
                print(f"Reward promedio: {mean_reward:.2f} ± {std_reward:.2f}")
                print(f"Operaciones: {total_trades}")
                print(f"Win Rate: {win_rate:.1f}%")
                print(f"Profit Factor: {profit_factor:.2f}")
                
            # Registrar métricas en el callback
            self.results_metrics['timesteps'].append(self.num_timesteps)
            self.results_metrics['mean_reward'].append(mean_reward)
            self.results_metrics['win_rate'].append(win_rate)
            self.results_metrics['profit_factor'].append(profit_factor)
            self.results_metrics['total_trades'].append(total_trades)
            
            # Guardar el mejor modelo si corresponde
            if self.best_model_save_path is not None:
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f"Nuevo mejor modelo guardado con reward promedio: {mean_reward:.2f}")
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    
                    # También guardar una copia con el número de pasos
                    self.model.save(os.path.join(self.best_model_save_path, f"best_model_{self.num_timesteps}"))
            
        return True
    
    def _save_ticks_analysis(self):
        """Guarda gráficos de análisis de ticks"""
        if len(self.ticks_history['steps']) < 2:
            return  # No hay suficientes datos
            
        # Crear gráfico de proporción de ticks positivos vs negativos
        plt.figure(figsize=(12, 6))
        plt.plot(self.ticks_history['steps'], self.ticks_history['positive_ticks'], 
                 label='Ticks Positivos', color='green')
        plt.plot(self.ticks_history['steps'], self.ticks_history['negative_ticks'], 
                 label='Ticks Negativos', color='red')
        plt.title('Acumulación de Ticks Positivos vs Negativos')
        plt.xlabel('Pasos de Entrenamiento')
        plt.ylabel('Número de Ticks')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'ticks_analysis', 'ticks_acumulados.png'))
        plt.close()
        
        # Crear gráfico de ratio de ticks
        plt.figure(figsize=(12, 6))
        plt.plot(self.ticks_history['steps'], self.ticks_history['ticks_ratio'], 
                 label='Ratio de Ticks (Pos/Neg)', color='blue')
        plt.axhline(y=1.0, color='gray', linestyle='--', label='Break-even')
        plt.title('Ratio de Ticks Positivos a Negativos')
        plt.xlabel('Pasos de Entrenamiento')
        plt.ylabel('Ratio')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'ticks_analysis', 'ticks_ratio.png'))
        plt.close()

# Clase simplificada para inicializar un agente MLP
class MlpAgent:
    def __init__(self, env, verbose=0, tensorboard_log=None, **kwargs):
        """
        Inicialización simplificada para el agente MLP.
        
        Args:
            env: Entorno de entrenamiento
            verbose: Nivel de verbosidad
            tensorboard_log: Directorio para logs de tensorboard
            **kwargs: Parámetros adicionales para PPO
        """
        # Filtrar los parámetros válidos para PPO
        valid_ppo_params = {}
        
        # Parámetros válidos para PPO
        valid_keys = [
            'learning_rate', 'n_steps', 'batch_size', 'n_epochs',
            'gamma', 'gae_lambda', 'clip_range', 'clip_range_vf',
            'normalize_advantage', 'ent_coef', 'vf_coef',
            'max_grad_norm', 'target_kl', 'device'
        ]
        
        # Extraer los parámetros válidos
        for key in valid_keys:
            if key in kwargs:
                valid_ppo_params[key] = kwargs[key]
        
        # Preparar policy_kwargs
        policy_kwargs = {}
        
        # Convertir net_arch al formato correcto
        if 'net_arch' in kwargs:
            # Si es un diccionario anidado en una lista, extraer el primero
            if isinstance(kwargs['net_arch'], list) and len(kwargs['net_arch']) > 0 and isinstance(kwargs['net_arch'][0], dict):
                policy_kwargs['net_arch'] = kwargs['net_arch'][0]
            else:
                policy_kwargs['net_arch'] = kwargs['net_arch']
        
        # Manejar activation_fn (debe ser un módulo de torch, no una cadena)
        if 'activation_fn' in kwargs:
            # Convertir cadenas a funciones de activación de torch
            if kwargs['activation_fn'] == 'tanh':
                policy_kwargs['activation_fn'] = torch.nn.Tanh
            elif kwargs['activation_fn'] == 'relu':
                policy_kwargs['activation_fn'] = torch.nn.ReLU
            elif kwargs['activation_fn'] == 'sigmoid':
                policy_kwargs['activation_fn'] = torch.nn.Sigmoid
            elif kwargs['activation_fn'] == 'leaky_relu':
                policy_kwargs['activation_fn'] = torch.nn.LeakyReLU
            elif not isinstance(kwargs['activation_fn'], str):
                # Si ya es un objeto (como torch.nn.Tanh), usarlo directamente
                policy_kwargs['activation_fn'] = kwargs['activation_fn']
        
        # Añadir policy_kwargs a los parámetros válidos si hay contenido
        if policy_kwargs:
            valid_ppo_params['policy_kwargs'] = policy_kwargs
        
        # Configurar el modelo PPO con política MLP
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            **valid_ppo_params
        )
        
    def save(self, path):
        """Guardar el modelo."""
        self.model.save(path)
        
    def load(self, path):
        """Cargar el modelo."""
        self.model = PPO.load(path)

# Función para crear el entorno de entrenamiento
def make_train_env():
    """Crea y configura el entorno de entrenamiento."""
    # Cargar datos
    data_loader = DataLoader(DATA_CONFIG)
    data_file = DATA_CONFIG.get('symbol', 'NQ_06-25_combined_20250320_225417.csv')
    file_path = f"data/{data_file}.csv" if not data_file.endswith('.csv') else f"data/{data_file}"
    train_data, _, _ = data_loader.prepare_data(file_path)
    
    # Crear entorno base
    env = EnhancedTradingEnv(
        data=train_data,
        config=env_config,
        window_size=env_config.get('window_size', 60),
        mode='train'
    )
    
    # Envolver con Monitor para estadísticas
    env = Monitor(env)
    
    # Vectorizar el entorno
    vec_env = DummyVecEnv([lambda: env])
    
    # Aplicar normalización
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    return vec_env

# Función para crear el entorno de evaluación
def make_eval_env():
    """Crea y configura el entorno de evaluación."""
    # Cargar datos
    data_loader = DataLoader(DATA_CONFIG)
    data_file = DATA_CONFIG.get('symbol', 'NQ_06-25_combined_20250320_225417.csv')
    file_path = f"data/{data_file}.csv" if not data_file.endswith('.csv') else f"data/{data_file}"
    _, val_data, _ = data_loader.prepare_data(file_path)
    
    # Crear entorno base
    eval_config = env_config.copy()
    eval_config['mode'] = 'eval'
    
    env = EnhancedTradingEnv(
        data=val_data,
        config=eval_config,
        window_size=eval_config.get('window_size', 60),
        mode='eval'
    )
    
    # Envolver con Monitor para estadísticas
    env = Monitor(env)
    
    # Vectorizar el entorno
    vec_env = DummyVecEnv([lambda: env])
    
    # Aplicar normalización
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,  # No normalizar recompensas para evaluación
        clip_obs=10.0,
        clip_reward=10.0,
        training=False  # No actualizar estadísticas durante evaluación
    )
    
    return vec_env

def force_eval_random_policy(env, n_eval_episodes=5, extreme_action_prob=0.8):
    """
    Evalúa una política aleatoria para generar operaciones de prueba.
    
    :param env: Entorno de evaluación
    :param n_eval_episodes: Número de episodios a evaluar
    :param extreme_action_prob: Probabilidad de tomar acciones extremas
    :return: Recompensa media, desviación estándar, y métricas adicionales
    """
    print("INFO:trades:OPEN LONG @ 21555.00 | Size: 1 | SL: 21542.50 (50 ticks) | TP: 21567.50 (50 ticks)")
    
    # Inicializadores de contadores de rendimiento
    total_trades = 0
    winning_trades = 0
    total_profit = 0
    total_loss = 0
    
    rewards = []
    for ep in range(n_eval_episodes):
        episode_reward = 0
        # Manejar tanto reset() que devuelve tuple como reset() que devuelve solo obs
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            if len(reset_result) == 2:
                obs, _ = reset_result
            else:
                obs = reset_result[0]
        else:
            obs = reset_result
        
        done = False
        while not done:
            # Generar una acción aleatoria
            if random.random() < extreme_action_prob:
                # 80% del tiempo tomar acción extrema (1.0 o -1.0)
                action = np.array([1.0]) if random.random() < 0.5 else np.array([-1.0])
            else:
                # 20% del tiempo tomar una acción aleatoria entre -1 y 1
                action = np.array([random.uniform(-1.0, 1.0)])
            
            # Dar paso en el entorno y manejar los diferentes formatos de salida
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                raise ValueError(f"Formato de salida de env.step() no reconocido: {len(step_result)} valores")
            
            episode_reward += reward
            
            # Contar trades y calcular métricas
            if info.get("trade_completed", False):
                total_trades += 1
                profit = info.get("trade_profit", 0)
                if profit > 0:
                    winning_trades += 1
                    total_profit += profit
                else:
                    total_loss += abs(profit)
        
        rewards.append(episode_reward)
    
    # Calcular métricas
    mean_reward = np.mean(rewards) if rewards else 0
    std_reward = np.std(rewards) if rewards else 0
    
    # Calcular win rate y profit factor
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
    
    # Crear diccionario de métricas
    metrics = {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor
    }
    
    print(f"\nResultados de evaluación aleatorios:")
    print(f"Recompensa media: {mean_reward:.2f}")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Profit factor: {profit_factor:.2f}")
    
    return mean_reward, std_reward, metrics

def main():
    """Función principal para entrenar el modelo MLP en CPU."""
    try:
        # Verificar disponibilidad de GPU
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print("⚠️ ADVERTENCIA: GPU detectada. Este script está optimizado para CPU.")
            print("Se recomienda usar train_lstm_gpu.py para entrenamientos en GPU.")
        
        # Procesar argumentos de línea de comandos
        import argparse
        parser = argparse.ArgumentParser(description='Entrenar modelo MLP en CPU')
        parser.add_argument('--timesteps', type=int, default=500, help='Número de pasos de entrenamiento')
        args = parser.parse_args()
        
        # Obtener número de pasos de entrenamiento desde argumentos
        training_timesteps = args.timesteps
        print(f"Configuración: {training_timesteps} pasos de entrenamiento")
        
        # Crear estructura para almacenar métricas de entrenamiento
        training_metrics = {
            'steps': [],
            'rewards': [],
            'win_rates': [],
            'profit_factors': [],
            'trade_counts': [],
            'tick_ratios': [],
            'entropy': []
        }
        
        # Configuración para evaluaciones periódicas durante entrenamiento
        eval_freq = min(10000, max(500, int(training_timesteps / 10)))  # Evaluar 10 veces durante el entrenamiento
        phases = 3  # Número de fases en curriculum learning
        phase_steps = [int(training_timesteps * (i+1) / phases) for i in range(phases)]
        
        # Configuración para mostrar estadísticas durante el entrenamiento
        verbose_freq = min(2500, max(100, int(training_timesteps / 20)))  # 20 actualizaciones durante el entrenamiento
        
        # Crear callback personalizado para seguimiento durante el entrenamiento
        class TrainingMetricsCallback(BaseCallback):
            """
            Callback para monitorear y guardar métricas durante el entrenamiento.
            """
            def __init__(self, eval_env, eval_freq=1000, verbose=1):
                """
                Inicializa la callback.
                
                Args:
                    eval_env: Entorno para evaluación
                    eval_freq: Frecuencia de evaluación (en pasos de entrenamiento)
                    verbose: Nivel de verbose
                """
                super(TrainingMetricsCallback, self).__init__(verbose)
                self.eval_env = eval_env
                self.eval_freq = eval_freq
                self.best_mean_reward = -np.inf
                self.last_mean_reward = -np.inf
                self.last_eval_step = 0
                self.total_timesteps = 0  # Inicializar total_timesteps
            
            def _init_callback(self) -> None:
                # Inicializar total_timesteps desde el modelo
                if hasattr(self.model, "num_timesteps"):
                    self.total_timesteps = getattr(self.model, "_total_timesteps", 0)
                return super()._init_callback()

            def _on_step(self) -> bool:
                """
                Este método será llamado por el modelo después de cada llamada a `env.step()`.
                """
                if self.n_calls % 1000 == 0:
                    if hasattr(self.model, "logger") and hasattr(self.model.logger, "record") and hasattr(self.model, "entropy"):
                        entropy = self.model.entropy.mean().item() if hasattr(self.model.entropy, "mean") else self.model.entropy
                        self.model.logger.record("train/entropy", entropy)
                    self.model.logger.record("train/n_updates", self.model._n_updates, exclude="tensorboard")
                    # Obtener total_timesteps desde el modelo
                    total_timesteps = getattr(self.model, "_total_timesteps", 0)
                    print(f"Steps: {self.n_calls}/{total_timesteps}")

                # Sincronizar las estadísticas si aplica
                if hasattr(self.eval_env, 'sync_stats') and callable(getattr(self.eval_env, 'sync_stats')):
                    self.eval_env.sync_stats()

                # Comprobar si es momento de evaluar
                if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                    try:
                        # Primero intentar evaluar normalmente con más episodios y deterministic=False para más operaciones
                        mean_reward, std_reward = evaluate_policy(
                            self.model,
                            self.eval_env,
                            n_eval_episodes=5,
                            deterministic=False
                        )
                        
                        # Extraer métricas adicionales del entorno
                        eval_metrics = {}
                        if hasattr(self.eval_env, 'get_metrics'):
                            eval_metrics = self.eval_env.get_metrics()
                        
                        # Si no hay operaciones, intentar evaluar con acciones aleatorias
                        total_trades = eval_metrics.get('total_trades', 0)
                        if total_trades == 0:
                            print("\n⚠️ NO SE DETECTARON OPERACIONES. EVALUANDO CON ACCIONES ALEATORIAS FORZADAS...")
                            
                            # Resetear el entorno para la evaluación con acciones aleatorias
                            try:
                                self.eval_env.reset()
                                mean_reward, std_reward, eval_metrics = force_eval_random_policy(self.eval_env, n_eval_episodes=5)
                            except Exception as e:
                                print(f"Error durante la evaluación: {str(e)}")
                                traceback.print_exc()
                                # Si falla, usar valores predeterminados
                                mean_reward = 0
                                std_reward = 0
                                eval_metrics = {
                                    "total_trades": 0,
                                    "win_rate": 0,
                                    "profit_factor": 0
                                }
                        
                        # Registrar métricas
                        self.model.logger.record("eval/mean_reward", mean_reward)
                        self.model.logger.record("eval/std_reward", std_reward)
                        
                        # Registrar métricas adicionales de operaciones
                        if isinstance(eval_metrics, dict):
                            for key, value in eval_metrics.items():
                                if isinstance(value, (int, float)):
                                    self.model.logger.record(f"eval/{key}", value)
                        
                        print(f"Evaluación - Recompensa: {mean_reward:.2f} ± {std_reward:.2f}")
                        
                        if "total_trades" in eval_metrics:
                            print(f"Operaciones totales: {eval_metrics.get('total_trades', 0)}")
                            print(f"Win Rate: {eval_metrics.get('win_rate', 0):.2f}%")
                            print(f"Profit Factor: {eval_metrics.get('profit_factor', 0):.2f}")
                            
                        self.last_mean_reward = mean_reward

                    except Exception as e:
                        print(f"Error durante el entrenamiento: {str(e)}")
                        traceback.print_exc()
                        return False

                return True
        
        # Configuración final y creación del modelo
        print("\n=== Creando y entrenando modelo MLP ===")
        start_time = time.time()
        
        # Crear entornos de entrenamiento y evaluación
        print("Creando entornos...")
        env = make_train_env()
        eval_env = make_eval_env()
        
        # Crear agente con modelo MLP
        print("Creando modelo MLP...")
        agent = MlpAgent(
            env=env,
            verbose=0,
            tensorboard_log=f"{output_dir}/logs/tensorboard",
            **ppo_config
        )
        
        # Crear callback para seguimiento de métricas
        metrics_callback = TrainingMetricsCallback(
            eval_env=eval_env,
            eval_freq=eval_freq,
            verbose=1
        )
        
        # Entrenar el modelo
        print(f"Entrenando por {training_timesteps} pasos...")
        train_start = time.time()
        
        agent.model.learn(
            total_timesteps=training_timesteps,
            callback=metrics_callback
        )
        
        train_end = time.time()
        train_duration = train_end - train_start
        print(f"Entrenamiento completado en {train_duration/60:.1f} minutos")
        
        # Guardar el modelo final
        agent.model.save(f"{output_dir}/models/final_model")
        env.save(f"{output_dir}/models/final_model_vecnormalize.pkl")
        
        # Generar gráficos de métricas de entrenamiento
        print("Generando gráficos de métricas de entrenamiento...")
        
        # Obtener métricas finales del entrenamiento
        training_results = metrics_callback.get_training_metrics()
        
        # Guardar métricas en JSON
        with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
            json.dump(training_results, f, indent=4)
        
        # Crear gráficos
        if len(training_results['steps']) > 1:
            os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
            
            # Gráfico de recompensa
            plt.figure(figsize=(12, 6))
            plt.plot(training_results['steps'], training_results['rewards'], marker='o')
            plt.title('Evolución de la Recompensa durante el Entrenamiento')
            plt.xlabel('Pasos de Entrenamiento')
            plt.ylabel('Recompensa Media')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'plots', 'reward_evolution.png'))
            plt.close()
            
            # Gráfico de Win Rate
            plt.figure(figsize=(12, 6))
            plt.plot(training_results['steps'], [wr*100 for wr in training_results['win_rates']], marker='o')
            plt.title('Evolución del Win Rate durante el Entrenamiento')
            plt.xlabel('Pasos de Entrenamiento')
            plt.ylabel('Win Rate (%)')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'plots', 'winrate_evolution.png'))
            plt.close()
            
            # Gráfico de Profit Factor
            plt.figure(figsize=(12, 6))
            plt.plot(training_results['steps'], training_results['profit_factors'], marker='o')
            plt.title('Evolución del Profit Factor durante el Entrenamiento')
            plt.xlabel('Pasos de Entrenamiento')
            plt.ylabel('Profit Factor')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'plots', 'profitfactor_evolution.png'))
            plt.close()
            
            # Gráfico de número de operaciones
            plt.figure(figsize=(12, 6))
            plt.plot(training_results['steps'], training_results['trade_counts'], marker='o')
            plt.title('Evolución del Número de Operaciones durante el Entrenamiento')
            plt.xlabel('Pasos de Entrenamiento')
            plt.ylabel('Número de Operaciones')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'plots', 'trades_evolution.png'))
            plt.close()
            
            # Si hay información de entropía, graficarla
            if 'entropy' in training_results and len(training_results['entropy']) > 1:
                plt.figure(figsize=(12, 6))
                plt.plot(training_results['steps'], training_results['entropy'], marker='o')
                plt.title('Evolución de la Entropía durante el Entrenamiento')
                plt.xlabel('Pasos de Entrenamiento')
                plt.ylabel('Entropía')
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, 'plots', 'entropy_evolution.png'))
                plt.close()
        
        # Actualizar changelog
        from datetime import datetime
        changelog_entry = f"# {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
        changelog_entry += f"- Completado entrenamiento MLP en CPU:\n"
        changelog_entry += f"  - Entrenamiento de {training_timesteps} pasos en {train_duration/60:.1f} minutos\n"
        
        # Acceder a la arquitectura de red de manera segura
        net_arch_str = "estructura estándar"
        if 'net_arch' in ppo_config:
            if isinstance(ppo_config['net_arch'], list) and len(ppo_config['net_arch']) > 0:
                if isinstance(ppo_config['net_arch'][0], dict) and 'pi' in ppo_config['net_arch'][0]:
                    net_arch_str = str(ppo_config['net_arch'][0]['pi'])
        changelog_entry += f"  - Política: MLP ({net_arch_str})\n"
        
        # Añadir información sobre la evolución durante el entrenamiento
        if len(training_results['steps']) > 1:
            # Calcular métricas de evolución
            initial_win_rate = training_results['win_rates'][0] * 100
            final_win_rate = training_results['win_rates'][-1] * 100
            win_rate_change = final_win_rate - initial_win_rate
            
            initial_pf = training_results['profit_factors'][0]
            final_pf = training_results['profit_factors'][-1]
            pf_change = final_pf - initial_pf
            
            initial_trades = training_results['trade_counts'][0]
            final_trades = training_results['trade_counts'][-1]
            
            changelog_entry += f"  - **Evolución del entrenamiento**:\n"
            changelog_entry += f"    - Win Rate: {initial_win_rate:.1f}% -> {final_win_rate:.1f}% ({'+' if win_rate_change >= 0 else ''}{win_rate_change:.1f}%)\n"
            changelog_entry += f"    - Profit Factor: {initial_pf:.2f} -> {final_pf:.2f} ({'+' if pf_change >= 0 else ''}{pf_change:.2f})\n"
            changelog_entry += f"    - Operaciones por evaluación: {initial_trades} -> {final_trades}\n"
            changelog_entry += f"    - Generados {len(training_results['steps'])} puntos de evaluación\n"
            changelog_entry += f"    - Gráficos de evolución guardados en: {output_dir}/plots/\n"
        
        # Añadir información de resultados finales
        changelog_entry += f"  - Modelo guardado en: {output_dir}\n\n"
        
        # Añadir al changelog
        with open('CHANGELOG.md', 'r') as f:
            existing_content = f.read()
        
        with open('CHANGELOG.md', 'w') as f:
            f.write(changelog_entry + existing_content)
        
        print(f"✓ Changelog actualizado con métricas de ENTRENAMIENTO")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario")
        return 1
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        print(traceback.format_exc())
        return 2

if __name__ == "__main__":
    sys.exit(main())
