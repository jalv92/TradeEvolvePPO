#!/usr/bin/env python
"""
Script para entrenar el modelo TradeEvolvePPO con arquitectura LSTM en GPU.
Diseñado para aprovechar aceleración por GPU y capturar dependencias temporales
en datos de trading mediante redes LSTM.
"""

import os
import sys
import time
import logging
import torch
import numpy as np
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from utils.adaptive_training import AdaptiveEntropy, EnhancedEarlyStopping, SmartCheckpointing
import gymnasium as gym
import torch.nn as nn

# Importar componentes del proyecto
from config.config import (
    BASE_CONFIG,
    DATA_CONFIG,
    ENV_CONFIG,
    REWARD_CONFIG,
    PPO_CONFIG,
    TRAINING_CONFIG
)
from data.data_loader import DataLoader
from environment.enhanced_trading_env import EnhancedTradingEnv
from agents.ppo_agent import PPOAgent
from agents.lstm_policy import LSTMPolicy  # Importamos la clase LSTMPolicy directamente
from utils.logger import setup_logger
import pandas as pd
import matplotlib.pyplot as plt

# Importar vectorización y normalización de entornos
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Optimizaciones CUDA para rendimiento
torch.backends.cudnn.benchmark = True  # Mejora rendimiento para inputs de tamaño fijo

# Verificar disponibilidad de CUDA
cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

if not cuda_available:
    print("⚠️ ADVERTENCIA: GPU no disponible. Este script está optimizado para GPU.")
    print("Se recomienda usar train_mlp_cpu.py para entrenamientos en CPU.")
    user_input = input("¿Desea continuar de todos modos? (s/n): ")
    if user_input.lower() != 's':
        sys.exit(0)
else:
    # Mostrar información de GPU
    print(f"GPU detectada: {device_name}")
    print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Versión CUDA: {torch.version.cuda}")
    # Reservar memoria caché para evitar fragmentación
    torch.cuda.empty_cache()

# Nombre del modelo
MODEL_NAME = "LstmTradeEvolvePPO"

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
    Implementa un curriculum learning más equilibrado y progresivo.
    
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
    
    # Curriculum learning equilibrado con 3 fases
    # Fase 1 (0-33%): Exploración amplia con mínimas restricciones
    # Fase 2 (33-66%): Refinamiento con restricciones moderadas
    # Fase 3 (66-100%): Optimización con restricciones razonables
    
    if progress < 0.33:
        # Fase 1: Exploración amplia con restricciones mínimas
        phase_progress = progress / 0.33
        
        # Parámetros para favorecer exploración inicial
        config['hold_reward'] = 0.05  # Recompensa mínima por mantener posición
        config['force_action_prob'] = 0.2 - 0.1 * phase_progress  # Decrece de 0.2 a 0.1
        config['min_hold_steps'] = 5  # Tiempo mínimo de operación muy bajo
        config['position_cooldown'] = 8  # Tiempo de enfriamiento mínimo
        config['force_min_hold'] = False  # No forzar duración mínima
        config['overtrade_penalty'] = -1.0  # Penalización leve por sobretrading
        config['short_trade_penalty_factor'] = 1.5  # Penalización leve por trades cortos
        
    elif progress < 0.66:
        # Fase 2: Refinamiento con restricciones moderadas
        phase_progress = (progress - 0.33) / 0.33
        
        config['hold_reward'] = 0.05 + 0.05 * phase_progress  # 0.05 a 0.1
        config['force_action_prob'] = 0.1  # Mantener estable
        config['min_hold_steps'] = 5 + int(5 * phase_progress)  # 5 a 10
        config['position_cooldown'] = 8 + int(7 * phase_progress)  # 8 a 15
        config['force_min_hold'] = phase_progress > 0.5  # Activar a mitad de fase
        config['overtrade_penalty'] = -1.0 - 1.0 * phase_progress  # -1.0 a -2.0
        config['short_trade_penalty_factor'] = 1.5 + 1.0 * phase_progress  # 1.5 a 2.5
        
    else:
        # Fase 3: Optimización con restricciones razonables
        phase_progress = (progress - 0.66) / 0.34
        
        config['hold_reward'] = 0.1 + 0.05 * phase_progress  # 0.1 a 0.15
        config['force_action_prob'] = max(0.05, 0.1 - 0.05 * phase_progress)  # 0.1 a 0.05
        config['min_hold_steps'] = 10  # Mantener estable
        config['position_cooldown'] = 15  # Mantener estable
        config['force_min_hold'] = phase_progress < 0.5  # Desactivar en etapas finales
        config['overtrade_penalty'] = -2.0 - 1.0 * phase_progress  # -2.0 a -3.0
        config['short_trade_penalty_factor'] = 2.5 + 0.5 * phase_progress  # 2.5 a 3.0
    
    # Asegurar valores mínimos razonables
    config['min_hold_steps'] = max(3, config['min_hold_steps'])
    config['position_cooldown'] = max(5, config['position_cooldown'])
    config['overtrade_penalty'] = min(-0.5, config['overtrade_penalty'])
    config['short_trade_penalty_factor'] = max(1.0, config['short_trade_penalty_factor'])
    
    # Añadir log para seguimiento
    if current_step % 50000 == 0:
        logger.info(f"Curriculum Learning - Paso {current_step}, Progreso {progress:.2f}")
        logger.info(f"  min_hold_steps: {config['min_hold_steps']}")
        logger.info(f"  position_cooldown: {config['position_cooldown']}")
        logger.info(f"  force_min_hold: {config['force_min_hold']}")
        logger.info(f"  overtrade_penalty: {config['overtrade_penalty']}")
    
    return config

# Función para calcular tasa de entropía decreciente - MEJORADA PARA MEJOR EXPLORACIÓN
def entropy_schedule(current_step, total_steps, initial=0.5, final=0.03):
    """
    Calcula la tasa de entropía con decaimiento mucho más lento para mantener
    exploración adecuada durante todo el entrenamiento.
    
    Entropía significativamente mayor y decaimiento más lento específicamente para LSTM.
    """
    # Para LSTM: mantener entropía alta por más tiempo para evitar convergencia prematura
    progress = min(0.8, current_step / total_steps)  # Limitar progreso a 80% para mantener entropía final
    
    # Decaimiento más lento con mayor valor inicial y final
    if progress < 0.3:
        # Primeros 30%: mantener entropía alta
        return initial
    else:
        # Después del 30%: decaimiento gradual
        decay_progress = (progress - 0.3) / 0.5  # Normalizar 0.3-0.8 a 0-1
        return max(initial * (1 - decay_progress * 0.7), final)  # Decaimiento muy lento

# Callback personalizado para seguimiento y ajuste de entrenamiento
class LstmTrainingCallback(BaseCallback):
    def __init__(self, eval_env, train_env, base_env_config, total_timesteps, n_eval_episodes=5, eval_freq=5000, log_path=None, verbose=1):
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
        
        # Tracking de recompensas
        self.reward_history = []
        self.win_rate_history = []
        self.profit_factor_history = []
        
        # Monitoreo de GPU
        self.gpu_memory = []
        self.last_eval_time = time.time()
        self.start_time = time.time()
        
        # Para mostrar información de progreso más frecuentemente
        self.verbose_freq = 2500  # Reducido de 10000 a 2500 pasos
        
        # Seguimiento adicional
        self.gpu_usage = []
        self.action_distributions = []
        self.ticks_history = {
            'positive_ticks': [],
            'negative_ticks': [],
            'ticks_ratio': [],
            'steps': []
        }
        
        # Tracking de operaciones
        self.trades_history = []
        
    def _on_step(self):
        # 1. Actualizar configuración basada en curriculum learning
        current_config = create_config_for_step(
            self.base_env_config, 
            self.n_calls, 
            self.total_timesteps
        )
        
        # 2. Actualizar la tasa de entropía
        new_entropy = entropy_schedule(self.n_calls, self.total_timesteps)
        
        # 3. Monitorear GPU solo cada 50000 pasos para reducir overhead
        if torch.cuda.is_available() and self.n_calls > 0 and self.n_calls % 50000 == 0:
            try:
                # Usar método alternativo sin pynvml
                gpu_mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)  # En GB
                gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # En GB
                self.gpu_memory.append((self.n_calls, gpu_mem_alloc))
                
                # Solo registrar uso de GPU cada 50000 pasos
                logger.debug(f"GPU Memoria: {gpu_mem_alloc:.2f} GB asignada, {gpu_mem_reserved:.2f} GB reservada")
            except Exception as e:
                logger.warning(f"Error al monitorear GPU: {e}")
            
        # 4. Registrar estadísticas de ticks (cada 10000 pasos en vez de 1000)
        if self.n_calls > 0 and self.n_calls % 10000 == 0 and hasattr(self.train_env.envs[0], 'env'):
            base_env = self.train_env.envs[0].env
            
            if hasattr(base_env, 'positive_ticks_total') and hasattr(base_env, 'negative_ticks_total'):
                pos_ticks = base_env.positive_ticks_total
                neg_ticks = base_env.negative_ticks_total
                ratio = pos_ticks / max(1, neg_ticks)
                
                self.ticks_history['steps'].append(self.n_calls)
                self.ticks_history['positive_ticks'].append(pos_ticks)
                self.ticks_history['negative_ticks'].append(neg_ticks)
                self.ticks_history['ticks_ratio'].append(ratio)
                    
            # Registrar operaciones completadas (trades)
            if hasattr(base_env, 'trades'):
                self.trades_history = base_env.trades.copy()
                
        # 5. Mostrar información detallada de progreso cada verbose_freq pasos
        # Evitar la impresión durante la inicialización o cuando no es un múltiplo exacto
        if self.n_calls > 0 and self.n_calls % self.verbose_freq == 0:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # Estimar tiempo restante
            if self.n_calls > 0:
                steps_per_second = self.n_calls / elapsed_time
                remaining_steps = self.total_timesteps - self.n_calls
                remaining_time = remaining_steps / steps_per_second
                r_hours, r_remainder = divmod(remaining_time, 3600)
                r_minutes, r_seconds = divmod(r_remainder, 60)
                time_estimate = f"{int(r_hours)}h {int(r_minutes)}m"
            else:
                time_estimate = "calculando..."
                steps_per_second = 0
            
            # Mostrar progreso global del entrenamiento con las métricas importantes
            progress = self.n_calls / self.total_timesteps * 100
            print(f"\n{'='*80}")
            print(f"PROGRESO: {progress:.1f}% | Paso {self.n_calls}/{self.total_timesteps}")
            print(f"Tiempo transcurrido: {int(hours)}h {int(minutes)}m {int(seconds)}s | Estimado restante: {time_estimate}")
            print(f"Velocidad: {steps_per_second:.1f} pasos/s | Entropía actual: {new_entropy:.4f}")
            
            # Mostrar métricas de ticks si están disponibles
            if len(self.ticks_history['steps']) > 0:
                latest_idx = len(self.ticks_history['steps']) - 1
                ratio = self.ticks_history['ticks_ratio'][latest_idx]
                print(f"Ratio ticks (Pos/Neg): {ratio:.2f} {'✓' if ratio > 1.0 else '✗'}")
            
            # Mostrar últimos resultados de evaluación si están disponibles
            if len(self.reward_history) > 0:
                print(f"\nÚLTIMAS MÉTRICAS DE EVALUACIÓN:")
                print(f"Recompensa media: {self.reward_history[-1]:.2f}")
                print(f"Win Rate: {self.win_rate_history[-1]*100:.1f}% {'✓' if self.win_rate_history[-1] > 0.25 else '✗'}")
                print(f"Profit Factor: {self.profit_factor_history[-1]:.2f} {'✓' if self.profit_factor_history[-1] > 1.0 else '✗'}")
                
                # Analizar tendencia
                if len(self.reward_history) >= 3:
                    reward_trend = self.reward_history[-1] - self.reward_history[-3]
                    pf_trend = self.profit_factor_history[-1] - self.profit_factor_history[-3]
                    wr_trend = self.win_rate_history[-1] - self.win_rate_history[-3]
                    
                    print(f"\nTENDENCIAS (últimas 3 evaluaciones):")
                    print(f"Recompensa: {'↑' if reward_trend > 0 else '↓'} {abs(reward_trend):.2f}")
                    print(f"Win Rate: {'↑' if wr_trend > 0 else '↓'} {abs(wr_trend)*100:.1f}%")
                    print(f"Profit Factor: {'↑' if pf_trend > 0 else '↓'} {abs(pf_trend):.2f}")
                    
                    # Detección de estancamiento
                    if reward_trend < 0 and pf_trend < 0 and wr_trend < 0:
                        print(f"\n⚠️ ALERTA: Posible estancamiento detectado - todas las métricas empeorando")
                    elif reward_trend <= 0 and self.n_calls > self.total_timesteps * 0.5:
                        print(f"\n⚠️ ALERTA: Recompensa estancada después del 50% del entrenamiento")
            
            print(f"{'='*80}\n")
                
        # 6. Evaluación periódica (menos frecuente para entrenamientos largos)
        if self.n_calls - self.last_eval_step < self.eval_freq:
            return True
            
        self.last_eval_step = self.n_calls
        current_time = time.time()
        eval_interval = current_time - self.last_eval_time
        self.last_eval_time = current_time
        
        # Velocidad de entrenamiento
        steps_per_second = self.eval_freq / max(1.0, eval_interval)
        
        # Sincronizar manualmente las estadísticas
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms
        
        # Evaluar el modelo actual
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=False  # Usar modo no determinístico para evaluación más realista
        )
        
        # Obtener métricas detalladas para análisis de aprendizaje profundo
        if hasattr(self.eval_env.envs[0], 'env') and hasattr(self.eval_env.envs[0].env, 'get_performance_summary'):
            metrics = self.eval_env.envs[0].env.get_performance_summary()
            
            win_rate = metrics.get('win_rate', 0)
            profit_factor = metrics.get('profit_factor', 0)
            total_trades = metrics.get('total_trades', 0)
            
            # Guardar métricas para análisis de tendencia
            self.reward_history.append(mean_reward)
            self.win_rate_history.append(win_rate)
            self.profit_factor_history.append(profit_factor)
            
            # Limitar el historial para evitar consumo excesivo de memoria
            if len(self.reward_history) > 20:
                self.reward_history = self.reward_history[-20:]
                self.win_rate_history = self.win_rate_history[-20:]
                self.profit_factor_history = self.profit_factor_history[-20:]
        else:
            win_rate = 0
            profit_factor = 0
            total_trades = 0
        
        # Guardar mejor modelo
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.log_path is not None:
                self.model.save(os.path.join(self.log_path, 'best_model'))
                print(f"[OK] Nuevo mejor modelo guardado con recompensa: {mean_reward:.2f}")
        
        # Mostrar progreso de evaluación
        progress = self.n_calls / self.total_timesteps * 100
        print(f"\n{'='*50}")
        print(f"EVALUACIÓN en paso {self.n_calls} ({progress:.1f}%):")
        print(f"Recompensa: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Operaciones: {total_trades} | Win Rate: {win_rate*100:.1f}% | Profit Factor: {profit_factor:.2f}")
        print(f"Velocidad: {steps_per_second:.1f} pasos/s")
        print(f"{'='*50}\n")
        
        # Guardar gráficos de análisis de ticks
        self._save_ticks_analysis()
        
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

class AdaptiveLstmCallback(BaseCallback):
    """
    Callback personalizado para ajuste dinámico del entrenamiento LSTM en GPU
    """
    def __init__(
        self, 
        eval_env,
        train_env,
        base_callback,
        adaptive_entropy,
        early_stopping,
        smart_checkpointing,
        eval_freq=25000,
        progress_bar_writer=None,
        entropy_schedule=None,
        training_timesteps=0,
        gpu_memory_threshold=0.8,
        verbose=1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.train_env = train_env
        self.base_callback = base_callback
        self.adaptive_entropy = adaptive_entropy
        self.early_stopping = early_stopping
        self.smart_checkpointing = smart_checkpointing
        self.eval_freq = eval_freq
        self.progress_bar_writer = progress_bar_writer
        self.entropy_schedule = entropy_schedule
        self.training_timesteps = training_timesteps
        self.gpu_memory_threshold = gpu_memory_threshold
        
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_progress_print = self.start_time
        self.steps_per_second = 0
        self.total_episodes = 0
        self.best_mean_reward = -np.inf
        self.total_reward = 0
        self.last_eval_step = 0
        self.last_eval_time = time.time()
        self.tick_ratio_history = []
        self.gpu_memory = []  # Lista para almacenar los valores de uso de memoria GPU
        self.verbose_freq = 10000  # Valor por defecto, 10000 pasos
        self.stagnation_counter = 0
        self.last_significant_improvement = 0
        
    def _on_step(self):
        # Evitar procesamiento durante la inicialización
        if self.n_calls == 0:
            return True
            
        # No ejecutar el callback base directamente, ya que Stable Baselines 3
        # ya lo está llamando automáticamente en cada paso
        # self.base_callback._on_step()  # Eliminamos esta línea
        
        # Actualización de entropía adaptativa (menos frecuente para entrenamientos largos)
        if hasattr(self.training_env, 'get_attr') and self.n_calls % 10000 == 0:
            # Extraer la última recompensa media
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                rewards = [v for k, v in self.model.logger.name_to_value.items() if 'reward' in k]
                if rewards:
                    recent_reward = rewards[-1]
                else:
                    recent_reward = -1000  # Valor por defecto
            else:
                recent_reward = -1000
            
            # Actualizar entropía adaptativa
            new_entropy = self.adaptive_entropy.update(self.n_calls, recent_reward)
            self.model.ent_coef = new_entropy
            
            # Registrar cambio de entropía solo cada varios pasos
            if self.n_calls % self.verbose_freq == 0:
                logger.info(f"Actualizada entropía a {new_entropy:.4f} en paso {self.n_calls}")
        
        # Evaluación periódica
        if self.n_calls - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.n_calls
            current_time = time.time()
            eval_interval = current_time - self.last_eval_time
            self.last_eval_time = current_time
            
            # Velocidad de entrenamiento
            steps_per_second = self.eval_freq / max(1.0, eval_interval)
            
            # Sincronizar estadísticas
            self.eval_env.obs_rms = self.train_env.obs_rms
            self.eval_env.ret_rms = self.train_env.ret_rms
            
            # Evaluar el modelo actual
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=2,
                deterministic=False  # Evaluación no determinística para mayor realismo
            )
            
            # Obtener métricas detalladas para análisis de aprendizaje profundo
            if hasattr(self.eval_env.envs[0], 'env') and hasattr(self.eval_env.envs[0].env, 'get_performance_summary'):
                metrics = self.eval_env.envs[0].env.get_performance_summary()
                win_rate = metrics.get('win_rate', 0) 
                profit_factor = metrics.get('profit_factor', 0)
                
                # Detección avanzada de estancamiento
                if len(self.base_callback.reward_history) > 5:
                    # Calcular tendencia de las últimas 5 evaluaciones
                    reward_trend = np.mean(np.diff(self.base_callback.reward_history[-5:]))
                    
                    # Si la tendencia es negativa o estancada por mucho tiempo
                    if reward_trend <= 0:
                        self.stagnation_counter += 1
                    else:
                        # Mejora significativa detectada
                        self.stagnation_counter = 0
                        self.last_significant_improvement = self.n_calls
                        
                    # Alerta de estancamiento prolongado
                    if self.stagnation_counter >= 3:
                        elapsed_since_improvement = self.n_calls - self.last_significant_improvement
                        pct_elapsed = elapsed_since_improvement / self.model._total_timesteps * 100
                        
                        if pct_elapsed > 10 and self.n_calls > self.model._total_timesteps * 0.3:
                            print(f"\n⚠️ ADVERTENCIA: Estancamiento prolongado detectado!")
                            print(f"Última mejora significativa: hace {elapsed_since_improvement} pasos ({pct_elapsed:.1f}% del entrenamiento)")
                            print(f"Considere ajustar hiperparámetros o utilizar early stopping si persiste\n")
                
                # Actualizar early stopping con criterios más estrictos para entrenamientos largos
                should_stop = self.early_stopping.update(self.n_calls, mean_reward)
                
                # Verificar si se debe guardar un checkpoint
                if self.smart_checkpointing.should_save(self.n_calls, mean_reward):
                    checkpoint_path = os.path.join(output_dir, 'models', f'checkpoint_{self.n_calls}')
                    self.model.save(checkpoint_path)
                    print(f"Guardado checkpoint en paso {self.n_calls} con recompensa {mean_reward:.2f}")
                
                # Verificar si se debe detener el entrenamiento
                if should_stop:
                    print(f"\n⚠️ Early stopping activado en paso {self.n_calls}")
                    print(f"No se han detectado mejoras significativas en el rendimiento.")
                    return False
        
        return True
        
    def on_training_end(self):
        # Generar gráfico de evolución de entropía
        self.adaptive_entropy.plot_entropy_history(output_dir)
        
        # Generar gráfico de uso de memoria GPU
        if len(self.gpu_memory) > 2:
            steps, mem_alloc = zip(*self.gpu_memory)
            
            plt.figure(figsize=(12, 6))
            plt.plot(steps, mem_alloc, label='Memoria Asignada (GB)', color='blue')
            plt.title('Uso de Memoria GPU durante el Entrenamiento')
            plt.xlabel('Pasos de Entrenamiento')
            plt.ylabel('Memoria (GB)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'gpu_memory_usage.png'))
            plt.close()
        
        super().on_training_end()

def main():
    """Función principal para entrenamiento LSTM en GPU."""
    try:
        # Verificar disponibilidad de GPU
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Entrenando en GPU: {device_name}")
        else:
            logger.warning("No se detectó GPU. Este script está optimizado para GPU.")
            logger.warning("Se recomienda usar train_mlp_cpu.py para entrenamientos en CPU.")
            response = input("¿Desea continuar de todos modos? (s/n): ")
            if response.lower() != 's':
                return
        
        # Configuración base para el entrenamiento
        base_config = BASE_CONFIG.copy()
        env_config = ENV_CONFIG.copy()
        ppo_config = PPO_CONFIG.copy()
        reward_config = REWARD_CONFIG.copy()
        training_config = TRAINING_CONFIG.copy()
        
        # Añadir la configuración de recompensa al config del entorno
        env_config['reward_config'] = reward_config
        
        # Modificar la arquitectura de la red LSTM para ser más estable
        lstm_params = {
            'lstm_hidden_size': 128,  # Reducido para evitar sobreajuste
            'num_lstm_layers': 2,
            'lstm_bidirectional': False
        }
        
        # Actualizar configuración PPO con los parámetros específicos de LSTM
        ppo_config = PPO_CONFIG.copy()
        
        # Reemplazar n_lstm_layers por num_lstm_layers para consistencia
        if 'n_lstm_layers' in ppo_config:
            ppo_config['num_lstm_layers'] = ppo_config.pop('n_lstm_layers')
            
        # Asegurar que los parámetros LSTM estén en la configuración
        for key, value in lstm_params.items():
            if key not in ppo_config:
                ppo_config[key] = value
        
        # Configuración específica para el entrenamiento
        training_config.update({
            'total_timesteps': 2000000,  # Aumentar para entrenamientos largos (2M pasos)
            'eval_freq': 25000,  # Evaluar cada 25k pasos
            'checkpoint_freq': 50000,  # Guardar modelo cada 50k pasos
            'deterministic_eval': False,  # Permitir exploración en evaluación
            'early_stopping_patience': 200000,  # Paciencia extendida para evitar detención prematura
        })
        
        # Obtener número total de pasos de entrenamiento
        training_timesteps = training_config.get('total_timesteps', 2000000)
        
        # Cargar los datos
        logger.info("Cargando datos...")
        data_loader = DataLoader(config=base_config)
        file_path = os.path.join('data', 'dataset', f"{base_config['symbol']}.csv")
        df = data_loader.load_csv_data(file_path)
        
        if df.empty or len(df) < 1000:
            logger.error(f"Error: DataFrame vacío o muy pequeño ({len(df)} filas)")
            raise ValueError("DataFrame insuficiente para entrenamiento.")
        
        logger.info(f"Datos cargados correctamente: {len(df)} filas")
        
        # Dividir los datos
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size+val_size].copy()
        test_df = df.iloc[train_size+val_size:].copy()
        
        logger.info(f"División de datos: Entrenamiento={len(train_df)}, Validación={len(val_df)}, Test={len(test_df)}")
        
        # Configurar directorio de salida
        # Ya configurado en la definición global anteriormente
        
        # Crear entorno de entrenamiento
        logger.info("Configuración del entorno:")
        for k, v in env_config.items():
            logger.info(f"  {k}: {v}")
            
        # Verificar y añadir campo de modo
        env_config['mode'] = 'train'
        
        # Crear entornos de entrenamiento y validación
        env = EnhancedTradingEnv(
            data=train_df,
            config=env_config,
            window_size=env_config.get('window_size', 60),
            mode='train'
        )
        
        # Crear entorno de validación
        val_env_config = env_config.copy()
        val_env_config['mode'] = 'validation'
        
        val_env = EnhancedTradingEnv(
            data=val_df,
            config=val_env_config,
            window_size=env_config.get('window_size', 60),
            mode='validation'
        )
        
        # Crear Monitor para registro
        env = gym.wrappers.RecordEpisodeStatistics(env)
        val_env = gym.wrappers.RecordEpisodeStatistics(val_env)
        
        # Creamos directorios para logs
        os.makedirs(f"{output_dir}/logs/train_env", exist_ok=True)
        os.makedirs(f"{output_dir}/logs/val_env", exist_ok=True)

        # Vectorizar entornos
        train_env = DummyVecEnv([lambda: env])
        eval_env = DummyVecEnv([lambda: val_env])
        
        # Vectorización opcional (Normalización)
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=ppo_config.get('gamma', 0.99),
            epsilon=1e-08
        )
        
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=ppo_config.get('gamma', 0.99),
            epsilon=1e-08
        )
        
        # Asegurar mismo tratamiento de normalización
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms
        
        # Crear el modelo PPO con arquitectura LSTM
        logger.info("Creando modelo PPO-LSTM...")
        try:
            # Monitorear memoria GPU inicial
            if cuda_available:
                torch.cuda.empty_cache()
                logger.info(f"GPU memoria inicial: {torch.cuda.memory_allocated(0)/1e9:.2f}GB asignada, {torch.cuda.memory_reserved(0)/1e9:.2f}GB reservada")
            
            # Clonar la configuración PPO para no modificar la original
            ppo_config_copy = ppo_config.copy()
            
            # Lista de parámetros que deben ir en policy_kwargs
            policy_params = [
                'activation_fn', 
                'net_arch', 
                'lstm_hidden_size',
                'num_lstm_layers',
                'lstm_bidirectional'
            ]
            
            # Asegurarse de que policy_kwargs existe
            if 'policy_kwargs' not in ppo_config_copy:
                ppo_config_copy['policy_kwargs'] = {}
                
            # Mover todos los parámetros de política a policy_kwargs
            for param in policy_params:
                if param in ppo_config_copy:
                    ppo_config_copy['policy_kwargs'][param] = ppo_config_copy.pop(param)
            
            # Convertir strings de activation_fn a clases reales
            if 'policy_kwargs' in ppo_config_copy and 'activation_fn' in ppo_config_copy['policy_kwargs']:
                activation_fn_str = ppo_config_copy['policy_kwargs']['activation_fn']
                if activation_fn_str == 'tanh':
                    ppo_config_copy['policy_kwargs']['activation_fn'] = nn.Tanh
                elif activation_fn_str == 'relu':
                    ppo_config_copy['policy_kwargs']['activation_fn'] = nn.ReLU
                elif activation_fn_str == 'elu':
                    ppo_config_copy['policy_kwargs']['activation_fn'] = nn.ELU
                elif activation_fn_str == 'selu':
                    ppo_config_copy['policy_kwargs']['activation_fn'] = nn.SELU
                elif activation_fn_str == 'gelu':
                    ppo_config_copy['policy_kwargs']['activation_fn'] = nn.GELU
                else:
                    # Default a Tanh si no se reconoce
                    ppo_config_copy['policy_kwargs']['activation_fn'] = nn.Tanh
            
            # Eliminar parámetros no soportados por PPO
            params_to_remove = ['exploration_config']
            for param in params_to_remove:
                if param in ppo_config_copy:
                    del ppo_config_copy[param]
            
            # Parámetros estándar para el modelo
            standard_params = [
                'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 
                'gamma', 'gae_lambda', 'clip_range', 'clip_range_vf',
                'normalize_advantage', 'ent_coef', 'vf_coef', 'max_grad_norm',
                'target_kl', 'device'
            ]
            
            # Crear un diccionario con solo los parámetros estándar
            final_params = {
                key: value for key, value in ppo_config_copy.items() 
                if key in standard_params or key == 'policy_kwargs'
            }
            
            model = PPO(
                LSTMPolicy,  # Usamos la clase directamente en lugar de "LstmPolicy"
                train_env,
                verbose=0,
                tensorboard_log=f"{output_dir}/logs/tensorboard",
                **final_params
            )
            
            logger.info(f"Modelo creado y asignado a dispositivo: {model.device}")
            
        except Exception as e:
            logger.error(f"Error al crear el modelo: {str(e)}")
            raise
        
        # Callbacks para el entrenamiento
        adaptive_entropy = AdaptiveEntropy(
            initial_entropy=ppo_config['ent_coef'],
            min_entropy=0.1,  # Valor mínimo de entropía
            base_decay=0.9995,  # Decaimiento ligero
            window_size=10000  # Ventana adecuada
        )
        
        early_stopping = EnhancedEarlyStopping(
            patience=200000,     # Paciencia extendida para entrenamientos largos
            min_delta=2000,      # Mejora mínima significativa
            stagnation_threshold=50000  # Umbral de estancamiento
        )
        
        smart_checkpointing = SmartCheckpointing(
            save_interval=training_config['checkpoint_freq'],
            max_checkpoints=10,  # Mantener hasta 10 checkpoints para mejor análisis
            min_reward_diff=1000
        )
        
        # Callback personalizado para curriculum learning y evaluación base
        lstm_callback = LstmTrainingCallback(
            eval_env=eval_env,
            train_env=train_env,
            base_env_config=env_config,
            total_timesteps=training_timesteps,
            n_eval_episodes=5,
            eval_freq=training_config['eval_freq'],
            log_path=os.path.join(output_dir, 'models')
        )
        
        # Crear callback adaptativo
        adaptive_callback = AdaptiveLstmCallback(
            eval_env=eval_env,
            train_env=train_env,
            base_callback=lstm_callback,
            adaptive_entropy=adaptive_entropy,
            early_stopping=early_stopping,
            smart_checkpointing=smart_checkpointing,
            eval_freq=training_config['eval_freq']
        )
        
        # Usar el callback adaptativo
        callback = adaptive_callback
        
        # Registro de inicio de entrenamiento
        print("\n=== ENTRENAMIENTO LSTM EN 3 FASES ===")
        print("Fase 1 (0-33%): Exploración con restricciones mínimas")
        print("Fase 2 (33-66%): Refinamiento con restricciones moderadas")
        print("Fase 3 (66-100%): Optimización con restricciones razonables")
        print(f"Dispositivo: {device_name if cuda_available else 'CPU'}")
        print(f"Total timesteps: {training_timesteps:,}")
        print(f"Modelo se guardará en: {output_dir}")
        print("=" * 50)
                
        try:
            # Entrenar el modelo
            logger.info(f"Iniciando entrenamiento para {training_timesteps:,} pasos...")
            start_time = time.time()
            
            model.learn(
                total_timesteps=training_timesteps,
                callback=callback,
                log_interval=1,  # No mostrar mensajes de progreso automáticos
                tb_log_name="lstm_ppo",
                reset_num_timesteps=True
            )
            
            # Guardar el modelo final
            model.save(f"{output_dir}/models/final_model")
            train_env.save(f"{output_dir}/models/final_model_vecnormalize.pkl")
            
            # Calcular tiempo total de entrenamiento
            training_time = time.time() - start_time
            hours, remainder = divmod(training_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            logger.info(f"Entrenamiento completado en {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            # Evaluar el modelo final
            logger.info("Evaluando modelo final...")
            mean_reward, std_reward = evaluate_policy(
                model, 
                eval_env, 
                n_eval_episodes=10,
                deterministic=False
            )
            
            logger.info(f"Recompensa media final: {mean_reward:.2f} ± {std_reward:.2f}")
            
            # Realizar evaluación detallada con número dinámico de operaciones
            logger.info("Realizando evaluación detallada con pruebas de trading...")
            
            # Crear un nuevo entorno de evaluación sin vectorizar para análisis detallado
            test_env_config = env_config.copy()
            test_env_config['mode'] = 'test'
            test_env_config['df'] = test_df
            
            test_env_raw = EnhancedTradingEnv(**test_env_config)
            
            # Ejecutar pruebas con un número dinámico de operaciones
            obs, _ = test_env_raw.reset()
            done = False
            steps = 0
            num_operations = 0
            max_steps = 10000
            
            # Definir número dinámico de operaciones máximas basado en el desempeño
            initial_max_operations = 30  # Valor base más realista
            max_operations = initial_max_operations
            
            # Almacenar información detallada de las operaciones
            operations_details = []
            
            logger.info("Ejecutando operaciones de prueba...")
            while not done and steps < max_steps:
                steps += 1
                action, _ = model.predict(obs, deterministic=False)
                
                # Registrar acción tomada para diagnóstico
                action_value = float(action[0])
                if steps % 500 == 0 or abs(action_value) > 0.7:
                    logger.debug(f"Paso {steps}: Acción={action_value:.4f}")
                
                obs, reward, terminated, truncated, info = test_env_raw.step(action)
                
                # Actualizar variable done para el bucle
                done = terminated or truncated
                
                # Verificar si se ha realizado una operación (apertura o cierre)
                if len(test_env_raw.trades) > num_operations:
                    num_operations = len(test_env_raw.trades)
                    # Obtener información detallada sobre la operación actual
                    latest_trade = test_env_raw.trades[-1]
                    trade_type = latest_trade.get('type', 'unknown')
                    direction = latest_trade.get('direction', 'unknown')
                    
                    # Guardar detalles de la operación para análisis posterior
                    trade_details = {
                        'step': steps,
                        'type': trade_type,
                        'direction': direction,
                        'entry_price': latest_trade.get('entry_price', 0),
                        'exit_price': latest_trade.get('exit_price', 0),
                        'pnl': latest_trade.get('net_pnl', 0),
                        'close_type': latest_trade.get('close_type', 'NORMAL')
                    }
                    operations_details.append(trade_details)
                    
                    logger.info(f"Paso {steps}: Nueva operación - {trade_type} {direction} | Close: {trade_details['close_type']} | PnL: {trade_details['pnl']:.2f} (total: {num_operations})")
                
                # Adaptar el límite máximo de operaciones basado en el desempeño
                if num_operations >= 10 and num_operations % 10 == 0:
                    # Calcular tasa de éxito de las últimas 10 operaciones
                    last_10_trades = operations_details[-10:]
                    wins = sum(1 for t in last_10_trades if t['pnl'] > 0)
                    win_rate = wins / 10
                    
                    # Si el desempeño es bueno, aumentar el límite para ver más operaciones
                    if win_rate > 0.4:
                        new_max = min(max_operations + 10, 100)  # Aumentar pero con límite global de 100
                        if new_max > max_operations:
                            max_operations = new_max
                            logger.info(f"Buen desempeño detectado (Win Rate: {win_rate:.1%}). Aumentando límite a {max_operations} operaciones")
                
                # Si ya tenemos suficientes operaciones, podemos terminar
                if num_operations >= max_operations:
                    logger.info(f"Se han registrado {max_operations} operaciones, terminando la prueba")
                    break
                
                # Mostrar progreso cada 1000 pasos
                if steps % 1000 == 0:
                    logger.info(f"Completados {steps} pasos, {num_operations} operaciones registradas")
            
            # Si hay alguna operación abierta, cerrarla para incluirla en las estadísticas
            if hasattr(test_env_raw, 'position') and test_env_raw.position != 0:
                logger.info("Cerrando operación abierta al final de la evaluación...")
                # Forzar cierre de posición con acción contraria
                close_action = -1.0 if test_env_raw.position > 0 else 1.0
                test_env_raw.step(np.array([close_action]))
            
            logger.info(f"\nPrueba completa: {steps} pasos ejecutados, {num_operations} operaciones registradas")
            
            # Realizar análisis detallado de las operaciones ejecutadas
            logger.info("\n=== ANÁLISIS DETALLADO DE OPERACIONES ===")
            
            # Obtener métricas detalladas incluyendo estadísticas de ticks
            detailed_metrics = test_env_raw.get_performance_summary()
            
            # Análisis de dirección de operaciones
            long_trades = sum(1 for t in operations_details if t['direction'] == 'LONG')
            short_trades = sum(1 for t in operations_details if t['direction'] == 'SHORT')
            logger.info(f"Operaciones LONG: {long_trades} ({long_trades/max(1, num_operations)*100:.1f}%)")
            logger.info(f"Operaciones SHORT: {short_trades} ({short_trades/max(1, num_operations)*100:.1f}%)")
            
            # Análisis de resultados por tipo de cierre
            tp_hits = sum(1 for t in operations_details if t['close_type'] == 'TAKE PROFIT')
            sl_hits = sum(1 for t in operations_details if t['close_type'] == 'STOP LOSS')
            manual_close = sum(1 for t in operations_details if t['close_type'] == 'NORMAL')
            
            logger.info(f"Cierres por Take Profit: {tp_hits} ({tp_hits/max(1, num_operations)*100:.1f}%)")
            logger.info(f"Cierres por Stop Loss: {sl_hits} ({sl_hits/max(1, num_operations)*100:.1f}%)")
            logger.info(f"Cierres manuales: {manual_close} ({manual_close/max(1, num_operations)*100:.1f}%)")
            
            # Análisis de rentabilidad
            if num_operations > 0:
                avg_pnl = sum(t['pnl'] for t in operations_details) / num_operations
                max_profit = max([t['pnl'] for t in operations_details]) if operations_details else 0
                max_loss = min([t['pnl'] for t in operations_details]) if operations_details else 0
                
                logger.info(f"PnL promedio por operación: {avg_pnl:.2f}")
                logger.info(f"Mejor operación: {max_profit:.2f}")
                logger.info(f"Peor operación: {max_loss:.2f}")
            
            # Añadir información detallada al reporte
            detailed_metrics.update({
                'long_trades': long_trades,
                'short_trades': short_trades,
                'take_profit_hits': tp_hits,
                'stop_loss_hits': sl_hits,
                'manual_close': manual_close,
                'direction_bias': 'LONG' if long_trades > short_trades else 'SHORT' if short_trades > long_trades else 'NEUTRAL',
                'long_pct': long_trades/max(1, num_operations)*100,
                'short_pct': short_trades/max(1, num_operations)*100,
                'tp_pct': tp_hits/max(1, num_operations)*100,
                'sl_pct': sl_hits/max(1, num_operations)*100,
                'manual_close_pct': manual_close/max(1, num_operations)*100
            })
            
            # Guardar resumen de evaluación
            with open(f"{output_dir}/evaluation_summary.json", "w") as f:
                import json
                json.dump(detailed_metrics, f, indent=4, default=str)
            
            # Actualizar el changelog con información más detallada
            from datetime import datetime
            changelog_entry = f"# {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
            changelog_entry += f"- Completado entrenamiento LSTM en {device_name if cuda_available else 'CPU'}:\n"
            changelog_entry += f"  - Entrenamiento de {training_timesteps} pasos en {training_time/60:.1f} minutos\n"
            changelog_entry += f"  - Política: LSTM ({ppo_config.get('lstm_hidden_size', 256)} unidades, {ppo_config.get('num_lstm_layers', 2)} capas)\n"
            changelog_entry += f"  - Total operaciones: {detailed_metrics.get('total_trades', 0)}\n"
            changelog_entry += f"  - Tasa de éxito: {detailed_metrics.get('win_rate', 0) * 100:.1f}%\n"
            changelog_entry += f"  - Profit Factor: {detailed_metrics.get('profit_factor', 0):.2f}\n"
            
            # Añadir información de dirección de operaciones
            direction_bias = detailed_metrics.get('direction_bias', 'NEUTRAL')
            changelog_entry += f"  - Sesgo de dirección: {direction_bias} (LONG: {detailed_metrics.get('long_pct', 0):.1f}%, SHORT: {detailed_metrics.get('short_pct', 0):.1f}%)\n"
            
            # Añadir información de tipos de cierre
            changelog_entry += f"  - Take profit alcanzados: {tp_hits} ({detailed_metrics.get('tp_pct', 0):.1f}%)\n"
            changelog_entry += f"  - Stop loss activados: {sl_hits} ({detailed_metrics.get('sl_pct', 0):.1f}%)\n"
            changelog_entry += f"  - Cierres manuales: {manual_close} ({detailed_metrics.get('manual_close_pct', 0):.1f}%)\n"
            
            changelog_entry += f"  - Modelo guardado en: {output_dir}\n\n"
            
            # Añadir al changelog
            with open('CHANGELOG.md', 'r') as f:
                existing_content = f.read()
            
            with open('CHANGELOG.md', 'w') as f:
                f.write(changelog_entry + existing_content)
            
            logger.info(f"Evaluación completa guardada en {output_dir}/evaluation_summary.json")
            logger.info(f"Changelog actualizado")
            
            # Mostrar mensaje para commit
            logger.info("\nMensaje para commit:")
            logger.info(f"Entrenamiento LSTM en {'GPU' if cuda_available else 'CPU'} ({training_timesteps} pasos) - {detailed_metrics.get('total_trades', 0)} operaciones, Win Rate: {detailed_metrics.get('win_rate', 0) * 100:.1f}%")
            
            return model, detailed_metrics
            
        except KeyboardInterrupt:
            logger.info("\nEntrenamiento interrumpido por el usuario")
            # Guardar modelo actual en caso de interrupción
            try:
                model.save(f"{output_dir}/models/interrupted_model")
                train_env.save(f"{output_dir}/models/interrupted_model_vecnormalize.pkl")
                logger.info(f"Modelo guardado en {output_dir}/models/interrupted_model")
            except:
                logger.error("No se pudo guardar el modelo interrumpido")
        
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"Error en la función principal: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
def run_detailed_evaluation(model, env, n_episodes=5):
    """
    Realiza una evaluación detallada del modelo con métricas de trading.
    
    Args:
        model: Modelo entrenado
        env: Entorno de evaluación
        n_episodes: Número de episodios para evaluación
        
    Returns:
        Dict: Métricas de evaluación
    """
    # Resultados
    episode_rewards = []
    episode_lengths = []
    trade_metrics = []
    
    # Recolectar información detallada de operaciones
    all_operations = []
    
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        operations_in_episode = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
            steps += 1
            
            # Verificar si se han realizado nuevas operaciones
            current_trades = env.get_attr('trades')[0]
            if len(current_trades) > operations_in_episode:
                # Obtener nuevas operaciones
                for idx in range(operations_in_episode, len(current_trades)):
                    trade = current_trades[idx]
                    
                    # Registrar detalles de la operación
                    if 'exit_price' in trade:  # Solo considerar operaciones completas
                        trade_details = {
                            'episode': i,
                            'step': steps,
                            'type': trade.get('type', 'unknown'),
                            'direction': trade.get('direction', 'unknown'),
                            'entry_price': trade.get('entry_price', 0),
                            'exit_price': trade.get('exit_price', 0),
                            'pnl': trade.get('net_pnl', 0),
                            'close_type': trade.get('close_type', 'NORMAL')
                        }
                        all_operations.append(trade_details)
                        
                        # Log para diagnóstico
                        logger.debug(f"Episodio {i}, Paso {steps}: Nueva operación - {trade_details['type']} {trade_details['direction']}, Close: {trade_details['close_type']}, PnL: {trade_details['pnl']:.2f}")
                
                operations_in_episode = len(current_trades)
        
        # Obtener información del entorno vectorizado
        env_info = env.get_attr('get_performance_summary')[0]()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        trade_metrics.append(env_info)
    
    # Calcular métricas promedio
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_steps = sum(episode_lengths) / len(episode_lengths)
    
    # Análisis detallado de operaciones
    total_operations = len(all_operations)
    
    # Análisis de dirección
    long_trades = sum(1 for t in all_operations if t['direction'] == 'LONG')
    short_trades = sum(1 for t in all_operations if t['direction'] == 'SHORT')
    
    # Análisis de cierres
    tp_hits = sum(1 for t in all_operations if t['close_type'] == 'TAKE PROFIT')
    sl_hits = sum(1 for t in all_operations if t['close_type'] == 'STOP LOSS')
    manual_close = sum(1 for t in all_operations if t['close_type'] == 'NORMAL')
    
    # Calcular rentabilidad
    avg_pnl = sum(t['pnl'] for t in all_operations) / total_operations if total_operations > 0 else 0
    max_profit = max([t['pnl'] for t in all_operations]) if all_operations else 0
    max_loss = min([t['pnl'] for t in all_operations]) if all_operations else 0
    
    # Combinar métricas de trading
    combined_metrics = {
        "avg_reward": float(avg_reward),
        "avg_steps": float(avg_steps),
        "win_rate": sum(m['win_rate'] for m in trade_metrics) / len(trade_metrics) * 100,
        "profit_factor": sum(m['profit_factor'] for m in trade_metrics) / len(trade_metrics),
        "total_trades": sum(m['total_trades'] for m in trade_metrics),
        "return_pct": sum(m['return_pct'] for m in trade_metrics) / len(trade_metrics),
        "max_drawdown": sum(m['max_drawdown'] for m in trade_metrics) / len(trade_metrics),
        
        # Añadir métricas detalladas
        "long_trades": long_trades,
        "short_trades": short_trades,
        "long_pct": (long_trades / max(1, total_operations) * 100),
        "short_pct": (short_trades / max(1, total_operations) * 100),
        "direction_bias": 'LONG' if long_trades > short_trades else 'SHORT' if short_trades > long_trades else 'NEUTRAL',
        "take_profit_hits": tp_hits,
        "stop_loss_hits": sl_hits,
        "manual_close": manual_close,
        "tp_pct": (tp_hits / max(1, total_operations) * 100),
        "sl_pct": (sl_hits / max(1, total_operations) * 100),
        "avg_pnl_per_trade": float(avg_pnl),
        "max_profit_trade": float(max_profit),
        "max_loss_trade": float(max_loss)
    }
    
    # Imprimir resumen para diagnóstico
    logger.info("\n=== ANÁLISIS DETALLADO DE OPERACIONES ===")
    logger.info(f"Total operaciones: {total_operations}")
    logger.info(f"Operaciones LONG: {long_trades} ({combined_metrics['long_pct']:.1f}%)")
    logger.info(f"Operaciones SHORT: {short_trades} ({combined_metrics['short_pct']:.1f}%)")
    logger.info(f"Cierres por Take Profit: {tp_hits} ({combined_metrics['tp_pct']:.1f}%)")
    logger.info(f"Cierres por Stop Loss: {sl_hits} ({combined_metrics['sl_pct']:.1f}%)")
    logger.info(f"Cierres manuales: {manual_close} ({manual_close/max(1, total_operations)*100:.1f}%)")
    
    return combined_metrics

if __name__ == "__main__":
    sys.exit(main())
