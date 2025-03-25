#!/usr/bin/env python
"""
Script para realizar una prueba rápida de entrenamiento con 10,000 pasos
y verificar que el entorno de trading ejecuta operaciones correctamente.
"""

import os
import sys
import time
import logging
import torch
import numpy as np
from datetime import datetime

# Importar componentes del proyecto
from config.config import DATA_CONFIG, ENV_CONFIG, REWARD_CONFIG, PPO_CONFIG, TRAINING_CONFIG
from data.data_loader import DataLoader
from environment.trading_env import TradingEnv
from agents.ppo_agent import PPOAgent
from utils.logger import setup_logger

# Importar vectorización y normalización de entornos
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Verificación de CUDA
cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

# Nombre del modelo
MODEL_NAME = "TestTradeEvolvePPO"

# Configuración de logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results/{MODEL_NAME}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)

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

# Nueva función para actualizar configuración a lo largo del entrenamiento (curriculum learning)
def create_config_for_step(base_config, current_step, total_steps):
    """
    Actualiza dinámicamente la configuración según la etapa de entrenamiento.
    
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
    
    # Fase 1 (0-25%): Alta exploración, muchas operaciones
    # Fase 2 (25-50%): Transición gradual
    # Fase 3 (50-100%): Consolidación de estrategia
    
    if progress < 0.25:
        # Fase de exploración: sin penalización por sobretrading
        config['hold_reward'] = 0.0
        config['overtrade_penalty'] = -1.0  # Penalización leve
        config['force_action_prob'] = 0.1  # 10% de probabilidad de forzar acción
    elif progress < 0.5:
        # Fase de transición: incentivo gradual para operaciones sostenibles
        config['hold_reward'] = 0.5 * ((progress - 0.25) / 0.25)
        config['overtrade_penalty'] = -2.0  # Penalización media
        config['force_action_prob'] = 0.05  # 5% de probabilidad
    else:
        # Fase de consolidación: mayor incentivo para mantener posiciones rentables
        config['hold_reward'] = 0.5
        config['overtrade_penalty'] = -3.0  # Penalización alta
        config['force_action_prob'] = 0.01  # 1% de probabilidad
    
    # Ajustar retraso de recompensa según progreso (más retraso en fases avanzadas)
    config['reward_delay_steps'] = int(3 + progress * 7)  # Entre 3 y 10 pasos
    
    return config

# Función para calcular tasa de entropía decreciente
def entropy_schedule(current_step, total_steps, initial=0.3, final=0.02):
    """Calcula la tasa de entropía decreciente a lo largo del entrenamiento."""
    progress = current_step / total_steps
    return max(initial * (1 - progress), final)

# Callback personalizado para la sincronización de estadísticas y curriculum learning
class CustomTrainingCallback(BaseCallback):
    def __init__(self, eval_env, train_env, base_env_config, total_timesteps, n_eval_episodes=2, eval_freq=1000, log_path=None, verbose=1):
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
        
    def _on_step(self):
        # 1. Actualizar configuración basada en curriculum learning
        current_config = create_config_for_step(
            self.base_env_config, 
            self.n_calls, 
            self.total_timesteps
        )
        
        # 2. Actualizar la tasa de entropía
        new_entropy = entropy_schedule(self.n_calls, self.total_timesteps)
        if hasattr(self.model, 'ent_coef') and self.n_calls % 1000 == 0:
            self.model.ent_coef = new_entropy
            if self.verbose > 0:
                print(f"Paso {self.n_calls}: Entropía actualizada a {new_entropy:.4f}")
        
        # 3. Evaluación periódica
        if self.n_calls - self.last_eval_step < self.eval_freq:
            return True
            
        self.last_eval_step = self.n_calls
        
        # Sincronizar manualmente las estadísticas
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms
        
        # Evaluar el modelo actual
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True
        )
        
        # Guardar mejor modelo
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.log_path is not None:
                self.model.save(os.path.join(self.log_path, 'best_model'))
            
        print(f"Evaluación en {self.n_calls} pasos: {mean_reward:.2f} ± {std_reward:.2f}")
        
        return True

def main():
    """Función principal para ejecutar la prueba de entrenamiento."""
    start_time = time.time()
    
    print(f"=== TradeEvolvePPO - Prueba de entrenamiento con aprendizaje progresivo ===")
    print(f"Dispositivo: {device_name}")
    print(f"Directorio de salida: {output_dir}")
    
    # Usar archivo de datos estándar
    data_file = "data/NQ_06-25_combined_20250320_225417.csv"
    
    # Cargar datos
    print("Cargando datos de entrenamiento...")
    data_loader = DataLoader(DATA_CONFIG)
    train_data, val_data, test_data = data_loader.prepare_data(data_file)
    print(f"Datos cargados: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Configurar parámetros para la prueba
    training_timesteps = 50000  # Aumentado a 50,000 pasos para permitir aprendizaje más completo
    
    # Configuración agresiva para fomentar operaciones
    reward_config = REWARD_CONFIG.copy()
    reward_config.update({
        'base_reward': -0.05,             # Penalización base para forzar acción
        'pnl_weight': 1.5,                # Peso para PnL
        'drawdown_weight': 0.05,          # Baja penalización por drawdown
        'trade_completion_bonus': 12.0,   # Bonus alto por completar operaciones
        'inactivity_weight': 3.0,         # Alta penalización por inactividad
        'scale_factor': 10.0,             # Factor de escala para recompensas
    })
    
    # Configuración de PPO para alta exploración inicial y aprendizaje eficiente
    ppo_config = PPO_CONFIG.copy()
    ppo_config.update({
        'ent_coef': 0.3,           # Alto coeficiente de entropía inicial (será decreciente)
        'gamma': 0.95,             # Mayor factor de descuento para valorar recompensas futuras
        'learning_rate': 0.0005,   # Tasa de aprendizaje inicial
        'n_steps': 2048,           # Mayor tamaño de buffer para capturar patrones a largo plazo
        'batch_size': 256,         # Mayor batch size para generalización
        'n_epochs': 15,            # Más épocas por cada batch para aprovechar los datos
        'gae_lambda': 0.97,        # Mayor lambda para mejor estimación de ventaja
        'device': 'cpu'            # Usar CPU para evitar problemas de compatibilidad
    })
    
    # Configuración de entrenamiento
    training_config = TRAINING_CONFIG.copy()
    training_config.update({
        'save_path': os.path.join(output_dir, 'models'),
        'log_path': os.path.join(output_dir, 'logs'),
        'total_timesteps': training_timesteps,
        'eval_freq': 5000,  # Evaluación cada 5000 pasos
    })
    
    # Configuración del entorno con parámetros adicionales para curriculum learning
    env_config = ENV_CONFIG.copy()
    env_config.update({
        'log_reward_components': True,  
        'inactivity_threshold': 20,      # Umbral de inactividad bajo
        'hold_reward': 0.0,              # Inicialmente 0, aumentará con el curriculum learning
        'overtrade_penalty': -1.0,       # Inicialmente bajo, aumentará con el curriculum
        'reward_delay_steps': 3,         # Inicialmente bajo, aumentará con el curriculum
        'force_action_prob': 0.1,        # 10% probabilidad inicial de forzar acciones
        
        # NUEVOS PARÁMETROS ANTI-SOBRETRADING
        'min_hold_steps': 5,             # Duración mínima recomendada para una operación
        'position_cooldown': 10,         # Tiempo de enfriamiento obligatorio entre operaciones
        
        'reward_config': reward_config   # Asignar configuración de recompensas
    })
    
    print("\n=== Configuración de la prueba ===")
    print(f"Pasos totales: {training_timesteps}")
    print(f"Entropía inicial: {ppo_config.get('ent_coef')}")
    print(f"Penalización por inactividad: {reward_config.get('inactivity_weight')}")
    print(f"Bonus por operación: {reward_config.get('trade_completion_bonus')}")
    print(f"Aprendizaje progresivo: Sí (3 fases)")
    
    try:
        # 1. Crear entornos base
        def make_train_env():
            env = TradingEnv(
                data=train_data,
                config=env_config,
                window_size=env_config.get('window_size', 60),
                mode='train'
            )
            return env
        
        def make_val_env():
            env = TradingEnv(
                data=val_data,
                config=env_config,
                window_size=env_config.get('window_size', 60),
                mode='validation'
            )
            return env
        
        # 2. Vectorizar los entornos
        train_vec_env = DummyVecEnv([lambda: Monitor(make_train_env())])
        eval_vec_env = DummyVecEnv([lambda: Monitor(make_val_env())])
        
        # 3. Aplicar normalización a ambos entornos
        train_env = VecNormalize(
            train_vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        
        eval_env = VecNormalize(
            eval_vec_env,
            norm_obs=True,
            norm_reward=False,  # No normalizar recompensas para evaluación
            clip_obs=10.0,
            clip_reward=10.0,
            training=False  # No actualizar estadísticas durante evaluación
        )
        
        # Asegurar que eval_env use las mismas estadísticas de normalización que train_env
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms
        
        # Crear configuración completa
        config = {
            'data_config': DATA_CONFIG,
            'env_config': env_config,
            'reward_config': reward_config,
            'ppo_config': ppo_config,
            'training_config': training_config
        }
        
        # Inicializar agente con el entorno vectorizado
        agent = PPOAgent(env=train_env, config=config)
        
        # Callback personalizado para curriculum learning y evaluación
        custom_callback = CustomTrainingCallback(
            eval_env=eval_env,
            train_env=train_env,
            base_env_config=env_config,
            total_timesteps=training_timesteps,
            n_eval_episodes=2,
            eval_freq=5000,
            log_path=os.path.join(output_dir, 'models')
        )
        
        print("\n=== Iniciando prueba de entrenamiento con curriculum learning ===")
        print("Fase 1 (0-25%): Alta exploración, muchas operaciones")
        print("Fase 2 (25-50%): Transición gradual")
        print("Fase 3 (50-100%): Consolidación de estrategia de largo plazo")
        
        try:
            agent.train(total_timesteps=training_timesteps, callback=custom_callback)
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
            import traceback
            print(traceback.format_exc())
        
        # Evaluar modelo final
        print("\n=== Evaluando modelo final ===")
        
        # Crear entorno de prueba
        def make_test_env():
            env = TradingEnv(
                data=test_data,
                config=env_config,
                window_size=env_config.get('window_size', 60),
                mode='test'
            )
            return env
        
        test_vec_env = DummyVecEnv([lambda: Monitor(make_test_env())])
        test_env = VecNormalize(
            test_vec_env,
            norm_obs=True,
            norm_reward=False,
            training=False  # No actualizar estadísticas durante evaluación
        )
        test_env.obs_rms = train_env.obs_rms
        test_env.ret_rms = train_env.ret_rms
        
        try:
            # Obtener el entorno sin vectorizar para obtener métricas detalladas
            test_env_raw = make_test_env()
            
            # Ejecutar un mayor número de operaciones para probar la funcionalidad
            print("\n=== Realizando operaciones de prueba con el modelo entrenado ===")
            obs, _ = test_env_raw.reset()
            done = False
            steps = 0
            num_operations = 0
            max_steps = 2000  # Mayor número de pasos para ver más operaciones
            
            while not done and steps < max_steps:
                steps += 1
                action, _ = agent.model.predict(obs, deterministic=False)  # Usar modo no determinístico para más exploración
                obs, reward, done, _, info = test_env_raw.step(action)
                
                # Verificar si se ha realizado una operación (apertura o cierre)
                if len(test_env_raw.trades) > num_operations:
                    num_operations = len(test_env_raw.trades)
                    print(f"Paso {steps}: Nueva operación realizada (total: {num_operations})")
                
                # Si ya tenemos suficientes operaciones, podemos terminar
                if num_operations >= 20:
                    print(f"Se han registrado {num_operations} operaciones, terminando la prueba")
                    break
                
                # Mostrar progreso cada 500 pasos
                if steps % 500 == 0:
                    print(f"Completados {steps} pasos, {num_operations} operaciones registradas")
            
            print(f"\nPrueba completa: {steps} pasos ejecutados, {num_operations} operaciones registradas")
            
            # Obtener métricas detalladas
            metrics = test_env_raw.get_performance_summary()
            
            # Mostrar resultados
            print("\n=== RESULTADOS DE LA PRUEBA ===")
            print(f"Recompensa media: {metrics.get('mean_reward', 0):.2f}")
            print(f"Operaciones totales: {metrics.get('total_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0) * 100:.1f}%")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            
            # Mostrar detalles de las operaciones
            trade_history = test_env_raw.trades
            print(f"\nHistorial de operaciones ({len(trade_history)} operaciones):")
            for i, trade in enumerate(trade_history[-10:]):  # Mostrar las últimas 10 operaciones
                try:
                    # Obtener dirección (con manejo de errores para compatibilidad con versiones anteriores)
                    if 'direction' in trade:
                        direction = trade['direction']
                    else:
                        # Determinar dirección a partir de la posición
                        position = trade.get('position', 0)
                        direction = 'long' if position > 0 else 'short' if position < 0 else 'none'
                    
                    # Obtener precio de salida (con manejo de errores)
                    exit_price = trade.get('exit_price', None)
                    exit_str = f"{exit_price:.2f}" if exit_price is not None else "Abierta"
                    
                    # Obtener PnL (con manejo de errores)
                    pnl = trade.get('pnl', 0)
                    
                    # Calcular duración de la operación
                    entry_time = trade.get('entry_time', 0)
                    exit_time = trade.get('exit_time', steps if exit_price is None else None)
                    duration = '-' if exit_time is None else (exit_time - entry_time)
                    
                    # Mostrar información completa
                    print(f"  {i+1}. Dirección: {direction}, Entry: {trade['entry_price']:.2f}, Exit: {exit_str}, PnL: {pnl:.2f}, Duración: {duration}")
                except KeyError as e:
                    # En caso de error, mostrar información parcial
                    print(f"  {i+1}. Trade con formato incompleto, faltan campos: {e}")
                    print(f"       Datos disponibles: {list(trade.keys())}")
                except Exception as e:
                    print(f"  {i+1}. Error al procesar trade: {e}")
            
        except Exception as e:
            print(f"Error durante la evaluación: {e}")
            import traceback
            print(traceback.format_exc())
        
        # Tiempo total
        training_time = time.time() - start_time
        minutes, seconds = divmod(training_time, 60)
        
        print(f"\nPrueba completada en {int(minutes)}m {int(seconds)}s")
        print(f"Resultados guardados en: {output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nPrueba interrumpida por el usuario")
        return 1
        
    except Exception as e:
        import traceback
        print(f"Error durante la prueba: {e}")
        print(traceback.format_exc())
        return 2

if __name__ == "__main__":
    sys.exit(main()) 