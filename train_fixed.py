#!/usr/bin/env python
"""
Script mejorado para entrenar el modelo TradeEvolvePPO.
Soluciona los problemas identificados con el entrenamiento original
usando los componentes robustos desarrollados para diagnosticar y corregir
el problema del agente demasiado conservador.
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

# Importar los componentes mejorados
from training.fix_callback import get_robust_callback

# Verificación de CUDA
cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

# Nombre del modelo
MODEL_NAME = "FixedTradeEvolvePPO"

# Configuración de logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results/{MODEL_NAME}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

# Configurar logger
logger = setup_logger(
    name=MODEL_NAME,
    log_file=f'{output_dir}/logs/main.log',
    level="INFO",
    console_level="INFO",  # Mostrar información relevante
    file_level="DEBUG"     # Guardar detalles en archivo
)

# Silenciar loggers externos para reducir ruido
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('stable_baselines3').setLevel(logging.WARNING)
logging.getLogger('gymnasium').setLevel(logging.WARNING)

def modify_reward_config(reward_config, behavior_profile="balanced"):
    """
    Modifica la configuración de recompensas según el perfil de comportamiento deseado.
    
    Args:
        reward_config (dict): Configuración original de recompensas
        behavior_profile (str): Perfil de comportamiento deseado:
            - "balanced": Balance entre exploración y explotación (default)
            - "aggressive": Mayor incentivo para operaciones
            - "conservative": Mayor énfasis en gestión de riesgo
            - "exploration": Máxima exploración, menos énfasis en resultados
    
    Returns:
        dict: Configuración de recompensas modificada
    """
    config = reward_config.copy()
    
    if behavior_profile == "balanced":
        # Configuración balanceada
        config.update({
            'base_reward': -0.03,             # Penalización modesta por cada paso
            'pnl_weight': 2.0,                # Peso moderado para PnL
            'drawdown_weight': 0.1,           # Peso moderado para drawdown
            'trade_completion_bonus': 6.0,    # Bonus significativo por completar operaciones
            'inactivity_weight': 0.8,         # Penalización moderada por inactividad
        })
    
    elif behavior_profile == "aggressive":
        # Configuración agresiva para fomentar trading
        config.update({
            'base_reward': -0.05,             # Mayor penalización base para forzar acción
            'pnl_weight': 1.0,                # Menor peso en PnL para favorecer actividad
            'drawdown_weight': 0.05,          # Baja penalización por drawdown
            'trade_completion_bonus': 10.0,   # Bonus muy alto por completar operaciones
            'inactivity_weight': 2.0,         # Alta penalización por inactividad
            'scale_factor': 15.0,             # Mayor escala para amplificar recompensas
        })
    
    elif behavior_profile == "conservative":
        # Configuración conservadora con énfasis en gestión de riesgo
        config.update({
            'base_reward': -0.01,             # Baja penalización por cada paso
            'pnl_weight': 3.0,                # Alto énfasis en PnL
            'drawdown_weight': 0.2,           # Mayor penalización por drawdown
            'trade_completion_bonus': 3.0,    # Bonus moderado por completar operaciones
            'inactivity_weight': 0.3,         # Baja penalización por inactividad
        })
    
    elif behavior_profile == "exploration":
        # Máxima exploración, especialmente útil para entrenamiento inicial
        config.update({
            'base_reward': -0.08,             # Alta penalización base para forzar acción
            'pnl_weight': 0.5,                # Mínimo énfasis en PnL
            'drawdown_weight': 0.03,          # Mínima penalización por drawdown
            'trade_completion_bonus': 15.0,   # Bonus extremo por completar operaciones
            'direction_change_bonus': 1.0,    # Alto bonus por cambiar de dirección
            'inactivity_weight': 5.0,         # Penalización extrema por inactividad
        })
    
    return config

def modify_ppo_config(ppo_config, exploration_level="medium"):
    """
    Modifica la configuración del agente PPO para aumentar/disminuir exploración.
    
    Args:
        ppo_config (dict): Configuración original de PPO
        exploration_level (str): Nivel de exploración deseado:
            - "low": Baja exploración, máxima explotación
            - "medium": Balance entre exploración y explotación (default)
            - "high": Alta exploración para diversificar comportamiento
            - "extreme": Máxima exploración, útil para entrenamiento inicial
    
    Returns:
        dict: Configuración de PPO modificada
    """
    config = ppo_config.copy()
    
    if exploration_level == "low":
        config.update({
            'ent_coef': 0.01,
            'gamma': 0.99,  # Alto factor de descuento
            'learning_rate': 0.0001
        })
    
    elif exploration_level == "medium":
        config.update({
            'ent_coef': 0.05,
            'gamma': 0.95,
            'learning_rate': 0.0003
        })
    
    elif exploration_level == "high":
        config.update({
            'ent_coef': 0.2,  # Alto coeficiente de entropía
            'gamma': 0.90,  # Menor factor de descuento para priorizar recompensas inmediatas
            'learning_rate': 0.0005,
            'n_steps': 8192,  # Mayor número de pasos para mejor exploración
            'batch_size': 1024  # Mayor batch size para generalización
        })
    
    elif exploration_level == "extreme":
        config.update({
            'ent_coef': 0.5,  # Coeficiente de entropía extremadamente alto
            'gamma': 0.85,  # Bajo factor de descuento para máxima prioridad a recompensas inmediatas
            'learning_rate': 0.001,
            'n_steps': 16384,
            'batch_size': 2048,
            'clip_range': 0.3,  # Mayor clip range para permitir más cambios
        })
    
    # Asignar dispositivo (CUDA/CPU)
    if cuda_available:
        config['device'] = 'cuda'
    else:
        config['device'] = 'cpu'
    
    return config

def main():
    """Función principal para entrenar el modelo con configuración mejorada."""
    start_time = time.time()
    
    print(f"=== TradeEvolvePPO - Entrenamiento con componentes mejorados ===")
    print(f"Dispositivo: {device_name}")
    print(f"Directorio de salida: {output_dir}")
    
    # Verificar si se especificó un archivo de datos personalizado
    data_file = "data/NQ_06-25_combined_20250320_225417.csv"
    
    if len(sys.argv) > 1:
        custom_data = sys.argv[1]
        if os.path.exists(custom_data):
            data_file = custom_data
            print(f"Usando archivo de datos personalizado: {data_file}")
        else:
            print(f"Archivo no encontrado: {custom_data}, usando el predeterminado")
    
    # Cargar datos
    print("Cargando datos de entrenamiento...")
    data_loader = DataLoader(DATA_CONFIG)
    train_data, val_data, test_data = data_loader.prepare_data(data_file)
    print(f"Datos cargados: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Configurar parámetros de entrenamiento para corregir el problema de agente demasiado conservador
    training_timesteps = 4000000  # 4 millones de pasos
    
    # Modificar las configuraciones
    reward_config = modify_reward_config(REWARD_CONFIG, behavior_profile="aggressive")
    ppo_config = modify_ppo_config(PPO_CONFIG, exploration_level="high")
    
    # Actualizar rutas y configuración de entrenamiento
    training_config = TRAINING_CONFIG.copy()
    training_config.update({
        'save_path': os.path.join(output_dir, 'models'),
        'log_path': os.path.join(output_dir, 'logs'),
        'total_timesteps': training_timesteps,
        'progressive_steps': [400000, 1200000, 2400000, 3600000],  # Etapas de curriculum learning adaptadas para 4M de pasos
        'eval_freq': 10000,  # Evaluación más frecuente
        'checkpoint_freq': 25000  # Guardar modelos más frecuentemente
    })
    
    # Crear entorno de entrenamiento con configuración mejorada
    env_config = ENV_CONFIG.copy()
    env_config.update({
        'log_reward_components': True,  # Activar logging de componentes de recompensa
        'inactivity_threshold': 25,  # Reducir umbral de inactividad (era 100)
        'reward_config': reward_config  # Asignar configuración de recompensas modificada
    })
    
    print("\n=== Configuración de entrenamiento ===")
    print(f"Pasos totales: {training_timesteps}")
    print(f"Entropía: {ppo_config.get('ent_coef')}")
    print(f"Penalización por inactividad: {reward_config.get('inactivity_weight')}")
    print(f"Bonus por operación: {reward_config.get('trade_completion_bonus')}")
    
    try:
        # Crear entornos
        train_env = TradingEnv(
            data=train_data,
            config=env_config,
            window_size=env_config.get('window_size', 60),
            mode='train'
        )
        
        val_env = TradingEnv(
            data=val_data,
            config=env_config,
            window_size=env_config.get('window_size', 60),
            mode='validation'
        )
        
        # Crear configuración completa
        config = {
            'data_config': DATA_CONFIG,
            'env_config': env_config,
            'reward_config': reward_config,
            'ppo_config': ppo_config,
            'training_config': training_config
        }
        
        # Inicializar agente
        agent = PPOAgent(env=train_env, config=config)
        
        # Crear callback robusto
        callback = get_robust_callback(config, eval_env=val_env)
        
        print("\n=== Iniciando entrenamiento ===")
        agent.train(total_timesteps=training_timesteps, callback=callback)
        
        # Guardar modelo final
        final_model_path = os.path.join(output_dir, 'models', 'final_model')
        agent.save(final_model_path)
        
        # Evaluar modelo final
        print("\n=== Evaluando modelo final ===")
        test_env = TradingEnv(
            data=test_data,
            config=env_config,
            window_size=env_config.get('window_size', 60),
            mode='test'
        )
        
        # Importar funciones de evaluación
        from stable_baselines3.common.evaluation import evaluate_policy
        
        mean_reward, std_reward = evaluate_policy(
            agent.model,
            test_env,
            n_eval_episodes=5,
            deterministic=True
        )
        
        # Obtener métricas detalladas
        metrics = test_env.get_performance_summary()
        
        # Mostrar resultados
        print("\n=== RESULTADOS FINALES ===")
        print(f"Recompensa media: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Operaciones totales: {metrics.get('total_trades', 0)}")
        print(f"Win Rate: {metrics.get('win_rate', 0) * 100:.1f}%")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        
        # Guardar métricas
        import json
        metrics_file = os.path.join(output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump({
                'reward': float(mean_reward),
                'reward_std': float(std_reward),
                'trades': metrics.get('total_trades', 0),
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'training_time': time.time() - start_time
            }, f, indent=4)
        
        # Generar visualizaciones
        try:
            import matplotlib.pyplot as plt
            from visualization.visualizer import plot_equity_curve, plot_drawdown
            
            # Generar gráficos básicos
            plot_equity_curve(test_env, save_path=os.path.join(output_dir, 'plots', 'equity_curve.png'))
            plot_drawdown(test_env, save_path=os.path.join(output_dir, 'plots', 'drawdown.png'))
            
            # Mostrar distribución de acciones
            if hasattr(callback, 'callbacks') and len(callback.callbacks) > 0:
                for cb in callback.callbacks:
                    if hasattr(cb, 'report_distribution'):
                        dist = cb.report_distribution()
                        
                        # Gráfico de distribución de acciones
                        plt.figure(figsize=(10, 6))
                        plt.bar(['Hold', 'Buy', 'Sell'], 
                                [dist.get('hold_pct', 0), dist.get('buy_pct', 0), dist.get('sell_pct', 0)],
                                color=['gray', 'green', 'red'])
                        plt.title('Distribución de Acciones')
                        plt.xlabel('Acción')
                        plt.ylabel('Porcentaje (%)')
                        plt.ylim(0, 100)
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        plt.savefig(os.path.join(output_dir, 'plots', 'action_distribution.png'))
                        plt.close()
                        
                        print(f"\nDistribución de acciones:")
                        print(f"  Hold: {dist.get('hold_pct', 0):.1f}%")
                        print(f"  Buy: {dist.get('buy_pct', 0):.1f}%")
                        print(f"  Sell: {dist.get('sell_pct', 0):.1f}%")
                        break
            
        except Exception as e:
            print(f"Error al generar visualizaciones: {e}")
        
        # Finalizar
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nEntrenamiento y evaluación completados en {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Resultados guardados en: {output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario")
        return 1
        
    except Exception as e:
        import traceback
        print(f"Error durante el entrenamiento: {e}")
        print(traceback.format_exc())
        return 2

if __name__ == "__main__":
    sys.exit(main())
