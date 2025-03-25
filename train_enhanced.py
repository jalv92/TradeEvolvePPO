#!/usr/bin/env python
"""
Script de entrenamiento mejorado con monitoreo activo del comportamiento de trading.
"""

import os
import sys
import time
import logging
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Importar componentes del proyecto
from config.config import DATA_CONFIG, ENV_CONFIG, REWARD_CONFIG, PPO_CONFIG, TRAINING_CONFIG
from data.data_loader import DataLoader
from environment.trading_env import TradingEnv
from agents.ppo_agent import PPOAgent
from utils.logger import setup_logger
from stable_baselines3.common.callbacks import BaseCallback

# Verificación de CUDA
cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

# Nombre del modelo
MODEL_NAME = "ActiveTradeEvolvePPO"

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
    console_level="INFO",
    file_level="DEBUG"
)

# Crear callback personalizado para monitorear comportamiento de trading
class TradingMonitorCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=5000, verbose=1):
        super(TradingMonitorCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.trading_stats = {
            'timesteps': [],
            'trades': [],
            'win_rate': [],
            'actions': {'hold': 0, 'buy': 0, 'sell': 0}
        }
        self.no_trade_counter = 0
        self.max_no_trade = 3  # Máximo número de evaluaciones consecutivas sin operaciones

    def _on_step(self):
        # Monitorear acciones
        if self.locals.get('actions') is not None:
            action = self.locals['actions'][0]
            if isinstance(action, np.ndarray):
                if action[0] > 0.3:
                    self.trading_stats['actions']['buy'] += 1
                elif action[0] < -0.3:
                    self.trading_stats['actions']['sell'] += 1
                else:
                    self.trading_stats['actions']['hold'] += 1
            else:
                if action == 1:
                    self.trading_stats['actions']['buy'] += 1
                elif action == 2:
                    self.trading_stats['actions']['sell'] += 1
                else:
                    self.trading_stats['actions']['hold'] += 1

        # Evaluar periódicamente
        if self.n_calls % self.eval_freq == 0:
            from stable_baselines3.common.evaluation import evaluate_policy
            
            # Reiniciar entorno
            self.eval_env.reset()
            
            # Evaluar modelo
            mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=2)
            
            # Obtener métricas de trading
            metrics = self.eval_env.get_performance_summary()
            trades = metrics.get('total_trades', 0)
            win_rate = metrics.get('win_rate', 0) * 100
            
            # Guardar estadísticas
            self.trading_stats['timesteps'].append(self.n_calls)
            self.trading_stats['trades'].append(trades)
            self.trading_stats['win_rate'].append(win_rate)
            
            # Verificar comportamiento sin operaciones
            if trades == 0:
                self.no_trade_counter += 1
                if self.no_trade_counter >= self.max_no_trade:
                    print("\n⚠️ ADVERTENCIA: ¡No se detectaron operaciones en evaluaciones consecutivas!")
                    print("Ajustando parámetros de exploración para forzar más operaciones...")
                    
                    # Aumento temporal de exploración
                    if hasattr(self.model, 'ent_coef'):
                        original_ent_coef = self.model.ent_coef
                        self.model.ent_coef = max(self.model.ent_coef * 2, 0.5)
                        print(f"Coeficiente de entropía aumentado de {original_ent_coef} a {self.model.ent_coef}")
                    
                    # Reiniciar contador
                    self.no_trade_counter = 0
            else:
                self.no_trade_counter = 0
            
            # Imprimir estadísticas
            total_actions = sum(self.trading_stats['actions'].values())
            action_distribution = {
                k: f"{v/total_actions*100:.1f}%" for k, v in self.trading_stats['actions'].items()
            } if total_actions > 0 else {k: "0%" for k in self.trading_stats['actions']}
            
            print(f"\nPaso {self.n_calls}: Recompensa={mean_reward:.1f}, Operaciones={trades}, Tasa de Victorias={win_rate:.1f}%")
            print(f"Distribución de Acciones: Hold={action_distribution['hold']}, Compra={action_distribution['buy']}, Venta={action_distribution['sell']}")
        
        return True

def main():
    """Función principal para entrenar el modelo con monitoreo mejorado."""
    # Iniciar cronometraje
    start_time = time.time()
    
    print(f"=== Entrenamiento Mejorado para {MODEL_NAME} en {device_name} ===")
    
    # Aplicar configuraciones mejoradas
    # 1. Modificar configuración de recompensas
    reward_config = REWARD_CONFIG.copy()
    reward_config.update({
        'base_reward': -0.002,
        'pnl_weight': 2.5,
        'risk_weight': 0.1,
        'drawdown_weight': 0.01,
        'inactivity_weight': 2.0,
        'trade_completion_bonus': 8.0,
        'direction_change_bonus': 0.5,
        'scale_factor': 5.0
    })
    
    # 2. Modificar configuración PPO para exploración
    ppo_config = PPO_CONFIG.copy()
    ppo_config.update({
        'learning_rate': 0.0003,
        'gamma': 0.95,
        'ent_coef': 0.3,
        'exploration_config': {
            'exploration_steps': 1000000,
            'exploration_prob': 0.5,
            'inactivity_threshold': 20
        }
    })
    
    # Establecer dispositivo para CUDA
    if cuda_available:
        ppo_config['device'] = 'cuda'
    
    # 3. Modificar configuración del entorno
    env_config = ENV_CONFIG.copy()
    env_config.update({
        'inactivity_threshold': 20,
        'trivial_trade_threshold': 5.0,
        'force_action_prob': 0.95
    })
    
    # 4. Actualizar configuración de entrenamiento
    training_config = TRAINING_CONFIG.copy()
    training_config.update({
        'save_path': os.path.join(output_dir, 'models'),
        'log_path': os.path.join(output_dir, 'logs'),
        'total_timesteps': 250000,  # Reducido de 2M a 250K pasos para prueba
        'progressive_steps': [50000, 100000, 150000, 200000],  # Ajustado para 250K pasos
        'eval_freq': 5000
    })
    
    # Cargar datos
    print("Cargando datos...")
    data_file = "data/NQ_06-25_combined_20250320_225417.csv"
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        data_file = sys.argv[1]
    
    data_loader = DataLoader(DATA_CONFIG)
    train_data, val_data, test_data = data_loader.prepare_data(data_file)
    print(f"Datos cargados: Entrenamiento={len(train_data)}, Validación={len(val_data)}, Prueba={len(test_data)}")
    
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
        
        # Guardar la configuración mejorada
        import json
        with open(os.path.join(output_dir, 'enhanced_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        # Inicializar agente
        agent = PPOAgent(env=train_env, config=config)
        
        # Crear callback personalizado
        monitor_callback = TradingMonitorCallback(
            eval_env=val_env, 
            eval_freq=training_config['eval_freq']
        )
        
        # Añadir callbacks estándar
        from training.callback import TradeCallback
        trade_callback = TradeCallback(
            log_dir=training_config['log_path'],
            save_path=training_config['save_path'],
            save_interval=20000,
            eval_interval=training_config['eval_freq'],
            eval_env=val_env,
            eval_episodes=3,
            verbose=0  # Verbosidad mínima para salida más limpia
        )
        
        # Combinar callbacks
        from stable_baselines3.common.callbacks import CallbackList
        callback = CallbackList([monitor_callback, trade_callback])
        
        print("\n=== Configuración de Entrenamiento ===")
        print(f"Pasos Totales: {training_config['total_timesteps']}")
        print(f"Coeficiente de Entropía: {ppo_config['ent_coef']}")
        print(f"Recompensa Base: {reward_config['base_reward']}")
        print(f"Peso de Inactividad: {reward_config['inactivity_weight']}")
        print(f"Bonus por Completar Operación: {reward_config['trade_completion_bonus']}")
        print("=" * 30)
        
        print("\n=== Iniciando Entrenamiento Mejorado ===")
        agent.train(total_timesteps=training_config['total_timesteps'], callback=callback)
        
        # Guardar modelo final
        final_model_path = os.path.join(output_dir, 'models', 'final_model')
        agent.save(final_model_path)
        
        # Evaluar modelo final
        print("\n=== Evaluación Final ===")
        test_env = TradingEnv(
            data=test_data,
            config=env_config,
            window_size=env_config.get('window_size', 60),
            mode='test'
        )
        
        from stable_baselines3.common.evaluation import evaluate_policy
        mean_reward, std_reward = evaluate_policy(
            agent.model, test_env, n_eval_episodes=5, deterministic=True
        )
        
        # Obtener métricas finales
        metrics = test_env.get_performance_summary()
        
        # Imprimir resultados
        print("\n=== RESULTADOS FINALES ===")
        print(f"Recompensa: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Operaciones Totales: {metrics.get('total_trades', 0)}")
        print(f"Tasa de Victorias: {metrics.get('win_rate', 0) * 100:.1f}%")
        print(f"Factor de Rentabilidad: {metrics.get('profit_factor', 0):.2f}")
        print(f"Drawdown Máximo: {metrics.get('max_drawdown', 0) * 100:.1f}%")
        
        # Tiempo de entrenamiento
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nEntrenamiento completado en {int(hours)}h {int(minutes)}m {int(seconds)}s")
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