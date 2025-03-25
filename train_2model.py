#!/usr/bin/env python
"""
Script para iniciar el entrenamiento del modelo "2Model" con 4 millones de pasos.
Este entrenamiento utiliza la configuración optimizada con:
- Mayor exploración (entropy coefficient = 0.15)
- Penalización por inactividad progresiva
- Gestión dinámica de posiciones
- Sistema completo de recompensas para trading efectivo
"""

import os
import sys
import time
import logging
import torch
from datetime import datetime

# Importar componentes del proyecto
from config.config import DATA_CONFIG, ENV_CONFIG, REWARD_CONFIG, PPO_CONFIG, TRAINING_CONFIG
from data.data_loader import DataLoader
from training.trainer import Trainer
from utils.logger import setup_logger

# Verificación de CUDA
cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

# Nombre del modelo
MODEL_NAME = "4M_TradeEvolvePPO"

# Configuración de logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results/{MODEL_NAME}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

# Configuración minimalista para el logger - solo mostrar información crítica
logger = setup_logger(
    name=MODEL_NAME,
    log_file=f'{output_dir}/logs/main.log',
    level="WARNING",
    console_level="ERROR",  # Solo errores críticos en consola
    file_level="INFO"       # Mantener INFO en archivo para diagnóstico posterior
)

# Silenciar loggers externos que generan ruido
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('stable_baselines3').setLevel(logging.ERROR)
logging.getLogger('gymnasium').setLevel(logging.ERROR)

def main():
    """Función principal para iniciar el entrenamiento del modelo con 4M de pasos."""
    start_time = time.time()
    
    # Minimizar logs iniciales
    print(f"=== Iniciando entrenamiento {MODEL_NAME} con {device_name} ===")
    
    # Crear directorios de salida
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # Actualizar rutas y configuración de entrenamiento
    training_config = TRAINING_CONFIG.copy()
    training_config['save_path'] = os.path.join(output_dir, 'models')
    training_config['log_path'] = os.path.join(output_dir, 'logs')
    training_config['total_timesteps'] = 4000000  # 4 millones de pasos
    
    # Actualizar etapas progresivas para 4M de pasos
    training_config['progressive_steps'] = [400000, 1200000, 2400000, 3600000]
    
    # Actualizar configuración de PPO para CUDA
    ppo_config = PPO_CONFIG.copy()
    if cuda_available:
        ppo_config['device'] = 'cuda'
    
    # Reducir verbosidad en configuración
    ppo_config['verbose'] = 0
    
    # Cargar datos
    print("Cargando datos...")
    data_file = "data/NQ_06-25_combined_20250320_225417.csv"
    data_loader = DataLoader(DATA_CONFIG)
    train_data, val_data, test_data = data_loader.prepare_data(data_file)
    print(f"Datos cargados: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Crear entrenador
    config = {
        'data_config': DATA_CONFIG,
        'env_config': ENV_CONFIG,
        'reward_config': REWARD_CONFIG,
        'ppo_config': ppo_config,
        'training_config': training_config
    }
    
    trainer = Trainer(config)
    
    # Configurar pipeline de entrenamiento
    trainer.setup(data_file)
    
    print("="*50)
    print(f"INICIANDO ENTRENAMIENTO DE {training_config['total_timesteps']} PASOS")
    print(f"- PnL weight: {REWARD_CONFIG['pnl_weight']}")
    print(f"- Coeficiente de entropía: {ppo_config['ent_coef']}")
    print(f"- Learning rate: {ppo_config['learning_rate']}")
    print("="*50)
    
    try:
        # Solo mostrar mensajes críticos durante el entrenamiento
        results = trainer.train(show_progress=True)
        
        # Guardar resultados
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("="*50)
        print(f"Entrenamiento completado en {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Evaluar modelo final
        print("Evaluando modelo final...")
        eval_metrics = trainer.evaluate()
        print("="*50)
        print("RESULTADOS FINALES:")
        print(f"- Trades: {eval_metrics.get('total_trades', 0)}")
        print(f"- Win Rate: {eval_metrics.get('win_rate', 0)*100:.2f}%")
        print(f"- Profit Factor: {eval_metrics.get('profit_factor', 0):.2f}")
        print(f"- Reward: {eval_metrics.get('mean_reward', 0):.2f}")
        print("="*50)
        
        return 0
    
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario")
        # Intentar guardar el modelo actual
        try:
            trainer.save_model(os.path.join(output_dir, 'models', 'interrupted_model'))
            print("Modelo guardado antes de interrumpir")
        except Exception as e:
            print(f"No se pudo guardar el modelo interrumpido: {e}")
        return 1
    
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main()) 