#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de ejemplo para ejecutar un benchmark de modelos de trading.
Este script muestra cómo utilizar el sistema de benchmarking para comparar
diferentes configuraciones y arquitecturas.
"""

import os
import argparse
import pandas as pd
from datetime import datetime

from utils.benchmarking import ModelBenchmark
from data.data_loader import DataLoader
from config.config import BASE_CONFIG, ENV_CONFIG, PPO_CONFIG, REWARD_CONFIG
from utils.logger import setup_logger

# Configurar logger
logger = setup_logger('benchmark_example', log_file='logs/benchmark_example.log')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark de modelos de trading')
    
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Número de pasos por experimento (default: 100000)')
    
    parser.add_argument('--parallel', action='store_true',
                       help='Ejecutar experimentos en paralelo')
    
    parser.add_argument('--workers', type=int, default=2,
                       help='Número de workers para ejecución paralela (default: 2)')
    
    parser.add_argument('--output', type=str, default=None,
                       help='Directorio de salida para resultados')
    
    parser.add_argument('--quick', action='store_true',
                       help='Ejecutar grid reducido para prueba rápida')
    
    return parser.parse_args()

def main():
    """Función principal para ejecutar el benchmark."""
    # Procesar argumentos
    args = parse_args()
    
    # Configurar salida
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/benchmark_{timestamp}"
    else:
        output_dir = args.output
    
    logger.info(f"Iniciando benchmark con {args.timesteps} pasos por experimento")
    if args.parallel:
        logger.info(f"Modo: Paralelo con {args.workers} workers")
    else:
        logger.info("Modo: Secuencial")
    
    # Cargar datos
    logger.info("Cargando datos...")
    base_config = BASE_CONFIG.copy()
    data_loader = DataLoader(config=base_config)
    file_path = os.path.join('data', 'dataset', f"{base_config['symbol']}.csv")
    df = data_loader.load_csv_data(file_path)
    
    # Dividir datos
    logger.info(f"Dividiendo datos ({len(df)} filas)...")
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size+val_size].copy()
    
    logger.info(f"Conjunto de entrenamiento: {len(train_df)} filas")
    logger.info(f"Conjunto de validación: {len(val_df)} filas")
    
    # Configuración completa
    logger.info("Preparando configuración...")
    config = {
        'env_config': ENV_CONFIG.copy(),
        'ppo_config': PPO_CONFIG.copy(),
        'reward_config': REWARD_CONFIG.copy(),
    }
    
    # Añadir reward_config a env_config
    config['env_config']['reward_config'] = config['reward_config']
    
    # Crear benchmark
    logger.info(f"Inicializando benchmark en: {output_dir}")
    benchmark = ModelBenchmark(
        data_train=train_df,
        data_val=val_df,
        base_config=config,
        output_dir=output_dir
    )
    
    # Definir grid de parámetros a probar
    if args.quick:
        # Grid reducido para pruebas rápidas
        logger.info("Usando grid reducido para prueba rápida")
        param_grid = {
            'model_type': ['lstm', 'mlp'],  # Tipo de arquitectura
            'ppo_learning_rate': [0.0001],  # Tasa de aprendizaje
            'ppo_ent_coef': [0.01, 0.1],  # Coeficiente de entropía
            'env_min_hold_steps': [10],  # Pasos mínimos de mantenimiento
        }
    else:
        # Grid completo para benchmarking exhaustivo
        logger.info("Usando grid completo para benchmarking exhaustivo")
        param_grid = {
            'model_type': ['lstm', 'mlp'],  # Tipo de arquitectura
            'ppo_learning_rate': [0.0001, 0.0003, 0.0005],  # Tasa de aprendizaje
            'ppo_n_steps': [1024, 2048],  # Pasos por actualización
            'ppo_ent_coef': [0.01, 0.05, 0.1],  # Coeficiente de entropía
            'ppo_batch_size': [128, 256],  # Tamaño de batch
            'env_min_hold_steps': [5, 10, 20],  # Pasos mínimos de mantenimiento
            'env_position_cooldown': [10, 20],  # Tiempo de espera entre operaciones
        }
    
    # Generar experimentos
    experiments = benchmark.generate_experiment_grid(param_grid)
    logger.info(f"Se ejecutarán {len(experiments)} experimentos")
    
    # Ejecutar benchmark
    results = benchmark.run_benchmarks(
        timesteps=args.timesteps,
        parallel=args.parallel,
        max_workers=args.workers
    )
    
    # Obtener mejores modelos
    best_reward = benchmark.get_best_models(metric='mean_reward', top_n=3)
    best_profit = benchmark.get_best_models(metric='profit_factor', top_n=3)
    best_winrate = benchmark.get_best_models(metric='win_rate', top_n=3)
    
    # Mostrar resumen
    logger.info("\n=== RESUMEN DE RESULTADOS ===")
    logger.info(f"Total de experimentos: {len(experiments)}")
    logger.info(f"Pasos por experimento: {args.timesteps}")
    logger.info(f"Directorio de resultados: {output_dir}")
    
    # Mostrar mejores por recompensa
    logger.info("\nTop 3 modelos por recompensa media:")
    for i, (_, row) in enumerate(best_reward.iterrows()):
        logger.info(f"#{i+1}: Experimento {row['experiment_id']} - Reward: {row['mean_reward']:.2f}")
        logger.info(f"    Configuración: {row['config']}")
    
    # Mostrar mejores por profit factor
    logger.info("\nTop 3 modelos por profit factor:")
    for i, (_, row) in enumerate(best_profit.iterrows()):
        logger.info(f"#{i+1}: Experimento {row['experiment_id']} - Profit Factor: {row['profit_factor']:.2f}")
        logger.info(f"    Configuración: {row['config']}")
    
    # Mostrar mejores por win rate
    logger.info("\nTop 3 modelos por win rate:")
    for i, (_, row) in enumerate(best_winrate.iterrows()):
        logger.info(f"#{i+1}: Experimento {row['experiment_id']} - Win Rate: {row['win_rate']:.2f}%")
        logger.info(f"    Configuración: {row['config']}")
    
    logger.info("\nBenchmark completado con éxito. Las visualizaciones están disponibles en:")
    logger.info(f"{output_dir}/visualizations/")
    
    return benchmark, results


if __name__ == "__main__":
    benchmark, results = main()
    print("\nBenchmark completado con éxito. Consulte el archivo de log para detalles completos.") 