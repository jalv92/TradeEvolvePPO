#!/usr/bin/env python
"""
Script de diagnóstico para detectar problemas en el entrenamiento de TradeEvolvePPO.

Este script realiza varias pruebas para identificar errores en:
1. Formato de datos y observaciones
2. Funcionamiento del entorno de trading
3. Sistema de recompensas
4. Callbacks de entrenamiento
5. Distribución de acciones del agente
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import torch
import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt

# Importar componentes del proyecto
from config.config import (
    DATA_CONFIG, ENV_CONFIG, REWARD_CONFIG, 
    PPO_CONFIG, TRAINING_CONFIG
)
from data.data_loader import DataLoader
from environment.trading_env import TradingEnv
from agents.ppo_agent import PPOAgent
from training.trainer import Trainer
from utils.logger import setup_logger
from training.callback import TradeCallback

# Configuración de logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/debug_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# Configurar logger
logger = setup_logger(
    name="debug_training",
    log_file=f"{log_dir}/debug.log",
    level="DEBUG",
    console_level="INFO",
    file_level="DEBUG"
)

def check_data_format(data_file: str) -> Dict[str, Any]:
    """
    Verifica el formato de los datos de entrada.
    
    Args:
        data_file (str): Ruta al archivo de datos
        
    Returns:
        Dict[str, Any]: Resultados de la verificación
    """
    logger.info(f"Verificando formato de datos: {data_file}")
    results = {}
    
    try:
        # Cargar datos
        if not os.path.exists(data_file):
            logger.error(f"Archivo de datos no encontrado: {data_file}")
            results["error"] = f"Archivo no encontrado: {data_file}"
            return results
        
        # Leer datos
        try:
            df = pd.read_csv(data_file)
            results["num_rows"] = len(df)
            results["num_columns"] = len(df.columns)
            results["columns"] = df.columns.tolist()
            
            # Verificar columnas mínimas necesarias
            required_columns = ["open", "high", "low", "close"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            results["missing_columns"] = missing_columns
            
            # Verificar valores NaN
            nan_counts = df.isna().sum()
            results["nan_counts"] = {col: count for col, count in nan_counts.items() if count > 0}
            
            # Verificar tipos de datos
            results["dtypes"] = {col: str(df[col].dtype) for col in df.columns}
            
            # Estadísticas básicas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            results["stats"] = {col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std())
            } for col in numeric_cols if col in ["open", "high", "low", "close", "volume"]}
            
            logger.info(f"Datos cargados correctamente: {results['num_rows']} filas, {results['num_columns']} columnas")
            results["success"] = True
            
        except Exception as e:
            logger.error(f"Error al leer datos: {e}")
            logger.error(traceback.format_exc())
            results["error"] = f"Error de lectura: {str(e)}"
            return results
        
        # Imprimir primeras 5 filas para verificación
        logger.info("Primeras filas del dataset:")
        logger.info(f"\n{df.head().to_string()}")
        
        # Verificar si el número de columnas coincide con lo esperado
        if results.get("num_columns", 0) < 25:
            logger.warning(f"El número de columnas ({results['num_columns']}) es menor que el esperado (25)")
            results["warning"] = f"Columnas insuficientes: {results['num_columns']}/25"
        
        return results
    
    except Exception as e:
        logger.error(f"Error en check_data_format: {e}")
        logger.error(traceback.format_exc())
        results["error"] = f"Error no controlado: {str(e)}"
        return results

def test_environment(data_file: str) -> Dict[str, Any]:
    """
    Prueba la funcionalidad básica del entorno de trading.
    
    Args:
        data_file (str): Ruta al archivo de datos
        
    Returns:
        Dict[str, Any]: Resultados de la prueba
    """
    logger.info("Probando entorno de trading")
    results = {}
    
    try:
        # Cargar datos
        data_loader = DataLoader(DATA_CONFIG)
        train_data, val_data, test_data = data_loader.prepare_data(data_file)
        
        # Crear entorno
        env = TradingEnv(
            data=train_data,
            config=ENV_CONFIG,
            window_size=ENV_CONFIG.get('window_size', 60),
            mode='train'
        )
        
        # Verificar espacios de observación y acción
        results["observation_space"] = {
            "shape": env.observation_space.shape,
            "low": float(env.observation_space.low.min()),
            "high": float(env.observation_space.high.max())
        }
        results["action_space"] = {
            "shape": env.action_space.shape if hasattr(env.action_space, 'shape') else None,
            "n": env.action_space.n if hasattr(env.action_space, 'n') else None,
            "low": env.action_space.low.tolist() if hasattr(env.action_space, 'low') else None,
            "high": env.action_space.high.tolist() if hasattr(env.action_space, 'high') else None
        }
        
        logger.info(f"Espacio de observación: {results['observation_space']}")
        logger.info(f"Espacio de acción: {results['action_space']}")
        
        # Resetear entorno y obtener observación inicial
        observation, info = env.reset()
        results["initial_observation"] = {
            "shape": observation.shape,
            "min": float(observation.min()),
            "max": float(observation.max()),
            "mean": float(observation.mean()),
            "std": float(observation.std()),
            "has_nan": bool(np.isnan(observation).any())
        }
        results["initial_info"] = info
        
        logger.info(f"Observación inicial: {results['initial_observation']}")
        
        # Probar acciones aleatorias
        actions_to_test = [0, 1, 2]  # Cerrar, comprar, vender
        action_results = []
        
        for action in actions_to_test:
            # Convertir acción a formato esperado por el entorno
            if isinstance(env.action_space, gym.spaces.Box):
                # Para espacios continuos, usar valores específicos
                if action == 0:  # Cerrar posición
                    formatted_action = np.array([0.0, 0.0], dtype=np.float32)
                elif action == 1:  # Comprar/Long
                    formatted_action = np.array([1.0, 0.0], dtype=np.float32)
                else:  # Vender/Short
                    formatted_action = np.array([-1.0, 0.0], dtype=np.float32)
            else:
                # Para espacios discretos, usar el valor directamente
                formatted_action = action
            
            # Ejecutar acción
            next_obs, reward, terminated, truncated, info = env.step(formatted_action)
            
            # Registrar resultados
            action_results.append({
                "action": int(action),
                "formatted_action": formatted_action.tolist() if isinstance(formatted_action, np.ndarray) else formatted_action,
                "reward": float(reward),
                "observation_shape": next_obs.shape,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": {k: (float(v) if isinstance(v, (int, float, np.number)) else v) 
                         for k, v in info.items() if k != 'action_mask'}
            })
            
            logger.info(f"Acción {action}: reward={reward}, terminated={terminated}")
            
            # Resetear después de cada acción para pruebas independientes
            if action != actions_to_test[-1]:
                observation, info = env.reset()
        
        results["action_tests"] = action_results
        
        # Verificar que el entorno funciona correctamente
        results["success"] = True
        logger.info("Prueba del entorno completada con éxito")
        
        return results
    
    except Exception as e:
        logger.error(f"Error en test_environment: {e}")
        logger.error(traceback.format_exc())
        results["error"] = f"Error no controlado: {str(e)}"
        return results

def test_reward_components(data_file: str) -> Dict[str, Any]:
    """
    Prueba los componentes de recompensa del entorno.
    
    Args:
        data_file (str): Ruta al archivo de datos
        
    Returns:
        Dict[str, Any]: Resultados de la prueba
    """
    logger.info("Probando componentes de recompensa")
    results = {}
    
    try:
        # Cargar datos
        data_loader = DataLoader(DATA_CONFIG)
        train_data, val_data, test_data = data_loader.prepare_data(data_file)
        
        # Copiar ENV_CONFIG para modificar
        env_config = ENV_CONFIG.copy()
        env_config['log_reward_components'] = True
        
        # Crear entorno
        env = TradingEnv(
            data=train_data,
            config=env_config,
            window_size=env_config.get('window_size', 60),
            mode='train'
        )
        
        # Resetear entorno
        observation, info = env.reset()
        
        # Simular una secuencia de acciones
        actions = [1, 0, 2, 0]  # Comprar, cerrar, vender, cerrar
        rewards = []
        components = []
        
        for action in actions:
            # Convertir acción a formato esperado por el entorno
            if isinstance(env.action_space, gym.spaces.Box):
                # Para espacios continuos, usar valores específicos
                if action == 0:  # Cerrar posición
                    formatted_action = np.array([0.0, 0.0], dtype=np.float32)
                elif action == 1:  # Comprar/Long
                    formatted_action = np.array([1.0, 0.0], dtype=np.float32)
                else:  # Vender/Short
                    formatted_action = np.array([-1.0, 0.0], dtype=np.float32)
            else:
                # Para espacios discretos, usar el valor directamente
                formatted_action = action
            
            # Ejecutar acción
            next_obs, reward, terminated, truncated, info = env.step(formatted_action)
            
            rewards.append(float(reward))
            
            # Obtener componentes de recompensa si están disponibles
            if hasattr(env, '_reward_components'):
                components.append(env._reward_components.copy())
            elif '_reward_components' in info:
                components.append(info['_reward_components'])
            else:
                components.append({'total': float(reward), 'unknown': float(reward)})
            
            logger.info(f"Acción {action}: reward={reward}")
            
            if terminated or truncated:
                break
        
        results["rewards"] = rewards
        results["components"] = components
        
        # Visualizar componentes
        if components:
            try:
                component_names = list(components[0].keys())
                for name in component_names:
                    plt.figure(figsize=(10, 5))
                    values = [comp.get(name, 0) for comp in components]
                    plt.plot(values, marker='o')
                    plt.title(f'Componente: {name}')
                    plt.xlabel('Paso')
                    plt.ylabel('Valor')
                    plt.grid(True)
                    plt.savefig(f"{log_dir}/reward_{name}.png")
                    plt.close()
                
                # Gráfico comparativo
                plt.figure(figsize=(12, 6))
                for name in component_names:
                    values = [comp.get(name, 0) for comp in components]
                    plt.plot(values, marker='o', label=name)
                plt.title('Comparación de Componentes de Recompensa')
                plt.xlabel('Paso')
                plt.ylabel('Valor')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{log_dir}/reward_components_comparison.png")
                plt.close()
            except Exception as plot_error:
                logger.error(f"Error al generar gráficos: {plot_error}")
        
        # Verificar que las recompensas son razonables
        results["success"] = True
        results["mean_reward"] = np.mean(rewards)
        results["std_reward"] = np.std(rewards)
        results["max_reward"] = np.max(rewards)
        results["min_reward"] = np.min(rewards)
        
        logger.info(f"Prueba de recompensas completada: media={results['mean_reward']}, desv={results['std_reward']}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error en test_reward_components: {e}")
        logger.error(traceback.format_exc())
        results["error"] = f"Error no controlado: {str(e)}"
        return results

def test_agent_initialization(data_file: str) -> Dict[str, Any]:
    """
    Prueba la inicialización y funcionamiento básico del agente PPO.
    
    Args:
        data_file (str): Ruta al archivo de datos
        
    Returns:
        Dict[str, Any]: Resultados de la prueba
    """
    logger.info("Probando inicialización del agente PPO")
    results = {}
    
    try:
        # Cargar datos
        data_loader = DataLoader(DATA_CONFIG)
        train_data, val_data, test_data = data_loader.prepare_data(data_file)
        
        # Crear entorno
        env = TradingEnv(
            data=train_data,
            config=ENV_CONFIG,
            window_size=ENV_CONFIG.get('window_size', 60),
            mode='train'
        )
        
        # Verificar disponibilidad de CUDA
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
        results["cuda"] = {
            "available": cuda_available,
            "device": device_name
        }
        logger.info(f"CUDA disponible: {cuda_available}, dispositivo: {device_name}")
        
        # Crear configuración para el agente
        config = {
            'data_config': DATA_CONFIG,
            'env_config': ENV_CONFIG,
            'reward_config': REWARD_CONFIG,
            'ppo_config': PPO_CONFIG,
            'training_config': TRAINING_CONFIG
        }
        
        # Inicializar agente
        agent = PPOAgent(env=env, config=config)
        
        # Verificar que el modelo se creó correctamente
        results["model_created"] = agent.model is not None
        
        # Probar predicción con observación aleatoria
        observation, info = env.reset()
        action, _ = agent.predict(observation)
        
        results["prediction"] = {
            "action_shape": action.shape if isinstance(action, np.ndarray) else None,
            "action": action.tolist() if isinstance(action, np.ndarray) else action
        }
        
        logger.info(f"Predicción de acción exitosa: {results['prediction']}")
        
        # Verificar que el agente funciona correctamente
        results["success"] = True
        logger.info("Prueba del agente completada con éxito")
        
        return results
    
    except Exception as e:
        logger.error(f"Error en test_agent_initialization: {e}")
        logger.error(traceback.format_exc())
        results["error"] = f"Error no controlado: {str(e)}"
        return results

def test_callback_integration(data_file: str) -> Dict[str, Any]:
    """
    Prueba la integración de callbacks durante el entrenamiento.
    
    Args:
        data_file (str): Ruta al archivo de datos
        
    Returns:
        Dict[str, Any]: Resultados de la prueba
    """
    logger.info("Probando integración de callbacks")
    results = {}
    
    try:
        # Cargar datos
        data_loader = DataLoader(DATA_CONFIG)
        train_data, val_data, test_data = data_loader.prepare_data(data_file)
        
        # Crear entorno
        env = TradingEnv(
            data=train_data,
            config=ENV_CONFIG,
            window_size=ENV_CONFIG.get('window_size', 60),
            mode='train'
        )
        
        # Crear entorno de validación
        val_env = TradingEnv(
            data=val_data,
            config=ENV_CONFIG,
            window_size=ENV_CONFIG.get('window_size', 60),
            mode='validation'
        )
        
        # Crear configuración para el agente
        config = {
            'data_config': DATA_CONFIG,
            'env_config': ENV_CONFIG,
            'reward_config': REWARD_CONFIG,
            'ppo_config': PPO_CONFIG,
            'training_config': TRAINING_CONFIG
        }
        
        # Inicializar agente
        agent = PPOAgent(env=env, config=config)
        
        # Crear callback de prueba
        callback = TradeCallback(
            log_dir=log_dir,
            save_path=log_dir,
            save_interval=100,
            eval_interval=50,
            eval_env=val_env,
            eval_episodes=2,
            verbose=2
        )
        
        # Entrenar por un número pequeño de pasos
        try:
            agent.train(total_timesteps=100, callback=callback)
            results["training_completed"] = True
            logger.info("Mini-entrenamiento completado correctamente")
        except Exception as train_error:
            logger.error(f"Error durante el mini-entrenamiento: {train_error}")
            logger.error(traceback.format_exc())
            results["training_completed"] = False
            results["training_error"] = str(train_error)
        
        # Verificar si se crearon los archivos esperados
        log_file = os.path.join(log_dir, "training.log")
        results["log_file_exists"] = os.path.exists(log_file)
        
        # Verificar si el callback funcionó correctamente
        results["success"] = results.get("training_completed", False)
        
        return results
    
    except Exception as e:
        logger.error(f"Error en test_callback_integration: {e}")
        logger.error(traceback.format_exc())
        results["error"] = f"Error no controlado: {str(e)}"
        return results

def test_trainer_workflow(data_file: str) -> Dict[str, Any]:
    """
    Prueba el flujo de trabajo completo del entrenador.
    
    Args:
        data_file (str): Ruta al archivo de datos
        
    Returns:
        Dict[str, Any]: Resultados de la prueba
    """
    logger.info("Probando flujo de trabajo del entrenador")
    results = {}
    
    try:
        # Crear configuración para el entrenador
        config = {
            'data_config': DATA_CONFIG,
            'env_config': ENV_CONFIG,
            'reward_config': REWARD_CONFIG,
            'ppo_config': PPO_CONFIG,
            'training_config': {
                'total_timesteps': 200,
                'eval_freq': 100,
                'save_path': log_dir,
                'log_path': log_dir,
                'n_eval_episodes': 2
            }
        }
        
        # Inicializar entrenador
        trainer = Trainer(config)
        
        # Configurar pipeline de entrenamiento
        try:
            trainer.setup(data_file)
            results["setup_completed"] = True
            logger.info("Configuración del entrenador completada con éxito")
        except Exception as setup_error:
            logger.error(f"Error durante la configuración del entrenador: {setup_error}")
            logger.error(traceback.format_exc())
            results["setup_completed"] = False
            results["setup_error"] = str(setup_error)
            return results
        
        # Entrenar por un número pequeño de pasos
        try:
            trainer.train(show_progress=True)
            results["training_completed"] = True
            logger.info("Mini-entrenamiento con Trainer completado correctamente")
        except Exception as train_error:
            logger.error(f"Error durante el entrenamiento con Trainer: {train_error}")
            logger.error(traceback.format_exc())
            results["training_completed"] = False
            results["training_error"] = str(train_error)
        
        # Evaluar modelo
        if results.get("training_completed", False):
            try:
                eval_results = trainer.evaluate()
                results["evaluation_completed"] = True
                results["evaluation_results"] = eval_results
                logger.info(f"Evaluación completada: {eval_results}")
            except Exception as eval_error:
                logger.error(f"Error durante la evaluación: {eval_error}")
                logger.error(traceback.format_exc())
                results["evaluation_completed"] = False
                results["evaluation_error"] = str(eval_error)
        
        # Verificar si el entrenador funcionó correctamente
        results["success"] = results.get("training_completed", False)
        
        return results
    
    except Exception as e:
        logger.error(f"Error en test_trainer_workflow: {e}")
        logger.error(traceback.format_exc())
        results["error"] = f"Error no controlado: {str(e)}"
        return results

def main():
    # Verificar que el dataset existe
    data_file = "data/NQ_06-25_combined_20250320_225417.csv"
    
    if not os.path.exists(data_file):
        # Intentar encontrar un archivo .csv en data/dataset
        csv_files = [f for f in os.listdir("data/dataset") if f.endswith(".csv")]
        if csv_files:
            data_file = os.path.join("data/dataset", csv_files[0])
            logger.info(f"Using alternative data file: {data_file}")
        else:
            logger.error(f"No se encontraron archivos de datos válidos.")
            sys.exit(1)
    
    # Ejecutar pruebas
    logger.info(f"=== INICIANDO DIAGNÓSTICO DE ENTRENAMIENTO ===")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Archivo de datos: {data_file}")
    
    # 1. Verificar formato de datos
    logger.info("=== PRUEBA 1: FORMATO DE DATOS ===")
    data_results = check_data_format(data_file)
    logger.info(f"Resultados: {'OK' if data_results.get('success', False) else 'ERROR'}")
    
    # 2. Probar entorno
    logger.info("=== PRUEBA 2: ENTORNO DE TRADING ===")
    env_results = test_environment(data_file)
    logger.info(f"Resultados: {'OK' if env_results.get('success', False) else 'ERROR'}")
    
    # 3. Probar sistema de recompensas
    logger.info("=== PRUEBA 3: SISTEMA DE RECOMPENSAS ===")
    reward_results = test_reward_components(data_file)
    logger.info(f"Resultados: {'OK' if reward_results.get('success', False) else 'ERROR'}")
    
    # 4. Probar inicialización del agente
    logger.info("=== PRUEBA 4: INICIALIZACIÓN DEL AGENTE ===")
    agent_results = test_agent_initialization(data_file)
    logger.info(f"Resultados: {'OK' if agent_results.get('success', False) else 'ERROR'}")
    
    # 5. Probar callbacks
    logger.info("=== PRUEBA 5: INTEGRACIÓN DE CALLBACKS ===")
    callback_results = test_callback_integration(data_file)
    logger.info(f"Resultados: {'OK' if callback_results.get('success', False) else 'ERROR'}")
    
    # 6. Probar trainer
    logger.info("=== PRUEBA 6: FLUJO DE TRABAJO DEL ENTRENADOR ===")
    trainer_results = test_trainer_workflow(data_file)
    logger.info(f"Resultados: {'OK' if trainer_results.get('success', False) else 'ERROR'}")
    
    # Guardar resultados
    all_results = {
        "timestamp": timestamp,
        "data_file": data_file,
        "data_format": data_results,
        "environment": env_results,
        "reward_system": reward_results,
        "agent_initialization": agent_results,
        "callback_integration": callback_results,
        "trainer_workflow": trainer_results
    }
    
    # Guardar resultados como JSON
    import json
    results_file = os.path.join(log_dir, "diagnostico_resultados.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4, default=str)
    
    # Generar un resumen simplificado de problemas encontrados
    problems_found = []
    
    if not data_results.get("success", False):
        problems_found.append(f"Problema en formato de datos: {data_results.get('error', 'Error desconocido')}")
    
    if not env_results.get("success", False):
        problems_found.append(f"Problema en entorno de trading: {env_results.get('error', 'Error desconocido')}")
    
    if not reward_results.get("success", False):
        problems_found.append(f"Problema en sistema de recompensas: {reward_results.get('error', 'Error desconocido')}")
    
    if not agent_results.get("success", False):
        problems_found.append(f"Problema en inicialización del agente: {agent_results.get('error', 'Error desconocido')}")
    
    if not callback_results.get("success", False):
        problems_found.append(f"Problema en integración de callbacks: {callback_results.get('error', 'Error desconocido')}")
    
    if not trainer_results.get("success", False):
        problems_found.append(f"Problema en flujo de trabajo del entrenador: {trainer_results.get('error', 'Error desconocido')}")
    
    # Crear archivo de resumen
    summary_file = os.path.join(log_dir, "resumen_diagnostico.txt")
    with open(summary_file, "w") as f:
        f.write(f"=== RESUMEN DE DIAGNÓSTICO DE ENTRENAMIENTO ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Archivo de datos: {data_file}\n\n")
        
        f.write("RESULTADOS DE LAS PRUEBAS:\n")
        f.write(f"1. Formato de datos: {'OK' if data_results.get('success', False) else 'ERROR'}\n")
        f.write(f"2. Entorno de trading: {'OK' if env_results.get('success', False) else 'ERROR'}\n")
        f.write(f"3. Sistema de recompensas: {'OK' if reward_results.get('success', False) else 'ERROR'}\n")
        f.write(f"4. Inicialización del agente: {'OK' if agent_results.get('success', False) else 'ERROR'}\n")
        f.write(f"5. Integración de callbacks: {'OK' if callback_results.get('success', False) else 'ERROR'}\n")
        f.write(f"6. Flujo de trabajo del entrenador: {'OK' if trainer_results.get('success', False) else 'ERROR'}\n\n")
        
        if problems_found:
            f.write("PROBLEMAS ENCONTRADOS:\n")
            for i, problem in enumerate(problems_found, 1):
                f.write(f"{i}. {problem}\n")
        else:
            f.write("No se encontraron problemas críticos en las pruebas básicas.\n")
    
    # Imprimir resultados
    print(f"\n=== DIAGNÓSTICO COMPLETADO ===")
    print(f"Resultados guardados en: {log_dir}")
    print(f"Resumen: {summary_file}")
    
    # Devolver código de salida
    return 0 if all(r.get("success", False) for r in [
        data_results, env_results, reward_results, 
        agent_results, callback_results, trainer_results
    ]) else 1


if __name__ == "__main__":
    sys.exit(main())
