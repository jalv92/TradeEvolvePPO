#!/usr/bin/env python
"""
Script para optimizar y diagnosticar el sistema de recompensas de TradeEvolvePPO.

Este script permite probar diferentes configuraciones de recompensas y visualizar
cómo afectan al comportamiento del agente, especialmente para resolver el problema
del agente demasiado conservador que no realiza suficientes operaciones.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

# Importar componentes del proyecto
from config.config import (
    DATA_CONFIG, ENV_CONFIG, REWARD_CONFIG, 
    PPO_CONFIG, TRAINING_CONFIG
)
from data.data_loader import DataLoader
from environment.trading_env import TradingEnv
from agents.ppo_agent import PPOAgent
from utils.logger import setup_logger
from training.callback import TradeCallback

# Configuración de logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/reward_opt_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# Configurar logger
logger = setup_logger(
    name="reward_optimizer",
    log_file=f"{log_dir}/reward_optimizer.log",
    level="DEBUG",
    console_level="INFO",
    file_level="DEBUG"
)

class RewardOptimizer:
    """
    Clase para optimizar y diagnosticar el sistema de recompensas.
    """
    
    def __init__(self, data_file: str):
        """
        Inicializa el optimizador de recompensas.
        
        Args:
            data_file (str): Ruta al archivo de datos
        """
        self.data_file = data_file
        self.data_loader = DataLoader(DATA_CONFIG)
        self.train_data, self.val_data, self.test_data = self.data_loader.prepare_data(data_file)
        
        # Verificar si hay datos suficientes
        if len(self.train_data) < ENV_CONFIG.get('window_size', 60) * 2:
            logger.error(f"Datos de entrenamiento insuficientes: {len(self.train_data)} filas")
            raise ValueError("Datos de entrenamiento insuficientes")
        
        logger.info(f"Datos cargados: Train={len(self.train_data)}, Val={len(self.val_data)}, Test={len(self.test_data)}")
        
        # Verificar disponibilidad de CUDA
        self.cuda_available = torch.cuda.is_available()
        self.device_name = torch.cuda.get_device_name(0) if self.cuda_available else "CPU"
        logger.info(f"Dispositivo: {self.device_name}")
        
        # Crear directorios para resultados
        self.plots_dir = os.path.join(log_dir, "plots")
        self.configs_dir = os.path.join(log_dir, "configs")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)
    
    def run_experiment(self, reward_config: Dict[str, Any], 
                      name: str, timesteps: int = 10000, 
                      eval_interval: int = 1000) -> Dict[str, Any]:
        """
        Ejecuta un experimento con una configuración de recompensas específica.
        
        Args:
            reward_config (Dict[str, Any]): Configuración de recompensas
            name (str): Nombre del experimento
            timesteps (int): Número de pasos de entrenamiento
            eval_interval (int): Intervalo de evaluación
            
        Returns:
            Dict[str, Any]: Resultados del experimento
        """
        logger.info(f"Iniciando experimento: {name}")
        
        # Crear directorio para este experimento
        experiment_dir = os.path.join(log_dir, name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Guardar configuración de recompensas
        config_file = os.path.join(self.configs_dir, f"{name}_config.json")
        with open(config_file, "w") as f:
            json.dump(reward_config, f, indent=4)
        
        # Crear entorno con la configuración de recompensas
        env_config = ENV_CONFIG.copy()
        env_config.update(reward_config)
        
        # Crear entorno de entrenamiento
        env = TradingEnv(
            data=self.train_data,
            config=env_config,
            window_size=env_config.get('window_size', 60),
            mode='train'
        )
        
        # Crear entorno de validación
        val_env = TradingEnv(
            data=self.val_data,
            config=env_config,
            window_size=env_config.get('window_size', 60),
            mode='validation'
        )
        
        # Configuración del agente
        ppo_config = PPO_CONFIG.copy()
        
        # Asignar dispositivo (CPU/CUDA)
        if self.cuda_available:
            ppo_config['device'] = 'cuda'
        else:
            ppo_config['device'] = 'cpu'
        
        # Configurar agent con verbosidad mínima
        ppo_config['verbose'] = 0
        
        # Crear configuración completa
        config = {
            'data_config': DATA_CONFIG,
            'env_config': env_config,
            'reward_config': reward_config,
            'ppo_config': ppo_config,
            'training_config': {
                'total_timesteps': timesteps,
                'eval_freq': eval_interval,
                'save_path': experiment_dir,
                'log_path': experiment_dir,
                'n_eval_episodes': 3
            }
        }
        
        # Inicializar agente
        agent = PPOAgent(env=env, config=config)
        
        # Crear callback para seguimiento
        callback = ExperimentCallback(
            log_dir=experiment_dir,
            eval_env=val_env,
            eval_interval=eval_interval,
            name=name
        )
        
        # Entrenar agente
        try:
            logger.info(f"Entrenamiento de {timesteps} pasos para: {name}")
            start_time = time.time()
            
            agent.train(total_timesteps=timesteps, callback=callback)
            
            train_time = time.time() - start_time
            logger.info(f"Entrenamiento completado en {train_time:.2f} segundos")
            
            # Guardar modelo final
            final_model_path = os.path.join(experiment_dir, "final_model")
            agent.save(final_model_path)
            
            # Evaluar modelo
            logger.info(f"Evaluando modelo final...")
            results = self._evaluate_model(agent, val_env, num_episodes=5)
            
            # Guardar resultados
            results_file = os.path.join(experiment_dir, "results.json")
            with open(results_file, "w") as f:
                json.dump(results, f, indent=4, default=str)
            
            # Incluir métricas de entrenamiento
            results.update({
                'training_time': train_time,
                'action_distribution': callback.action_distribution,
                'reward_history': callback.reward_history,
                'component_history': callback.component_history
            })
            
            # Generar gráficos
            self._generate_plots(results, name)
            
            logger.info(f"Experimento completado: {name}")
            return results
            
        except Exception as e:
            logger.error(f"Error durante el experimento {name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'action_distribution': callback.action_distribution,
                'reward_history': callback.reward_history,
                'component_history': callback.component_history
            }
    
    def run_multiple_experiments(self, configurations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Ejecuta múltiples experimentos con diferentes configuraciones de recompensas.
        
        Args:
            configurations (List[Dict[str, Any]]): Lista de configuraciones a probar
            
        Returns:
            Dict[str, Dict[str, Any]]: Resultados de todos los experimentos
        """
        results = {}
        
        for i, config in enumerate(configurations):
            name = config.pop('name', f"experiment_{i}")
            timesteps = config.pop('timesteps', 10000)
            
            logger.info(f"Iniciando experimento {i+1}/{len(configurations)}: {name}")
            
            # Ejecutar experimento
            experiment_results = self.run_experiment(
                reward_config=config,
                name=name,
                timesteps=timesteps
            )
            
            results[name] = experiment_results
        
        # Generar comparativa final
        self._generate_comparative_plots(results)
        
        return results
    
    def _evaluate_model(self, agent: PPOAgent, env: TradingEnv, num_episodes: int = 5) -> Dict[str, Any]:
        """
        Evalúa un modelo en un entorno.
        
        Args:
            agent (PPOAgent): Agente a evaluar
            env (TradingEnv): Entorno de evaluación
            num_episodes (int): Número de episodios
            
        Returns:
            Dict[str, Any]: Resultados de la evaluación
        """
        rewards = []
        trades = []
        win_rates = []
        profit_factors = []
        position_durations = []
        action_counts = {'hold': 0, 'buy': 0, 'sell': 0}
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                
                # Registrar acción tomada
                if isinstance(action, np.ndarray):
                    if action[0] > 0.5:
                        action_counts['buy'] += 1
                    elif action[0] < -0.5:
                        action_counts['sell'] += 1
                    else:
                        action_counts['hold'] += 1
                else:
                    if action == 1:
                        action_counts['buy'] += 1
                    elif action == 2:
                        action_counts['sell'] += 1
                    else:
                        action_counts['hold'] += 1
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated
            
            # Recopilar métricas
            rewards.append(episode_reward)
            
            if hasattr(env, 'get_performance_summary'):
                metrics = env.get_performance_summary()
                trades.append(metrics.get('total_trades', 0))
                win_rates.append(metrics.get('win_rate', 0) * 100)
                profit_factors.append(metrics.get('profit_factor', 0))
                
                # Duración promedio de posiciones (si está disponible)
                if 'avg_position_duration' in metrics:
                    position_durations.append(metrics['avg_position_duration'])
        
        # Calcular promedios
        avg_reward = np.mean(rewards)
        avg_trades = np.mean(trades)
        avg_win_rate = np.mean(win_rates)
        avg_profit_factor = np.mean(profit_factors)
        avg_position_duration = np.mean(position_durations) if position_durations else 0
        
        # Calcular proporciones de acciones
        total_actions = sum(action_counts.values())
        action_percentages = {
            action: (count / total_actions * 100) if total_actions > 0 else 0 
            for action, count in action_counts.items()
        }
        
        return {
            'avg_reward': float(avg_reward),
            'avg_trades': float(avg_trades),
            'avg_win_rate': float(avg_win_rate),
            'avg_profit_factor': float(avg_profit_factor),
            'avg_position_duration': float(avg_position_duration),
            'action_distribution': action_counts,
            'action_percentages': action_percentages,
            'episodes_completed': num_episodes
        }
    
    def _generate_plots(self, results: Dict[str, Any], name: str) -> None:
        """
        Genera gráficos para visualizar los resultados de un experimento.
        
        Args:
            results (Dict[str, Any]): Resultados del experimento
            name (str): Nombre del experimento
        """
        # 1. Gráfico de distribución de acciones
        try:
            if 'action_percentages' in results:
                actions = list(results['action_percentages'].keys())
                percentages = list(results['action_percentages'].values())
                
                plt.figure(figsize=(10, 6))
                plt.bar(actions, percentages, color=['gray', 'green', 'red'])
                plt.title(f'Distribución de Acciones - {name}')
                plt.xlabel('Acción')
                plt.ylabel('Porcentaje (%)')
                plt.ylim(0, 100)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(self.plots_dir, f"{name}_action_distribution.png"))
                plt.close()
        except Exception as e:
            logger.error(f"Error generando gráfico de distribución de acciones: {e}")
        
        # 2. Gráfico de historia de recompensas
        try:
            if 'reward_history' in results and results['reward_history']:
                plt.figure(figsize=(12, 6))
                plt.plot(results['reward_history'], marker='', linewidth=2)
                plt.title(f'Historia de Recompensas - {name}')
                plt.xlabel('Evaluación')
                plt.ylabel('Recompensa Media')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(self.plots_dir, f"{name}_reward_history.png"))
                plt.close()
        except Exception as e:
            logger.error(f"Error generando gráfico de historia de recompensas: {e}")
        
        # 3. Gráfico de componentes de recompensa
        try:
            if 'component_history' in results and results['component_history']:
                component_history = results['component_history']
                # Promediar componentes por evaluación
                avg_components = {}
                for eval_components in component_history:
                    for component, value in eval_components.items():
                        if component not in avg_components:
                            avg_components[component] = []
                        avg_components[component].append(value)
                
                plt.figure(figsize=(14, 8))
                for component, values in avg_components.items():
                    plt.plot(values, label=component, linewidth=2)
                
                plt.title(f'Componentes de Recompensa - {name}')
                plt.xlabel('Paso')
                plt.ylabel('Valor')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(self.plots_dir, f"{name}_reward_components.png"))
                plt.close()
        except Exception as e:
            logger.error(f"Error generando gráfico de componentes de recompensa: {e}")
    
    def _generate_comparative_plots(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Genera gráficos comparativos de múltiples experimentos.
        
        Args:
            all_results (Dict[str, Dict[str, Any]]): Resultados de todos los experimentos
        """
        # 1. Comparación de métricas clave
        try:
            metrics = ['avg_reward', 'avg_trades', 'avg_win_rate', 'avg_profit_factor']
            metric_names = ['Recompensa Media', 'Operaciones Promedio', 'Win Rate (%)', 'Profit Factor']
            
            experiment_names = list(all_results.keys())
            metric_values = {
                metric: [all_results[name].get(metric, 0) for name in experiment_names]
                for metric in metrics
            }
            
            fig, axs = plt.subplots(2, 2, figsize=(18, 12))
            axs = axs.flatten()
            
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                values = metric_values[metric]
                axs[i].bar(experiment_names, values, color='skyblue')
                axs[i].set_title(name)
                axs[i].set_xticklabels(experiment_names, rotation=45, ha='right')
                axs[i].grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "comparative_metrics.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Error generando comparativa de métricas: {e}")
        
        # 2. Comparación de distribución de acciones
        try:
            action_types = ['hold', 'buy', 'sell']
            
            # Extraer porcentajes de cada acción para cada experimento
            hold_pcts = []
            buy_pcts = []
            sell_pcts = []
            
            for name, results in all_results.items():
                if 'action_percentages' in results:
                    action_pcts = results['action_percentages']
                    hold_pcts.append(action_pcts.get('hold', 0))
                    buy_pcts.append(action_pcts.get('buy', 0))
                    sell_pcts.append(action_pcts.get('sell', 0))
                else:
                    hold_pcts.append(0)
                    buy_pcts.append(0)
                    sell_pcts.append(0)
            
            # Crear gráfico de barras agrupadas
            x = np.arange(len(experiment_names))
            width = 0.25
            
            fig, ax = plt.figure(figsize=(14, 8))
            ax = plt.subplot(111)
            
            ax.bar(x - width, hold_pcts, width, label='Hold', color='gray')
            ax.bar(x, buy_pcts, width, label='Buy', color='green')
            ax.bar(x + width, sell_pcts, width, label='Sell', color='red')
            
            ax.set_title('Comparación de Distribución de Acciones')
            ax.set_xticks(x)
            ax.set_xticklabels(experiment_names, rotation=45, ha='right')
            ax.set_ylabel('Porcentaje (%)')
            ax.set_ylim(0, 100)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "comparative_actions.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Error generando comparativa de acciones: {e}")
        
        # 3. Comparación de recompensas a lo largo del tiempo
        try:
            plt.figure(figsize=(14, 8))
            
            for name, results in all_results.items():
                if 'reward_history' in results and results['reward_history']:
                    plt.plot(results['reward_history'], label=name, linewidth=2)
            
            plt.title('Comparación de Recompensas')
            plt.xlabel('Evaluación')
            plt.ylabel('Recompensa Media')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(self.plots_dir, "comparative_rewards.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Error generando comparativa de recompensas: {e}")


class ExperimentCallback(TradeCallback):
    """
    Callback personalizado para experimentos de optimización de recompensas.
    """
    
    def __init__(self, log_dir: str, eval_env: TradingEnv, eval_interval: int, name: str):
        super(ExperimentCallback, self).__init__(
            log_dir=log_dir,
            save_path=log_dir,
            save_interval=eval_interval * 5,  # Guardar menos frecuentemente
            eval_interval=eval_interval,
            eval_env=eval_env,
            eval_episodes=3,
            verbose=0  # Minimal verbosity
        )
        self.name = name
        self.reward_history = []
        self.action_distribution = {'hold': 0, 'buy': 0, 'sell': 0}
        self.component_history = []
        self.last_actions = []
        
    def _on_step(self) -> bool:
        # Registrar distribución de acciones
        if self.locals.get('actions') is not None:
            action = self.locals['actions'][0]  # Primera acción (entorno único)
            
            if isinstance(action, np.ndarray):
                if action[0] > 0.5:  # Buy/Long
                    self.action_distribution['buy'] += 1
                    self.last_actions.append('buy')
                elif action[0] < -0.5:  # Sell/Short
                    self.action_distribution['sell'] += 1
                    self.last_actions.append('sell')
                else:  # Hold/Close
                    self.action_distribution['hold'] += 1
                    self.last_actions.append('hold')
            else:
                if action == 1:  # Buy/Long
                    self.action_distribution['buy'] += 1
                    self.last_actions.append('buy')
                elif action == 2:  # Sell/Short
                    self.action_distribution['sell'] += 1
                    self.last_actions.append('sell')
                else:  # Hold/Close
                    self.action_distribution['hold'] += 1
                    self.last_actions.append('hold')
            
            # Mantener solo las últimas 100 acciones
            if len(self.last_actions) > 100:
                self.last_actions.pop(0)
        
        # Capturar componentes de recompensa si están disponibles
        if hasattr(self.model.env.unwrapped, 'get_env_attr'):
            if self.n_calls % self.eval_interval == 0:
                try:
                    env = self.model.env.unwrapped.get_env_attr('envs')[0]
                    if hasattr(env, '_reward_components'):
                        self.component_history.append(env._reward_components.copy())
                except:
                    pass
        
        return super()._on_step()
    
    def _evaluate_policy(self) -> None:
        super()._evaluate_policy()
        
        # Guardar histórico de recompensas
        if hasattr(self, 'best_mean_reward'):
            self.reward_history.append(self.best_mean_reward)
        
        # Log breve de progreso
        if self.n_calls % self.eval_interval == 0:
            action_total = sum(self.action_distribution.values())
            if action_total > 0:
                hold_pct = self.action_distribution['hold'] / action_total * 100
                buy_pct = self.action_distribution['buy'] / action_total * 100
                sell_pct = self.action_distribution['sell'] / action_total * 100
                
                print(f"[{self.name}] Paso {self.n_calls}: "
                      f"Hold={hold_pct:.1f}%, Buy={buy_pct:.1f}%, Sell={sell_pct:.1f}%")


def create_default_configurations() -> List[Dict[str, Any]]:
    """
    Crea una lista de configuraciones por defecto para experimentar.
    
    Returns:
        List[Dict[str, Any]]: Lista de configuraciones
    """
    configs = []
    
    # Usar la configuración actual como base
    base_config = REWARD_CONFIG.copy()
    
    # 1. Configuración actual (baseline)
    configs.append({
        'name': 'baseline',
        'timesteps': 20000,
        **base_config
    })
    
    # 2. Configuración con mayor incentivo por trading
    configs.append({
        'name': 'high_trading_incentive',
        'timesteps': 20000,
        **base_config,
        'pnl_weight': base_config.get('pnl_weight', 3.0) * 0.7,  # Reducir énfasis en PnL
        'trade_completion_bonus': base_config.get('trade_completion_bonus', 5.0) * 2.0,  # Aumentar bonus por completar operaciones
        'base_reward': -0.01,  # Menor penalización base
        'inactivity_weight': base_config.get('inactivity_weight', 2.0) * 1.5  # Mayor penalización por inactividad
    })
    
    # 3. Configuración con exploración forzada
    configs.append({
        'name': 'forced_exploration',
        'timesteps': 20000,
        **base_config,
        'base_reward': -0.02,  # Mayor penalización base para forzar acciones
        'pnl_weight': base_config.get('pnl_weight', 3.0) * 0.5,  # Menor énfasis en PnL
        'trade_completion_bonus': base_config.get('trade_completion_bonus', 5.0) * 3.0,  # Mayor bonus por completar operaciones
        'drawdown_weight': base_config.get('drawdown_weight', 0.05) * 0.5,  # Menor penalización por drawdown
        'inactivity_weight': 5.0  # Penalización severa por inactividad
    })
    
    # 4. Configuración "solo acción", no importa el resultado
    configs.append({
        'name': 'action_only',
        'timesteps': 20000,
        **base_config,
        'base_reward': -0.1,  # Alta penalización base para forzar acciones
        'pnl_weight': 0.5,  # Mínimo énfasis en PnL
        'trade_completion_bonus': 10.0,  # Bonus extremadamente alto por completar operaciones
        'drawdown_weight': 0.01,  # Mínima penalización por drawdown
        'risk_weight': 0.05,  # Mínima preocupación por riesgo
        'inactivity_weight': 10.0  # Penalización extrema por inactividad
    })
    
    return configs


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
    
    # Imprimir información inicial
    logger.info(f"=== OPTIMIZADOR DE SISTEMA DE RECOMPENSAS ===")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Archivo de datos: {data_file}")
    
    # Crear optimizador
    optimizer = RewardOptimizer(data_file)
    
    # Usar configuraciones por defecto o personalizadas
    if len(sys.argv) > 1 and sys.argv[1] == '--custom':
        # Cargar configuración personalizada
        if len(sys.argv) > 2:
            config_file = sys.argv[2]
            
            if not os.path.exists(config_file):
                logger.error(f"Archivo de configuración no encontrado: {config_file}")
                sys.exit(1)
            
            with open(config_file, 'r') as f:
                configs = json.load(f)
            
            logger.info(f"Configuraciones personalizadas cargadas desde: {config_file}")
        else:
            logger.error("Debe especificar un archivo de configuración con --custom")
            sys.exit(1)
    else:
        # Usar configuraciones por defecto
        configs = create_default_configurations()
        logger.info(f"Usando {len(configs)} configuraciones por defecto")
    
    # Ejecutar experimentos
    results = optimizer.run_multiple_experiments(configs)
    
    # Guardar resultados finales
    final_results_file = os.path.join(log_dir, "final_results.json")
    with open(final_results_file, "w") as f:
        json.dump(results, f, indent=4, default=str)
    
    logger.info(f"Resultados finales guardados en: {final_results_file}")
    
    # Imprimir resumen de resultados
    print("\n=== RESUMEN DE RESULTADOS ===")
    print(f"{'Experimento':<25} {'Recompensa':<12} {'Trades':<8} {'Win Rate':<10} {'Profit Factor':<12}")
    print("-" * 70)
    
    for name, result in results.items():
        reward = result.get('avg_reward', 0)
        trades = result.get('avg_trades', 0)
        win_rate = result.get('avg_win_rate', 0)
        profit_factor = result.get('avg_profit_factor', 0)
        
        print(f"{name:<25} {reward:<12.2f} {trades:<8.1f} {win_rate:<10.1f}% {profit_factor:<12.2f}")
    
    print("\n=== ANÁLISIS DE DISTRIBUCIÓN DE ACCIONES ===")
    print(f"{'Experimento':<25} {'Hold %':<10} {'Buy %':<10} {'Sell %':<10}")
    print("-" * 60)
    
    for name, result in results.items():
        if 'action_percentages' in result:
            pcts = result['action_percentages']
            hold = pcts.get('hold', 0)
            buy = pcts.get('buy', 0)
            sell = pcts.get('sell', 0)
            
            print(f"{name:<25} {hold:<10.1f} {buy:<10.1f} {sell:<10.1f}")
    
    print(f"\nGráficos guardados en: {log_dir}/plots")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
