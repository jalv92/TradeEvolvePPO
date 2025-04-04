"""
Sistema de benchmarking para comparar diferentes arquitecturas y configuraciones
de modelos de trading basados en aprendizaje por refuerzo.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import logging

from environment.enhanced_trading_env import EnhancedTradingEnv
from utils.logger import setup_logger
from utils.helpers import create_output_dir
from utils.adaptive_training import AdaptiveLstmCallback

# Configurar logger
logger = setup_logger("benchmarking", log_file="logs/benchmarking.log")

class ModelBenchmark:
    """Clase para realizar benchmarking sistemático de modelos de trading."""
    
    def __init__(self, data_train, data_val, base_config, output_dir=None):
        """
        Inicializa el benchmark con los datos y configuración base.
        
        Args:
            data_train: DataFrame con datos de entrenamiento
            data_val: DataFrame con datos de validación
            base_config: Configuración base para todos los experimentos
            output_dir: Directorio donde guardar los resultados
        """
        self.data_train = data_train
        self.data_val = data_val
        self.base_config = base_config
        
        # Crear directorio de resultados si no existe
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"results/benchmark_{timestamp}"
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/models", exist_ok=True)
        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)
        os.makedirs(f"{self.output_dir}/visualizations", exist_ok=True)
        
        # Variables para tracking de experimentos
        self.experiments = []
        self.results = []
        
        logger.info(f"Benchmark inicializado - Resultados en: {self.output_dir}")
    
    def generate_experiment_grid(self, param_grid):
        """
        Genera una lista de configuraciones de experimentos basados en la rejilla de parámetros.
        
        Args:
            param_grid: Diccionario con parámetros a variar y sus posibles valores
                        Formato: {'param1': [val1, val2], 'param2': [val1, val2]}
        
        Returns:
            Lista de configuraciones para experimentos
        """
        # Obtener todas las combinaciones de parámetros
        keys = param_grid.keys()
        values = param_grid.values()
        experiments = []
        
        # Generar todas las combinaciones posibles
        for combination in product(*values):
            # Crear un diccionario con la combinación actual
            experiment_config = dict(zip(keys, combination))
            experiments.append(experiment_config)
        
        self.experiments = experiments
        logger.info(f"Generados {len(experiments)} experimentos para el grid de parámetros")
        return experiments
    
    def run_single_experiment(self, experiment_id, config_override, timesteps=100000):
        """
        Ejecuta un único experimento con la configuración dada.
        
        Args:
            experiment_id: Identificador único del experimento
            config_override: Diccionario con parámetros específicos para este experimento
            timesteps: Número de pasos de entrenamiento
            
        Returns:
            Diccionario con resultados del experimento
        """
        # Crear carpeta para este experimento
        exp_dir = f"{self.output_dir}/models/exp_{experiment_id}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Combinar configuración base con override
        env_config = self.base_config['env_config'].copy()
        ppo_config = self.base_config['ppo_config'].copy()
        
        # Actualizar con los parámetros específicos del experimento
        for key, value in config_override.items():
            if key.startswith('env_'):
                # Parámetro de entorno
                param_name = key[4:]  # Quitar prefijo 'env_'
                env_config[param_name] = value
            elif key.startswith('ppo_'):
                # Parámetro de PPO
                param_name = key[4:]  # Quitar prefijo 'ppo_'
                ppo_config[param_name] = value
        
        # Guardar la configuración
        with open(f"{exp_dir}/config.json", 'w') as f:
            json.dump({
                'experiment_id': experiment_id,
                'env_config': env_config,
                'ppo_config': ppo_config,
                'timesteps': timesteps,
                'override': config_override
            }, f, indent=4, default=str)
        
        # Crear entornos
        train_env = self._create_env(self.data_train, env_config, 'train')
        eval_env = self._create_env(self.data_val, env_config, 'validation')
        
        # Sincronizar estadísticas de normalización
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms
        
        # Crear modelo con la configuración específica
        try:
            # Determinar la política basada en los parámetros
            if config_override.get('model_type', 'lstm').lower() == 'mlp':
                policy_type = "MlpPolicy"
            else:
                policy_type = "LstmPolicy"
                
            model = PPO(
                policy_type,
                train_env,
                verbose=0,
                tensorboard_log=f"{exp_dir}/tensorboard",
                device="cuda" if torch.cuda.is_available() else "cpu",
                **ppo_config
            )
            
            # Entrenar el modelo
            start_time = time.time()
            model.learn(
                total_timesteps=timesteps,
                tb_log_name=f"exp_{experiment_id}",
                progress_bar=False
            )
            training_time = time.time() - start_time
            
            # Guardar el modelo
            model.save(f"{exp_dir}/model")
            train_env.save(f"{exp_dir}/vec_normalize.pkl")
            
            # Evaluar el modelo
            mean_reward, std_reward = evaluate_policy(
                model, 
                eval_env, 
                n_eval_episodes=5,
                deterministic=False
            )
            
            # Realizar evaluación detallada
            detailed_metrics = self._evaluate_detailed(model, eval_env)
            
            # Guardar resultados
            results = {
                'experiment_id': experiment_id,
                'config': config_override,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'training_time': training_time,
                'win_rate': detailed_metrics.get('win_rate', 0),
                'profit_factor': detailed_metrics.get('profit_factor', 0),
                'total_trades': detailed_metrics.get('total_trades', 0),
                'return_pct': detailed_metrics.get('return_pct', 0),
                'max_drawdown': detailed_metrics.get('max_drawdown', 0),
                'sharpe_ratio': detailed_metrics.get('sharpe_ratio', 0),
            }
            
            # Guardar resultados
            with open(f"{exp_dir}/results.json", 'w') as f:
                json.dump(results, f, indent=4)
                
            logger.info(f"Experimento {experiment_id} completado - Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error en experimento {experiment_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'experiment_id': experiment_id,
                'error': str(e),
                'status': 'failed'
            }
    
    def run_benchmarks(self, timesteps=100000, parallel=False, max_workers=None):
        """
        Ejecuta todos los experimentos generados.
        
        Args:
            timesteps: Número de pasos de entrenamiento por experimento
            parallel: Si es True, ejecuta experimentos en paralelo
            max_workers: Número máximo de workers para ejecución paralela
            
        Returns:
            DataFrame con resultados de los experimentos
        """
        if not self.experiments:
            logger.error("No hay experimentos configurados. Ejecute generate_experiment_grid primero.")
            return None
            
        logger.info(f"Iniciando benchmark de {len(self.experiments)} experimentos con {timesteps} pasos cada uno")
        
        results = []
        
        if parallel and torch.cuda.is_available():
            logger.warning("Ejecución paralela con GPU puede causar problemas de memoria. Considere reducir max_workers.")
        
        if parallel:
            # Ejecución paralela
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, config in enumerate(self.experiments):
                    futures.append(executor.submit(
                        self.run_single_experiment, i, config, timesteps
                    ))
                
                # Recoger resultados
                for future in futures:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error en experimento: {e}")
        else:
            # Ejecución secuencial
            for i, config in enumerate(self.experiments):
                result = self.run_single_experiment(i, config, timesteps)
                results.append(result)
        
        # Convertir resultados a DataFrame
        self.results = results
        df_results = pd.DataFrame(results)
        
        # Guardar resultados
        df_results.to_csv(f"{self.output_dir}/benchmark_results.csv", index=False)
        
        # Generar visualizaciones
        self.generate_visualizations(df_results)
        
        logger.info(f"Benchmark completado - Resultados guardados en {self.output_dir}")
        
        return df_results
    
    def generate_visualizations(self, results_df):
        """
        Genera visualizaciones a partir de los resultados del benchmark.
        
        Args:
            results_df: DataFrame con resultados de experimentos
        """
        if results_df.empty:
            logger.warning("No hay resultados para visualizar")
            return
            
        # Configurar estilo de los gráficos
        plt.style.use('ggplot')
        
        # 1. Comparativa de recompensas por experimento
        plt.figure(figsize=(12, 6))
        sns.barplot(x='experiment_id', y='mean_reward', data=results_df, 
                   palette='viridis', errorbar=('ci', 95))
        plt.title('Recompensa media por experimento')
        plt.xlabel('ID de Experimento')
        plt.ylabel('Recompensa Media')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/visualizations/reward_comparison.png")
        
        # 2. Tiempo de entrenamiento vs Recompensa
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='training_time', y='mean_reward', hue='experiment_id', 
                       data=results_df, palette='viridis', s=100)
        plt.title('Relación entre Tiempo de Entrenamiento y Recompensa')
        plt.xlabel('Tiempo de Entrenamiento (s)')
        plt.ylabel('Recompensa Media')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/visualizations/time_vs_reward.png")
        
        # 3. Win Rate vs Profit Factor
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='win_rate', y='profit_factor', hue='experiment_id', 
                       data=results_df, palette='viridis', s=100, legend='full')
        plt.title('Win Rate vs Profit Factor')
        plt.xlabel('Win Rate (%)')
        plt.ylabel('Profit Factor')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/visualizations/winrate_vs_profitfactor.png")
        
        # 4. Heatmap de correlación entre métricas
        metrics_cols = ['mean_reward', 'win_rate', 'profit_factor', 'total_trades', 
                       'return_pct', 'max_drawdown', 'training_time']
        
        # Filtrar columnas que existen en el DataFrame
        available_cols = [col for col in metrics_cols if col in results_df.columns]
        
        if len(available_cols) > 1:  # Necesitamos al menos 2 columnas para la correlación
            plt.figure(figsize=(10, 8))
            correlation = results_df[available_cols].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                       linewidths=.5)
            plt.title('Correlación entre Métricas')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/visualizations/metrics_correlation.png")
        
        logger.info(f"Visualizaciones generadas en {self.output_dir}/visualizations/")
    
    def get_best_models(self, metric='mean_reward', top_n=3, ascending=False):
        """
        Obtiene los mejores modelos según la métrica especificada.
        
        Args:
            metric: Métrica para ordenar los resultados
            top_n: Número de mejores modelos a retornar
            ascending: Si es True, ordena de menor a mayor (útil para métricas donde menor es mejor)
            
        Returns:
            DataFrame con los mejores modelos
        """
        if not self.results:
            logger.warning("No hay resultados disponibles")
            return None
            
        df_results = pd.DataFrame(self.results)
        
        if metric not in df_results.columns:
            logger.error(f"Métrica '{metric}' no encontrada en los resultados")
            return None
            
        # Ordenar por la métrica especificada
        sorted_df = df_results.sort_values(by=metric, ascending=ascending)
        
        # Obtener los mejores modelos
        best_models = sorted_df.head(top_n)
        
        logger.info(f"Top {top_n} modelos según {metric}:")
        for i, (_, row) in enumerate(best_models.iterrows()):
            logger.info(f"#{i+1}: Experimento {row['experiment_id']} - {metric}: {row[metric]}")
            
        return best_models
    
    def _create_env(self, data, config, mode='train'):
        """
        Crea un entorno vectorizado normalizado.
        
        Args:
            data: DataFrame con datos para el entorno
            config: Configuración del entorno
            mode: Modo de operación ('train', 'validation', 'test')
            
        Returns:
            Entorno vectorizado normalizado
        """
        # Copiar config para no modificar la original
        env_config = config.copy()
        env_config['mode'] = mode
        
        # Crear entorno base
        env = EnhancedTradingEnv(
            data=data,
            config=env_config,
            window_size=env_config.get('window_size', 60),
            mode=mode
        )
        
        # Vectorizar
        vec_env = DummyVecEnv([lambda: env])
        
        # Normalizar
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
            epsilon=1e-08
        )
        
        return vec_env
    
    def _evaluate_detailed(self, model, env, n_episodes=5):
        """
        Realiza una evaluación detallada del modelo.
        
        Args:
            model: Modelo entrenado
            env: Entorno de evaluación
            n_episodes: Número de episodios para evaluación
            
        Returns:
            Dict: Métricas detalladas
        """
        # Resultados
        episode_rewards = []
        episode_lengths = []
        trade_metrics = []
        
        for i in range(n_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, _ = env.step(action)
                total_reward += reward[0]
                steps += 1
            
            # Obtener métricas del entorno
            env_info = env.get_attr('get_performance_summary')[0]()
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            trade_metrics.append(env_info)
        
        # Calcular métricas promedio
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        avg_steps = sum(episode_lengths) / len(episode_lengths)
        
        # Combinar métricas de trading
        metrics = {
            "avg_reward": float(avg_reward),
            "avg_steps": float(avg_steps),
            "win_rate": sum(m['win_rate'] for m in trade_metrics) / len(trade_metrics) * 100,
            "profit_factor": sum(m['profit_factor'] for m in trade_metrics) / len(trade_metrics),
            "total_trades": sum(m['total_trades'] for m in trade_metrics),
            "return_pct": sum(m.get('return_pct', 0) for m in trade_metrics) / len(trade_metrics),
            "max_drawdown": sum(m.get('max_drawdown', 0) for m in trade_metrics) / len(trade_metrics),
        }
        
        return metrics


def run_benchmark_example():
    """Ejemplo de uso del sistema de benchmarking."""
    from data.data_loader import DataLoader
    from config.config import BASE_CONFIG, ENV_CONFIG, PPO_CONFIG, REWARD_CONFIG
    
    # Cargar datos
    base_config = BASE_CONFIG.copy()
    data_loader = DataLoader(config=base_config)
    file_path = os.path.join('data', 'dataset', f"{base_config['symbol']}.csv")
    df = data_loader.load_csv_data(file_path)
    
    # Dividir datos
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size+val_size].copy()
    
    # Configuración completa
    config = {
        'env_config': ENV_CONFIG.copy(),
        'ppo_config': PPO_CONFIG.copy(),
        'reward_config': REWARD_CONFIG.copy(),
    }
    
    # Añadir reward_config a env_config
    config['env_config']['reward_config'] = config['reward_config']
    
    # Crear benchmark
    benchmark = ModelBenchmark(
        data_train=train_df,
        data_val=val_df,
        base_config=config
    )
    
    # Definir grid de parámetros a probar
    param_grid = {
        'model_type': ['lstm', 'mlp'],  # Tipo de arquitectura
        'ppo_learning_rate': [0.0001, 0.0003],  # Tasa de aprendizaje
        'ppo_n_steps': [1024, 2048],  # Pasos por actualización
        'ppo_ent_coef': [0.01, 0.05],  # Coeficiente de entropía
        'env_min_hold_steps': [10, 20],  # Pasos mínimos de mantenimiento
        'env_position_cooldown': [15, 30],  # Tiempo de espera entre operaciones
    }
    
    # Generar experimentos
    benchmark.generate_experiment_grid(param_grid)
    
    # Ejecutar benchmark (usar un número bajo de pasos para el ejemplo)
    results = benchmark.run_benchmarks(timesteps=50000, parallel=False)
    
    # Obtener mejores modelos
    best_models = benchmark.get_best_models(metric='mean_reward', top_n=3)
    
    return benchmark, results, best_models


if __name__ == "__main__":
    benchmark, results, best_models = run_benchmark_example()
    print("Benchmark completado con éxito.") 