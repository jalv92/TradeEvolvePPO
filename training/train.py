import os
import time
import gymnasium as gym
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from environment.trading_env import TradingEnv
from utils.data_loader import load_data
from config.config import (
    BASE_CONFIG, ENV_CONFIG, AGENT_CONFIG, 
    TRAINING_CONFIG, VISUALIZATION_CONFIG
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import shutil
from datetime import datetime
import glob
from evaluation.metrics import TradingMetrics
from utils.helpers import format_time  # Importamos format_time para el formateo de tiempos

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradeMetricsCallback(BaseCallback):
    """
    Callback para trackear métricas detalladas durante el entrenamiento.
    """
    
    def __init__(self, eval_env, log_dir: str, eval_freq: int = 20000, 
                 verbose: int = 1, plot_freq: int = 100000, 
                 best_model_save_path: Optional[str] = None):
        """
        Inicializar callback.
        
        Args:
            eval_env: Entorno de evaluación
            log_dir: Directorio para guardar logs
            eval_freq: Frecuencia de evaluación (en steps) - aumentada para reducir verbosidad
            verbose: Nivel de detalle
            plot_freq: Frecuencia para generar gráficos
            best_model_save_path: Ruta para guardar mejor modelo
        """
        super(TradeMetricsCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.plot_freq = plot_freq
        self.best_model_save_path = best_model_save_path
        self.best_mean_reward = -float('inf')
        
        # Crear directorio de métricas
        self.metrics_dir = os.path.join(log_dir, 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Metricas para seguimiento
        self.metrics_history = {
            'timesteps': [],
            'win_rate': [],
            'profit_factor': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'total_trades': [],
            'avg_trade_pnl': [],
            'market_exposure': [],
            'avg_position_duration': [],
            'small_trades_pct': [],
            'direction_bias': [],
            'avg_rr_ratio': [],
            'sortino_ratio': []
        }
        
        # Seguimiento de componentes de reward
        self.reward_components_history = {
            'timesteps': [],
            'pnl_reward': [],
            'win_rate_reward': [],
            'risk_reward': [],
            'inactivity_penalty': [],
            'size_reward': [],
            'risk_penalty': []
        }
        
        # Contadores para mostrar progreso
        self.last_eval_trades = 0
        self.total_trades_all_evals = 0
        self.evaluation_count = 0
        
        # Mejores métricas
        self.best_metrics = {
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0
        }
        
        # Progreso del entrenamiento
        self.start_time = time.time()
        
        # Imprimir encabezado de resumen
        if verbose > 0:
            print("\n" + "="*80)
            print(f"{'Pasos':>10} | {'Reward':>10} | {'Trades':>8} | {'Win%':>6} | {'PF':>6} | {'DD%':>6} | {'FPS':>6} | {'Tiempo':>10}")
            print("-"*80)
    
    def _on_step(self) -> bool:
        """
        Método llamado en cada paso del entrenamiento.
        """
        # Mostrar progreso simple cada 100,000 pasos
        if self.n_calls % 100000 == 0 and self.verbose > 0:
            elapsed = time.time() - self.start_time
            fps = self.num_timesteps / max(1, elapsed)
            print(f"{self.num_timesteps:>10,d} | {'--':>10} | {'--':>8} | {'--':>6} | {'--':>6} | {'--':>6} | {fps:>6.0f} | {format_time(elapsed):>10}")
        
        # Evaluar cada eval_freq pasos
        if self.n_calls % self.eval_freq == 0:
            self.evaluation_count += 1
            
            # Evaluar el modelo en el entorno de evaluación
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=3,  # Reducido para acelerar evaluaciones
                return_episode_rewards=True
            )
            
            # Obtener métricas detalladas del último episodio
            metrics = self.eval_env.metrics.get_summary_stats() if hasattr(self.eval_env, 'metrics') else {}
            
            # Registrar métricas a lo largo del tiempo
            self.metrics_history['timesteps'].append(self.num_timesteps)
            self.metrics_history['win_rate'].append(metrics.get('win_rate', 0))
            self.metrics_history['profit_factor'].append(metrics.get('profit_factor', 0))
            self.metrics_history['sharpe_ratio'].append(metrics.get('sharpe_ratio', 0))
            self.metrics_history['max_drawdown'].append(metrics.get('max_drawdown', 0))
            self.metrics_history['total_trades'].append(metrics.get('total_trades', 0))
            self.metrics_history['avg_trade_pnl'].append(metrics.get('avg_trade_pnl', 0))
            self.metrics_history['market_exposure'].append(metrics.get('market_exposure_pct', 0))
            self.metrics_history['avg_position_duration'].append(metrics.get('avg_position_duration', 0))
            self.metrics_history['small_trades_pct'].append(metrics.get('small_trades_pct', 0))
            self.metrics_history['direction_bias'].append(metrics.get('direction_bias', 0))
            self.metrics_history['avg_rr_ratio'].append(metrics.get('avg_rr_ratio', 0))
            self.metrics_history['sortino_ratio'].append(metrics.get('sortino_ratio', 0))
            
            # Registrar componentes de reward si están disponibles
            info = self.eval_env.get_info() if hasattr(self.eval_env, 'get_info') else {}
            if 'reward_components' in info:
                components = info['reward_components']
                self.reward_components_history['timesteps'].append(self.num_timesteps)
                for key in components:
                    if key not in self.reward_components_history:
                        self.reward_components_history[key] = []
                    self.reward_components_history[key].append(components[key])
            
            # Actualizar contadores
            current_trades = metrics.get('total_trades', 0)
            new_trades = current_trades - self.last_eval_trades if self.evaluation_count > 1 else current_trades
            self.total_trades_all_evals += new_trades
            self.last_eval_trades = current_trades
            
            # Actualizar mejores métricas
            win_rate = metrics.get('win_rate', 0)
            profit_factor = metrics.get('profit_factor', 0)
            total_trades = metrics.get('total_trades', 0)
            
            if win_rate > self.best_metrics['win_rate'] and total_trades >= max(3, self.best_metrics['total_trades']/2):
                self.best_metrics['win_rate'] = win_rate
            
            if profit_factor > self.best_metrics['profit_factor'] and total_trades >= max(3, self.best_metrics['total_trades']/2):
                self.best_metrics['profit_factor'] = profit_factor
                
            if total_trades > self.best_metrics['total_trades']:
                self.best_metrics['total_trades'] = total_trades
            
            # Registrar métricas en el logger de Stable-Baselines
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            
            # Calcular velocidad y tiempo
            elapsed = time.time() - self.start_time
            fps = self.num_timesteps / max(1, elapsed)
            
            # Imprimir resumen
            if self.verbose > 0:
                print(f"{self.num_timesteps:>10,d} | {mean_reward:>10.2f} | {total_trades:>8d} | {win_rate:>6.1f} | {profit_factor:>6.2f} | {metrics.get('max_drawdown', 0):>6.2f} | {fps:>6.0f} | {format_time(elapsed):>10}")
            
            # Guardar el mejor modelo
            if self.best_model_save_path is not None and mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                
                # Guardar también las métricas del mejor modelo
                metrics_path = os.path.join(self.best_model_save_path, "best_model_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                    
                if self.verbose > 0:
                    print(f"✓ Nuevo mejor modelo guardado: {mean_reward:.2f}")
            
            # Guardar historial de trades en csv (solo si hay trades)
            if metrics.get('total_trades', 0) > 0 and hasattr(self.eval_env, 'get_trade_history_df'):
                trades_df = self.eval_env.get_trade_history_df()
                if not trades_df.empty:
                    trades_path = os.path.join(self.metrics_dir, f"trades_{self.num_timesteps}.csv")
                    trades_df.to_csv(trades_path, index=False)
        
        # Generar y guardar gráficos periódicamente
        if self.n_calls % self.plot_freq == 0 and self.n_calls > 0 and self.verbose > 0:
            self._generate_plots()
            
            # Mostrar resumen de mejores resultados
            print("\n" + "="*40)
            print(f"MEJORES MÉTRICAS ({self.num_timesteps/1000:.0f}K pasos)")
            print(f"Win Rate: {self.best_metrics['win_rate']:.1f}%")
            print(f"Profit Factor: {self.best_metrics['profit_factor']:.2f}")
            print(f"Total Trades: {self.best_metrics['total_trades']}")
            print("="*40 + "\n")
        
        return True
    
    def _generate_plots(self):
        """Generar gráficos de métricas."""
        # Crear directorio para gráficos
        plots_dir = os.path.join(self.metrics_dir, f"plots_{self.num_timesteps}")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generar gráficos de métricas de trading
        self._plot_metrics_history(plots_dir)
        
        # Generar gráficos de componentes de reward
        self._plot_reward_components(plots_dir)
        
        # Generar gráficos del entorno si hay operaciones
        try:
            self.eval_env.plot_metrics(save_path=plots_dir)
        except Exception as e:
            logger.warning(f"Error al generar gráficos del entorno: {e}")
    
    def _plot_metrics_history(self, save_dir):
        """Generar gráficos de métricas a lo largo del tiempo."""
        if not self.metrics_history['timesteps']:
            return
        
        # Convertir a df para facilitar plotting
        metrics_df = pd.DataFrame(self.metrics_history)
        
        # Plotear métricas agrupadas por tipo
        
        # 1. Métricas de performance
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(metrics_df['timesteps'], metrics_df['win_rate'], label='Win Rate (%)')
        ax.plot(metrics_df['timesteps'], metrics_df['profit_factor'] * 10, label='Profit Factor (x10)')
        ax.set_title('Performance Metrics')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        fig.savefig(os.path.join(save_dir, 'performance_metrics.png'), dpi=300)
        plt.close(fig)
        
        # 2. Métricas de riesgo
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(metrics_df['timesteps'], metrics_df['sharpe_ratio'], label='Sharpe Ratio')
        ax.plot(metrics_df['timesteps'], metrics_df['sortino_ratio'], label='Sortino Ratio')
        ax.plot(metrics_df['timesteps'], metrics_df['max_drawdown'], label='Max Drawdown (%)')
        ax.set_title('Risk Metrics')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        fig.savefig(os.path.join(save_dir, 'risk_metrics.png'), dpi=300)
        plt.close(fig)
        
        # 3. Métricas de comportamiento
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(metrics_df['timesteps'], metrics_df['market_exposure'], label='Market Exposure (%)')
        ax.plot(metrics_df['timesteps'], metrics_df['small_trades_pct'], label='Small Trades (%)')
        ax.plot(metrics_df['timesteps'], metrics_df['direction_bias'] * 100, label='Direction Bias (x100)')
        ax.set_title('Behavior Metrics')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        fig.savefig(os.path.join(save_dir, 'behavior_metrics.png'), dpi=300)
        plt.close(fig)
        
        # 4. Métricas de trading
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(metrics_df['timesteps'], metrics_df['total_trades'], label='Total Trades')
        ax.plot(metrics_df['timesteps'], metrics_df['avg_position_duration'], label='Avg Position Duration')
        ax.plot(metrics_df['timesteps'], metrics_df['avg_rr_ratio'], label='Avg R:R Ratio')
        ax.set_title('Trading Metrics')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        fig.savefig(os.path.join(save_dir, 'trading_metrics.png'), dpi=300)
        plt.close(fig)
    
    def _plot_reward_components(self, save_dir):
        """Generar gráficos de componentes de reward."""
        if not self.reward_components_history['timesteps']:
            return
        
        # Convertir a df para facilitar plotting
        reward_df = pd.DataFrame(self.reward_components_history)
        
        # Plotear componentes de reward
        components = [col for col in reward_df.columns if col != 'timesteps']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        for comp in components:
            if comp in reward_df.columns:
                ax.plot(reward_df['timesteps'], reward_df[comp], label=comp)
        
        ax.set_title('Reward Components Over Time')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        fig.savefig(os.path.join(save_dir, 'reward_components.png'), dpi=300)
        plt.close(fig)


class ProgressiveTrainingCallback(BaseCallback):
    """
    Callback para entrenamiento progresivo (curriculum learning).
    Permite modificar gradualmente parámetros del entorno durante el entrenamiento.
    """
    
    def __init__(self, env, progressive_steps: List[int], 
                 target_params: Dict[str, List[Any]], verbose: int = 1):
        """
        Inicializar callback.
        
        Args:
            env: Entorno de trading
            progressive_steps: Lista de steps donde cambiar parámetros
            target_params: Diccionario de parámetros y sus valores progresivos
            verbose: Nivel de detalle
        """
        super(ProgressiveTrainingCallback, self).__init__(verbose)
        self.env = env
        self.progressive_steps = progressive_steps
        self.target_params = target_params
        self.current_level = 0
        self.levels = len(progressive_steps)
    
    def _on_step(self) -> bool:
        """
        Método llamado en cada paso del entrenamiento.
        """
        # Verificar si es momento de cambiar de nivel
        if (self.current_level < self.levels and 
            self.num_timesteps >= self.progressive_steps[self.current_level]):
            
            # Actualizar parámetros del entorno
            for param_name, param_values in self.target_params.items():
                if self.current_level < len(param_values):
                    # Obtener nuevo valor
                    new_value = param_values[self.current_level]
                    
                    # Actualizar parámetro en el entorno
                    if hasattr(self.env, param_name):
                        setattr(self.env, param_name, new_value)
                    
                    logger.info(f"Actualizado parámetro {param_name} a {new_value}")
            
            self.current_level += 1
            logger.info(f"Progresando a nivel {self.current_level}/{self.levels}")
        
        return True


def make_env(data, config, validation_segment=None, seed=0):
    """
    Crear función de fábrica de entornos para vectorización.
    
    Args:
        data: Datos de mercado
        config: Configuración del entorno
        validation_segment: Segmento para validación (None=entrenamiento)
        seed: Semilla aleatoria
    
    Returns:
        Función que crea un entorno
    """
    def _init():
        # Si es validación, usar segmento específico de datos
        if validation_segment is not None:
            # Dividir datos en segmentos
            segment_size = len(data) // 5  # 5 segmentos
            start_idx = validation_segment * segment_size
            end_idx = start_idx + segment_size
            segment_data = data.iloc[start_idx:end_idx].reset_index(drop=True)
        else:
            segment_data = data
        
        env = TradingEnv(config)
        env.set_data(segment_data)
        env.reset(seed=seed)
        return env
    
    return _init


def train_model(config: Dict = None, model_name: str = "ppo_trader") -> Tuple[str, Dict]:
    """
    Entrena un modelo PPO para trading.
    
    Args:
        config: Configuración personalizada (opcional)
        model_name: Nombre del modelo
    
    Returns:
        Tuple: (ruta al modelo guardado, métricas de evaluación final)
    """
    # Cargar configuración
    if config is None:
        config = {
            **BASE_CONFIG,
            **ENV_CONFIG,
            **AGENT_CONFIG,
            **TRAINING_CONFIG
        }
    
    # Crear directorio para resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", f"{model_name}_{timestamp}")
    models_dir = os.path.join(result_dir, "models")
    logs_dir = os.path.join(result_dir, "logs")
    best_model_dir = os.path.join(result_dir, "best_model")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Guardar configuración
    with open(os.path.join(result_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Cargar datos
    data = load_data(
        symbol=config['symbol'],
        timeframe=config['timeframe'],
        start_date=config['start_date'],
        end_date=config['end_date']
    )
    
    # Dividir en train/validation
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size].reset_index(drop=True)
    validation_data = data.iloc[train_size:].reset_index(drop=True)
    
    # Crear entorno de entrenamiento
    train_env = TradingEnv(config)
    train_env.set_data(train_data)
    
    # Configurar entornos de validación multi-segmento
    val_envs = []
    for i in range(5):  # 5 segmentos de validación cruzada
        val_env = TradingEnv(config)
        
        # Dividir validation_data en 5 segmentos
        segment_size = len(validation_data) // 5
        start_idx = i * segment_size
        end_idx = start_idx + segment_size
        segment_data = validation_data.iloc[start_idx:end_idx].reset_index(drop=True)
        
        val_env.set_data(segment_data)
        val_envs.append(val_env)
    
    # Usar el primer segmento como entorno de evaluación principal
    eval_env = val_envs[0]
    
    # Configurar logger
    new_logger = configure(logs_dir, ["stdout", "csv", "tensorboard"])
    
    # Crear callback de checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=config['checkpoint_freq'],
        save_path=models_dir,
        name_prefix=model_name,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Crear callback de métricas
    metrics_callback = TradeMetricsCallback(
        eval_env=eval_env,
        log_dir=logs_dir,
        eval_freq=config['log_freq'],
        plot_freq=config['log_freq'] * 5,
        best_model_save_path=best_model_dir,
        verbose=1
    )
    
    # Crear callback de entrenamiento progresivo (curriculum learning)
    progressive_callback = ProgressiveTrainingCallback(
        env=train_env,
        progressive_steps=config['progressive_steps'],
        target_params={
            'inactivity_threshold': [50, 40, 30, 25],  # Progresivamente reducir umbral
            'risk_aversion': [0.8, 0.6, 0.4, 0.2],  # Progresivamente reducir aversión al riesgo
            'trivial_trade_threshold': [5, 10, 15, 20]  # Progresivamente aumentar umbral
        },
        verbose=1
    )
    
    # Crear lista de callbacks
    callbacks = CallbackList([
        checkpoint_callback,
        metrics_callback,
        progressive_callback
    ])
    
    # Crear y entrenar modelo
    logger.info(f"Iniciando entrenamiento de {model_name} por {config['total_timesteps']} steps")
    logger.info(f"Configuración: {config}")
    
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        clip_range_vf=config['clip_range_vf'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        max_grad_norm=config['max_grad_norm'],
        policy_kwargs=config['policy_kwargs'],
        tensorboard_log=logs_dir,
        seed=config['seed']
    )
    
    model.set_logger(new_logger)
    
    # Entrenar modelo
    start_time = time.time()
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=callbacks,
    )
    training_time = time.time() - start_time
    
    # Guardar modelo final
    final_model_path = os.path.join(models_dir, "final_model")
    model.save(final_model_path)
    
    # Evaluar modelo en todos los segmentos de validación
    logger.info("Evaluando modelo final en todos los segmentos de validación...")
    segment_metrics = []
    
    for i, val_env in enumerate(val_envs):
        logger.info(f"Evaluando en segmento {i+1}/5...")
        # Evaluar política
        mean_reward, std_reward = evaluate_policy(
            model, val_env, n_eval_episodes=5, deterministic=True
        )
        
        # Obtener métricas detalladas
        metrics = val_env.metrics.get_summary_stats()
        metrics['segment'] = i+1
        metrics['mean_reward'] = mean_reward
        metrics['std_reward'] = std_reward
        segment_metrics.append(metrics)
        
        # Guardar gráficos para este segmento
        segment_dir = os.path.join(result_dir, f"validation_segment_{i+1}")
        os.makedirs(segment_dir, exist_ok=True)
        val_env.plot_metrics(save_path=segment_dir)
        
        # Guardar historial de trades
        trades_df = val_env.get_trade_history_df()
        if not trades_df.empty:
            trades_df.to_csv(os.path.join(segment_dir, "trades.csv"), index=False)
    
    # Calcular métricas agregadas
    agg_metrics = {
        'mean_reward': np.mean([m['mean_reward'] for m in segment_metrics]),
        'win_rate': np.mean([m.get('win_rate', 0) for m in segment_metrics]),
        'profit_factor': np.mean([m.get('profit_factor', 0) for m in segment_metrics]),
        'max_drawdown': np.mean([m.get('max_drawdown', 0) for m in segment_metrics]),
        'sharpe_ratio': np.mean([m.get('sharpe_ratio', 0) for m in segment_metrics]),
        'sortino_ratio': np.mean([m.get('sortino_ratio', 0) for m in segment_metrics]),
        'total_trades': np.mean([m.get('total_trades', 0) for m in segment_metrics]),
        'market_exposure_pct': np.mean([m.get('market_exposure_pct', 0) for m in segment_metrics]),
        'avg_rr_ratio': np.mean([m.get('avg_rr_ratio', 0) for m in segment_metrics]),
        'segments': len(segment_metrics),
        'training_time': training_time,
        'total_timesteps': config['total_timesteps']
    }
    
    # Guardar métricas agregadas
    with open(os.path.join(result_dir, "validation_metrics.json"), 'w') as f:
        json.dump({
            'aggregated': agg_metrics,
            'segments': segment_metrics
        }, f, indent=4)
    
    # Log final de resultados
    logger.info(f"Entrenamiento completado en {training_time:.2f} segundos")
    logger.info(f"Métricas promedio en validación:")
    logger.info(f"  Reward: {agg_metrics['mean_reward']:.2f}")
    logger.info(f"  Win Rate: {agg_metrics['win_rate']:.2f}%")
    logger.info(f"  Profit Factor: {agg_metrics['profit_factor']:.2f}")
    logger.info(f"  Sharpe Ratio: {agg_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {agg_metrics['max_drawdown']:.2f}%")
    
    return result_dir, agg_metrics


if __name__ == "__main__":
    # Por defecto usar configuración estándar
    result_dir, metrics = train_model()
    
    print(f"Entrenamiento completado. Resultados en: {result_dir}")
    print(f"Métricas finales: {metrics}") 