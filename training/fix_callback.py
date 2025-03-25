#!/usr/bin/env python
"""
Script para solucionar problemas con los callbacks en el entrenamiento de TradeEvolvePPO.

Este script implementa una versión mejorada de los callbacks que corrige varios problemas:
1. Falta de atributos en los objetos de callback
2. Errores en la actualización de progreso
3. Problemas en la detección de entornos
"""

import os
import gymnasium as gym
import numpy as np
from typing import Dict, List, Union, Optional, Callable, Any, Tuple
import logging
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.logger import Logger
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TqdmCallback(BaseCallback):
    """
    Callback que muestra una barra de progreso usando tqdm.
    Incluye mejor manejo de errores y métricas en tiempo real.
    """
    
    def __init__(self, total_timesteps: int = None, verbose: int = 0):
        """
        Args:
            total_timesteps (int, optional): Número total de pasos para la barra.
            verbose (int, optional): Nivel de detalle.
        """
        super(TqdmCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.last_update = 0
        # Métricas para mostrar en la barra
        self.metrics = {
            'reward': 0.0,
            'trades': 0,
            'win_rate': 0.0
        }
    
    def _on_training_start(self) -> None:
        self.total_timesteps = self.total_timesteps or self.model.num_timesteps
        # Inicializar la barra de progreso
        self.pbar = tqdm(total=self.total_timesteps, desc="Entrenando")
        self.last_update = 0
    
    def _on_step(self) -> bool:
        # Actualizar la barra de progreso
        steps_done = self.model.num_timesteps - self.last_update
        self.pbar.update(steps_done)
        self.last_update = self.model.num_timesteps
        
        # Intentar obtener métricas de los últimos episodios
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            # Recompensa media
            self.metrics['reward'] = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            
            # Intentar obtener métricas de trading si están disponibles
            try:
                # Extraer entorno original de VecEnv
                if hasattr(self.model, 'env') and hasattr(self.model.env, 'envs'):
                    env = self.model.env.envs[0]
                    # Si el entorno está envuelto en Monitor, obtener el entorno original
                    while hasattr(env, 'env'):
                        env = env.env
                    
                    # Si tiene método get_performance_summary, obtener métricas
                    if hasattr(env, 'get_performance_summary'):
                        metrics = env.get_performance_summary()
                        self.metrics['trades'] = metrics.get('total_trades', 0)
                        self.metrics['win_rate'] = metrics.get('win_rate', 0) * 100
            except Exception as e:
                if self.verbose > 0:
                    logger.warning(f"Error obteniendo métricas de trading: {e}")
        
        # Actualizar descripción de la barra con métricas
        self.pbar.set_description(
            f"R:{self.metrics['reward']:.1f} | T:{self.metrics['trades']} | WR:{self.metrics['win_rate']:.1f}%"
        )
        
        return True
    
    def _on_training_end(self) -> None:
        # Cerrar la barra de progreso al finalizar
        if self.pbar is not None:
            self.pbar.close()


class RobustTradeCallback(BaseCallback):
    """
    Versión robusta del callback de trading que incluye mejor manejo de errores
    y compatibilidad con todas las versiones recientes de stable-baselines3.
    """
    
    def __init__(
        self,
        log_dir: str,
        save_path: str,
        save_interval: int = 10000,
        eval_interval: int = 5000,
        eval_env: gym.Env = None,
        eval_episodes: int = 5,
        verbose: int = 1
    ):
        """
        Inicializa el callback.
        
        Args:
            log_dir (str): Directorio para logs
            save_path (str): Directorio para guardar modelos
            save_interval (int): Frecuencia de guardado
            eval_interval (int): Frecuencia de evaluación
            eval_env (gym.Env): Entorno para evaluación
            eval_episodes (int): Número de episodios para evaluación
            verbose (int): Nivel de detalle
        """
        super(RobustTradeCallback, self).__init__(verbose)
        
        # Crear directorios si no existen
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        
        # Guardar parámetros
        self.log_dir = log_dir
        self.save_path = save_path
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        
        # Inicializar métricas
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.reward_history = []
        self.eval_count = 0
        
        # Estadísticas de trading
        self.trade_stats = {
            'total_trades': 0,
            'win_trades': 0,
            'loss_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_trade_pnl': 0.0
        }
        
        # Archivo de log
        self.log_file = os.path.join(log_dir, "training.log")
        with open(self.log_file, "w") as f:
            f.write("Timestep,Reward,Trades,WinRate,ProfitFactor,Status\n")
    
    def _on_step(self) -> bool:
        """
        Método llamado en cada paso del entrenamiento.
        
        Returns:
            bool: True para continuar entrenamiento, False para detener
        """
        # Guardar modelo periódicamente
        if self.n_calls % self.save_interval == 0:
            self._save_model()
        
        # Evaluar modelo periódicamente
        if self.eval_env is not None and self.n_calls % self.eval_interval == 0:
            self._evaluate_model()
        
        return True
    
    def _evaluate_model(self) -> None:
        """
        Evalúa el modelo actual y actualiza métricas.
        """
        # Aumentar contador de evaluaciones
        self.eval_count += 1
        
        try:
            # Asegurarse de que existe eval_env
            if self.eval_env is None:
                logger.warning("No se ha proporcionado eval_env. Saltando evaluación.")
                return
            
            # Importar funciones de evaluación
            from stable_baselines3.common.evaluation import evaluate_policy
            
            # Evaluar política con manejo de errores
            try:
                mean_reward, std_reward = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.eval_episodes,
                    deterministic=True
                )
            except Exception as e:
                logger.error(f"Error durante la evaluación de política: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Valores predeterminados en caso de error
                mean_reward = -999.0
                std_reward = 0.0
            
            self.last_mean_reward = mean_reward
            self.reward_history.append(mean_reward)
            
            # Extraer métricas de trading
            try:
                # Obtener env original (desenrollando posibles wrappers)
                env = self.eval_env
                # Si es VecEnv, obtener el primer entorno
                if hasattr(env, 'envs'):
                    env = env.envs[0]
                # Desenrollar wrappers
                while hasattr(env, 'env'):
                    env = env.env
                
                # Obtener métricas de rendimiento si están disponibles
                if hasattr(env, 'get_performance_summary'):
                    metrics = env.get_performance_summary()
                    self.trade_stats = {
                        'total_trades': metrics.get('total_trades', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'profit_factor': metrics.get('profit_factor', 0),
                        'avg_trade_pnl': metrics.get('average_pnl', 0)
                    }
            except Exception as e:
                logger.warning(f"Error obteniendo métricas de trading: {e}")
            
            # Actualizar mejor recompensa si corresponde
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                
                # Guardar mejor modelo
                best_model_path = os.path.join(self.save_path, "best_model")
                self.model.save(best_model_path)
                
                if self.verbose > 0:
                    print(f"Nuevo mejor modelo (#{self.eval_count}): Reward={mean_reward:.2f}, Trades={self.trade_stats['total_trades']}, WR={self.trade_stats['win_rate']*100:.1f}%")
            
            # Logging adicional
            self._log_to_file(mean_reward, std_reward)
            
        except Exception as e:
            logger.error(f"Error general en _evaluate_model: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _save_model(self) -> None:
        """
        Guarda el modelo actual.
        """
        try:
            # Ruta para el modelo actual
            model_path = os.path.join(self.save_path, f"model_{self.n_calls}_steps")
            
            # Guardar modelo
            self.model.save(model_path)
            
            if self.verbose > 1:  # Verbose alto
                print(f"Modelo guardado: {model_path}")
                
            # Logging
            self._log_message(f"Model saved: {model_path}")
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _log_to_file(self, mean_reward: float, std_reward: float) -> None:
        """
        Escribe métricas de evaluación en el archivo de log.
        
        Args:
            mean_reward (float): Recompensa media
            std_reward (float): Desviación estándar de la recompensa
        """
        try:
            with open(self.log_file, "a") as f:
                f.write(f"{self.n_calls},{mean_reward:.2f},{self.trade_stats['total_trades']},"
                        f"{self.trade_stats['win_rate']*100:.1f},{self.trade_stats['profit_factor']:.2f},"
                        f"{'BEST' if mean_reward >= self.best_mean_reward else 'REG'}\n")
        except Exception as e:
            logger.error(f"Error escribiendo en log: {e}")
    
    def _log_message(self, message: str) -> None:
        """
        Escribe un mensaje al archivo de log.
        
        Args:
            message (str): Mensaje a escribir
        """
        try:
            with open(self.log_file, "a") as f:
                f.write(f"# {message}\n")
        except Exception as e:
            logger.error(f"Error escribiendo mensaje en log: {e}")
    
    def on_training_end(self) -> None:
        """
        Método llamado al final del entrenamiento.
        """
        # Guardar modelo final
        try:
            final_model_path = os.path.join(self.save_path, "final_model")
            self.model.save(final_model_path)
            
            if self.verbose > 0:
                print(f"Modelo final guardado: {final_model_path}")
            
            self._log_message(f"Final model saved: {final_model_path}")
            
            # Estadísticas finales
            if len(self.reward_history) > 0:
                # Calcular estadísticas
                mean_reward = np.mean(self.reward_history)
                max_reward = np.max(self.reward_history)
                min_reward = np.min(self.reward_history)
                
                # Escribir resumen
                self._log_message(f"Training completed: {self.n_calls} steps")
                self._log_message(f"Best reward: {self.best_mean_reward:.2f}")
                self._log_message(f"Average reward: {mean_reward:.2f}")
                self._log_message(f"Max reward: {max_reward:.2f}")
                self._log_message(f"Min reward: {min_reward:.2f}")
                
                if self.verbose > 0:
                    print(f"Entrenamiento completado: {self.n_calls} pasos")
                    print(f"Mejor recompensa: {self.best_mean_reward:.2f}")
                    print(f"Recompensa media: {mean_reward:.2f}")
                    
        except Exception as e:
            logger.error(f"Error en on_training_end: {e}")
            import traceback
            logger.error(traceback.format_exc())


class ActionDistributionCallback(BaseCallback):
    """
    Callback para monitorear la distribución de acciones durante el entrenamiento.
    Ayuda a detectar problemas de sesgo en el comportamiento del agente.
    """
    
    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        """
        Inicializa el callback.
        
        Args:
            log_interval (int): Intervalo de pasos para registrar distribución
            verbose (int): Nivel de detalle
        """
        super(ActionDistributionCallback, self).__init__(verbose)
        self.log_interval = log_interval
        
        # Contadores para acciones
        self.action_counts = {'hold': 0, 'buy': 0, 'sell': 0}
        self.total_actions = 0
        
        # Historial reciente para detectar estancamiento
        self.recent_actions = []
        self.max_recent = 100  # Últimas 100 acciones
        
        # Historial de distribución
        self.distribution_history = []
    
    def _on_step(self) -> bool:
        """
        Método llamado en cada paso.
        
        Returns:
            bool: True para continuar entrenamiento
        """
        # Obtener acción actual
        if self.locals.get('actions') is not None:
            action = self.locals['actions'][0]  # Primera acción (asumiendo entorno único)
            
            # Clasificar la acción
            if isinstance(action, np.ndarray):
                if len(action.shape) > 0 and action.shape[0] > 0:
                    # Acciones continuas (common case)
                    if action[0] > 0.5:  # Buy/Long
                        self.action_counts['buy'] += 1
                        self.recent_actions.append('buy')
                    elif action[0] < -0.5:  # Sell/Short
                        self.action_counts['sell'] += 1
                        self.recent_actions.append('sell')
                    else:  # Hold/Close
                        self.action_counts['hold'] += 1
                        self.recent_actions.append('hold')
                else:
                    # Fallback para acciones escalares
                    self.action_counts['hold'] += 1
                    self.recent_actions.append('hold')
            else:
                # Acciones discretas
                if action == 1:  # Buy/Long
                    self.action_counts['buy'] += 1
                    self.recent_actions.append('buy')
                elif action == 2:  # Sell/Short
                    self.action_counts['sell'] += 1
                    self.recent_actions.append('sell')
                else:  # Hold/Close
                    self.action_counts['hold'] += 1
                    self.recent_actions.append('hold')
            
            # Mantener solo las últimas max_recent acciones
            if len(self.recent_actions) > self.max_recent:
                self.recent_actions.pop(0)
            
            # Actualizar total
            self.total_actions += 1
        
        # Registrar distribución periódicamente
        if self.n_calls % self.log_interval == 0:
            if self.total_actions > 0:
                # Calcular distribución actual
                distribution = {
                    'step': self.n_calls,
                    'hold_pct': self.action_counts['hold'] / self.total_actions * 100,
                    'buy_pct': self.action_counts['buy'] / self.total_actions * 100,
                    'sell_pct': self.action_counts['sell'] / self.total_actions * 100
                }
                
                # Guardar en historial
                self.distribution_history.append(distribution)
                
                # Verificar sesgo extremo en acciones recientes
                if len(self.recent_actions) >= 50:  # Al menos 50 acciones para análisis
                    recent_counts = {'hold': 0, 'buy': 0, 'sell': 0}
                    for act in self.recent_actions:
                        recent_counts[act] += 1
                    
                    # Calcular porcentajes recientes
                    recent_total = len(self.recent_actions)
                    recent_hold_pct = recent_counts['hold'] / recent_total * 100
                    recent_buy_pct = recent_counts['buy'] / recent_total * 100
                    recent_sell_pct = recent_counts['sell'] / recent_total * 100
                    
                    # Detectar sesgo extremo (>80% en una acción)
                    max_pct = max(recent_hold_pct, recent_buy_pct, recent_sell_pct)
                    if max_pct > 80:
                        if max_pct == recent_hold_pct:
                            action_type = 'HOLD'
                        elif max_pct == recent_buy_pct:
                            action_type = 'BUY'
                        else:
                            action_type = 'SELL'
                        
                        if self.verbose > 0:
                            print(f"\n⚠️ ALERTA: Sesgo extremo hacia {action_type} ({max_pct:.1f}%)")
                            print(f"  Hold: {recent_hold_pct:.1f}%, Buy: {recent_buy_pct:.1f}%, Sell: {recent_sell_pct:.1f}%")
                
                # Mostrar distribución actual
                if self.verbose > 0:
                    print(f"Paso {self.n_calls}: Hold={distribution['hold_pct']:.1f}%, "
                          f"Buy={distribution['buy_pct']:.1f}%, Sell={distribution['sell_pct']:.1f}%")
    
        return True
    
    def report_distribution(self) -> Dict[str, float]:
        """
        Proporciona un informe de la distribución actual de acciones.
        
        Returns:
            Dict[str, float]: Distribución de acciones en porcentajes
        """
        if self.total_actions == 0:
            return {'hold_pct': 0.0, 'buy_pct': 0.0, 'sell_pct': 0.0}
        
        return {
            'hold_pct': self.action_counts['hold'] / self.total_actions * 100,
            'buy_pct': self.action_counts['buy'] / self.total_actions * 100,
            'sell_pct': self.action_counts['sell'] / self.total_actions * 100
        }
    
    def get_recent_bias(self) -> Dict[str, Any]:
        """
        Analiza el sesgo en las acciones recientes.
        
        Returns:
            Dict[str, Any]: Información sobre el sesgo reciente
        """
        if len(self.recent_actions) < 20:
            return {'bias_detected': False}
        
        # Contar acciones recientes
        recent_counts = {'hold': 0, 'buy': 0, 'sell': 0}
        for act in self.recent_actions:
            recent_counts[act] += 1
        
        # Calcular porcentajes
        recent_total = len(self.recent_actions)
        percentages = {
            k: (v / recent_total * 100) for k, v in recent_counts.items()
        }
        
        # Determinar si hay sesgo
        max_pct = max(percentages.values())
        max_action = max(percentages.items(), key=lambda x: x[1])[0]
        
        return {
            'bias_detected': max_pct > 70,
            'max_action': max_action,
            'max_percentage': max_pct,
            'percentages': percentages
        }


def create_combined_callback(
    save_path: str,
    eval_env = None,
    log_dir: str = None,
    total_timesteps: int = None,
    eval_interval: int = 20000,
    save_interval: int = 50000,
    verbose: int = 1
) -> BaseCallback:
    """
    Crea un callback combinado con todas las funcionalidades necesarias.
    
    Args:
        save_path (str): Directorio para guardar modelos
        eval_env: Entorno para evaluación
        log_dir (str, optional): Directorio para logs
        total_timesteps (int, optional): Total de pasos para la barra de progreso
        eval_interval (int, optional): Intervalo de evaluación
        save_interval (int, optional): Intervalo de guardado
        verbose (int, optional): Nivel de detalle
    
    Returns:
        BaseCallback: Callback combinado
    """
    # Crear directorios si no existen
    if log_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/training_{timestamp}"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    
    # Crear callbacks
    callbacks = []
    
    # Callback de barra de progreso
    tqdm_callback = TqdmCallback(total_timesteps=total_timesteps, verbose=verbose)
    callbacks.append(tqdm_callback)
    
    # Callback de distribución de acciones
    action_callback = ActionDistributionCallback(
        log_interval=eval_interval // 4,  # 4 veces más frecuente
        verbose=verbose
    )
    callbacks.append(action_callback)
    
    # Callback principal para evaluación y guardado
    trade_callback = RobustTradeCallback(
        log_dir=log_dir,
        save_path=save_path,
        save_interval=save_interval,
        eval_interval=eval_interval,
        eval_env=eval_env,
        eval_episodes=5,
        verbose=verbose
    )
    callbacks.append(trade_callback)
    
    # Crear callback combinado
    from stable_baselines3.common.callbacks import CallbackList
    combined_callback = CallbackList(callbacks)
    
    return combined_callback


# Función principal para usar en el entrenamiento
def get_robust_callback(
    config: Dict[str, Any],
    eval_env = None
) -> BaseCallback:
    """
    Obtiene un callback robusto basado en la configuración del entrenamiento.
    
    Args:
        config (Dict[str, Any]): Configuración completa
        eval_env: Entorno de evaluación
    
    Returns:
        BaseCallback: Callback combinado
    """
    training_config = config.get('training_config', {})
    
    return create_combined_callback(
        save_path=training_config.get('save_path', './models'),
        eval_env=eval_env,
        log_dir=training_config.get('log_path', './logs'),
        total_timesteps=training_config.get('total_timesteps', 1000000),
        eval_interval=training_config.get('eval_freq', 20000),
        save_interval=training_config.get('checkpoint_freq', 50000),
        verbose=training_config.get('verbose', 1)
    )
