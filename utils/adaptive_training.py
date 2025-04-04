"""
Sistemas adaptativos para mejorar la estabilidad y rendimiento del entrenamiento
en el proyecto TradeEvolvePPO.
"""

import numpy as np
import os
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class AdaptiveEntropy:
    """
    Sistema de gestión adaptativa de entropía que ajusta dinámicamente
    el coeficiente de entropía basado en el rendimiento y la volatilidad.
    """
    def __init__(self, initial_entropy=0.3, min_entropy=0.15, 
                 base_decay=0.995, window_size=50000):
        """
        Inicializa el sistema de entropía adaptativa.
        
        Args:
            initial_entropy (float): Valor inicial de entropía.
            min_entropy (float): Valor mínimo permitido de entropía.
            base_decay (float): Factor de decaimiento base para reducción gradual.
            window_size (int): Tamaño de la ventana para el historial de recompensas.
        """
        self.initial_entropy = initial_entropy
        self.min_entropy = min_entropy
        self.base_decay = base_decay
        self.window_size = window_size
        self.reward_history = []
        self.entropy_history = []
        self.current_entropy = initial_entropy
        
    def update(self, step, reward):
        """
        Actualiza el valor de entropía basado en el rendimiento reciente.
        
        Args:
            step (int): Paso actual de entrenamiento.
            reward (float): Recompensa actual.
            
        Returns:
            float: Valor actualizado de entropía.
        """
        self.reward_history.append(reward)
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)
            
        # Calcular métricas
        if len(self.reward_history) > 10:
            recent_rewards = self.reward_history[-min(10000, len(self.reward_history)):]
            mean_reward = np.mean(recent_rewards)
            reward_std = np.std(recent_rewards)
            
            # Ajuste dinámico basado en el rendimiento
            if mean_reward < -100000:  # Rendimiento pobre
                new_entropy = min(self.initial_entropy * 0.8, 
                                 self.current_entropy * 1.2)
                logger.info(f"Paso {step}: Aumentando entropía a {new_entropy:.4f} debido a rendimiento pobre")
            elif mean_reward > -50000:  # Buen rendimiento
                new_entropy = max(self.min_entropy,
                                 self.current_entropy * self.base_decay)
                logger.info(f"Paso {step}: Reduciendo entropía a {new_entropy:.4f} debido a buen rendimiento")
            else:
                # Mantener la entropía actual con una pequeña reducción
                new_entropy = max(self.min_entropy,
                               self.current_entropy * (self.base_decay + 0.003))
            
            # Ajuste basado en la volatilidad
            if reward_std > 100000:  # Alta volatilidad
                volatility_adj = min(self.initial_entropy * 0.9,
                                   self.current_entropy * 1.1)
                logger.info(f"Paso {step}: Ajustando entropía a {volatility_adj:.4f} debido a alta volatilidad")
                new_entropy = volatility_adj
                
            self.current_entropy = new_entropy
        else:
            # No hay suficientes datos para ajustar, aplicar decaimiento estándar
            self.current_entropy = max(self.min_entropy,
                                     self.current_entropy * self.base_decay)
        
        # Guardar historial de entropía para visualización
        self.entropy_history.append((step, self.current_entropy))
        
        return self.current_entropy
    
    def plot_entropy_history(self, output_dir):
        """
        Genera un gráfico de la evolución de la entropía durante el entrenamiento.
        
        Args:
            output_dir (str): Directorio donde guardar el gráfico.
        """
        if len(self.entropy_history) < 2:
            return
            
        steps, entropies = zip(*self.entropy_history)
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, entropies, label='Entropía', color='blue')
        plt.title('Evolución Adaptativa de la Entropía')
        plt.xlabel('Pasos de Entrenamiento')
        plt.ylabel('Valor de Entropía')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'entropy_evolution.png'))
        plt.close()


class EnhancedEarlyStopping:
    """
    Sistema de early stopping mejorado que considera tanto la falta de mejora
    como la estabilidad de las recompensas.
    """
    def __init__(self, patience=50000, min_delta=1000, stagnation_threshold=10000):
        """
        Inicializa el sistema de early stopping.
        
        Args:
            patience (int): Número de pasos para esperar mejoras.
            min_delta (float): Mejora mínima requerida para considerar progreso.
            stagnation_threshold (int): Umbral de estancamiento para verificaciones adicionales.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.stagnation_threshold = stagnation_threshold
        self.best_reward = float('-inf')
        self.best_step = 0
        self.stagnation_counter = 0
        self.reward_history = []
        
    def update(self, step, reward):
        """
        Actualiza el estado del early stopping con una nueva recompensa.
        
        Args:
            step (int): Paso actual de entrenamiento.
            reward (float): Recompensa actual.
        """
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
            
        if reward > self.best_reward + self.min_delta:
            self.best_reward = reward
            self.best_step = step
            self.stagnation_counter = 0
            logger.info(f"Paso {step}: Nueva mejor recompensa: {reward:.2f}")
        else:
            self.stagnation_counter += 1
            
            # Verificación adicional de estancamiento prolongado
            if self.stagnation_counter % self.stagnation_threshold == 0:
                logger.warning(f"Paso {step}: Estancamiento prolongado ({self.stagnation_counter} pasos sin mejora)")
    
    def should_stop(self, step):
        """
        Determina si el entrenamiento debe detenerse según los criterios de early stopping.
        
        Args:
            step (int): Paso actual de entrenamiento.
            
        Returns:
            bool: True si el entrenamiento debe detenerse, False en caso contrario.
        """
        if self.stagnation_counter >= self.patience:
            logger.warning(f"Early stopping activado en paso {step}: {self.stagnation_counter} pasos sin mejora")
            return True
            
        # Verificación adicional: deterioro severo del rendimiento
        if len(self.reward_history) > 20:
            recent_mean = np.mean(self.reward_history[-10:])
            historical_mean = np.mean(self.reward_history)
            
            if recent_mean < historical_mean * 0.5 and step > self.best_step + self.patience/2:
                logger.warning(f"Early stopping activado en paso {step}: Deterioro severo del rendimiento")
                return True
                
        return False


class SmartCheckpointing:
    """
    Sistema de checkpointing inteligente que guarda modelos basado en rendimiento
    y diversidad de puntos de guardado.
    """
    def __init__(self, save_interval=50000, max_checkpoints=5, min_reward_diff=5000):
        """
        Inicializa el sistema de checkpointing.
        
        Args:
            save_interval (int): Intervalo base entre guardados.
            max_checkpoints (int): Número máximo de checkpoints a mantener.
            min_reward_diff (float): Diferencia mínima de recompensa para guardar un checkpoint.
        """
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        self.min_reward_diff = min_reward_diff
        self.checkpoints = []
        self.last_save_step = 0
        self.best_reward = float('-inf')
        
    def should_save(self, step, reward):
        """
        Determina si se debe guardar un checkpoint en el paso actual.
        
        Args:
            step (int): Paso actual de entrenamiento.
            reward (float): Recompensa actual.
            
        Returns:
            bool: True si se debe guardar un checkpoint, False en caso contrario.
        """
        # Guardar periódicamente según el intervalo
        interval_save = step % self.save_interval == 0 and step > 0
        
        # Guardar si hay una mejora significativa
        reward_save = reward > self.best_reward + self.min_reward_diff
        
        # Asegurar un mínimo de pasos entre guardados (evitar guardar demasiado frecuentemente)
        min_step_diff = step - self.last_save_step >= self.save_interval/5
        
        should_save = (interval_save or reward_save) and min_step_diff
        
        if should_save:
            # Actualizar referencias
            if reward > self.best_reward:
                self.best_reward = reward
                
            self.checkpoints.append((step, reward))
            self.last_save_step = step
            
            # Mantener solo los mejores checkpoints
            if len(self.checkpoints) > self.max_checkpoints:
                # Ordenar por recompensa (de menor a mayor) y eliminar el peor
                self.checkpoints.sort(key=lambda x: x[1])
                removed = self.checkpoints.pop(0)
                logger.info(f"Eliminando checkpoint del paso {removed[0]} (recompensa: {removed[1]:.2f})")
            
            logger.info(f"Guardando checkpoint en paso {step} (recompensa: {reward:.2f})")
            
        return should_save
    
    def get_checkpoint_names(self):
        """
        Obtiene los nombres de los checkpoints actuales.
        
        Returns:
            list: Lista de nombres de checkpoints.
        """
        return [f"checkpoint_{step}" for step, _ in self.checkpoints]
