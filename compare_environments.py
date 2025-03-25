#!/usr/bin/env python
"""
Script para comparar el comportamiento de los dos entornos de trading:
1. TradingEnv (original)
2. SimpleTradingEnv (simplificado)

Este script ejecuta pruebas idénticas en ambos entornos y compara los resultados.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from gym import spaces

# Importar componentes del proyecto
from config.config import DATA_CONFIG, ENV_CONFIG, REWARD_CONFIG
from data.data_loader import DataLoader
from environment.trading_env import TradingEnv
from environment.simple_trading_env import SimpleTradingEnv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("compare_environments.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("compare_envs")

def run_environment_test(env_class, env_name, data, config, num_steps=500, seed=42):
    """
    Ejecutar prueba en un entorno específico.
    
    Args:
        env_class: Clase del entorno a probar
        env_name: Nombre del entorno para el registro
        data: Datos para el entorno
        config: Configuración
        num_steps: Número de pasos
        seed: Semilla para reproducibilidad
        
    Returns:
        dict: Resultados de la prueba
    """
    logger.info(f"Ejecutando prueba en {env_name}...")
    
    # Crear entorno
    env = env_class(
        data=data,
        config=config,
        window_size=config.get('window_size', 60),
        mode='train'
    )
    
    # Resultados a recopilar
    results = {
        'env_name': env_name,
        'rewards': [],
        'balances': [],
        'positions': [],
        'action_results': [],
        'trades': []
    }
    
    # Reiniciar entorno
    np.random.seed(seed)
    observation, _ = env.reset(seed=seed)
    
    # Patrón de prueba: alternar entre 1 (compra) y 2 (venta) cada 10 pasos
    pattern = []
    for i in range(num_steps // 20 + 1):
        pattern.extend([1] * 10)  # 10 pasos de compra
        pattern.extend([2] * 10)  # 10 pasos de venta
    pattern = pattern[:num_steps]
    
    # Ejecutar pasos
    for step in tqdm(range(num_steps), desc=f"Probando {env_name}"):
        # Determinar acción
        action = pattern[step]
        
        # Para acciones continuas, convertir
        if hasattr(env, 'action_space_type') and env.action_space_type == 'continuous':
            if action == 1:  # Comprar
                action_value = np.array([1.0])
            elif action == 2:  # Vender
                action_value = np.array([-1.0])
            else:  # Mantener
                action_value = np.array([0.0])
        elif isinstance(env.action_space, spaces.Box):
            # TradingEnv usa un espacio de acción Box con 2 dimensiones
            if action == 1:  # Comprar
                action_value = np.array([1.0, 0.0])
            elif action == 2:  # Vender
                action_value = np.array([-1.0, 0.0]) 
            else:  # Mantener
                action_value = np.array([0.0, 0.0])
        else:
            action_value = action
        
        # Ejecutar paso
        observation, reward, done, truncated, info = env.step(action_value)
        
        # Registrar resultados
        results['rewards'].append(reward)
        results['balances'].append(info.get('balance', 0.0))
        
        # Obtener posición actual
        if hasattr(env, 'current_position'):
            position = env.current_position
        else:
            position = info.get('position', 0)
        
        results['positions'].append(position)
        
        # Verificar si la posición cambió como se esperaba
        expected_position = 0
        if action == 1:
            expected_position = 1
        elif action == 2:
            expected_position = -1
        
        # Solo considerar éxito si la posición es exactamente la esperada
        # O si ya hay una posición del mismo signo
        action_success = (position == expected_position) or \
                         (action == 1 and position > 0) or \
                         (action == 2 and position < 0)
        
        results['action_results'].append({
            'step': step,
            'action': action,
            'position': position,
            'expected': expected_position,
            'success': action_success
        })
        
        # Verificar terminación
        if done:
            logger.info(f"Episodio terminado en paso {step}")
            break
    
    # Obtener resumen de rendimiento
    if hasattr(env, 'get_performance_summary'):
        performance = env.get_performance_summary()
        results['performance'] = performance
    else:
        results['performance'] = {"total_trades": len(getattr(env, 'trades', [])), "win_rate": 0.0}
    
    # Obtener lista de trades
    if hasattr(env, 'trades'):
        results['trades'] = env.trades
    elif hasattr(env, 'trade_history'):
        results['trades'] = env.trade_history
    else:
        results['trades'] = []
    
    logger.info(f"Prueba completada en {env_name}. Trades: {len(results['trades'])}")
    
    return results

def plot_comparison(original_results, simple_results, save_path=None):
    """
    Visualizar comparación entre los dos entornos.
    
    Args:
        original_results: Resultados del entorno original
        simple_results: Resultados del entorno simplificado
        save_path: Ruta para guardar el gráfico
    """
    plt.figure(figsize=(18, 15))
    
    # Plot 1: Balance
    plt.subplot(3, 1, 1)
    plt.plot(original_results['balances'], label='Original')
    plt.plot(simple_results['balances'], label='Simplificado')
    plt.title('Balance durante la prueba')
    plt.ylabel('Balance')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Posición
    plt.subplot(3, 1, 2)
    plt.plot(original_results['positions'], label='Original')
    plt.plot(simple_results['positions'], label='Simplificado')
    plt.title('Posición durante la prueba')
    plt.ylabel('Posición')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Recompensas
    plt.subplot(3, 1, 3)
    plt.plot(original_results['rewards'], label='Original')
    plt.plot(simple_results['rewards'], label='Simplificado')
    plt.title('Recompensas durante la prueba')
    plt.xlabel('Paso')
    plt.ylabel('Recompensa')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico guardado en {save_path}")
    else:
        plt.show()

def analyze_action_results(results, env_name):
    """
    Analizar resultados de las acciones.
    
    Args:
        results: Resultados del entorno
        env_name: Nombre del entorno
        
    Returns:
        dict: Análisis de resultados
    """
    # Contar éxitos y fracasos por tipo de acción
    action_counts = {1: 0, 2: 0}
    action_success = {1: 0, 2: 0}
    
    for result in results['action_results']:
        action = result['action']
        if action in [1, 2]:
            action_counts[action] += 1
            if result['success']:
                action_success[action] += 1
    
    # Calcular tasas de éxito
    success_rates = {}
    for action in [1, 2]:
        if action_counts[action] > 0:
            success_rates[action] = action_success[action] / action_counts[action] * 100
        else:
            success_rates[action] = 0.0
    
    # Análisis general
    total_actions = sum(action_counts.values())
    total_success = sum(action_success.values())
    overall_success_rate = total_success / total_actions * 100 if total_actions > 0 else 0.0
    
    logger.info(f"=== Análisis de acciones para {env_name} ===")
    logger.info(f"Acciones de compra (1): {action_counts[1]} intentos, {action_success[1]} éxitos ({success_rates[1]:.2f}%)")
    logger.info(f"Acciones de venta (2): {action_counts[2]} intentos, {action_success[2]} éxitos ({success_rates[2]:.2f}%)")
    logger.info(f"Tasa de éxito global: {overall_success_rate:.2f}%")
    
    return {
        'env_name': env_name,
        'action_counts': action_counts,
        'action_success': action_success,
        'success_rates': success_rates,
        'overall_success_rate': overall_success_rate
    }

def main():
    """Función principal para ejecutar la comparación."""
    # Cargar datos
    data_file = "data/NQ_06-25_combined_20250320_225417.csv"
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        data_file = sys.argv[1]
    
    print(f"Usando archivo de datos: {data_file}")
    
    data_loader = DataLoader(DATA_CONFIG)
    train_data, val_data, test_data = data_loader.prepare_data(data_file)
    
    # Configuración modificada para pruebas
    env_config = ENV_CONFIG.copy()
    env_config.update({
        'log_reward_components': True,
        'force_action_prob': 0.0  # Desactivar forzado de acciones para pruebas
    })
    
    # Ejecutar pruebas
    original_results = run_environment_test(
        TradingEnv, "Entorno Original", train_data, env_config
    )
    
    simple_results = run_environment_test(
        SimpleTradingEnv, "Entorno Simplificado", train_data, env_config
    )
    
    # Analizar resultados
    original_analysis = analyze_action_results(original_results, "Entorno Original")
    simple_analysis = analyze_action_results(simple_results, "Entorno Simplificado")
    
    # Visualizar comparación
    plot_comparison(original_results, simple_results, "comparacion_entornos.png")
    
    # Mostrar estadísticas finales
    print("\n" + "="*50)
    print("COMPARACIÓN DE ENTORNOS DE TRADING")
    print("="*50)
    
    print("\n=== Entorno Original ===")
    print(f"Operaciones completadas: {len(original_results['trades'])}")
    print(f"Balance final: {original_results['balances'][-1]:.2f}")
    print(f"Tasa de éxito de acciones: {original_analysis['overall_success_rate']:.2f}%")
    
    print("\n=== Entorno Simplificado ===")
    print(f"Operaciones completadas: {len(simple_results['trades'])}")
    print(f"Balance final: {simple_results['balances'][-1]:.2f}")
    print(f"Tasa de éxito de acciones: {simple_analysis['overall_success_rate']:.2f}%")
    
    # Identificar problemas
    print("\n=== DIAGNÓSTICO ===")
    
    # 1. Verificar operaciones
    ops_diff = len(simple_results['trades']) - len(original_results['trades'])
    if ops_diff > 0:
        print(f"⚠️ El entorno simplificado ejecuta {ops_diff} operaciones más que el original")
        print("   Esto indica que hay restricciones excesivas en el entorno original")
    elif ops_diff < 0:
        print(f"⚠️ El entorno simplificado ejecuta {abs(ops_diff)} operaciones menos que el original")
        print("   Esto indica un posible problema en la implementación simplificada")
    else:
        print("✅ Ambos entornos ejecutan el mismo número de operaciones")
    
    # 2. Verificar cambios de posición
    pos_original = np.array(original_results['positions'])
    pos_simple = np.array(simple_results['positions'])
    pos_changes_original = np.sum(np.abs(np.diff(pos_original) != 0))
    pos_changes_simple = np.sum(np.abs(np.diff(pos_simple) != 0))
    
    if pos_changes_original < pos_changes_simple:
        print(f"⚠️ El entorno original cambia de posición {pos_changes_original} veces vs {pos_changes_simple} en el simplificado")
        print("   Esto confirma restricciones excesivas en el entorno original")
    
    # 3. Análisis de rechazos
    if original_analysis['overall_success_rate'] < 90 and simple_analysis['overall_success_rate'] > 90:
        print(f"⚠️ Tasa de éxito muy diferente: {original_analysis['overall_success_rate']:.2f}% vs {simple_analysis['overall_success_rate']:.2f}%")
        print("   Identificado problema fundamental en el entorno original que rechaza operaciones válidas")
    
    # Conclusión
    print("\n=== CONCLUSIÓN ===")
    if len(original_results['trades']) == 0 and len(simple_results['trades']) > 0:
        print("PROBLEMA CRÍTICO IDENTIFICADO: El entorno original no ejecuta ninguna operación, mientras que el simplificado sí lo hace")
        print("Revisar el código del entorno original para identificar y corregir las restricciones que impiden ejecutar operaciones")
    elif len(original_results['trades']) > 0:
        print("El entorno original está ejecutando operaciones. El problema puede estar en:")
        print("1. Cómo se registran las operaciones")
        print("2. Cómo se calculan las métricas de rendimiento")
        print("3. Cómo se comunican estas operaciones durante el entrenamiento")
    
    print("\nComprobación completada. Revise los resultados y archivos de registro para más detalles.")

if __name__ == "__main__":
    main() 