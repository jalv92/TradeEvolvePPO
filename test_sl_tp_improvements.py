"""
Script para probar las mejoras implementadas en el sistema de stop loss y take profit.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import random

from environment.trading_env import TradingEnv
from environment.enhanced_trading_env import EnhancedTradingEnv
from utils.trade_diagnostics import analyze_trades_from_env, diagnose_sl_tp_behavior
from config.config import BASE_CONFIG, ENV_CONFIG

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples=5000, volatility=0.01, trend=0.0001):
    """
    Genera datos sintéticos de precios para pruebas.
    
    Args:
        n_samples: Número de muestras
        volatility: Volatilidad de los precios
        trend: Tendencia de los precios
        
    Returns:
        pd.DataFrame: DataFrame con datos OHLCV
    """
    # Generar precios de cierre con caminata aleatoria
    np.random.seed(42)
    close = 100.0
    closes = [close]
    
    for _ in range(n_samples - 1):
        # Añadir ruido aleatorio y tendencia
        close = close * (1 + np.random.normal(trend, volatility))
        closes.append(close)
    
    closes = np.array(closes)
    
    # Generar OHLC basado en los precios de cierre
    highs = closes * (1 + np.random.uniform(0, volatility, n_samples))
    lows = closes * (1 - np.random.uniform(0, volatility, n_samples))
    opens = closes[:-1].copy()
    opens = np.append([closes[0]], opens)
    
    # Generar volumen aleatorio
    volumes = np.random.uniform(1000, 5000, n_samples)
    
    # Crear DataFrame con índice numérico (no fechas)
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    # Asegurar que el índice sea numérico para evitar problemas de fechas
    df.index = np.arange(len(df))
    
    return df

def test_environment(env_class, config, data, n_steps=1000, random_seed=42):
    """
    Ejecuta una prueba del entorno con acciones aleatorias.
    
    Args:
        env_class: Clase del entorno a probar
        config: Configuración del entorno
        data: DataFrame con datos OHLCV
        n_steps: Número de pasos a ejecutar
        random_seed: Semilla aleatoria
        
    Returns:
        env: Entorno después de la prueba
    """
    # Crear entorno
    env = env_class(
        data=data,
        config=config,
        initial_balance=100000.0,
        window_size=60,
        mode='eval'  # Usar modo eval para forzar más operaciones
    )
    
    # Reiniciar entorno
    np.random.seed(random_seed)
    random.seed(random_seed)
    obs, _ = env.reset(seed=random_seed)
    
    # Ejecutar pasos
    for i in range(n_steps):
        # Tomar acción aleatoria
        if i % 10 == 0:  # Cada 10 pasos, forzar una acción más extrema
            action = np.random.uniform(-0.8, 0.8)
        else:
            action = np.random.uniform(-0.3, 0.3)
            
        # Ejecutar paso
        obs, reward, done, truncated, info = env.step(action)
        
        if done:
            break
    
    return env

def main():
    """
    Función principal para ejecutar las pruebas.
    """
    # Generar datos sintéticos
    logger.info("Generando datos sintéticos...")
    data = generate_synthetic_data(n_samples=5000)
    
    # Configuración base
    config = BASE_CONFIG.copy()
    config.update(ENV_CONFIG)
    
    # Configuración para pruebas
    test_config = config.copy()
    test_config.update({
        'min_hold_steps': 10,  # Período mínimo de mantener posición
        'force_min_hold': True,  # Forzar período mínimo
        'sl_buffer_ticks': 5,  # Buffer para SL
        'tp_buffer_ticks': 5,  # Buffer para TP
        'force_action_prob': 0.3,  # Probabilidad de forzar acción
        'log_ticks': True,  # Activar logging de ticks
    })
    
    # Crear directorio para resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/sl_tp_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Probar entorno original
    logger.info("Probando entorno original...")
    env_original = test_environment(TradingEnv, test_config, data, n_steps=2000)
    
    # Probar entorno mejorado
    logger.info("Probando entorno mejorado...")
    env_enhanced = test_environment(EnhancedTradingEnv, test_config, data, n_steps=2000)
    
    # Analizar resultados
    logger.info("Analizando resultados del entorno original...")
    analyzer_original, results_original = analyze_trades_from_env(
        env_original, output_dir=f"{output_dir}/original"
    )
    
    logger.info("Analizando resultados del entorno mejorado...")
    analyzer_enhanced, results_enhanced = analyze_trades_from_env(
        env_enhanced, output_dir=f"{output_dir}/enhanced"
    )
    
    # Mostrar resultados
    logger.info("\n=== RESULTADOS DE LA PRUEBA ===")
    
    # Duración de operaciones
    logger.info("\n--- Duración de Operaciones ---")
    logger.info(f"Original: Media={results_original['duration_stats']['mean']:.1f}, % < 5 barras={results_original['duration_stats']['pct_below_5']:.1f}%")
    logger.info(f"Mejorado: Media={results_enhanced['duration_stats']['mean']:.1f}, % < 5 barras={results_enhanced['duration_stats']['pct_below_5']:.1f}%")
    
    # Tipos de cierre
    logger.info("\n--- Tipos de Cierre ---")
    logger.info(f"Original: SL={results_original['sl_tp_stats']['sl_count']}, TP={results_original['sl_tp_stats']['tp_count']}, Manual={results_original['sl_tp_stats']['manual_count']}")
    logger.info(f"Mejorado: SL={results_enhanced['sl_tp_stats']['sl_count']}, TP={results_enhanced['sl_tp_stats']['tp_count']}, Manual={results_enhanced['sl_tp_stats']['manual_count']}")
    
    # Ratio TP/SL
    logger.info("\n--- Ratio TP/SL ---")
    logger.info(f"Original: {results_original['sl_tp_stats']['tp_sl_ratio']:.2f}")
    logger.info(f"Mejorado: {results_enhanced['sl_tp_stats']['tp_sl_ratio']:.2f}")
    
    # Ticks
    logger.info("\n--- Ticks ---")
    logger.info(f"Original: Positivos={results_original['ticks_stats']['positive_mean']:.1f}, Negativos={results_original['ticks_stats']['negative_mean']:.1f}, Ratio={results_original['ticks_stats']['ratio_mean']:.2f}")
    logger.info(f"Mejorado: Positivos={results_enhanced['ticks_stats']['positive_mean']:.1f}, Negativos={results_enhanced['ticks_stats']['negative_mean']:.1f}, Ratio={results_enhanced['ticks_stats']['ratio_mean']:.2f}")
    
    # Crear gráfico comparativo de duración
    plt.figure(figsize=(12, 6))
    
    # Extraer duraciones
    durations_original = [t.get('duration', 0) for t in env_original.trades if 'duration' in t]
    durations_enhanced = [t.get('duration', 0) for t in env_enhanced.trades if 'duration' in t]
    
    plt.hist(durations_original, bins=20, alpha=0.5, label='Original', color='blue')
    plt.hist(durations_enhanced, bins=20, alpha=0.5, label='Mejorado', color='green')
    plt.axvline(x=5, color='red', linestyle='--', label='5 barras')
    plt.axvline(x=10, color='orange', linestyle='--', label='10 barras')
    plt.title('Comparación de Duración de Operaciones')
    plt.xlabel('Duración (barras)')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/duration_comparison.png")
    plt.close()
    
    # Crear gráfico comparativo de tipos de cierre
    labels = ['Stop Loss', 'Take Profit', 'Manual']
    original_values = [
        results_original['sl_tp_stats']['sl_count'],
        results_original['sl_tp_stats']['tp_count'],
        results_original['sl_tp_stats']['manual_count']
    ]
    enhanced_values = [
        results_enhanced['sl_tp_stats']['sl_count'],
        results_enhanced['sl_tp_stats']['tp_count'],
        results_enhanced['sl_tp_stats']['manual_count']
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, original_values, width, label='Original', color='blue', alpha=0.7)
    rects2 = ax.bar(x + width/2, enhanced_values, width, label='Mejorado', color='green', alpha=0.7)
    
    ax.set_ylabel('Número de Operaciones')
    ax.set_title('Comparación de Tipos de Cierre')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Añadir etiquetas con valores
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(f"{output_dir}/close_type_comparison.png")
    plt.close()
    
    # Crear gráfico comparativo de balance
    equity_original = env_original.equity_curve
    equity_enhanced = env_enhanced.equity_curve
    
    # Asegurar misma longitud
    min_len = min(len(equity_original), len(equity_enhanced))
    equity_original = equity_original[:min_len]
    equity_enhanced = equity_enhanced[:min_len]
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_original, label='Original', color='blue')
    plt.plot(equity_enhanced, label='Mejorado', color='green')
    plt.title('Comparación de Curva de Equity')
    plt.xlabel('Pasos')
    plt.ylabel('Balance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/equity_comparison.png")
    plt.close()
    
    logger.info(f"\nResultados guardados en: {output_dir}")
    logger.info("Prueba completada.")

if __name__ == "__main__":
    main()
