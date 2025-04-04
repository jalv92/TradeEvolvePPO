"""
Script para diagnosticar el comportamiento del entorno de trading y verificar
que las mejoras implementadas resuelven el problema de hipertrading.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from datetime import datetime
import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple

from environment.trading_env import TradingEnv
from environment.enhanced_trading_env import EnhancedTradingEnv
from utils.trade_diagnostics import TradeAnalyzer, analyze_trades_from_env, diagnose_sl_tp_behavior
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

def run_environment_test(env_class, config, data, n_steps=1000, random_seed=42):
    """
    Ejecuta una prueba del entorno con acciones aleatorias.
    
    Args:
        env_class: Clase del entorno a probar
        config: Configuración del entorno
        data: DataFrame con datos OHLCV
        n_steps: Número de pasos a ejecutar
        random_seed: Semilla aleatoria
        
    Returns:
        Tuple: (env, rewards, infos)
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
    
    # Variables para tracking
    rewards = []
    infos = []
    
    # Ejecutar pasos
    for i in range(n_steps):
        # Tomar acción aleatoria
        if i % 10 == 0:  # Cada 10 pasos, forzar una acción más extrema
            action = np.random.uniform(-0.8, 0.8)
        else:
            action = np.random.uniform(-0.3, 0.3)
            
        # Ejecutar paso
        obs, reward, done, truncated, info = env.step(action)
        
        # Guardar resultados
        rewards.append(reward)
        infos.append(info)
        
        if done:
            break
    
    return env, rewards, infos

def compare_environments(n_steps=2000):
    """
    Compara el comportamiento de los entornos original y mejorado.
    
    Args:
        n_steps: Número de pasos a ejecutar
        
    Returns:
        Dict: Resultados de la comparación
    """
    # Generar datos sintéticos
    data = generate_synthetic_data(n_samples=n_steps + 100)
    
    # Configuración base
    config = BASE_CONFIG.copy()
    config.update(ENV_CONFIG)
    
    # Modificar configuración para pruebas
    test_config = config.copy()
    test_config.update({
        'min_hold_steps': 10,  # Reducido para pruebas
        'force_min_hold': True,  # Forzar período mínimo
        'sl_buffer_ticks': 5,  # Buffer para SL
        'tp_buffer_ticks': 5,  # Buffer para TP
        'force_action_prob': 0.3,  # Probabilidad de forzar acción
        'log_ticks': True,  # Activar logging de ticks
    })
    
    # Ejecutar pruebas
    logger.info("Ejecutando prueba con TradingEnv original...")
    env_original, rewards_original, infos_original = run_environment_test(
        TradingEnv, test_config, data, n_steps=n_steps
    )
    
    logger.info("Ejecutando prueba con EnhancedTradingEnv mejorado...")
    env_enhanced, rewards_enhanced, infos_enhanced = run_environment_test(
        EnhancedTradingEnv, test_config, data, n_steps=n_steps
    )
    
    # Analizar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/env_comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Analizar trades
    analyzer_original, results_original = analyze_trades_from_env(
        env_original, output_dir=f"{output_dir}/original"
    )
    
    analyzer_enhanced, results_enhanced = analyze_trades_from_env(
        env_enhanced, output_dir=f"{output_dir}/enhanced"
    )
    
    # Comparar métricas clave
    comparison = {
        'original': {
            'total_trades': len(env_original.trades),
            'avg_duration': results_original['duration_stats'].get('mean', 0),
            'pct_below_5': results_original['duration_stats'].get('pct_below_5', 0),
            'win_rate': results_original['entry_exit_stats'].get('positive_pnl_pct', 0),
            'tp_sl_ratio': results_original['sl_tp_stats'].get('tp_sl_ratio', 0),
            'final_balance': env_original.balance
        },
        'enhanced': {
            'total_trades': len(env_enhanced.trades),
            'avg_duration': results_enhanced['duration_stats'].get('mean', 0),
            'pct_below_5': results_enhanced['duration_stats'].get('pct_below_5', 0),
            'win_rate': results_enhanced['entry_exit_stats'].get('positive_pnl_pct', 0),
            'tp_sl_ratio': results_enhanced['sl_tp_stats'].get('tp_sl_ratio', 0),
            'final_balance': env_enhanced.balance
        }
    }
    
    # Guardar comparación
    with open(f"{output_dir}/comparison.json", 'w') as f:
        json.dump(comparison, f, indent=4)
    
    # Crear gráficos comparativos
    create_comparison_plots(
        env_original, env_enhanced, rewards_original, rewards_enhanced, output_dir
    )
    
    logger.info(f"Comparación completada. Resultados guardados en {output_dir}")
    
    return comparison

def create_comparison_plots(env_original, env_enhanced, rewards_original, rewards_enhanced, output_dir):
    """
    Crea gráficos comparativos entre los dos entornos.
    
    Args:
        env_original: Entorno original
        env_enhanced: Entorno mejorado
        rewards_original: Recompensas del entorno original
        rewards_enhanced: Recompensas del entorno mejorado
        output_dir: Directorio de salida
    """
    # 1. Comparar duración de operaciones
    durations_original = [t.get('duration', 0) for t in env_original.trades if 'duration' in t]
    durations_enhanced = [t.get('duration', 0) for t in env_enhanced.trades if 'duration' in t]
    
    plt.figure(figsize=(12, 6))
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
    
    # 2. Comparar balance final
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
    
    # 3. Comparar recompensas
    min_len = min(len(rewards_original), len(rewards_enhanced))
    rewards_original = rewards_original[:min_len]
    rewards_enhanced = rewards_enhanced[:min_len]
    
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_original, label='Original', color='blue', alpha=0.5)
    plt.plot(rewards_enhanced, label='Mejorado', color='green', alpha=0.5)
    plt.title('Comparación de Recompensas')
    plt.xlabel('Pasos')
    plt.ylabel('Recompensa')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/rewards_comparison.png")
    plt.close()
    
    # 4. Comparar distribución de tipos de cierre (SL/TP/Manual)
    sl_tp_original = {'sl': 0, 'tp': 0, 'manual': 0}
    sl_tp_enhanced = {'sl': 0, 'tp': 0, 'manual': 0}
    
    for trade in env_original.trades:
        if 'close_type' in trade:
            if 'STOP LOSS' in trade['close_type']:
                sl_tp_original['sl'] += 1
            elif 'TAKE PROFIT' in trade['close_type']:
                sl_tp_original['tp'] += 1
            else:
                sl_tp_original['manual'] += 1
                
    for trade in env_enhanced.trades:
        if 'close_type' in trade:
            if 'STOP LOSS' in trade['close_type']:
                sl_tp_enhanced['sl'] += 1
            elif 'TAKE PROFIT' in trade['close_type']:
                sl_tp_enhanced['tp'] += 1
            else:
                sl_tp_enhanced['manual'] += 1
    
    # Crear gráfico de barras
    labels = ['Stop Loss', 'Take Profit', 'Manual']
    original_values = [sl_tp_original['sl'], sl_tp_original['tp'], sl_tp_original['manual']]
    enhanced_values = [sl_tp_enhanced['sl'], sl_tp_enhanced['tp'], sl_tp_enhanced['manual']]
    
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

def diagnose_environment():
    """
    Ejecuta diagnóstico completo del entorno mejorado.
    """
    # Generar datos sintéticos
    data = generate_synthetic_data(n_samples=5000)
    
    # Configuración para diagnóstico
    config = BASE_CONFIG.copy()
    config.update(ENV_CONFIG)
    config.update({
        'min_hold_steps': 10,  # Reducido para pruebas
        'force_min_hold': True,  # Forzar período mínimo
        'sl_buffer_ticks': 5,  # Buffer para SL
        'tp_buffer_ticks': 5,  # Buffer para TP
        'log_ticks': True,  # Activar logging de ticks
    })
    
    # Crear entorno mejorado
    env = EnhancedTradingEnv(
        data=data,
        config=config,
        initial_balance=100000.0,
        window_size=60,
        mode='eval'  # Usar modo eval para forzar más operaciones
    )
    
    # Ejecutar diagnóstico de SL/TP
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/env_diagnosis_{timestamp}"
    
    logger.info("Ejecutando diagnóstico de SL/TP...")
    sl_tp_results = diagnose_sl_tp_behavior(env, num_steps=2000, output_dir=f"{output_dir}/sl_tp")
    
    # Reiniciar entorno y ejecutar prueba completa
    env.reset()
    
    logger.info("Ejecutando prueba completa...")
    _, rewards, infos = run_environment_test(
        EnhancedTradingEnv, config, data, n_steps=3000
    )
    
    # Analizar trades
    analyzer, results = analyze_trades_from_env(
        env, output_dir=f"{output_dir}/trades"
    )
    
    # Crear informe HTML
    create_html_report(env, results, sl_tp_results, output_dir)
    
    logger.info(f"Diagnóstico completado. Resultados guardados en {output_dir}")
    
    return env, results, sl_tp_results

def create_html_report(env, trade_results, sl_tp_results, output_dir):
    """
    Crea un informe HTML con los resultados del diagnóstico.
    
    Args:
        env: Entorno de trading
        trade_results: Resultados del análisis de trades
        sl_tp_results: Resultados del diagnóstico de SL/TP
        output_dir: Directorio de salida
    """
    # Extraer métricas clave
    total_trades = len(env.trades)
    avg_duration = trade_results['duration_stats'].get('mean', 0)
    pct_below_5 = trade_results['duration_stats'].get('pct_below_5', 0)
    win_rate = trade_results['entry_exit_stats'].get('positive_pnl_pct', 0)
    tp_sl_ratio = trade_results['sl_tp_stats'].get('tp_sl_ratio', 0)
    final_balance = env.balance
    
    # Crear contenido HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Diagnóstico del Entorno de Trading</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .section {{ margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .stat-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 15px; }}
            .stat-box {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
            .stat-label {{ font-size: 14px; color: #7f8c8d; }}
            .good {{ color: #27ae60; }}
            .bad {{ color: #e74c3c; }}
            .neutral {{ color: #f39c12; }}
            .image-container {{ margin-top: 20px; text-align: center; }}
            .image-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Diagnóstico del Entorno de Trading Mejorado</h1>
        <p>Generado el: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="section">
            <h2>Resumen de Resultados</h2>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-value">{total_trades}</div>
                    <div class="stat-label">Total de Operaciones</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value {('good' if avg_duration >= 10 else 'bad')}">{avg_duration:.1f}</div>
                    <div class="stat-label">Duración Media (barras)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value {('bad' if pct_below_5 > 50 else 'neutral')}">{pct_below_5:.1f}%</div>
                    <div class="stat-label">% Operaciones < 5 barras</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value {('good' if win_rate > 50 else 'bad')}">{win_rate:.1f}%</div>
                    <div class="stat-label">Win Rate</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value {('good' if tp_sl_ratio > 1 else 'bad')}">{tp_sl_ratio:.2f}</div>
                    <div class="stat-label">Ratio TP/SL</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value {('good' if final_balance > 100000 else 'bad')}">
                        ${final_balance:.2f}
                    </div>
                    <div class="stat-label">Balance Final</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Análisis de Duración de Operaciones</h2>
            <p>La duración media de las operaciones es de <strong>{avg_duration:.1f}</strong> barras.</p>
            <p>El {pct_below_5:.1f}% de las operaciones duran menos de 5 barras.</p>
            <div class="image-container">
                <img src="trades/plots/trade_durations.png" alt="Distribución de Duración de Operaciones">
            </div>
        </div>
        
        <div class="section">
            <h2>Análisis de Stop Loss y Take Profit</h2>
            <p>El ratio de operaciones cerradas por Take Profit vs Stop Loss es <strong>{tp_sl_ratio:.2f}</strong>.</p>
            <p>Premature closures: {sl_tp_results.get('premature_closures', 'N/A')} 
               ({sl_tp_results.get('premature_closure_pct', 0):.1f}% of total)</p>
            <div class="image-container">
                <img src="trades/plots/sl_tp_distribution.png" alt="Distribución de Tipos de Cierre">
            </div>
            <div class="image-container">
                <img src="sl_tp/plots/price_movement_vs_duration.png" alt="Movimiento de Precio vs Duración">
            </div>
        </div>
        
        <div class="section">
            <h2>Análisis de Ticks</h2>
            <p>Ratio medio de ticks positivos/negativos: <strong>{trade_results['ticks_stats'].get('ratio_mean', 0):.2f}</strong></p>
            <div class="image-container">
                <img src="trades/plots/ticks_comparison.png" alt="Comparación de Ticks">
            </div>
        </div>
        
        <div class="section">
            <h2>Conclusiones</h2>
            <p>Basado en el análisis de las {total_trades} operaciones, se observa que:</p>
            <ul>
                <li><strong>Duración de operaciones:</strong> {
                    "Las operaciones son extremadamente cortas, lo que indica que el problema de hipertrading persiste." 
                    if pct_below_5 > 50 
                    else "La duración de las operaciones ha mejorado significativamente, lo que indica que las mejoras implementadas están funcionando."
                }</li>
                <li><strong>Efectividad de SL/TP:</strong> {
                    "Hay un desequilibrio significativo hacia cierres por Stop Loss, lo que indica problemas en la configuración de SL/TP." 
                    if tp_sl_ratio < 0.5 
                    else "La distribución de cierres por SL/TP es más equilibrada, lo que indica una mejor gestión del riesgo."
                }</li>
                <li><strong>Análisis de ticks:</strong> {
                    "Hay un problema con el registro de ticks positivos, lo que sugiere que las operaciones se cierran antes de poder registrar movimientos favorables." 
                    if trade_results['ticks_stats'].get('zero_positive_pct', 0) > 50 
                    else "El registro de ticks funciona correctamente, lo que permite una mejor evaluación del rendimiento de las operaciones."
                }</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Guardar informe HTML
    with open(f"{output_dir}/report.html", 'w') as f:
        f.write(html_content)
        
    logger.info(f"Informe HTML generado: {output_dir}/report.html")

if __name__ == "__main__":
    # Ejecutar comparación de entornos
    logger.info("Iniciando comparación de entornos...")
    comparison_results = compare_environments(n_steps=2000)
    
    # Ejecutar diagnóstico completo
    logger.info("Iniciando diagnóstico completo...")
    env, results, sl_tp_results = diagnose_environment()
    
    logger.info("Diagnóstico completado.")
