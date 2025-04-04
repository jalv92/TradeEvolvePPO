"""
Utilidades para diagnosticar y analizar el comportamiento de trading.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)

class TradeAnalyzer:
    """
    Clase para analizar operaciones de trading y generar estadísticas.
    """
    
    def __init__(self, trades: List[Dict[str, Any]], output_dir: Optional[str] = None):
        """
        Inicializa el analizador de operaciones.
        
        Args:
            trades: Lista de operaciones a analizar
            output_dir: Directorio para guardar resultados
        """
        self.trades = trades
        self.output_dir = output_dir
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    def analyze_durations(self) -> Dict[str, Any]:
        """
        Analiza la duración de las operaciones.
        
        Returns:
            Dict: Estadísticas de duración
        """
        # Extraer duraciones
        durations = [t.get('duration', 0) for t in self.trades if 'duration' in t]
        
        if not durations:
            return {
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0,
                'std': 0,
                'pct_below_5': 0,
                'pct_below_10': 0
            }
        
        # Calcular estadísticas
        mean_duration = np.mean(durations)
        median_duration = np.median(durations)
        min_duration = np.min(durations)
        max_duration = np.max(durations)
        std_duration = np.std(durations)
        
        # Calcular porcentajes por debajo de umbrales
        pct_below_5 = 100 * sum(1 for d in durations if d < 5) / len(durations)
        pct_below_10 = 100 * sum(1 for d in durations if d < 10) / len(durations)
        
        # Crear gráfico de distribución
        if self.output_dir:
            plt.figure(figsize=(10, 6))
            plt.hist(durations, bins=20, alpha=0.7, color='blue')
            plt.axvline(x=5, color='red', linestyle='--', label='5 barras')
            plt.axvline(x=10, color='orange', linestyle='--', label='10 barras')
            plt.axvline(x=mean_duration, color='green', linestyle='-', label=f'Media: {mean_duration:.1f}')
            plt.title('Distribución de Duración de Operaciones')
            plt.xlabel('Duración (barras)')
            plt.ylabel('Frecuencia')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'plots', 'trade_durations.png'))
            plt.close()
        
        return {
            'mean': mean_duration,
            'median': median_duration,
            'min': min_duration,
            'max': max_duration,
            'std': std_duration,
            'pct_below_5': pct_below_5,
            'pct_below_10': pct_below_10
        }
    
    def analyze_entry_exit(self) -> Dict[str, Any]:
        """
        Analiza las entradas y salidas de las operaciones.
        
        Returns:
            Dict: Estadísticas de entradas y salidas
        """
        # Filtrar operaciones completas
        completed_trades = [t for t in self.trades if 'entry_price' in t and 'exit_price' in t]
        
        if not completed_trades:
            return {
                'total_trades': 0,
                'positive_pnl': 0,
                'negative_pnl': 0,
                'positive_pnl_pct': 0,
                'avg_pnl': 0,
                'avg_positive_pnl': 0,
                'avg_negative_pnl': 0
            }
        
        # Calcular estadísticas
        total_trades = len(completed_trades)
        positive_pnl = sum(1 for t in completed_trades if t.get('net_pnl', 0) > 0)
        negative_pnl = sum(1 for t in completed_trades if t.get('net_pnl', 0) <= 0)
        
        positive_pnl_pct = 100 * positive_pnl / total_trades if total_trades > 0 else 0
        
        # Calcular PnL promedio
        pnls = [t.get('net_pnl', 0) for t in completed_trades]
        positive_pnls = [p for p in pnls if p > 0]
        negative_pnls = [p for p in pnls if p <= 0]
        
        avg_pnl = np.mean(pnls) if pnls else 0
        avg_positive_pnl = np.mean(positive_pnls) if positive_pnls else 0
        avg_negative_pnl = np.mean(negative_pnls) if negative_pnls else 0
        
        # Crear gráfico de PnL
        if self.output_dir:
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(pnls)), pnls, color=['green' if p > 0 else 'red' for p in pnls])
            plt.axhline(y=0, color='black', linestyle='-')
            plt.title('PnL por Operación')
            plt.xlabel('Operación')
            plt.ylabel('PnL')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'plots', 'trade_pnl.png'))
            plt.close()
        
        return {
            'total_trades': total_trades,
            'positive_pnl': positive_pnl,
            'negative_pnl': negative_pnl,
            'positive_pnl_pct': positive_pnl_pct,
            'avg_pnl': avg_pnl,
            'avg_positive_pnl': avg_positive_pnl,
            'avg_negative_pnl': avg_negative_pnl
        }
    
    def analyze_sl_tp(self) -> Dict[str, Any]:
        """
        Analiza los cierres por stop loss y take profit.
        
        Returns:
            Dict: Estadísticas de SL/TP
        """
        # Contar cierres por SL/TP
        sl_count = 0
        tp_count = 0
        manual_count = 0
        
        for trade in self.trades:
            if 'close_type' in trade:
                if 'STOP LOSS' in trade['close_type']:
                    sl_count += 1
                elif 'TAKE PROFIT' in trade['close_type']:
                    tp_count += 1
                else:
                    manual_count += 1
        
        total_count = sl_count + tp_count + manual_count
        
        # Calcular ratios
        sl_pct = 100 * sl_count / total_count if total_count > 0 else 0
        tp_pct = 100 * tp_count / total_count if total_count > 0 else 0
        manual_pct = 100 * manual_count / total_count if total_count > 0 else 0
        
        tp_sl_ratio = tp_count / sl_count if sl_count > 0 else float('inf') if tp_count > 0 else 0
        
        # Crear gráfico de distribución
        if self.output_dir and total_count > 0:
            labels = ['Stop Loss', 'Take Profit', 'Manual']
            sizes = [sl_count, tp_count, manual_count]
            colors = ['red', 'green', 'blue']
            
            # Solo crear gráficos si hay datos
            if sum(sizes) > 0:
                plt.figure(figsize=(10, 6))
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                plt.axis('equal')
                plt.title('Distribución de Tipos de Cierre')
                plt.savefig(os.path.join(self.output_dir, 'plots', 'sl_tp_distribution.png'))
                plt.close()
                
                # Crear gráfico de barras
                plt.figure(figsize=(10, 6))
                plt.bar(labels, sizes, color=colors)
                plt.title('Número de Operaciones por Tipo de Cierre')
                plt.ylabel('Número de Operaciones')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(self.output_dir, 'plots', 'sl_tp_counts.png'))
                plt.close()
        
        return {
            'sl_count': sl_count,
            'tp_count': tp_count,
            'manual_count': manual_count,
            'sl_pct': sl_pct,
            'tp_pct': tp_pct,
            'manual_pct': manual_pct,
            'tp_sl_ratio': tp_sl_ratio
        }
    
    def analyze_ticks(self) -> Dict[str, Any]:
        """
        Analiza los ticks positivos y negativos.
        
        Returns:
            Dict: Estadísticas de ticks
        """
        # Extraer datos de ticks
        positive_ticks = [t.get('positive_ticks', 0) for t in self.trades if 'positive_ticks' in t]
        negative_ticks = [t.get('negative_ticks', 0) for t in self.trades if 'negative_ticks' in t]
        
        if not positive_ticks or not negative_ticks:
            return {
                'positive_mean': 0,
                'negative_mean': 0,
                'ratio_mean': 0,
                'zero_positive_pct': 0,
                'zero_negative_pct': 0
            }
        
        # Calcular estadísticas
        positive_mean = np.mean(positive_ticks)
        negative_mean = np.mean(negative_ticks)
        
        # Calcular ratios (evitando división por cero)
        ratios = []
        for p, n in zip(positive_ticks, negative_ticks):
            if n > 0:
                ratios.append(p / n)
            elif p > 0:
                ratios.append(float('inf'))
            else:
                ratios.append(0)
        
        ratio_mean = np.mean([r for r in ratios if r != float('inf')])
        
        # Calcular porcentajes de ceros
        zero_positive_pct = 100 * sum(1 for p in positive_ticks if p == 0) / len(positive_ticks)
        zero_negative_pct = 100 * sum(1 for n in negative_ticks if n == 0) / len(negative_ticks)
        
        # Crear gráfico de comparación
        if self.output_dir:
            plt.figure(figsize=(10, 6))
            
            # Crear índices para las barras
            indices = np.arange(len(positive_ticks))
            width = 0.35
            
            plt.bar(indices - width/2, positive_ticks, width, label='Ticks Positivos', color='green', alpha=0.7)
            plt.bar(indices + width/2, negative_ticks, width, label='Ticks Negativos', color='red', alpha=0.7)
            
            plt.title('Comparación de Ticks Positivos y Negativos por Operación')
            plt.xlabel('Operación')
            plt.ylabel('Número de Ticks')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'plots', 'ticks_comparison.png'))
            plt.close()
            
            # Crear gráfico de ratio
            plt.figure(figsize=(10, 6))
            plt.bar(indices, [min(r, 10) for r in ratios], color='blue', alpha=0.7)  # Limitar a 10 para visualización
            plt.axhline(y=1, color='red', linestyle='--', label='Ratio = 1')
            plt.title('Ratio de Ticks Positivos/Negativos por Operación')
            plt.xlabel('Operación')
            plt.ylabel('Ratio (limitado a 10)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'plots', 'ticks_ratio.png'))
            plt.close()
        
        return {
            'positive_mean': positive_mean,
            'negative_mean': negative_mean,
            'ratio_mean': ratio_mean,
            'zero_positive_pct': zero_positive_pct,
            'zero_negative_pct': zero_negative_pct
        }
    
    def run_analysis(self) -> Dict[str, Any]:
        """
        Ejecuta todos los análisis y devuelve los resultados.
        
        Returns:
            Dict: Resultados de todos los análisis
        """
        duration_stats = self.analyze_durations()
        entry_exit_stats = self.analyze_entry_exit()
        sl_tp_stats = self.analyze_sl_tp()
        ticks_stats = self.analyze_ticks()
        
        results = {
            'duration_stats': duration_stats,
            'entry_exit_stats': entry_exit_stats,
            'sl_tp_stats': sl_tp_stats,
            'ticks_stats': ticks_stats
        }
        
        # Guardar resultados en JSON
        if self.output_dir:
            # Convertir valores numpy a Python nativos para serialización JSON
            def convert_numpy(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(i) for i in obj]
                else:
                    return obj
            
            # Convertir resultados para serialización JSON
            json_results = convert_numpy(results)
            
            with open(os.path.join(self.output_dir, 'analysis_results.json'), 'w') as f:
                json.dump(json_results, f, indent=4)
        
        return results

def analyze_trades_from_env(env, output_dir=None):
    """
    Analiza las operaciones de un entorno de trading.
    
    Args:
        env: Entorno de trading
        output_dir: Directorio para guardar resultados
        
    Returns:
        Tuple: (analizador, resultados)
    """
    # Crear directorio si no existe
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Crear analizador
    analyzer = TradeAnalyzer(env.trades, output_dir=output_dir)
    
    # Ejecutar análisis
    results = analyzer.run_analysis()
    
    return analyzer, results

def diagnose_sl_tp_behavior(env, num_steps=1000, output_dir=None):
    """
    Diagnostica el comportamiento de stop loss y take profit.
    
    Args:
        env: Entorno de trading
        num_steps: Número de pasos a ejecutar
        output_dir: Directorio para guardar resultados
        
    Returns:
        Dict: Resultados del diagnóstico
    """
    # Crear directorio si no existe
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # Reiniciar entorno
    env.reset()
    
    # Variables para tracking
    premature_closures = 0
    total_closures = 0
    price_movements = []
    durations = []
    
    # Ejecutar pasos
    for i in range(num_steps):
        # Tomar acción aleatoria
        if i % 10 == 0:  # Cada 10 pasos, forzar una acción más extrema
            action = np.random.uniform(-0.8, 0.8)
        else:
            action = np.random.uniform(-0.3, 0.3)
            
        # Ejecutar paso
        _, _, done, _, info = env.step(action)
        
        # Verificar si se cerró una posición
        if hasattr(env, '_just_closed_position') and env._just_closed_position:
            total_closures += 1
            
            # Verificar si fue un cierre prematuro
            if hasattr(env, 'position_duration') and env.position_duration < env.min_hold_steps:
                premature_closures += 1
            
            # Registrar movimiento de precio y duración
            if hasattr(env, 'trades') and env.trades:
                last_trade = env.trades[-1]
                if 'entry_price' in last_trade and 'exit_price' in last_trade:
                    price_movement = abs(last_trade['exit_price'] - last_trade['entry_price']) / last_trade['entry_price']
                    price_movements.append(price_movement)
                    
                if 'duration' in last_trade:
                    durations.append(last_trade['duration'])
        
        if done:
            break
    
    # Calcular estadísticas
    premature_closure_pct = 100 * premature_closures / total_closures if total_closures > 0 else 0
    
    # Crear gráfico de movimiento de precio vs duración
    if output_dir and price_movements and durations:
        plt.figure(figsize=(10, 6))
        plt.scatter(durations, price_movements, alpha=0.7)
        plt.axvline(x=env.min_hold_steps, color='red', linestyle='--', label=f'Min Hold: {env.min_hold_steps}')
        plt.title('Movimiento de Precio vs Duración')
        plt.xlabel('Duración (barras)')
        plt.ylabel('Movimiento de Precio (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'plots', 'price_movement_vs_duration.png'))
        plt.close()
    
    return {
        'premature_closures': premature_closures,
        'total_closures': total_closures,
        'premature_closure_pct': premature_closure_pct
    }
