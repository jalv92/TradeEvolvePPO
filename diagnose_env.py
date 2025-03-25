#!/usr/bin/env python
"""
Script de diagnóstico para analizar el entorno de trading y detectar problemas.
Este script fuerza acciones específicas y monitorea el comportamiento del entorno.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym

# Importar componentes del proyecto
from environment.trading_env import TradingEnv
from environment.simple_trading_env import SimpleTradingEnv
from data.data_loader import DataLoader

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("diagnose_env.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("diagnostico")

class EnvDiagnostic:
    """Clase para diagnosticar el comportamiento del entorno de trading."""
    
    def __init__(self, env, num_steps=500):
        """
        Inicializa el diagnóstico.
        
        Args:
            env: Entorno de trading a diagnosticar
            num_steps: Número de pasos para la ejecución del diagnóstico
        """
        self.env = env
        self.num_steps = num_steps
        self.logger = logger
        
        # Inicializar estadísticas
        self.operations_attempted = {
            "buy": {"total": 0, "success": 0, "reasons": {}},
            "sell": {"total": 0, "success": 0, "reasons": {}},
            "hold": {"total": 0, "success": 0, "reasons": {}}
        }
    
    def _log_operation_attempt(self, action_type, success, reason=""):
        """
        Registra un intento de operación.
        
        Args:
            action_type: Tipo de acción (0=hold, 1=buy, 2=sell)
            success: Si la operación fue exitosa
            reason: Razón del fallo (si aplica)
        """
        action_names = ["hold", "buy", "sell"]
        action_name = action_names[action_type]
        
        self.operations_attempted[action_name]["total"] += 1
        
        if success:
            self.operations_attempted[action_name]["success"] += 1
        elif reason:
            if reason not in self.operations_attempted[action_name]["reasons"]:
                self.operations_attempted[action_name]["reasons"][reason] = 0
            self.operations_attempted[action_name]["reasons"][reason] += 1
            
        return action_name
    
    def run_fixed_pattern(self):
        """Ejecuta un patrón fijo de acciones para diagnosticar el entorno."""
        self.logger.info("Iniciando diagnóstico con patrón fijo de acciones")
        
        actions_results = []
        self.env.reset()
        
        # Determinar tipo de espacio de acción
        is_continuous = isinstance(self.env.action_space, gym.spaces.Box)
        
        for i in tqdm(range(self.num_steps), desc="Ejecutando patrón fijo"):
            # Ciclo de acciones: comprar (10 pasos) -> mantener (10 pasos) -> vender (10 pasos)
            action_type = (i // 10) % 3  # 0: hold, 1: buy, 2: sell
            
            # Ejecutar acción basada en el tipo de acción y espacio de acción
            if is_continuous:
                if action_type == 0:  # hold
                    action = np.array([0.0])
                elif action_type == 1:  # buy
                    action = np.array([0.7])
                else:  # sell
                    action = np.array([-0.7])
            else:
                action = action_type
            
            # Registrar intento de acción
            self._log_operation_attempt(action_type, False)
            
            # Ejecutar acción en el entorno
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Determinar si la operación fue exitosa
            operation_executed = info.get("operation_executed", False)
            if operation_executed:
                # Actualizar el éxito si la operación fue ejecutada
                self._log_operation_attempt(action_type, True)
            
            # Almacenar resultados
            actions_results.append({
                "step": i,
                "action": action if not is_continuous else action[0],
                "action_type": ["hold", "buy", "sell"][action_type],
                "reward": reward,
                "position": info.get("position", 0),
                "balance": info.get("balance", 0),
                "operation_executed": operation_executed
            })
            
            if done:
                self.env.reset()
        
        # Calcular estadísticas
        success_by_type = {
            "buy": {"attempts": 0, "success": 0},
            "sell": {"attempts": 0, "success": 0},
            "hold": {"attempts": 0, "success": 0}
        }
        
        for result in actions_results:
            action_type = result["action_type"]
            success_by_type[action_type]["attempts"] += 1
            if result["operation_executed"]:
                success_by_type[action_type]["success"] += 1
        
        # Calcular tasas de éxito
        for action_type in success_by_type:
            attempts = success_by_type[action_type]["attempts"]
            if attempts > 0:
                success_rate = success_by_type[action_type]["success"] / attempts * 100
                success_by_type[action_type]["success_rate"] = success_rate
            else:
                success_by_type[action_type]["success_rate"] = 0
        
        return {
            "actions_results": actions_results,
            "success_by_type": success_by_type
        }
    
    def run_random_actions(self, seed=42):
        """
        Ejecuta acciones aleatorias para diagnosticar el entorno.
        
        Args:
            seed: Semilla para reproducibilidad
        
        Returns:
            Diccionario con resultados del diagnóstico
        """
        np.random.seed(seed)
        self.logger.info("Iniciando diagnóstico con acciones aleatorias")
        
        actions_results = []
        obs, _ = self.env.reset()
        
        # Determinar tipo de espacio de acción
        is_continuous = isinstance(self.env.action_space, gym.spaces.Box)
        
        for i in tqdm(range(self.num_steps), desc="Ejecutando acciones aleatorias"):
            # Generar acción aleatoria
            if is_continuous:
                action = self.env.action_space.sample()
            else:
                action = np.random.randint(0, self.env.action_space.n)
            
            # Determinar tipo de acción para estadísticas
            if is_continuous:
                action_value = action[0]
                if action_value > 0.3:
                    action_type = 1  # buy
                elif action_value < -0.3:
                    action_type = 2  # sell
                else:
                    action_type = 0  # hold
            else:
                action_type = action
            
            # Registrar intento de acción
            self._log_operation_attempt(action_type, False)
            
            # Ejecutar acción
            obs, reward, done, truncated, info = self.env.step(action)
            
            # Determinar si la operación fue exitosa
            operation_executed = info.get("operation_executed", False)
            if operation_executed:
                # Actualizar el éxito si la operación fue ejecutada
                self._log_operation_attempt(action_type, True)
            
            # Almacenar resultados
            actions_results.append({
                "step": i,
                "action": action if not is_continuous else action[0],
                "action_type": ["hold", "buy", "sell"][action_type],
                "reward": reward,
                "position": info.get("position", 0),
                "balance": info.get("balance", 0),
                "operation_executed": operation_executed
            })
            
            if done:
                obs, _ = self.env.reset()
        
        # Calcular estadísticas
        success_by_type = {
            "buy": {"attempts": 0, "success": 0},
            "sell": {"attempts": 0, "success": 0},
            "hold": {"attempts": 0, "success": 0}
        }
        
        for result in actions_results:
            action_type = result["action_type"]
            success_by_type[action_type]["attempts"] += 1
            if result["operation_executed"]:
                success_by_type[action_type]["success"] += 1
        
        # Calcular tasas de éxito
        for action_type in success_by_type:
            attempts = success_by_type[action_type]["attempts"]
            if attempts > 0:
                success_rate = success_by_type[action_type]["success"] / attempts * 100
                success_by_type[action_type]["success_rate"] = success_rate
            else:
                success_by_type[action_type]["success_rate"] = 0
        
        return {
            "actions_results": actions_results,
            "success_by_type": success_by_type
        }
    
    def plot_results(self, results, title="Diagnóstico del Entorno de Trading"):
        """
        Genera gráficos con los resultados del diagnóstico.
        
        Args:
            results: Resultados del diagnóstico
            title: Título del gráfico
        """
        actions_results = results["actions_results"]
        
        # Crear DataFrame para análisis
        df = pd.DataFrame(actions_results)
        
        # Gráfico de acciones tomadas y posiciones
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Balance
        plt.subplot(3, 1, 1)
        plt.plot(df["step"], df["balance"])
        plt.title("Balance")
        plt.grid(True)
        
        # Subplot 2: Posición
        plt.subplot(3, 1, 2)
        plt.plot(df["step"], df["position"])
        plt.title("Posición")
        plt.grid(True)
        
        # Subplot 3: Recompensa
        plt.subplot(3, 1, 3)
        plt.plot(df["step"], df["reward"])
        plt.title("Recompensa")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("diagnostico_resultados.png")
        plt.close()
        
        # Gráfico de tasa de éxito por tipo de acción
        success_rates = {
            action_type: results["success_by_type"][action_type]["success_rate"]
            for action_type in results["success_by_type"]
        }
        
        plt.figure(figsize=(10, 6))
        plt.bar(success_rates.keys(), success_rates.values())
        plt.title("Tasa de Éxito por Tipo de Acción")
        plt.xlabel("Tipo de Acción")
        plt.ylabel("Tasa de Éxito (%)")
        plt.grid(True, axis='y')
        plt.savefig("diagnostico_exito_acciones.png")
        plt.close()
    
    def print_summary(self, results):
        """
        Imprime un resumen de los resultados del diagnóstico.
        
        Args:
            results: Resultados del diagnóstico
        """
        print("\n========== DIAGNÓSTICO DEL ENTORNO DE TRADING ==========")
        print("\nEstadísticas de operaciones:")
        
        for action_type in results["success_by_type"]:
            data = results["success_by_type"][action_type]
            attempts = data["attempts"]
            success = data["success"]
            success_rate = data["success_rate"]
            
            print(f"  - {action_type.title()}: {success}/{attempts} operaciones exitosas ({success_rate:.2f}%)")
        
        # Analizar balance
        df = pd.DataFrame(results["actions_results"])
        initial_balance = df["balance"].iloc[0] if not df.empty else 0
        final_balance = df["balance"].iloc[-1] if not df.empty else 0
        profit = final_balance - initial_balance
        
        print(f"\nResultados financieros:")
        print(f"  - Balance inicial: {initial_balance:.2f}")
        print(f"  - Balance final: {final_balance:.2f}")
        print(f"  - Beneficio/Pérdida: {profit:.2f} ({(profit/initial_balance)*100:.2f}%)")
        
        # Contar operaciones totales exitosas
        total_operations = sum(data["success"] for data in results["success_by_type"].values())
        
        print(f"\nTotales:")
        print(f"  - Operaciones ejecutadas: {total_operations}")
        print(f"  - Pasos totales: {self.num_steps}")
        
        print("\n===== DIAGNÓSTICO COMPLETADO =====")
        print("Gráficos guardados como 'diagnostico_resultados.png' y 'diagnostico_exito_acciones.png'")
        print("Registro detallado disponible en 'diagnose_env.log'")

def main():
    # Obtener archivo de datos
    data_file = None
    for file in os.listdir("data"):
        if file.endswith(".csv") and "combined" in file:
            data_file = os.path.join("data", file)
            break
    
    if not data_file:
        print("Error: No se encontró un archivo de datos adecuado. Asegúrate de tener datos en la carpeta 'data'.")
        sys.exit(1)
    
    print(f"Usando archivo de datos: {data_file}")
    
    # Configuración básica para DataLoader
    data_config = {
        'date_column': ['datetime'],
        'index_column': None,
        'handle_missing': True,
        'missing_strategy': 'ffill',
        'normalize': False,
        'train_ratio': 0.7,
        'val_ratio': 0.15
    }
    
    # Cargar datos
    data_loader = DataLoader(data_config)
    train_data, val_data, test_data = data_loader.prepare_data(data_file)
    
    # Configuración
    config = {
        "window_size": 60,
        "initial_balance": 100000,
        "commission_pct": 0.01,
        "max_position_size": 1,
        "reward": {
            "pnl_weight": 1.0,
            "win_rate_weight": 0.3,
            "trade_frequency_weight": 0.2,
            "inactivity_penalty": 0.05,
            "inactivity_threshold": 20
        }
    }
    
    # Crear entorno
    env = TradingEnv(
        data=train_data,
        config=config,
        window_size=config["window_size"],
        mode="train"
    )
    
    # Crear diagnóstico
    diagnostic = EnvDiagnostic(env, num_steps=500)
    
    # Ejecutar diagnóstico con patrón fijo
    print("\nEjecutando diagnóstico con patrón fijo de acciones...")
    results_fixed = diagnostic.run_fixed_pattern()
    
    # Imprimir resultados y generar gráficas
    diagnostic.plot_results(results_fixed, "Diagnóstico con Patrón Fijo")
    diagnostic.print_summary(results_fixed)
    
    # Opcional: ejecutar diagnóstico con acciones aleatorias
    print("\nEjecutando diagnóstico con acciones aleatorias...")
    results_random = diagnostic.run_random_actions()
    
    # Imprimir resultados y generar gráficas
    diagnostic.plot_results(results_random, "Diagnóstico con Acciones Aleatorias")
    diagnostic.print_summary(results_random)

if __name__ == "__main__":
    main() 