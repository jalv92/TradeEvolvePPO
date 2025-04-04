# TradeEvolvePPO

Sistema de trading basado en RL usando PPO para entrenar agentes capaces de operar en mercados financieros.

## Estructura del Proyecto

```
TradeEvolvePPO/
├── agents/               # Implementaciones de agentes (PPO, LSTM)
├── config/               # Archivos de configuración
├── connector_nt8/        # Conexión con NinjaTrader 8
├── data/                 # Manejo de datos
│   └── dataset/          # Conjuntos de datos para entrenamiento
├── debug/                # Scripts de depuración y diagnóstico
├── docs/                 # Documentación (incluye nt8.pdf)
├── environment/          # Entornos de trading para RL
├── evaluation/           # Evaluación y backtesting
├── indicator_NT8/        # Indicadores para NinjaTrader 8
├── logs/                 # Logs de ejecución
├── plots/                # Gráficos generados
├── results/              # Resultados de entrenamiento
├── training/             # Módulos de entrenamiento
├── utils/                # Utilidades generales
└── visualization/        # Visualización de resultados
```

## Archivos Principales

- `main.py`: Punto de entrada principal para entrenamiento/backtesting
- `cleanup.py`: Herramienta para mantenimiento y limpieza del proyecto
- `training/train.py`: Implementación principal del entrenamiento
- `connector_nt8/nt8_trader.py`: Trader en vivo para NinjaTrader 8
- `environment/trading_env.py`: Entorno de trading para RL

## Modos de Ejecución

El sistema puede ejecutarse en diferentes modos:

1. **Entrenamiento**: Entrenar un modelo desde cero
   ```
   python main.py --mode train --data path/to/data.csv --output results/new_run
   ```

2. **Prueba**: Evaluar un modelo entrenado en datos de prueba
   ```
   python main.py --mode test --data path/to/test_data.csv --model path/to/model.zip --output results/test_run
   ```

3. **Backtest**: Realizar backtest con un modelo entrenado
   ```
   python main.py --mode backtest --data path/to/data.csv --model path/to/model.zip --output results/backtest_run
   ```

## Mantenimiento

Para mantener el proyecto organizado:

```
python cleanup.py --keep 5 --zip
```

Esto mantiene las 5 ejecuciones más recientes y comprime las más antiguas.

## Requisitos

Ver `requirements.txt` para la lista completa de dependencias.
