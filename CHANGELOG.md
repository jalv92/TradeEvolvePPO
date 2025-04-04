# 30/03/2025 20:06:10
- Implementado entorno de trading mejorado con características anti-hipertrading:
  - Creado nuevo archivo `environment/enhanced_trading_env.py` que extiende el entorno base
  - Configuración mejorada con valores predeterminados optimizados:
    - Aumentado `min_hold_steps` a 30 barras para evitar operaciones demasiado cortas
    - Incrementado `position_cooldown` a 40 pasos para reducir frecuencia de operaciones
    - Activado `force_min_hold` para garantizar duración mínima obligatoria
    - Aumentados `min_sl_ticks` y `min_tp_ticks` a 50 ticks para mejor gestión de riesgo
  - Implementado sistema de buffer para SL/TP:
    - Añadidos parámetros `sl_buffer_ticks` y `tp_buffer_ticks` (5 ticks cada uno)
    - Cálculo de niveles efectivos de SL/TP con margen adicional
    - Mantenimiento de precios originales para cálculo de PnL
  - Mejorado sistema de cooldown después de cerrar posiciones
  - Implementado tracking detallado de ticks positivos y negativos
  - Añadido diagnóstico de cierres prematuros para análisis posterior
  - Mejoradas métricas de rendimiento con estadísticas adicionales
  - Corregido error en `utils/trade_diagnostics.py` para manejar correctamente valores numpy en serialización JSON

# 30/03/2025 17:30:00
- Implementada solución completa para el problema de hipertrading:
  - Modificado método `_check_sl_tp` en `environment/trading_env.py`:
    - Añadido período mínimo obligatorio para mantener posiciones (configurable)
    - Implementado buffer para niveles de SL/TP para evitar cierres prematuros
    - Añadido diagnóstico detallado para cada cierre por SL/TP
  - Mejorado método `_update_ticks` en `environment/enhanced_trading_env.py`:
    - Registro de ticks basado en movimiento real del precio, no solo en PnL
    - Garantizado registro mínimo de ticks incluso en operaciones muy cortas
    - Añadido diagnóstico detallado para cada tick registrado
  - Creadas herramientas de diagnóstico en `utils/trade_diagnostics.py`:
    - Análisis completo de duración de operaciones
    - Evaluación de efectividad de SL/TP
    - Análisis de ticks positivos/negativos
    - Generación de informes HTML con visualizaciones
  - Implementado script de diagnóstico `diagnose_env.py`:
    - Comparación entre entorno original y mejorado
    - Diagnóstico detallado del comportamiento de SL/TP
    - Generación de datos sintéticos para pruebas controladas
    - Visualizaciones comparativas de resultados

# 30/03/2025 15:40:00
- Corregidos errores en el script de entrenamiento LSTM para GPU:
  - Corregido manejo de parámetros `features_extractor` y `activation_fn` en la configuración
  - Implementada solución para convertir strings de activación (ej: 'tanh') a clases reales (nn.Tanh)
  - Mejorado el procesamiento de observaciones 1D en LSTM para manejar dinámicamente diferentes tamaños de entrada
  - Implementado ajuste automático de la arquitectura LSTM para adaptarse a cualquier formato de observación
  - Corregido problema de dispositivos moviendo el LSTM dinámicamente a la GPU
  - Añadido código para calcular automáticamente las dimensiones de entrada basándose en las observaciones reales
  - Mejorada la gestión de errores durante la inicialización del modelo

# 30/03/2025 18:35:00
- Corregidos errores en el script de entrenamiento:
  - Solucionado error "not enough values to unpack (expected 5, got 4)" en la función `force_eval_random_policy`
  - Corregido error "TrainingMetricsCallback object has no attribute 'total_timesteps'"
  - El modelo ahora ejecuta correctamente tanto operaciones LONG como SHORT
  - Aplicación adecuada de stop loss y take profit durante el entrenamiento
  - Cálculo correcto de PnL, comisiones y balance

# 30/03/2025 14:41:34
- Completado entrenamiento MLP en CPU:
  - Entrenamiento de 50000 pasos en 8.6 minutos
  - Política: MLP ([256, 128, 64])
  - **Evolución del entrenamiento**:
    - Win Rate: 0.0% -> 0.0% (+0.0%)
    - Profit Factor: 0.00 -> 0.00 (+0.00)
    - Operaciones por evaluación: 0 -> 0
    - Generados 10 puntos de evaluación
    - Gráficos de evolución guardados en: results/MlpTradeEvolvePPO_20250330_143258/plots/
  - Modelo guardado en: results/MlpTradeEvolvePPO_20250330_143258

# 30/03/2025 13:48:54
- Completado entrenamiento MLP en CPU:
  - Entrenamiento de 50000 pasos en 5.7 minutos
  - Política: MLP ([256, 128, 64])
  - Total operaciones: 99
  - Tasa de éxito: 49.5%
  - Profit Factor: 0.80
  - Modelo guardado en: results/MlpTradeEvolvePPO_20250330_134308

# [30/03/2025 13:45:00] - v1.6.1

### Mejorado
- **Eliminada confirmación de usuario en train_mlp_cpu.py**:
  - Removida pregunta "¿Desea continuar de todos modos? (s/n)" al detectar GPU disponible
  - El script ahora continúa automáticamente con el entrenamiento tras mostrar la advertencia
  - Cambio orientado a facilitar ejecuciones automatizadas y entrenamiento por lotes
  - Mantiene el mensaje informativo recomendando usar train_lstm_gpu.py para GPU

# [30/03/2025 05:45:00] - v1.6.0

### Mejorado
- **Implementado sistema avanzado de gestión dinámica de stop loss y take profit**:
  - Modificada configuración en `config/config.py`:
    - Añadidos parámetros para trailing stop dinámico: `enable_dynamic_trailing`, `trailing_stop_distance_ticks`, `trailing_activation_threshold`
    - Añadidos umbrales para activación de trailing stop y break-even basados en % de ganancias
    - Añadido parámetro `reward_for_good_sl_adjustment` para recompensar ajustes acertados
  - Ampliado espacio de acción en `trading_env.py`:
    - Segunda dimensión de la acción (0.0-1.0) para gestión de SL/TP
    - 0.0-0.33: mantener SL/TP actual
    - 0.33-0.66: mover SL a break-even (cuando hay ganancias suficientes)
    - 0.66-1.0: activar trailing stop (cuando hay ganancias suficientes)
  - Mejorado método `_update_trailing_stop()` en `trading_env.py`:
    - Ahora utiliza distancia en ticks en lugar de porcentajes para mayor precisión
    - Actualización automática del registro de operaciones cuando el stop se mueve
    - Logging detallado de ajustes para análisis posterior
  - Expandido método `_get_observation()` con información de gestión de riesgo:
    - Añadidas 7 nuevas características: dirección de posición, precio relativo, SL relativo, TP relativo
    - Añadidas distancias normalizadas a SL/TP y estado del trailing stop
    - Datos normalizados para facilitar el aprendizaje del agente
  - Mejorado método `step()` para procesar acciones de gestión de SL/TP:
    - Verificación de umbrales mínimos de ganancia antes de permitir modificaciones
    - Actualización del estado de la operación en el historial de trades
    - Información ampliada en el diccionario `info` devuelto por `step()`
  - Modificado método `_calculate_simple_reward_with_ticks()` en `enhanced_trading_env.py`:
    - Nuevas recompensas por uso efectivo de trailing stops y movimiento a break-even
    - Bonificación especial por operaciones cerradas con beneficios usando trailing stop
    - Balance entre incentivar protección de capital y maximización de ganancias
  - Sistema preparado para que el agente aprenda a gestionar dinámicamente el riesgo
  - Resultados esperados: mayor captura de beneficios, menores drawdowns y mejor manejo de volatilidad

# [30/03/2025 04:45:30] - v1.5.9

### Mejorado
- **Implementado sistema obligatorio de stop loss y take profit basado en ticks**:
  - Modificada la configuración en `config/config.py`:
    - Eliminados parámetros `stop_loss_pct` y `take_profit_pct`
    - Configurado `min_sl_ticks` y `min_tp_ticks` a 50 ticks cada uno
    - Activado `enforce_min_trade_size = True` para forzar el uso de los ticks mínimos
    - Habilitado `reward_larger_trades = True` para incentivar operaciones de mayor tamaño
  - Rediseñado método `_open_position()` en `trading_env.py`:
    - Eliminado completamente el cálculo basado en porcentajes
    - Implementada lógica que siempre usa el número mínimo de ticks configurado
    - Sistema simplificado de cálculo directamente en ticks
  - Añadido método `_calculate_simple_reward_with_ticks()` en `enhanced_trading_env.py`:
    - Recompensa principal basada en PnL normalizado
    - Penalización por drawdowns significativos (>5%)
    - Bonificación por mantener posiciones rentables
    - Penalización suave por inactividad
    - **Nueva**: Bonificación especial por usar operaciones con los ticks mínimos establecidos
  - Actualizado método `step()` para usar la nueva función de recompensa
  - Añadido método `_update_ticks()` para seguimiento consistente de ticks positivos/negativos
  - Sistema preparado para entrenamientos largos que incentiva operaciones con protección adecuada
  - Resultados esperados: menor cantidad de operaciones, mayor win rate, mejor profit factor

# [30/03/2025 04:15:22] - v1.5.8

### Mejorado
- **Implementadas soluciones completas para mejorar el sistema de entrenamiento**:
  - Simplificado el sistema de recompensas en `environment/enhanced_trading_env.py`:
    - Nuevo método `_calculate_simple_reward()` con solo 4 componentes principales
    - Enfoque en PnL normalizado como componente primario
    - Penalización clara por drawdowns significativos (>5%)
    - Bonus por mantener posiciones rentables
    - Penalización suave por inactividad tras 100 pasos
  - Reducidas drásticamente las restricciones anti-hipertrading en `config/config.py`:
    - `min_hold_steps` reducido de 50 a 10
    - `position_cooldown` reducido de 60 a 15
    - Desactivado `force_min_hold` para mayor flexibilidad
    - Reducido `short_trade_penalty_factor` de 15.0 a 3.0
    - Ajustado `duration_scaling_factor` de 3.0 a 1.5
  - Rediseñado el curriculum learning en `train_lstm_gpu.py`:
    - Estructura de 3 fases bien definidas:
      - Fase 1 (0-33%): Exploración con restricciones mínimas
      - Fase 2 (33-66%): Refinamiento con restricciones moderadas
      - Fase 3 (66-100%): Optimización con restricciones razonables
    - Mejor balance de parámetros para progresión natural del aprendizaje
  - Implementado protocolo de entrenamiento consistente:
    - Mayor duración de entrenamiento (2M pasos)
    - Checkpoints automáticos cada 50k pasos
    - Evaluación detallada cada 25k pasos
    - Mayor paciencia para early stopping (200k pasos)
  - Creado sistema de benchmarking en `utils/benchmarking.py`:
    - Comparación sistemática de arquitecturas y configuraciones
    - Generación automática de grid de experimentos para múltiples parámetros
    - Evaluación detallada de métricas críticas (win rate, profit factor, etc.)
    - Visualizaciones automáticas de resultados para análisis comparativo
    - Sistema de ejecución paralela para máximo aprovechamiento de recursos

## [30/03/2025 00:03:25] - v1.5.7

### Corregido
- **Mejorado sistema de logging en train_lstm_gpu.py**:
  - Añadida creación automática del directorio de logs
  - Configurado archivo de log específico para entrenamiento LSTM
  - Asegurada persistencia de logs entre ejecuciones
  - Mejorada estructura de directorios para logs

## [30/03/2025 00:02:50] - v1.5.6

### Corregido
- **Mejorado sistema de cálculo de recompensas en enhanced_trading_env.py**:
  - Corregido error en cálculo de duración de operaciones
  - Implementada verificación robusta de campos entry_step y exit_step
  - Añadido manejo de operaciones cerradas sin exit_step
  - Mejorada lógica de penalización por hipertrading
  - Simplificada estructura de recompensas y penalizaciones
  - Optimizado sistema de penalización exponencial

## [30/03/2025 00:02:15] - v1.5.5

### Corregido
- **Añadida definición de training_timesteps en train_lstm_gpu.py**:
  - Agregada obtención del número total de pasos desde training_config
  - Establecido valor por defecto de 500 pasos si no está definido
  - Eliminada definición duplicada de total_timesteps

## [30/03/2025 00:01:40] - v1.5.4

### Corregido
- **Corregido método de carga de datos en train_lstm_gpu.py**:
  - Cambiado `load_data()` a `load_csv_data(file_path)`
  - Añadida construcción correcta de la ruta del archivo
  - Mejorado manejo de carga de datos desde directorio dataset

## [30/03/2025 00:01:05] - v1.5.3

### Corregido
- **Añadida importación faltante en train_lstm_gpu.py**:
  - Agregada importación de BASE_CONFIG desde config.config
  - Reorganizadas importaciones para mejor legibilidad
  - Corregida variable de tiempo total de entrenamiento

## [30/03/2025 00:00:30] - v1.5.2

### Corregido
- **Refactorizado train_lstm_gpu.py para mejor manejo de configuración**:
  - Corregido error de acceso a configuración del símbolo
  - Simplificada la inicialización de configuraciones
  - Mejorada la detección y configuración de GPU
  - Unificado el manejo de parámetros anti-hipertrading
  - Optimizada la estructura del código principal
  - Actualizada la forma de cargar datos usando base_config

## [29/03/2025 23:59:55] - v1.5.1

### Corregido
- **Actualizada configuración del símbolo de datos**:
  - Modificado `symbol` en `BASE_CONFIG` para usar el archivo de datos existente
  - Cambiado de 'NQ' a 'NQ_06-25_combined_20250320_225417'
  - Asegurada compatibilidad con el conjunto de datos actual en data/dataset

## [29/03/2025 23:59:22] - v1.5.0

### Mejorado
- **Implementadas correcciones drásticas anti-hipertrading para modelo LSTM**:
  - Modificada configuración del entorno con restricciones extremadamente estrictas:
    - Aumento de `min_hold_steps` a 50 (duplicado)
    - Aumento de `position_cooldown` a 60 (duplicado)
    - Incremento de `short_trade_penalty_factor` a 15.0 (casi duplicado)
    - Penalización por sobretrading aumentada a -10.0 (duplicada)
    - Retraso de recompensa aumentado a 15 pasos
    - Ratio asimétrico de 2.0 para penalizar más las pérdidas que premiar ganancias
  - Reescrito algoritmo de curriculum learning con 4 fases progresivas anti-hipertrading
  - Implementada función de cálculo de ticks mejorada con penalización exponencial para pérdidas
  - Añadido detector específico de hipertrading con penalización cuadrática
  - Modificados hiperparámetros PPO para LSTM:
    - Entropía aumentada a 0.5 para mayor exploración
    - Tasa de aprendizaje reducida a 0.00005 para cambios más lentos
    - Tamaño de lote aumentado a 256 para mayor estabilidad
    - Gamma aumentado a 0.99 para priorizar recompensas a largo plazo
    - Recorte (clipping) reducido a 0.1 para cambios de política más graduales
    - Añadido target_kl de 0.01 para limitar divergencia de política
  - Sistema preparado para entrenamiento de larga duración (5M pasos)

## [29/03/2025 23:17:18] - v1.4.1

### Ejecutado
- **Completado entrenamiento MLP en CPU**:
  - Entrenamiento de 500 pasos en 0.1 minutos
  - Política: MLP ([256, 128, 64])
  - Total operaciones: 8
  - Tasa de éxito: 12.5%
  - Profit Factor: 0.00
  - Modelo guardado en: results/MlpTradeEvolvePPO_20250329_231655

# CHANGELOG

## [29/03/2025 23:10:17] - v1.4.0

### Ejecutado
- **Completado entrenamiento LSTM en NVIDIA GeForce RTX 3060 Laptop GPU (parámetros anti-hipertrading v2)**:
  - Entrenamiento de 500 pasos en 0.2 minutos
  - Política: LSTM (256 unidades, 2 capas)
  - Total operaciones: 99
  - Tasa de éxito: 14.1%
  - Profit Factor: 0.25
  - Modelo guardado en: results/LstmTradeEvolvePPO_20250329_231001
  - Problema persistente: hipertrading (99 operaciones, mayoría con stop loss)

## [29/03/2025 23:09:33] - v1.3.9

### Mejorado
- **Configuración drásticamente más estricta anti-hipertrading**:
  - `min_hold_steps` aumentado drásticamente de 15 a 25
  - `position_cooldown` aumentado de 20 a 30
  - `short_trade_penalty_factor` aumentado de 4.0 a 8.0
  - `overtrade_penalty` aumentado de -3.0 a -5.0
  - `reward_delay_steps` aumentado de 5 a 10
  - Modificado sistema de recompensas con más peso a riesgo:
    - `pnl_weight` reducido de 3.0 a 2.5
    - `risk_weight` aumentado de 0.5 a 0.7
    - `drawdown_weight` aumentado de 0.2 a 0.4
    - `inactivity_weight` aumentado de 0.2 a 0.5

## [29/03/2025 23:08:30] - v1.3.8

### Ejecutado
- **Completado entrenamiento LSTM en NVIDIA GeForce RTX 3060 Laptop GPU (parámetros anti-hipertrading v1)**:
  - Entrenamiento de 500 pasos en 0.2 minutos
  - Política: LSTM (256 unidades, 2 capas)
  - Total operaciones: 99
  - Tasa de éxito: 15.2%
  - Profit Factor: 0.35
  - Modelo guardado en: results/LstmTradeEvolvePPO_20250329_230814
  - Problema identificado: hipertrading persistente pese a ajustes en el script LSTM

## [29/03/2025 23:07:44] - v1.3.7

### Corregido
- **Ajustes específicos para el script LSTM para estabilizar comportamiento**:
  - Aumentado el coeficiente de entropía (`ent_coef`) de 0.4 a 0.5
  - Cambio de función de activación de `ReLU` a `Tanh`
  - Reducida la tasa de aprendizaje de 0.0003 a 0.0002
  - Modificaciones específicas para LSTM sin afectar al MLP

## [29/03/2025 22:55:33] - v1.3.6

### Ejecutado
- **Completado entrenamiento MLP en CPU**:
  - Entrenamiento de 500 pasos en 0.1 minutos
  - Política: MLP ([256, 128, 64])
  - Total operaciones: 13
  - Tasa de éxito: 15.4%
  - Profit Factor: 3.26
  - Modelo guardado en: results/MlpTradeEvolvePPO_20250329_225516
  - Mejora significativa: reducción del número de operaciones (13 vs 99 del LSTM)

## [29/03/2025 22:50:11] - v1.3.5

### Ejecutado
- **Completado entrenamiento LSTM en NVIDIA GeForce RTX 3060 Laptop GPU**:
  - Entrenamiento de 500 pasos en 0.2 minutos
  - Política: LSTM (256 unidades, 2 capas)
  - Total operaciones: 99
  - Tasa de éxito: 18.2%
  - Profit Factor: 0.33
  - Problema identificado: 99 operaciones activaron stop loss (comportamiento hipertrader)
  - Modelo guardado en: results/LstmTradeEvolvePPO_20250329_224935

## [29/03/2025 22:47:03] - v1.3.4

### Ejecutado
- **Completado entrenamiento MLP en CPU**:
  - Entrenamiento de 500 pasos en 0.1 minutos
  - Política: MLP ([256, 128, 64])
  - Total operaciones: 7
  - Tasa de éxito: 28.6%
  - Profit Factor: 2.60
  - Modelo guardado en: results/MlpTradeEvolvePPO_20250329_224703

## [29/03/2025 22:45:00] - v1.3.3

### Corregido
- **Reorganización completa del CHANGELOG.md**:
  - Estructura estandarizada para todas las entradas con formato consistente
  - Secuencia correcta de versiones incrementales (v1.0.1 a v1.3.5)
  - Organización cronológica de todos los cambios
  - Categorización clara de cada tipo de cambio (Corregido, Mejorado, Optimizado, etc.)

## [29/03/2025 19:59:00] - v1.3.2

### Corregido
- **Cálculo de PnL en cierres por SL/TP**: Modificada la función `_close_position` en `environment/trading_env.py` para usar el precio exacto del nivel de SL/TP en lugar del precio de cierre de la barra, asegurando un registro de PnL más preciso para estas operaciones.
- **Errores de indentación**: Corregida indentación incorrecta introducida previamente en `environment/trading_env.py`.

## [29/03/2025 19:35:20] - v1.3.1

### Optimizado
- **Sistema de logging en train_lstm_gpu.py**:
  - Reducida frecuencia de monitorización de GPU de cada 5000 pasos a cada 50000 pasos
  - Eliminada duplicación de logs de monitorización de GPU en LstmTrainingCallback
  - Reducida frecuencia de registro de ticks de cada 1000 pasos a cada 10000 pasos
  - Redirigidos todos los mensajes de consola al logger estructurado en lugar de print()
  - Aumentada frecuencia de actualización de entropía adaptativa a cada 10000 pasos
  - Limpiada la salida de consola para mostrar solo información relevante del entrenamiento

## [29/03/2025 14:38:43] - v1.3.0

### Ejecutado
- **Completado entrenamiento MLP en CPU**:
  - Entrenamiento de 100000 pasos en 9.4 minutos
  - Política: MLP ([256, 128, 64])
  - Total operaciones: 2
  - Tasa de éxito: 0.0%
  - Profit Factor: 0.00
  - Modelo guardado en: results/MlpTradeEvolvePPO_20250329_142905

## [29/03/2025 13:47:30] - v1.2.9

### Ejecutado
- **Completado entrenamiento MLP en CPU**:
  - Entrenamiento de 500000 pasos en 13.9 minutos
  - Política: MLP ([256, 128, 64])
  - Total operaciones: 3
  - Tasa de éxito: 0.0%
  - Profit Factor: 0.00
  - Modelo guardado en: results/MlpTradeEvolvePPO_20250329_133308

## [28/03/2025 17:00:31] - v1.2.8

### Ejecutado
- **Completado entrenamiento MLP en CPU**:
  - Entrenamiento de 1000000 pasos en 15.5 minutos
  - Política: MLP ([256, 128, 64])
  - Total operaciones: 5
  - Tasa de éxito: 40.0%
  - Profit Factor: 10.00
  - Modelo guardado en: results/MlpTradeEvolvePPO_20250328_164441

## [28/03/2025 15:52:50] - v1.2.7

### Ejecutado
- **Completado entrenamiento MLP en CPU con early stopping activado**:
  - Entrenamiento de 1000000 pasos interrumpido en paso 105000 por early stopping
  - Early stopping activado por deterioro severo del rendimiento
  - Política: MLP ([256, 128, 64])
  - Total operaciones: solo 1 completada en evaluación
  - Tasa de éxito: 0.0%
  - Profit Factor: 0.00
  - Modelo persistiendo con acción constante (1.0) durante toda la evaluación
  - Posible causa: configuración demasiado restrictiva o problemas de convergencia
  - Modelo guardado en: results/MlpTradeEvolvePPO_20250328_153928

## [28/03/2025 15:24:56] - v1.2.6

### Optimizado
- **Mejora del sistema de aprendizaje para mayor rentabilidad**:
  - Simplificación del sistema de recompensas para enfocarse principalmente en PnL total
  - Reducción de parámetros complejos de curriculum learning de 5 a 3 fases más graduales
  - Equilibrio de parámetros asimétricos para no penalizar excesivamente las pérdidas
  - Mejora del balance exploración/explotación con decaimiento más lento de entropía
  - Establecimiento de ratios take-profit/stop-loss más equilibrados (1:1)
  - Cambio de arquitecturas neuronales y funciones de activación para mejor convergencia

## [28/03/2025 14:34:26] - v1.2.5

### Ejecutado
- **Completado entrenamiento MLP en CPU**:
  - Entrenamiento de 1000000 pasos en 10.1 minutos
  - Política: MLP ([384, 256, 128])
  - Total operaciones: 99
  - Tasa de éxito: 20.2%
  - Profit Factor: 0.14
  - Modelo guardado en: results/MlpTradeEvolvePPO_20250328_142421

## [28/03/2025 14:15:00] - v1.2.4

### Optimizado
- **Ajustada frecuencia de evaluación y logs de progreso**:
  - Reducida frecuencia de evaluación (eval_freq) de 25000 a 5000 pasos en train_lstm_gpu.py
  - Reducida frecuencia de impresión detallada (verbose_freq) de 10000 a 2500 pasos
  - Eliminada duplicación de definición de variables de seguimiento de métricas
  - Reorganizado código de inicialización en LstmTrainingCallback para mayor claridad
  - Resultado: información de progreso mostrada 5 veces más frecuentemente
  - Mayor visibilidad del avance del entrenamiento durante ejecuciones largas

## [28/03/2025 14:05:00] - v1.2.3

### Corregido
- **Solucionado problema de logs repetitivos en scripts de entrenamiento**:
  - Corregido comportamiento en `LstmTrainingCallback` y `AdaptiveLstmCallback` que causaba impresión constante de "Paso 0"
  - Eliminada la llamada recursiva `self.base_callback._on_step()` que generaba
