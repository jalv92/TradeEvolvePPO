# Changelog

## [0.1.86] - 2025-03-24 22:10:52

### Corregido
- Optimizado sistema de recompensas para mejorar el entrenamiento:
  - Reducida penalización base por cada paso de -0.05 a -0.01
  - Disminuido peso del PnL de 3.0 a 1.0 para evitar inestabilidad
  - Reducida penalización por inactividad de 2.0 a 0.5
  - Disminuido factor de escala de recompensa de 10.0 a 2.0
  - Reducido bonus por completar operación de 5.0 a 3.0

### Optimizado
- Ajustados hiperparámetros del agente PPO para trading:
  - Reducida tasa de aprendizaje de 0.001 a 0.0001
  - Disminuidos n_steps de 16384 a 8192
  - Reducido batch_size de 2048 a 1024
  - Aumentado factor de descuento gamma de 0.90 a 0.99
  - Disminuido coeficiente de entropía de 0.5 a 0.1
  - Reducidos exploration_steps de 1000000 a 500000

### Mejorado
- Optimizada gestión de riesgo:
  - Reducido risk_per_trade_pct de 0.01 a 0.005
  - Disminuido stop_loss_pct de 0.02 a 0.01
  - Aumentado take_profit_ratio de 1.5 a 2.0
  - Reducido pnl_scale de 10.0 a 5.0

## [0.1.85] - 2025-03-24 19:08:49

### Añadido
- Herramientas de diagnóstico y optimización completas para resolver problemas de entrenamiento:
  - Creado script `debug_training.py` para realizar diagnóstico exhaustivo del sistema
  - Implementado `reward_optimizer.py` para experimentar con diferentes configuraciones de recompensas
  - Desarrollado `training/fix_callback.py` con callbacks robustos para manejar errores comunes
  - Creado script `train_fixed.py` que integra todas las mejoras y correcciones

### Mejorado
- Solución integral al problema del agente demasiado conservador:
  - Implementados múltiples perfiles de comportamiento (balanceado, agresivo, conservador, exploración)
  - Desarrollado sistema adaptativo de incentivos para operaciones de trading
  - Creado mecanismo avanzado de detección de sesgo hacia la inactividad
  - Optimizado curriculum learning con transiciones suaves entre etapas

### Corregido
- Implementadas soluciones para los principales problemas identificados:
  - Error en la actualización de progreso durante el entrenamiento
  - Problemas con atributos faltantes en objetos de callback
  - Detección incorrecta de entornos vectorizados
  - Sistema de recompensas desequilibrado que inhibía el trading

## [0.1.84] - 2025-03-23 20:00

### Corregido
- Solucionado problema de convergencia en el entrenamiento:
  - Modificada la arquitectura de la red neuronal para balancear representación y generalización
  - Implementada normalización de batch para reducir dependencia de la escala de los datos
  - Añadido dropout para prevenir sobreajuste durante entrenamiento prolongado
  - Optimizados hiperparámetros para evitar mínimos locales durante el aprendizaje

## [0.1.83] - 2025-03-23 19:45

### Corregido
- Error de importación en training/trainer.py:
  - Eliminada la importación incorrecta de setup_logger desde utils.helpers
  - La función setup_logger solo debe importarse desde utils.logger
  - Resuelto conflicto de doble importación que causaba ImportError

## [0.1.82] - 2024-03-23 14:08

### Corregido
- Solucionado error de importación `ModuleNotFoundError: No module named 'utils.metrics'` modificando el archivo `trainer.py` para importar `calculate_metrics` desde `evaluation.metrics` en lugar de `utils.metrics`.

## [0.1.81] - 2025-03-23 19:05

### Corregido
- Error de importación en training/trainer.py:
  - Eliminadas importaciones de funciones inexistentes 'get_device' y 'configure_torch'
  - Mantenida la importación válida de setup_logger
  - Evitado conflicto con importación duplicada de setup_logger (de utils.logger)

## [0.1.80] - 2025-03-23 18:58

### Corregido
- Error en el entrenamiento: 'PPOAgent' object has no attribute 'reset_metrics'
  - Implementado método reset_metrics en PPOAgent
  - Añadida verificación para comprobar si el método existe antes de llamarlo
- Error en la función de seguimiento: 'name traceback is not defined'
  - Añadida importación del módulo traceback en trainer.py

## [0.1.79] - 2025-03-23 17:10

### Modificado
- Implementación completa de recomendaciones para resolver problema de agente inactivo:
  - Modificado sistema de recompensas para forzar trading activo:
    - Aumentado pnl_weight a 3.0 (antes 2.0)
    - Reducido drawdown_weight a 0.05 (antes 0.15)
    - Aumentado inactivity_weight a 2.0 (antes 0.3)
    - Añadida penalización base por cada paso (-0.05)
    - Implementado bonus por operaciones (5.0) y cambio de dirección (0.2)
  - Forzada exploración obligatoria:
    - Extendido período de exploración forzada a 1,000,000 pasos
    - Implementado mecanismo de forzado con 40% de probabilidad
    - Añadido forzado adicional tras inactividad prolongada
  - Optimizados hiperparámetros para mayor exploración:
    - Aumentado coeficiente de entropía a 0.5 (antes 0.25)
    - Reducido gamma a 0.90 (antes 0.99)
    - Aumentado learning_rate a 0.001 (antes 0.0003)
  - Implementado curriculum learning progresivo más agresivo
  - Añadido sistema de monitoreo de distribución de acciones

### Agregado
- Nuevo sistema de monitoreo de métricas en tiempo real:
  - Implementado ActionDistributionCallback para visualizar acciones tomadas
  - Mejorado TqdmCallback para mostrar métricas relevantes en tiempo real
  - Añadido seguimiento de mejores métricas históricas
  - Creado sistema para detectar sesgo en las decisiones del modelo

## [0.1.78] - 2025-03-23 16:30

### Analizado
- Resumen del proceso de entrenamiento hasta 265,000 pasos:
  - El modelo ha alcanzado una recompensa máxima de 1691.89 en 225,000 pasos
  - No se han registrado operaciones de trading (0 trades) en ninguna evaluación
  - Problema persistente de inactividad a pesar de las optimizaciones en el sistema de recompensas
  - Las puntuaciones de recompensa han mejorado significativamente desde el inicio (-7323 → +1691)

### Problema identificado
- El agente está optimizando la recompensa evitando operar, posiblemente debido a:
  - Penalización por drawdown demasiado severa en comparación con recompensa por operaciones
  - Sistema de recompensa actual permite obtener puntuación positiva sin realizar trades
  - Posible fallo en el mecanismo de exploración forzada durante entrenamiento

## [0.1.76] - 2025-03-23 04:30

### Corregido
- Error al cargar modelo en el script `nt8_trader.py`:
  - Solucionado problema con el parámetro `device` al cargar el modelo
  - Cambiado `None` por `"auto"` para correcta detección de dispositivo
  - Añadido mensaje informativo sobre el dispositivo seleccionado
  - Mejorada documentación interna sobre parámetros aceptados

## [0.1.75] - 2025-03-23 12:50

### Corregido
- Solucionadas advertencias y errores en la ejecución del entrenamiento:
  - Corregido error de SettingWithCopyWarning en environment/trading_env.py usando .loc[]
  - Solucionado problema de codificación del carácter □ en los logs de entrenamiento
  - Mejorada la gestión de columnas adicionales en los datos de observación

## [0.1.74] - 2025-03-23 12:30

### Corregido
- Solucionado error crítico con callbacks en el entrenamiento:
  - Reemplazada función de actualización de progreso con clase TqdmCallback que extiende BaseCallback
  - Corregidos errores de atributos faltantes en objetos de callback
  - Implementada detección robusta de entornos para mostrar métricas en la barra de progreso
  - Mejorado manejo de excepciones durante actualización de métricas

## [0.1.73] - 2025-03-23 12:15

### Corregido
- Solucionado error crítico durante el inicio del entrenamiento:
  - Implementado método `create_callback()` faltante en la clase `Trainer`
  - Corregida incompatibilidad de atributos en la clase `TradeCallback`
  - Actualizado sistema de almacenamiento y visualización de métricas
  - Corregido error en la ejecución del comando de entrenamiento

## [0.1.72] - 2025-03-23 12:00

### Mejorado
- Reducida significativamente la verbosidad de logs para enfocarse solo en información esencial:
  - Cambiado nivel de logging general a WARNING
  - Cambiado nivel de consola a ERROR para mostrar solo errores críticos
  - Desactivados logs de componentes de recompensa y estadísticas del sistema
  - Incrementada frecuencia de logging a cada 20000 pasos
  - Implementada barra de progreso simple con métricas clave de trading
  - Simplificada presentación de resultados, mostrando solo datos de operaciones relevantes

### Optimizado
- Modificado `train_2model.py` para mostrar solo información crítica:
  - Añadida barra de progreso con tqdm para seguimiento visual
  - Silenciadas bibliotecas externas que generan ruido en logs (matplotlib, PIL)
  - Formato más conciso para resultados de entrenamiento y evaluación

## [0.1.71] - 2025-03-23 11:00

### Modificado
- Aumentado dramáticamente el énfasis en exploración para resolver el problema de agente demasiado conservador:
  - Incrementado coeficiente de entropía de 0.1 a 0.25 para forzar mayor exploración
  - Reducido umbral de inactividad de 50 a 25 pasos
  - Aumentado peso de penalización por inactividad de 0.1 a 0.3
  - Implementado mecanismo de exploración forzada durante los primeros 500 pasos
  - Añadida recompensa directa por abrir posiciones (exploration_weight: 0.5)
  - Forzada acción aleatoria después de períodos prolongados de inactividad

### Corregido
- Solucionado problema de agente demasiado conservador que no abre operaciones:
  - Simplificado sistema de recompensas para priorizar acciones de trading
  - Implementada penalización exponencial por inactividad
  - Añadida recompensa específica por mantener posiciones rentables

## [0.1.70] - 2025-03-23 10:25

### Añadido
- Iniciado entrenamiento de 4 millones de pasos utilizando CUDA con el archivo de datos "NQ_06-25_combined_20250320_225417.csv"
- Configurado entrenamiento progresivo con pasos [400000, 1200000, 2400000, 3600000]

### Modificado
- Actualizado script train_2model.py para utilizar CUDA durante el entrenamiento
- Ajustados parámetros para utilizar el nuevo sistema de recompensas

## [0.1.69] - 2025-03-22 20:15

### Corregido
- Error crítico `Unexpected observation shape (60, 9) for Box environment, please use (60, 25)`:
  - Implementado nuevo método `_expand_data_for_model` para adaptar datos de NinjaTrader al formato esperado
  - Añadido cálculo automático de indicadores técnicos (SMA, RSI, MACD) para completar las 25 características
  - Incorporado sistema de relleno de columnas faltantes hasta llegar a dimensiones (n, 25)
  - Mejorada la detección de datos insuficientes con manejo adecuado cuando no hay suficientes filas

### Mejorado
- Ampliado conjunto de indicadores técnicos disponibles para operación:
  - Medias móviles simples (SMA) de 5, 10 y 20 períodos
  - RSI con período de 14 barras
  - MACD con configuración estándar (12, 26, 9)
  - Sistema de normalización y prevención de NaN en los indicadores
- Implementada mejor detección de requisitos de columnas, eliminando dependencia estricta de 'timestamp'

## [0.1.68] - 2025-03-22 19:40

### Corregido
- Error crítico `KeyError: "['timestamp'] not in index"` al procesar datos sin columna timestamp:
  - Eliminada dependencia de `self.feature_names` que podía contener columnas no existentes
  - Implementada detección automática de columnas disponibles en el DataFrame actual
  - Agregado sistema de ajuste dinámico de `self.feature_columns` y `self.num_features`
  - Mejorada inicialización de `self.normalize_columns` para prevenir errores AttributeError

### Mejorado
- Robustez del sistema de extracción de datos de la ventana de observación:
  - Implementada captura de excepciones detallada durante la extracción de datos
  - Añadido registro extenso de errores con trazas completas para mejor diagnóstico
  - Agregados mensajes de depuración sobre forma y dimensiones de los datos
  - Mejorada adaptabilidad a diferentes estructuras de datos

## [0.1.67] - 2025-03-22 19:05

### Corregido
- Error crítico `single positional indexer is out-of-bounds` cuando se actualizan datos del entorno:
  - Implementada verificación exhaustiva del índice current_step dentro de los límites del DataFrame
  - Añadido manejo defensivo para DataFrames vacíos retornando observaciones con ceros
  - Corregida validación de ventanas de datos para asegurar índices válidos
  - Mejorado manejo de excepciones al acceder a precios en el método _get_observation

### Mejorado
- Flexibilidad en el formato de datos recibidos de NinjaTrader 8:
  - Eliminado requisito estricto de columna "timestamp", usando el índice del DataFrame en su lugar
  - Implementada detección automática de índices datetime y conversión cuando es necesario
  - Añadido soporte para índices numéricos simples cuando no hay timestamps disponibles
  - Mejorado logging con información detallada sobre la estructura de datos y errores

## [0.1.66] - 2025-03-22 18:10

### Corregido
- Error crítico `float() argument must be a string or a real number, not 'Timestamp'`:
  - Modificado el método de actualización del entorno para usar sólo datos numéricos
  - Eliminada la columna timestamp del DataFrame enviado al modelo
  - Implementado nuevo enfoque usando el timestamp como índice del DataFrame
  - Añadida conversión explícita a numérico para todos los campos
  - Mejorado sistema de reemplazo de valores NaN por ceros

### Mejorado
- Proceso de depuración para problemas de actualización del entorno:
  - Añadidos mensajes detallados sobre tipos de datos en cada etapa
  - Implementado registro completo de trazas de error para diagnóstico
  - Mejorada legibilidad de la información sobre el DataFrame usado en el modelo

## [0.1.65] - 2025-03-22 17:50

### Corregido
- Errores críticos en el procesamiento de datos de barras en NT8Client:
  - Implementada validación exhaustiva de campos requeridos en los datos de barras
  - Añadida conversión robusta de tipos para todos los campos (timestamp, valores numéricos)
  - Corregido método de actualización de barras existentes para evitar problemas con índices
  - Mejorado manejo de excepciones con trazas completas para facilitar diagnóstico

### Mejorado
- Robustez del sistema de almacenamiento de datos históricos:
  - Implementada conversión explícita de todos los campos a sus tipos apropiados
  - Añadido sistema de logging para seguimiento detallado de cada actualización
  - Mejorada legibilidad de mensajes de error con información contextual completa

## [0.1.64] - 2025-03-22 17:35

### Añadido
- Nueva opción `--force-cpu` para el script `nt8_trader.py`:
  - Permite forzar el uso de CPU para el modelo PPO
  - Evita advertencias y problemas relacionados con CUDA/GPU
  - Mejora la estabilidad en sistemas sin GPU dedicada

### Mejorado
- Manejo de dispositivo (CPU/GPU) en `PPOAgentInference`:
  - Detección automática de configuración forzada de CPU
  - Adición de parámetro `device` en la carga del modelo
  - Mejor información de diagnóstico sobre el dispositivo utilizado
- Sistema de detección de variables de entorno para CUDA

## [0.1.63] - 2025-03-22 17:05

### Corregido
- Error persistente "Must have equal len keys and value when setting with an iterable":
  - Implementada solución alternativa creando el DataFrame desde cero usando un diccionario
  - Añadido sistema de depuración detallada para identificar la fuente exacta del problema
  - Agregado mecanismo para convertir automáticamente el campo 'timestamp' a datetime
  - Mejorado el manejo de errores con trazas completas para identificar problemas subyacentes

### Mejorado
- Información de diagnóstico durante la conexión con NinjaTrader 8:
  - Añadidos logs detallados mostrando la estructura completa de los datos recibidos
  - Implementada visualización de la primera fila del DataFrame para verificar formato
  - Agregado sistema de depuración que muestra todos los tipos de datos en el DataFrame

## [0.1.62] - 2025-03-22 16:35

### Corregido
- Error crítico "Must have equal len keys and value when setting with an iterable":
  - Implementada limpieza exhaustiva de datos recibidos de NinjaTrader 8
  - Filtrado de columnas para asegurar que solo las necesarias se pasan al entorno
  - Eliminación de columna 'instrument' que causaba conflictos con las expectativas del entorno
  - Añadida depuración detallada para identificar problemas en estructura de datos

### Mejorado
- Robustez del sistema para manejar datos inconsistentes de NinjaTrader 8:
  - Verificación explícita de columnas requeridas
  - Limpieza preventiva de datos antes de pasarlos al entorno
  - Reorganización de índices para garantizar consistencia

## [0.1.61] - 2025-03-22 16:10

### Añadido
- Método `update_data` en la clase `TradingEnv` para actualizar datos en tiempo real:
  - Solucionado error "Must have equal len keys and value when setting with an iterable"
  - Implementada validación de columnas requeridas en los datos recibidos
  - Añadido soporte para actualizar datos sin reiniciar el entorno

### Mejorado
- Proceso de actualización de entorno con datos de NinjaTrader 8

## [0.1.60] - 2025-03-22 15:45

### Añadido
- Nueva clase `PPOAgentInference` para cargar modelos pre-entrenados:
  - Implementada versión simplificada para uso exclusivo en inferencia
  - Solucionado error `PPOAgent.__init__() missing 1 required positional argument: 'config'`
  - Añadido soporte para carga directa de modelos Stable-Baselines3

### Cambiado
- Mejorado método `_setup_environment_and_model()` para usar la nueva clase de inferencia
- Simplificado proceso de carga de modelos para operación en vivo

## [0.1.59] - 2025-03-22 15:35

### Corregido
- Error en el entorno TradingEnv que impedía su inicialización con DataFrame vacío:
  - Eliminada validación `Data length (0) must be greater than window size (60)` para el modo "inference"
  - Permitida inicialización del entorno sin datos históricos para operación en vivo
  - Adaptado el entorno para recibir datos en tiempo real de NinjaTrader 8

## [0.1.58] - 2025-03-22 15:20

### Corregido
- Error crítico al intentar utilizar el modelo con NinjaTrader 8:
  - Añadida verificación para manejar DataFrames vacíos en TradingEnv
  - Implementada solución al error "zero-size array to reduction operation minimum which has no identity"
  - Añadidos valores predeterminados para la normalización de precios cuando no hay datos iniciales

## [0.1.57] - 2025-03-22 15:10

### Corregido
- Error al inicializar el entorno TradingEnv:
  - Añadido DataFrame vacío como parámetro requerido en la creación del entorno
  - Implementado manejo adecuado de columnas del DataFrame inicial
  - Solucionado error "TradingEnv.__init__() missing 1 required positional argument: 'data'"

### Cambiado
- Mejorada implementación del método `_setup_environment_and_model()` para compatibilidad con NinjaTrader 8

## [0.1.56] - 2025-03-22 14:45

### Añadido
- Configurado sistema para evaluar el modelo "2Model" en NinjaTrader 8:
  - Corregido script `nt8_trader.py` para funcionar con el modelo entrenado
  - Preparada conexión con NinjaTrader 8 a través del puerto 5555
  - Configurado ambiente de ejecución para evaluación en tiempo real

### Cambiado
- Ajustado método `_setup_environment_and_model()` para cargar correctamente el modelo
- Corregida inicialización del DataLoader

## [0.1.55] - 2025-03-22 14:23

### Añadido
- Iniciado entrenamiento del modelo "2Model":
  - Training ejecutándose satisfactoriamente con 2 millones de pasos objetivo
  - Corregido configuración del logger para usar niveles de logging como strings
  - Añadido logging detallado para seguimiento de progreso

### Cambiado
- Mejorado script `train_2model.py` para facilitar monitoreo de entrenamiento:
  - Implementado mejor manejo de excepciones
  - Agregado reportes de métricas en tiempo real

## [0.1.54] - 2025-03-22 14:02

### Añadido
- Creado nuevo script `train_2model.py` para iniciar entrenamiento optimizado:
  - Configurado para ejecutar 2 millones de pasos (antes 800k)
  - Implementado sistema de directorios específicos para resultados
  - Agregado logging detallado de parámetros de entrenamiento

### Cambiado
- Optimizado sistema de recompensas para fomentar trading más activo:
  - Aumentado peso de recompensa PnL de 1.0 a 1.5
  - Incrementado penalización por inactividad de -0.05 a -0.08
  - Reducido umbral de inactividad de 100 a 50 pasos
  - Balanceado mejor las recompensas (open_reward: 0.15, win_reward: 0.4)
- Mejorado algoritmo PPO para mayor exploración y eficiencia:
  - Aumentado coeficiente de entropía de 0.1 a 0.15
  - Reducido factor gamma de 0.95 a 0.92 para priorizar recompensas inmediatas
  - Duplicado tamaño de n_steps de 256 a 512 para capturar secuencias más largas
- Actualizado curriculum learning para 2M de pasos:
  - Nuevos puntos de progresión: [200k, 600k, 1.2M, 1.8M]
  - Umbrales de inactividad progresivamente más estrictos: [50, 40, 30, 20]

## [0.1.53] - 2025-03-22 09:45

### Añadido
- Implementado nuevo sistema de recompensas basado en principios de trading profesional:
  - Recompensa fija (+0.1) por tomar acción para fomentar la exploración desde el inicio
  - Recompensa por PnL normalizada por tamaño de posición y precio de entrada
  - Recompensa específica (+0.5) por usar correctamente stop-loss y take-profit
  - Sistema de penalización por operar en alta volatilidad sin éxito
  - Mecanismo completo de seguimiento de componentes de recompensa para análisis

### Cambiado
- Rediseñada completamente la función de recompensa en el entorno de trading:
  - Reemplazado sistema anterior por arquitectura más clara y modular
  - Actualizado método de seguimiento de pasos con posición e inactividad
  - Modificada la gestión de posiciones para el nuevo modelo de recompensas
  - Ajustado el espacio de acción a formato continuo para mejor control
  - Balanceados valores de componentes para equilibrar recompensas/penalizaciones

### Optimizado
- Mejorado el sistema de entrenamiento forzado para garantizar exploración:
  - Periodo inicial de 300 pasos con 90% de probabilidad de acciones aleatorias
  - Implementación de forzado adicional cuando hay inactividad prolongada
  - Sistema de cierre forzado de posiciones mantenidas por más de 40 pasos
  - Actualización de hiperparámetros PPO para mejor aprovechamiento de GPU

## [0.1.52] - 2025-03-22 04:15

### Cambiado
- Mejorado sistema de logging para reducir la verbosidad en la consola:
  - Separación de niveles de log para consola (WARNING) y archivos (INFO)
  - Consola muestra solo mensajes importantes mientras archivos mantienen detalles completos
  - Implementado contador para limitar logs de depuración LSTM (solo cada 100 iteraciones)

### Optimizado
- Visualización en terminal más clara durante entrenamiento:
  - Eliminado exceso de mensajes de depuración que dificultaban seguir el progreso
  - Mantenida toda la información detallada en archivos de log para análisis posterior
  - Mejorada capacidad de monitoreo de progreso de entrenamiento en tiempo real

## [0.1.51] - 2025-03-22 03:55

### Corregido
- Error crítico en la implementación de LSTMPolicy con DiagGaussianDistribution:
  - Añadido parámetro self.log_std inicializado como nn.Parameter(torch.zeros(action_space.shape[0]))
  - Actualizado método forward() para pasar correctamente mean_actions y log_std a la distribución
  - Corregidas todas las referencias a proba_distribution() para incluir el parámetro log_std
  - Modificados métodos evaluate_actions() y _predict() para utilizar el parámetro log_std
  - Solucionado error TypeError: DiagGaussianDistribution.proba_distribution() missing 1 required positional argument: 'log_std'

### Optimizado
- Implementación mejorada de LSTM para mejor rendimiento:
  - Refinado el sistema de preprocesamiento de observaciones para LSTM
  - Mejorada la gestión de dimensiones de entrada/salida
  - Añadidos registros detallados para facilitar la depuración durante el entrenamiento
  - Verificado rendimiento con aceleración CUDA

## [0.1.50] - 2025-03-22 04:05

### Cambiado
- Reemplazada arquitectura híbrida por arquitectura puramente LSTM:
  - Eliminada completamente la arquitectura MLP posterior al LSTM
  - Implementado modelo LSTM puro para política y función de valor
  - Simplificada la estructura de la red neuronal eliminando capas innecesarias
  - Optimizada la implementación para aprovechar mejor la GPU

### Corregido
- Inconsistencias en la implementación anterior:
  - Eliminada la clase LSTMExtractor que servía como intermediaria
  - Corregidos parámetros para utilizar el mismo LSTM para política y valor
  - Simplificada la configuración al eliminar configuraciones MLP redundantes
  - Mejorada la compatibilidad con stable-baselines3 implementando todos los métodos necesarios

## [0.1.49] - 2025-03-22 03:35

### Añadido
- Implementación de redes LSTM para mejorar el aprovechamiento de GPU:
  - Desarrollado archivo `agents/lstm_policy.py` con arquitectura LSTM completa
  - Creadas clases LSTMPolicy y LSTMExtractor para procesamiento de series temporales
  - Incorporado soporte para LSTM bidireccional (configurable)
  - Configurado uso de múltiples capas LSTM apiladas (2 por defecto)

### Optimizado
- Mejoras en la configuración para aprovechar GPU:
  - Reducción del tamaño de red MLP posterior al procesamiento LSTM
  - Configuración de tamaño de LSTM (256) y dimensiones de características (128)
  - Implementación de batch_first=True para mejor rendimiento en GPU
  - Parámetros específicos para LSTM centralizados en archivo de configuración

### Cambiado
- Modificado `agents/ppo_agent.py` para usar LSTM:
  - Añadida detección de política LSTM personalizada
  - Ajustada creación del modelo PPO para utilizar la nueva arquitectura
  - Mantenida compatibilidad con políticas anteriores

## [0.1.48] - 2025-03-22 08:30

### Añadido
- Sistema de recompensa por exploración y curiosidad:
  - Nueva componente de recompensa específica para exploración con peso 1.0
  - Bonus significativo (0.2) por abrir operaciones nuevas
  - Incentivo (0.1) por alternar entre posiciones largas y cortas
  - Mecanismo de entrenamiento forzado extendido a 300 pasos (era 150)
  - Recompensa adicional por mantener posiciones activas en el mercado

### Cambiado
- Reestructuración completa del sistema de recompensas:
  - Reducción del peso de PnL de 1.2 a 0.8 para favorecer exploración
  - Aumento de win_rate_weight de 0.3 a 0.4 para incentivar aciertos
  - Reducción de penalizaciones por pérdidas de 0.8 a 0.5
  - Forzado probabilístico de operaciones cuando hay inactividad (25+ pasos)
  - Aumento del período de mantenimiento máximo de posiciones de 30 a 40 pasos

### Optimizado
- Hiperparámetros de PPO para maximizar exploración:
  - Increment
