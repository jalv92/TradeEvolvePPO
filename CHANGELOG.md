# Changelog

## [0.4.1] - 2025-03-25 22:30:45

### Implementado
- **Sistema anti-sobretrading para operaciones más sostenibles**:
  - Implementado tiempo de enfriamiento obligatorio entre operaciones (10 pasos)
  - Aumentados significativamente los umbrales para cambio de posición (0.4 → 0.7)
  - Añadida penalización severa (-8.0) por operaciones de corta duración
  - Implementado factor de duración que aumenta el PnL según el tiempo de operación (hasta 2x)
  - Añadido bonus por duración de operaciones que crece con el tiempo

### Mejorado
- **Sistema de recompensas rebalanceado**:
  - Reforzado el factor de tiempo para posiciones rentables (hasta 2.0x)
  - Incrementado factor de beneficio para posiciones rentables (hasta 2.0x)
  - Duplicado el multiplicador para recompensas por mantener (5.0 → 10.0)
  - Registrada duración de posiciones para análisis y ajuste de recompensas
  - Mejorado logging para visibilidad del impacto de duración en recompensas

### Optimizado
- **Parámetros de entrenamiento anti-sobretrading**:
  - Añadida configuración para duración mínima de operaciones (5 pasos)
  - Implementado registro de factores de duración aplicados a operaciones
  - Aplicación selectiva de penalizaciones (mayores para operaciones muy cortas)
  - Mecanismo de cierre diferido para posiciones altamente rentables
  - Ignoradas acciones de apertura durante períodos de enfriamiento

## [0.4.0] - 2025-03-25 21:45:55

### Implementado
- **Curriculum Learning para desarrollo de estrategias sostenibles**:
  - Fase 1 (0-25%): Alta exploración, muchas operaciones, sin penalización por sobretrading
  - Fase 2 (25-50%): Transición gradual con incentivos crecientes para operaciones de mayor duración
  - Fase 3 (50-100%): Consolidación de estrategia, con fuerte incentivo para mantener posiciones rentables
  - Evolución dinámica de parámetros a lo largo del entrenamiento

- **Sistema de recompensas mejorado**:
  - Añadida recompensa específica para mantener posiciones rentables a lo largo del tiempo
  - Implementado factor de tiempo (mayor recompensa cuanto más tiempo se mantiene posición)
  - Implementado factor de beneficio (mayor recompensa cuanto más rentable es la posición)
  - Añadida penalización configurable por sobretrading (evitar operaciones muy seguidas)
  - Implementado sistema de recompensas retrasadas para aprendizaje a largo plazo

- **Aprendizaje supervisado avanzado**:
  - Entropía decreciente progresiva (desde 0.3 hasta 0.02)
  - Umbral más alto para cambiar posiciones (0.3 → 0.4)
  - Parámetros PPO optimizados para capturar patrones a largo plazo

### Mejorado
- **Optimización de hiperparámetros**:
  - Aumentado tamaño de buffer (n_steps) a 2048 para series temporales más largas
  - Incrementado batch_size a 256 para mejor generalización
  - Aumentado número de épocas a 15 para aprovechar mejor los datos
  - Mejorado gae_lambda a 0.97 para mejor estimación de ventaja
  - Aumentado gamma a 0.95 para valorar más las recompensas futuras

- **Sistema de visualización de operaciones**:
  - Añadida duración de operaciones en la visualización
  - Implementado manejo mejorado de errores para visualización de operaciones
  - Detección y conversión automática de formato de operaciones

- **Normalización de recompensas**:
  - Implementada compresión logarítmica para valores extremos de recompensa
  - Añadido clipping para mantener recompensas en rangos razonables
  - Mejor balanceo entre diferentes componentes de recompensa

## [0.3.8] - 2025-03-25 20:15:40

### Corregido
- Solucionado error crítico en el historial de operaciones:
  - Error: "KeyError: 'direction'" al intentar mostrar el historial de operaciones
  - Añadido campo 'direction' a todas las operaciones registradas
  - Implementado manejo robusto de errores en la visualización de operaciones
  - Conversión automática de posición numérica a etiqueta textual ('long'/'short')

### Mejorado
- Sistema de recompensas optimizado para incentivar más operaciones:
  - Añadido bonus explícito (2.0) por abrir posición
  - Incrementado el bonus por completar operaciones de 10.0 a 12.0
  - Aumentado el bonus por operaciones ganadoras de 50% a 80%
  - Reducido el umbral de penalización por mantener posiciones (de 20 a 15 pasos)
  - Aumentado el factor de escala para amplificar todas las recompensas (de 10.0 a 12.0)
  - Nuevo sistema de registro detallado de componentes de recompensa

### Implementado
- Monitorización adecuada de entornos:
  - Añadido wrapper Monitor a todos los entornos para registro de métricas
  - Solucionada advertencia "Evaluation environment is not wrapped with Monitor"
  - Implementados flags para rastrear aperturas y cierres de posiciones
  - Mejorado el seguimiento del PnL y recompensas por operación

## [0.3.7] - 2025-03-25 19:15:30

### Corregido
- Solucionado error crítico en la clase `PPOAgent`:
  - Error: "The environment is of type VecNormalize, not a Gymnasium environment"
  - Implementada detección inteligente para verificar si el entorno ya está vectorizado
  - Eliminada la doble vectorización que estaba causando el error
  - Mejorado el sistema de logging para proporcionar información sobre el estado del entorno

### Implementado
- Compatibilidad mejorada con entornos pre-vectorizados:
  - Manejo adecuado de VecEnv y sus subclases (VecNormalize, etc.)
  - Mayor flexibilidad para trabajar con diferentes configuraciones de entorno
  - Conservación de las propiedades de normalización configuradas externamente

## [0.3.6] - 2025-03-25 18:45:10

### Corregido
- Solucionado error crítico en el sistema de evaluación durante el entrenamiento:
  - Error: "Error while synchronizing normalization stats"
  - Implementado callback personalizado (CustomEvalCallback) para manejo adecuado de la sincronización
  - Corregida la inicialización de entornos para mantener coherencia entre entrenamiento y evaluación
  - Mejorada la gestión de nombres de variables para claridad (train_vec_env, eval_vec_env)
  - Añadida limitación de pasos máximos (500) para evitar loops infinitos en evaluación

### Mejorado
- Robustez general del sistema de entrenamiento:
  - Forzado el uso de CPU para el modelo para evitar problemas de incompatibilidad con GPU
  - Mejorado el manejo de errores para evitar terminaciones abruptas
  - Optimizado el formato de visualización de operaciones
  - Añadido sistema de almacenamiento estructurado para modelos en la carpeta 'models'
  - Silenciado el logger de gymnasium para evitar advertencias innecesarias

## [0.3.5] - 2025-03-25 17:55:20

### Corregido
- Solucionado error crítico en la configuración de los entornos de vectorización y normalización:
  - Error: "Error while synchronizing normalization stats: expected the eval env to be a VecEnvWrapper"
  - Implementada vectorización consistente para entornos de entrenamiento y evaluación
  - Añadida normalización de observaciones con VecNormalize a todos los entornos
  - Sincronizadas estadísticas entre entornos para garantizar consistencia
  - Mejorada la evaluación final con entorno dedicado para probar el modelo entrenado

### Mejorado
- Robustez en el flujo de entrenamiento y evaluación:
  - Implementado manejo de excepciones durante el entrenamiento
  - Añadida captura detallada de errores en la fase de evaluación
  - Optimizado el proceso de evaluación para generar suficientes operaciones de muestra
  - Configurados parámetros específicos para cada tipo de entorno (training, validation, test)

## [0.3.4] - 2025-03-25 17:20:35

### Corregido
- Solucionado error crítico en el método `_get_observation` que impedía el entrenamiento:
  - Error: "could not broadcast input array from shape (60,21) into shape (60,25)"
  - Implementada adaptación dinámica de las dimensiones de observación
  - Añadido relleno automático para datos con menos de 25 características
  - Recorte automático para datos con más de 25 características
  - Verificación final de forma para garantizar observaciones de tamaño (window_size, 25)

### Mejorado
- Robustez del sistema de observación del entorno:
  - Manejo correcto de excepciones durante la extracción de datos
  - Respuesta adecuada ante datos insuficientes o malformados
  - Mensajes de depuración detallados sobre la forma de los datos
  - Normalización preservando las dimensiones correctas

## [0.3.3] - 2025-03-25 16:45:12

### Añadido
- Creado script `train_test.py` para ejecución rápida de pruebas de entrenamiento:
  - Configurado para ejecutar 10,000 pasos de entrenamiento
  - Optimizado para fomentar la apertura de operaciones con alta entropía (0.3)
  - Implementada evaluación cada 1,000 pasos para monitoreo de progreso
  - Añadido sistema de visualización de últimas operaciones realizadas

### Implementado
- Configuración específica para pruebas de entrenamiento:
  - Penalización por inactividad incrementada (2.0) para forzar operaciones
  - Bonus por completar operaciones aumentado (10.0)
  - Umbral de inactividad reducido a 20 pasos
  - Recompensa base negativa (-0.05) para incentivar acciones

## [0.3.2] - 2025-03-25 15:30:45

### Corregido
- Solucionado error en el método `_get_observation` del entorno:
  - Corregido problema con el atributo `normalize_observation` inexistente
  - Añadido manejo adecuado de configuración de normalización desde el diccionario config
  - Implementada normalización opcional con StandardScaler cuando está habilitada
  - Mejorado filtrado de datos para manejar correctamente las columnas no numéricas
  - Optimizado manejo de excepciones con logging detallado de trazas de error

### Validado
- Verificado funcionamiento del entorno con diagnósticos exhaustivos:
  - Ejecutado script de diagnóstico con múltiples patrones de acción
  - Confirmados ciclos completos de apertura y cierre de posiciones
  - Validada la gestión adecuada de balance y operaciones

## [0.3.1] - 2025-03-25 14:25:30

### Corregido
- Solución definitiva al problema crítico que impedía la ejecución de operaciones:
  - Identificados y corregidos errores en los métodos `_open_position` y `_close_position`
  - Mejorada la implementación del método `get_performance_summary` para usar el nuevo formato de trades
  - Corregidas validaciones en la función `step` que provocaban que no se ejecutaran operaciones
  - Eliminadas dependencias de atributos que podían no existir en ciertas configuraciones

### Implementado
- Herramientas de diagnóstico completas para verificar el funcionamiento del entorno:
  - Creado script de comparación `compare_environments.py` para analizar el comportamiento entre entornos
  - Implementado entorno simplificado `SimpleTradingEnv` como referencia para testing
  - Añadida instrumentación con logs detallados en puntos críticos del código

### Confirmado
- Verificado funcionamiento correcto del entorno mediante pruebas:
  - Validado que el entorno ahora ejecuta operaciones de compra y venta correctamente
  - Confirmado que el sistema registra adecuadamente las transacciones en la lista de trades
  - Comprobado balance actualizado correctamente después de operaciones

## [0.1.87] - 2025-03-25 13:17:45

### Corregido
- Solucionado problema crítico de falta de operaciones durante el entrenamiento:
  - Implementada solución integral al desequilibrio del sistema de recompensas
  - Corregida penalización por inactividad que era demasiado débil (dividida por 10 en lugar de 100)
  - Mejorada interpretación de acciones con umbrales más decisivos (0.3 en lugar de 0.5)
  - Implementado sistema de monitoreo en tiempo real de comportamiento de trading

### Mejorado
- Sistema de recompensas optimizado para incentivar el trading:
  - Reducida penalización base por cada paso de -0.01 a -0.002
  - Aumentado el peso del PnL de 1.0 a 2.5
  - Reducido peso de drawdown de 0.05 a 0.01
  - Aumentado peso de inactividad de 0.5 a 2.0
  - Incrementado bonus por completar operaciones de 3.0 a 8.0
  - Aumentado factor de escala de recompensa de 2.0 a 5.0

### Optimizado
- Parámetros de exploración del agente PPO:
  - Aumentado coeficiente de entropía de 0.1 a 0.3
  - Incrementados pasos de exploración de 500K a 1M
  - Aumentada probabilidad de exploración de 0.3 a 0.5
  - Reducido umbral de inactividad de 50 a 20
  - Disminuido gamma de 0.99 a 0.95 para priorizar recompensas a corto plazo

### Añadido
- Nuevo script `train_enhanced.py` con características avanzadas:
  - Sistema de monitoreo de distribución de acciones en tiempo real
  - Detección automática de comportamiento sin operaciones
  - Ajuste dinámico de parámetros de exploración durante el entrenamiento
  - Métricas detalladas de trading con visualización mejorada
  - Sistema de guardado de configuración y modelos con timestamps

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
- Método `update_data`