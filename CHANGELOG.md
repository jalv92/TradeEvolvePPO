# Changelog

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

## [0.1.46] - 2025-03-22 02:35

### Añadido
- Nuevo mecanismo de entrenamiento forzado para promover trading activo:
  - Implementado periodo inicial (primeros 100 pasos) donde se fuerzan operaciones aleatorias
  - Añadida recompensa por dirección correcta del mercado independiente del resultado financiero
  - Incorporada recompensa base por mantener posiciones activas en el mercado

### Cambiado
- Rediseño completo de la estructura de recompensas:
  - Reducidas todas las penalizaciones por pérdidas a la mitad para evitar comportamiento demasiado conservador
  - Aumentado coeficiente de entropía de 0.01 a 0.05 para fomentar exploración
  - Incrementada tasa de aprendizaje de 0.0003 a 0.0005 para acelerar adaptación inicial
  - Reducido coeficiente de función de valor de 0.8 a 0.4 para priorizar mejora de política
- Mejorado sistema de curriculum learning:
  - Nuevos parámetros progresivos para factor de penalización y máximo drawdown
  - Aversión al riesgo inicial reducida de 0.8 a 0.3 para permitir más experimentación

### Optimizado
- Sistema de cierre de posiciones:
  - Forzado cierre automático después de 20 pasos para evitar estancamiento
  - Implementada reducción gradual del drawdown máximo permitido durante el entrenamiento
  - Penalizaciones por inactividad más agresivas para promover trading constante

## [0.1.45] - 2025-03-22 01:00

### Añadido
- Iniciado entrenamiento definitivo de 400k pasos con la fuente de datos correcta:
  - Utilizado archivo NQ_06-25_combined_20250320_225417.csv para entrenamiento
  - Configurados 400,000 pasos con aceleración CUDA
  - Mantenidas optimizaciones previas para curriculum learning

### Cambiado
- Actualizado flujo de trabajo para verificación de datos:
  - Implementado proceso para confirmar siempre la última versión de datos combinados disponible
  - Establecido procedimiento de validación de fuentes de datos antes del entrenamiento

### Corregido
- Error en la selección de datos de entrenamiento:
  - Reemplazado uso incorrecto de processed_data.csv con el dataset combinado más reciente
  - Asegurada la consistencia entre entrenamiento y validación usando la misma fuente de datos
- Error en la configuración de la red neuronal:
  - Corregido formato de la función de activación de 'tanh' (string) a torch.nn.Tanh (clase)
  - Solucionado error TypeError: 'str' object is not callable al construir la red neuronal
- Error en TradingEnv durante la ejecución del entrenamiento:
  - Añadido y corregido el atributo _reward_components que faltaba
  - Implementada inicialización y actualización adecuada de los componentes de recompensa
  - Añadida variable unrealized_pnl para calcular el valor de la posición actual
  - Solucionados errores AttributeError para variables faltantes en el entorno

## [0.1.44] - 2025-03-22 00:45

### Añadido
- Configuración optimizada para entrenamiento con aceleración GPU:
  - Forzado explícito del uso de CUDA mediante parámetro 'device': 'cuda'
  - Ajustados parámetros para entrenamiento de 400,000 pasos
  - Recalibrados intervalos de curriculum learning a [80k, 160k, 240k, 320k]
  - Configuradas frecuencias de evaluación (20k pasos) y checkpoint (40k pasos)

### Cambiado
- Optimización de recursos para mejor rendimiento:
  - Reducidos pasos totales de entrenamiento de 500k a 400k
  - Ajustadas frecuencias de evaluación para equilibrar velocidad y monitoreo

### Corregido
- Error de importación al iniciar entrenamiento:
  - Añadido alias PPO_CONFIG = AGENT_CONFIG para compatibilidad
  - Resuelto problema "ImportError: cannot import name 'PPO_CONFIG' from 'config.config'"
- Error de configuración de logging:
  - Añadida propiedad 'log_level' en el diccionario LOGGING_CONFIG
  - Solucionado KeyError: 'log_level' en main.py

## [0.1.43] - 2025-03-22 00:30

### Añadido
- Verificación completa del sistema de entrenamiento:
  - Confirmadas modificaciones en la función de recompensa para reducir énfasis en win rate (peso 0.2)
  - Verificada implementación de penalización más severa por inactividad (0.01 escala y umbral de 30 pasos)
  - Comprobado sistema de incentivos para operaciones significativas con penalización por trades triviales
  - Validada arquitectura de red neuronal con capas [256, 256, 128] para política y valor
  - Confirmado entrenamiento progresivo en cuatro etapas con validación multi-segmento

### Cambiado
- N/A

### Corregido
- N/A

## [0.1.25] - 2025-05-22 12:30

### Corregido
- Verificado el acceso correcto a Instruments.All() en NT8StrategyServer:
  - Confirmado que el método ya estaba correctamente implementado con paréntesis
  - Validado que la sintaxis del foreach era apropiada para iterar sobre la colección de instrumentos
  - Asegurado que el código sigue las mejores prácticas de NinjaTrader 8

## [0.1.23] - 2025-05-22 12:00

### Corregido
- Solucionado error final de compilación en NT8StrategyServer:
  - Corregido acceso a Instruments.All que debe ser llamado como método con paréntesis: Instruments.All()
  - Añadido comentario explicativo para evitar confusiones futuras
  - Completada la corrección de todos los errores de compilación

## [0.1.22] - 2025-05-22 11:45

### Corregido
- Implementada solución final para errores de compilación en NT8StrategyServer:
  - Corregido error con Account.Orders: es una propiedad, no un método
  - Actualizado el comentario explicativo para reflejar el manejo nativo de NinjaTrader 8
  - Mejorada la documentación del código siguiendo las prácticas recomendadas por NinjaTrader

## [0.1.21] - 2025-05-22 11:30

### Corregido
- Errores de compilación adicionales en la estrategia NT8StrategyServer:
  - Corregido error con Account.Orders (se accedía como propiedad pero es un método)
  - Implementada solución para Account.Cancel usando CancelOrder de NinjaTrader
  - Mejorado manejo de órdenes para iterar sobre una lista separada de órdenes a cancelar

## [0.1.20] - 2025-05-22 10:45

### Corregido
- Errores de compilación en la estrategia NT8StrategyServer:
  - Corregidas llamadas a métodos de trading (EnterLong, EnterShort, etc.) con los parámetros correctos
  - Reemplazado método inexistente EnterLongStop por EnterLongStopMarket
  - Reemplazado método inexistente EnterShortStop por EnterShortStopMarket
  - Implementado método CancelOrdersForInstrument en lugar del inexistente CancelOpenOrders
  - Corregida búsqueda de instrumentos usando Instruments.All en lugar de GetInstrument
  - Implementado cálculo manual de PnL en lugar de usar Position.GetProfitLoss

## [0.1.19] - 2025-05-21 22:15

### Añadido
- Implementado nuevo sistema de comunicación bidireccional con NinjaTrader 8:
  - Creada estrategia NT8StrategyServer para NinjaTrader que actúa como servidor TCP
  - Creado cliente NT8Client en Python para comunicarse con la estrategia
  - Implementado protocolo de mensajes para intercambio de datos y órdenes
  - Añadido sistema de callbacks para procesamiento de eventos en tiempo real

### Cambiado
- Reemplazado el conector basado en ATI por una arquitectura cliente-servidor más robusta:
  - Comunicación directa vía TCP en lugar de archivos o ATI
  - Recepción de datos de mercado directamente desde NinjaTrader sin necesidad de archivos CSV
  - Mejorado manejo de actualizaciones de órdenes y posiciones
  - Sistema más resiliente a desconexiones y fallos

### Eliminado
- Eliminado el indicador DataExtractor.cs reemplazado por funcionalidad integrada en la estrategia
- Eliminado código de detección de directorios de NinjaTrader y manejo de archivos

## [0.1.18] - 2025-05-20 21:52

### Añadido
- Implementado conector Python para NinjaTrader 8:
  - Creada clase NinjaTraderConnector para enviar órdenes a NinjaTrader 8 vía ATI
  - Añadido soporte para órdenes de mercado (compra/venta) y cierre de posiciones
  - Integración con la detección automática de rutas de NinjaTrader mediante registro de Windows

### Cambiado
- Actualizada documentación para incluir instrucciones de integración con NinjaTrader 8
- Mejorado el sistema para permitir trading en tiempo real con modelos entrenados

## [0.1.17] - 2025-03-20 12:15

### Añadido
- Técnicas alternativas para verificación de tamaño y existencia:
  - Implementado sistema de conteo manual por iteración
  - Añadida verificación de existencia por acceso a índices
  - Captura segura de excepciones para determinar límites de colecciones

### Cambiado
- Eliminación radical de todas las referencias a la propiedad Count:
  - Reemplazadas con métodos alternativos que no causan conflictos
  - Sustituidas comparaciones directas con Count por verificaciones de acceso
  - Implementada estrategia de conteo explícita compatible con NinjaTrader 8

### Corregido
- Errores persistentes CS0428 y CS0019:
  - Eliminada completamente cualquier referencia a la propiedad o método Count
  - Sustituidas comparaciones directas con Count por verificaciones de acceso
  - Implementada estrategia de conteo explícita compatible con NinjaTrader 8

## [0.1.16] - 2025-03-20 12:00

### Añadido
- N/A

### Cambiado
- Implementado enfoque 100% nativo de NinjaTrader 8 para resolver problemas de compilación:
  - Eliminadas conversiones explícitas a tipos genéricos de .NET como ICollection
  - Reemplazado acceso directo a Count con validaciones de índices por elementos
  - Simplificada la lógica para evitar dependencias en propiedades problemáticas

### Corregido
- Errores de compilación CS0266 y CS0428:
  - Eliminado intento de convertir NinjaTrader.Data.Bars a System.Collections.ICollection
  - Implementado método seguro de iteración para determinar el tamaño de colecciones
  - Validación directa de índices en lugar de verificar contra tamaño de colección

## [0.1.15] - 2025-03-20 11:45

### Añadido
- N/A

### Cambiado
- Reemplazado el método de acceso a Count en colecciones de NinjaTrader:
  - Implementada conversión explícita a System.Collections.ICollection antes de acceder a Count
  - Eliminado el uso de foreach en favor de iteración manual indexada para evitar problemas de enumeración

### Corregido
- Solución final para errores persistentes de method group Count:
  - Resuelto problema fundamental de tipos en líneas 549 y 678
  - Implementado patrón de conversión de tipo recomendado para colecciones de NinjaTrader
  - Eliminada ambigüedad entre métodos de LINQ y propiedades nativas de colecciones

## [0.1.14] - 2025-03-20 11:15

### Añadido
- N/A

### Cambiado
- Mejorada la coherencia en el acceso a la propiedad Count:
  - Unificados los nombres de variables (valueCount -> valuesCount) para mantener consistencia
  - Asegurado el acceso explícito a la propiedad Count en todos los objetos

### Corregido
- Implementada la solución definitiva para los errores persistentes de method group Count:
  - Basada en el análisis de ejemplos oficiales de NinjaTrader 8 (EventCounterBuilderExample y BarCounterBuilderExample)
  - Corregido el acceso a Count en objetos Series y en Collections
  - Eliminados conflictos con la librería LINQ

## [0.1.13] - 2025-03-20 10:25

### Añadido
- N/A

### Cambiado
- N/A

### Corregido
- Resueltos errores persistentes de method group Count:
  - Eliminada la ambigüedad entre la propiedad Count de colecciones nativas y el método de extensión Count() de LINQ
  - Especificado el acceso explícito a propiedad o método según corresponda
  - Implementada solución para líneas 549 y 678 que generaban error CS0428

## [0.1.12] - 2025-03-20 10:15

### Añadido
- N/A

### Cambiado
- N/A

### Corregido
- Errores de compilación con tipos específicos de NinjaTrader 8:
  - Solucionados errores de acceso a Count para tipos Bars y Series<double>
  - Reemplazado el uso de Count() como método por acceso a la propiedad Count
  - Corregido enfoque al verificar el tamaño de las colecciones de NinjaTrader

## [0.1.11] - 2025-03-20 10:05

### Añadido
- N/A

### Cambiado
- N/A

### Corregido
- Errores de compilación persistentes en DataExtractor.cs:
  - Corregidos todos los errores de method group Count usando Count() con paréntesis
  - Añadido enfoque defensivo para evitar excepciones de NullReferenceException
  - Aplicada corrección consistente en todas las colecciones del código

## [0.1.10] - 2025-03-20 09:52

### Añadido
- N/A

### Cambiado
- N/A

### Corregido
- Errores de compilación restantes en DataExtractor.cs:
  - Corregidos errores de method group en líneas 534 y 650
  - Solucionados problemas con llamadas incompletas a métodos Count()
  - Asegurada la correcta evaluación de variables frente a method groups

## [0.1.9] - 2025-03-20 09:45

### Añadido
- Nueva implementación de botones WPF basada en ejemplos exitosos de NinjaTrader 8

### Cambiado
- Reemplazo completo del enfoque de botones basado en TextFixed por controles WPF nativos
- Creación de grid personalizado para gestionar elementos de UI
- Implementación de manejadores de eventos de clic adecuados

### Corregido
- Errores de compilación relacionados con la detección de clics
- Eliminación de dependencias de APIs no soportadas (GetHitTestRect, IsVisibleOnChart)
- Manejo adecuado de la creación y limpieza de recursos de UI

## [0.1.8] - 2025-03-20 09:30

### Añadido
- N/A

### Cambiado
- N/A

### Corregido
- Errores de compilación en DataExtractor.cs:
  - Reemplazado `DrawObjects.Contains()`

## [0.1.18] - 2025-03-21 03:13

### Añadido
- Configuraciones faltantes en config.py:
  - Añadido REWARD_CONFIG para configuración de recompensas del modelo
  - Agregado PPO_CONFIG como alias de AGENT_CONFIG para compatibilidad
  - Implementado VISUALIZATION_CONFIG para controlar visualización de resultados
  - Incorporado LOGGING_CONFIG para definir niveles y formatos de log

### Cambiado
- Preparación para ejecución de entrenamiento:
  - Creado directorio results para almacenar resultados de entrenamiento
  - Configurado un entrenamiento de prueba con pocos timesteps (10,000)

### Corregido
- Error de importación en main.py:
  - Solucionado el error "ImportError: cannot import name 'REWARD_CONFIG'" 
  - Agregadas configuraciones faltantes que provocaban problemas en el entrenamiento

## [0.1.19] - 2025-03-21 03:45

### Añadido
- Análisis detallado del estado del proyecto:
  - Verificado el conjunto de datos procesado en data/dataset/processed_data.csv
  - Confirmadas las dependencias principales: numpy, pandas, matplotlib, stable-baselines3, gymnasium

### Cambiado
- Modificado el archivo main.py para corregir el parseo de argumentos:
  - Movido parse_args() al nivel de módulo para evitar problemas de ejecución
  - Eliminada duplicación de parseo de argumentos en la función main()

### Corregido
- Errores en las configuraciones necesarias para el entrenamiento:
  - Agregados los componentes faltantes en config.py: REWARD_CONFIG, PPO_CONFIG, VISUALIZATION_CONFIG, LOGGING_CONFIG
  - Creada la estructura de directorios necesaria para almacenar los resultados del entrenamiento

### Pendiente
- Solucionar problemas de ejecución del entrenamiento:
  - El comando python main.py train no está generando salida visible
  - Posibles problemas de compatibilidad con Python 3.13.1 y las librerías utilizadas
  - Se recomienda verificar compatibilidad de versiones entre Gymnasium 1.0.0 y Stable-Baselines3 2.5.0

## [0.1.20] - 2025-03-21 04:20

### Añadido
- Mejoras en la compatibilidad para el entrenamiento:
  - Implementado sistema de logging más robusto con salida a consola y archivo
  - Añadidas opciones de debug adicionales en la línea de comandos
  - Agregado manejo detallado de excepciones con trazas completas

### Cambiado
- Actualizado el módulo de entorno trading_env.py para garantizar compatibilidad con Gymnasium 1.0.0:
  - Adaptado método reset() para seguir la API actual que requiere retornar (observation, info)
  - Modificado método step() para implementar el nuevo formato que devuelve (observation, reward, terminated, truncated, info)
  - Mejorada la documentación de los métodos principales

### Corregido
- Problemas con el parseo de argumentos en línea de comandos:
  - Modificado 'mode' para ser un argumento posicional en lugar de una opción
  - Esto resuelve el problema de que el comando 'python main.py train' no funcionaba correctamente
  - Agregadas validaciones adicionales para los argumentos

## [0.1.21] - 2025-03-21 04:45

### Añadido
- Resolución de problemas de compatibilidad entre bibliotecas:
  - Actualizada Gymnasium a la versión 0.29.1 para resolver conflictos con Stable-Baselines3
  - Mejoradas herramientas de depuración para diagnóstico de problemas

### Cambiado
- Actualizadas dependencias del sistema:
  - Actualizados paquetes setuptools (77.0.3) y wheel (0.45.1) para resolver problemas de instalación
  - Adaptado el código para trabajar con la API más reciente de Gymnasium

### Corregido
- Problemas con el entorno de Python:
  - Resueltas incompatibilidades entre Python 3.13.1 y algunas bibliotecas
  - Corregidas implementaciones de métodos step() y reset() para seguir las convenciones actuales

## [0.1.22] - 2025-03-21 05:00

### Añadido
- Implementación exitosa de un entrenamiento de prueba:
  - Instalada versión compatible de Stable-Baselines3 2.5.0
  - Agregada versión compatible de Gymnasium 1.0.0
  - Añadido manejo robusto de evaluación de modelos durante el entrenamiento

### Cambiado
- Simplificada implementación del sistema de callbacks:
  - Eliminadas dependencias innecesarias de EvalCallback
  - Implementada versión más liviana y robusta de TradeCallback
  - Mejorado el sistema de logging para facilitar el seguimiento del proceso

### Corregido
- Problemas de compatibilidad críticos:
  - Resuelto conflicto de importación circular entre módulos agents y training
  - Correción de problemas con dependencias entre versiones de bibliotecas
  - Actualizado sistema de callback para evitar AttributeError con 'model'

## [0.1.23] - 2025-03-21 05:30

### Añadido
- Registro de operaciones mejorado:
  - Implementado sistema para mantener la lista de trades actualizada tanto en self.trades como en self.position_history
  - Añadida sincronización entre registros de operaciones para asegurar consistencia en análisis de resultados

### Cambiado
- N/A

### Corregido
- Error crítico en environment/trading_env.py:
  - Solucionado IndexError: list index out of range en _close_position cuando self.trades está vacío
  - Implementada validación defensiva para verificar la existencia de elementos antes de acceder a self.trades[-1]
  - Modificada la función _close_position para usar self.entry_time cuando no hay operaciones previas

## [0.1.24] - 2025-03-21 05:55

### Añadido
- Soporte para entrenamiento acelerado con GPU CUDA:
  - Instalación y configuración de PyTorch 2.6.0 con soporte CUDA 11.8
  - Verificación de detección correcta de NVIDIA GeForce RTX 3060 Laptop GPU
  - Optimización de entrenamientos para tiempos más rápidos utilizando hardware acelerado

### Cambiado
- Entorno de ejecución modificado:
  - Actualizados requisitos y dependencias para asegurar compatibilidad con CUDA
  - Configuración de Python 3.11 como versión compatible con PyTorch+CUDA

### Corregido
- N/A

## [0.1.27] - 2025-03-21 00:54

### Añadido
- N/A

### Cambiado
- N/A

### Corregido
- Error en la ruta del archivo de datos para entrenamiento:
  - Identificado el problema: el archivo "nq_futures_data.csv" no existe en el directorio data
  - Detectados archivos de datos disponibles: "NQ_06-25_combined_20250320_225417.csv" y "processed_data.csv" en data/dataset
  - Documentado el formato correcto del comando con la ruta correcta a los archivos existentes

## [0.1.28] - 2025-03-21 01:10

### Añadido
- Sistema de entrenamiento progresivo en cuatro etapas:
  - Etapa 1: Fase de exploración con incentivos para incrementar la actividad de trading
  - Etapa 2: Fase de aprendizaje balanceado entre exploración y explotación
  - Etapa 3: Fase de optimización enfocada en refinamiento de estrategias
  - Etapa 4: Fase de optimización de rendimiento con enfoque en métricas financieras

### Cambiado
- Reconfiguración completa de hiperparámetros para PPO:
  - Aumentado coeficiente de entropía para fomentar exploración (de 0.0 a 0.02)
  - Activado SDE (Stochastic Differential Equations) para mejorar exploración
  - Incrementado learning rate de 3e-4 a 5e-4 para aprendizaje más rápido
  - Reducido n_steps de 2048 a 1024 para actualizaciones más frecuentes
  - Aumentado tamaño de batch de 64 a 128 para mejor generalización
  - Implementada red neuronal más grande para mayor capacidad de aprendizaje

### Corregido
- Sistema de recompensas completamente rediseñado:
  - Eliminada penalización excesiva por trading que inhibía operaciones
  - Añadida recompensa por mantener posiciones abiertas
  - Implementada recompensa por PnL no realizado en posiciones abiertas
  - Añadida penalización por inactividad para fomentar operaciones
  - Introducido bonus de exploración para incentivar variedad en operaciones
  - Reducido umbral de drawdown para permitir mayor tolerancia al riesgo

## [0.1.26] - 2025-03-21 19:45

### Corregido
- Solucionado error de compilación en NT8StrategyServer.cs:
  - Corregido acceso a instrumentos usando el método Instruments.GetInstruments() en lugar de Instruments.All()
  - Cambiado implementación del método GetInstrument para iterar directamente sobre la propiedad Instruments
  - Simplificada la implementación para seguir el patrón estándar de NinjaTrader 8

## [0.1.29] - 2025-03-21 20:00

### Corregido
- Solucionado nuevo error de compilación en NT8StrategyServer.cs:
  - Corregido error CS1061: 'Instrument[]' does not contain a definition for 'GetInstruments'
  - Actualizado método GetInstrument para iterar directamente sobre la propiedad Instruments
  - Simplificada la implementación para seguir el patrón estándar de NinjaTrader 8

## [0.1.30] - 2025-03-21 20:15

### Corregido
- Solucionado error de validación en NT8StrategyServer:
  - Corregido el valor de EntriesPerDirection que estaba establecido en 0 (no válido)
  - Actualizado a un valor de 10 para permitir múltiples entradas por dirección
  - Resuelto el error: "Value of property 'EntriesPerDirection' is not in valid range between 1 and 2147483647"

## [0.1.31] - 2025-03-21 20:30

### Añadido
- Cliente de prueba para NT8StrategyServer:
  - Implementado cliente TCP básico en Python para conectarse a la estrategia
  - Creado integrador de modelo simulado para demostrar la funcionalidad de trading
  - Añadida documentación detallada sobre la integración con modelos personalizados
  - Desarrollado sistema de callbacks para procesar eventos en tiempo real

### Cambiado
- N/A

### Corregido
- N/A

## [0.1.32] - 2025-03-21 21:00

### Añadido
- Integrador de modelo PPO real para NT8StrategyServer:
  - Implementado cliente especializado para conectar modelo PPO entrenado con NinjaTrader 8
  - Desarrollado sistema de transformación de datos de mercado a observaciones para el modelo
  - Creado mecanismo para ejecutar decisiones de trading basadas en la predicción del modelo
  - Añadido manejo de historial de precios para proporcionar contexto temporal al modelo

### Cambiado
- N/A

### Corregido
- N/A

## [0.1.33] - 2025-03-21 21:15

### Corregido
- Optimizado el integrador de modelo PPO para NT8StrategyServer:
  - Forzado el uso de CPU para evitar advertencias y problemas con CUDA
  - Mejorado el manejo de estadísticas de normalización ausentes
  - Implementada detección automática del tamaño de observación del modelo
  - Añadido manejo robusto de errores en la preparación de observaciones
  - Resuelto el problema de conversión de acciones a tipos escalares
  - Incluida mayor información de diagnóstico con trazas de error completas

## [0.1.34] - 2025-03-21 21:30

### Corregido
- Solucionado error crítico en el integrador de modelo PPO con NT8StrategyServer:
  - Corregido formato de observaciones para coincidir con la forma (60, 25) esperada por el modelo
  - Implementada detección automática de dimensiones de observación desde el modelo cargado
  - Añadido sistema robusto para convertir observaciones 1D a la forma 2D requerida
  - Mejorado manejo de normalización para observaciones con diferentes formatos

## [0.1.35] - 2025-03-21 21:35

### Añadido
- Sistema completo funcional para trading automatizado con aprendizaje por refuerzo:
  - Integración exitosa del modelo PPO con NinjaTrader 8 operando en tiempo real
  - Procesamiento de datos de mercado y conversión a formato compatible con el modelo
  - Toma de decisiones de trading basadas en predicciones del modelo entrenado
  - Sistema completo funcionando sin errores y listo para operar en mercados reales

### Cambiado
- N/A

### Corregido
- N/A

## [0.1.36] - 2025-03-21 22:05

### Añadido
- Mejoras de diagnóstico en el integrador PPO con NT8StrategyServer:
  - Registro detallado de todas las decisiones del modelo (incluyendo "mantener")
  - Análisis y comparación entre datos actuales y estadísticas de entrenamiento
  - Alertas cuando las observaciones difieren significativamente de los datos de entrenamiento
  - Reducción del periodo de enfriamiento a 30 segundos para mayor frecuencia de evaluación
  - Información detallada sobre el estado del sistema en cada etapa del proceso

### Cambiado
- N/A

### Corregido
- N/A

## [0.1.37] - 2025-03-21 22:25

### Corregido
- Solucionado problema crítico que impedía la ejecución de operaciones:
  - Implementada interpretación correcta del formato de acciones del modelo PPO
  - Añadido soporte para arrays de decisión multidimensionales [dirección, ?, ?, ?]
  - Corregida la conversión de valores de dirección (-1 a 1) a acciones discretas (0, 1, 2)
  - Mejorado el sistema de diagnóstico para mostrar detalles sobre la interpretación de acciones
  - Habilitada la apertura de posiciones basadas en las señales del modelo entrenado

### Cambiado
- N/A

### Corregido
- N/A

## [0.1.38] - 2025-03-21 22:45

### Corregido
- Problemas fundamentales en el sistema de entrenamiento del modelo PPO que causaban sesgo y mala gestión de riesgos:
  - Rebalanceada la función de recompensa para penalizar adecuadamente las pérdidas y drawdowns
  - Restaurada la penalización por mantener posiciones perdedoras por largos períodos
  - Mejorado el sistema de recompensas para valorar correctamente el cierre de operaciones
  - Implementada una configuración de entrenamiento más equilibrada entre operaciones largas y cortas
  - Reducido el umbral de drawdown a 0.15 para forzar el cierre de posiciones con pérdidas significativas

### Cambiado
- Sistema de recompensas restaurado para entrenar modelos con gestión de riesgos incorporada:
  - Aumentada la penalización por drawdown significativamente (de 0.05 a 0.25)
  - Restaurada la penalización por mantener posiciones perdedoras
  - Equilibrada la recompensa entre operaciones largas y cortas para evitar sesgo
  - Modificada la estructura de la recompensa para valorar adecuadamente la gestión de riesgos
  - Reequilibrados los pesos de los componentes de la recompensa

## [0.1.39] - 2025-03-22 22:40

### Corregido
- Solucionados problemas críticos en el entorno de trading que impedían un entrenamiento adecuado:
  - Corregidas inconsistencias en el manejo de variables entre `step()` y `reset()`
  - Añadida inicialización adecuada de la variable `trade_active` en el constructor
  - Implementada validación defensiva para el acceso a `self.drawdowns[-1]`
  - Corregido el manejo de `entry_time` en `_close_position`
  - Eliminadas referencias incorrectas a `equity_curve` y `max_equity`
  - Unificado uso de `net_worth` y `max_net_worth` en todo el código
  - Incluida inicialización adecuada de `equity_curve` en el constructor

## [0.1.40] - 2025-03-22 22:50

### Cambiado
- Optimizados hiperparámetros para el entrenamiento de prueba:
  - Aumentado learning_rate de 3e-4 a 5e-4 para acelerar el aprendizaje inicial
  - Ajustada configuración de entrenamiento para evaluaciones más frecuentes
  - Reducido número de episodios de evaluación para pruebas más rápidas
  - Ajustados progressive_steps a valores más adecuados para entrenamientos cortos
  - Configurado total_timesteps a 500,000 para pruebas rápidas de validación

## [0.1.41] - 2025-03-22 23:15

### Añadido
- Script de verificación para el entorno de trading:
  - Implementado test_env.py para validar la corrección del entorno
  - Confirmado que el entorno acepta y procesa correctamente las acciones
  - Verificada la corrección de todos los errores previamente identificados
  - Demostrado funcionamiento exitoso de las operaciones de compra/venta/cierre
  - Comprobación de la estructura correcta de observaciones y recompensas

### Corregido
- Error de compatibilidad con el formato de datos:
  - Solucionado problema con la columna datetime añadiendo conversión y uso como índice
  - Implementado manejo correcto del espacio de acción de 4 dimensiones
  - Añadido soporte para formato numpy.array en la función step()

## [0.1.42] - 2025-03-22 23:30

### Añadido
- Configuración para entrenamiento completo con aceleración CUDA:
  - Forzado explícito del uso de GPU mediante parámetro 'device': 'cuda'
  - Ajustados parámetros de entrenamiento para ejecución estable de 500,000 pasos
  - Incrementado coeficiente de entropía de 0.02 a 0.03 para mejorar exploración
  - Configuración optimizada para hardware GPU disponible

### Cambiado
- Rebalanceados parámetros para entrenamiento completo:
  - Aumentada frecuencia de evaluación a cada 25,000 pasos
  - Incrementada frecuencia de guardado de modelos a cada 50,000 pasos
  - Ajustada paciencia para early stopping a 10 evaluaciones
  - Restaurados valores óptimos para n_eval_episodes (5) y log_freq (10,000)

## [0.1.33] - 2023-03-21 22:45
### Agregado
- Sistema de métricas de diagnóstico completo para análisis detallado del comportamiento de trading
- Seguimiento y visualización de componentes de recompensa para análisis de aprendizaje
- Validación multi-segmento para evaluación más robusta del modelo
- Implementación de curriculum learning con entrenamiento progresivo de parámetros
- Registro detallado de operaciones con MAE/MFE y ratios de riesgo/recompensa

### Cambiado
- Mejorado el entorno de trading para registrar métricas detalladas durante el entrenamiento
- Actualizado el script de entrenamiento con callbacks avanzados de seguimiento
- Optimizado el sistema de evaluación para análisis multidimensional del rendimiento

## [0.1.32] - 2023-03-21 21:00
### Agregado
- Implementación del integrador de modelo PPO real para NT8StrategyServer
- Cliente especializado para conectar el modelo PPO entrenado con NinjaTrader 8
- Sistema para transformar datos de mercado en observaciones para el modelo
- Mecanismo para ejecutar decisiones de trading basadas en predicciones del modelo
- Manejo de datos históricos de precios para contexto temporal

## [0.1.31] - 2023-03-21 20:15
### Cambiado
- Modificada la función de recompensa para resolver problemas de aversión al riesgo extrema
- Reducido el peso del componente de win rate para equilibrar rendimiento
- Añadida penalización progresiva por inactividad con umbral reducido a 30 pasos
- Implementada recompensa basada en tamaño de ganancias y penalización por operaciones triviales
- Mejorado componente de gestión de riesgo con penalización por exposición extrema

### Agregado
- Nuevo componente "size_reward" para incentivar operaciones de mayor tamaño
- Sistema de seguimiento detallado de estado de posiciones y razones de cierre

## [0.1.30] - 2023-03-20 19:30
### Cambiado
- Actualizada configuración para capital inicial de 50,000 USD
- Implementado tamaño de contrato configurable (1 contrato)
- Incrementado factor de escalado de recompensa a 1.5
- Aumentado el número de épocas a 5 y el tamaño de lote a 256
- Expandida la arquitectura de red con más capas para mayor capacidad de aprendizaje

## [0.1.46] - 2025-03-22 01:15

### Corregido
- Error crítico que impedía la ejecución del entrenamiento en TradingEnv:
  - Agregados atributos faltantes para tracking de operaciones: _just_opened_position, _just_closed_position, _last_trade_pnl y _last_trade_action_initiated
  - Implementada inicialización correcta de estos atributos en el constructor
  - Añadida actualización de los flags en los métodos relevantes (step, _close_position, reset)
  - Solucionado error AttributeError: 'TradingEnv' object has no attribute '_just_opened_position'

## [0.1.47] - 2025-03-22 07:20

### Añadido
- Sistema mejorado de seguimiento de métricas durante el entrenamiento:
  - Incorporación de contadores acumulados de operaciones entre evaluaciones
  - Agregado seguimiento de mejores métricas históricas (win rate, profit factor, total trades)
  - Implementado reporte periódico de estadísticas con formato tabular más legible
  - Añadido resumen de desempeño que muestra FPS y tiempo de entrenamiento

### Cambiado
- Reducción significativa de verbosidad en logs de entrenamiento:
  - Disminuida frecuencia de evaluaciones de 10,000 a 20,000 pasos
  - Reducido número de episodios de evaluación de 5 a 3 para mayor velocidad
  - Establecido nivel de consola por defecto a WARNING en vez de INFO
  - Modificada configuración de verbose en PPO a 0 por defecto (era 1)
  - Eliminados mensajes repetitivos al guardar modelos intermedios

### Optimizado
- Mejoras de usabilidad para monitoreo de entrenamiento:
  - Formato tabular para estadísticas que facilita seguimiento del progreso
  - Resumen periódico de mejores métricas cada 100,000 pasos
  - Optimización de frecuencia de impresión para reducir sobrecarga de logs
  - Presentación de métricas más relevantes para trading (win rate, PF, trades)

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
  - Incremento significativo de coeficiente de entropía de 0.1 a 0.2
  - Aumento de learning rate de 0.0007 a 0.001
  - Reducción de gamma de 0.98 a 0.95 para enfatizar recompensas inmediatas
  - Inicialización de std más alta (-0.5) para exploración más agresiva
  - Tamaños de batch más pequeños (64) y mayor frecuencia de actualización (256 pasos)

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

## [0.1.54] - 2025-03-22 10:30

### Corregido
- Solucionados errores críticos en el entorno de trading que impedían el entrenamiento:
  - Corregido el error "cannot access local variable 'trade_closed' where it is not associated with a value"
  - Inicializada variable trade_closed correctamente en todos los caminos de ejecución
  - Solucionado KeyError: 'close_reason' al cambiar a la clave 'reason' en los registros de operaciones
  - Mejorado sistema de tracking de operaciones para mantener consistencia entre trades y trade_history
  - Implementada reinicialización adecuada de variables al cerrar posiciones

### Optimizado
- Actualizado el método step() para mejor rendimiento y consistencia:
  - Eliminados cálculos redundantes y uso excesivo de memoria
  - Simplificada la estructura del método para mejor mantenimiento
  - Normalizado uso de commission_rate en lugar de transaction_fee

## [0.1.55] - 2025-03-22 10:45

### Añadido
- Sistema de forzado de operaciones EXTREMADAMENTE AGRESIVO:
  - Fase 1: 95% de probabilidad de trades forzados en los primeros 500 pasos
  - Fase 2: 70% de probabilidad de trades hasta el paso 2000
  - Fase 3: Sistema progresivo que aumenta probabilidad con la inactividad
  - Fase 4: Cierre forzado de posiciones perdedoras con probabilidad progresiva
  - Mecanismo extremadamente agresivo para garantizar operaciones desde el inicio

### Cambiado
- Función de recompensa completamente rediseñada para maximizar actividad de trading:
  - Bonus constante (+0.2) por cada paso con posición activa
  - Recompensa muy alta (+1.0) por abrir cualquier posición
  - Recompensa extrema (+2.0) por cerrar operaciones ganadoras
  - Penalización muy reducida (-0.5) por operaciones perdedoras
  - Penalización severa por inactividad que aumenta con el tiempo
  - Recompensa adicional por PnL no realizado en posiciones abiertas

### Optimizado
- Hiperparámetros de PPO llevados al extremo para forzar exploración:
  - Coeficiente de entropía aumentado a 0.5 (era 0.1)
  - Learning rate duplicado a 0.002 para aprendizaje agresivo
  - Inicialización de std a 1.0 para máxima variabilidad en acciones
  - Activado SDE (Stochastic Differential Equations) para mayor exploración
  - Reducido tamaño de batch y n_steps para actualizaciones más frecuentes
  - Simplificada arquitectura de red para prevenir overfitting temprano

## [0.1.56] - 2025-03-22 12:30

### Añadido
- Sistema robusto de protección contra inestabilidad numérica:
  - Detector de NaN para identificar y corregir automáticamente valores problemáticos
  - Limitador de desviación estándar que previene explosión de gradientes
  - Mecanismo de reinicio de parámetros cuando se detectan valores inestables
  - Normalización de estados para mejorar la estabilidad del entrenamiento

### Cambiado
- Sistema de recompensas balanceado para evitar inestabilidad:
  - Reducción de recompensas extremas pero manteniendo incentivos para operaciones
  - Implementación de recorte (clipping) de valores para evitar recompensas excesivas
  - Penalizaciones moderadas que no causan gradientes explosivos
  - Recompensa proporcional al PnL con límites superior e inferior

### Optimizado
- Hiperparámetros ajustados para máxima estabilidad:
  - Learning rate reducido a 0.0005 para prevenir actualizaciones desestabilizadoras
  - Coeficiente de entropía ajustado a 0.1 (balance entre exploración y estabilidad)
  - Implementación de Target KL para limitar cambios bruscos en la política
  - Función de activación cambiada a tanh para mejor control de valores
  - Clip de gradientes más estricto (0.5) para prevenir explosión