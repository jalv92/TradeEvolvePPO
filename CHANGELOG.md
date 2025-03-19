# Changelog

## [0.1.2] - 2025-03-18 21:55

### Añadido
- Dependencia PyYAML para el manejo de archivos YAML

### Cambiado
- Actualizado requirements.txt para incluir PyYAML

### Corregido
- Error de importación en utils/helpers.py al resolver el módulo yaml

## [0.1.1] - 2025-03-18 21:45

### Añadido
- Funcionalidad para recibir indicadores directamente de NinjaTrader 8
- Configuración detallada en config.py para todos los módulos del sistema
- Implementación del DataLoader para procesar datos con indicadores externos

### Cambiado
- Eliminado el archivo indicators.py ya que los indicadores se obtendrán directamente de NinjaTrader 8
- Modificado data_loader.py para verificar la presencia de indicadores en lugar de calcularlos
- Actualizado data/__init__.py para reflejar la nueva estructura

### Corregido
- N/A

## [0.1.0] - 2025-03-18 21:35

### Añadido
- Estructura inicial del proyecto
- Configuración básica de carpetas y archivos
- Creación de todos los archivos necesarios según la estructura definida en el README
- Configuración de requirements.txt con las dependencias iniciales

### Cambiado
- N/A

### Corregido
- N/A
