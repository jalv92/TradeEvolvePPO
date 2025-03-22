# Conector para NinjaTrader 8

Este paquete proporciona un sistema de comunicación entre Python y NinjaTrader 8, permitiendo ejecutar operaciones de trading utilizando modelos de aprendizaje automático.

## Características

- Comunicación TCP bidireccional entre Python y NinjaTrader 8
- Arquitectura cliente-servidor:
  - **Servidor**: Estrategia NT8StrategyServer ejecutada en NinjaTrader 8
  - **Cliente**: Clase NT8Client en Python que se conecta al servidor
- Envío y recepción de datos de mercado en tiempo real
- Ejecución de órdenes de mercado, límite y stop
- Gestión de posiciones y cancelación de órdenes
- Actualizaciones en tiempo real de posiciones y órdenes

## Requisitos

1. NinjaTrader 8 instalado en Windows
2. Python 3.7 o superior
3. Pandas y NumPy instalados

## Instalación

El sistema se incluye como parte del proyecto TradeEvolvePPO. No es necesario realizar una instalación adicional.

## Configuración

### En NinjaTrader 8:

1. Copiar el archivo `NT8StrategyServer.cs` a la carpeta de estrategias de NinjaTrader
2. En NinjaTrader 8, ir a Herramientas (Tools) > Importar (Import) > NinjaScript Add-On
3. Seleccionar el archivo `NT8StrategyServer.cs`
4. Una vez importado, añadir la estrategia a un gráfico del instrumento que deseas operar:
   - Hacer clic derecho en el gráfico > Strategies > Add Strategy...
   - Seleccionar "NT8StrategyServer" de la lista
   - Configurar los parámetros (puerto, actualizaciones de datos, etc.)
   - Aceptar y verificar que la estrategia se está ejecutando

### Parámetros de la estrategia NT8StrategyServer:

- **Puerto TCP**: Puerto en el que escuchará el servidor (por defecto 5555)
- **Enviar datos de barras**: Activa/desactiva el envío de datos OHLCV
- **Enviar actualizaciones de órdenes**: Notificar sobre cambios en órdenes
- **Enviar actualizaciones de posiciones**: Notificar sobre cambios en posiciones
- **Permitir órdenes remotas**: Permitir que Python envíe órdenes
- **Intervalo de actualización**: Cada cuántas barras se envían datos

## Uso básico

### Cliente Python

```python
from connector_nt8.ninjatrader_client import NT8Client

# Crear una instancia del cliente
client = NT8Client(host="localhost", port=5555)

# Conectar con el servidor (estrategia en NinjaTrader)
client.connect()

# Obtener datos históricos
bar_history = client.get_bar_history(instrument="NQ 06-23")

# Enviar una orden de compra a mercado
client.market_buy(instrument="NQ 06-23", quantity=1)

# Enviar una orden de venta a mercado
client.market_sell(instrument="NQ 06-23", quantity=1)

# Cerrar posiciones
client.close_position(instrument="NQ 06-23")

# Recibir actualizaciones (usando callbacks)
def on_bar_data(bar_data):
    print(f"Nueva barra: {bar_data['timestamp']} - Cierre: {bar_data['close']}")

def on_position_update(position):
    print(f"Posición: {position['market_position']} - Cantidad: {position['quantity']}")

client.set_on_bar_data(on_bar_data)
client.set_on_position_update(on_position_update)

# Desconectar
client.disconnect()
```

### Live Trader con Modelo PPO

Para ejecutar operaciones en tiempo real utilizando un modelo entrenado:

```bash
python -m connector_nt8.nt8_trader --model ./results/models/final_model.zip --instrument "NQ 06-23" --host localhost --port 5555 --quantity 1 --interval 60
```

Parámetros:
- `--model`: Ruta al modelo entrenado (.zip)
- `--instrument`: Instrumento a operar (formato NinjaTrader)
- `--host`: Dirección IP donde se ejecuta NinjaTrader (por defecto "localhost")
- `--port`: Puerto TCP de la estrategia NT8StrategyServer (por defecto 5555)
- `--quantity`: Cantidad de contratos a operar (por defecto 1)
- `--interval`: Intervalo de actualización en segundos (por defecto 60)
- `--data`: Ruta a datos históricos para inicializar (opcional)
- `--min-bars`: Mínimo de barras requeridas antes de empezar a operar (por defecto 20)

## Protocolo de comunicación

El protocolo de comunicación entre Python y NinjaTrader 8 utiliza mensajes de texto con formato delimitado por puntos y coma (`;`). Cada mensaje termina con un salto de línea (`\n`).

### Mensajes del Servidor (NinjaTrader) a Cliente (Python)

- **Datos de barras**: `BARDATA;instrumento;timestamp;open;high;low;close;volume`
- **Datos de órdenes**: `ORDER;id;instrumento;acción;tipo;cantidad;limitPrice;stopPrice;estado`
- **Datos de posiciones**: `POSITION;instrumento;marketPosition;cantidad;precioPromedio;pnlMoneda;pnlPorciento`
- **Respuesta a comandos**: `ORDERPLACED;...`, `POSITIONCLOSED;...`, `ORDERSCANCELED;...`, `PONG`
- **Errores**: `ERROR;mensaje`

### Comandos del Cliente (Python) a Servidor (NinjaTrader)

- **Solicitar datos**: `GETDATA`, `GETORDERS`, `GETPOSITIONS`
- **Verificar conexión**: `PING`
- **Órdenes**: 
  - `BUY;instrumento;cantidad;tipo;precio`
  - `SELL;instrumento;cantidad;tipo;precio`
  - `CLOSEPOSITION;instrumento`
  - `CANCELORDERS;instrumento`

## Consideraciones importantes

- **Siempre prueba en cuenta simulada**: Nunca utilices una cuenta real sin haber probado exhaustivamente en simulación.
- **Fallos de conexión**: El sistema intenta manejar las desconexiones, pero es posible que sea necesario reiniciar NinjaTrader o el script en caso de problemas persistentes.
- **Limitaciones**: Las operaciones están limitadas por la velocidad de la comunicación TCP, que puede tener cierta latencia. Para estrategias de muy alta frecuencia, considera otros enfoques.
- **Seguridad**: La comunicación no está cifrada, por lo que se recomienda utilizar este sistema solo en un entorno local o en una red privada segura.
- **Compatibilidad**: La estrategia NT8StrategyServer está diseñada para NinjaTrader 8. No es compatible con versiones anteriores.

## Solución de problemas

Si encuentras problemas:

1. Verifica que la estrategia NT8StrategyServer esté correctamente cargada en un gráfico en NinjaTrader 8
2. Comprueba que el puerto configurado en la estrategia coincide con el utilizado en el cliente Python
3. Verifica en la ventana de salida de NinjaTrader si hay errores relacionados con la estrategia
4. Si hay problemas de conexión, reinicia NinjaTrader y el script Python
5. Asegúrate de que no hay firewalls bloqueando la comunicación TCP en el puerto configurado 