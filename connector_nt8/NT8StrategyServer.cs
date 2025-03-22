#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Net;
using System.Net.Sockets;
using System.IO;
using System.Windows;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;
using System.Text.RegularExpressions;
#endregion

//This namespace holds Strategies in this folder and is required. Do not change it. 
namespace NinjaTrader.NinjaScript.Strategies
{
    public class NT8StrategyServer : Strategy
    {
        private TcpListener server;
        private bool serverRunning;
        private Thread serverThread;
        private Dictionary<string, TcpClient> clients;
        private Dictionary<string, NetworkStream> clientStreams;
        private Dictionary<string, Thread> clientThreads;
        private bool isDisposed;
        private readonly object lockObject = new object();
        private List<Order> currentOrders;
        private List<Position> currentPositions;
        private string lastBarData;
        private int dataUpdateCount;
        
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description                = @"Estrategia servidor para integración con TradeEvolvePPO";
                Name                       = "NT8StrategyServer";
                Calculate                  = Calculate.OnBarClose;
                EntriesPerDirection        = 10;
                EntryHandling              = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds  = 30;
                IsFillLimitOnTouch         = false;
                MaximumBarsLookBack        = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution        = OrderFillResolution.Standard;
                Slippage                   = 0;
                StartBehavior              = StartBehavior.WaitUntilFlat;
                TimeInForce                = TimeInForce.Gtc;
                TraceOrders                = false;
                RealtimeErrorHandling      = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling         = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade        = 20;
                IsInstantiatedOnEachOptimizationIteration = true;
                
                // Parámetros configurables
                Port                       = 5555;
                SendBarData                = true;
                SendMarketDepth            = false;
                SendOrderUpdates           = true;
                SendPositionUpdates        = true;
                AllowRemoteOrders          = true;
                DataUpdateInterval         = 1; // Cada cuántas barras se envían datos (1 = cada barra)
            }
            else if (State == State.Configure)
            {
                // Inicializar variables
                clients = new Dictionary<string, TcpClient>();
                clientStreams = new Dictionary<string, NetworkStream>();
                clientThreads = new Dictionary<string, Thread>();
                serverRunning = false;
                isDisposed = false;
                currentOrders = new List<Order>();
                currentPositions = new List<Position>();
                dataUpdateCount = 0;
                
                // Registrar manejadores de eventos
                if (SendOrderUpdates)
                {
                    Account.OrderUpdate += Account_OrderUpdate;
                }
                
                if (SendPositionUpdates)
                {
                    Account.PositionUpdate += Account_PositionUpdate;
                }
            }
            else if (State == State.DataLoaded)
            {
                // Iniciar el servidor TCP
                StartServer();
            }
            else if (State == State.Terminated)
            {
                // Detener el servidor y limpiar recursos
                StopServer();
                
                // Deregistrar eventos
                if (SendOrderUpdates)
                {
                    Account.OrderUpdate -= Account_OrderUpdate;
                }
                
                if (SendPositionUpdates)
                {
                    Account.PositionUpdate -= Account_PositionUpdate;
                }
                
                isDisposed = true;
            }
        }
        
        protected override void OnBarUpdate()
        {
            // Solo procesar en tiempo real
            if (State < State.Realtime)
                return;
            
            if (CurrentBar < BarsRequiredToTrade)
                return;
            
            // Verificar si debemos enviar datos en esta barra
            dataUpdateCount++;
            if (SendBarData && dataUpdateCount >= DataUpdateInterval)
            {
                dataUpdateCount = 0;
                SendBarDataToClients();
            }
        }
        
        #region Servidor TCP
        
        private void StartServer()
        {
            if (serverRunning)
                return;
            
            try
            {
                // Iniciar el servidor en un hilo separado
                serverThread = new Thread(new ThreadStart(RunServer));
                serverThread.IsBackground = true;
                serverThread.Start();
                
                Print("NT8StrategyServer: Servidor iniciado en puerto " + Port);
            }
            catch (Exception ex)
            {
                Print("NT8StrategyServer Error al iniciar servidor: " + ex.Message);
            }
        }
        
        private void StopServer()
        {
            if (!serverRunning)
                return;
            
            serverRunning = false;
            
            try
            {
                // Cerrar todos los clientes
                lock (lockObject)
                {
                    foreach (var clientId in clients.Keys.ToList())
                    {
                        CloseClient(clientId);
                    }
                    
                    clients.Clear();
                    clientStreams.Clear();
                    clientThreads.Clear();
                }
                
                // Cerrar el servidor
                if (server != null)
                {
                    server.Stop();
                    server = null;
                }
                
                // Esperar a que el hilo del servidor termine
                if (serverThread != null && serverThread.IsAlive)
                {
                    serverThread.Join(1000);
                    if (serverThread.IsAlive)
                    {
                        try { serverThread.Abort(); } catch { }
                    }
                    serverThread = null;
                }
                
                Print("NT8StrategyServer: Servidor detenido");
            }
            catch (Exception ex)
            {
                Print("NT8StrategyServer Error al detener servidor: " + ex.Message);
            }
        }
        
        private void RunServer()
        {
            try
            {
                server = new TcpListener(IPAddress.Any, Port);
                server.Start();
                serverRunning = true;
                
                Print("NT8StrategyServer: Escuchando en puerto " + Port);
                
                while (serverRunning && !isDisposed)
                {
                    try
                    {
                        // Esperar por conexiones de clientes
                        if (server.Pending())
                        {
                            TcpClient client = server.AcceptTcpClient();
                            string clientId = Guid.NewGuid().ToString();
                            
                            // Guardar el cliente
                            lock (lockObject)
                            {
                                clients[clientId] = client;
                                clientStreams[clientId] = client.GetStream();
                                
                                // Crear un hilo para manejar la comunicación con este cliente
                                Thread clientThread = new Thread(new ParameterizedThreadStart(HandleClient));
                                clientThread.IsBackground = true;
                                clientThread.Start(clientId);
                                clientThreads[clientId] = clientThread;
                            }
                            
                            Print("NT8StrategyServer: Cliente conectado: " + clientId);
                            
                            // Enviar datos iniciales al cliente
                            if (SendBarData)
                            {
                                SendBarDataToClient(clientId);
                            }
                            
                            if (SendOrderUpdates)
                            {
                                SendOrdersToClient(clientId);
                            }
                            
                            if (SendPositionUpdates)
                            {
                                SendPositionsToClient(clientId);
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Print("NT8StrategyServer Error en bucle de aceptación: " + ex.Message);
                    }
                    
                    Thread.Sleep(100);
                }
            }
            catch (Exception ex)
            {
                Print("NT8StrategyServer Error en hilo del servidor: " + ex.Message);
            }
            finally
            {
                serverRunning = false;
                
                if (server != null)
                {
                    server.Stop();
                    server = null;
                }
            }
        }
        
        private void HandleClient(object clientIdObj)
        {
            string clientId = (string)clientIdObj;
            TcpClient client = null;
            NetworkStream stream = null;
            
            lock (lockObject)
            {
                if (!clients.TryGetValue(clientId, out client) || client == null)
                    return;
                
                if (!clientStreams.TryGetValue(clientId, out stream) || stream == null)
                    return;
            }
            
            byte[] buffer = new byte[4096];
            bool clientRunning = true;
            
            try
            {
                while (clientRunning && serverRunning && !isDisposed && client.Connected)
                {
                    try
                    {
                        // Verificar si hay datos disponibles
                        if (client.Available > 0)
                        {
                            int bytesRead = stream.Read(buffer, 0, buffer.Length);
                            if (bytesRead > 0)
                            {
                                string message = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                                
                                // Procesar los comandos del cliente
                                ProcessClientCommand(clientId, message);
                            }
                            else
                            {
                                // El cliente cerró la conexión
                                clientRunning = false;
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Print("NT8StrategyServer Error en comunicación con cliente " + clientId + ": " + ex.Message);
                        clientRunning = false;
                    }
                    
                    Thread.Sleep(100);
                }
            }
            catch (Exception ex)
            {
                Print("NT8StrategyServer Error en hilo de cliente " + clientId + ": " + ex.Message);
            }
            finally
            {
                // Cerrar la conexión con este cliente
                CloseClient(clientId);
            }
        }
        
        private void CloseClient(string clientId)
        {
            lock (lockObject)
            {
                try
                {
                    // Cerrar y liberar recursos del cliente
                    if (clientStreams.ContainsKey(clientId))
                    {
                        NetworkStream stream = clientStreams[clientId];
                        if (stream != null)
                        {
                            stream.Close();
                        }
                        clientStreams.Remove(clientId);
                    }
                    
                    if (clients.ContainsKey(clientId))
                    {
                        TcpClient client = clients[clientId];
                        if (client != null)
                        {
                            client.Close();
                        }
                        clients.Remove(clientId);
                    }
                    
                    if (clientThreads.ContainsKey(clientId))
                    {
                        Thread thread = clientThreads[clientId];
                        if (thread != null && thread.IsAlive)
                        {
                            try { thread.Abort(); } catch { }
                        }
                        clientThreads.Remove(clientId);
                    }
                    
                    Print("NT8StrategyServer: Cliente desconectado: " + clientId);
                }
                catch (Exception ex)
                {
                    Print("NT8StrategyServer Error al cerrar cliente " + clientId + ": " + ex.Message);
                }
            }
        }
        
        private void SendToClient(string clientId, string message)
        {
            NetworkStream stream = null;
            
            lock (lockObject)
            {
                if (!clientStreams.TryGetValue(clientId, out stream) || stream == null)
                    return;
                
                if (!clients.TryGetValue(clientId, out TcpClient client) || client == null || !client.Connected)
                    return;
            }
            
            try
            {
                byte[] data = Encoding.UTF8.GetBytes(message + "\n");
                stream.Write(data, 0, data.Length);
                stream.Flush();
            }
            catch (Exception ex)
            {
                Print("NT8StrategyServer Error al enviar datos a cliente " + clientId + ": " + ex.Message);
                CloseClient(clientId);
            }
        }
        
        private void BroadcastToClients(string message)
        {
            lock (lockObject)
            {
                foreach (string clientId in clients.Keys.ToList())
                {
                    SendToClient(clientId, message);
                }
            }
        }
        
        #endregion
        
        #region Procesamiento de comandos
        
        private void ProcessClientCommand(string clientId, string command)
        {
            try
            {
                // Dividir el comando por líneas (puede haber varios comandos)
                string[] commands = command.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
                
                foreach (string cmd in commands)
                {
                    string trimmedCmd = cmd.Trim();
                    
                    if (string.IsNullOrEmpty(trimmedCmd))
                        continue;
                    
                    // Parsear el comando
                    // Formato esperado: ACCIÓN;PARAM1;PARAM2;...
                    string[] parts = trimmedCmd.Split(';');
                    
                    if (parts.Length < 1)
                        continue;
                    
                    string action = parts[0].ToUpper();
                    
                    switch (action)
                    {
                        case "PING":
                            // Comando simple para verificar conectividad
                            SendToClient(clientId, "PONG");
                            break;
                            
                        case "GETDATA":
                            // Enviar datos de precio actuales
                            SendBarDataToClient(clientId);
                            break;
                            
                        case "GETORDERS":
                            // Enviar órdenes actuales
                            SendOrdersToClient(clientId);
                            break;
                            
                        case "GETPOSITIONS":
                            // Enviar posiciones actuales
                            SendPositionsToClient(clientId);
                            break;
                            
                        case "BUY":
                            // Colocar orden de compra
                            if (AllowRemoteOrders && parts.Length >= 3)
                            {
                                string instrument = parts[1];
                                int quantity = int.Parse(parts[2]);
                                
                                // Verificar parámetros adicionales
                                double price = 0;
                                string orderType = "MARKET";
                                
                                if (parts.Length >= 4)
                                    orderType = parts[3].ToUpper();
                                
                                if (parts.Length >= 5 && !string.IsNullOrEmpty(parts[4]))
                                    price = double.Parse(parts[4]);
                                
                                PlaceOrder(clientId, true, instrument, quantity, orderType, price);
                            }
                            break;
                            
                        case "SELL":
                            // Colocar orden de venta
                            if (AllowRemoteOrders && parts.Length >= 3)
                            {
                                string instrument = parts[1];
                                int quantity = int.Parse(parts[2]);
                                
                                // Verificar parámetros adicionales
                                double price = 0;
                                string orderType = "MARKET";
                                
                                if (parts.Length >= 4)
                                    orderType = parts[3].ToUpper();
                                
                                if (parts.Length >= 5 && !string.IsNullOrEmpty(parts[4]))
                                    price = double.Parse(parts[4]);
                                
                                PlaceOrder(clientId, false, instrument, quantity, orderType, price);
                            }
                            break;
                            
                        case "CLOSEPOSITION":
                            // Cerrar posición
                            if (AllowRemoteOrders && parts.Length >= 2)
                            {
                                string instrument = parts[1];
                                ClosePosition(clientId, instrument);
                            }
                            break;
                            
                        case "CANCELORDERS":
                            // Cancelar órdenes
                            if (AllowRemoteOrders && parts.Length >= 2)
                            {
                                string instrument = parts[1];
                                CancelOrders(clientId, instrument);
                            }
                            break;
                            
                        default:
                            // Comando desconocido
                            SendToClient(clientId, "ERROR;Unknown command: " + action);
                            break;
                    }
                }
            }
            catch (Exception ex)
            {
                Print("NT8StrategyServer Error al procesar comando de cliente " + clientId + ": " + ex.Message);
                SendToClient(clientId, "ERROR;" + ex.Message);
            }
        }
        
        #endregion
        
        #region Operaciones de trading
        
        private void PlaceOrder(string clientId, bool isBuy, string instrument, int quantity, string orderType, double price)
        {
            if (!AllowRemoteOrders)
            {
                SendToClient(clientId, "ERROR;Remote orders not allowed");
                return;
            }
            
            try
            {
                // Encontrar el instrumento
                Instrument instr = GetInstrument(instrument);
                
                if (instr == null)
                {
                    SendToClient(clientId, "ERROR;Instrument not found: " + instrument);
                    return;
                }
                
                // Determinar el tipo de orden
                OrderType type = OrderType.Market;
                
                switch (orderType)
                {
                    case "MARKET":
                        type = OrderType.Market;
                        break;
                    case "LIMIT":
                        type = OrderType.Limit;
                        break;
                    case "STOP":
                        type = OrderType.StopMarket;
                        break;
                    default:
                        SendToClient(clientId, "ERROR;Invalid order type: " + orderType);
                        return;
                }
                
                // Colocar la orden
                if (isBuy)
                {
                    if (type == OrderType.Market)
                    {
                        EnterLong(quantity, instr.FullName);
                    }
                    else if (type == OrderType.Limit)
                    {
                        EnterLongLimit(quantity, price, instr.FullName);
                    }
                    else if (type == OrderType.StopMarket)
                    {
                        EnterLongStopMarket(quantity, price, instr.FullName);
                    }
                }
                else
                {
                    if (type == OrderType.Market)
                    {
                        EnterShort(quantity, instr.FullName);
                    }
                    else if (type == OrderType.Limit)
                    {
                        EnterShortLimit(quantity, price, instr.FullName);
                    }
                    else if (type == OrderType.StopMarket)
                    {
                        EnterShortStopMarket(quantity, price, instr.FullName);
                    }
                }
                
                string action = isBuy ? "BUY" : "SELL";
                SendToClient(clientId, "ORDERPLACED;" + action + ";" + instrument + ";" + quantity + ";" + orderType + ";" + price);
            }
            catch (Exception ex)
            {
                SendToClient(clientId, "ERROR;Failed to place order: " + ex.Message);
                Print("NT8StrategyServer Error al colocar orden: " + ex.Message);
            }
        }
        
        private void ClosePosition(string clientId, string instrument)
        {
            if (!AllowRemoteOrders)
            {
                SendToClient(clientId, "ERROR;Remote orders not allowed");
                return;
            }
            
            try
            {
                // Encontrar el instrumento
                Instrument instr = GetInstrument(instrument);
                
                if (instr == null)
                {
                    SendToClient(clientId, "ERROR;Instrument not found: " + instrument);
                    return;
                }
                
                // Cerrar todas las posiciones para este instrumento
                ExitLong(instr.FullName);
                ExitShort(instr.FullName);
                
                SendToClient(clientId, "POSITIONCLOSED;" + instrument);
            }
            catch (Exception ex)
            {
                SendToClient(clientId, "ERROR;Failed to close position: " + ex.Message);
                Print("NT8StrategyServer Error al cerrar posición: " + ex.Message);
            }
        }
        
        private void CancelOrders(string clientId, string instrument)
        {
            if (!AllowRemoteOrders)
            {
                SendToClient(clientId, "ERROR;Remote orders not allowed");
                return;
            }
            
            try
            {
                // Encontrar el instrumento
                Instrument instr = GetInstrument(instrument);
                
                if (instr == null)
                {
                    SendToClient(clientId, "ERROR;Instrument not found: " + instrument);
                    return;
                }
                
                // Cancelar todas las órdenes para este instrumento
                CancelOrdersForInstrument(instr.FullName);
                
                SendToClient(clientId, "ORDERSCANCELED;" + instrument);
            }
            catch (Exception ex)
            {
                SendToClient(clientId, "ERROR;Failed to cancel orders: " + ex.Message);
                Print("NT8StrategyServer Error al cancelar órdenes: " + ex.Message);
            }
        }
        
        private void CancelOrdersForInstrument(string instrumentName)
        {
            // Obtener todas las órdenes activas
            // En NinjaTrader 8, Account.Orders es una propiedad, no un método
            List<Order> ordersToCancel = new List<Order>();
            
            // Iterar sobre Account.Orders para encontrar órdenes para el instrumento dado
            foreach (Order order in Account.Orders)
            {
                if (order.Instrument.FullName == instrumentName && 
                    (order.OrderState == OrderState.Working || order.OrderState == OrderState.Accepted))
                {
                    ordersToCancel.Add(order);
                }
            }
            
            // Cancelar las órdenes
            foreach (Order order in ordersToCancel)
            {
                // En NinjaTrader 8, usamos el método heredado CancelOrder para cancelar
                // órdenes individuales en lugar de Account.Cancel
                CancelOrder(order);
            }
        }
        
        private Instrument GetInstrument(string instrumentName)
        {
            // Si estamos operando el mismo instrumento, usar el actual
            if (Instrument != null && Instrument.FullName == instrumentName)
                return Instrument;
            
            // De lo contrario, buscar el instrumento por nombre
            try
            {
                // En NinjaTrader 8, Instruments es una propiedad que devuelve un array de Instrument
                foreach (Instrument instr in Instruments)
                {
                    if (instr.FullName == instrumentName)
                        return instr;
                }
                
                return null;
            }
            catch
            {
                return null;
            }
        }
        
        #endregion
        
        #region Envío de datos y eventos
        
        private void SendBarDataToClients()
        {
            if (CurrentBar < 1)
                return;
            
            // Crear mensaje con datos de la barra actual
            string data = CreateBarDataMessage(0);
            
            lastBarData = data;
            
            // Enviar a todos los clientes
            BroadcastToClients(data);
        }
        
        private void SendBarDataToClient(string clientId)
        {
            if (CurrentBar < 1)
                return;
            
            // Crear mensaje con datos para este cliente
            string data = CreateBarDataMessage(0);
            
            // Enviar al cliente
            SendToClient(clientId, data);
        }
        
        private string CreateBarDataMessage(int barsAgo)
        {
            if (CurrentBar < barsAgo)
                return null;
            
            // Formatear timestamp
            string timestamp = Time[barsAgo].ToString("yyyy-MM-dd HH:mm:ss");
            
            // Datos OHLCV
            string data = string.Format(
                "BARDATA;{0};{1};{2};{3};{4};{5};{6}",
                Instrument.FullName,
                timestamp,
                Open[barsAgo],
                High[barsAgo],
                Low[barsAgo],
                Close[barsAgo],
                Volume[barsAgo]
            );
            
            return data;
        }
        
        private void SendOrdersToClient(string clientId)
        {
            try
            {
                lock (lockObject)
                {
                    // Enviar cada orden como un mensaje separado
                    foreach (Order order in currentOrders)
                    {
                        string orderData = FormatOrderData(order);
                        SendToClient(clientId, orderData);
                    }
                }
            }
            catch (Exception ex)
            {
                Print("NT8StrategyServer Error al enviar órdenes a cliente " + clientId + ": " + ex.Message);
            }
        }
        
        private void SendPositionsToClient(string clientId)
        {
            try
            {
                lock (lockObject)
                {
                    // Enviar cada posición como un mensaje separado
                    foreach (Position position in currentPositions)
                    {
                        string positionData = FormatPositionData(position);
                        SendToClient(clientId, positionData);
                    }
                }
            }
            catch (Exception ex)
            {
                Print("NT8StrategyServer Error al enviar posiciones a cliente " + clientId + ": " + ex.Message);
            }
        }
        
        private string FormatOrderData(Order order)
        {
            return string.Format(
                "ORDER;{0};{1};{2};{3};{4};{5};{6};{7}",
                order.Id,
                order.Instrument.FullName,
                order.OrderAction.ToString(),
                order.OrderType.ToString(),
                order.Quantity,
                order.LimitPrice,
                order.StopPrice,
                order.OrderState.ToString()
            );
        }
        
        private string FormatPositionData(Position position)
        {
            // Si position.MarketPosition es Flat, significa que no hay posición
            if (position.MarketPosition == MarketPosition.Flat)
                return string.Format("POSITION;{0};FLAT;0;0;0;0", position.Instrument.FullName);
            
            // Calcular PnL para la posición
            double pnlCurrency = 0;
            double pnlPercent = 0;
            
            try {
                pnlCurrency = position.Quantity * (position.MarketPosition == MarketPosition.Long ? 
                    Close[0] - position.AveragePrice : 
                    position.AveragePrice - Close[0]) * position.Instrument.MasterInstrument.PointValue;
                
                if (position.AveragePrice != 0)
                    pnlPercent = (pnlCurrency / (position.AveragePrice * position.Quantity * position.Instrument.MasterInstrument.PointValue)) * 100;
            }
            catch (Exception ex) {
                Print("Error al calcular PnL: " + ex.Message);
            }
            
            return string.Format(
                "POSITION;{0};{1};{2};{3};{4};{5}",
                position.Instrument.FullName,
                position.MarketPosition.ToString(),
                position.Quantity,
                position.AveragePrice,
                pnlCurrency,
                pnlPercent
            );
        }
        
        #endregion
        
        #region Manejadores de eventos
        
        private void Account_OrderUpdate(object sender, OrderEventArgs e)
        {
            try
            {
                lock (lockObject)
                {
                    // Actualizar la lista de órdenes
                    // Primero eliminar la orden si existe
                    currentOrders.RemoveAll(o => o.Id == e.Order.Id);
                    
                    // Añadir la orden actualizada si no está completada o cancelada
                    if (e.Order.OrderState != OrderState.Filled && e.Order.OrderState != OrderState.Cancelled)
                    {
                        currentOrders.Add(e.Order);
                    }
                    
                    // Enviar la actualización a todos los clientes
                    string orderData = FormatOrderData(e.Order);
                    BroadcastToClients(orderData);
                }
            }
            catch (Exception ex)
            {
                Print("NT8StrategyServer Error en actualización de orden: " + ex.Message);
            }
        }
        
        private void Account_PositionUpdate(object sender, PositionEventArgs e)
        {
            try
            {
                lock (lockObject)
                {
                    // Actualizar la lista de posiciones
                    // Primero eliminar la posición si existe
                    currentPositions.RemoveAll(p => p.Instrument.FullName == e.Position.Instrument.FullName);
                    
                    // Añadir la posición actualizada
                    currentPositions.Add(e.Position);
                    
                    // Enviar la actualización a todos los clientes
                    string positionData = FormatPositionData(e.Position);
                    BroadcastToClients(positionData);
                }
            }
            catch (Exception ex)
            {
                Print("NT8StrategyServer Error en actualización de posición: " + ex.Message);
            }
        }
        
        #endregion
        
        #region Properties
        
        [NinjaScriptProperty]
        [Range(1000, 65535)]
        [Display(Name = "Puerto TCP", Description = "Puerto para el servidor TCP", Order = 1, GroupName = "Configuración de conexión")]
        public int Port { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Enviar datos de barras", Description = "Enviar datos OHLCV a los clientes", Order = 2, GroupName = "Configuración de datos")]
        public bool SendBarData { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Enviar profundidad de mercado", Description = "Enviar datos de book/profundidad a los clientes", Order = 3, GroupName = "Configuración de datos")]
        public bool SendMarketDepth { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Enviar actualizaciones de órdenes", Description = "Notificar a los clientes sobre cambios en órdenes", Order = 4, GroupName = "Configuración de datos")]
        public bool SendOrderUpdates { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Enviar actualizaciones de posiciones", Description = "Notificar a los clientes sobre cambios en posiciones", Order = 5, GroupName = "Configuración de datos")]
        public bool SendPositionUpdates { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Permitir órdenes remotas", Description = "Permitir que los clientes envíen órdenes", Order = 6, GroupName = "Configuración de seguridad")]
        public bool AllowRemoteOrders { get; set; }
        
        [NinjaScriptProperty]
        [Range(1, 100)]
        [Display(Name = "Intervalo de actualización", Description = "Cada cuántas barras se envían datos (1 = cada barra)", Order = 7, GroupName = "Configuración de datos")]
        public int DataUpdateInterval { get; set; }
        
        #endregion
    }
} 