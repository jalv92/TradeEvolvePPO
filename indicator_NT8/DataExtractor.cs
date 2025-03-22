using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
// Restauramos Linq pero usaremos calificación completa para sus métodos
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
using NinjaTrader.Core.FloatingPoint;

namespace NinjaTrader.NinjaScript.Indicators
{
    public class DataExtractor : Indicator
    {
        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Export Folder", Description = "Path to export CSV files", Order = 1, GroupName = "Export Settings")]
        public string ExportFolder { get; set; }

        [NinjaScriptProperty]
        [Range(100, 1000)]
        [Display(Name = "Block Size", Description = "Number of bars in each export block", Order = 2, GroupName = "Export Settings")]
        public int BlockSize { get; set; }

        [NinjaScriptProperty]
        [Range(0, int.MaxValue)]
        [Display(Name = "Max Blocks", Description = "Maximum number of blocks to export (0 = all)", Order = 3, GroupName = "Export Settings")]
        public int MaxBlocks { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Auto Export", Description = "Automatically export when indicator is added", Order = 4, GroupName = "Export Settings")]
        public bool AutoExport { get; set; }

        [NinjaScriptProperty]
        [Range(1, 100)]
        [Display(Name = "Fast EMA Period", Description = "Period for fast EMA", Order = 5, GroupName = "Indicators")]
        public int FastEMA { get; set; }

        [NinjaScriptProperty]
        [Range(10, 200)]
        [Display(Name = "Slow EMA Period", Description = "Period for slow EMA", Order = 6, GroupName = "Indicators")]
        public int SlowEMA { get; set; }

        [NinjaScriptProperty]
        [Range(5, 50)]
        [Display(Name = "RSI Period", Description = "Period for RSI", Order = 7, GroupName = "Indicators")]
        public int RSIPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(10, 50)]
        [Display(Name = "Bollinger Period", Description = "Period for Bollinger Bands", Order = 8, GroupName = "Indicators")]
        public int BollingerPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(1.0, 3.0)]
        [Display(Name = "Bollinger StdDev", Description = "StdDev for Bollinger Bands", Order = 9, GroupName = "Indicators")]
        public double BollingerStdDev { get; set; }

        [NinjaScriptProperty]
        [Range(5, 50)]
        [Display(Name = "MACD Fast", Description = "Fast period for MACD", Order = 10, GroupName = "Indicators")]
        public int MACDFast { get; set; }

        [NinjaScriptProperty]
        [Range(10, 100)]
        [Display(Name = "MACD Slow", Description = "Slow period for MACD", Order = 11, GroupName = "Indicators")]
        public int MACDSlow { get; set; }

        [NinjaScriptProperty]
        [Range(3, 20)]
        [Display(Name = "MACD Signal", Description = "Signal period for MACD", Order = 12, GroupName = "Indicators")]
        public int MACDSignal { get; set; }

        [NinjaScriptProperty]
        [Range(7, 100)]
        [Display(Name = "ATR Period", Description = "Period for ATR", Order = 13, GroupName = "Indicators")]
        public int ATRPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(3, 30)]
        [Display(Name = "ADX Period", Description = "Period for ADX", Order = 14, GroupName = "Indicators")]
        public int ADXPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(5, 50)]
        [Display(Name = "Stochastic Period", Description = "Period for Stochastic", Order = 15, GroupName = "Indicators")]
        public int StochPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(3, 20)]
        [Display(Name = "Stochastic K Smoothing", Description = "K smoothing for Stochastic", Order = 16, GroupName = "Indicators")]
        public int StochK { get; set; }

        [NinjaScriptProperty]
        [Range(3, 20)]
        [Display(Name = "Stochastic D Smoothing", Description = "D smoothing for Stochastic", Order = 17, GroupName = "Indicators")]
        public int StochD { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Debug Mode", Description = "Enable detailed logging", Order = 18, GroupName = "Debug")]
        public bool DebugMode { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Clean Files", Description = "Clean up temporary files after combining", Order = 19, GroupName = "Debug")]
        public bool CleanFiles { get; set; }
        #endregion

        #region Variables
        private EMA emaFast;
        private EMA emaSlow;
        private RSI rsiIndicator;
        private Bollinger bbandsIndicator;
        private MACD macdIndicator;
        private ATR atrIndicator;
        private ADX adxIndicator;
        private Stochastics stochIndicator;
        private VOL volumeIndicator;
        private OBV obvIndicator;

        private bool isExporting = false;
        private string statusText = "";
        private string combinedFileName = "";
        private List<string> tempFiles = new List<string>();
        private ChartControl chartControl;
        
        // Nuevas variables para los controles WPF
        private System.Windows.Controls.Grid controlGrid;
        private System.Windows.Controls.Button exportButton;
        private System.Windows.Controls.TextBlock statusTextBlock;
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Data extractor for TradeEvolvePPO with batched export";
                Name = "DataExtractor";
                Calculate = Calculate.OnBarClose;
                IsOverlay = false;
                DisplayInDataBox = false;
                DrawOnPricePanel = false;
                IsSuspendedWhileInactive = true;

                // Default values
                ExportFolder = @"C:\TradeEvolvePPO\data";
                BlockSize = 200;
                MaxBlocks = 0;  // 0 = extract all available data
                AutoExport = true;

                // Indicator defaults
                FastEMA = 9;
                SlowEMA = 21;
                RSIPeriod = 14;
                BollingerPeriod = 20;
                BollingerStdDev = 2.0;
                MACDFast = 12;
                MACDSlow = 26;
                MACDSignal = 9;
                ATRPeriod = 14;
                ADXPeriod = 14;
                StochPeriod = 14;
                StochK = 3;
                StochD = 3;

                // Debug
                DebugMode = true;
                CleanFiles = true;

                Log("SetDefaults complete");
            }
            else if (State == State.Configure)
            {
                Log($"Configuring indicator - Export Folder: {ExportFolder}, Block Size: {BlockSize}");

                try
                {
                    // Create our own instances of indicators with exact parameters
                    emaFast = EMA(FastEMA);
                    emaSlow = EMA(SlowEMA);
                    rsiIndicator = RSI(RSIPeriod, 1);
                    bbandsIndicator = Bollinger(BollingerPeriod, (int)BollingerStdDev);
                    macdIndicator = MACD(MACDFast, MACDSlow, MACDSignal);
                    atrIndicator = ATR(ATRPeriod);
                    adxIndicator = ADX(ADXPeriod);
                    stochIndicator = Stochastics(StochPeriod, StochK, StochD);
                    volumeIndicator = VOL();
                    obvIndicator = OBV();
                    
                    Log("All indicators configured successfully");
                }
                catch (Exception ex)
                {
                    Print($"ERROR in Configure: {ex.Message}");
                    if (ex.InnerException != null)
                        Print($"Inner Exception: {ex.InnerException.Message}");
                }
            }
            else if (State == State.Historical)
            {
                Log("Starting historical processing");
            }
            else if (State == State.Realtime)
            {
                Log("Transitioning to realtime");
                chartControl = ChartControl;

                // Crear los controles de UI
                Dispatcher.InvokeAsync(() => CreateUIControls());

                // Auto-export if enabled
                if (AutoExport)
                {
                    Log("Auto-export is enabled, starting export process");
                    ExportDataInBlocks();
                }
            }
            else if (State == State.Terminated)
            {
                Log("Indicator terminating");
                Dispatcher.InvokeAsync(() => RemoveUIControls());
            }
        }

        private void CreateUIControls()
        {
            try
            {
                if (UserControlCollection.Contains(controlGrid))
                    return;

                // Crear el grid principal
                controlGrid = new System.Windows.Controls.Grid
                {
                    Name = "DataExtractorGrid",
                    HorizontalAlignment = HorizontalAlignment.Right,
                    VerticalAlignment = VerticalAlignment.Bottom,
                    Margin = new Thickness(0, 0, 10, 10)
                };

                // Definir filas para botón y estado
                System.Windows.Controls.RowDefinition row1 = new System.Windows.Controls.RowDefinition();
                System.Windows.Controls.RowDefinition row2 = new System.Windows.Controls.RowDefinition();
                controlGrid.RowDefinitions.Add(row1);
                controlGrid.RowDefinitions.Add(row2);

                // Crear el botón de exportación
                exportButton = new System.Windows.Controls.Button
                {
                    Name = "ExportButton",
                    Content = "EXPORT DATA",
                    Foreground = Brushes.White,
                    Background = Brushes.DarkBlue,
                    Padding = new Thickness(10, 5, 10, 5),
                    Margin = new Thickness(0, 0, 0, 5)
                };
                
                // Asignar el evento de clic
                exportButton.Click += OnExportButtonClick;
                
                // Añadir botón al grid
                System.Windows.Controls.Grid.SetRow(exportButton, 0);
                controlGrid.Children.Add(exportButton);

                // Crear el bloque de texto para status
                statusTextBlock = new System.Windows.Controls.TextBlock
                {
                    Name = "StatusTextBlock",
                    Text = "",
                    Foreground = Brushes.White,
                    Background = Brushes.Transparent,
                    TextAlignment = TextAlignment.Center,
                    Padding = new Thickness(5),
                    MaxWidth = 200,
                    TextWrapping = TextWrapping.Wrap
                };
                
                // Añadir bloque de texto al grid
                System.Windows.Controls.Grid.SetRow(statusTextBlock, 1);
                controlGrid.Children.Add(statusTextBlock);

                // Añadir el grid a la colección de controles
                UserControlCollection.Add(controlGrid);
                
                Log("UI controls created successfully");
            }
            catch (Exception ex)
            {
                Print($"Error creating UI controls: {ex.Message}");
                if (ex.InnerException != null)
                    Print($"Inner Exception: {ex.InnerException.Message}");
            }
        }

        private void RemoveUIControls()
        {
            try
            {
                if (controlGrid != null)
                {
                    if (exportButton != null)
                    {
                        controlGrid.Children.Remove(exportButton);
                        exportButton.Click -= OnExportButtonClick;
                        exportButton = null;
                    }

                    if (statusTextBlock != null)
                    {
                        controlGrid.Children.Remove(statusTextBlock);
                        statusTextBlock = null;
                    }

                    UserControlCollection.Remove(controlGrid);
                    controlGrid = null;
                }
                
                Log("UI controls removed successfully");
            }
            catch (Exception ex)
            {
                Print($"Error removing UI controls: {ex.Message}");
            }
        }

        private void OnExportButtonClick(object sender, RoutedEventArgs e)
        {
            if (!isExporting)
            {
                Log("Export button clicked");
                ExportDataInBlocks();
            }
            else
            {
                Log("Export already in progress, button click ignored");
            }
        }

        protected override void OnBarUpdate()
        {
            // This indicator doesn't need to do calculations on each bar
            // It only exports data on demand
        }

        private void ExportDataInBlocks()
        {
            if (isExporting) return;
            isExporting = true;

            try
            {
                UpdateStatus("Starting export process...");

                if (!Directory.Exists(ExportFolder))
                {
                    try
                    {
                        Directory.CreateDirectory(ExportFolder);
                        Log($"Created export directory: {ExportFolder}");
                    }
                    catch (Exception ex)
                    {
                        Print($"ERROR creating directory: {ex.Message}");
                        
                        ExportFolder = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
                        Log($"Using desktop as fallback: {ExportFolder}");
                    }
                }

                int totalBars = 0;
                if (BarsArray != null && BarsArray.Length > 0 && BarsArray[0] != null)
                {
                    totalBars = BarsArray[0].Count;
                }
                Log($"Total bars available: {totalBars}");

                if (totalBars <= 0)
                {
                    UpdateStatus("No data available for export");
                    isExporting = false;
                    return;
                }

                int numBlocks = (int)Math.Ceiling((double)totalBars / BlockSize);
                Log($"Will export {numBlocks} blocks of {BlockSize} bars each");

                if (MaxBlocks > 0 && numBlocks > MaxBlocks)
                {
                    numBlocks = MaxBlocks;
                    Log($"Limiting to {MaxBlocks} blocks as per settings");
                }

                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string instrumentName = SanitizeFileName(Instrument.FullName);
                
                combinedFileName = Path.Combine(ExportFolder, $"{instrumentName}_combined_{timestamp}.csv");
                Log($"Combined file will be: {combinedFileName}");
                
                tempFiles.Clear();

                for (int block = 0; block < numBlocks; block++)
                {
                    int startIdx = totalBars - (block + 1) * BlockSize;
                    int endIdx = totalBars - block * BlockSize - 1;
                    
                    if (startIdx < 0)
                        startIdx = 0;
                        
                    UpdateStatus($"Exporting block {block+1}/{numBlocks}...");
                    string blockFile = ExportDataBlock(startIdx, endIdx, block, timestamp);
                    
                    if (!string.IsNullOrEmpty(blockFile))
                        tempFiles.Add(blockFile);
                }

                CombineBlockFiles();
                
                UpdateStatus($"Export complete!\nFile: {Path.GetFileName(combinedFileName)}");
            }
            catch (Exception ex)
            {
                Print($"ERROR initiating export: {ex.Message}");
                UpdateStatus($"Failed to start export: {ex.Message}");
            }
            finally
            {
                isExporting = false;
            }
        }

        private void UpdateStatus(string message)
        {
            statusText = message;
            
            try
            {
                // Actualizar el texto de estado en la UI
                Dispatcher.InvokeAsync(() =>
                {
                    if (statusTextBlock != null)
                        statusTextBlock.Text = message;
                });
                
                Print($"STATUS: {message}");
            }
            catch (Exception ex)
            {
                Print($"ERROR updating status: {ex.Message}");
            }
        }

        private string ExportDataBlock(int startIdx, int endIdx, int blockNum, string timestamp)
        {
            try
            {
                if (startIdx < 0) startIdx = 0;
                
                int barsCount = 0;
                if (BarsArray != null && BarsArray.Length > 0 && BarsArray[0] != null)
                {
                    barsCount = BarsArray[0].Count;
                }
                
                if (endIdx >= barsCount) endIdx = barsCount - 1;
                
                Log($"Exporting block {blockNum}: bars {startIdx} to {endIdx}");
                
                string instrumentName = SanitizeFileName(Instrument.FullName);
                string blockFile = Path.Combine(ExportFolder, $"{instrumentName}_block{blockNum}_{timestamp}.csv");
                
                var chartIndicators = GetChartIndicators();
                // No use el operador ternario con Count que puede causar ambigüedad
                int indicatorCount = 0;
                if (chartIndicators != null)
                { 
                    // Contar uno por uno en lugar de acceder a Count
                    try
                    {
                        foreach (var item in chartIndicators)
                        {
                            indicatorCount++;
                        }
                    }
                    catch { }
                }
                Log($"Found {indicatorCount} additional indicators on chart");
                
                List<string> headerColumns = new List<string>
                {
                    "datetime", "open", "high", "low", "close", "volume",
                    "ema_fast", "ema_slow", "rsi", 
                    "bb_upper", "bb_middle", "bb_lower",
                    "macd_line", "macd_signal", "macd_histogram",
                    "atr", "adx", "pos_di", "neg_di",
                    "stoch_k", "stoch_d", "obv"
                };
                
                foreach (var indInfo in chartIndicators)
                {
                    headerColumns.Add(indInfo.Key);
                }
                
                List<string> csvLines = new List<string>();
                csvLines.Add(string.Join(",", headerColumns));
                
                for (int i = startIdx; i <= endIdx; i++)
                {
                    try
                    {
                        List<string> row = new List<string>();
                        
                        row.Add(Time[i].ToString("yyyy-MM-dd HH:mm:ss"));
                        row.Add(FormatValue(Open[i]));
                        row.Add(FormatValue(High[i]));
                        row.Add(FormatValue(Low[i]));
                        row.Add(FormatValue(Close[i]));
                        row.Add(FormatValue(Volume[i]));
                        
                        row.Add(FormatValue(emaFast[i]));
                        row.Add(FormatValue(emaSlow[i]));
                        row.Add(FormatValue(rsiIndicator[i]));
                        
                        row.Add(FormatValue(bbandsIndicator.Upper[i]));
                        row.Add(FormatValue(bbandsIndicator.Middle[i]));
                        row.Add(FormatValue(bbandsIndicator.Lower[i]));
                        
                        row.Add(FormatValue(macdIndicator[i]));
                        row.Add(FormatValue(macdIndicator.Avg[i]));
                        row.Add(FormatValue(macdIndicator.Diff[i]));
                        
                        row.Add(FormatValue(atrIndicator[i]));
                        row.Add(FormatValue(adxIndicator[i]));
                        
                        double posDI = 0, negDI = 0;
                        CalculateDI(i, out posDI, out negDI);
                        row.Add(FormatValue(posDI));
                        row.Add(FormatValue(negDI));
                        
                        row.Add(FormatValue(stochIndicator.K[i]));
                        row.Add(FormatValue(stochIndicator.D[i]));
                        
                        row.Add(FormatValue(obvIndicator[i]));
                        
                        foreach (var indInfo in chartIndicators)
                        {
                            try
                            {
                                var ind = indInfo.Value.Item1;
                                int seriesIdx = indInfo.Value.Item2;
                                
                                int valCount = 0;
                                if (ind != null && ind.Values != null)
                                {
                                    // Determinar el conteo evitando el acceso directo a Count
                                    var values = ind.Values;
                                    // Enumerar uno por uno hasta encontrar el fin (enfoque más seguro)
                                    for (int j = 0; j < 100; j++) // Límite razonable para evitar bucles infinitos
                                    {
                                        try
                                        {
                                            // En lugar de usar Count, intentamos acceder al elemento
                                            // Si no existe, se lanzará una excepción
                                            var tempValue = values[j];
                                            valCount = j + 1; // Si llegamos aquí, el índice es válido
                                        }
                                        catch
                                        {
                                            break; // Si hay una excepción, hemos llegado al final
                                        }
                                    }
                                }
                                
                                if (ind != null && ind.Values != null && seriesIdx < valCount)
                                {
                                    var series = ind.Values[seriesIdx];
                                    
                                    int seriesCount = 0;
                                    if (series != null)
                                    {
                                        // Determinar el conteo de series sin usar Count
                                        for (int j = 0; j < 20000; j++) // Suficiente para datos de trading
                                        {
                                            try
                                            {
                                                var tempVal = series[j];
                                                seriesCount = j + 1;
                                            }
                                            catch
                                            {
                                                break;
                                            }
                                        }
                                    }
                                    
                                    if (series != null && i < seriesCount)
                                    {
                                        double val = series[i];
                                        row.Add(FormatValue(val));
                                    }
                                    else
                                    {
                                        row.Add("");
                                    }
                                }
                                else
                                {
                                    row.Add("");
                                }
                            }
                            catch
                            {
                                row.Add("");
                            }
                        }
                        
                        csvLines.Add(string.Join(",", row));
                    }
                    catch (Exception ex)
                    {
                        Print($"ERROR processing bar {i}: {ex.Message}");
                        continue;
                    }
                }
                
                File.WriteAllLines(blockFile, csvLines);
                // No use el operador ternario con Count que puede causar ambigüedad
                int numRows = 0;
                if (csvLines != null)
                {
                    // Contar manualmente los elementos
                    foreach (var line in csvLines)
                    {
                        numRows++;
                    }
                }
                Log($"Block {blockNum} exported: {numRows} rows to {blockFile}");
                
                return blockFile;
            }
            catch (Exception ex)
            {
                Print($"ERROR exporting block {blockNum}: {ex.Message}");
                return null;
            }
        }

        private void CombineBlockFiles()
        {
            // No use el operador ternario con Count que puede causar ambigüedad
            int tempFilesCount = 0;
            if (tempFiles != null)
            {
                // Contar manualmente los elementos
                foreach (var file in tempFiles)
                {
                    tempFilesCount++;
                }
            }
            
            if (tempFilesCount == 0)
            {
                Log("No block files to combine");
                return;
            }
            
            try
            {
                Log($"Combining {tempFilesCount} block files into {combinedFileName}");
                
                // Usamos calificación completa para los métodos de Linq
                string header = System.Linq.Enumerable.First(File.ReadLines(tempFiles[0]));
                
                using (StreamWriter writer = new StreamWriter(combinedFileName))
                {
                    writer.WriteLine(header);
                    
                    for (int i = tempFilesCount - 1; i >= 0; i--)
                    {
                        string blockFile = tempFiles[i];
                        // Usamos calificación completa para los métodos de Linq
                        var lines = System.Linq.Enumerable.Skip(File.ReadLines(blockFile), 1);
                        
                        foreach (var line in lines)
                        {
                            writer.WriteLine(line);
                        }
                    }
                }
                
                Log($"Successfully combined all blocks into: {combinedFileName}");
                
                if (CleanFiles)
                {
                    foreach (string file in tempFiles)
                    {
                        try
                        {
                            File.Delete(file);
                        }
                        catch (Exception ex)
                        {
                            Print($"Failed to delete temp file {file}: {ex.Message}");
                        }
                    }
                    Log("Temporary block files cleaned up");
                }
            }
            catch (Exception ex)
            {
                Print($"ERROR combining block files: {ex.Message}");
            }
        }

        private Dictionary<string, Tuple<NinjaTrader.Gui.NinjaScript.IndicatorRenderBase, int>> GetChartIndicators()
        {
            var result = new Dictionary<string, Tuple<NinjaTrader.Gui.NinjaScript.IndicatorRenderBase, int>>();
            
            try
            {
                if (ChartControl != null && ChartControl.Indicators != null)
                {
                    // Enfoque nativo de NinjaTrader: usar directamente la colección y enumerar los elementos
                    foreach (var ind in ChartControl.Indicators)
                    {
                        if (ind == this || IsStandardIndicator(ind)) 
                            continue;
                        
                        try
                        {
                            // Enfoque simplificado para obtener el conteo de valores
                            // Evitamos acceder directamente a Count
                            int valCount = 0;
                            
                            if (ind != null && ind.Values != null)
                            {
                                // Determinar el conteo evitando el acceso directo a Count
                                var values = ind.Values;
                                // Enumerar uno por uno hasta encontrar el fin (enfoque más seguro)
                                for (int i = 0; i < 100; i++) // Límite razonable para evitar bucles infinitos
                                {
                                    try
                                    {
                                        // En lugar de usar Count, intentamos acceder al elemento
                                        // Si no existe, se lanzará una excepción
                                        var tempValue = values[i];
                                        valCount = i + 1; // Si llegamos aquí, el índice es válido
                                    }
                                    catch
                                    {
                                        break; // Si hay una excepción, hemos llegado al final
                                    }
                                }
                            }
                            
                            for (int i = 0; i < valCount; i++)
                            {
                                string name = $"{SanitizeName(ind.Name)}_{i}";
                                int suffix = 1;
                                string originalName = name;
                                
                                // Verificamos si la clave ya existe sin usar ContainsKey
                                bool keyExists = false;
                                try {
                                    var tempVal = result[name];
                                    keyExists = true;
                                } catch {
                                    keyExists = false;
                                }
                                
                                while (keyExists)
                                {
                                    name = $"{originalName}_{suffix}";
                                    suffix++;
                                    
                                    // Verificamos de nuevo
                                    try {
                                        var tempVal = result[name];
                                        keyExists = true;
                                    } catch {
                                        keyExists = false;
                                    }
                                }
                                
                                result.Add(name, Tuple.Create(ind, i));
                            }
                        }
                        catch (Exception ex)
                        {
                            Print($"Warning: Failed to process indicator {ind.Name}: {ex.Message}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"ERROR getting chart indicators: {ex.Message}");
            }
            
            return result;
        }

        private bool IsStandardIndicator(NinjaTrader.Gui.NinjaScript.IndicatorRenderBase ind)
        {
            return (ind == emaFast || ind == emaSlow || ind == rsiIndicator ||
                    ind == bbandsIndicator || ind == macdIndicator || ind == atrIndicator ||
                    ind == adxIndicator || ind == stochIndicator || ind == volumeIndicator ||
                    ind == obvIndicator);
        }

        private void CalculateDI(int barIndex, out double posDI, out double negDI)
        {
            posDI = 0;
            negDI = 0;
            
            try
            {
                // Enfoque nativo y simplificado para determinar cuántas barras hay
                bool barIndexValid = false;
                
                if (Bars != null)
                {
                    // Validar el índice directamente
                    try 
                    {
                        // Si podemos acceder a este índice, es válido
                        var temp = High[barIndex];
                        var tempPrev = Close[barIndex-1];
                        barIndexValid = true;
                    }
                    catch 
                    {
                        barIndexValid = false;
                    }
                }
                
                if (!barIndexValid || barIndex <= 0)
                    return;
                
                double trueRange = Math.Max(High[barIndex] - Low[barIndex], 
                                    Math.Max(Math.Abs(High[barIndex] - Close[barIndex-1]),
                                             Math.Abs(Low[barIndex] - Close[barIndex-1])));
                
                double upMove = High[barIndex] - High[barIndex-1];
                double downMove = Low[barIndex-1] - Low[barIndex];
                
                if (trueRange > 0)
                {
                    if (upMove > downMove && upMove > 0)
                        posDI = (upMove / trueRange) * 100;
                    
                    if (downMove > upMove && downMove > 0)
                        negDI = (downMove / trueRange) * 100;
                }
            }
            catch (Exception ex)
            {
                Print($"ERROR calculating DI: {ex.Message}");
            }
        }

        private string SanitizeFileName(string name)
        {
            foreach (char c in Path.GetInvalidFileNameChars())
            {
                name = name.Replace(c, '_');
            }
            
            name = name.Replace(' ', '_')
                       .Replace(':', '_')
                       .Replace('/', '_')
                       .Replace('\\', '_')
                       .Replace('[', '_')
                       .Replace(']', '_');
            
            return name;
        }

        private string SanitizeName(string name)
        {
            StringBuilder sb = new StringBuilder();
            
            for (int i = 0; i < name.Length; i++)
            {
                char c = name[i];
                if (char.IsLetterOrDigit(c) || c == '_')
                    sb.Append(c);
                else
                    sb.Append('_');
            }
            
            string result = sb.ToString().ToLower();
            
            if (result.Length > 0 && char.IsDigit(result[0]))
                result = "ind_" + result;
                
            return result;
        }

        private string FormatValue(double value)
        {
            return value.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }

        private void Log(string message)
        {
            if (DebugMode)
            {
                Print($"[DEBUG] {Name}: {message}");
            }
        }
    }
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private DataExtractor[] cacheDataExtractor;
		public DataExtractor DataExtractor(string exportFolder, int blockSize, int maxBlocks, bool autoExport, int fastEMA, int slowEMA, int rSIPeriod, int bollingerPeriod, double bollingerStdDev, int mACDFast, int mACDSlow, int mACDSignal, int aTRPeriod, int aDXPeriod, int stochPeriod, int stochK, int stochD, bool debugMode, bool cleanFiles)
		{
			return DataExtractor(Input, exportFolder, blockSize, maxBlocks, autoExport, fastEMA, slowEMA, rSIPeriod, bollingerPeriod, bollingerStdDev, mACDFast, mACDSlow, mACDSignal, aTRPeriod, aDXPeriod, stochPeriod, stochK, stochD, debugMode, cleanFiles);
		}

		public DataExtractor DataExtractor(ISeries<double> input, string exportFolder, int blockSize, int maxBlocks, bool autoExport, int fastEMA, int slowEMA, int rSIPeriod, int bollingerPeriod, double bollingerStdDev, int mACDFast, int mACDSlow, int mACDSignal, int aTRPeriod, int aDXPeriod, int stochPeriod, int stochK, int stochD, bool debugMode, bool cleanFiles)
		{
			if (cacheDataExtractor != null)
				for (int idx = 0; idx < cacheDataExtractor.Length; idx++)
					if (cacheDataExtractor[idx] != null && cacheDataExtractor[idx].ExportFolder == exportFolder && cacheDataExtractor[idx].BlockSize == blockSize && cacheDataExtractor[idx].MaxBlocks == maxBlocks && cacheDataExtractor[idx].AutoExport == autoExport && cacheDataExtractor[idx].FastEMA == fastEMA && cacheDataExtractor[idx].SlowEMA == slowEMA && cacheDataExtractor[idx].RSIPeriod == rSIPeriod && cacheDataExtractor[idx].BollingerPeriod == bollingerPeriod && cacheDataExtractor[idx].BollingerStdDev == bollingerStdDev && cacheDataExtractor[idx].MACDFast == mACDFast && cacheDataExtractor[idx].MACDSlow == mACDSlow && cacheDataExtractor[idx].MACDSignal == mACDSignal && cacheDataExtractor[idx].ATRPeriod == aTRPeriod && cacheDataExtractor[idx].ADXPeriod == aDXPeriod && cacheDataExtractor[idx].StochPeriod == stochPeriod && cacheDataExtractor[idx].StochK == stochK && cacheDataExtractor[idx].StochD == stochD && cacheDataExtractor[idx].DebugMode == debugMode && cacheDataExtractor[idx].CleanFiles == cleanFiles && cacheDataExtractor[idx].EqualsInput(input))
						return cacheDataExtractor[idx];
			return CacheIndicator<DataExtractor>(new DataExtractor(){ ExportFolder = exportFolder, BlockSize = blockSize, MaxBlocks = maxBlocks, AutoExport = autoExport, FastEMA = fastEMA, SlowEMA = slowEMA, RSIPeriod = rSIPeriod, BollingerPeriod = bollingerPeriod, BollingerStdDev = bollingerStdDev, MACDFast = mACDFast, MACDSlow = mACDSlow, MACDSignal = mACDSignal, ATRPeriod = aTRPeriod, ADXPeriod = aDXPeriod, StochPeriod = stochPeriod, StochK = stochK, StochD = stochD, DebugMode = debugMode, CleanFiles = cleanFiles }, input, ref cacheDataExtractor);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.DataExtractor DataExtractor(string exportFolder, int blockSize, int maxBlocks, bool autoExport, int fastEMA, int slowEMA, int rSIPeriod, int bollingerPeriod, double bollingerStdDev, int mACDFast, int mACDSlow, int mACDSignal, int aTRPeriod, int aDXPeriod, int stochPeriod, int stochK, int stochD, bool debugMode, bool cleanFiles)
		{
			return indicator.DataExtractor(Input, exportFolder, blockSize, maxBlocks, autoExport, fastEMA, slowEMA, rSIPeriod, bollingerPeriod, bollingerStdDev, mACDFast, mACDSlow, mACDSignal, aTRPeriod, aDXPeriod, stochPeriod, stochK, stochD, debugMode, cleanFiles);
		}

		public Indicators.DataExtractor DataExtractor(ISeries<double> input , string exportFolder, int blockSize, int maxBlocks, bool autoExport, int fastEMA, int slowEMA, int rSIPeriod, int bollingerPeriod, double bollingerStdDev, int mACDFast, int mACDSlow, int mACDSignal, int aTRPeriod, int aDXPeriod, int stochPeriod, int stochK, int stochD, bool debugMode, bool cleanFiles)
		{
			return indicator.DataExtractor(input, exportFolder, blockSize, maxBlocks, autoExport, fastEMA, slowEMA, rSIPeriod, bollingerPeriod, bollingerStdDev, mACDFast, mACDSlow, mACDSignal, aTRPeriod, aDXPeriod, stochPeriod, stochK, stochD, debugMode, cleanFiles);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.DataExtractor DataExtractor(string exportFolder, int blockSize, int maxBlocks, bool autoExport, int fastEMA, int slowEMA, int rSIPeriod, int bollingerPeriod, double bollingerStdDev, int mACDFast, int mACDSlow, int mACDSignal, int aTRPeriod, int aDXPeriod, int stochPeriod, int stochK, int stochD, bool debugMode, bool cleanFiles)
		{
			return indicator.DataExtractor(Input, exportFolder, blockSize, maxBlocks, autoExport, fastEMA, slowEMA, rSIPeriod, bollingerPeriod, bollingerStdDev, mACDFast, mACDSlow, mACDSignal, aTRPeriod, aDXPeriod, stochPeriod, stochK, stochD, debugMode, cleanFiles);
		}

		public Indicators.DataExtractor DataExtractor(ISeries<double> input , string exportFolder, int blockSize, int maxBlocks, bool autoExport, int fastEMA, int slowEMA, int rSIPeriod, int bollingerPeriod, double bollingerStdDev, int mACDFast, int mACDSlow, int mACDSignal, int aTRPeriod, int aDXPeriod, int stochPeriod, int stochK, int stochD, bool debugMode, bool cleanFiles)
		{
			return indicator.DataExtractor(input, exportFolder, blockSize, maxBlocks, autoExport, fastEMA, slowEMA, rSIPeriod, bollingerPeriod, bollingerStdDev, mACDFast, mACDSlow, mACDSignal, aTRPeriod, aDXPeriod, stochPeriod, stochK, stochD, debugMode, cleanFiles);
		}
	}
}

#endregion
