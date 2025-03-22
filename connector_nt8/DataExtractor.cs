#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

//This namespace holds Indicators in this folder and is required. Do not change it. 
namespace NinjaTrader.NinjaScript.Indicators
{
    public class DataExtractor : Indicator
    {
        private string exportFilePath;
        private string exportFileName;
        private string fullExportPath;
        private DateTime lastExportTime;
        private bool headerWritten;
        private int barCount;
        private StreamWriter writer;
        
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description                 = @"Extrae datos de trading a CSV para uso con TradeEvolvePPO";
                Name                        = "DataExtractor";
                Calculate                   = Calculate.OnBarClose;
                IsOverlay                   = true;
                DisplayInDataBox            = true;
                DrawOnPricePanel            = true;
                DrawHorizontalGridLines     = true;
                DrawVerticalGridLines       = true;
                PaintPriceMarkers           = true;
                ScaleJustification          = NinjaTrader.Gui.Chart.ScaleJustification.Right;
                
                // Parámetros configurables
                ExportDirectory             = "exportdata";
                ExportInterval              = 60;  // En segundos
                IncludeHeader               = true;
                ExportOnBarClose            = true;
                MaxBarsToExport             = 500;
                ExportVolumeProfile         = false;
                CustomIndicatorsToExport    = "";
            }
            else if (State == State.Configure)
            {
                // Inicializar variables
                headerWritten = false;
                lastExportTime = DateTime.MinValue;
                barCount = 0;
                
                // Crear el directorio de exportación si no existe
                string ninjaTraderDocumentsDir = NinjaTrader.Core.Globals.UserDataDir;
                exportFilePath = Path.Combine(ninjaTraderDocumentsDir, ExportDirectory);
                
                if (!Directory.Exists(exportFilePath))
                {
                    try
                    {
                        Directory.CreateDirectory(exportFilePath);
                    }
                    catch (Exception ex)
                    {
                        Print("DataExtractor Error: No se pudo crear el directorio de exportación: " + ex.Message);
                    }
                }
                
                // Crear nombre de archivo basado en instrumento y timeframe
                exportFileName = string.Format("{0}_{1}_{2}.csv", 
                    Instrument.MasterInstrument.Name.Replace(" ", "_"),
                    BarsPeriod.Value,
                    BarsPeriod.BarsPeriodType.ToString());
                
                fullExportPath = Path.Combine(exportFilePath, exportFileName);
                
                Print("DataExtractor: Los datos se exportarán a " + fullExportPath);
            }
            else if (State == State.Terminated)
            {
                // Cerrar el StreamWriter si está abierto
                if (writer != null)
                {
                    try
                    {
                        writer.Close();
                        writer.Dispose();
                        writer = null;
                    }
                    catch (Exception ex)
                    {
                        Print("DataExtractor Error al cerrar el archivo: " + ex.Message);
                    }
                }
            }
        }

        protected override void OnBarUpdate()
        {
            // Salir si se alcanzó el límite de barras a exportar
            if (MaxBarsToExport > 0 && CurrentBar >= MaxBarsToExport)
                return;
                
            barCount = Math.Min(CurrentBar + 1, MaxBarsToExport);
            
            // Si ExportOnBarClose está habilitado, exportamos solo al cierre de la barra
            if (ExportOnBarClose && State == State.Realtime && IsFirstTickOfBar)
            {
                ExportData();
            }
            
            // Si no está habilitado ExportOnBarClose, exportamos según el intervalo
            if (!ExportOnBarClose && State == State.Realtime)
            {
                DateTime now = DateTime.Now;
                double secondsSinceLastExport = (now - lastExportTime).TotalSeconds;
                
                if (secondsSinceLastExport >= ExportInterval)
                {
                    ExportData();
                    lastExportTime = now;
                }
            }
        }
        
        private void ExportData()
        {
            try
            {
                // Crear o abrir el archivo CSV
                bool fileExists = File.Exists(fullExportPath);
                
                // Abrir el archivo en modo append
                using (writer = new StreamWriter(fullExportPath, true))
                {
                    // Escribir el encabezado si es necesario
                    if (IncludeHeader && (!fileExists || !headerWritten))
                    {
                        StringBuilder header = new StringBuilder();
                        header.Append("Timestamp,Open,High,Low,Close,Volume");
                        
                        // Agregar indicadores personalizados al encabezado si están definidos
                        if (!string.IsNullOrWhiteSpace(CustomIndicatorsToExport))
                        {
                            string[] indicators = CustomIndicatorsToExport.Split(',');
                            foreach (string indicator in indicators)
                            {
                                header.Append("," + indicator.Trim());
                            }
                        }
                        
                        // Agregar datos del perfil de volumen si está habilitado
                        if (ExportVolumeProfile)
                        {
                            header.Append(",VolumeAtPrice,PriceLevel");
                        }
                        
                        writer.WriteLine(header.ToString());
                        headerWritten = true;
                    }
                    
                    // Escribir datos para cada barra
                    StringBuilder data = new StringBuilder();
                    
                    // Iterar sobre las barras disponibles, limitado por MaxBarsToExport
                    for (int i = Math.Max(0, CurrentBar - barCount + 1); i <= CurrentBar; i++)
                    {
                        StringBuilder line = new StringBuilder();
                        
                        // Datos básicos OHLCV
                        line.Append(Time[i].ToString("yyyy-MM-dd HH:mm:ss") + ",");
                        line.Append(Open[i].ToString() + ",");
                        line.Append(High[i].ToString() + ",");
                        line.Append(Low[i].ToString() + ",");
                        line.Append(Close[i].ToString() + ",");
                        line.Append(Volume[i].ToString());
                        
                        // Agregar indicadores personalizados si están definidos
                        if (!string.IsNullOrWhiteSpace(CustomIndicatorsToExport))
                        {
                            string[] indicators = CustomIndicatorsToExport.Split(',');
                            foreach (string indicator in indicators)
                            {
                                string indicatorName = indicator.Trim();
                                
                                // Intentar obtener el valor del indicador
                                double value = 0;
                                try
                                {
                                    // Obtener el indicador del gráfico
                                    var chartIndicators = ChartControl.Indicators;
                                    
                                    // Buscar el indicador por nombre
                                    foreach (var chartIndicator in chartIndicators)
                                    {
                                        if (chartIndicator.Name.Contains(indicatorName))
                                        {
                                            // Obtener el valor del primer plot del indicador
                                            if (chartIndicator.Values.Any())
                                            {
                                                value = chartIndicator.Values[0].GetValueAt(chartIndicator.BarsInProgress, i);
                                                break;
                                            }
                                        }
                                    }
                                }
                                catch (Exception ex)
                                {
                                    Print("DataExtractor Error al obtener indicador " + indicatorName + ": " + ex.Message);
                                }
                                
                                line.Append("," + value.ToString());
                            }
                        }
                        
                        // Agregar datos del perfil de volumen si está habilitado
                        if (ExportVolumeProfile && State == State.Realtime)
                        {
                            try
                            {
                                // Esta parte es compleja y depende de cómo NinjaTrader implementa el perfil de volumen
                                // Aquí hay una implementación simplificada que puede necesitar ajustes
                                var priceVol = GetVolumeAtPrice(i);
                                if (priceVol != null && priceVol.Any())
                                {
                                    foreach (var pv in priceVol)
                                    {
                                        // Crear una línea separada para cada nivel de precio
                                        StringBuilder volumeLine = new StringBuilder(line.ToString());
                                        volumeLine.Append("," + pv.Volume.ToString() + "," + pv.Price.ToString());
                                        data.AppendLine(volumeLine.ToString());
                                    }
                                    continue; // Saltar la adición de la línea normal
                                }
                            }
                            catch (Exception ex)
                            {
                                Print("DataExtractor Error al obtener perfil de volumen: " + ex.Message);
                            }
                        }
                        
                        data.AppendLine(line.ToString());
                    }
                    
                    writer.Write(data.ToString());
                }
                
                if (State == State.Realtime)
                {
                    Draw.TextFixed(this, "ExportStatus", "Datos exportados: " + DateTime.Now.ToString("HH:mm:ss"), TextPosition.BottomRight);
                }
            }
            catch (Exception ex)
            {
                Print("DataExtractor Error al exportar datos: " + ex.Message);
                if (State == State.Realtime)
                {
                    Draw.TextFixed(this, "ExportError", "Error al exportar: " + ex.Message, TextPosition.BottomRight);
                }
            }
        }
        
        private List<PriceVolumeItem> GetVolumeAtPrice(int barIndex)
        {
            // Esta es una implementación simplificada
            // En un indicador real, necesitarías acceder a los datos de volumen por precio
            // que NinjaTrader proporciona internamente
            
            List<PriceVolumeItem> result = new List<PriceVolumeItem>();
            
            try
            {
                // Esto es solo un ejemplo y no funcionará tal cual
                // NinjaTrader no expone directamente la API para obtener volumen por precio
                // Podrías aproximarlo usando Open, High, Low, Close y Volume
                
                // Para una implementación más precisa, consulta la documentación de NinjaTrader
                // o usa un enfoque basado en la recopilación de datos tick por tick
                
                double price = Close[barIndex];
                long volume = Convert.ToInt64(Volume[barIndex]);
                
                result.Add(new PriceVolumeItem { Price = price, Volume = volume });
            }
            catch (Exception ex)
            {
                Print("Error al obtener volumen por precio: " + ex.Message);
            }
            
            return result;
        }
        
        // Clase auxiliar para almacenar volumen por precio
        private class PriceVolumeItem
        {
            public double Price { get; set; }
            public long Volume { get; set; }
        }

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Directorio de exportación", Description = "Directorio donde se guardarán los archivos CSV", Order = 1, GroupName = "Configuración de exportación")]
        public string ExportDirectory { get; set; }
        
        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Intervalo de exportación (seg)", Description = "Intervalo en segundos para la exportación de datos (solo si ExportOnBarClose = false)", Order = 2, GroupName = "Configuración de exportación")]
        public int ExportInterval { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Incluir encabezado", Description = "Incluir encabezado con nombres de columnas", Order = 3, GroupName = "Configuración de exportación")]
        public bool IncludeHeader { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Exportar al cierre de barra", Description = "Si es verdadero, los datos se exportan solo cuando se cierra una barra", Order = 4, GroupName = "Configuración de exportación")]
        public bool ExportOnBarClose { get; set; }
        
        [NinjaScriptProperty]
        [Range(0, int.MaxValue)]
        [Display(Name = "Máximo de barras a exportar", Description = "Número máximo de barras a exportar (0 = sin límite)", Order = 5, GroupName = "Configuración de exportación")]
        public int MaxBarsToExport { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Exportar perfil de volumen", Description = "Incluir datos de volumen por precio", Order = 6, GroupName = "Configuración avanzada")]
        public bool ExportVolumeProfile { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Indicadores personalizados", Description = "Lista de indicadores a exportar separados por comas", Order = 7, GroupName = "Configuración avanzada")]
        public string CustomIndicatorsToExport { get; set; }
        #endregion
    }
} 