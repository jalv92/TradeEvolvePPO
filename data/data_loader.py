"""
Data loading and preprocessing module.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

def load_data(symbol='NQ', timeframe='5min', start_date='2022-01-01', end_date='2022-12-31'):
    """
    Carga los datos directamente desde un archivo CSV específico o desde la carpeta de datos.
    
    Args:
        symbol (str): Símbolo del instrumento ('NQ', 'ES', etc.)
        timeframe (str): Timeframe de los datos ('1min', '5min', etc.)
        start_date (str): Fecha de inicio para filtrar datos
        end_date (str): Fecha final para filtrar datos
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    # Intentar encontrar un archivo que coincida con los parámetros
    data_dir = os.path.join('data', 'dataset')
    filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.csv"
    
    # Si no existe el archivo específico, buscar cualquier archivo con el símbolo
    if not os.path.exists(os.path.join(data_dir, filename)):
        files = [f for f in os.listdir(data_dir) if f.startswith(f"{symbol}_") and f.endswith('.csv')]
        if files:
            # Usar el archivo más reciente por fecha de modificación
            filename = sorted(files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)[0]
            logger.info(f"Usando archivo existente: {filename}")
        else:
            # Buscar cualquier archivo CSV si no hay específicos del símbolo
            files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if files:
                filename = sorted(files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)[0]
                logger.info(f"Usando archivo alternativo: {filename}")
            else:
                raise FileNotFoundError(f"No se encontraron archivos de datos para {symbol} en {data_dir}")
    
    # Cargar los datos
    file_path = os.path.join(data_dir, filename)
    try:
        logger.info(f"Cargando datos desde {file_path}")
        data = pd.read_csv(file_path)
        
        # Convertir la columna datetime si está presente
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
            data.set_index('datetime', inplace=True)
        
        # Filtrar por fechas si es necesario
        if start_date and end_date:
            mask = (data.index >= start_date) & (data.index <= end_date)
            data = data[mask]
        
        logger.info(f"Datos cargados correctamente: {len(data)} filas")
        return data
        
    except Exception as e:
        logger.error(f"Error cargando datos: {str(e)}")
        # Generar datos sintéticos de ejemplo como último recurso
        logger.warning("Generando datos sintéticos de ejemplo")
        return _generate_sample_data()

def _generate_sample_data(rows=1000):
    """
    Genera datos sintéticos para pruebas cuando no hay datos reales disponibles.
    
    Args:
        rows (int): Número de filas a generar
        
    Returns:
        pd.DataFrame: DataFrame con datos sintéticos
    """
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=rows, freq='5min')
    
    # Generar precios siguiendo un proceso aleatorio
    close = np.random.normal(15000, 150, rows).cumsum() + 15000
    # Generar OHLC basado en close
    high = close + np.random.normal(0, 20, rows).cumsum()
    low = close - np.random.normal(0, 20, rows).cumsum()
    open_price = low + np.random.rand(rows) * (high - low)
    
    # Generar volumen
    volume = np.random.randint(100, 1000, rows)
    
    # Indicadores técnicos básicos
    sma_20 = pd.Series(close).rolling(20).mean().values
    sma_50 = pd.Series(close).rolling(50).mean().values
    sma_200 = pd.Series(close).rolling(200).mean().values
    
    # Generar DataFrame
    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'sma_20': sma_20,
        'sma_50': sma_50,
        'sma_200': sma_200
    }, index=dates)
    
    # Llenar NaNs
    data.fillna(method='bfill', inplace=True)
    data.fillna(method='ffill', inplace=True)
    
    logger.warning(f"Se han generado {len(data)} filas de datos sintéticos para pruebas")
    return data

class DataLoader:
    """
    Class for loading and preprocessing financial data.
    """
    def __init__(self, config):
        """
        Initialize the DataLoader.
        
        Args:
            config (dict): Configuration dictionary for data loading and preprocessing
        """
        self.config = config
        self.data = None
        
    def load_csv_data(self, file_path):
        """
        Load data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            file_path = Path(file_path)
            logger.info(f"Loading data from {file_path}")
            
            data = pd.read_csv(file_path, parse_dates=self.config.get('date_column', ['datetime']), 
                              index_col=self.config.get('index_column', 0))
            
            logger.info(f"Loaded {len(data)} rows of data")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def add_indicators(self, data):
        """
        Verifica que los indicadores necesarios estén presentes en los datos.
        
        Args:
            data (pd.DataFrame): DataFrame con datos OHLCV e indicadores
            
        Returns:
            pd.DataFrame: El mismo DataFrame de entrada
        """
        indicators_config = self.config.get('indicators', [])
        
        # Verificar que todos los indicadores necesarios estén en los datos
        missing_indicators = [ind for ind in indicators_config if ind not in data.columns]
        if missing_indicators:
            logger.warning(f"Los siguientes indicadores no están en los datos: {missing_indicators}")
            
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess the data.
        
        Args:
            data (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Handle missing values
        if self.config.get('handle_missing', True):
            data = self.handle_missing_values(data)
        
        # Normalize features if required
        if self.config.get('normalize', False):
            data = self.normalize_features(data)
        
        return data
    
    def handle_missing_values(self, data):
        """
        Handle missing values in the data.
        
        Args:
            data (pd.DataFrame): Data with potential missing values
            
        Returns:
            pd.DataFrame: Data with handled missing values
        """
        strategy = self.config.get('missing_strategy', 'ffill')
        
        if strategy == 'ffill':
            data = data.ffill()
            # Handle any remaining NaNs at the beginning
            data = data.bfill()
        elif strategy == 'mean':
            data = data.fillna(data.mean())
        elif strategy == 'median':
            data = data.fillna(data.median())
        elif strategy == 'drop':
            data = data.dropna()
        
        return data
    
    def normalize_features(self, data):
        """
        Normalize features in the data.
        
        Args:
            data (pd.DataFrame): Data to normalize
            
        Returns:
            pd.DataFrame: Normalized data
        """
        columns_to_normalize = self.config.get('normalize_columns', [])
        
        if not columns_to_normalize:
            columns_to_normalize = data.columns
        
        for column in columns_to_normalize:
            if column in data.columns:
                mean = data[column].mean()
                std = data[column].std()
                if std != 0:
                    data[column] = (data[column] - mean) / std
        
        return data
    
    def split_data(self, data, train_ratio=0.7, val_ratio=0.15):
        """
        Split data into training, validation, and test sets.
        
        Args:
            data (pd.DataFrame): Data to split
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def prepare_data(self, file_path):
        """
        Full pipeline to load, process, and split data.
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        data = self.load_csv_data(file_path)
        data = self.add_indicators(data)
        data = self.preprocess_data(data)
        
        train_ratio = self.config.get('train_ratio', 0.7)
        val_ratio = self.config.get('val_ratio', 0.15)
        
        return self.split_data(data, train_ratio, val_ratio)

    def get_data_stats(self):
        """
        Get basic statistics about the data.
        
        Returns:
            dict: Dictionary with data statistics
        """
        stats = {}
        
        if self.data is not None:
            stats['data_shape'] = self.data.shape
            stats['data_period'] = (self.data.index[0], self.data.index[-1])
            stats['data_mean'] = self.data['close'].mean()
            stats['data_std'] = self.data['close'].std()
        
        return stats
