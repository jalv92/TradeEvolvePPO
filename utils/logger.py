"""
Logger module for TradeEvolvePPO.
Provides logging functionality.
"""

import os
import logging
from typing import Optional


def setup_logger(name: str, 
                log_file: Optional[str] = None, 
                level: str = "INFO",
                format_str: Optional[str] = None) -> logging.Logger:
    """
    Setup logger with file and stream handlers.
    
    Args:
        name (str): Logger name
        log_file (Optional[str], optional): Path to log file. Defaults to None.
        level (str, optional): Logging level. Defaults to "INFO".
        format_str (Optional[str], optional): Log format string. Defaults to None.
        
    Returns:
        logging.Logger: Configured logger
    """
    # Convert string level to logging level
    level_dict = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    level_num = level_dict.get(level.upper(), logging.INFO)
    
    # Set default format if not provided
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(level_num)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is provided
    if log_file is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TradeLogger:
    """
    Logger class for trading operations.
    Provides specialized logging for trades and performance.
    """
    
    def __init__(self, 
                name: str = "trade_logger", 
                log_dir: str = "./logs",
                level: str = "INFO"):
        """
        Initialize TradeLogger.
        
        Args:
            name (str, optional): Logger name. Defaults to "trade_logger".
            log_dir (str, optional): Log directory. Defaults to "./logs".
            level (str, optional): Logging level. Defaults to "INFO".
        """
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup main logger
        self.log_file = os.path.join(log_dir, f"{name}.log")
        self.logger = setup_logger(name, self.log_file, level)
        
        # Setup trade logger
        self.trade_log_file = os.path.join(log_dir, "trades.csv")
        self.has_trade_header = os.path.exists(self.trade_log_file) and os.path.getsize(self.trade_log_file) > 0
        
        # Setup performance logger
        self.performance_log_file = os.path.join(log_dir, "performance.csv")
        self.has_performance_header = os.path.exists(self.performance_log_file) and os.path.getsize(self.performance_log_file) > 0
    
    def log_trade(self, trade_info: dict) -> None:
        """
        Log trade information to CSV file.
        
        Args:
            trade_info (dict): Trade information
        """
        # Define expected fields
        expected_fields = ['timestamp', 'action', 'price', 'position', 'pnl', 'balance', 'reason']
        
        # Create CSV header if needed
        write_header = not self.has_trade_header
        
        # Open file in append mode
        with open(self.trade_log_file, 'a') as f:
            # Write header if needed
            if write_header:
                header = ','.join(expected_fields)
                f.write(f"{header}\n")
                self.has_trade_header = True
            
            # Format trade information
            trade_values = []
            for field in expected_fields:
                if field in trade_info:
                    value = trade_info[field]
                    if isinstance(value, float):
                        value = f"{value:.6f}"
                    trade_values.append(str(value))
                else:
                    trade_values.append("")
            
            # Write trade information
            f.write(','.join(trade_values) + '\n')
        
        # Also log to main logger
        self.logger.info(f"Trade: {trade_info}")
    
    def log_performance(self, performance_info: dict) -> None:
        """
        Log performance information to CSV file.
        
        Args:
            performance_info (dict): Performance information
        """
        # Define expected fields
        expected_fields = ['timestamp', 'step', 'balance', 'net_worth', 'position', 'return', 'drawdown']
        
        # Create CSV header if needed
        write_header = not self.has_performance_header
        
        # Open file in append mode
        with open(self.performance_log_file, 'a') as f:
            # Write header if needed
            if write_header:
                header = ','.join(expected_fields)
                f.write(f"{header}\n")
                self.has_performance_header = True
            
            # Format performance information
            performance_values = []
            for field in expected_fields:
                if field in performance_info:
                    value = performance_info[field]
                    if isinstance(value, float):
                        value = f"{value:.6f}"
                    performance_values.append(str(value))
                else:
                    performance_values.append("")
            
            # Write performance information
            f.write(','.join(performance_values) + '\n')
    
    def log_metrics(self, metrics: dict) -> None:
        """
        Log performance metrics.
        
        Args:
            metrics (dict): Performance metrics
        """
        # Log each metric
        self.logger.info("Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_training_step(self, step: int, stats: dict) -> None:
        """
        Log training step information.
        
        Args:
            step (int): Training step
            stats (dict): Training statistics
        """
        # Format statistics
        stats_str = ', '.join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()])
        
        # Log step information
        self.logger.info(f"Step {step}: {stats_str}")