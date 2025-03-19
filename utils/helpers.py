"""
Helper utilities for TradeEvolvePPO.
Provides helper functions for various tasks.
"""

import os
import json
import yaml
import argparse
from typing import Dict, Any, Optional, Union, List


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Determine file type from extension
    _, ext = os.path.splitext(config_path)
    
    # Load configuration
    if ext.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif ext.lower() == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file type: {ext}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        config_path (str): Path to save configuration
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Determine file type from extension
    _, ext = os.path.splitext(config_path)
    
    # Save configuration
    if ext.lower() in ['.yaml', '.yml']:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif ext.lower() == '.json':
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    else:
        raise ValueError(f"Unsupported config file type: {ext}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='TradeEvolvePPO - Reinforcement Learning for Trading')
    
    # Mode argument
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'backtest'], default='train',
                        help='Operation mode: train, test, or backtest')
    
    # Configuration arguments
    parser.add_argument('--config', type=str, default='config/config.py',
                        help='Path to configuration file')
    
    # Data arguments
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data file')
    
    # Model arguments
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (for test or backtest mode)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='./results',
                        help='Path to output directory')
    
    # Training arguments
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Number of timesteps for training')
    
    # Device argument
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, auto)')
    
    # Visualization argument
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization')
    
    # Verbose argument
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0: silent, 1: normal, 2: debug)')
    
    return parser.parse_args()


def create_directories(paths: List[str]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        paths (List[str]): List of directory paths
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"