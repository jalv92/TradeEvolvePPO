#!/usr/bin/env python
"""
Data preprocessor for TradeEvolvePPO

This script prepares and optimizes CSV data exported from NinjaTrader 8
for training with the TradeEvolvePPO reinforcement learning system.

Usage:
    python preprocess_data.py input.csv output.csv [--normalize] [--dropna] [--fillna method]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess data for TradeEvolvePPO')
    parser.add_argument('input', help='Input CSV file from NinjaTrader 8')
    parser.add_argument('output', help='Output CSV file for TradeEvolvePPO')
    parser.add_argument('--normalize', action='store_true', help='Normalize all feature columns')
    parser.add_argument('--dropna', action='store_true', help='Drop rows with missing values')
    parser.add_argument('--fillna', choices=['ffill', 'bfill', 'zero', 'mean', 'median'], 
                      help='Method to fill missing values')
    parser.add_argument('--resample', choices=['1min', '5min', '15min', '30min', '1h', '4h', 'D', 'W'], 
                      help='Resample data to different timeframe')
    parser.add_argument('--min-bars', type=int, default=1000, 
                      help='Minimum number of bars required (default: 1000)')
    parser.add_argument('--validate', action='store_true', 
                      help='Validate that all required columns for TradeEvolvePPO are present')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    return parser.parse_args()


def load_data(file_path, verbose=False):
    """Load data from CSV file."""
    if verbose:
        print(f"Loading data from {file_path}...")
    
    try:
        # First attempt - assume datetime is properly formatted
        data = pd.read_csv(file_path, parse_dates=['datetime'])
    except Exception as e:
        if verbose:
            print(f"First loading attempt failed: {e}")
        try:
            # Second attempt - try to infer datetime format
            data = pd.read_csv(file_path)
            data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    
    if verbose:
        print(f"Loaded {len(data)} rows with {len(data.columns)} columns")
    
    return data


def validate_required_columns(data, verbose=False):
    """Check if all required columns for TradeEvolvePPO are present."""
    required_columns = [
        'datetime', 'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd_line', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower',
        'ema_fast', 'ema_slow', 'atr'
    ]
    
    optional_columns = [
        'stoch_k', 'stoch_d', 'adx', 'pos_di', 'neg_di', 'vwap', 'obv'
    ]
    
    # Check for missing required columns
    missing_required = [col for col in required_columns if col not in data.columns]
    if missing_required:
        print(f"WARNING: Missing required columns: {missing_required}")
        return False
        
    # Check for missing optional columns
    missing_optional = [col for col in optional_columns if col not in data.columns]
    if missing_optional and verbose:
        print(f"INFO: Missing optional columns: {missing_optional}")
    
    # Report extra columns
    all_expected = required_columns + optional_columns
    extra_columns = [col for col in data.columns if col not in all_expected]
    if extra_columns and verbose:
        print(f"INFO: Additional columns found: {extra_columns}")
        
    return len(missing_required) == 0


def clean_data(data, args):
    """Clean and prepare the data."""
    # Ensure datetime is set as index
    if 'datetime' in data.columns:
        data.set_index('datetime', inplace=True)
    
    # Remove duplicate timestamps
    if data.index.duplicated().any():
        print(f"Found {data.index.duplicated().sum()} duplicate timestamps, keeping last values")
        data = data[~data.index.duplicated(keep='last')]
    
    # Sort by datetime
    data = data.sort_index()
    
    # Handle missing values
    if args.dropna:
        original_length = len(data)
        data = data.dropna()
        if args.verbose:
            print(f"Dropped {original_length - len(data)} rows with missing values")
    elif args.fillna:
        if args.fillna == 'ffill':
            data = data.fillna(method='ffill').fillna(method='bfill')
        elif args.fillna == 'bfill':
            data = data.fillna(method='bfill').fillna(method='ffill')
        elif args.fillna == 'zero':
            data = data.fillna(0)
        elif args.fillna == 'mean':
            data = data.fillna(data.mean())
        elif args.fillna == 'median':
            data = data.fillna(data.median())
    
    # Resample if requested
    if args.resample:
        # Define aggregation methods for different column types
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # For all other columns, use the last value
        for col in data.columns:
            if col not in agg_dict:
                agg_dict[col] = 'last'
        
        # Resample
        data = data.resample(args.resample).agg(agg_dict)
        if args.verbose:
            print(f"Resampled data to {args.resample}, new size: {len(data)} rows")
    
    # Check if we have enough data
    if len(data) < args.min_bars:
        print(f"WARNING: Dataset contains only {len(data)} bars, fewer than the minimum {args.min_bars}")
    
    return data


def normalize_features(data, verbose=False):
    """Normalize all feature columns between 0 and 1."""
    # Keep original data for reference
    original_data = data.copy()
    
    # Columns to exclude from normalization
    exclude_columns = ['datetime', 'volume']
    
    # Normalize each column
    for column in data.columns:
        if column in exclude_columns:
            continue
            
        min_val = data[column].min()
        max_val = data[column].max()
        
        # Skip if min and max are the same (no variation)
        if min_val == max_val:
            if verbose:
                print(f"Skipping normalization for {column}: no variation")
            continue
            
        # Apply normalization
        data[column] = (data[column] - min_val) / (max_val - min_val)
        
    # Special handling for volume - use log normalization
    if 'volume' in data.columns:
        data['volume'] = np.log1p(data['volume'])
        data['volume'] = (data['volume'] - data['volume'].min()) / (data['volume'].max() - data['volume'].min())
    
    if verbose:
        print("Data normalized")
    
    return data


def add_derived_features(data, verbose=False):
    """Add derived features that might be useful for the model."""
    # Ensure we have a datetime index for time-based features
    if not isinstance(data.index, pd.DatetimeIndex):
        if verbose:
            print("Cannot add time-based features: index is not datetime")
        return data
    
    # Add hour of day feature (cyclical encoding)
    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
    
    # Add day of week feature (cyclical encoding)
    data['day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
    data['day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
    
    # Add previous close to current open ratio
    data['prev_close_to_open'] = data['open'] / data['close'].shift(1)
    
    # Add average directional movement index indicator
    if all(col in data.columns for col in ['adx', 'pos_di', 'neg_di']):
        data['adx_diff'] = data['pos_di'] - data['neg_di']
    
    # Add candlestick pattern features
    data['body_size'] = abs(data['close'] - data['open'])
    data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
    data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
    
    if verbose:
        print("Added derived features")
    
    return data


def main():
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Load data from CSV
    data = load_data(args.input, args.verbose)
    
    # Validate required columns if requested
    if args.validate:
        if not validate_required_columns(data, args.verbose):
            print("WARNING: Missing required columns, proceeding anyway")
    
    # Clean data
    data = clean_data(data, args)
    
    # Add derived features
    data = add_derived_features(data, args.verbose)
    
    # Normalize features if requested
    if args.normalize:
        data = normalize_features(data, args.verbose)
    
    # Reset index to make datetime a column again
    data = data.reset_index()
    
    # Save processed data
    try:
        data.to_csv(args.output, index=False)
        print(f"Processed data saved to {args.output}")
        print(f"Final dataset: {len(data)} rows, {len(data.columns)} columns")
    except Exception as e:
        print(f"Error saving data: {e}")


if __name__ == "__main__":
    main()