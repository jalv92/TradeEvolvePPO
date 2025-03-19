"""
Main entry point for TradeEvolvePPO.
Handles command line arguments and runs the appropriate mode.
"""

import os
import sys
import time
import importlib.util
from typing import Dict, Any

from config.config import (
    DATA_CONFIG, ENV_CONFIG, REWARD_CONFIG, 
    PPO_CONFIG, TRAINING_CONFIG, VISUALIZATION_CONFIG, LOGGING_CONFIG
)
from data.data_loader import DataLoader
from environment.trading_env import TradingEnv
from agents.ppo_agent import PPOAgent
from training.trainer import Trainer
from evaluation.backtest import Backtester
from utils.logger import setup_logger
from utils.helpers import parse_args, create_directories, format_time


def load_dynamic_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a Python file dynamically.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config = {}
    
    try:
        # Check if it's the default config file
        if config_path == 'config/config.py':
            # Use imported config
            config = {
                'data_config': DATA_CONFIG,
                'env_config': ENV_CONFIG,
                'reward_config': REWARD_CONFIG,
                'ppo_config': PPO_CONFIG,
                'training_config': TRAINING_CONFIG,
                'visualization_config': VISUALIZATION_CONFIG,
                'logging_config': LOGGING_CONFIG
            }
        else:
            # Load custom config file
            spec = importlib.util.spec_from_file_location("custom_config", config_path)
            custom_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_config)
            
            # Extract configuration variables
            for var in dir(custom_config):
                if var.endswith('_CONFIG') and not var.startswith('__'):
                    config_type = var.lower()
                    config[config_type] = getattr(custom_config, var)
        
        return config
    
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def train_mode(args):
    """
    Run training mode.
    
    Args:
        args: Command line arguments
    """
    print("=== TradeEvolvePPO Training Mode ===")
    
    # Load configuration
    config = load_dynamic_config(args.config)
    
    # Override timesteps if provided
    if args.timesteps is not None:
        config['training_config']['total_timesteps'] = args.timesteps
    
    # Setup directories
    output_dir = args.output
    create_directories([
        output_dir,
        os.path.join(output_dir, 'models'),
        os.path.join(output_dir, 'logs'),
        os.path.join(output_dir, 'plots')
    ])
    
    # Update paths in config
    config['training_config']['save_path'] = os.path.join(output_dir, 'models')
    config['training_config']['log_path'] = os.path.join(output_dir, 'logs')
    config['visualization_config']['plot_path'] = os.path.join(output_dir, 'plots')
    
    # Set device if provided
    if args.device != 'auto':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if args.device == 'cpu' else '0'
    
    # Setup logger
    logger = setup_logger(
        name='tradeevolveppo',
        log_file=os.path.join(output_dir, 'logs', 'main.log'),
        level=config['logging_config']['log_level']
    )
    
    logger.info("Starting TradeEvolvePPO in training mode")
    logger.info(f"Configuration loaded from {args.config}")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Setup training pipeline
    try:
        logger.info("Setting up training pipeline")
        trainer.setup(args.data)
        logger.info("Training pipeline setup complete")
    except Exception as e:
        logger.error(f"Error setting up training pipeline: {e}")
        sys.exit(1)
    
    # Run training
    try:
        logger.info("Starting training")
        start_time = time.time()
        
        if config['training_config'].get('progressive_rewards', False):
            results = trainer.run_progressive_training()
            logger.info("Progressive training completed")
        else:
            results = {'stages': [trainer.train()]}
            logger.info("Training completed")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {format_time(training_time)}")
        
        # Save training results
        trainer.save_training_results(results, os.path.join(output_dir, 'training_results.json'))
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        sys.exit(1)
    
    # Run evaluation on test set
    try:
        logger.info("Evaluating on test set")
        eval_metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {eval_metrics}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
    
    logger.info("Training mode completed")


def test_mode(args):
    """
    Run test mode.
    
    Args:
        args: Command line arguments
    """
    print("=== TradeEvolvePPO Test Mode ===")
    
    if args.model is None:
        print("Error: Model path must be provided in test mode")
        sys.exit(1)
    
    # Load configuration
    config = load_dynamic_config(args.config)
    
    # Setup directories
    output_dir = args.output
    create_directories([
        output_dir,
        os.path.join(output_dir, 'logs'),
        os.path.join(output_dir, 'plots')
    ])
    
    # Update paths in config
    config['visualization_config']['plot_path'] = os.path.join(output_dir, 'plots')
    
    # Set device if provided
    if args.device != 'auto':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if args.device == 'cpu' else '0'
    
    # Setup logger
    logger = setup_logger(
        name='tradeevolveppo',
        log_file=os.path.join(output_dir, 'logs', 'main.log'),
        level=config['logging_config']['log_level']
    )
    
    logger.info("Starting TradeEvolvePPO in test mode")
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Model loaded from {args.model}")
    
    # Load data
    try:
        logger.info("Loading data")
        data_loader = DataLoader(config['data_config'])
        _, _, test_data = data_loader.prepare_data(args.data)
        logger.info(f"Data loaded: Test={len(test_data)}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Create environment
    try:
        logger.info("Creating test environment")
        test_env = TradingEnv(
            data=test_data,
            config=config['env_config'],
            window_size=config['data_config'].get('sequence_length', 60),
            mode='test'
        )
        logger.info("Test environment created")
    except Exception as e:
        logger.error(f"Error creating test environment: {e}")
        sys.exit(1)
    
    # Create agent and load model
    try:
        logger.info("Creating agent and loading model")
        agent = PPOAgent(test_env, config)
        agent.load(args.model)
        logger.info("Agent created and model loaded")
    except Exception as e:
        logger.error(f"Error creating agent or loading model: {e}")
        sys.exit(1)
    
    # Run backtest
    try:
        logger.info("Running backtest")
        backtester = Backtester(test_env, agent, config)
        results = backtester.evaluate()
        logger.info(f"Backtest completed with results: {results}")
        
        # Save results
        backtester.save_results(output_dir)
        logger.info(f"Results saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error during backtest: {e}")
        sys.exit(1)
    
    logger.info("Test mode completed")


def backtest_mode(args):
    """
    Run backtest mode.
    
    Args:
        args: Command line arguments
    """
    print("=== TradeEvolvePPO Backtest Mode ===")
    
    if args.model is None:
        print("Error: Model path must be provided in backtest mode")
        sys.exit(1)
    
    # Load configuration
    config = load_dynamic_config(args.config)
    
    # Setup directories
    output_dir = args.output
    create_directories([
        output_dir,
        os.path.join(output_dir, 'logs'),
        os.path.join(output_dir, 'plots')
    ])
    
    # Update paths in config
    config['visualization_config']['plot_path'] = os.path.join(output_dir, 'plots')
    
    # Set device if provided
    if args.device != 'auto':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if args.device == 'cpu' else '0'
    
    # Setup logger
    logger = setup_logger(
        name='tradeevolveppo',
        log_file=os.path.join(output_dir, 'logs', 'main.log'),
        level=config['logging_config']['log_level']
    )
    
    logger.info("Starting TradeEvolvePPO in backtest mode")
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Model loaded from {args.model}")
    
    # Load data
    try:
        logger.info("Loading data")
        data_loader = DataLoader(config['data_config'])
        data = data_loader.load_data(args.data)
        data = data_loader.preprocess_data(data)
        logger.info(f"Data loaded: {len(data)} rows")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Create environment
    try:
        logger.info("Creating backtest environment")
        backtest_env = TradingEnv(
            data=data,
            config=config['env_config'],
            window_size=config['data_config'].get('sequence_length', 60),
            mode='backtest'
        )
        logger.info("Backtest environment created")
    except Exception as e:
        logger.error(f"Error creating backtest environment: {e}")
        sys.exit(1)
    
    # Create agent and load model
    try:
        logger.info("Creating agent and loading model")
        agent = PPOAgent(backtest_env, config)
        agent.load(args.model)
        logger.info("Agent created and model loaded")
    except Exception as e:
        logger.error(f"Error creating agent or loading model: {e}")
        sys.exit(1)
    
    # Run backtest
    try:
        logger.info("Running backtest")
        backtester = Backtester(backtest_env, agent, config)
        results = backtester.evaluate()
        logger.info(f"Backtest completed with results: {results}")
        
        # Save results
        backtester.save_results(output_dir)
        logger.info(f"Results saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error during backtest: {e}")
        sys.exit(1)
    
    logger.info("Backtest mode completed")


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Run the appropriate mode
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'test':
        test_mode(args)
    elif args.mode == 'backtest':
        backtest_mode(args)
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)


if __name__ == "__main__":
    main()