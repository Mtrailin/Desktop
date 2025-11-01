# Standard library imports
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Third-party imports
from dotenv import load_dotenv

def load_config() -> Dict[str, Dict[str, Any]]:
    """Load configuration from .env file and create necessary directories"""
    # Load environment variables
    load_dotenv()

    # Create configuration dictionary
    config = {
        'exchange': {
            'api_key': os.getenv('BINANCE_API_KEY'),
            'secret_key': os.getenv('BINANCE_SECRET_KEY'),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        },
        'trading': {
            'risk_per_trade': float(os.getenv('RISK_PER_TRADE', 0.02)),
            'stop_loss_percent': float(os.getenv('STOP_LOSS_PERCENT', 0.02)),
            'take_profit_percent': float(os.getenv('TAKE_PROFIT_PERCENT', 0.03)),
            'max_leverage': int(os.getenv('MAX_LEVERAGE', 5))
        },
        'model': {
            'sequence_length': int(os.getenv('SEQUENCE_LENGTH', 60)),
            'hidden_size': int(os.getenv('HIDDEN_SIZE', 64)),
            'batch_size': int(os.getenv('BATCH_SIZE', 32)),
            'learning_rate': float(os.getenv('LEARNING_RATE', 0.001))
        }
    }

    # Create necessary directories
    create_directories()

    return config

def create_directories():
    """Create necessary directories for the project"""
    directories = ['models', 'logs', 'data']
    for dir_name in directories:
        Path(dir_name).mkdir(parents=True, exist_ok=True)

def setup_logging():
    """Setup logging configuration"""
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_file = Path('logs') / f'crypto_trader_{datetime.now().strftime("%Y%m%d")}.log'

    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger('crypto_trader')
