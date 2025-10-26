# Example script showing how to use the crypto trader in training mode
from crypto_trader import CryptoTrader
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('crypto_trainer')

def train_trader():
    # Initialize trader in training mode
    trader = CryptoTrader(
        exchange_id='binance',
        symbol=['BTC/USDT', 'ETH/USDT', 'LTC/USDT', 'XRP/USDT', 'BCH/USDT', 'DOGE/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 'SHIB/USDT', 'TRX/USDT'],
        train_mode=True
    )

    try:
        # Step 1: Collect historical data
        logger.info("Collecting historical data...")
        trader.collect_training_data(days_history=60)  # Get 60 days of historical data

        # Step 2: Train on historical data
        logger.info("Starting training on historical data...")
        trader.train_on_historical(epochs=20)

        # Step 3: Validate model performance
        logger.info("Running validation...")
        validation_metrics = trader.validate_model()
        logger.info(f"Validation metrics: {validation_metrics}")

        # Step 4: Save the trained model
        logger.info("Saving model...")
        trader.save_model()

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

def backtest_strategy():
    # Initialize trader in training mode for backtesting
    trader = CryptoTrader(
        exchange_id='binance',
        symbol=['BTC/USDT', 'ETH/USDT', 'LTC/USDT', 'XRP/USDT', 'BCH/USDT', 'DOGE/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 'SHIB/USDT', 'TRX/USDT'],
        timeframe='1h',
        train_mode=True
    )

    try:
        # Load the trained model
        trader.load_model()

        # Run backtesting
        logger.info("Running backtesting...")
        results = trader.backtest(
            start_date='2025-09-01',
            end_date='2025-10-16',
            initial_balance=10000  # USDT
        )

        # Print backtesting results
        logger.info("Backtesting Results:")
        logger.info(f"Total Returns: {results['total_returns']:.2f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        logger.info(f"Win Rate: {results['win_rate']:.2f}%")

    except Exception as e:
        logger.error(f"Error during backtesting: {e}")
        raise

if __name__ == "__main__":
    # First train the model
    train_trader()

    # Then backtest the strategy
    backtest_strategy()
