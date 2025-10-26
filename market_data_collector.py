import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import time
from pathlib import Path
import json
import logging
from typing import List, Dict, Optional, Tuple

class MarketDataCollector:
    def __init__(self, exchange_id: str = 'binance', symbols: List[str] = None,
                 timeframes: List[str] = None, data_dir: str = 'data'):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        self.timeframes = timeframes or ['1m', '5m', '15m', '1h', '4h', '1d']
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger('market_data_collector')

    def collect_historical_data(self, symbol: str, timeframe: str,
                              start_date: datetime, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Collect historical OHLCV data for a given symbol and timeframe"""
        end_date = end_date or datetime.now()

        # Calculate time parameters
        since = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)

        all_candles = []
        while since < end_timestamp:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=since,
                    limit=1000  # Maximum allowed by most exchanges
                )

                if not candles:
                    break

                all_candles.extend(candles)

                # Update since timestamp
                since = candles[-1][0] + 1

                # Rate limiting
                self.exchange.sleep(self.exchange.rateLimit)

            except Exception as e:
                self.logger.error(f"Error collecting data for {symbol} {timeframe}: {e}")
                time.sleep(10)  # Wait longer on error

        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save collected data to CSV file"""
        filename = self.data_dir / f"{symbol.replace('/', '_')}_{timeframe}.csv"
        df.to_csv(filename)
        self.logger.info(f"Saved data to {filename}")

    def load_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load saved data from CSV file"""
        filename = self.data_dir / f"{symbol.replace('/', '_')}_{timeframe}.csv"
        if filename.exists():
            return pd.read_csv(filename, index_col='timestamp', parse_dates=True)
        return None

    def update_dataset(self, symbol: str, timeframe: str):
        """Update existing dataset with new data"""
        existing_data = self.load_data(symbol, timeframe)

        if existing_data is not None:
            last_timestamp = pd.to_datetime(existing_data.index[-1])
            new_data = self.collect_historical_data(
                symbol,
                timeframe,
                start_date=last_timestamp + timedelta(minutes=1)
            )

            if not new_data.empty:
                updated_data = pd.concat([existing_data, new_data])
                self.save_data(updated_data, symbol, timeframe)
        else:
            # If no existing data, collect last 30 days
            start_date = datetime.now() - timedelta(days=30)
            data = self.collect_historical_data(symbol, timeframe, start_date)
            self.save_data(data, symbol, timeframe)

    def collect_all_markets(self, days_history: int = 30):
        """Collect data for all specified symbols and timeframes"""
        start_date = datetime.now() - timedelta(days=days_history)

        for symbol in self.symbols:
            for timeframe in self.timeframes:
                self.logger.info(f"Collecting data for {symbol} {timeframe}")
                try:
                    df = self.collect_historical_data(symbol, timeframe, start_date)
                    self.save_data(df, symbol, timeframe)
                except Exception as e:
                    self.logger.error(f"Failed to collect data for {symbol} {timeframe}: {e}")
                    continue

    def get_training_data(self, symbol: str, timeframe: str,
                         sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from collected market data"""
        df = self.load_data(symbol, timeframe)
        if df is None:
            raise ValueError(f"No data found for {symbol} {timeframe}")

        # Calculate additional features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'] = self.calculate_macd(df['close'])

        # Remove rows with NaN values
        df.dropna(inplace=True)

        # Normalize features
        features = ['open', 'high', 'low', 'close', 'volume', 'returns', 'volatility', 'rsi', 'macd']
        normalized_data = self.normalize_features(df[features])

        # Create sequences for training
        X, y = self.create_sequences(normalized_data, sequence_length)

        return X, y

    @staticmethod
    def normalize_features(data: pd.DataFrame) -> np.ndarray:
        """Normalize features using min-max scaling"""
        normalized = (data - data.min()) / (data.max() - data.min())
        return normalized.values

    @staticmethod
    def create_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length, 3])  # predict close price
        return np.array(X), np.array(y)

    @staticmethod
    def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        return exp1 - exp2

if __name__ == "__main__":
    # Example usage
    collector = MarketDataCollector()

    # Collect initial dataset
    collector.collect_all_markets(days_history=30)

    # Get training data for BTC/USDT
    X, y = collector.get_training_data('BTC/USDT', '1h')
    print(f"Training data shape: X={X.shape}, y={y.shape}")
