import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import logging
import json
import os

class CryptoDataset(Dataset):
    def __init__(self, data, sequence_length=60):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + self.sequence_length]
        return sequence, target

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class MarketState:
    def __init__(self):
        self.position = None  # 'long', 'short', or None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.position_size = None

class CryptoTrader:
    def __init__(self, exchange_id='binance', symbol='BTC/USDT', timeframe='1m', train_mode=True):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}  # Use futures market
        })
        self.symbol = symbol
        self.timeframe = timeframe
        self.market_state = MarketState()

        # Model parameters
        self.sequence_length = 60
        self.hidden_size = 64
        self.batch_size = 32
        self.learning_rate = 0.001

        # Initialize model
        self.model = LSTM(input_size=5, hidden_size=self.hidden_size)  # 5 features: OHLCV
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Data processing
        self.scaler = MinMaxScaler()

        # Setup logging
        logging.basicConfig(
            filename=f'crypto_trader_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Load existing model if available
        self.model_path = f'models/{symbol.replace("/", "_")}_{timeframe}_model.pth'
        if os.path.exists(self.model_path):
            self.load_model()

    def fetch_data(self, limit=1000):
        """Fetch OHLCV data from exchange"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return None

    def preprocess_data(self, df):
        """Preprocess data for model training"""
        features = df[['open', 'high', 'low', 'close', 'volume']].values
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features

    def train_model(self, data):
        """Train the model on new data"""
        dataset = CryptoDataset(data, self.sequence_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(5):  # Quick update with new data
            total_loss = 0
            for sequences, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logging.info(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

    def predict(self, sequence):
        """Make price prediction"""
        self.model.eval()
        with torch.no_grad():
            sequence = torch.FloatTensor(sequence).unsqueeze(0)
            prediction = self.model(sequence)
            return prediction.item()

    def generate_signals(self, prediction, current_price):
        """Generate trading signals based on predictions"""
        prediction = self.scaler.inverse_transform([[prediction]])[0][0]

        # Simple strategy: Buy if predicted price is higher, Sell if lower
        if prediction > current_price * 1.01:  # 1% threshold
            return 'buy'
        elif prediction < current_price * 0.99:
            return 'sell'
        return 'hold'

    def execute_trade(self, signal):
        """Execute trades based on signals"""
        try:
            if signal == 'buy' and self.market_state.position != 'long':
                # Close any existing short position
                if self.market_state.position == 'short':
                    self.exchange.create_market_buy_order(
                        self.symbol,
                        self.market_state.position_size
                    )

                # Open long position
                price = self.exchange.fetch_ticker(self.symbol)['last']
                amount = self.calculate_position_size(price)
                order = self.exchange.create_market_buy_order(
                    self.symbol,
                    amount
                )

                self.market_state.position = 'long'
                self.market_state.entry_price = price
                self.market_state.position_size = amount
                self.market_state.stop_loss = price * 0.98  # 2% stop loss
                self.market_state.take_profit = price * 1.03  # 3% take profit

                logging.info(f"Opened long position: {order}")

            elif signal == 'sell' and self.market_state.position != 'short':
                # Close any existing long position
                if self.market_state.position == 'long':
                    self.exchange.create_market_sell_order(
                        self.symbol,
                        self.market_state.position_size
                    )

                # Open short position
                price = self.exchange.fetch_ticker(self.symbol)['last']
                amount = self.calculate_position_size(price)
                order = self.exchange.create_market_sell_order(
                    self.symbol,
                    amount
                )

                self.market_state.position = 'short'
                self.market_state.entry_price = price
                self.market_state.position_size = amount
                self.market_state.stop_loss = price * 1.02  # 2% stop loss
                self.market_state.take_profit = price * 0.97  # 3% take profit

                logging.info(f"Opened short position: {order}")

        except Exception as e:
            logging.error(f"Error executing trade: {e}")

    def calculate_position_size(self, price):
        """Calculate position size based on risk management"""
        balance = self.exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        risk_per_trade = 0.02  # 2% risk per trade

        position_size = (usdt_balance * risk_per_trade) / price
        return position_size

    def check_stop_loss_take_profit(self):
        """Check and handle stop loss and take profit conditions"""
        if self.market_state.position is None:
            return

        current_price = self.exchange.fetch_ticker(self.symbol)['last']

        if self.market_state.position == 'long':
            if current_price <= self.market_state.stop_loss:
                self.exchange.create_market_sell_order(
                    self.symbol,
                    self.market_state.position_size
                )
                logging.info(f"Stop loss triggered for long position")
                self.market_state.position = None

            elif current_price >= self.market_state.take_profit:
                self.exchange.create_market_sell_order(
                    self.symbol,
                    self.market_state.position_size
                )
                logging.info(f"Take profit triggered for long position")
                self.market_state.position = None

        elif self.market_state.position == 'short':
            if current_price >= self.market_state.stop_loss:
                self.exchange.create_market_buy_order(
                    self.symbol,
                    self.market_state.position_size
                )
                logging.info(f"Stop loss triggered for short position")
                self.market_state.position = None

            elif current_price <= self.market_state.take_profit:
                self.exchange.create_market_buy_order(
                    self.symbol,
                    self.market_state.position_size
                )
                logging.info(f"Take profit triggered for short position")
                self.market_state.position = None

    def save_model(self):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler
        }, self.model_path)
        logging.info(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load a trained model"""
        try:
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler = checkpoint['scaler']
            logging.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def collect_training_data(self, days_history=60):
        """Collect historical data for training"""
        try:
            # Calculate the number of data points needed
            limit = days_history * 24 * 60  # Assuming 1-minute timeframe
            if self.timeframe == '5m':
                limit = days_history * 24 * 12
            elif self.timeframe == '15m':
                limit = days_history * 24 * 4
            elif self.timeframe == '1h':
                limit = days_history * 24
            elif self.timeframe == '4h':
                limit = days_history * 6
            elif self.timeframe == '1d':
                limit = days_history
            
            # Fetch data
            df = self.fetch_data(limit=min(limit, 1000))  # API limits
            if df is not None:
                logging.info(f"Collected {len(df)} data points for training")
                return df
            else:
                raise ValueError("Failed to fetch training data")
        except Exception as e:
            logging.error(f"Error collecting training data: {e}")
            raise

    def train_on_historical(self, epochs=20):
        """Train model on historical data"""
        try:
            # Fetch historical data
            df = self.fetch_data(limit=1000)
            if df is None:
                raise ValueError("No data available for training")
            
            # Preprocess data
            data = self.preprocess_data(df)
            
            # Create dataset
            dataset = CryptoDataset(data, self.sequence_length)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for sequences, targets in dataloader:
                    self.optimizer.zero_grad()
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            logging.info("Training completed successfully")
            
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise

    def validate_model(self):
        """Validate model performance"""
        try:
            # Fetch validation data
            df = self.fetch_data(limit=500)
            if df is None:
                raise ValueError("No data available for validation")
            
            # Preprocess data
            data = self.preprocess_data(df)
            
            # Create dataset
            dataset = CryptoDataset(data, self.sequence_length)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            
            # Validation
            self.model.eval()
            total_loss = 0
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for sequences, targets in dataloader:
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()
                    predictions.extend(outputs.numpy().flatten())
                    actuals.extend(targets.numpy().flatten())
            
            avg_loss = total_loss / len(dataloader)
            
            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            mse = np.mean((predictions - actuals) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - actuals))
            
            metrics = {
                'avg_loss': avg_loss,
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            }
            
            logging.info(f"Validation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logging.error(f"Error during validation: {e}")
            raise

    def backtest(self, start_date, end_date, initial_balance=10000):
        """Backtest strategy on historical data"""
        try:
            # Fetch historical data for backtesting period
            df = self.fetch_data(limit=1000)
            if df is None:
                raise ValueError("No data available for backtesting")
            
            # Filter by date range
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
            
            if len(df) < self.sequence_length:
                raise ValueError("Insufficient data for backtesting period")
            
            # Initialize backtesting variables
            balance = initial_balance
            position = None
            trades = []
            equity_curve = []
            
            # Preprocess data
            data = self.preprocess_data(df)
            
            # Run backtest
            for i in range(self.sequence_length, len(data)):
                sequence = data[i-self.sequence_length:i]
                prediction = self.predict(sequence)
                current_price = df.iloc[i]['close']
                
                signal = self.generate_signals(prediction, current_price)
                
                if signal == 'buy' and position is None:
                    # Enter long position
                    position = {
                        'entry_price': current_price,
                        'size': balance * 0.95 / current_price,  # 95% of balance
                        'entry_time': df.iloc[i]['timestamp']
                    }
                elif signal == 'sell' and position is not None:
                    # Exit position
                    exit_price = current_price
                    pnl = (exit_price - position['entry_price']) * position['size']
                    balance += pnl
                    
                    trades.append({
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'return': (exit_price / position['entry_price'] - 1) * 100
                    })
                    
                    position = None
                
                # Track equity
                current_equity = balance
                if position is not None:
                    current_equity += (current_price - position['entry_price']) * position['size']
                equity_curve.append(current_equity)
            
            # Calculate performance metrics
            total_returns = (balance - initial_balance) / initial_balance * 100
            
            if len(trades) > 0:
                winning_trades = [t for t in trades if t['pnl'] > 0]
                win_rate = len(winning_trades) / len(trades) * 100
                
                returns = [t['return'] for t in trades]
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
                
                equity_curve = np.array(equity_curve)
                peak = np.maximum.accumulate(equity_curve)
                drawdown = (peak - equity_curve) / peak * 100
                max_drawdown = np.max(drawdown)
            else:
                win_rate = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            results = {
                'total_returns': total_returns,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': len(trades),
                'final_balance': balance
            }
            
            logging.info(f"Backtest results: {results}")
            return results
            
        except Exception as e:
            logging.error(f"Error during backtesting: {e}")
            raise

    def run(self):
        """Main trading loop"""
        while True:
            try:
                # Fetch new data
                df = self.fetch_data()
                if df is None:
                    time.sleep(13)
                    continue

                # Preprocess data
                data = self.preprocess_data(df)

                # Update model with new data
                self.train_model(data)

                # Make prediction
                current_sequence = data[-self.sequence_length:]
                prediction = self.predict(current_sequence)

                # Generate and execute trading signals
                current_price = df['close'].iloc[-1]
                signal = self.generate_signals(prediction, current_price)
                self.execute_trade(signal)

                # Check stop loss and take profit
                self.check_stop_loss_take_profit()

                # Save model periodically
                if datetime.now().minute == 0:  # Save every hour
                    self.save_model()

                time.sleep(60)  # Wait for next minute

            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    # Load configuration and setup logging
    from config import load_config, setup_logging

    config = load_config()
    logger = setup_logging()

    # Initialize exchange with API credentials
    exchange_config = config['exchange']

    # Initialize and run trader
    trader = CryptoTrader(
        exchange_id='binance',
        symbol=['BTC/USDT', 'ETH/USDT', 'LTC/USDT', 'XRP/USDT', 'ADA/USDT', 'DOT/USDT', 'SOL/USDT', 'SHIB/USDT', 'MATIC/USDT', 'DOGE/USDT', 'BNB/USDT', 'TRX/USDT', 'USDT/USDT'],
        timeframe='1m'
    )
    # Update trader with configuration
    trader.exchange.apiKey = exchange_config['api_key']
    trader.exchange.secret = exchange_config['secret_key']

    # Update trading parameters
    trading_config = config['trading']
    model_config = config['model']

    trader.sequence_length = model_config['sequence_length']
    trader.hidden_size = model_config['hidden_size']
    trader.batch_size = model_config['batch_size']
    trader.learning_rate = model_config['learning_rate']

    # Initialize new model with updated parameters
    trader.model = LSTM(input_size=5, hidden_size=trader.hidden_size)
    trader.optimizer = torch.optim.Adam(trader.model.parameters(), lr=trader.learning_rate)

    logger.info("Starting crypto trader with configuration loaded")
    trader.run()
