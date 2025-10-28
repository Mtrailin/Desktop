# Build Your Own Trading System - User Guide

## Overview

The **Build Your Own Trading System** is a comprehensive GUI application that allows you to create, test, and deploy custom cryptocurrency trading strategies with advanced multi-level settings configuration.

## Features

### 1. Strategy Builder
- **Custom Strategy Creation**: Design your own trading strategies from scratch
- **Technical Indicators**: Select from 15+ technical indicators including:
  - Moving Averages (SMA, EMA)
  - Momentum Indicators (RSI, Stochastic, Williams %R)
  - Trend Indicators (MACD, ADX, Ichimoku)
  - Volatility Indicators (Bollinger Bands, ATR)
  - Volume Indicators (OBV, VWAP)
- **Entry/Exit Conditions**: Define custom conditions for trade entry and exit
- **Strategy Types**: Choose from predefined templates or build completely custom strategies

### 2. Multi-Level Settings System

The application features a comprehensive multi-level settings system with the following categories:

#### Exchange Configuration
- Exchange selection (Binance, KuCoin, Bybit)
- API credentials management
- Trading pairs selection
- Timeframe selection
- Testnet/Live mode toggle

#### Risk Management
- Maximum position size
- Stop loss percentage
- Take profit percentage
- Trailing stop configuration
- Maximum drawdown limits
- Risk per trade settings

#### Order Execution
- Order type selection (Market, Limit)
- Slippage tolerance
- Timeout settings
- Partial fills handling
- Post-only orders

#### Advanced Settings
- Machine learning integration
- Model type selection (LSTM, etc.)
- Retraining intervals
- Feature engineering
- Portfolio optimization

### 3. Backtesting

- Historical data testing
- Configurable date ranges
- Initial capital settings
- Commission calculations
- Comprehensive results analysis
- Export functionality

### 4. Live Trading

- Real-time trading execution
- Live market monitoring
- Emergency stop functionality
- Trade logging
- Performance tracking

### 5. Performance Monitoring

- Real-time P&L tracking
- Win rate calculation
- Sharpe ratio
- Maximum drawdown
- Performance charts and visualizations

## Getting Started

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your environment:
   - Copy `.env.example` to `.env`
   - Add your exchange API credentials

### Running the Application

#### Main Build-Your-Own-Trading Application:
```bash
python build_your_own_trading.py
```

#### Alternative Trading GUIs:
```bash
# Comprehensive Trading Suite
python crypto_trading_suite_gui.py

# Simple Trader Control Panel
python crypto_trader_gui.py
```

## Usage Guide

### Creating a Strategy

1. Go to the **Strategy Builder** tab
2. Enter a strategy name
3. Select a strategy type or choose "custom"
4. Add technical indicators from the available list
5. Define entry and exit conditions
6. Save your strategy

### Configuring Settings

1. Navigate to the **Settings** tab
2. Select a category from the tree view:
   - Exchange Configuration
   - Risk Management
   - Order Execution
   - Advanced Settings
3. Modify settings as needed
4. Click "Save All Settings" to persist changes

#### Using the Advanced Settings Editor

For comprehensive multi-level settings:
1. Go to Settings menu → "All Settings"
2. Navigate through categories using the tree view
3. Edit settings in the right panel
4. Click "Save" to apply changes

### Backtesting a Strategy

1. Create or load a strategy
2. Go to the **Backtesting** tab
3. Configure:
   - Start date
   - End date
   - Initial capital
4. Click "Run Backtest"
5. Review results
6. Export results if needed

### Live Trading

⚠️ **Warning**: Live trading involves real money. Always test thoroughly first!

1. Ensure your strategy is tested
2. Configure risk management settings
3. Go to **Live Trading** tab
4. Review all settings
5. Click "Start Live Trading"
6. Monitor performance in real-time

### Emergency Stop

If you need to immediately stop all trading:
1. Click "Emergency Stop" in the Live Trading tab
2. All open positions will be closed
3. Trading will be halted

## Configuration Files

### Main Configuration
Location: `config/build_your_own_trading.json`

Contains all application settings organized by category.

### Strategy Files
Saved strategies are stored as JSON files and can be loaded/saved through the File menu.

## Settings Categories Reference

### Exchange Settings
```json
{
  "id": "binance",
  "api_key": "your_api_key",
  "secret_key": "your_secret_key",
  "testnet": true,
  "symbols": ["BTC/USDT", "ETH/USDT"],
  "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
}
```

### Risk Management
```json
{
  "max_position_size": 0.1,
  "stop_loss_percent": 2.0,
  "take_profit_percent": 3.0,
  "trailing_stop": true,
  "max_drawdown": 10.0,
  "risk_per_trade": 1.0
}
```

### Order Execution
```json
{
  "order_type": "limit",
  "slippage_tolerance": 0.1,
  "timeout": 30,
  "partial_fills": true,
  "post_only": false
}
```

## Menu Reference

### File Menu
- **New Strategy**: Create a new strategy
- **Load Strategy**: Load an existing strategy file
- **Save Strategy**: Save current strategy
- **Exit**: Close the application

### Settings Menu
- **Exchange Settings**: Configure exchange parameters
- **Risk Management**: Set risk parameters
- **Order Execution**: Configure order settings
- **Advanced Settings**: Access advanced options
- **All Settings**: Open comprehensive settings editor

### Tools Menu
- **Backtest Strategy**: Run backtest on current strategy
- **Optimize Parameters**: Parameter optimization (coming soon)
- **Validate Data**: Validate market data integrity

### Help Menu
- **Documentation**: View this documentation
- **About**: Application information

## Tips and Best Practices

1. **Always Test First**: Use testnet mode or backtesting before live trading
2. **Start Small**: Begin with small position sizes
3. **Set Stop Losses**: Always configure stop losses to limit potential losses
4. **Monitor Regularly**: Keep an eye on your trading bot's performance
5. **Keep Records**: Export backtest results and trading logs
6. **Update Regularly**: Keep your trading strategy and settings updated

## Troubleshooting

### Common Issues

**Issue**: Cannot connect to exchange
- **Solution**: Check API credentials, ensure API keys have trading permissions

**Issue**: Backtest shows no results
- **Solution**: Verify date range has available data, check strategy conditions

**Issue**: Settings not saving
- **Solution**: Check file permissions in config directory

## Support

For issues, questions, or contributions, please refer to the repository documentation.

## Disclaimer

⚠️ **Important**: Cryptocurrency trading involves substantial risk of loss. This software is provided "as is" without warranty. Use at your own risk. Always test thoroughly and never invest more than you can afford to lose.
