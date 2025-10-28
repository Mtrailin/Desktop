# Crypto Trader Quick Start Guide

## Overview

Professional cryptocurrency trading system with support for multiple exchanges including **Binance**, **Coinbase Advanced Trade** (with official SDK integration), **KuCoin**, **Bybit**, and more.

## Setup Instructions

1. Install Dependencies:

```bash
pip install -r requirements.txt
```

2. Configure Environment:
   - Copy the `.env` file from the example
   - Add your exchange API credentials:
     - For Binance: BINANCE_API_KEY, BINANCE_SECRET_KEY
     - For Coinbase: COINBASE_API_KEY, COINBASE_API_SECRET
   - Adjust trading parameters if needed

3. Directory Structure:

```
├── crypto_trader.py      # Main trading bot
├── config.py            # Configuration handling
├── .env                # Environment variables
├── requirements.txt    # Dependencies
├── models/            # Saved model states
├── logs/             # Trading logs
└── data/             # Market data
```

4. Running the Bot:

```bash
# Main trading bot
python crypto_trader.py

# GUI interface with multiple exchange support
python crypto_trader_gui.py

# Advanced build-your-own trading system
python build_your_own_trading.py
```

## Supported Exchanges

- **Binance** - via CCXT
- **Coinbase Advanced Trade** - via official SDK (recommended) and CCXT
- **KuCoin** - via CCXT
- **Bybit** - via CCXT
- **MEXC** - via CCXT
- **Gate.io** - via CCXT
- **Huobi** - via CCXT

### Coinbase Integration

This system includes native integration with the **official Coinbase Advanced Trade Python SDK**. See [COINBASE_INTEGRATION.md](COINBASE_INTEGRATION.md) for detailed setup and usage instructions.

**Benefits of Coinbase native SDK:**
- Official API support with better reliability
- Access to advanced Coinbase features
- Support for USD, CAD, and USDT trading pairs
- Automatic failover to CCXT if SDK unavailable

## Important Notes

- The bot uses LSTM for price prediction
- Default timeframe is 1 minute
- Default trading pair is BTC/USDT
- Risk management:
  - 2% risk per trade
  - 2% stop loss
  - 3% take profit
- Model automatically saves every hour
- All trades are logged in the logs directory

## Monitoring

- Check the logs directory for detailed trading logs
- Use tensorboard for model training visualization:

```bash
tensorboard --logdir=./logs
```

## Warning

This is a live trading bot. Make sure to:

1. Start with small amounts
2. Test thoroughly in testnet first
3. Monitor the bot's performance
4. Understand the risks involved
