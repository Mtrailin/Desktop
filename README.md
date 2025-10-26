# Crypto Trader Quick Start Guide

## Setup Instructions

1. Install Dependencies:

```bash
pip install -r requirements.txt
```

2. Configure Environment:
   - Copy the `.env` file from the example
   - Add your Binance API credentials:
     - BINANCE_API_KEY: Your API key
     - BINANCE_SECRET_KEY: Your API secret
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
python crypto_trader.py
```

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
