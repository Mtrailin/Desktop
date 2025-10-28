# Quick Start Guide - You Have Coinbase API! 🚀

Congratulations! Since you have Coinbase API credentials, you can immediately start using the advanced Coinbase integration.

## Immediate Next Steps

### 1. Add Your API Credentials

Edit the `.env` file and replace the placeholder values:

```bash
# Open .env file and update these lines:
COINBASE_API_KEY=your_actual_coinbase_api_key
COINBASE_API_SECRET=your_actual_coinbase_api_secret
```

### 2. Verify Your Setup

Run the verification script:

```bash
python setup_coinbase.py
```

This will:
- ✓ Check that the Coinbase SDK is installed
- ✓ Verify your API credentials work
- ✓ Show your account balances
- ✓ Display available trading pairs
- ✓ Fetch current market prices

### 3. Try the Quick Start

See your Coinbase data immediately:

```bash
python quick_start_coinbase.py
```

This displays:
- Your account balances (USD, BTC, ETH, etc.)
- Current prices for BTC/USD, ETH/USD, SOL/USD
- Recent hourly price data

### 4. Run the Full Example

See all features in action:

```bash
python example_coinbase_integration.py
```

## What You Can Do Now

### View Your Portfolio

```python
from exchange_manager import ExchangeManager
import os
from dotenv import load_dotenv

load_dotenv()

manager = ExchangeManager(
    'coinbase',
    api_key=os.getenv('COINBASE_API_KEY'),
    api_secret=os.getenv('COINBASE_API_SECRET')
)

# Get your balance
balance = manager.fetch_balance()
print(balance['total'])  # Shows all your balances
```

### Fetch Market Data

```python
# Get historical data
df = manager.fetch_ohlcv('BTC/USD', timeframe='1h', limit=100)
print(df)  # 100 hours of BTC/USD data

# Get current price
ticker = manager.fetch_ticker('BTC/USD')
print(f"BTC Price: ${ticker['last']}")
```

### Place Orders (Use with Caution!)

```python
# Market buy order - $10 worth of BTC
order = manager.create_market_order('BTC/USD', 'buy', cost=10)

# Limit buy order - 0.001 BTC at $50,000
order = manager.create_limit_order('BTC/USD', 'buy', amount=0.001, price=50000)

# Check order status
orders = manager.fetch_orders('BTC/USD', status='open')
```

### Use the GUI

```bash
python crypto_trader_gui.py
```

Then:
1. Go to **Exchange** tab
2. Select **coinbase** from dropdown
3. Your credentials from `.env` will be loaded automatically
4. Start trading!

## Supported Trading Pairs

With your Coinbase account, you can trade:

### USD Pairs
- BTC/USD, ETH/USD, SOL/USD
- XRP/USD, ADA/USD, DOGE/USD
- MATIC/USD, LINK/USD, LTC/USD

### CAD Pairs (if you're in Canada)
- BTC/CAD, ETH/CAD, SOL/CAD
- And more...

### USDT Pairs
- BTC/USDT, ETH/USDT, SOL/USDT
- And more...

## Important Notes

### API Permissions

Make sure your API key has these permissions:
- ✅ View account information
- ✅ View market data
- ✅ Trade (if you want to place orders)

### Security

- ✅ Your `.env` file is in `.gitignore` (credentials won't be committed)
- ✅ Never share your API secret
- ✅ Consider IP whitelisting on Coinbase for extra security

### Trading Safely

1. **Start Small**: Test with small amounts first
2. **Use Limit Orders**: More control than market orders
3. **Set Stop Losses**: Always have a risk management plan
4. **Monitor Your Bot**: Don't leave it unattended initially
5. **Check Fees**: Review [Coinbase's current fee schedule](https://help.coinbase.com/en/advanced-trade/trading-and-funding/trading-fees) (varies by volume)

## Useful Commands

```bash
# Verify setup
python setup_coinbase.py

# Quick portfolio check
python quick_start_coinbase.py

# Run full example
python example_coinbase_integration.py

# Run tests
python test_coinbase_integration.py

# Start GUI
python crypto_trader_gui.py
```

## Documentation

- `COINBASE_INTEGRATION.md` - Full integration guide
- `README.md` - Main project README
- [Coinbase API Docs](https://docs.cdp.coinbase.com/advanced-trade/docs)

## Troubleshooting

### "Invalid API credentials"
- Double-check your API key and secret in `.env`
- Ensure there are no extra spaces
- Verify the API key is active on Coinbase

### "Permission denied"
- Check that your API key has the required permissions
- Re-create the API key if needed

### "Rate limit exceeded"
- Wait a minute and try again
- The SDK handles rate limiting automatically for most operations

## Next Steps

1. ✅ Add your credentials to `.env`
2. ✅ Run `python setup_coinbase.py`
3. ✅ Try `python quick_start_coinbase.py`
4. ✅ Read `COINBASE_INTEGRATION.md` for advanced features
5. ✅ Build your own trading strategies!

Happy Trading! 📈

---

**Need Help?**
- Check `COINBASE_INTEGRATION.md` for detailed examples
- Run `python setup_coinbase.py` to diagnose issues
- Review test file: `test_coinbase_integration.py`
