# Coinbase Advanced Trade Integration Guide

## Overview

This trading system now integrates the official **Coinbase Advanced Trade Python SDK** (`coinbase-advanced-py`), providing native support for Coinbase's trading API alongside the existing CCXT library support.

## Features

### Native Coinbase SDK Benefits

- **Official SDK**: Uses Coinbase's official Python SDK for better reliability and support
- **Advanced Features**: Access to all Coinbase Advanced Trade features
- **Better Performance**: Direct API integration optimized for Coinbase
- **WebSocket Support**: Real-time market data and order updates
- **Automatic Failover**: Falls back to CCXT if native SDK is unavailable

### Supported Operations

- Market and limit order execution
- Historical candlestick (OHLCV) data fetching
- Account balance management
- Order status tracking and cancellation
- Trade history retrieval
- Real-time ticker information
- Multi-currency support (USD, CAD, USDT)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `coinbase-advanced-py` - Official Coinbase Advanced Trade SDK
- All other required dependencies

### 2. Get API Credentials

1. Log in to [Coinbase Advanced Trade](https://www.coinbase.com/advanced-trade)
2. Navigate to **Settings** → **API**
3. Click **New API Key**
4. Configure permissions:
   - ✅ View account information
   - ✅ Trade
   - ✅ View market data
5. Copy your API Key and API Secret
6. **Important**: Save your API Secret securely - it's shown only once!

### 3. Configure Environment

Add your credentials to `.env` file:

```env
COINBASE_API_KEY=your_api_key_here
COINBASE_API_SECRET=your_api_secret_here
```

Or configure them in the GUI under **Exchange Settings**.

## Usage

### Using the Exchange Manager

The `ExchangeManager` class provides a unified interface that automatically uses the native Coinbase SDK when available:

```python
from exchange_manager import ExchangeManager

# Initialize with Coinbase
manager = ExchangeManager(
    exchange_id='coinbase',
    api_key='your_api_key',
    api_secret='your_api_secret',
    use_native_sdk=True  # Use native SDK (default)
)

# Fetch market data
df = manager.fetch_ohlcv('BTC/USD', timeframe='1h', limit=100)

# Create market order
order = manager.create_market_order('BTC/USD', 'buy', amount=0.001)

# Create limit order
order = manager.create_limit_order('BTC/USD', 'buy', amount=0.001, price=50000)

# Check balance
balance = manager.fetch_balance()

# Get ticker
ticker = manager.fetch_ticker('BTC/USD')
```

### Using the Coinbase Adapter Directly

For Coinbase-specific features, use the `CoinbaseAdapter` class:

```python
from coinbase_adapter import CoinbaseAdapter

# Initialize adapter
adapter = CoinbaseAdapter(api_key='your_key', api_secret='your_secret')

# Get all accounts
accounts = adapter.get_accounts()

# List all products
products = adapter.list_products()

# Get specific product info
product = adapter.get_product('BTC-USD')

# Fetch candles with specific granularity
df = adapter.get_candles(
    product_id='BTC-USD',
    start='2024-01-01T00:00:00Z',
    end='2024-01-02T00:00:00Z',
    granularity='ONE_HOUR'
)

# Get trade fills
fills = adapter.get_fills(product_id='BTC-USD')
```

### GUI Integration

The trading GUI automatically detects and uses the Coinbase SDK when Coinbase is selected as the exchange:

1. Open the trading application:
   ```bash
   python crypto_trader_gui.py
   # or
   python build_your_own_trading.py
   ```

2. Navigate to **Exchange Settings**
3. Select **Coinbase** from the dropdown
4. Enter your API credentials
5. Select trading pairs (supports USD, CAD, and USDT pairs)

The GUI will automatically use the native SDK for better performance and features.

## Supported Trading Pairs

### Fiat Pairs (USD)
- BTC/USD, ETH/USD, XRP/USD
- ADA/USD, DOGE/USD, SOL/USD
- MATIC/USD, LINK/USD, LTC/USD

### Canadian Dollar Pairs (CAD)
- BTC/CAD, ETH/CAD, XRP/CAD
- ADA/CAD, DOGE/CAD, SOL/CAD
- MATIC/CAD, LINK/CAD, LTC/CAD

### Stablecoin Pairs (USDT)
- BTC/USDT, ETH/USDT, XRP/USDT
- ADA/USDT, DOGE/USDT, SOL/USDT
- MATIC/USDT, LINK/USDT, LTC/USDT

## Timeframes

Supported timeframes for historical data:
- `1m` - 1 minute
- `5m` - 5 minutes
- `15m` - 15 minutes
- `30m` - 30 minutes
- `1h` - 1 hour
- `2h` - 2 hours
- `6h` - 6 hours
- `1d` - 1 day

## Configuration

### Exchange Configuration

Coinbase is configured in `exchange_config.py` with the following settings:

```python
{
    "id": "coinbase",
    "name": "Coinbase Advanced Trade",
    "priority": "primary",
    "has_websocket": True,
    "requires_vpn": False,
    "trading_fees": {
        "maker": 0.004,  # 0.4%
        "taker": 0.006   # 0.6%
    },
    "leverage_available": False,
    "api_rate_limit": 300  # requests per minute
}
```

### Best Exchange Selection

The system automatically selects Coinbase for USD and CAD trading pairs when available:

```python
from exchange_config import get_best_exchange

# Automatically selects Coinbase for fiat pairs
best = get_best_exchange('BTC/USD')  # Returns 'coinbase'
best = get_best_exchange('ETH/CAD')  # Returns 'coinbase'

# Uses fee/priority logic for other pairs
best = get_best_exchange('BTC/USDT')  # May return 'coinbase' or other exchanges
```

## Error Handling

The integration includes robust error handling:

```python
from coinbase_adapter import COINBASE_SDK_AVAILABLE

if not COINBASE_SDK_AVAILABLE:
    print("Coinbase SDK not installed. Using CCXT fallback.")
    # System automatically falls back to CCXT
```

All API calls include try-except blocks and logging for debugging.

## Examples

### Example 1: Fetch Historical Data

```python
from exchange_manager import ExchangeManager
from datetime import datetime, timedelta

manager = ExchangeManager('coinbase', api_key='key', api_secret='secret')

# Fetch last 24 hours of hourly data
df = manager.fetch_ohlcv('BTC/USD', timeframe='1h', limit=24)
print(df.head())
```

### Example 2: Place a Trade

```python
# Buy $100 worth of Bitcoin
order = manager.create_market_order('BTC/USD', 'buy', cost=100)

if order:
    print(f"Order placed: {order}")
```

### Example 3: Monitor Balance

```python
balance = manager.fetch_balance()

# Check BTC balance
btc_balance = balance['total'].get('BTC', 0)
print(f"Total BTC: {btc_balance}")

# Check USD balance
usd_balance = balance['total'].get('USD', 0)
print(f"Total USD: {usd_balance}")
```

### Example 4: Strategy Backtesting with Coinbase Data

```python
from market_data_collector import MarketDataCollector
from datetime import datetime, timedelta

# Note: MarketDataCollector can be extended to use ExchangeManager
# for Coinbase native SDK support

# For now, use ExchangeManager directly for Coinbase data
manager = ExchangeManager('coinbase', api_key='key', api_secret='secret')

# Collect 30 days of data
data = []
for day in range(30):
    end = datetime.now() - timedelta(days=day)
    start = end - timedelta(days=1)
    df = manager.fetch_ohlcv('BTC/USD', timeframe='1h', limit=24)
    data.append(df)

# Combine and use for backtesting
import pandas as pd
all_data = pd.concat(data)
```

## API Rate Limits

Coinbase Advanced Trade has the following rate limits:
- **300 requests per minute** for most endpoints
- Higher limits for authenticated users
- WebSocket connections for real-time data (recommended for high-frequency updates)

The system automatically handles rate limiting through the SDK.

## Troubleshooting

### SDK Not Available
```
ImportError: Coinbase Advanced Trade SDK is not installed
```
**Solution**: Run `pip install coinbase-advanced-py`

### Authentication Errors
```
Error: Invalid API credentials
```
**Solution**: 
1. Verify API key and secret are correct
2. Ensure API key has required permissions
3. Check if API key is active (not revoked)

### Product Not Found
```
Error: Product BTC/USD not found
```
**Solution**: 
1. Use correct product ID format (BTC-USD for Coinbase)
2. The adapter automatically converts BTC/USD to BTC-USD
3. Verify the trading pair is supported on Coinbase

### Rate Limit Exceeded
```
Error: Rate limit exceeded
```
**Solution**: 
1. Reduce request frequency
2. Use WebSocket for real-time data instead of polling
3. Wait and retry with exponential backoff

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for credentials
3. **Restrict API key permissions** to only what's needed
4. **Enable IP whitelisting** if available
5. **Rotate API keys** periodically
6. **Monitor API usage** for unusual activity
7. **Use testnet** for development when available

## Additional Resources

- [Coinbase Advanced Trade API Documentation](https://docs.cdp.coinbase.com/advanced-trade/docs)
- [coinbase-advanced-py GitHub Repository](https://github.com/coinbase/coinbase-advanced-py)
- [Coinbase Developer Portal](https://developers.coinbase.com/)
- [API Status Page](https://status.coinbase.com/)

## Support

For issues related to:
- **Coinbase SDK**: Check [official repository issues](https://github.com/coinbase/coinbase-advanced-py/issues)
- **Trading System**: Check this repository's issues
- **Coinbase API**: Contact [Coinbase Support](https://help.coinbase.com/)
