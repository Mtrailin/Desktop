#!/usr/bin/env python3
"""
Example: Using Coinbase Advanced Trade SDK Integration

This script demonstrates how to use the Coinbase integration in the trading system.
"""

import os
from exchange_manager import ExchangeManager
from coinbase_adapter import CoinbaseAdapter, COINBASE_SDK_AVAILABLE


def main():
    print("=" * 60)
    print("Coinbase Advanced Trade SDK Integration Example")
    print("=" * 60)
    
    # Check if SDK is available
    if COINBASE_SDK_AVAILABLE:
        print("✓ Coinbase Advanced Trade SDK is installed")
    else:
        print("✗ Coinbase Advanced Trade SDK is NOT installed")
        print("  Install it with: pip install coinbase-advanced-py")
        return
    
    # Check for API credentials
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    
    if not api_key or not api_secret:
        print("\n⚠ Warning: No API credentials found in environment variables")
        print("  Set COINBASE_API_KEY and COINBASE_API_SECRET to use authenticated features")
        print("\nRunning in read-only mode with CCXT fallback...")
        
        # Use CCXT fallback for public data
        manager = ExchangeManager('coinbase', use_native_sdk=False)
        
        print("\n" + "-" * 60)
        print("Example 1: Fetching Public Market Data (via CCXT)")
        print("-" * 60)
        
        try:
            # Fetch recent OHLCV data
            print("\nFetching BTC/USD hourly candles (last 5 hours)...")
            df = manager.fetch_ohlcv('BTC/USD', timeframe='1h', limit=5)
            
            if not df.empty:
                print(f"\nReceived {len(df)} candles:")
                print(df[['open', 'high', 'low', 'close', 'volume']])
                
                # Calculate some basic stats
                avg_price = df['close'].mean()
                print(f"\nAverage closing price: ${avg_price:,.2f}")
            else:
                print("No data received")
        except Exception as e:
            print(f"Error fetching data: {e}")
        
        print("\n" + "-" * 60)
        print("Example 2: Getting Ticker Information")
        print("-" * 60)
        
        try:
            ticker = manager.fetch_ticker('BTC/USD')
            if ticker:
                print(f"\nBTC/USD Ticker:")
                print(f"  Last Price: ${ticker.get('last', 0):,.2f}")
                print(f"  24h Volume: {ticker.get('volume', 0):,.2f}")
        except Exception as e:
            print(f"Error fetching ticker: {e}")
        
    else:
        # Full demo with authentication
        print("✓ API credentials found")
        
        print("\n" + "-" * 60)
        print("Using Native Coinbase Advanced Trade SDK")
        print("-" * 60)
        
        try:
            # Initialize with native SDK
            manager = ExchangeManager(
                'coinbase',
                api_key=api_key,
                api_secret=api_secret,
                use_native_sdk=True
            )
            
            print(f"\nExchange Info: {manager.get_exchange_info()}")
            
            # Example 1: Get account balance
            print("\n" + "-" * 60)
            print("Example 1: Account Balance")
            print("-" * 60)
            
            balance = manager.fetch_balance()
            if balance:
                print("\nAccount Balances:")
                for currency in ['USD', 'BTC', 'ETH', 'USDT']:
                    total = balance['total'].get(currency, 0)
                    if total > 0:
                        print(f"  {currency}: {total}")
            
            # Example 2: Get market data
            print("\n" + "-" * 60)
            print("Example 2: Historical Market Data (Native SDK)")
            print("-" * 60)
            
            print("\nFetching BTC/USD hourly candles (last 10 hours)...")
            df = manager.fetch_ohlcv('BTC/USD', timeframe='1h', limit=10)
            
            if not df.empty:
                print(f"\nReceived {len(df)} candles:")
                print(df[['open', 'high', 'low', 'close', 'volume']].tail())
                
                # Calculate volatility
                price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
                print(f"\nPrice change over period: {price_change:+.2f}%")
            
            # Example 3: List open orders (if any)
            print("\n" + "-" * 60)
            print("Example 3: Open Orders")
            print("-" * 60)
            
            orders = manager.fetch_orders(status='open')
            if orders:
                print(f"\nFound {len(orders)} open orders")
                for order in orders[:5]:  # Show first 5
                    print(f"  {order.get('product_id', 'N/A')}: {order.get('side', 'N/A')} "
                          f"{order.get('size', 'N/A')} @ {order.get('price', 'N/A')}")
            else:
                print("\nNo open orders")
            
            # Example 4: Get recent fills
            print("\n" + "-" * 60)
            print("Example 4: Recent Trades (Fills)")
            print("-" * 60)
            
            fills = manager.adapter.get_fills()
            if fills:
                print(f"\nFound {len(fills)} recent fills")
                for fill in fills[:5]:  # Show first 5
                    print(f"  {fill.get('product_id', 'N/A')}: {fill.get('side', 'N/A')} "
                          f"{fill.get('size', 'N/A')} @ ${fill.get('price', 'N/A')}")
            else:
                print("\nNo recent fills")
                
        except Exception as e:
            print(f"\nError: {e}")
            print("Make sure your API credentials are correct and have the required permissions")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    
    print("\nNext Steps:")
    print("1. Set up API credentials in .env file")
    print("2. See COINBASE_INTEGRATION.md for detailed documentation")
    print("3. Use ExchangeManager in your trading strategies")
    print("4. Explore the crypto_trader_gui.py for a full GUI interface")


if __name__ == '__main__':
    main()
