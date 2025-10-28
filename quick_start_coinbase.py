#!/usr/bin/env python3
"""
Quick Start: Coinbase Trading with Your API

This script provides a simple interface to start trading with your Coinbase API.
"""

import os
from dotenv import load_dotenv
from exchange_manager import ExchangeManager


def get_coinbase_manager():
    """
    Get a configured ExchangeManager for Coinbase using credentials from .env
    
    Returns:
        ExchangeManager instance configured for Coinbase
    """
    load_dotenv()
    
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    
    if not api_key or not api_secret:
        raise ValueError(
            "Coinbase API credentials not found. "
            "Please set COINBASE_API_KEY and COINBASE_API_SECRET in your .env file"
        )
    
    return ExchangeManager(
        exchange_id='coinbase',
        api_key=api_key,
        api_secret=api_secret,
        use_native_sdk=True
    )


def main():
    """Example usage of your Coinbase API"""
    print("Coinbase Quick Start")
    print("=" * 60)
    
    # Initialize
    print("\nInitializing Coinbase connection...")
    manager = get_coinbase_manager()
    
    info = manager.get_exchange_info()
    print(f"✓ Connected to: {info['id']}")
    print(f"✓ Using native SDK: {info['using_native_sdk']}")
    
    # Get balance
    print("\n" + "-" * 60)
    print("Your Account Balance:")
    print("-" * 60)
    
    balance = manager.fetch_balance()
    if balance:
        # Show balances for common currencies
        for currency in ['USD', 'BTC', 'ETH', 'USDT', 'SOL', 'ADA']:
            total = balance['total'].get(currency, 0)
            if total > 0:
                free = balance['free'].get(currency, 0)
                print(f"{currency:6} - Total: {total:>15} (Available: {free})")
    
    # Get BTC/USD price
    print("\n" + "-" * 60)
    print("Current Market Prices:")
    print("-" * 60)
    
    for symbol in ['BTC/USD', 'ETH/USD', 'SOL/USD']:
        try:
            ticker = manager.fetch_ticker(symbol)
            if ticker:
                price = ticker.get('last', 0)
                print(f"{symbol:10} ${price:>10,.2f}")
        except:
            pass
    
    # Get recent data
    print("\n" + "-" * 60)
    print("Recent BTC/USD Data (last 5 hours):")
    print("-" * 60)
    
    df = manager.fetch_ohlcv('BTC/USD', timeframe='1h', limit=5)
    if not df.empty:
        print(df[['open', 'high', 'low', 'close', 'volume']])
        
        # Calculate change
        price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        print(f"\nPrice change (5h): {price_change:+.2f}%")
    
    print("\n" + "=" * 60)
    print("To customize this script, edit quick_start_coinbase.py")
    print("See COINBASE_INTEGRATION.md for more examples")
    print("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. You have set COINBASE_API_KEY and COINBASE_API_SECRET in .env")
        print("2. Your API credentials are valid")
        print("3. Run: python setup_coinbase.py to verify setup")
