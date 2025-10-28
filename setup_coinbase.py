#!/usr/bin/env python3
"""
Coinbase API Setup and Verification Script

This script helps you verify your Coinbase API credentials and test the connection.
"""

import os
import sys
from dotenv import load_dotenv
from coinbase_adapter import CoinbaseAdapter, COINBASE_SDK_AVAILABLE
from exchange_manager import ExchangeManager


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text):
    """Print a formatted section"""
    print("\n" + "-" * 70)
    print(f"  {text}")
    print("-" * 70)


def check_sdk_installed():
    """Check if Coinbase SDK is installed"""
    print_section("Step 1: Checking SDK Installation")
    
    if COINBASE_SDK_AVAILABLE:
        print("✓ Coinbase Advanced Trade SDK is installed")
        return True
    else:
        print("✗ Coinbase Advanced Trade SDK is NOT installed")
        print("\nPlease install it with:")
        print("  pip install coinbase-advanced-py")
        return False


def load_credentials():
    """Load API credentials from environment"""
    print_section("Step 2: Loading API Credentials")
    
    # Load .env file
    load_dotenv()
    
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    
    if not api_key or api_key == 'your_coinbase_api_key_here':
        print("✗ COINBASE_API_KEY not found or not set in .env file")
        return None, None
    
    if not api_secret or api_secret == 'your_coinbase_api_secret_here':
        print("✗ COINBASE_API_SECRET not found or not set in .env file")
        return None, None
    
    print(f"✓ API Key found: {api_key[:10]}...")
    print(f"✓ API Secret found: {api_secret[:10]}...")
    
    return api_key, api_secret


def test_connection(api_key, api_secret):
    """Test connection to Coinbase API"""
    print_section("Step 3: Testing API Connection")
    
    try:
        # Initialize adapter
        print("Initializing Coinbase adapter...")
        adapter = CoinbaseAdapter(api_key, api_secret)
        
        # Test 1: Get accounts
        print("\nTest 1: Fetching account information...")
        accounts = adapter.get_accounts()
        
        if accounts:
            print(f"✓ Successfully retrieved {len(accounts)} account(s)")
            print("\nAccount Summary:")
            for account in accounts[:5]:  # Show first 5 accounts
                currency = account.get('currency', 'N/A')
                available = account.get('available_balance', {}).get('value', '0')
                print(f"  {currency}: {available}")
        else:
            print("⚠ API connection successful, but no accounts returned")
            print("  This may indicate your account has no funded currencies")
        
        # Test 2: List products
        print("\nTest 2: Fetching available trading products...")
        products = adapter.list_products()
        
        if products:
            print(f"✓ Successfully retrieved {len(products)} trading product(s)")
            
            # Show some popular products
            popular = ['BTC-USD', 'ETH-USD', 'BTC-USDT', 'ETH-USDT']
            print("\nPopular Products Status:")
            for product_id in popular:
                available = any(p.get('product_id') == product_id for p in products)
                status = "✓ Available" if available else "✗ Not found"
                print(f"  {product_id}: {status}")
        else:
            print("⚠ No products found")
        
        # Test 3: Get market data
        print("\nTest 3: Fetching market data (BTC-USD)...")
        product = adapter.get_product('BTC-USD')
        
        if product:
            price = product.get('price', 'N/A')
            volume = product.get('volume_24h', 'N/A')
            print(f"✓ BTC-USD Price: ${price}")
            print(f"✓ 24h Volume: {volume}")
        else:
            print("⚠ Could not fetch BTC-USD market data")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error testing connection: {e}")
        print("\nTroubleshooting:")
        print("1. Verify your API credentials are correct")
        print("2. Check that your API key has the required permissions:")
        print("   - View account information")
        print("   - View market data")
        print("   - Trade (if you plan to trade)")
        print("3. Ensure your API key is active and not revoked")
        return False


def test_exchange_manager(api_key, api_secret):
    """Test ExchangeManager with Coinbase"""
    print_section("Step 4: Testing ExchangeManager Integration")
    
    try:
        # Initialize manager
        print("Initializing ExchangeManager with Coinbase...")
        manager = ExchangeManager(
            'coinbase',
            api_key=api_key,
            api_secret=api_secret,
            use_native_sdk=True
        )
        
        info = manager.get_exchange_info()
        print(f"✓ Exchange: {info['id']}")
        print(f"✓ Using Native SDK: {info['using_native_sdk']}")
        print(f"✓ Has Credentials: {info['has_api_credentials']}")
        
        # Test fetching ticker
        print("\nFetching BTC/USD ticker...")
        ticker = manager.fetch_ticker('BTC/USD')
        
        if ticker:
            print(f"✓ Last Price: ${ticker.get('last', 'N/A')}")
            print(f"✓ Volume: {ticker.get('volume', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error with ExchangeManager: {e}")
        return False


def print_next_steps():
    """Print next steps for the user"""
    print_section("Setup Complete! Next Steps")
    
    print("""
Your Coinbase API integration is ready to use! Here's what you can do:

1. Run the example script:
   python example_coinbase_integration.py

2. Use in your trading scripts:
   from exchange_manager import ExchangeManager
   
   manager = ExchangeManager('coinbase', api_key='...', api_secret='...')
   df = manager.fetch_ohlcv('BTC/USD', timeframe='1h', limit=100)

3. Use the GUI applications:
   python crypto_trader_gui.py
   # Select Coinbase from the exchange dropdown

4. Read the documentation:
   cat COINBASE_INTEGRATION.md

5. Run the tests:
   python test_coinbase_integration.py

For more information, see COINBASE_INTEGRATION.md
""")


def main():
    """Main setup function"""
    print_header("Coinbase API Setup and Verification")
    
    # Step 1: Check SDK
    if not check_sdk_installed():
        sys.exit(1)
    
    # Step 2: Load credentials
    api_key, api_secret = load_credentials()
    
    if not api_key or not api_secret:
        print("\n" + "=" * 70)
        print("Setup Instructions:")
        print("=" * 70)
        print("""
1. Get your Coinbase API credentials:
   - Go to https://www.coinbase.com/settings/api
   - Click "New API Key"
   - Enable required permissions:
     ✓ View account information
     ✓ View market data
     ✓ Trade (optional, for live trading)
   - Copy your API Key and API Secret

2. Add them to your .env file:
   COINBASE_API_KEY=your_actual_api_key
   COINBASE_API_SECRET=your_actual_api_secret

3. Run this script again:
   python setup_coinbase.py
""")
        sys.exit(1)
    
    # Step 3: Test connection
    if not test_connection(api_key, api_secret):
        sys.exit(1)
    
    # Step 4: Test ExchangeManager
    if not test_exchange_manager(api_key, api_secret):
        sys.exit(1)
    
    # Success!
    print_next_steps()
    
    print_header("✓ All Tests Passed!")
    print("\nYour Coinbase integration is ready to use!\n")


if __name__ == '__main__':
    main()
