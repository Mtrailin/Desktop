"""
Unit tests for Coinbase Integration

Tests the coinbase_adapter and exchange_manager modules.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime

from coinbase_adapter import CoinbaseAdapter, COINBASE_SDK_AVAILABLE
from exchange_manager import ExchangeManager


class TestCoinbaseAdapter(unittest.TestCase):
    """Test cases for CoinbaseAdapter class"""
    
    def test_sdk_availability(self):
        """Test that SDK availability is detected"""
        self.assertTrue(COINBASE_SDK_AVAILABLE, 
                       "Coinbase SDK should be available after installation")
    
    def test_symbol_to_product_id_conversion(self):
        """Test symbol format conversion"""
        test_cases = [
            ('BTC/USD', 'BTC-USD'),
            ('ETH/USDT', 'ETH-USDT'),
            ('XRP/CAD', 'XRP-CAD'),
        ]
        
        for symbol, expected_product_id in test_cases:
            with self.subTest(symbol=symbol):
                product_id = CoinbaseAdapter.convert_symbol_to_product_id(symbol)
                self.assertEqual(product_id, expected_product_id)
    
    def test_product_id_to_symbol_conversion(self):
        """Test product ID to symbol conversion"""
        test_cases = [
            ('BTC-USD', 'BTC/USD'),
            ('ETH-USDT', 'ETH/USDT'),
            ('XRP-CAD', 'XRP/CAD'),
        ]
        
        for product_id, expected_symbol in test_cases:
            with self.subTest(product_id=product_id):
                symbol = CoinbaseAdapter.convert_product_id_to_symbol(product_id)
                self.assertEqual(symbol, expected_symbol)
    
    def test_timeframe_to_granularity_conversion(self):
        """Test timeframe to granularity mapping"""
        test_cases = [
            ('1m', 'ONE_MINUTE'),
            ('5m', 'FIVE_MINUTE'),
            ('15m', 'FIFTEEN_MINUTE'),
            ('1h', 'ONE_HOUR'),
            ('1d', 'ONE_DAY'),
        ]
        
        for timeframe, expected_granularity in test_cases:
            with self.subTest(timeframe=timeframe):
                granularity = CoinbaseAdapter.get_granularity_from_timeframe(timeframe)
                self.assertEqual(granularity, expected_granularity)
    
    def test_adapter_requires_credentials(self):
        """Test that adapter requires API credentials"""
        with self.assertRaises(TypeError):
            # Should fail without credentials
            CoinbaseAdapter()
    
    @patch('coinbase_adapter.RESTClient')
    def test_adapter_initialization(self, mock_rest_client):
        """Test adapter initializes correctly with credentials"""
        api_key = 'test_key'
        api_secret = 'test_secret'
        
        adapter = CoinbaseAdapter(api_key=api_key, api_secret=api_secret)
        
        # Verify RESTClient was called with correct parameters
        mock_rest_client.assert_called_once_with(
            api_key=api_key,
            api_secret=api_secret
        )
        
        self.assertIsNotNone(adapter.client)
        self.assertIsNotNone(adapter.logger)


class TestExchangeManager(unittest.TestCase):
    """Test cases for ExchangeManager class"""
    
    def test_manager_with_binance_uses_ccxt(self):
        """Test that Binance uses CCXT"""
        manager = ExchangeManager('binance', use_native_sdk=True)
        
        info = manager.get_exchange_info()
        self.assertEqual(info['id'], 'binance')
        self.assertFalse(info['using_native_sdk'])
        self.assertIsNotNone(manager.ccxt_exchange)
        self.assertIsNone(manager.adapter)
    
    def test_manager_with_coinbase_no_credentials_uses_ccxt(self):
        """Test that Coinbase without credentials uses CCXT"""
        manager = ExchangeManager('coinbase', use_native_sdk=True)
        
        info = manager.get_exchange_info()
        self.assertEqual(info['id'], 'coinbase')
        self.assertFalse(info['using_native_sdk'])
        self.assertIsNotNone(manager.ccxt_exchange)
        self.assertIsNone(manager.adapter)
    
    @patch('exchange_manager.CoinbaseAdapter')
    def test_manager_with_coinbase_credentials_uses_native(self, mock_adapter):
        """Test that Coinbase with credentials uses native SDK"""
        api_key = 'test_key'
        api_secret = 'test_secret'
        
        manager = ExchangeManager(
            'coinbase',
            api_key=api_key,
            api_secret=api_secret,
            use_native_sdk=True
        )
        
        info = manager.get_exchange_info()
        self.assertEqual(info['id'], 'coinbase')
        self.assertTrue(info['using_native_sdk'])
        self.assertIsNotNone(manager.adapter)
        self.assertIsNone(manager.ccxt_exchange)
        
        # Verify adapter was initialized with credentials
        mock_adapter.assert_called_once_with(api_key, api_secret)
    
    def test_manager_can_disable_native_sdk(self):
        """Test that native SDK can be explicitly disabled"""
        manager = ExchangeManager(
            'coinbase',
            api_key='test_key',
            api_secret='test_secret',
            use_native_sdk=False  # Explicitly disable
        )
        
        info = manager.get_exchange_info()
        self.assertFalse(info['using_native_sdk'])
        self.assertIsNone(manager.adapter)
        self.assertIsNotNone(manager.ccxt_exchange)
    
    @patch('exchange_manager.ccxt')
    def test_manager_initializes_ccxt_correctly(self, mock_ccxt):
        """Test that CCXT exchange is initialized with correct config"""
        api_key = 'test_key'
        api_secret = 'test_secret'
        
        mock_exchange = Mock()
        mock_ccxt.binance = Mock(return_value=mock_exchange)
        
        manager = ExchangeManager(
            'binance',
            api_key=api_key,
            api_secret=api_secret,
            use_native_sdk=False
        )
        
        # Verify exchange was created with credentials
        mock_ccxt.binance.assert_called_once()
        call_args = mock_ccxt.binance.call_args[0][0]
        
        self.assertTrue(call_args['enableRateLimit'])
        self.assertEqual(call_args['apiKey'], api_key)
        self.assertEqual(call_args['secret'], api_secret)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_roundtrip_symbol_conversion(self):
        """Test that symbol conversion is reversible"""
        original_symbol = 'BTC/USD'
        
        product_id = CoinbaseAdapter.convert_symbol_to_product_id(original_symbol)
        back_to_symbol = CoinbaseAdapter.convert_product_id_to_symbol(product_id)
        
        self.assertEqual(original_symbol, back_to_symbol)
    
    def test_all_timeframes_have_granularity(self):
        """Test that all common timeframes are supported"""
        timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '6h', '1d']
        
        for timeframe in timeframes:
            with self.subTest(timeframe=timeframe):
                granularity = CoinbaseAdapter.get_granularity_from_timeframe(timeframe)
                self.assertIsNotNone(granularity)
                self.assertTrue(granularity.endswith('MINUTE') or 
                              granularity.endswith('HOUR') or 
                              granularity.endswith('DAY'))


def run_tests():
    """Run all tests"""
    print("Running Coinbase Integration Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestCoinbaseAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestExchangeManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
