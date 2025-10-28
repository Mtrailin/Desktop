"""
Coinbase Advanced Trade SDK Adapter
Integrates the official Coinbase Advanced Trade Python SDK with the trading system.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import time

try:
    from coinbase.rest import RESTClient
    COINBASE_SDK_AVAILABLE = True
except ImportError:
    COINBASE_SDK_AVAILABLE = False
    logging.warning("Coinbase Advanced Trade SDK not available. Install with: pip install coinbase-advanced-py")


class CoinbaseAdapter:
    """
    Adapter class for Coinbase Advanced Trade API.
    Provides a unified interface compatible with the existing trading system.
    """
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize Coinbase adapter.
        
        Args:
            api_key: Coinbase API key
            api_secret: Coinbase API secret
        """
        if not COINBASE_SDK_AVAILABLE:
            raise ImportError(
                "Coinbase Advanced Trade SDK is not installed. "
                "Install it with: pip install coinbase-advanced-py"
            )
        
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        self.logger = logging.getLogger('coinbase_adapter')
        
    def get_accounts(self) -> List[Dict[str, Any]]:
        """
        Get all accounts.
        
        Returns:
            List of account dictionaries
        """
        try:
            response = self.client.get_accounts()
            return response.get('accounts', [])
        except Exception as e:
            self.logger.error(f"Error fetching accounts: {e}")
            return []
    
    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific product.
        
        Args:
            product_id: Product ID (e.g., 'BTC-USD')
            
        Returns:
            Product information dictionary
        """
        try:
            response = self.client.get_product(product_id)
            return response
        except Exception as e:
            self.logger.error(f"Error fetching product {product_id}: {e}")
            return None
    
    def list_products(self) -> List[Dict[str, Any]]:
        """
        List all available trading products.
        
        Returns:
            List of product dictionaries
        """
        try:
            response = self.client.get_products()
            return response.get('products', [])
        except Exception as e:
            self.logger.error(f"Error listing products: {e}")
            return []
    
    def get_candles(self, product_id: str, start: str, end: str, 
                    granularity: str = "ONE_MINUTE") -> pd.DataFrame:
        """
        Get historical candle data.
        
        Args:
            product_id: Product ID (e.g., 'BTC-USD')
            start: Start time (ISO 8601 format)
            end: End time (ISO 8601 format)
            granularity: Candle granularity (ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            response = self.client.get_candles(
                product_id=product_id,
                start=start,
                end=end,
                granularity=granularity
            )
            
            candles = response.get('candles', [])
            if not candles:
                return pd.DataFrame()
            
            # Convert to DataFrame with standard format
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['start'], unit='s')
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Select and order columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            self.logger.error(f"Error fetching candles for {product_id}: {e}")
            return pd.DataFrame()
    
    def create_market_order(self, product_id: str, side: str, 
                           size: Optional[str] = None,
                           funds: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Create a market order.
        
        Args:
            product_id: Product ID (e.g., 'BTC-USD')
            side: Order side ('BUY' or 'SELL')
            size: Amount in base currency (optional if funds specified)
            funds: Amount in quote currency (optional if size specified)
            
        Returns:
            Order response dictionary
        """
        try:
            # Build order configuration
            order_config = {
                "market_market_ioc": {
                    "quote_size" if funds else "base_size": funds or size
                }
            }
            
            response = self.client.create_order(
                client_order_id=f"order_{int(time.time() * 1000)}",
                product_id=product_id,
                side=side,
                order_configuration=order_config
            )
            
            self.logger.info(f"Created market order: {response}")
            return response
        except Exception as e:
            self.logger.error(f"Error creating market order: {e}")
            return None
    
    def create_limit_order(self, product_id: str, side: str, 
                          price: str, size: str,
                          post_only: bool = False) -> Optional[Dict[str, Any]]:
        """
        Create a limit order.
        
        Args:
            product_id: Product ID (e.g., 'BTC-USD')
            side: Order side ('BUY' or 'SELL')
            price: Limit price
            size: Order size
            post_only: Whether to use post-only mode
            
        Returns:
            Order response dictionary
        """
        try:
            order_config = {
                "limit_limit_gtc": {
                    "base_size": size,
                    "limit_price": price,
                    "post_only": post_only
                }
            }
            
            response = self.client.create_order(
                client_order_id=f"order_{int(time.time() * 1000)}",
                product_id=product_id,
                side=side,
                order_configuration=order_config
            )
            
            self.logger.info(f"Created limit order: {response}")
            return response
        except Exception as e:
            self.logger.error(f"Error creating limit order: {e}")
            return None
    
    def cancel_orders(self, order_ids: List[str]) -> Optional[Dict[str, Any]]:
        """
        Cancel multiple orders.
        
        Args:
            order_ids: List of order IDs to cancel
            
        Returns:
            Cancellation response dictionary
        """
        try:
            response = self.client.cancel_orders(order_ids=order_ids)
            self.logger.info(f"Canceled orders: {order_ids}")
            return response
        except Exception as e:
            self.logger.error(f"Error canceling orders: {e}")
            return None
    
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order details.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order details dictionary
        """
        try:
            response = self.client.get_order(order_id=order_id)
            return response.get('order', {})
        except Exception as e:
            self.logger.error(f"Error fetching order {order_id}: {e}")
            return None
    
    def list_orders(self, product_id: Optional[str] = None,
                    order_status: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List orders with optional filters.
        
        Args:
            product_id: Filter by product ID (optional)
            order_status: Filter by order status (optional)
            
        Returns:
            List of order dictionaries
        """
        try:
            response = self.client.list_orders(
                product_id=product_id,
                order_status=order_status
            )
            return response.get('orders', [])
        except Exception as e:
            self.logger.error(f"Error listing orders: {e}")
            return []
    
    def get_fills(self, order_id: Optional[str] = None,
                  product_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get fills (trade history).
        
        Args:
            order_id: Filter by order ID (optional)
            product_id: Filter by product ID (optional)
            
        Returns:
            List of fill dictionaries
        """
        try:
            response = self.client.get_fills(
                order_id=order_id,
                product_id=product_id
            )
            return response.get('fills', [])
        except Exception as e:
            self.logger.error(f"Error fetching fills: {e}")
            return []
    
    @staticmethod
    def convert_symbol_to_product_id(symbol: str) -> str:
        """
        Convert standard symbol format to Coinbase product ID.
        
        Args:
            symbol: Symbol in format 'BTC/USD' or 'BTC/USDT'
            
        Returns:
            Product ID in format 'BTC-USD' or 'BTC-USDT'
        """
        return symbol.replace('/', '-')
    
    @staticmethod
    def convert_product_id_to_symbol(product_id: str) -> str:
        """
        Convert Coinbase product ID to standard symbol format.
        
        Args:
            product_id: Product ID in format 'BTC-USD'
            
        Returns:
            Symbol in format 'BTC/USD'
        """
        return product_id.replace('-', '/')
    
    @staticmethod
    def get_granularity_from_timeframe(timeframe: str) -> str:
        """
        Convert standard timeframe to Coinbase granularity.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '1h')
            
        Returns:
            Coinbase granularity string
        """
        timeframe_map = {
            '1m': 'ONE_MINUTE',
            '5m': 'FIVE_MINUTE',
            '15m': 'FIFTEEN_MINUTE',
            '30m': 'THIRTY_MINUTE',
            '1h': 'ONE_HOUR',
            '2h': 'TWO_HOUR',
            '6h': 'SIX_HOUR',
            '1d': 'ONE_DAY'
        }
        return timeframe_map.get(timeframe, 'ONE_MINUTE')
