"""
Exchange Manager - Unified interface for multiple exchanges
Supports both CCXT and native Coinbase Advanced Trade SDK
"""

import logging
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime
import ccxt

from coinbase_adapter import CoinbaseAdapter, COINBASE_SDK_AVAILABLE


class ExchangeManager:
    """
    Unified exchange manager that supports both CCXT and native exchange SDKs.
    Provides a consistent interface regardless of the underlying implementation.
    """
    
    def __init__(self, exchange_id: str, api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None, use_native_sdk: bool = True):
        """
        Initialize exchange manager.
        
        Args:
            exchange_id: Exchange identifier (e.g., 'coinbase', 'binance')
            api_key: API key for authentication
            api_secret: API secret for authentication
            use_native_sdk: Whether to use native SDK when available (default: True)
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.logger = logging.getLogger('exchange_manager')
        
        # Determine which implementation to use
        self.use_native = use_native_sdk and exchange_id == 'coinbase' and COINBASE_SDK_AVAILABLE
        
        if self.use_native:
            self.logger.info(f"Using native Coinbase Advanced Trade SDK")
            if not api_key or not api_secret:
                raise ValueError("API key and secret required for Coinbase SDK")
            self.adapter = CoinbaseAdapter(api_key, api_secret)
            self.ccxt_exchange = None
        else:
            self.logger.info(f"Using CCXT for {exchange_id}")
            self.adapter = None
            self.ccxt_exchange = self._init_ccxt_exchange()
    
    def _init_ccxt_exchange(self):
        """Initialize CCXT exchange instance."""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            config = {'enableRateLimit': True}
            
            if self.api_key and self.api_secret:
                config.update({
                    'apiKey': self.api_key,
                    'secret': self.api_secret
                })
            
            return exchange_class(config)
        except Exception as e:
            self.logger.error(f"Error initializing CCXT exchange: {e}")
            raise
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', 
                    limit: int = 100, since: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            limit: Number of candles to fetch
            since: Timestamp in milliseconds (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        if self.use_native:
            # Use Coinbase native SDK
            product_id = CoinbaseAdapter.convert_symbol_to_product_id(symbol)
            granularity = CoinbaseAdapter.get_granularity_from_timeframe(timeframe)
            
            # Calculate time range
            end_time = datetime.utcnow()
            if since:
                start_time = datetime.fromtimestamp(since / 1000)
            else:
                # Calculate start time based on timeframe and limit
                minutes_map = {
                    '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                    '1h': 60, '2h': 120, '6h': 360, '1d': 1440
                }
                minutes = minutes_map.get(timeframe, 1)
                from datetime import timedelta
                start_time = end_time - timedelta(minutes=minutes * limit)
            
            df = self.adapter.get_candles(
                product_id=product_id,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                granularity=granularity
            )
            return df
        else:
            # Use CCXT
            try:
                ohlcv = self.ccxt_exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=limit
                )
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            except Exception as e:
                self.logger.error(f"Error fetching OHLCV data: {e}")
                return pd.DataFrame()
    
    def create_market_order(self, symbol: str, side: str, 
                           amount: Optional[float] = None,
                           cost: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Create a market order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            side: Order side ('buy' or 'sell')
            amount: Amount in base currency (optional if cost specified)
            cost: Amount in quote currency (optional if amount specified)
            
        Returns:
            Order response dictionary
        """
        if self.use_native:
            product_id = CoinbaseAdapter.convert_symbol_to_product_id(symbol)
            return self.adapter.create_market_order(
                product_id=product_id,
                side=side.upper(),
                size=str(amount) if amount else None,
                funds=str(cost) if cost else None
            )
        else:
            try:
                if not amount:
                    # Convert cost to amount if needed
                    ticker = self.ccxt_exchange.fetch_ticker(symbol)
                    amount = cost / ticker['last']
                
                return self.ccxt_exchange.create_market_order(symbol, side, amount)
            except Exception as e:
                self.logger.error(f"Error creating market order: {e}")
                return None
    
    def create_limit_order(self, symbol: str, side: str, 
                          amount: float, price: float,
                          post_only: bool = False) -> Optional[Dict[str, Any]]:
        """
        Create a limit order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Limit price
            post_only: Whether to use post-only mode
            
        Returns:
            Order response dictionary
        """
        if self.use_native:
            product_id = CoinbaseAdapter.convert_symbol_to_product_id(symbol)
            return self.adapter.create_limit_order(
                product_id=product_id,
                side=side.upper(),
                price=str(price),
                size=str(amount),
                post_only=post_only
            )
        else:
            try:
                params = {'postOnly': post_only} if post_only else {}
                return self.ccxt_exchange.create_limit_order(
                    symbol, side, amount, price, params
                )
            except Exception as e:
                self.logger.error(f"Error creating limit order: {e}")
                return None
    
    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol (required for some exchanges)
            
        Returns:
            Cancellation response dictionary
        """
        if self.use_native:
            return self.adapter.cancel_orders([order_id])
        else:
            try:
                return self.ccxt_exchange.cancel_order(order_id, symbol)
            except Exception as e:
                self.logger.error(f"Error canceling order: {e}")
                return None
    
    def fetch_balance(self) -> Optional[Dict[str, Any]]:
        """
        Fetch account balance.
        
        Returns:
            Balance dictionary
        """
        if self.use_native:
            accounts = self.adapter.get_accounts()
            # Convert to CCXT-like format
            balance = {'free': {}, 'used': {}, 'total': {}}
            for account in accounts:
                currency = account.get('currency', '')
                available = float(account.get('available_balance', {}).get('value', 0))
                hold = float(account.get('hold', {}).get('value', 0))
                
                balance['free'][currency] = available
                balance['used'][currency] = hold
                balance['total'][currency] = available + hold
            
            return balance
        else:
            try:
                return self.ccxt_exchange.fetch_balance()
            except Exception as e:
                self.logger.error(f"Error fetching balance: {e}")
                return None
    
    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch ticker information.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Ticker dictionary
        """
        if self.use_native:
            product_id = CoinbaseAdapter.convert_symbol_to_product_id(symbol)
            product = self.adapter.get_product(product_id)
            if product:
                # Convert to CCXT-like format
                return {
                    'symbol': symbol,
                    'last': float(product.get('price', 0)),
                    'bid': float(product.get('quote_increment', 0)),
                    'ask': float(product.get('quote_increment', 0)),
                    'volume': float(product.get('volume_24h', 0))
                }
            return None
        else:
            try:
                return self.ccxt_exchange.fetch_ticker(symbol)
            except Exception as e:
                self.logger.error(f"Error fetching ticker: {e}")
                return None
    
    def fetch_orders(self, symbol: Optional[str] = None,
                    status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch orders.
        
        Args:
            symbol: Trading pair symbol (optional)
            status: Order status filter (optional)
            
        Returns:
            List of order dictionaries
        """
        if self.use_native:
            product_id = None
            if symbol:
                product_id = CoinbaseAdapter.convert_symbol_to_product_id(symbol)
            
            order_status = None
            if status:
                # Convert to Coinbase status format
                status_map = {
                    'open': ['OPEN'],
                    'closed': ['FILLED'],
                    'canceled': ['CANCELLED']
                }
                order_status = status_map.get(status.lower())
            
            return self.adapter.list_orders(
                product_id=product_id,
                order_status=order_status
            )
        else:
            try:
                if status == 'open':
                    return self.ccxt_exchange.fetch_open_orders(symbol)
                elif status == 'closed':
                    return self.ccxt_exchange.fetch_closed_orders(symbol)
                else:
                    return self.ccxt_exchange.fetch_orders(symbol)
            except Exception as e:
                self.logger.error(f"Error fetching orders: {e}")
                return []
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information.
        
        Returns:
            Dictionary with exchange information
        """
        return {
            'id': self.exchange_id,
            'using_native_sdk': self.use_native,
            'has_api_credentials': bool(self.api_key and self.api_secret)
        }
