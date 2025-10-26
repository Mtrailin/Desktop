# Standard library imports
from datetime import datetime
from typing import Dict, List, Optional, TypedDict, Union, Protocol

# Third-party imports
import numpy as np
import pandas as pd

class MarketData(TypedDict):
    price: float
    volume: float
    timestamp: float
    exchange: str
    symbol: str

class AggregatedData(TypedDict):
    price: float
    volume: float
    timestamp: float
    sources: int

class OHLCVData(TypedDict):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class EndpointMetrics(TypedDict):
    connected: bool
    last_update: float
    latency: float
    success_rate: float
    error_rate: float

class ExchangeConfig(TypedDict):
    id: str
    api_key: Optional[str]
    api_secret: Optional[str]

class MarketDataSourceProtocol(Protocol):
    async def connect_websocket(self, symbols: List[str]) -> None:
        ...

    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> List[OHLCVData]:
        ...

    def get_health_metrics(self) -> EndpointMetrics:
        ...

class DataValidationResult(TypedDict):
    is_valid: bool
    message: Optional[str]
    data: Optional[MarketData]

class StrategySignal(TypedDict):
    action: str  # 'buy' or 'sell'
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: str
    confidence: float

class PerformanceMetrics(TypedDict):
    total_pnl: float
    open_pnl: float
    closed_pnl: float
    roi: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    successful_trades: int
    failed_trades: int
    average_profit: float
    average_loss: float
    largest_profit: float
    largest_loss: float
    average_holding_time: float

# Type aliases for commonly used types
Timeframe = str  # e.g., '1m', '5m', '1h', '1d'
Symbol = str  # e.g., 'BTC/USDT'
ExchangeId = str  # e.g., 'binance', 'kucoin'
TimeseriesData = pd.DataFrame  # DataFrame with OHLCV data
PriceData = Union[float, np.float64]
OrderSide = str  # 'buy' or 'sell'
OrderType = str  # 'market' or 'limit'
OrderStatus = str  # 'open', 'closed', 'canceled'

# Custom exceptions
class MarketDataError(Exception):
    """Raised when there's an error with market data operations"""
    pass

class ValidationError(Exception):
    """Raised when data validation fails"""
    pass

class ConnectionError(Exception):
    """Raised when connection to exchange fails"""
    pass

class OrderError(Exception):
    """Raised when order operations fail"""
    pass

class StrategyError(Exception):
    """Raised when strategy operations fail"""
    pass

# Constants
TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
ORDER_SIDES = ['buy', 'sell']
ORDER_TYPES = ['market', 'limit']
ORDER_STATUSES = ['open', 'closed', 'canceled']

# Validation functions
def validate_timeframe(timeframe: str) -> bool:
    """Validate that a timeframe string is supported"""
    return timeframe in TIMEFRAMES

def validate_order_side(side: str) -> bool:
    """Validate that an order side is supported"""
    return side in ORDER_SIDES

def validate_order_type(order_type: str) -> bool:
    """Validate that an order type is supported"""
    return order_type in ORDER_TYPES

def validate_symbol(symbol: str) -> bool:
    """Validate that a symbol string is properly formatted"""
    try:
        base, quote = symbol.split('/')
        return bool(base and quote)
    except ValueError:
        return False
