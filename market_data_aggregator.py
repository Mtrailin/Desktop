# Standard library imports
import asyncio
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Union, TypeVar, TypedDict, Any

# Third-party imports
import ccxt
import numpy as np
import pandas as pd
import websockets

# Local imports
from endpoint_validator import EndpointManager
from trading_types import MarketData, AggregatedData

# Type aliases
WSConnection = TypeVar('WSConnection', bound=websockets.WebSocketClientProtocol)
Exchange = TypeVar('Exchange', bound=ccxt.Exchange)
DataFrame = pd.DataFrame

class ExchangeConfig(TypedDict):
    id: str
    api_key: Optional[str]
    api_secret: Optional[str]

class AggregatorConfig(TypedDict):
    exchanges: List[ExchangeConfig]
    symbols: List[str]
    refresh_rate: int
    buffer_size: int
    min_sources: int

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class MarketDataSource:
    def __init__(self, exchange_id: str, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.exchange_id: str = exchange_id
        self.exchange: Any = getattr(ccxt, exchange_id)({
            'apiKey': api_key or '',
            'secret': api_secret or '',
            'enableRateLimit': True,
            'timeout': 30000,
        })
        self.endpoint_manager: EndpointManager = EndpointManager()
        self.last_update: float = 0.0
        self.data_buffer: List[MarketData] = []
        self.is_running: bool = False
        self.reconnect_delay: float = 1.0  # Initial reconnect delay in seconds
        self.max_reconnect_delay: float = 300.0  # Maximum reconnect delay in seconds

        # Setup logging
        self.logger = logging.getLogger(f'MarketDataSource.{exchange_id}')

        # Setup logging
        self.logger = logging.getLogger(f'MarketDataSource.{exchange_id}')

    def _get_ws_endpoint(self) -> str:
        """Get WebSocket endpoint for the exchange"""
        ws_endpoints = {
            'binance': 'wss://stream.binance.com:9443/ws',
            'kucoin': 'wss://ws-api.kucoin.com/',
            'bybit': 'wss://stream.bybit.com/realtime',
            'huobi': 'wss://api.huobi.pro/ws',
            'okx': 'wss://ws.okx.com:8443/ws/v5/public',
        }
        return ws_endpoints.get(self.exchange_id, '')

    async def connect_websocket(self, symbols: List[str]) -> None:
        """Establish WebSocket connection with automatic failover"""
        while self.is_running:
            try:
                # Get best available endpoint
                ws_endpoint = await self.endpoint_manager.get_active_endpoint(self.exchange_id, 'ws')
                if not ws_endpoint:
                    self.logger.warning(f"No valid endpoint available for {self.exchange_id}")
                    await asyncio.sleep(self.reconnect_delay)
                    self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                    continue

                async with websockets.connect(ws_endpoint) as websocket:
                    # Subscribe to market data
                    subscribe_msg = self._create_subscribe_message(symbols)
                    await websocket.send(subscribe_msg)

                    # Reset reconnect delay on successful connection
                    self.reconnect_delay = 1

                    while self.is_running:
                        try:
                            message = await websocket.recv()
                            data = self._process_ws_message(message)
                            if data:
                                self.data_buffer.append(data)
                                self.last_update = time.time()
                        except websockets.exceptions.ConnectionClosed:
                            self.logger.warning(f"WebSocket connection closed for {self.exchange_id}")
                            break

            except Exception as e:
                self.logger.error(f"WebSocket error for {self.exchange_id}: {str(e)}")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

    def _create_subscribe_message(self, symbols: List[str]) -> str:
        """Create exchange-specific subscription message"""
        if self.exchange_id == 'binance':
            streams = [f"{s.lower().replace('/', '')}@trade" for s in symbols]
            return json.dumps({'method': 'SUBSCRIBE', 'params': streams, 'id': 1})
        elif self.exchange_id == 'kucoin':
            return json.dumps({'type': 'subscribe', 'topic': f'/market/match:{",".join(symbols)}'})
        # Add more exchange-specific subscription formats
        return ""

    def _process_ws_message(self, message: str) -> Optional[MarketData]:
        """Process WebSocket messages based on exchange format"""
        try:
            data = json.loads(message) if isinstance(message, str) else message
            if self.exchange_id == 'binance':
                if 'e' in data and data['e'] == 'trade':
                    return MarketData(
                        exchange=self.exchange_id,
                        symbol=data['s'],
                        price=float(data['p']),
                        volume=float(data['q']),
                        timestamp=float(data['T'])
                    )
            # Add more exchange-specific message processing
            return None
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            return None

class MarketDataAggregator:
    def __init__(self, config: AggregatorConfig):
        self.sources: List[MarketDataSource] = []
        self.symbols: List[str] = config['symbols']
        self.refresh_rate: int = config.get('refresh_rate', 100)  # ms
        self.buffer_size: int = config.get('buffer_size', 1000)
        self.min_sources: int = config.get('min_sources', 2)  # Minimum number of sources required for aggregation

        # Initialize data structures
        self.price_buffer: Dict[str, List[float]] = defaultdict(list)
        self.volume_buffer: Dict[str, List[float]] = defaultdict(list)
        self.last_aggregation: Dict[str, float] = defaultdict(float)

        # Create default empty AggregatedData
        def create_empty_data() -> AggregatedData:
            return AggregatedData(
                price=0.0,
                volume=0.0,
                timestamp=0.0,
                sources=0
            )

        self.aggregated_data: Dict[str, AggregatedData] = defaultdict(create_empty_data)

        # Initialize exchange connections
        self._initialize_sources(config['exchanges'])

        # Setup threading and async event loop
        self.executor = ThreadPoolExecutor(max_workers=len(self.sources) + 2)
        self.loop = asyncio.new_event_loop()
        self.is_running = False

    def _initialize_sources(self, exchange_configs: List[ExchangeConfig]) -> None:
        """Initialize market data sources"""
        for config in exchange_configs:
            source = MarketDataSource(
                exchange_id=config['id'],
                api_key=config.get('api_key', ''),
                api_secret=config.get('api_secret', '')
            )
            self.sources.append(source)

    async def _aggregate_data(self):
        """Aggregate data from multiple sources"""
        while self.is_running:
            try:
                current_time = time.time()

                # Process each symbol
                for symbol in self.symbols:
                    if current_time - self.last_aggregation[symbol] < self.refresh_rate / 1000:
                        continue

                    # Get data from buffers
                    prices = self.price_buffer[symbol]
                    volumes = self.volume_buffer[symbol]

                    if len(prices) >= self.min_sources:
                        # Calculate VWAP and aggregate data
                        vwap = np.average(prices, weights=volumes)
                        total_volume = sum(volumes)

                        self.aggregated_data[symbol] = AggregatedData(
                            price=vwap,
                            volume=total_volume,
                            timestamp=current_time,
                            sources=len(prices)
                        )

                        # Clear buffers
                        self.price_buffer[symbol].clear()
                        self.volume_buffer[symbol].clear()
                        self.last_aggregation[symbol] = current_time

                await asyncio.sleep(self.refresh_rate / 1000)

            except Exception as e:
                logging.error(f"Aggregation error: {str(e)}")
                await asyncio.sleep(1)

    def start(self):
        """Start the market data aggregation"""
        self.is_running = True

        # Start WebSocket connections for each source
        for source in self.sources:
            source.is_running = True
            self.loop.create_task(source.connect_websocket(self.symbols))

        # Start data aggregation
        self.loop.create_task(self._aggregate_data())

        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the market data aggregation"""
        self.is_running = False
        for source in self.sources:
            source.is_running = False
        self.loop.stop()
        self.executor.shutdown(wait=False)

    def get_aggregated_data(self, symbol: str) -> Optional[AggregatedData]:
        """Get the latest aggregated data for a symbol"""
        return self.aggregated_data.get(symbol)

    def get_all_data(self) -> Dict[str, AggregatedData]:
        """Get all aggregated data"""
        return dict(self.aggregated_data)

    def get_source_health(self) -> Dict[str, Dict[str, Union[bool, float, int]]]:
        """Get health metrics for each data source"""
        health = {}
        current_time = time.time()

        for source in self.sources:
            health[source.exchange_id] = {
                'connected': source.is_running,
                'last_update': round(current_time - source.last_update, 2),
                'buffer_size': len(source.data_buffer),
            }

        return health

    async def get_historical_data(self,
                                symbol: str,
                                timeframe: str = '1m',
                                limit: int = 1000) -> DataFrame:
        """Get historical data from multiple sources and aggregate"""
        dfs: List[pd.DataFrame] = []

        # Gather data from all sources
        for source in self.sources:
            try:
                ohlcv: List[List[float]] = await self.loop.run_in_executor(
                    self.executor,
                    lambda: source.exchange.fetch_ohlcv(symbol, timeframe, limit)
                )

                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['exchange'] = source.exchange_id
                    dfs.append(df)

            except Exception as e:
                logging.warning(f"Error fetching historical data from {source.exchange_id}: {str(e)}")

        if not dfs:
            return pd.DataFrame()

        # Combine and aggregate data
        combined = pd.concat(dfs)
        grouped = combined.groupby('timestamp').agg({
            'open': 'mean',
            'high': 'max',
            'low': 'min',
            'close': 'mean',
            'volume': 'sum'
        }).reset_index()

        return grouped

# Default aggregator configuration using alternative exchanges
AGGREGATOR_CONFIG: AggregatorConfig = {
    'exchanges': [
        {'id': 'kucoin', 'api_key': None, 'api_secret': None},     # Primary exchange
        {'id': 'bybit', 'api_key': None, 'api_secret': None},      # Primary exchange
        {'id': 'mexc', 'api_key': None, 'api_secret': None},       # Secondary exchange
        {'id': 'gate', 'api_key': None, 'api_secret': None},       # Secondary exchange
        {'id': 'huobi', 'api_key': None, 'api_secret': None}       # Backup exchange
    ],
    'symbols': [
        'BTC/USDT', 'ETH/USDT', 'XRP/USDT',
        'ADA/USDT', 'DOGE/USDT', 'SOL/USDT',
        'MATIC/USDT', 'LINK/USDT', 'LTC/USDT'
    ],
    'refresh_rate': 100,  # Refresh every 100ms
    'buffer_size': 1000,
    'min_sources': 2  # Minimum number of sources required for aggregation
}
