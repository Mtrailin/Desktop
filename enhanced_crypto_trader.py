from market_data_aggregator import MarketDataAggregator, AGGREGATOR_CONFIG, AggregatorConfig
from trading_strategy import SystematicTradingStrategy, TradingParameters
import asyncio
from typing import Dict, List, Optional, Any, TypedDict # pyright: ignore[reportUnusedImport]
import logging
from datetime import datetime
import pandas as pd
from trading_types import MarketData, AggregatedData, StrategySignal

class Position(TypedDict):
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    order_id: str

class Order(TypedDict):
    id: str
    symbol: str
    side: str
    amount: float
    price: float
    status: str

class EnhancedCryptoTrader:
    def __init__(self, trading_params: TradingParameters, config: Optional[AggregatorConfig] = None):
        self.trading_params: TradingParameters = trading_params
        self.config: AggregatorConfig = config or AGGREGATOR_CONFIG
        self.strategy: SystematicTradingStrategy = SystematicTradingStrategy(trading_params)
        self.data_aggregator: MarketDataAggregator = MarketDataAggregator(self.config)
        self.is_running: bool = False
        self.positions: Dict[str, Position] = {}
        self.loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()

        # Setup logging
        self.logger: logging.Logger = logging.getLogger('EnhancedCryptoTrader')
        self.logger.setLevel(logging.INFO)

    async def start(self):
        """Start the enhanced trading system"""
        self.is_running = True

        # Start market data aggregation
        await self.loop.run_in_executor(None, self.data_aggregator.start)

        # Start trading loop
        while self.is_running:
            try:
                # Get aggregated market data
                market_data = self.data_aggregator.get_all_data()

                # Check data source health
                health = self.data_aggregator.get_source_health()
                active_sources = sum(1 for h in health.values() if h['connected'])

                if active_sources < self.config['min_sources']:
                    self.logger.warning(f"Insufficient data sources ({active_sources} < {self.config['min_sources']})")
                    await asyncio.sleep(1)
                    continue

                # Process each symbol
                for symbol, data in market_data.items():
                    await self._process_symbol(symbol, data)

                # Short sleep to prevent CPU overuse
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(1)

    async def _process_symbol(self, symbol: str, data: AggregatedData):
        """Process trading logic for a single symbol"""
        try:
            # Get historical data for analysis
            historical_data = await self.data_aggregator.get_historical_data(
                symbol=symbol,
                timeframe='1m',
                limit=1000
            )

            if historical_data.empty:
                return

            # Prepare multi-timeframe data
            timeframes = {
                '1m': historical_data,
                '5m': self._resample_data(historical_data, '5T'),
                '15m': self._resample_data(historical_data, '15T'),
                '1h': self._resample_data(historical_data, '1H'),
                '4h': self._resample_data(historical_data, '4H'),
            }

            # Generate trading signals
            signals = self.strategy.generate_trading_signals(timeframes)

            if signals:
                await self._execute_trades(symbol, signals, data['price'])

        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {str(e)}")

    def _resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to a different timeframe"""
        resampled = df.resample(timeframe, on='timestamp').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        return resampled.dropna()

    async def _execute_trades(self, symbol: str, signals: StrategySignal, current_price: float) -> None:
        """Execute trading signals"""
        # Check if we have an existing position
        position = self.positions.get(symbol)

        if signals.get('action') == 'buy' and not position:
            # Calculate position size
            size = await self.strategy.position_sizer.calculate(
                capital=self.get_available_capital(),
                price=current_price,
                stop_loss=signals['stop_loss']
            )

            # Execute buy order
            if size > 0:
                try:
                    order = await self._place_order(
                        symbol=symbol,
                        side='buy',
                        amount=size,
                        price=current_price
                    )

                    if order:
                        self.positions[symbol] = {
                            'side': 'long',
                            'entry_price': current_price,
                            'stop_loss': signals['stop_loss'],
                            'take_profit': signals['take_profit'],
                            'size': size,
                            'order_id': order['id']
                        }

                        self.logger.info(f"Opened long position in {symbol} at {current_price}")

                except Exception as e:
                    self.logger.error(f"Error executing buy order for {symbol}: {str(e)}")

        elif signals.get('action') == 'sell' and position:
            # Close existing position
            try:
                order = await self._place_order(
                    symbol=symbol,
                    side='sell',
                    amount=position['size'],
                    price=current_price
                )

                if order:
                    del self.positions[symbol]
                    self.logger.info(f"Closed position in {symbol} at {current_price}")

            except Exception as e:
                self.logger.error(f"Error executing sell order for {symbol}: {str(e)}")

    async def _place_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[Order]:
        """Place an order on the exchange"""
        # Implement actual order placement logic here
        return None

    def get_available_capital(self) -> float:
        """Get available capital for trading"""
        # Implement actual capital calculation logic here
        return 0.0

    def stop(self) -> None:
        """Stop the trading system"""
        self.is_running = False
        self.data_aggregator.stop()

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        # Implement performance tracking logic here
        return {
            'total_pnl': 0.0,
            'roi': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
