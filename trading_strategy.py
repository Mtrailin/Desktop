# Standard library imports
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Optional, Any

# Third-party imports
import numpy as np
from numpy.typing import NDArray
import pandas as pd

class TimeFrame(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

@dataclass
class TradingParameters:
    risk_per_trade: float  # Percentage of account to risk per trade (e.g., 0.02 for 2%)
    max_position_size: float  # Maximum position size as percentage of account (e.g., 0.1 for 10%)
    min_risk_reward: float  # Minimum risk/reward ratio (e.g., 2.0 for 2:1)
    max_correlation: float  # Maximum correlation between assets (e.g., 0.7 for 70%)
    min_volatility: float  # Minimum volatility required for trading (e.g., 0.5%)
    max_volatility: float  # Maximum volatility allowed for trading (e.g., 5.0%)
    max_drawdown: float  # Maximum drawdown allowed (e.g., 0.15 for 15%)

class SystematicTradingStrategy:
    def __init__(self, params: TradingParameters):
        self.params = params
        self.technical_indicators = TechnicalIndicators()
        self.risk_manager = RiskManager(params)
        self.position_sizer = PositionSizer(params)
        self.market_regime = MarketRegimeClassifier()
        self.signal_generator = SignalGenerator()

    def analyze_market_structure(self, data: Dict[TimeFrame, pd.DataFrame]) -> Dict[str, float]:
        """
        Analyze market structure across multiple timeframes
        """
        structure = {}

        # Analyze trend strength and direction on each timeframe
        for timeframe, df in data.items():
            # Calculate trend strength using ADX
            adx = self.technical_indicators.adx(df)
            # Calculate trend direction using multiple EMAs
            trend_direction = self.technical_indicators.trend_direction(df)

            structure[f"{timeframe.value}_trend_strength"] = adx[-1]
            structure[f"{timeframe.value}_trend_direction"] = trend_direction

        return structure

    def evaluate_volatility(self, data: pd.DataFrame) -> Tuple[float, bool]:
        """
        Calculate and evaluate market volatility
        """
        volatility = self.technical_indicators.atr_percent(data)
        is_valid = self.params.min_volatility <= volatility <= self.params.max_volatility

        return volatility, is_valid

    def generate_trading_signals(self, data: Dict[TimeFrame, pd.DataFrame]) -> Optional[Dict[str, float]]:
        """
        Generate trading signals based on multi-timeframe analysis
        """
        # Get market structure
        structure = self.analyze_market_structure(data)

        # Check market regime
        regime = self.market_regime.classify(data[TimeFrame.H4])

        # Generate signals for each timeframe
        signals = self.signal_generator.generate(data, structure, regime)

        # Check signal confluence
        if self._check_signal_confluence(signals, structure):
            # Calculate position size
            position_size = self.position_sizer.calculate(
                data[TimeFrame.H1],
                signals["entry_price"],
                signals["stop_loss"]
            )

            signals["position_size"] = position_size
            return signals

        return None

    def _check_signal_confluence(self, signals: Dict[str, float], structure: Dict[str, float]) -> bool:
        """
        Check if signals align across multiple timeframes
        """
        # Check trend alignment
        trend_aligned = all(
            structure[f"{tf.value}_trend_direction"] == signals["direction"]
            for tf in [TimeFrame.H4, TimeFrame.H1, TimeFrame.M15]
        )

        # Check momentum alignment
        momentum_aligned = all(
            structure[f"{tf.value}_trend_strength"] > 25  # Strong trend threshold
            for tf in [TimeFrame.H4, TimeFrame.H1]
        )

        return trend_aligned and momentum_aligned

class TechnicalIndicators:
    @staticmethod
    def sma(data: pd.DataFrame, period: int) -> NDArray[np.float64]:
        return data['close'].rolling(window=period).mean().to_numpy()

    @staticmethod
    def ema(data: pd.DataFrame, period: int) -> NDArray[np.float64]:
        return data['close'].ewm(span=period, adjust=False).mean().to_numpy()

    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> NDArray[np.float64]:
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr.to_numpy()

    def atr_percent(self, data: pd.DataFrame, period: int = 14) -> float:
        atr = self.atr(data, period)
        current_price = float(data['close'].iloc[-1])
        return (float(atr[-1]) / current_price) * 100.0

    @staticmethod
    def adx(data: pd.DataFrame, period: int = 14) -> NDArray[np.float64]:
        high = data['high']
        low = data['low']
        close = data['close']

        plus_dm = high.diff()
        minus_dm = low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr = pd.DataFrame({
            'tr1': high - low,
            'tr2': abs(high - close.shift(1)),
            'tr3': abs(low - close.shift(1))
        }).max(axis=1)

        atr = tr.rolling(window=period).mean()

        plus_di = 100.0 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = abs(100.0 * (minus_dm.rolling(window=period).mean() / atr))

        dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx.to_numpy()

    def trend_direction(self, data: pd.DataFrame) -> float:
        """
        Determine trend direction using multiple EMAs
        Returns: 1 for uptrend, -1 for downtrend, 0 for neutral
        """
        ema20 = self.ema(data, 20)
        ema50 = self.ema(data, 50)
        ema200 = self.ema(data, 200)

        if ema20[-1] > ema50[-1] > ema200[-1]:
            return 1
        elif ema20[-1] < ema50[-1] < ema200[-1]:
            return -1
        return 0

class RiskManager:
    def __init__(self, params: TradingParameters):
        self.params = params

    def validate_trade(self, entry: float, stop: float, target: float, position_size: float) -> bool:
        """
        Validate if trade meets risk management criteria
        """
        # Check risk-reward ratio
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr_ratio = reward / risk

        if rr_ratio < self.params.min_risk_reward:
            return False

        # Check position size limits
        if position_size > self.params.max_position_size:
            return False

        return True

class PositionSizer:
    def __init__(self, params: TradingParameters):
        self.params = params

    def calculate(self, data: pd.DataFrame, entry: float, stop: float) -> float:
        """
        Calculate position size based on risk parameters
        """
        risk_amount = self.params.risk_per_trade
        price_risk = abs(entry - stop)
        position_size = (risk_amount / price_risk) * entry

        # Ensure position size doesn't exceed maximum
        return min(position_size, self.params.max_position_size)

class MarketRegimeClassifier:
    def classify(self, data: pd.DataFrame) -> str:
        """
        Classify market regime (trending, ranging, volatile)
        """
        # Calculate volatility using ATR
        atr = TechnicalIndicators.atr(data)
        std_atr = np.std(atr[-20:])  # Volatility of volatility

        # Calculate trend strength using ADX
        adx = TechnicalIndicators.adx(data)

        if adx[-1] > 25:
            if std_atr > np.mean(atr[-20:]) * 0.1:
                return "volatile_trend"
            return "trending"
        elif std_atr > np.mean(atr[-20:]) * 0.15:
            return "volatile"
        return "ranging"

class SignalGenerator:
    def generate(self, data: Dict[TimeFrame, pd.DataFrame],
                structure: Dict[str, float],
                regime: str) -> Dict[str, float]:
        """
        Generate trading signals based on market conditions
        """
        signals = {}

        # Adjust strategy based on market regime
        if regime == "trending":
            signals = self._trend_following_strategy(data, structure)
        elif regime == "ranging":
            signals = self._range_trading_strategy(data, structure)
        elif regime == "volatile":
            signals = self._volatility_strategy(data, structure)
        elif regime == "volatile_trend":
            signals = self._volatile_trend_strategy(data, structure)

        return signals

    def _trend_following_strategy(self, data: Dict[TimeFrame, pd.DataFrame],
                                structure: Dict[str, float]) -> Dict[str, float]:
        # Implement trend following logic
        return {'direction': 0.0, 'entry_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0}

    def _range_trading_strategy(self, data: Dict[TimeFrame, pd.DataFrame],
                              structure: Dict[str, float]) -> Dict[str, float]:
        # Implement range trading logic
        return {'direction': 0.0, 'entry_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0}

    def _volatility_strategy(self, data: Dict[TimeFrame, pd.DataFrame],
                           structure: Dict[str, float]) -> Dict[str, float]:
        # Implement volatility-based strategy
        return {'direction': 0.0, 'entry_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0}

    def _volatile_trend_strategy(self, data: Dict[TimeFrame, pd.DataFrame],
                               structure: Dict[str, float]) -> Dict[str, float]:
        # Implement volatile trend strategy
        return {'direction': 0.0, 'entry_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0}
