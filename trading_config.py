# Standard library imports
from typing import Dict, List

# Local imports
from trading_strategy import TradingParameters, TimeFrame

# Default trading parameters
TRADING_PARAMS = TradingParameters(
    risk_per_trade=0.02,  # 2% risk per trade
    max_position_size=0.1,  # Maximum 10% of account in single position
    min_risk_reward=2.0,  # Minimum 2:1 reward-to-risk ratio
    max_correlation=0.7,  # Maximum 0.7 correlation between traded assets
    min_volatility=0.5,  # Minimum 0.5% ATR
    max_volatility=5.0,  # Maximum 5% ATR
    max_drawdown=0.15,  # Maximum 15% drawdown
)

# Timeframe configuration
ANALYSIS_TIMEFRAMES: List[TimeFrame] = [
    TimeFrame.M15,  # Entry timing
    TimeFrame.H1,   # Trend confirmation
    TimeFrame.H4,   # Primary trend
    TimeFrame.D1    # Market regime
]

# Technical indicator parameters
INDICATOR_PARAMS: Dict[str, Dict[str, int | float]] = {
    'trend': {
        'fast_ema': 20,
        'medium_ema': 50,
        'slow_ema': 200,
        'adx_period': 14,
        'adx_threshold': 25,
    },
    'momentum': {
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
    },
    'volatility': {
        'atr_period': 14,
        'bollinger_period': 20,
        'bollinger_std': 2.0,
    }
}

# Position sizing configuration
POSITION_SIZING: Dict[str, bool | float] = {
    'base_risk': 0.02,  # Base risk per trade (2%)
    'volatility_adjustment': True,  # Adjust position size based on volatility
    'momentum_adjustment': True,  # Adjust position size based on momentum
    'max_position_increase': 1.5,  # Maximum position size multiplier
    'min_position_decrease': 0.5,  # Minimum position size multiplier
}

# Market regime classification parameters
REGIME_PARAMS: Dict[str, int | float] = {
    'lookback_period': 20,
    'volatility_threshold': 0.15,
    'trend_threshold': 25,
    'range_threshold': 0.5,
}

# Risk management rules
RISK_RULES: Dict[str, int] = {
    'max_trades_per_direction': 3,  # Maximum number of trades in same direction
    'max_correlated_trades': 2,  # Maximum number of correlated positions
    'min_trades_spacing': 15,  # Minimum minutes between trades
    'max_daily_trades': 5,  # Maximum number of trades per day
    'required_confirmation_timeframes': 2,  # Number of timeframes that must confirm trend
}
