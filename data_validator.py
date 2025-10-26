from typing import Dict, List
import pandas as pd
import numpy as np
from trading_types import (
    MarketData, ValidationError, DataValidationResult,
    validate_timeframe, validate_symbol
)

class DataValidator:
    """
    Comprehensive data validation for crypto trading system
    """

    @staticmethod
    def validate_market_data(data: Dict) -> DataValidationResult:
        """
        Validate market data structure and values

        Args:
            data: Dictionary containing market data

        Returns:
            DataValidationResult with validation status and details

        Raises:
            ValidationError: If data structure is invalid
        """
        try:
            # Required fields
            required_fields = {'price', 'volume', 'timestamp', 'exchange', 'symbol'}
            if not all(field in data for field in required_fields):
                missing = required_fields - set(data.keys())
                return DataValidationResult(
                    is_valid=False,
                    message=f"Missing required fields: {missing}",
                    data=None
                )

            # Type validation
            if not isinstance(data['price'], (int, float)):
                return DataValidationResult(
                    is_valid=False,
                    message="Price must be numeric",
                    data=None
                )

            if not isinstance(data['volume'], (int, float)):
                return DataValidationResult(
                    is_valid=False,
                    message="Volume must be numeric",
                    data=None
                )

            if not isinstance(data['timestamp'], (int, float)):
                return DataValidationResult(
                    is_valid=False,
                    message="Timestamp must be numeric",
                    data=None
                )

            # Value validation
            if data['price'] <= 0:
                return DataValidationResult(
                    is_valid=False,
                    message="Price must be positive",
                    data=None
                )

            if data['volume'] < 0:
                return DataValidationResult(
                    is_valid=False,
                    message="Volume cannot be negative",
                    data=None
                )

            if not validate_symbol(data['symbol']):
                return DataValidationResult(
                    is_valid=False,
                    message="Invalid symbol format",
                    data=None
                )

            # Create validated market data
            validated_data: MarketData = {
                'price': float(data['price']),
                'volume': float(data['volume']),
                'timestamp': float(data['timestamp']),
                'exchange': str(data['exchange']),
                'symbol': str(data['symbol'])
            }

            return DataValidationResult(
                is_valid=True,
                message=None,
                data=validated_data
            )

        except Exception as e:
            return DataValidationResult(
                is_valid=False,
                message=f"Validation error: {str(e)}",
                data=None
            )

    @staticmethod
    def validate_ohlcv_data(data: pd.DataFrame) -> bool:
        """
        Validate OHLCV DataFrame structure and values

        Args:
            data: DataFrame containing OHLCV data

        Returns:
            bool: True if valid, False otherwise

        Raises:
            ValidationError: If data structure is invalid
        """
        try:
            # Required columns
            required_columns = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
            if not all(col in data.columns for col in required_columns):
                missing = required_columns - set(data.columns)
                raise ValidationError(f"Missing required columns: {missing}")

            # Check for NaN values
            if data.isnull().any().any():
                raise ValidationError("Data contains NaN values")

            # Value validation
            if (data[['open', 'high', 'low', 'close', 'volume']] <= 0).any().any():
                raise ValidationError("OHLCV values must be positive")

            # High/Low validation
            if not all(data['high'] >= data['low']):
                raise ValidationError("High prices must be >= low prices")

            # OHLC consistency
            if not all(
                (data['high'] >= data['open']) &
                (data['high'] >= data['close']) &
                (data['low'] <= data['open']) &
                (data['low'] <= data['close'])
            ):
                raise ValidationError("OHLC values are inconsistent")

            return True

        except Exception as e:
            raise ValidationError(f"OHLCV validation error: {str(e)}")

    @staticmethod
    def validate_timeframe_data(
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Validate data for multiple timeframes

        Args:
            data: Dictionary of timeframes to DataFrames

        Returns:
            Dict[str, pd.DataFrame]: Validated data dictionary

        Raises:
            ValidationError: If any timeframe data is invalid
        """
        validated_data = {}

        for timeframe, df in data.items():
            # Validate timeframe format
            if not validate_timeframe(timeframe):
                raise ValidationError(f"Invalid timeframe format: {timeframe}")

            # Validate DataFrame
            if DataValidator.validate_ohlcv_data(df):
                validated_data[timeframe] = df

        return validated_data

    @staticmethod
    def validate_aggregated_data(data: List[MarketData]) -> List[MarketData]:
        """
        Validate a list of market data points

        Args:
            data: List of market data dictionaries

        Returns:
            List[MarketData]: List of validated market data

        Raises:
            ValidationError: If any data point is invalid
        """
        validated_data = []

        for item in data:
            result = DataValidator.validate_market_data(item)
            if result['is_valid'] and result['data']:
                validated_data.append(result['data'])
            else:
                raise ValidationError(
                    f"Invalid market data: {result['message']}"
                )

        return validated_data

    @staticmethod
    def detect_anomalies(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Detect anomalies in price data using statistical methods

        Args:
            data: DataFrame with price data
            window: Rolling window size for calculations

        Returns:
            pd.Series: Boolean series indicating anomalies
        """
        # Calculate rolling statistics
        rolling_mean = data['close'].rolling(window=window).mean()
        rolling_std = data['close'].rolling(window=window).std()

        # Z-score calculation
        z_scores = np.abs((data['close'] - rolling_mean) / rolling_std)

        # Detect anomalies (z-score > 3)
        return z_scores > 3

    @staticmethod
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare data for analysis

        Args:
            data: Raw DataFrame

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Remove duplicates
        data = data.drop_duplicates()

        # Sort by timestamp
        data = data.sort_values('timestamp')

        # Forward fill missing values
        data = data.ffill()

        # Remove outliers
        anomalies = DataValidator.detect_anomalies(data)
        data.loc[anomalies, ['open', 'high', 'low', 'close']] = np.nan
        data = data.ffill()

        return data
