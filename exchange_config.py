from typing import Dict, List, TypedDict
from enum import Enum

class ExchangePriority(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    BACKUP = "backup"

class ExchangeConfig(TypedDict):
    id: str
    name: str
    priority: str
    has_websocket: bool
    requires_vpn: bool
    supported_pairs: List[str]
    trading_fees: Dict[str, float]
    withdrawal_fees: Dict[str, float]
    min_trade_amount: Dict[str, float]
    leverage_available: bool
    api_rate_limit: int  # requests per minute

# Alternative exchanges configuration
ALTERNATIVE_EXCHANGES = {
    "coinbase": ExchangeConfig(
        id="coinbase",
        name="Coinbase Advanced Trade",
        priority=ExchangePriority.PRIMARY.value,
        has_websocket=True,
        requires_vpn=False,
        supported_pairs=[
            "BTC/USD", "ETH/USD", "XRP/USD",
            "ADA/USD", "DOGE/USD", "SOL/USD",
            "MATIC/USD", "LINK/USD", "LTC/USD",
            "BTC/USDT", "ETH/USDT", "XRP/USDT",
            "ADA/USDT", "DOGE/USDT", "SOL/USDT",
            "MATIC/USDT", "LINK/USDT", "LTC/USDT",
            # Canadian CAD pairs
            "BTC/CAD", "ETH/CAD", "XRP/CAD",
            "ADA/CAD", "DOGE/CAD", "SOL/CAD",
            "MATIC/CAD", "LINK/CAD", "LTC/CAD"
        ],
        trading_fees={
            "maker": 0.004,  # 0.4% standard, can be lower with volume
            "taker": 0.006   # 0.6% standard, can be lower with volume
        },
        withdrawal_fees={
            "BTC": 0.0001,
            "ETH": 0.002,
            "USDT": 1,
            "USD": 0,  # Free for USD to US bank
            "CAD": 0,  # Free for CAD to Canadian bank
            "XRP": 0.02,
            "ADA": 1,
            "DOGE": 2,
            "SOL": 0.01,
            "MATIC": 1,
            "LINK": 0.1,
            "LTC": 0.001
        },
        min_trade_amount={
            "BTC": 0.0001,
            "ETH": 0.001,
            "USDT": 1,
            "USD": 1,
            "CAD": 1,
            "XRP": 1,
            "ADA": 5,
            "DOGE": 10,
            "SOL": 0.1,
            "MATIC": 5,
            "LINK": 0.1,
            "LTC": 0.1
        },
        leverage_available=False,  # Coinbase doesn't offer leverage
        api_rate_limit=300
    ),

    "kucoin": ExchangeConfig(
        id="kucoin",
        name="KuCoin",
        priority=ExchangePriority.PRIMARY.value,
        has_websocket=True,
        requires_vpn=False,
        supported_pairs=[
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT",
            "ADA/USDT", "DOGE/USDT", "SOL/USDT", "DOT/USDT",
            "AVAX/USDT", "MATIC/USDT", "LINK/USDT", "UNI/USDT",
            "ATOM/USDT", "LTC/USDT", "ETC/USDT"
        ],
        trading_fees={
            "maker": 0.001,  # 0.1%
            "taker": 0.001   # 0.1%
        },
        withdrawal_fees={
            "BTC": 0.0005,
            "ETH": 0.004,
            "USDT": 1,
            "XRP": 0.01,
            "ADA": 1,
            "DOGE": 2,
            "SOL": 0.01,
            "MATIC": 1,
            "LINK": 0.1,
            "LTC": 0.001
        },
        min_trade_amount={
            "BTC": 0.0001,
            "ETH": 0.001,
            "USDT": 1,
            "XRP": 1,
            "ADA": 5,
            "DOGE": 10,
            "SOL": 0.1,
            "MATIC": 5,
            "LINK": 0.1,
            "LTC": 0.1
        },
        leverage_available=True,
        api_rate_limit=600
    ),

    "bybit": ExchangeConfig(
        id="bybit",
        name="Bybit",
        priority=ExchangePriority.PRIMARY.value,
        has_websocket=True,
        requires_vpn=False,
        supported_pairs=[
            "BTC/USDT", "ETH/USDT", "XRP/USDT",
            "ADA/USDT", "DOGE/USDT", "SOL/USDT",
            "MATIC/USDT", "LINK/USDT", "UNI/USDT",
            "ATOM/USDT", "LTC/USDT"
        ],
        trading_fees={
            "maker": 0.0006,  # 0.06%
            "taker": 0.001    # 0.1%
        },
        withdrawal_fees={
            "BTC": 0.0005,
            "ETH": 0.005,
            "USDT": 2,
            "XRP": 0.02,
            "ADA": 2,
            "DOGE": 5,
            "SOL": 0.02,
            "MATIC": 2,
            "LINK": 0.2,
            "LTC": 0.002
        },
        min_trade_amount={
            "BTC": 0.0001,
            "ETH": 0.001,
            "USDT": 1,
            "XRP": 1,
            "ADA": 5,
            "DOGE": 10,
            "SOL": 0.1,
            "MATIC": 5,
            "LINK": 0.1,
            "LTC": 0.1
        },
        leverage_available=True,
        api_rate_limit=600
    ),

    "mexc": ExchangeConfig(
        id="mexc",
        name="MEXC Global",
        priority=ExchangePriority.SECONDARY.value,
        has_websocket=True,
        requires_vpn=False,
        supported_pairs=[
            "BTC/USDT", "ETH/USDT", "XRP/USDT",
            "ADA/USDT", "DOGE/USDT", "SOL/USDT",
            "MATIC/USDT", "LINK/USDT", "UNI/USDT",
            "ATOM/USDT", "LTC/USDT"
        ],
        trading_fees={
            "maker": 0.002,  # 0.2%
            "taker": 0.002   # 0.2%
        },
        withdrawal_fees={
            "BTC": 0.0004,
            "ETH": 0.003,
            "USDT": 1,
            "XRP": 0.01,
            "ADA": 1,
            "DOGE": 2,
            "SOL": 0.01,
            "MATIC": 1,
            "LINK": 0.1,
            "LTC": 0.001
        },
        min_trade_amount={
            "BTC": 0.0001,
            "ETH": 0.001,
            "USDT": 1,
            "XRP": 1,
            "ADA": 5,
            "DOGE": 10,
            "SOL": 0.1,
            "MATIC": 5,
            "LINK": 0.1,
            "LTC": 0.1
        },
        leverage_available=True,
        api_rate_limit=500
    ),

    "gate": ExchangeConfig(
        id="gate",
        name="Gate.io",
        priority=ExchangePriority.SECONDARY.value,
        has_websocket=True,
        requires_vpn=False,
        supported_pairs=[
            "BTC/USDT", "ETH/USDT", "XRP/USDT",
            "ADA/USDT", "DOGE/USDT", "SOL/USDT",
            "MATIC/USDT", "LINK/USDT", "UNI/USDT",
            "ATOM/USDT", "LTC/USDT"
        ],
        trading_fees={
            "maker": 0.002,  # 0.2%
            "taker": 0.002   # 0.2%
        },
        withdrawal_fees={
            "BTC": 0.0004,
            "ETH": 0.004,
            "USDT": 1,
            "XRP": 0.01,
            "ADA": 1,
            "DOGE": 2,
            "SOL": 0.01,
            "MATIC": 1,
            "LINK": 0.1,
            "LTC": 0.001
        },
        min_trade_amount={
            "BTC": 0.0001,
            "ETH": 0.001,
            "USDT": 1,
            "XRP": 1,
            "ADA": 5,
            "DOGE": 10,
            "SOL": 0.1,
            "MATIC": 5,
            "LINK": 0.1,
            "LTC": 0.1
        },
        leverage_available=True,
        api_rate_limit=400
    ),

    "huobi": ExchangeConfig(
        id="huobi",
        name="Huobi Global",
        priority=ExchangePriority.BACKUP.value,
        has_websocket=True,
        requires_vpn=False,
        supported_pairs=[
            "BTC/USDT", "ETH/USDT", "XRP/USDT",
            "ADA/USDT", "DOGE/USDT", "SOL/USDT",
            "MATIC/USDT", "LINK/USDT", "UNI/USDT",
            "ATOM/USDT", "LTC/USDT"
        ],
        trading_fees={
            "maker": 0.002,  # 0.2%
            "taker": 0.002   # 0.2%
        },
        withdrawal_fees={
            "BTC": 0.0005,
            "ETH": 0.004,
            "USDT": 2,
            "XRP": 0.02,
            "ADA": 2,
            "DOGE": 5,
            "SOL": 0.02,
            "MATIC": 2,
            "LINK": 0.2,
            "LTC": 0.002
        },
        min_trade_amount={
            "BTC": 0.0001,
            "ETH": 0.001,
            "USDT": 1,
            "XRP": 1,
            "ADA": 5,
            "DOGE": 10,
            "SOL": 0.1,
            "MATIC": 5,
            "LINK": 0.1,
            "LTC": 0.1
        },
        leverage_available=True,
        api_rate_limit=300
    )
}

# Default trading pairs available across most exchanges
DEFAULT_TRADING_PAIRS = [
    "BTC/USDT", "ETH/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "SOL/USDT",
    "MATIC/USDT", "LINK/USDT", "LTC/USDT"
]

# Exchange selection helper functions
def get_available_exchanges() -> List[str]:
    """Get list of all available exchanges"""
    return list(ALTERNATIVE_EXCHANGES.keys())

def get_primary_exchanges() -> List[str]:
    """Get list of primary exchanges"""
    return [
        exchange_id for exchange_id, config in ALTERNATIVE_EXCHANGES.items()
        if config["priority"] == ExchangePriority.PRIMARY.value
    ]

def get_exchange_for_pair(pair: str) -> List[str]:
    """Get list of exchanges that support a specific trading pair"""
    return [
        exchange_id for exchange_id, config in ALTERNATIVE_EXCHANGES.items()
        if pair in config["supported_pairs"]
    ]

def get_best_exchange(pair: str) -> str:
    """
    Get best exchange for a specific trading pair based on fees, liquidity, and pair type

    Args:
        pair: Trading pair (e.g., 'BTC/USD' or 'BTC/USDT')

    Returns:
        str: Best exchange ID for the given pair

    Raises:
        ValueError: If no exchange supports the trading pair
    """
    available_exchanges = get_exchange_for_pair(pair)

    if not available_exchanges:
        raise ValueError(f"No exchange supports trading pair {pair}")

    # Special handling for fiat pairs - prefer Coinbase for USD/CAD pairs if available
    if (pair.endswith('/USD') or pair.endswith('/CAD')) and 'coinbase' in available_exchanges:
        return 'coinbase'

    # For other pairs, prioritize exchanges by fees, priority, and features
    best_exchange = min(
        available_exchanges,
        key=lambda x: (
            # Lower priority number is better
            ["primary", "secondary", "backup"].index(
                ALTERNATIVE_EXCHANGES[x]["priority"]
            ),
            # Lower fees are better
            ALTERNATIVE_EXCHANGES[x]["trading_fees"]["taker"],
            # Prefer exchanges with WebSocket support
            0 if ALTERNATIVE_EXCHANGES[x]["has_websocket"] else 1,
            # Prefer exchanges that don't require VPN
            0 if not ALTERNATIVE_EXCHANGES[x]["requires_vpn"] else 1
        )
    )

    return best_exchange
