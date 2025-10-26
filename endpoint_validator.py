# Standard library imports
import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

# Third-party imports
import aiohttp

class EndpointStatus(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"

@dataclass
class EndpointHealth:
    status: EndpointStatus
    response_time: float
    last_check: float
    success_rate: float
    consecutive_failures: int
    is_primary: bool

class EndpointValidator:
    def __init__(self):
        self.logger = logging.getLogger('EndpointValidator')
        self.endpoints: Dict[str, Dict[str, EndpointHealth]] = {}
        self.fallback_endpoints = {
            'binance': [
                'wss://stream.binance.com:9443/ws',
                'wss://stream.binancefuture.com/ws',
                'wss://fstream.binance.com/ws'
            ],
            'kucoin': [
                'wss://ws-api.kucoin.com/',
                'wss://ws-api-spot.kucoin.com/',
                'wss://ws-api-futures.kucoin.com/'
            ],
            'bybit': [
                'wss://stream.bybit.com/realtime',
                'wss://stream.bybit.com/spot/quote/ws/v2',
                'wss://stream.bybit.com/contract/usdt/public/v3'
            ],
            'huobi': [
                'wss://api.huobi.pro/ws',
                'wss://api-aws.huobi.pro/ws',
                'wss://api.huobi.pro/linear-swap-ws'
            ],
            'okx': [
                'wss://ws.okx.com:8443/ws/v5/public',
                'wss://wsaws.okx.com:8443/ws/v5/public',
                'wss://ws.okx.com:8443/ws/v5/business'
            ]
        }

        # REST API endpoints for health checks
        self.rest_endpoints = {
            'binance': [
                'https://api.binance.com/api/v3/ping',
                'https://api1.binance.com/api/v3/ping',
                'https://api2.binance.com/api/v3/ping'
            ],
            'kucoin': [
                'https://api.kucoin.com/api/v1/timestamp',
                'https://api1.kucoin.com/api/v1/timestamp',
                'https://api2.kucoin.com/api/v1/timestamp'
            ],
            'bybit': [
                'https://api.bybit.com/v2/public/time',
                'https://api.bytick.com/v2/public/time',
                'https://api-testnet.bybit.com/v2/public/time'
            ],
            'huobi': [
                'https://api.huobi.pro/v1/common/timestamp',
                'https://api-aws.huobi.pro/v1/common/timestamp',
                'https://api.huobi.pro/market/history/kline'
            ],
            'okx': [
                'https://www.okx.com/api/v5/public/time',
                'https://aws.okx.com/api/v5/public/time',
                'https://www.okx.com/api/v5/public/instruments'
            ]
        }

    async def validate_endpoint(self, exchange: str, endpoint: str) -> Tuple[bool, float]:
        """
        Validate a single endpoint's health
        Returns: (is_valid, response_time)
        """
        try:
            start_time = time.time()
            if endpoint.startswith('wss://'):
                # WebSocket validation
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.ws_connect(endpoint) as ws:
                        await ws.ping()
                        await ws.pong()
                        response_time = time.time() - start_time
                        return True, response_time
            else:
                # REST API validation
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(endpoint) as response:
                        response_time = time.time() - start_time
                        return response.status == 200, response_time
        except Exception as e:
            self.logger.warning(f"Endpoint validation failed for {endpoint}: {str(e)}")
            return False, 999.0

    async def get_best_endpoint(self, exchange: str, endpoint_type: str = 'ws') -> Optional[str]:
        """
        Get the best available endpoint for an exchange
        """
        endpoints = (self.fallback_endpoints if endpoint_type == 'ws' else self.rest_endpoints).get(exchange, [])

        if not endpoints:
            return None

        best_endpoint = None
        best_response_time = float('inf')

        for endpoint in endpoints:
            is_valid, response_time = await self.validate_endpoint(exchange, endpoint)
            if is_valid and response_time < best_response_time:
                best_endpoint = endpoint
                best_response_time = response_time

        return best_endpoint

    async def monitor_endpoints(self):
        """
        Continuously monitor endpoint health
        """
        while True:
            for exchange in self.fallback_endpoints.keys():
                # Check WebSocket endpoints
                ws_status = await self._check_exchange_endpoints(
                    exchange,
                    self.fallback_endpoints[exchange],
                    'ws'
                )

                # Check REST endpoints
                rest_status = await self._check_exchange_endpoints(
                    exchange,
                    self.rest_endpoints[exchange],
                    'rest'
                )

                # Update endpoint health metrics
                if exchange not in self.endpoints:
                    self.endpoints[exchange] = {}

                self.endpoints[exchange].update({
                    'ws': ws_status,
                    'rest': rest_status
                })

            # Wait before next check
            await asyncio.sleep(60)  # Check every minute

    async def _check_exchange_endpoints(self,
                                     exchange: str,
                                     endpoints: List[str],
                                     endpoint_type: str) -> EndpointHealth:
        """
        Check all endpoints for an exchange and return health status
        """
        total_checks = len(endpoints)
        successful_checks = 0
        total_response_time = 0
        consecutive_failures = 0

        for endpoint in endpoints:
            is_valid, response_time = await self.validate_endpoint(exchange, endpoint)

            if is_valid:
                successful_checks += 1
                total_response_time += response_time
                consecutive_failures = 0
            else:
                consecutive_failures += 1

        success_rate = successful_checks / total_checks if total_checks > 0 else 0
        avg_response_time = total_response_time / successful_checks if successful_checks > 0 else 999.0

        # Determine status based on health metrics
        if success_rate >= 0.8:
            status = EndpointStatus.ACTIVE
        elif success_rate >= 0.5:
            status = EndpointStatus.DEGRADED
        else:
            status = EndpointStatus.DOWN

        return EndpointHealth(
            status=status,
            response_time=avg_response_time,
            last_check=time.time(),
            success_rate=success_rate,
            consecutive_failures=consecutive_failures,
            is_primary=endpoint_type == 'ws'
        )

    def get_exchange_health(self, exchange: str) -> Dict[str, EndpointHealth]:
        """
        Get current health status for an exchange's endpoints
        """
        return self.endpoints.get(exchange, {})

    def is_exchange_healthy(self, exchange: str) -> bool:
        """
        Check if an exchange's endpoints are healthy enough for trading
        """
        health = self.get_exchange_health(exchange)

        if not health:
            return False

        ws_health = health.get('ws')
        rest_health = health.get('rest')

        if not ws_health or not rest_health:
            return False

        # Consider exchange healthy if either WebSocket or REST endpoints are active
        # and the other is at least degraded
        return (
            (ws_health.status == EndpointStatus.ACTIVE and rest_health.status != EndpointStatus.DOWN) or
            (rest_health.status == EndpointStatus.ACTIVE and ws_health.status != EndpointStatus.DOWN)
        )

    def get_all_exchange_health(self) -> Dict[str, Dict[str, EndpointHealth]]:
        """
        Get health status for all exchanges
        """
        return self.endpoints

class EndpointManager:
    def __init__(self):
        self.validator = EndpointValidator()
        self.active_endpoints: Dict[str, Dict[str, str]] = {}
        self.is_monitoring: bool = False

    async def start_monitoring(self):
        """
        Start endpoint health monitoring
        """
        if not self.is_monitoring:
            self.is_monitoring = True
            asyncio.create_task(self.validator.monitor_endpoints())
            asyncio.create_task(self._manage_endpoints())

    async def _manage_endpoints(self):
        """
        Continuously manage and update active endpoints
        """
        while self.is_monitoring:
            for exchange in self.validator.fallback_endpoints.keys():
                # Get current best endpoints
                ws_endpoint = await self.validator.get_best_endpoint(exchange, 'ws')
                rest_endpoint = await self.validator.get_best_endpoint(exchange, 'rest')

                # Update active endpoints
                if ws_endpoint and rest_endpoint:
                    self.active_endpoints[exchange] = {
                        'ws': ws_endpoint,
                        'rest': rest_endpoint
                    }
                elif exchange in self.active_endpoints:
                    del self.active_endpoints[exchange]

            await asyncio.sleep(300)  # Update every 5 minutes

    async def get_active_endpoint(self, exchange: str, endpoint_type: str = 'ws') -> Optional[str]:
        """
        Get the current active endpoint for an exchange
        """
        if exchange not in self.active_endpoints:
            return None
        return self.active_endpoints[exchange].get(endpoint_type)

    def stop_monitoring(self) -> None:
        """
        Stop endpoint health monitoring
        """
        self.is_monitoring = False
