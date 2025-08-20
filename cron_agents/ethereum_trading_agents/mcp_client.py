"""
MCP Client for communicating with Ethereum Trading and Market Data servers
"""

import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self, ethereum_trading_url: str, market_data_url: str):
        """Initialize MCP client"""
        self.ethereum_trading_url = ethereum_trading_url
        self.market_data_url = market_data_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_ethereum_balance(self, address: str) -> Dict[str, Any]:
        """Get Ethereum balance via MCP"""
        try:
            async with self.session.get(f"{self.ethereum_trading_url}/balance/{address}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Failed to get Ethereum balance: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_gas_price(self) -> Dict[str, Any]:
        """Get current gas price via MCP"""
        try:
            async with self.session.get(f"{self.ethereum_trading_url}/gas-price") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Failed to get gas price: {e}")
            return {"status": "error", "message": str(e)}
    
    async def send_ethereum_transaction(self, to_address: str, amount_eth: float, 
                                      gas_limit: int = 21000) -> Dict[str, Any]:
        """Send Ethereum transaction via MCP"""
        try:
            payload = {
                "to_address": to_address,
                "amount_eth": amount_eth,
                "gas_limit": gas_limit
            }
            
            async with self.session.post(f"{self.ethereum_trading_url}/send-transaction", 
                                       json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Failed to send Ethereum transaction: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction status via MCP"""
        try:
            async with self.session.get(f"{self.ethereum_trading_url}/transaction-status/{tx_hash}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Failed to get transaction status: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_ethereum_price(self) -> Dict[str, Any]:
        """Get Ethereum price via MCP"""
        try:
            async with self.session.get(f"{self.market_data_url}/ethereum-price") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Failed to get Ethereum price: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_market_trends(self, timeframe: str = "24h") -> Dict[str, Any]:
        """Get market trends via MCP"""
        try:
            async with self.session.get(f"{self.market_data_url}/market-trends?timeframe={timeframe}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Failed to get market trends: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_technical_indicators(self) -> Dict[str, Any]:
        """Get technical indicators via MCP"""
        try:
            async with self.session.get(f"{self.market_data_url}/technical-indicators") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Failed to get technical indicators: {e}")
            return {"status": "error", "message": str(e)}
    
    async def search_market_news(self, query: str = "ethereum") -> Dict[str, Any]:
        """Search market news via MCP"""
        try:
            async with self.session.get(f"{self.market_data_url}/search-news?query={query}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Failed to search market news: {e}")
            return {"status": "error", "message": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of MCP servers"""
        try:
            health_status = {}
            
            # Check Ethereum Trading MCP
            try:
                async with self.session.get(f"{self.ethereum_trading_url}/health") as response:
                    health_status['ethereum_trading'] = {
                        'status': 'healthy' if response.status == 200 else 'unhealthy',
                        'response_time': response.headers.get('X-Response-Time', 'unknown')
                    }
            except Exception as e:
                health_status['ethereum_trading'] = {'status': 'error', 'error': str(e)}
            
            # Check Market Data MCP
            try:
                async with self.session.get(f"{self.market_data_url}/health") as response:
                    health_status['market_data'] = {
                        'status': 'healthy' if response.status == 200 else 'unhealthy',
                        'response_time': response.headers.get('X-Response-Time', 'unknown')
                    }
            except Exception as e:
                health_status['market_data'] = {'status': 'error', 'error': str(e)}
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "servers": health_status
            }
            
        except Exception as e:
            logger.error(f"Failed to check MCP health: {e}")
            return {"status": "error", "message": str(e)}
    
    async def batch_market_data(self) -> Dict[str, Any]:
        """Get all market data in a single batch call"""
        try:
            # Execute multiple requests concurrently
            tasks = [
                self.get_ethereum_price(),
                self.get_market_trends(),
                self.get_technical_indicators()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "ethereum_price": results[0] if not isinstance(results[0], Exception) else {"status": "error", "message": str(results[0])},
                "market_trends": results[1] if not isinstance(results[1], Exception) else {"status": "error", "message": str(results[1])},
                "technical_indicators": results[2] if not isinstance(results[2], Exception) else {"status": "error", "message": str(results[2])}
            }
            
        except Exception as e:
            logger.error(f"Failed to get batch market data: {e}")
            return {"status": "error", "message": str(e)}
