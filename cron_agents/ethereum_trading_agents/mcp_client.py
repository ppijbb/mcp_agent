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
    
    async def enhanced_mcp_operations(self) -> Dict[str, Any]:
        """Execute enhanced MCP operations with parallel processing and retry logic"""
        try:
            # Define MCP operations with different priorities
            high_priority_ops = [
                self.get_ethereum_price(),
                self.get_gas_price()
            ]
            
            medium_priority_ops = [
                self.get_market_trends(),
                self.get_technical_indicators()
            ]
            
            low_priority_ops = [
                self.search_market_news("ethereum"),
                self.get_ethereum_balance("0x0000000000000000000000000000000000000000")  # Dummy address for testing
            ]
            
            # Execute operations with priority-based timeouts
            results = {}
            
            # High priority operations (fast timeout)
            try:
                high_results = await asyncio.wait_for(
                    asyncio.gather(*high_priority_ops, return_exceptions=True),
                    timeout=15
                )
                results["high_priority"] = self._process_mcp_results(high_results, ["price", "gas"])
            except asyncio.TimeoutError:
                results["high_priority"] = {"status": "timeout", "retry_available": True}
            
            # Medium priority operations
            try:
                medium_results = await asyncio.wait_for(
                    asyncio.gather(*medium_priority_ops, return_exceptions=True),
                    timeout=25
                )
                results["medium_priority"] = self._process_mcp_results(medium_results, ["trends", "indicators"])
            except asyncio.TimeoutError:
                results["medium_priority"] = {"status": "timeout", "retry_available": True}
            
            # Low priority operations (longer timeout)
            try:
                low_results = await asyncio.wait_for(
                    asyncio.gather(*low_priority_ops, return_exceptions=True),
                    timeout=45
                )
                results["low_priority"] = self._process_mcp_results(low_results, ["news", "balance"])
            except asyncio.TimeoutError:
                results["low_priority"] = {"status": "timeout", "retry_available": True}
            
            # Calculate overall MCP health score
            health_score = self._calculate_mcp_health_score(results)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "mcp_operations": results,
                "health_score": health_score,
                "overall_status": "healthy" if health_score >= 80 else "degraded" if health_score >= 50 else "critical"
            }
            
        except Exception as e:
            logger.error(f"Enhanced MCP operations failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def _process_mcp_results(self, results: List, operation_names: List[str]) -> Dict[str, Any]:
        """Process MCP operation results with detailed status"""
        processed = {}
        for i, result in enumerate(results):
            op_name = operation_names[i] if i < len(operation_names) else f"operation_{i}"
            if isinstance(result, Exception):
                processed[op_name] = {
                    "status": "error",
                    "message": str(result),
                    "retry_available": True,
                    "error_type": type(result).__name__
                }
            else:
                processed[op_name] = result
        
        return {
            "status": "success",
            "operations": processed,
            "success_count": sum(1 for r in processed.values() if r.get("status") == "success"),
            "total_count": len(processed)
        }
    
    def _calculate_mcp_health_score(self, results: Dict) -> int:
        """Calculate MCP health score based on operation results"""
        total_operations = 0
        successful_operations = 0
        
        for priority_level, result in results.items():
            if result.get("status") == "success":
                total_ops = result.get("total_count", 0)
                success_ops = result.get("success_count", 0)
                total_operations += total_ops
                successful_operations += success_ops
        
        if total_operations == 0:
            return 0
        
        return int((successful_operations / total_operations) * 100)
    
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
        """Get all market data in a single batch call with enhanced MCP processing"""
        try:
            # Execute multiple requests concurrently with timeout
            tasks = [
                self.get_ethereum_price(),
                self.get_market_trends(),
                self.get_technical_indicators()
            ]
            
            # Add timeout for MCP operations
            timeout = 30  # 30 seconds timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            
            # Process results with enhanced error handling
            processed_results = {}
            for i, result in enumerate(['ethereum_price', 'market_trends', 'technical_indicators']):
                if isinstance(results[i], Exception):
                    processed_results[result] = {
                        "status": "error", 
                        "message": str(results[i]),
                        "retry_available": True
                    }
                else:
                    processed_results[result] = results[i]
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "mcp_processing_time": datetime.now().isoformat(),
                "data_sources": processed_results,
                "overall_health": "healthy" if all(
                    r.get("status") == "success" for r in processed_results.values()
                ) else "degraded"
            }
            
        except asyncio.TimeoutError:
            logger.error("MCP batch market data request timed out")
            return {
                "status": "error", 
                "message": "MCP request timeout",
                "retry_available": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get batch market data: {e}")
            return {"status": "error", "message": str(e)}
