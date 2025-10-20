"""
MCP Client for communicating with Ethereum Trading and Market Data servers
"""

import aiohttp
import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self, ethereum_trading_url: str = None, market_data_url: str = None, bitcoin_trading_url: str = None):
        """Initialize MCP client with Bitcoin support"""
        self.ethereum_trading_url = ethereum_trading_url or os.getenv("MCP_ETHEREUM_TRADING_URL", "http://localhost:3005")
        self.market_data_url = market_data_url or os.getenv("MCP_MARKET_DATA_URL", "http://localhost:3006")
        self.bitcoin_trading_url = bitcoin_trading_url or os.getenv("MCP_BITCOIN_TRADING_URL", "http://localhost:3008")
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def connect(self):
        """Connect to MCP servers"""
        try:
            self.session = aiohttp.ClientSession()
            logger.info("MCP client connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect MCP client: {e}")
            raise
    
    async def close(self):
        """Close MCP client connection"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
                logger.info("MCP client connection closed")
        except Exception as e:
            logger.error(f"Error closing MCP client: {e}")
    
    async def get_ethereum_balance(self, address: str) -> Dict[str, Any]:
        """Get Ethereum balance via MCP"""
        try:
            async with self.session.get(f"{self.ethereum_trading_url}/balance/{address}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Ethereum balance request failed: HTTP {response.status}")
                    raise RuntimeError(f"Ethereum MCP server error: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to get Ethereum balance: {e}")
            raise RuntimeError(f"Ethereum balance retrieval failed: {e}")
    
    async def get_gas_price(self) -> Dict[str, Any]:
        """Get current gas price via MCP"""
        try:
            async with self.session.get(f"{self.ethereum_trading_url}/gas-price") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Gas price request failed: HTTP {response.status}")
                    raise RuntimeError(f"Ethereum MCP server error: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to get gas price: {e}")
            raise RuntimeError(f"Gas price retrieval failed: {e}")
    
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
                    logger.error(f"Ethereum transaction request failed: HTTP {response.status}")
                    raise RuntimeError(f"Ethereum MCP server error: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to send Ethereum transaction: {e}")
            raise RuntimeError(f"Ethereum transaction failed: {e}")
    
    async def get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction status via MCP"""
        try:
            async with self.session.get(f"{self.ethereum_trading_url}/transaction-status/{tx_hash}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Transaction status request failed: HTTP {response.status}")
                    raise RuntimeError(f"Ethereum MCP server error: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to get transaction status: {e}")
            raise RuntimeError(f"Transaction status retrieval failed: {e}")
    
    # Bitcoin MCP Methods - NO FALLBACKS
    async def get_bitcoin_balance(self, address: str) -> Dict[str, Any]:
        """Get Bitcoin balance via MCP"""
        try:
            async with self.session.get(f"{self.bitcoin_trading_url}/balance/{address}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Bitcoin balance request failed: HTTP {response.status}")
                    raise RuntimeError(f"Bitcoin MCP server error: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to get Bitcoin balance: {e}")
            raise RuntimeError(f"Bitcoin balance retrieval failed: {e}")
    
    async def get_bitcoin_fee_estimate(self) -> Dict[str, Any]:
        """Get Bitcoin fee estimate via MCP"""
        try:
            async with self.session.get(f"{self.bitcoin_trading_url}/fee-estimate") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Bitcoin fee estimate request failed: HTTP {response.status}")
                    raise RuntimeError(f"Bitcoin MCP server error: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to get Bitcoin fee estimate: {e}")
            raise RuntimeError(f"Bitcoin fee estimation failed: {e}")
    
    async def send_bitcoin_transaction(self, to_address: str, amount_btc: float, 
                                     fee_rate: float = None) -> Dict[str, Any]:
        """Send Bitcoin transaction via MCP"""
        try:
            payload = {
                "to_address": to_address,
                "amount_btc": amount_btc,
                "fee_rate": fee_rate
            }
            
            async with self.session.post(f"{self.bitcoin_trading_url}/send-transaction", 
                                       json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Bitcoin transaction request failed: HTTP {response.status}")
                    raise RuntimeError(f"Bitcoin MCP server error: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to send Bitcoin transaction: {e}")
            raise RuntimeError(f"Bitcoin transaction failed: {e}")
    
    async def get_bitcoin_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """Get Bitcoin transaction status via MCP"""
        try:
            async with self.session.get(f"{self.bitcoin_trading_url}/transaction-status/{tx_hash}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Bitcoin transaction status request failed: HTTP {response.status}")
                    raise RuntimeError(f"Bitcoin MCP server error: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to get Bitcoin transaction status: {e}")
            raise RuntimeError(f"Bitcoin transaction status retrieval failed: {e}")
    
    async def get_bitcoin_market_data(self) -> Dict[str, Any]:
        """Get Bitcoin market data via MCP"""
        try:
            async with self.session.get(f"{self.bitcoin_trading_url}/market-data") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Bitcoin market data request failed: HTTP {response.status}")
                    raise RuntimeError(f"Bitcoin MCP server error: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to get Bitcoin market data: {e}")
            raise RuntimeError(f"Bitcoin market data retrieval failed: {e}")
    
    async def get_ethereum_price(self) -> Dict[str, Any]:
        """Get Ethereum price via MCP"""
        try:
            async with self.session.get(f"{self.market_data_url}/ethereum-price") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Ethereum price request failed: HTTP {response.status}")
                    raise RuntimeError(f"Market data MCP server error: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to get Ethereum price: {e}")
            raise RuntimeError(f"Ethereum price retrieval failed: {e}")
    
    async def get_bitcoin_price(self) -> Dict[str, Any]:
        """Get Bitcoin price via MCP"""
        try:
            async with self.session.get(f"{self.market_data_url}/bitcoin-price") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Bitcoin price request failed: HTTP {response.status}")
                    raise RuntimeError(f"Market data MCP server error: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to get Bitcoin price: {e}")
            raise RuntimeError(f"Bitcoin price retrieval failed: {e}")
    
    async def get_market_trends(self, timeframe: str = "24h") -> Dict[str, Any]:
        """Get market trends via MCP"""
        try:
            async with self.session.get(f"{self.market_data_url}/market-trends?timeframe={timeframe}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Market trends request failed: HTTP {response.status}")
                    raise RuntimeError(f"Market data MCP server error: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to get market trends: {e}")
            raise RuntimeError(f"Market trends retrieval failed: {e}")
    
    async def get_technical_indicators(self) -> Dict[str, Any]:
        """Get technical indicators via MCP"""
        try:
            async with self.session.get(f"{self.market_data_url}/technical-indicators") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Technical indicators request failed: HTTP {response.status}")
                    raise RuntimeError(f"Market data MCP server error: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to get technical indicators: {e}")
            raise RuntimeError(f"Technical indicators retrieval failed: {e}")
    
    async def search_market_news(self, query: str = "ethereum") -> Dict[str, Any]:
        """Search market news via MCP"""
        try:
            async with self.session.get(f"{self.market_data_url}/search-news?query={query}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Market news search request failed: HTTP {response.status}")
                    raise RuntimeError(f"Market data MCP server error: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to search market news: {e}")
            raise RuntimeError(f"Market news search failed: {e}")
    
    async def enhanced_mcp_operations(self) -> Dict[str, Any]:
        """Execute enhanced MCP operations with parallel processing and retry logic"""
        try:
            # Validate MCP client state
            if not self.session:
                raise ValueError("MCP client session not available")
            
            # Define MCP operations with different priorities and validation
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
                # Test MCP connection with actual address
                test_result = await self.get_ethereum_balance(self.config.ETHEREUM_ADDRESS)
            ]
            
            # Execute operations with priority-based timeouts and enhanced error handling
            results = {}
            
            # High priority operations (fast timeout)
            try:
                high_results = await asyncio.wait_for(
                    asyncio.gather(*high_priority_ops, return_exceptions=True),
                    timeout=15
                )
                results["high_priority"] = self._process_mcp_results(high_results, ["price", "gas"])
            except asyncio.TimeoutError:
                logger.warning("High priority MCP operations timed out")
                results["high_priority"] = {"status": "timeout", "retry_available": True, "priority": "high"}
            except Exception as high_error:
                logger.error(f"High priority MCP operations failed: {high_error}")
                results["high_priority"] = {"status": "error", "error": str(high_error), "retry_available": True, "priority": "high"}
            
            # Medium priority operations
            try:
                medium_results = await asyncio.wait_for(
                    asyncio.gather(*medium_priority_ops, return_exceptions=True),
                    timeout=25
                )
                results["medium_priority"] = self._process_mcp_results(medium_results, ["trends", "indicators"])
            except asyncio.TimeoutError:
                logger.warning("Medium priority MCP operations timed out")
                results["medium_priority"] = {"status": "timeout", "retry_available": True, "priority": "medium"}
            except Exception as medium_error:
                logger.error(f"Medium priority MCP operations failed: {medium_error}")
                results["medium_priority"] = {"status": "error", "error": str(medium_error), "retry_available": True, "priority": "medium"}
            
            # Low priority operations (longer timeout)
            try:
                low_results = await asyncio.wait_for(
                    asyncio.gather(*low_priority_ops, return_exceptions=True),
                    timeout=45
                )
                results["low_priority"] = self._process_mcp_results(low_results, ["news", "balance"])
            except asyncio.TimeoutError:
                logger.warning("Low priority MCP operations timed out")
                results["low_priority"] = {"status": "timeout", "retry_available": True, "priority": "low"}
            except Exception as low_error:
                logger.error(f"Low priority MCP operations failed: {low_error}")
                results["low_priority"] = {"status": "error", "error": str(low_error), "retry_available": True, "priority": "low"}
            
            # Calculate overall MCP health score with validation
            health_score = self._calculate_mcp_health_score(results)
            
            # Determine overall status based on health score and critical operations
            overall_status = "healthy"
            if health_score < 50:
                overall_status = "critical"
            elif health_score < 80:
                overall_status = "degraded"
            
            # Check if critical operations (high priority) failed
            high_priority_status = results.get("high_priority", {}).get("status")
            if high_priority_status in ["error", "timeout"]:
                overall_status = "critical"
                logger.error("Critical MCP operations failed - system status degraded")
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "mcp_operations": results,
                "health_score": health_score,
                "overall_status": overall_status,
                "critical_operations_status": high_priority_status,
                "retry_recommendations": self._generate_retry_recommendations(results)
            }
            
        except Exception as e:
            logger.error(f"Enhanced MCP operations failed: {e}")
            return {"status": "error", "message": str(e), "error_type": type(e).__name__}
    
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
    
    def _generate_retry_recommendations(self, results: Dict) -> Dict[str, Any]:
        """Generate retry recommendations based on MCP operation results"""
        recommendations = {
            "immediate_retry": [],
            "delayed_retry": [],
            "no_retry": [],
            "system_impact": "none"
        }
        
        for priority, result in results.items():
            if result.get("status") in ["error", "timeout"]:
                if priority == "high_priority":
                    recommendations["immediate_retry"].append(priority)
                    recommendations["system_impact"] = "critical"
                elif priority == "medium_priority":
                    recommendations["delayed_retry"].append(priority)
                    if recommendations["system_impact"] != "critical":
                        recommendations["system_impact"] = "moderate"
                else:
                    recommendations["delayed_retry"].append(priority)
            elif result.get("status") == "success":
                recommendations["no_retry"].append(priority)
        
        # Add timing recommendations
        if recommendations["immediate_retry"]:
            recommendations["retry_delay_seconds"] = 5
        elif recommendations["delayed_retry"]:
            recommendations["retry_delay_seconds"] = 30
        
        return recommendations
    
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
