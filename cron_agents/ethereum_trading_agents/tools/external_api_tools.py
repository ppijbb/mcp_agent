"""
External API Integration Tools
Integration with external APIs and MCP servers for real-time data
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import aiohttp
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class APISource(Enum):
    COINGECKO = "coingecko"
    COINMARKETCAP = "coinmarketcap"
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    ETHERSCAN = "etherscan"
    GLASSNODE = "glassnode"
    SANTIMENT = "santiment"
    DEFI_PULSE = "defi_pulse"
    WEB_SEARCH = "web_search"
    NEWS_API = "news_api"
    TWITTER_API = "twitter_api"
    REDDIT_API = "reddit_api"

@dataclass
class ExternalAPIConfig:
    """Configuration for external API integrations"""
    coingecko_api_key: Optional[str] = None
    coinmarketcap_api_key: Optional[str] = None
    etherscan_api_key: Optional[str] = None
    glassnode_api_key: Optional[str] = None
    santiment_api_key: Optional[str] = None
    news_api_key: Optional[str] = None
    twitter_api_key: Optional[str] = None
    reddit_api_key: Optional[str] = None
    mcp_server_url: Optional[str] = None
    request_timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 0.1

class ExternalAPIManager:
    """Manager for external API integrations and MCP server connections"""
    
    def __init__(self, config: ExternalAPIConfig):
        self.config = config
        self.session = None
        self.rate_limiter = asyncio.Semaphore(10)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_comprehensive_market_data(self, symbol: str = "ethereum") -> Dict[str, Any]:
        """Get comprehensive market data from multiple external sources"""
        try:
            async with self.rate_limiter:
                # Collect data from multiple external sources in parallel
                tasks = [
                    self._get_coingecko_data(symbol),
                    self._get_coinmarketcap_data(symbol),
                    self._get_binance_data(symbol),
                    self._get_coinbase_data(symbol),
                    self._get_kraken_data(symbol),
                    self._get_etherscan_data(symbol),
                    self._get_glassnode_data(symbol),
                    self._get_defi_pulse_data(),
                    self._get_news_data(symbol),
                    self._get_social_sentiment_data(symbol)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                market_data = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "status": "success",
                    "sources": {},
                    "aggregated_data": {}
                }
                
                source_names = [
                    "coingecko", "coinmarketcap", "binance", "coinbase", "kraken",
                    "etherscan", "glassnode", "defi_pulse", "news", "social_sentiment"
                ]
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to get {source_names[i]} data: {result}")
                        market_data["sources"][source_names[i]] = {"status": "error", "error": str(result)}
                    else:
                        market_data["sources"][source_names[i]] = result
                
                # Aggregate data from all sources
                market_data["aggregated_data"] = self._aggregate_market_data(market_data["sources"])
                
                return market_data
                
        except Exception as e:
            logger.error(f"Failed to get comprehensive market data: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _get_coingecko_data(self, symbol: str) -> Dict[str, Any]:
        """Get data from CoinGecko API"""
        try:
            url = "https://api.coingecko.com/api/v3/coins/ethereum"
            params = {
                "localization": "false",
                "tickers": "true",
                "market_data": "true",
                "community_data": "true",
                "developer_data": "true",
                "sparkline": "false"
            }
            
            if self.config.coingecko_api_key:
                params["x_cg_demo_api_key"] = self.config.coingecko_api_key
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "source": "coingecko",
                        "data": {
                            "price_usd": data.get("market_data", {}).get("current_price", {}).get("usd", 0),
                            "market_cap": data.get("market_data", {}).get("market_cap", {}).get("usd", 0),
                            "volume_24h": data.get("market_data", {}).get("total_volume", {}).get("usd", 0),
                            "price_change_24h": data.get("market_data", {}).get("price_change_percentage_24h", 0),
                            "market_cap_rank": data.get("market_cap_rank", 0),
                            "circulating_supply": data.get("market_data", {}).get("circulating_supply", 0),
                            "total_supply": data.get("market_data", {}).get("total_supply", 0),
                            "ath": data.get("market_data", {}).get("ath", {}).get("usd", 0),
                            "atl": data.get("market_data", {}).get("atl", {}).get("usd", 0),
                            "developer_data": data.get("developer_data", {}),
                            "community_data": data.get("community_data", {})
                        }
                    }
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
                    
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _get_coinmarketcap_data(self, symbol: str) -> Dict[str, Any]:
        """Get data from CoinMarketCap API"""
        try:
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            params = {"symbol": "ETH"}
            headers = {}
            
            if self.config.coinmarketcap_api_key:
                headers["X-CMC_PRO_API_KEY"] = self.config.coinmarketcap_api_key
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    eth_data = data.get("data", {}).get("ETH", {})
                    return {
                        "status": "success",
                        "source": "coinmarketcap",
                        "data": {
                            "price_usd": eth_data.get("quote", {}).get("USD", {}).get("price", 0),
                            "market_cap": eth_data.get("quote", {}).get("USD", {}).get("market_cap", 0),
                            "volume_24h": eth_data.get("quote", {}).get("USD", {}).get("volume_24h", 0),
                            "price_change_24h": eth_data.get("quote", {}).get("USD", {}).get("percent_change_24h", 0),
                            "market_cap_rank": eth_data.get("cmc_rank", 0),
                            "circulating_supply": eth_data.get("circulating_supply", 0),
                            "total_supply": eth_data.get("total_supply", 0),
                            "max_supply": eth_data.get("max_supply", 0)
                        }
                    }
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
                    
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _get_binance_data(self, symbol: str) -> Dict[str, Any]:
        """Get data from Binance API"""
        try:
            # Get 24hr ticker
            ticker_url = "https://api.binance.com/api/v3/ticker/24hr"
            ticker_params = {"symbol": "ETHUSDT"}
            
            # Get order book
            orderbook_url = "https://api.binance.com/api/v3/depth"
            orderbook_params = {"symbol": "ETHUSDT", "limit": 100}
            
            # Get recent trades
            trades_url = "https://api.binance.com/api/v3/trades"
            trades_params = {"symbol": "ETHUSDT", "limit": 100}
            
            ticker_response, orderbook_response, trades_response = await asyncio.gather(
                self.session.get(ticker_url, params=ticker_params),
                self.session.get(orderbook_url, params=orderbook_params),
                self.session.get(trades_url, params=trades_params)
            )
            
            if ticker_response.status == 200:
                ticker_data = await ticker_response.json()
                orderbook_data = await orderbook_response.json() if orderbook_response.status == 200 else {}
                trades_data = await trades_response.json() if trades_response.status == 200 else []
                
                return {
                    "status": "success",
                    "source": "binance",
                    "data": {
                        "price_usd": float(ticker_data.get("lastPrice", 0)),
                        "volume_24h": float(ticker_data.get("volume", 0)) * float(ticker_data.get("lastPrice", 0)),
                        "price_change_24h": float(ticker_data.get("priceChangePercent", 0)),
                        "high_24h": float(ticker_data.get("highPrice", 0)),
                        "low_24h": float(ticker_data.get("lowPrice", 0)),
                        "bid_price": float(ticker_data.get("bidPrice", 0)),
                        "ask_price": float(ticker_data.get("askPrice", 0)),
                        "order_book": orderbook_data,
                        "recent_trades": trades_data[:10]  # Last 10 trades
                    }
                }
            else:
                return {"status": "error", "message": f"HTTP {ticker_response.status}"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _get_coinbase_data(self, symbol: str) -> Dict[str, Any]:
        """Get data from Coinbase API"""
        try:
            # Get product ticker
            ticker_url = "https://api.exchange.coinbase.com/products/ETH-USD/ticker"
            
            # Get product stats
            stats_url = "https://api.exchange.coinbase.com/products/ETH-USD/stats"
            
            # Get product book
            book_url = "https://api.exchange.coinbase.com/products/ETH-USD/book"
            book_params = {"level": 2}
            
            ticker_response, stats_response, book_response = await asyncio.gather(
                self.session.get(ticker_url),
                self.session.get(stats_url),
                self.session.get(book_url, params=book_params)
            )
            
            if ticker_response.status == 200:
                ticker_data = await ticker_response.json()
                stats_data = await stats_response.json() if stats_response.status == 200 else {}
                book_data = await book_response.json() if book_response.status == 200 else {}
                
                return {
                    "status": "success",
                    "source": "coinbase",
                    "data": {
                        "price_usd": float(ticker_data.get("price", 0)),
                        "volume_24h": float(stats_data.get("volume", 0)),
                        "high_24h": float(stats_data.get("high", 0)),
                        "low_24h": float(stats_data.get("low", 0)),
                        "bid_price": float(ticker_data.get("bid", 0)),
                        "ask_price": float(ticker_data.get("ask", 0)),
                        "order_book": book_data
                    }
                }
            else:
                return {"status": "error", "message": f"HTTP {ticker_response.status}"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _get_kraken_data(self, symbol: str) -> Dict[str, Any]:
        """Get data from Kraken API"""
        try:
            url = "https://api.kraken.com/0/public/Ticker"
            params = {"pair": "ETHUSD"}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("error"):
                        return {"status": "error", "message": data["error"][0]}
                    
                    result = data.get("result", {})
                    eth_data = result.get("XETHZUSD", {})
                    
                    return {
                        "status": "success",
                        "source": "kraken",
                        "data": {
                            "price_usd": float(eth_data.get("c", [0])[0]),
                            "volume_24h": float(eth_data.get("v", [0])[0]) * float(eth_data.get("c", [0])[0]),
                            "high_24h": float(eth_data.get("h", [0])[0]),
                            "low_24h": float(eth_data.get("l", [0])[0]),
                            "bid_price": float(eth_data.get("b", [0])[0]),
                            "ask_price": float(eth_data.get("a", [0])[0])
                        }
                    }
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
                    
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _get_etherscan_data(self, symbol: str) -> Dict[str, Any]:
        """Get data from Etherscan API"""
        try:
            if not self.config.etherscan_api_key:
                return {"status": "error", "message": "Etherscan API key not provided"}
            
            # Get ETH supply
            supply_url = "https://api.etherscan.io/api"
            supply_params = {
                "module": "stats",
                "action": "ethsupply",
                "apikey": self.config.etherscan_api_key
            }
            
            # Get gas price
            gas_url = "https://api.etherscan.io/api"
            gas_params = {
                "module": "gastracker",
                "action": "gasoracle",
                "apikey": self.config.etherscan_api_key
            }
            
            # Get latest block
            block_url = "https://api.etherscan.io/api"
            block_params = {
                "module": "proxy",
                "action": "eth_blockNumber",
                "apikey": self.config.etherscan_api_key
            }
            
            supply_response, gas_response, block_response = await asyncio.gather(
                self.session.get(supply_url, params=supply_params),
                self.session.get(gas_url, params=gas_params),
                self.session.get(block_url, params=block_params)
            )
            
            supply_data = await supply_response.json() if supply_response.status == 200 else {}
            gas_data = await gas_response.json() if gas_response.status == 200 else {}
            block_data = await block_response.json() if block_response.status == 200 else {}
            
            return {
                "status": "success",
                "source": "etherscan",
                "data": {
                    "total_supply": int(supply_data.get("result", 0)) / 1e18 if supply_data.get("result") else 0,
                    "gas_price": {
                        "slow": int(gas_data.get("result", {}).get("SafeGasPrice", 0)),
                        "standard": int(gas_data.get("result", {}).get("ProposeGasPrice", 0)),
                        "fast": int(gas_data.get("result", {}).get("FastGasPrice", 0))
                    },
                    "latest_block": int(block_data.get("result", "0x0"), 16) if block_data.get("result") else 0
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _get_glassnode_data(self, symbol: str) -> Dict[str, Any]:
        """Get data from Glassnode API"""
        try:
            if not self.config.glassnode_api_key:
                return {"status": "error", "message": "Glassnode API key not provided"}
            
            base_url = "https://api.glassnode.com/v1/metrics"
            headers = {"X-API-KEY": self.config.glassnode_api_key}
            
            # Get active addresses
            active_url = f"{base_url}/addresses/active_count"
            active_params = {
                "a": "ETH",
                "f": "1d",
                "i": "1d",
                "s": int((datetime.now() - timedelta(days=30)).timestamp()),
                "u": int(datetime.now().timestamp())
            }
            
            # Get exchange flows
            exchange_url = f"{base_url}/distribution/exchange_flows"
            exchange_params = {
                "a": "ETH",
                "f": "1d",
                "i": "1d",
                "s": int((datetime.now() - timedelta(days=30)).timestamp()),
                "u": int(datetime.now().timestamp())
            }
            
            active_response, exchange_response = await asyncio.gather(
                self.session.get(active_url, params=active_params, headers=headers),
                self.session.get(exchange_url, params=exchange_params, headers=headers)
            )
            
            active_data = await active_response.json() if active_response.status == 200 else []
            exchange_data = await exchange_response.json() if exchange_response.status == 200 else []
            
            return {
                "status": "success",
                "source": "glassnode",
                "data": {
                    "active_addresses": active_data[-1] if active_data else 0,
                    "exchange_flows": exchange_data[-1] if exchange_data else 0,
                    "historical_active_addresses": active_data,
                    "historical_exchange_flows": exchange_data
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _get_defi_pulse_data(self) -> Dict[str, Any]:
        """Get data from DeFi Pulse API"""
        try:
            url = "https://api.defipulse.com/v1/defipulseapi/GetHistory"
            params = {
                "period": "30d",
                "length": "30"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "source": "defi_pulse",
                        "data": {
                            "current_tvl": data[-1] if data else 0,
                            "historical_tvl": data,
                            "tvl_change_24h": (data[-1] - data[-2]) / data[-2] * 100 if len(data) > 1 else 0
                        }
                    }
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
                    
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _get_news_data(self, symbol: str) -> Dict[str, Any]:
        """Get news data from News API"""
        try:
            if not self.config.news_api_key:
                return {"status": "error", "message": "News API key not provided"}
            
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": f"{symbol} OR ethereum OR ETH",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "apiKey": self.config.news_api_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get("articles", [])
                    
                    return {
                        "status": "success",
                        "source": "news_api",
                        "data": {
                            "articles": articles,
                            "total_articles": len(articles),
                            "sentiment": self._analyze_news_sentiment(articles)
                        }
                    }
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
                    
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _get_social_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """Get social sentiment data from various sources"""
        try:
            # This would integrate with Twitter API, Reddit API, etc.
            # For now, return mock data structure
            return {
                "status": "success",
                "source": "social_sentiment",
                "data": {
                    "twitter_sentiment": {"score": 0.2, "mentions": 1500},
                    "reddit_sentiment": {"score": 0.1, "mentions": 800},
                    "overall_sentiment": "bullish",
                    "sentiment_score": 0.15
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _analyze_news_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of news articles"""
        try:
            if not articles:
                return {"sentiment": "neutral", "score": 0.0}
            
            # Simple keyword-based sentiment analysis
            positive_keywords = ["bullish", "surge", "rally", "growth", "adoption", "breakthrough"]
            negative_keywords = ["bearish", "crash", "decline", "concern", "risk", "warning"]
            
            total_score = 0
            for article in articles:
                title = article.get("title", "").lower()
                description = article.get("description", "").lower()
                text = f"{title} {description}"
                
                positive_count = sum(1 for keyword in positive_keywords if keyword in text)
                negative_count = sum(1 for keyword in negative_keywords if keyword in text)
                
                if positive_count + negative_count > 0:
                    article_score = (positive_count - negative_count) / (positive_count + negative_count)
                    total_score += article_score
            
            avg_score = total_score / len(articles) if articles else 0
            
            if avg_score > 0.1:
                sentiment = "bullish"
            elif avg_score < -0.1:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "score": avg_score,
                "articles_analyzed": len(articles)
            }
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "score": 0.0}
    
    def _aggregate_market_data(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate data from all sources"""
        try:
            aggregated = {
                "price_usd": 0,
                "volume_24h": 0,
                "market_cap": 0,
                "price_change_24h": 0,
                "sources_count": 0,
                "confidence": 0
            }
            
            valid_prices = []
            valid_volumes = []
            valid_market_caps = []
            valid_changes = []
            
            for source_name, source_data in sources.items():
                if source_data.get("status") == "success" and "data" in source_data:
                    data = source_data["data"]
                    
                    if "price_usd" in data and data["price_usd"] > 0:
                        valid_prices.append(data["price_usd"])
                    
                    if "volume_24h" in data and data["volume_24h"] > 0:
                        valid_volumes.append(data["volume_24h"])
                    
                    if "market_cap" in data and data["market_cap"] > 0:
                        valid_market_caps.append(data["market_cap"])
                    
                    if "price_change_24h" in data:
                        valid_changes.append(data["price_change_24h"])
            
            # Calculate aggregated values
            if valid_prices:
                aggregated["price_usd"] = sum(valid_prices) / len(valid_prices)
                aggregated["sources_count"] = len(valid_prices)
            
            if valid_volumes:
                aggregated["volume_24h"] = sum(valid_volumes) / len(valid_volumes)
            
            if valid_market_caps:
                aggregated["market_cap"] = sum(valid_market_caps) / len(valid_market_caps)
            
            if valid_changes:
                aggregated["price_change_24h"] = sum(valid_changes) / len(valid_changes)
            
            # Calculate confidence based on number of sources
            total_sources = len(sources)
            successful_sources = len([s for s in sources.values() if s.get("status") == "success"])
            aggregated["confidence"] = successful_sources / total_sources if total_sources > 0 else 0
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Data aggregation failed: {e}")
            return {}
