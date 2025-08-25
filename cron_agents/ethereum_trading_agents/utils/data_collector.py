"""
Enhanced Data Collector for Ethereum Trading Agents

This module provides comprehensive data collection from multiple sources:
1. News APIs for real-time market news
2. Social media monitoring for sentiment analysis
3. Technical data from various exchanges
4. On-chain data for blockchain metrics
5. Expert opinions and analysis
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class DataCollector:
    """Comprehensive data collector for trading agents"""
    
    def __init__(self):
        self.session = None
        self.api_keys = self._load_api_keys()
        self.data_cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables"""
        return {
            "newsapi": os.getenv("NEWS_API_KEY"),
            "alphavantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
            "cryptocompare": os.getenv("CRYPTOCOMPARE_API_KEY"),
            "coingecko": os.getenv("COINGECKO_API_KEY"),
            "etherscan": os.getenv("ETHERSCAN_API_KEY"),
            "glassnode": os.getenv("GLASSNODE_API_KEY"),
            "twitter": os.getenv("TWITTER_BEARER_TOKEN"),
            "reddit": os.getenv("REDDIT_CLIENT_ID"),
            "telegram": os.getenv("TELEGRAM_BOT_TOKEN")
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def connect(self):
        """Connect to data sources"""
        try:
            self.session = aiohttp.ClientSession()
            logger.info("Data collector connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect data collector: {e}")
            raise
    
    async def close(self):
        """Close data collector connection"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
                logger.info("Data collector connection closed")
        except Exception as e:
            logger.error(f"Error closing data collector: {e}")
    
    async def collect_comprehensive_data(self, asset: str = "ETH") -> Dict[str, Any]:
        """Collect comprehensive data from all sources"""
        try:
            tasks = [
                self._collect_news_data(asset),
                self._collect_social_sentiment(asset),
                self._collect_technical_data(asset),
                self._collect_onchain_data(asset),
                self._collect_expert_opinions(asset)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                "news_data": results[0] if not isinstance(results[0], Exception) else {},
                "social_sentiment": results[1] if not isinstance(results[1], Exception) else {},
                "technical_data": results[2] if not isinstance(results[2], Exception) else {},
                "onchain_data": results[3] if not isinstance(results[3], Exception) else {},
                "expert_opinions": results[4] if not isinstance(results[4], Exception) else {},
                "timestamp": datetime.now().isoformat(),
                "asset": asset
            }
            
        except Exception as e:
            logger.error(f"Error collecting comprehensive data: {e}")
            return {"error": str(e)}
    
    async def _collect_news_data(self, asset: str) -> Dict[str, Any]:
        """Collect news data from multiple sources"""
        try:
            news_data = {}
            
            # NewsAPI.org
            if self.api_keys.get("newsapi"):
                news_data["newsapi"] = await self._fetch_newsapi_news(asset)
            
            # Alpha Vantage News
            if self.api_keys.get("alphavantage"):
                news_data["alphavantage"] = await self._fetch_alphavantage_news(asset)
            
            # CryptoCompare News
            if self.api_keys.get("cryptocompare"):
                news_data["cryptocompare"] = await self._fetch_cryptocompare_news(asset)
            
            return news_data
            
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
            return {}
    
    async def _fetch_newsapi_news(self, asset: str) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI.org"""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": f"{asset} cryptocurrency",
                "apiKey": self.api_keys["newsapi"],
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("articles", [])
                else:
                    logger.warning(f"NewsAPI returned status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching NewsAPI data: {e}")
            return []
    
    async def _fetch_alphavantage_news(self, asset: str) -> List[Dict[str, Any]]:
        """Fetch news from Alpha Vantage"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": f"{asset}USD",
                "apikey": self.api_keys["alphavantage"],
                "limit": 20
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("feed", [])
                else:
                    logger.warning(f"Alpha Vantage returned status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data: {e}")
            return []
    
    async def _fetch_cryptocompare_news(self, asset: str) -> List[Dict[str, Any]]:
        """Fetch news from CryptoCompare"""
        try:
            url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={asset}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("Data", [])
                else:
                    logger.warning(f"CryptoCompare returned status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching CryptoCompare data: {e}")
            return []
    
    async def _collect_social_sentiment(self, asset: str) -> Dict[str, Any]:
        """Collect social media sentiment data"""
        try:
            sentiment_data = {}
            
            # Twitter/X sentiment (if API access available)
            if self.api_keys.get("twitter"):
                sentiment_data["twitter"] = await self._fetch_twitter_sentiment(asset)
            
            # Reddit sentiment
            if self.api_keys.get("reddit"):
                sentiment_data["reddit"] = await self._fetch_reddit_sentiment(asset)
            
            # Telegram sentiment
            if self.api_keys.get("telegram"):
                sentiment_data["telegram"] = await self._fetch_telegram_sentiment(asset)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error collecting social sentiment: {e}")
            return {}
    
    async def _fetch_twitter_sentiment(self, asset: str) -> Dict[str, Any]:
        """Fetch Twitter sentiment data"""
        try:
            # Note: Twitter API v2 requires proper authentication
            # This is a simplified example
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                "Authorization": f"Bearer {self.api_keys['twitter']}"
            }
            params = {
                "query": f"{asset} crypto",
                "max_results": 100
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._analyze_twitter_sentiment(data.get("data", []))
                else:
                    logger.warning(f"Twitter API returned status {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching Twitter data: {e}")
            return {}
    
    def _analyze_twitter_sentiment(self, tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment from Twitter data"""
        # Simplified sentiment analysis
        positive_keywords = ["bullish", "moon", "pump", "buy", "long", "hodl"]
        negative_keywords = ["bearish", "dump", "sell", "short", "crash"]
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for tweet in tweets:
            text = tweet.get("text", "").lower()
            if any(keyword in text for keyword in positive_keywords):
                positive_count += 1
            elif any(keyword in text for keyword in negative_keywords):
                negative_count += 1
            else:
                neutral_count += 1
        
        total = len(tweets) if tweets else 1
        
        return {
            "positive_percentage": (positive_count / total) * 100,
            "negative_percentage": (negative_count / total) * 100,
            "neutral_percentage": (neutral_count / total) * 100,
            "total_tweets": total,
            "sentiment_score": (positive_count - negative_count) / total
        }
    
    async def _fetch_reddit_sentiment(self, asset: str) -> Dict[str, Any]:
        """Fetch Reddit sentiment data"""
        try:
            # Reddit API requires proper OAuth setup
            # This is a simplified example
            subreddits = ["cryptocurrency", "ethereum", "cryptomarkets"]
            sentiment_data = {}
            
            for subreddit in subreddits:
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    "q": asset,
                    "restrict_sr": "true",
                    "sort": "hot",
                    "t": "day"
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = data.get("data", {}).get("children", [])
                        sentiment_data[subreddit] = self._analyze_reddit_sentiment(posts)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error fetching Reddit data: {e}")
            return {}
    
    def _analyze_reddit_sentiment(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment from Reddit posts"""
        total_score = 0
        total_posts = len(posts)
        
        for post in posts:
            post_data = post.get("data", {})
            score = post_data.get("score", 0)
            total_score += score
        
        avg_score = total_score / total_posts if total_posts > 0 else 0
        
        return {
            "total_posts": total_posts,
            "average_score": avg_score,
            "total_score": total_score,
            "sentiment": "positive" if avg_score > 0 else "negative" if avg_score < 0 else "neutral"
        }
    
    async def _fetch_telegram_sentiment(self, asset: str) -> Dict[str, Any]:
        """Fetch Telegram sentiment data"""
        try:
            # Telegram Bot API implementation
            # This is a simplified example
            return {
                "message_count": 0,
                "sentiment": "neutral",
                "active_channels": 0
            }
            
        except Exception as e:
            logger.error(f"Error fetching Telegram data: {e}")
            return {}
    
    async def _collect_technical_data(self, asset: str) -> Dict[str, Any]:
        """Collect technical data from exchanges and APIs"""
        try:
            technical_data = {}
            
            # CoinGecko data
            if self.api_keys.get("coingecko"):
                technical_data["coingecko"] = await self._fetch_coingecko_data(asset)
            
            # CoinMarketCap data (if available)
            technical_data["market_data"] = await self._fetch_market_data(asset)
            
            return technical_data
            
        except Exception as e:
            logger.error(f"Error collecting technical data: {e}")
            return {}
    
    async def _fetch_coingecko_data(self, asset: str) -> Dict[str, Any]:
        """Fetch data from CoinGecko"""
        try:
            # Map asset to CoinGecko ID
            asset_mapping = {
                "ETH": "ethereum",
                "BTC": "bitcoin",
                "ADA": "cardano"
            }
            
            coin_id = asset_mapping.get(asset, asset.lower())
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "current_price": data.get("market_data", {}).get("current_price", {}),
                        "market_cap": data.get("market_data", {}).get("market_cap", {}),
                        "volume": data.get("market_data", {}).get("total_volume", {}),
                        "price_change_24h": data.get("market_data", {}).get("price_change_percentage_24h"),
                        "market_cap_rank": data.get("market_cap_rank"),
                        "developer_score": data.get("developer_score"),
                        "community_score": data.get("community_score")
                    }
                else:
                    logger.warning(f"CoinGecko returned status {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching CoinGecko data: {e}")
            return {}
    
    async def _fetch_market_data(self, asset: str) -> Dict[str, Any]:
        """Fetch basic market data"""
        try:
            # Simplified market data collection
            return {
                "asset": asset,
                "timestamp": datetime.now().isoformat(),
                "price_usd": 0.0,  # Would be fetched from actual API
                "volume_24h": 0.0,
                "market_cap": 0.0,
                "price_change_24h": 0.0
            }
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}
    
    async def _collect_onchain_data(self, asset: str) -> Dict[str, Any]:
        """Collect on-chain blockchain data"""
        try:
            onchain_data = {}
            
            # Etherscan data for Ethereum
            if asset == "ETH" and self.api_keys.get("etherscan"):
                onchain_data["etherscan"] = await self._fetch_etherscan_data()
            
            # Glassnode data
            if self.api_keys.get("glassnode"):
                onchain_data["glassnode"] = await self._fetch_glassnode_data(asset)
            
            return onchain_data
            
        except Exception as e:
            logger.error(f"Error collecting on-chain data: {e}")
            return {}
    
    async def _fetch_etherscan_data(self) -> Dict[str, Any]:
        """Fetch data from Etherscan"""
        try:
            url = "https://api.etherscan.io/api"
            params = {
                "module": "proxy",
                "action": "eth_blockNumber",
                "apikey": self.api_keys["etherscan"]
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "latest_block": data.get("result"),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    logger.warning(f"Etherscan returned status {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching Etherscan data: {e}")
            return {}
    
    async def _fetch_glassnode_data(self, asset: str) -> Dict[str, Any]:
        """Fetch data from Glassnode"""
        try:
            # Glassnode API implementation
            # This is a simplified example
            return {
                "active_addresses": 0,
                "transaction_count": 0,
                "network_hash_rate": 0
            }
            
        except Exception as e:
            logger.error(f"Error fetching Glassnode data: {e}")
            return {}
    
    async def _collect_expert_opinions(self, asset: str) -> Dict[str, Any]:
        """Collect expert opinions and analysis"""
        try:
            expert_data = {}
            
            # Collect from various expert sources
            expert_data["analyst_reports"] = await self._fetch_analyst_reports(asset)
            expert_data["podcast_insights"] = await self._fetch_podcast_insights(asset)
            expert_data["blog_analysis"] = await self._fetch_blog_analysis(asset)
            
            return expert_data
            
        except Exception as e:
            logger.error(f"Error collecting expert opinions: {e}")
            return {}
    
    async def _fetch_analyst_reports(self, asset: str) -> List[Dict[str, Any]]:
        """Fetch analyst reports and research"""
        try:
            # This would integrate with various financial research platforms
            return [
                {
                    "source": "Example Research Firm",
                    "title": f"{asset} Market Analysis",
                    "summary": "Sample analysis summary",
                    "rating": "bullish",
                    "timestamp": datetime.now().isoformat()
                }
            ]
            
        except Exception as e:
            logger.error(f"Error fetching analyst reports: {e}")
            return []
    
    async def _fetch_podcast_insights(self, asset: str) -> List[Dict[str, Any]]:
        """Fetch insights from cryptocurrency podcasts"""
        try:
            # This would integrate with podcast platforms and transcripts
            return [
                {
                    "podcast": "Example Crypto Podcast",
                    "episode": f"{asset} Market Update",
                    "key_points": ["Sample insight 1", "Sample insight 2"],
                    "sentiment": "neutral",
                    "timestamp": datetime.now().isoformat()
                }
            ]
            
        except Exception as e:
            logger.error(f"Error fetching podcast insights: {e}")
            return []
    
    async def _fetch_blog_analysis(self, asset: str) -> List[Dict[str, Any]]:
        """Fetch analysis from cryptocurrency blogs"""
        try:
            # This would integrate with various crypto blog platforms
            return [
                {
                    "blog": "Example Crypto Blog",
                    "title": f"{asset} Technical Analysis",
                    "author": "Sample Analyst",
                    "summary": "Sample blog summary",
                    "sentiment": "bullish",
                    "timestamp": datetime.now().isoformat()
                }
            ]
            
        except Exception as e:
            logger.error(f"Error fetching blog analysis: {e}")
            return []
    
    def get_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of collected data"""
        try:
            summary = {
                "total_sources": len([k for k, v in data.items() if v and k != "timestamp" and k != "asset"]),
                "data_quality": "high" if len([k for k, v in data.items() if v and k != "timestamp" and k != "asset"]) >= 3 else "medium",
                "last_updated": data.get("timestamp"),
                "asset": data.get("asset"),
                "data_points": {}
            }
            
            for key, value in data.items():
                if key not in ["timestamp", "asset"] and value:
                    if isinstance(value, dict):
                        summary["data_points"][key] = len(value)
                    elif isinstance(value, list):
                        summary["data_points"][key] = len(value)
                    else:
                        summary["data_points"][key] = 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {e}")
            return {"error": str(e)}

# Utility functions for external use
async def collect_market_data(asset: str = "ETH") -> Dict[str, Any]:
    """Utility function to collect market data"""
    async with DataCollector() as collector:
        return await collector.collect_comprehensive_data(asset)

async def collect_news_sentiment(asset: str = "ETH") -> Dict[str, Any]:
    """Utility function to collect news and sentiment data"""
    async with DataCollector() as collector:
        news_data = await collector._collect_news_data(asset)
        social_data = await collector._collect_social_sentiment(asset)
        
        return {
            "news_data": news_data,
            "social_sentiment": social_data,
            "timestamp": datetime.now().isoformat(),
            "asset": asset
        }
