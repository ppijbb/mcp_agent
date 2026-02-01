"""
MCP (Model Context Protocol) Integration Tools
Integration with MCP servers for enhanced data collection and analysis
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MCPServerType(Enum):
    WEB_SEARCH = "web_search"
    KIS_TRADING = "kis_trading"
    NEWS_ANALYSIS = "news_analysis"
    SOCIAL_MONITORING = "social_monitoring"
    BLOCKCHAIN_ANALYSIS = "blockchain_analysis"
    MARKET_RESEARCH = "market_research"


@dataclass
class MCPConfig:
    """Configuration for MCP server integrations"""
    web_search_mcp_url: Optional[str] = None
    kis_trading_mcp_url: Optional[str] = None
    news_analysis_mcp_url: Optional[str] = None
    social_monitoring_mcp_url: Optional[str] = None
    blockchain_analysis_mcp_url: Optional[str] = None
    market_research_mcp_url: Optional[str] = None
    request_timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 0.1


class MCPIntegrationManager:
    """Manager for MCP server integrations"""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.session = None
        self.rate_limiter = asyncio.Semaphore(5)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_comprehensive_analysis(self, symbol: str = "ethereum") -> Dict[str, Any]:
        """Get comprehensive analysis using multiple MCP servers"""
        try:
            async with self.rate_limiter:
                # Collect data from multiple MCP servers in parallel
                tasks = [
                    self._web_search_analysis(symbol),
                    self._news_analysis(symbol),
                    self._social_monitoring(symbol),
                    self._blockchain_analysis(symbol),
                    self._market_research(symbol)
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                analysis = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "status": "success",
                    "mcp_servers": {},
                    "aggregated_insights": {}
                }

                server_names = [
                    "web_search", "news_analysis", "social_monitoring",
                    "blockchain_analysis", "market_research"
                ]

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to get {server_names[i]} analysis: {result}")
                        analysis["mcp_servers"][server_names[i]] = {"status": "error", "error": str(result)}
                    else:
                        analysis["mcp_servers"][server_names[i]] = result

                # Aggregate insights from all MCP servers
                analysis["aggregated_insights"] = self._aggregate_mcp_insights(analysis["mcp_servers"])

                return analysis

        except Exception as e:
            logger.error(f"MCP analysis failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _web_search_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get web search analysis using MCP server"""
        try:
            if not self.config.web_search_mcp_url:
                return {"status": "error", "message": "Web search MCP URL not configured"}

            # Search queries for comprehensive analysis
            queries = [
                f"{symbol} price analysis",
                f"{symbol} market trends",
                f"{symbol} news today",
                f"{symbol} technical analysis",
                f"{symbol} fundamental analysis"
            ]

            search_results = []
            for query in queries:
                result = await self._call_mcp_server(
                    self.config.web_search_mcp_url,
                    "search",
                    {"query": query, "max_results": 5}
                )
                if result.get("status") == "success":
                    search_results.extend(result.get("results", []))

            # Analyze search results
            analysis = self._analyze_search_results(search_results, symbol)

            return {
                "status": "success",
                "source": "web_search_mcp",
                "data": {
                    "search_results": search_results,
                    "analysis": analysis,
                    "total_results": len(search_results)
                }
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _news_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get news analysis using MCP server"""
        try:
            if not self.config.news_analysis_mcp_url:
                return {"status": "error", "message": "News analysis MCP URL not configured"}

            # Get news analysis
            result = await self._call_mcp_server(
                self.config.news_analysis_mcp_url,
                "analyze_news",
                {
                    "symbol": symbol,
                    "timeframe": "24h",
                    "sources": ["reuters", "bloomberg", "coindesk", "cointelegraph"],
                    "analysis_type": "sentiment_and_impact"
                }
            )

            if result.get("status") == "success":
                return {
                    "status": "success",
                    "source": "news_analysis_mcp",
                    "data": result.get("data", {})
                }
            else:
                return {"status": "error", "message": result.get("error", "Unknown error")}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _social_monitoring(self, symbol: str) -> Dict[str, Any]:
        """Get social media monitoring using MCP server"""
        try:
            if not self.config.social_monitoring_mcp_url:
                return {"status": "error", "message": "Social monitoring MCP URL not configured"}

            # Get social media analysis
            result = await self._call_mcp_server(
                self.config.social_monitoring_mcp_url,
                "monitor_sentiment",
                {
                    "symbol": symbol,
                    "platforms": ["twitter", "reddit", "telegram", "discord"],
                    "timeframe": "24h",
                    "analysis_depth": "comprehensive"
                }
            )

            if result.get("status") == "success":
                return {
                    "status": "success",
                    "source": "social_monitoring_mcp",
                    "data": result.get("data", {})
                }
            else:
                return {"status": "error", "message": result.get("error", "Unknown error")}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _blockchain_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get blockchain analysis using MCP server"""
        try:
            if not self.config.blockchain_analysis_mcp_url:
                return {"status": "error", "message": "Blockchain analysis MCP URL not configured"}

            # Get blockchain analysis
            result = await self._call_mcp_server(
                self.config.blockchain_analysis_mcp_url,
                "analyze_blockchain",
                {
                    "symbol": symbol,
                    "analysis_types": [
                        "transaction_analysis",
                        "address_analysis",
                        "gas_analysis",
                        "network_health",
                        "defi_metrics"
                    ],
                    "timeframe": "7d"
                }
            )

            if result.get("status") == "success":
                return {
                    "status": "success",
                    "source": "blockchain_analysis_mcp",
                    "data": result.get("data", {})
                }
            else:
                return {"status": "error", "message": result.get("error", "Unknown error")}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _market_research(self, symbol: str) -> Dict[str, Any]:
        """Get market research using MCP server"""
        try:
            if not self.config.market_research_mcp_url:
                return {"status": "error", "message": "Market research MCP URL not configured"}

            # Get market research
            result = await self._call_mcp_server(
                self.config.market_research_mcp_url,
                "research_market",
                {
                    "symbol": symbol,
                    "research_areas": [
                        "competitor_analysis",
                        "market_trends",
                        "regulatory_analysis",
                        "adoption_metrics",
                        "institutional_activity"
                    ],
                    "depth": "comprehensive"
                }
            )

            if result.get("status") == "success":
                return {
                    "status": "success",
                    "source": "market_research_mcp",
                    "data": result.get("data", {})
                }
            else:
                return {"status": "error", "message": result.get("error", "Unknown error")}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _call_mcp_server(self, server_url: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP server with specified method and parameters"""
        try:
            payload = {
                "method": method,
                "params": params,
                "timestamp": datetime.now().isoformat()
            }

            async with self.session.post(server_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    return {
                        "status": "error",
                        "message": f"HTTP {response.status}",
                        "error": await response.text()
                    }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _analyze_search_results(self, search_results: List[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
        """Analyze web search results for insights"""
        try:
            if not search_results:
                return {"insights": [], "sentiment": "neutral", "key_themes": []}

            # Extract key themes and sentiment
            themes = {}
            sentiment_scores = []

            for result in search_results:
                title = result.get("title", "").lower()
                snippet = result.get("snippet", "").lower()
                text = f"{title} {snippet}"

                # Extract themes
                theme_keywords = {
                    "price": ["price", "cost", "value", "valuation"],
                    "technology": ["technology", "tech", "innovation", "upgrade"],
                    "adoption": ["adoption", "usage", "users", "growth"],
                    "regulation": ["regulation", "regulatory", "legal", "compliance"],
                    "partnership": ["partnership", "collaboration", "integration"],
                    "competition": ["competition", "competitor", "rival", "alternative"]
                }

                for theme, keywords in theme_keywords.items():
                    if any(keyword in text for keyword in keywords):
                        themes[theme] = themes.get(theme, 0) + 1

                # Simple sentiment analysis
                positive_words = ["bullish", "positive", "growth", "success", "breakthrough"]
                negative_words = ["bearish", "negative", "decline", "concern", "risk"]

                positive_count = sum(1 for word in positive_words if word in text)
                negative_count = sum(1 for word in negative_words if word in text)

                if positive_count + negative_count > 0:
                    sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
                    sentiment_scores.append(sentiment_score)

            # Calculate overall sentiment
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                if avg_sentiment > 0.1:
                    overall_sentiment = "bullish"
                elif avg_sentiment < -0.1:
                    overall_sentiment = "bearish"
                else:
                    overall_sentiment = "neutral"
            else:
                overall_sentiment = "neutral"
                avg_sentiment = 0.0

            # Get top themes
            top_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:5]

            return {
                "insights": search_results[:10],  # Top 10 results
                "sentiment": overall_sentiment,
                "sentiment_score": avg_sentiment,
                "key_themes": [theme for theme, count in top_themes],
                "theme_counts": dict(top_themes),
                "total_results": len(search_results)
            }

        except Exception as e:
            logger.error(f"Search results analysis failed: {e}")
            return {"insights": [], "sentiment": "neutral", "key_themes": []}

    def _aggregate_mcp_insights(self, mcp_servers: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate insights from all MCP servers"""
        try:
            aggregated = {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "key_insights": [],
                "risk_factors": [],
                "opportunities": [],
                "confidence": 0.0
            }

            sentiments = []
            insights = []
            risk_factors = []
            opportunities = []
            successful_servers = 0

            for server_name, server_data in mcp_servers.items():
                if server_data.get("status") == "success" and "data" in server_data:
                    data = server_data["data"]
                    successful_servers += 1

                    # Collect sentiment data
                    if "sentiment" in data:
                        sentiment = data["sentiment"]
                        if isinstance(sentiment, str):
                            sentiment_map = {"bullish": 0.5, "bearish": -0.5, "neutral": 0.0}
                            sentiments.append(sentiment_map.get(sentiment, 0.0))
                        elif isinstance(sentiment, (int, float)):
                            sentiments.append(sentiment)

                    if "sentiment_score" in data:
                        sentiments.append(data["sentiment_score"])

                    # Collect insights
                    if "insights" in data:
                        insights.extend(data["insights"] if isinstance(data["insights"], list) else [data["insights"]])

                    if "key_insights" in data:
                        insights.extend(data["key_insights"] if isinstance(data["key_insights"], list) else [data["key_insights"]])

                    # Collect risk factors
                    if "risk_factors" in data:
                        risk_factors.extend(data["risk_factors"] if isinstance(data["risk_factors"], list) else [data["risk_factors"]])

                    # Collect opportunities
                    if "opportunities" in data:
                        opportunities.extend(data["opportunities"] if isinstance(data["opportunities"], list) else [data["opportunities"]])

            # Calculate overall sentiment
            if sentiments:
                aggregated["sentiment_score"] = sum(sentiments) / len(sentiments)
                if aggregated["sentiment_score"] > 0.1:
                    aggregated["overall_sentiment"] = "bullish"
                elif aggregated["sentiment_score"] < -0.1:
                    aggregated["overall_sentiment"] = "bearish"
                else:
                    aggregated["overall_sentiment"] = "neutral"

            # Set aggregated data
            aggregated["key_insights"] = insights[:10]  # Top 10 insights
            aggregated["risk_factors"] = risk_factors[:5]  # Top 5 risk factors
            aggregated["opportunities"] = opportunities[:5]  # Top 5 opportunities

            # Calculate confidence based on successful servers
            total_servers = len(mcp_servers)
            aggregated["confidence"] = successful_servers / total_servers if total_servers > 0 else 0

            return aggregated

        except Exception as e:
            logger.error(f"MCP insights aggregation failed: {e}")
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "key_insights": [],
                "risk_factors": [],
                "opportunities": [],
                "confidence": 0.0
            }

    async def get_real_time_news(self, symbol: str, max_articles: int = 10) -> Dict[str, Any]:
        """Get real-time news using MCP server"""
        try:
            if not self.config.news_analysis_mcp_url:
                return {"status": "error", "message": "News analysis MCP URL not configured"}

            result = await self._call_mcp_server(
                self.config.news_analysis_mcp_url,
                "get_latest_news",
                {
                    "symbol": symbol,
                    "max_articles": max_articles,
                    "timeframe": "1h",
                    "sources": ["all"]
                }
            )

            return result

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def get_social_trends(self, symbol: str, platform: str = "all") -> Dict[str, Any]:
        """Get social media trends using MCP server"""
        try:
            if not self.config.social_monitoring_mcp_url:
                return {"status": "error", "message": "Social monitoring MCP URL not configured"}

            result = await self._call_mcp_server(
                self.config.social_monitoring_mcp_url,
                "get_trends",
                {
                    "symbol": symbol,
                    "platform": platform,
                    "timeframe": "24h",
                    "trend_type": "hashtags_and_mentions"
                }
            )

            return result

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def get_market_intelligence(self, symbol: str) -> Dict[str, Any]:
        """Get market intelligence using MCP server"""
        try:
            if not self.config.market_research_mcp_url:
                return {"status": "error", "message": "Market research MCP URL not configured"}

            result = await self._call_mcp_server(
                self.config.market_research_mcp_url,
                "get_intelligence",
                {
                    "symbol": symbol,
                    "intelligence_types": [
                        "institutional_activity",
                        "whale_movements",
                        "exchange_flows",
                        "derivatives_data",
                        "options_flow"
                    ]
                }
            )

            return result

        except Exception as e:
            return {"status": "error", "error": str(e)}
