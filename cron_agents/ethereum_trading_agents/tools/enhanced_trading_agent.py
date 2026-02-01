"""
Enhanced Trading Agent with External API and MCP Integration
Comprehensive trading agent using external data sources and MCP servers
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass

from .external_api_tools import ExternalAPIManager, ExternalAPIConfig
from .mcp_integration_tools import MCPIntegrationManager, MCPConfig
from .technical_analysis_tools import AdvancedTechnicalAnalyzer, TechnicalAnalysisConfig
from .fundamental_analysis_tools import AdvancedFundamentalAnalyzer, FundamentalAnalysisConfig
from .sentiment_analysis_tools import AdvancedSentimentAnalyzer, SentimentAnalysisConfig
from .risk_management_tools import AdvancedRiskManager, RiskManagementConfig
from .portfolio_management_tools import AdvancedPortfolioManager, PortfolioConfig
from .execution_tools import AdvancedExecutionManager, ExecutionConfig

logger = logging.getLogger(__name__)


@dataclass
class EnhancedTradingAgentConfig:
    """Configuration for enhanced trading agent"""
    # External API configuration
    external_api_config: ExternalAPIConfig

    # MCP configuration
    mcp_config: MCPConfig

    # Analysis tool configurations
    technical_config: TechnicalAnalysisConfig
    fundamental_config: FundamentalAnalysisConfig
    sentiment_config: SentimentAnalysisConfig

    # Management configurations
    risk_config: RiskManagementConfig
    portfolio_config: PortfolioConfig
    execution_config: ExecutionConfig

    # Agent settings
    symbol: str = "ethereum"
    analysis_interval: int = 300  # 5 minutes
    max_concurrent_analyses: int = 5


class EnhancedTradingAgent:
    """Enhanced trading agent with external API and MCP integration"""

    def __init__(self, config: EnhancedTradingAgentConfig):
        self.config = config
        self.external_api_manager = None
        self.mcp_manager = None
        self.technical_analyzer = AdvancedTechnicalAnalyzer(config.technical_config)
        self.fundamental_analyzer = AdvancedFundamentalAnalyzer(config.fundamental_config)
        self.sentiment_analyzer = AdvancedSentimentAnalyzer(config.sentiment_config)
        self.risk_manager = AdvancedRiskManager(config.risk_config)
        self.portfolio_manager = AdvancedPortfolioManager(config.portfolio_config)
        self.execution_manager = AdvancedExecutionManager(config.execution_config)

    async def __aenter__(self):
        self.external_api_manager = ExternalAPIManager(self.config.external_api_config)
        self.mcp_manager = MCPIntegrationManager(self.config.mcp_config)
        await self.external_api_manager.__aenter__()
        await self.mcp_manager.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.external_api_manager:
            await self.external_api_manager.__aexit__(exc_type, exc_val, exc_tb)
        if self.mcp_manager:
            await self.mcp_manager.__aexit__(exc_type, exc_val, exc_tb)

    async def execute_comprehensive_analysis(self) -> Dict[str, Any]:
        """Execute comprehensive analysis using all available tools"""
        try:
            logger.info(f"Starting comprehensive analysis for {self.config.symbol}")

            # Collect data from external sources in parallel
            analysis_tasks = [
                self._collect_external_market_data(),
                self._collect_mcp_analysis(),
                self._collect_technical_analysis(),
                self._collect_fundamental_analysis(),
                self._collect_sentiment_analysis()
            ]

            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            # Process results
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "symbol": self.config.symbol,
                "status": "success",
                "analysis_types": {},
                "integrated_insights": {},
                "trading_recommendations": [],
                "risk_assessment": {},
                "portfolio_analysis": {}
            }

            analysis_names = [
                "external_market_data", "mcp_analysis", "technical_analysis",
                "fundamental_analysis", "sentiment_analysis"
            ]

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Analysis {analysis_names[i]} failed: {result}")
                    analysis["analysis_types"][analysis_names[i]] = {"status": "error", "error": str(result)}
                else:
                    analysis["analysis_types"][analysis_names[i]] = result

            # Integrate insights from all analyses
            analysis["integrated_insights"] = await self._integrate_analysis_insights(analysis["analysis_types"])

            # Generate trading recommendations
            analysis["trading_recommendations"] = await self._generate_trading_recommendations(analysis)

            # Perform risk assessment
            analysis["risk_assessment"] = await self._perform_risk_assessment(analysis)

            # Analyze portfolio
            analysis["portfolio_analysis"] = await self._analyze_portfolio(analysis)

            logger.info("Comprehensive analysis completed successfully")
            return analysis

        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _collect_external_market_data(self) -> Dict[str, Any]:
        """Collect market data from external APIs"""
        try:
            if not self.external_api_manager:
                return {"status": "error", "message": "External API manager not initialized"}

            market_data = await self.external_api_manager.get_comprehensive_market_data(self.config.symbol)
            return market_data

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _collect_mcp_analysis(self) -> Dict[str, Any]:
        """Collect analysis from MCP servers"""
        try:
            if not self.mcp_manager:
                return {"status": "error", "message": "MCP manager not initialized"}

            mcp_analysis = await self.mcp_manager.get_comprehensive_analysis(self.config.symbol)
            return mcp_analysis

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _collect_technical_analysis(self) -> Dict[str, Any]:
        """Collect technical analysis"""
        try:
            # Get historical data for technical analysis
            if self.external_api_manager:
                market_data = await self.external_api_manager.get_comprehensive_market_data(self.config.symbol)
                if market_data.get("status") == "success":
                    # Convert to DataFrame for technical analysis
                    historical_data = self._prepare_historical_data(market_data)
                    technical_analysis = self.technical_analyzer.analyze_comprehensive(historical_data)
                    return technical_analysis

            return {"status": "error", "message": "No market data available for technical analysis"}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _collect_fundamental_analysis(self) -> Dict[str, Any]:
        """Collect fundamental analysis"""
        try:
            if not self.mcp_manager:
                return {"status": "error", "message": "MCP manager not initialized"}

            # Get blockchain analysis from MCP
            blockchain_analysis = await self.mcp_manager._blockchain_analysis(self.config.symbol)

            # Perform fundamental analysis
            fundamental_analysis = await self.fundamental_analyzer.analyze_comprehensive(self.config.symbol)

            # Combine MCP and fundamental analysis
            combined_analysis = {
                "status": "success",
                "mcp_blockchain_data": blockchain_analysis,
                "fundamental_analysis": fundamental_analysis
            }

            return combined_analysis

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _collect_sentiment_analysis(self) -> Dict[str, Any]:
        """Collect sentiment analysis"""
        try:
            if not self.mcp_manager:
                return {"status": "error", "message": "MCP manager not initialized"}

            # Get social sentiment from MCP
            social_sentiment = await self.mcp_manager._social_monitoring(self.config.symbol)

            # Get news sentiment from MCP
            news_sentiment = await self.mcp_manager._news_analysis(self.config.symbol)

            # Perform comprehensive sentiment analysis
            sentiment_analysis = await self.sentiment_analyzer.analyze_comprehensive_sentiment(self.config.symbol)

            # Combine all sentiment sources
            combined_sentiment = {
                "status": "success",
                "mcp_social_sentiment": social_sentiment,
                "mcp_news_sentiment": news_sentiment,
                "comprehensive_sentiment": sentiment_analysis
            }

            return combined_sentiment

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _prepare_historical_data(self, market_data: Dict[str, Any]) -> Any:
        """Prepare historical data for technical analysis"""
        try:
            import pandas as pd

            # Extract real historical data from market_data
            historical_data = market_data.get("historical_data", [])
            if not historical_data:
                raise ValueError("No historical data available for analysis")

            # Convert to pandas DataFrame
            df = pd.DataFrame(historical_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Failed to prepare historical data: {e}")
            return None

    async def _integrate_analysis_insights(self, analysis_types: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate insights from all analysis types"""
        try:
            integrated_insights = {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "technical_signal": "hold",
                "fundamental_score": 0.5,
                "risk_level": "medium",
                "confidence": 0.0,
                "key_factors": [],
                "market_outlook": "neutral"
            }

            # Extract sentiment data
            sentiments = []
            technical_signals = []
            fundamental_scores = []
            risk_levels = []
            successful_analyses = 0

            for analysis_name, analysis_data in analysis_types.items():
                if analysis_data.get("status") == "success":
                    successful_analyses += 1

                    # Extract sentiment
                    if "sentiment" in analysis_data:
                        sentiment = analysis_data["sentiment"]
                        if isinstance(sentiment, str):
                            sentiment_map = {"bullish": 0.5, "bearish": -0.5, "neutral": 0.0}
                            sentiments.append(sentiment_map.get(sentiment, 0.0))
                        elif isinstance(sentiment, (int, float)):
                            sentiments.append(sentiment)

                    if "sentiment_score" in analysis_data:
                        sentiments.append(analysis_data["sentiment_score"])

                    # Extract technical signals
                    if "signals" in analysis_data and "overall_signal" in analysis_data["signals"]:
                        technical_signals.append(analysis_data["signals"]["overall_signal"])

                    # Extract fundamental scores
                    if "overall_score" in analysis_data:
                        fundamental_scores.append(analysis_data["overall_score"])

                    # Extract risk levels
                    if "overall_risk_level" in analysis_data:
                        risk_levels.append(analysis_data["overall_risk_level"])

            # Calculate integrated metrics
            if sentiments:
                integrated_insights["sentiment_score"] = sum(sentiments) / len(sentiments)
                if integrated_insights["sentiment_score"] > 0.1:
                    integrated_insights["overall_sentiment"] = "bullish"
                elif integrated_insights["sentiment_score"] < -0.1:
                    integrated_insights["overall_sentiment"] = "bearish"

            if technical_signals:
                # Count signal types
                buy_signals = technical_signals.count("buy")
                sell_signals = technical_signals.count("sell")
                hold_signals = technical_signals.count("hold")

                if buy_signals > sell_signals and buy_signals > hold_signals:
                    integrated_insights["technical_signal"] = "buy"
                elif sell_signals > buy_signals and sell_signals > hold_signals:
                    integrated_insights["technical_signal"] = "sell"
                else:
                    integrated_insights["technical_signal"] = "hold"

            if fundamental_scores:
                integrated_insights["fundamental_score"] = sum(fundamental_scores) / len(fundamental_scores)

            if risk_levels:
                # Determine overall risk level
                risk_counts = {}
                for risk in risk_levels:
                    risk_counts[risk] = risk_counts.get(risk, 0) + 1

                integrated_insights["risk_level"] = max(risk_counts, key=risk_counts.get)

            # Calculate confidence
            total_analyses = len(analysis_types)
            integrated_insights["confidence"] = successful_analyses / total_analyses if total_analyses > 0 else 0

            # Determine market outlook
            if (integrated_insights["overall_sentiment"] == "bullish" and
                integrated_insights["technical_signal"] == "buy" and
                integrated_insights["fundamental_score"] > 0.6):
                integrated_insights["market_outlook"] = "bullish"
            elif (integrated_insights["overall_sentiment"] == "bearish" and
                  integrated_insights["technical_signal"] == "sell" and
                  integrated_insights["fundamental_score"] < 0.4):
                integrated_insights["market_outlook"] = "bearish"
            else:
                integrated_insights["market_outlook"] = "neutral"

            return integrated_insights

        except Exception as e:
            logger.error(f"Failed to integrate analysis insights: {e}")
            return {"overall_sentiment": "neutral", "sentiment_score": 0.0}

    async def _generate_trading_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading recommendations based on analysis"""
        try:
            recommendations = []
            integrated_insights = analysis.get("integrated_insights", {})

            # Technical recommendation
            technical_signal = integrated_insights.get("technical_signal", "hold")
            if technical_signal == "buy":
                recommendations.append({
                    "type": "technical",
                    "action": "buy",
                    "confidence": 0.8,
                    "reason": "Technical indicators suggest bullish momentum",
                    "priority": "high"
                })
            elif technical_signal == "sell":
                recommendations.append({
                    "type": "technical",
                    "action": "sell",
                    "confidence": 0.8,
                    "reason": "Technical indicators suggest bearish momentum",
                    "priority": "high"
                })

            # Sentiment recommendation
            sentiment = integrated_insights.get("overall_sentiment", "neutral")
            if sentiment == "bullish":
                recommendations.append({
                    "type": "sentiment",
                    "action": "buy",
                    "confidence": 0.7,
                    "reason": "Market sentiment is bullish",
                    "priority": "medium"
                })
            elif sentiment == "bearish":
                recommendations.append({
                    "type": "sentiment",
                    "action": "sell",
                    "confidence": 0.7,
                    "reason": "Market sentiment is bearish",
                    "priority": "medium"
                })

            # Fundamental recommendation
            fundamental_score = integrated_insights.get("fundamental_score", 0.5)
            if fundamental_score > 0.7:
                recommendations.append({
                    "type": "fundamental",
                    "action": "buy",
                    "confidence": 0.9,
                    "reason": "Strong fundamental analysis",
                    "priority": "high"
                })
            elif fundamental_score < 0.3:
                recommendations.append({
                    "type": "fundamental",
                    "action": "sell",
                    "confidence": 0.9,
                    "reason": "Weak fundamental analysis",
                    "priority": "high"
                })

            # Risk-based recommendation
            risk_level = integrated_insights.get("risk_level", "medium")
            if risk_level == "high":
                recommendations.append({
                    "type": "risk",
                    "action": "reduce_exposure",
                    "confidence": 0.8,
                    "reason": "High risk environment detected",
                    "priority": "critical"
                })

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate trading recommendations: {e}")
            return []

    async def _perform_risk_assessment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        try:
            if not self.risk_management:
                raise ValueError("Risk management not initialized")

            # Use real risk management with actual portfolio data
            portfolio_data = analysis.get("portfolio_data", {})
            market_data = analysis.get("market_data", {})

            return self.risk_management.calculate_comprehensive_risk(portfolio_data, market_data)

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            raise ValueError(f"Risk assessment failed: {e}")

    async def _analyze_portfolio(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current portfolio"""
        try:
            if not self.portfolio_management:
                raise ValueError("Portfolio management not initialized")

            # Use real portfolio management with actual portfolio data
            portfolio_data = analysis.get("portfolio_data", {})
            market_data = analysis.get("market_data", {})
            risk_metrics = analysis.get("risk_metrics", {})

            return self.portfolio_management.manage_portfolio(portfolio_data, market_data, risk_metrics)

        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            raise ValueError(f"Portfolio analysis failed: {e}")

    async def execute_trade(self, recommendation: Dict[str, Any],
                           market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade based on recommendation"""
        try:
            # Prepare trade decision
            trade_decision = {
                "action": recommendation.get("action", "hold"),
                "amount_eth": recommendation.get("amount_eth", 0),
                "target_price": market_data.get("aggregated_data", {}).get("price_usd", 0),
                "stop_loss": recommendation.get("stop_loss", 0),
                "take_profit": recommendation.get("take_profit", 0),
                "reason": recommendation.get("reason", ""),
                "risk_level": recommendation.get("risk_level", "medium")
            }

            # Get real account info
            if not self.execution_manager:
                raise ValueError("Execution manager not initialized")

            account_info = await self._get_account_info()

            # Execute trade
            execution_result = self.execution_manager.execute_trade(
                trade_decision, market_data, account_info
            )

            return execution_result

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            raise ValueError(f"Trade execution failed: {e}")

    async def _get_account_info(self) -> Dict[str, Any]:
        """Get real account information"""
        try:
            if not self.mcp_client:
                raise ValueError("MCP client not initialized")

            # Get real account balance
            balance_result = await self.mcp_client.get_ethereum_balance(
                self.config.external_api_config.ethereum_address
            )

            return {
                "balance_eth": balance_result.get("balance_eth", 0),
                "total_value": balance_result.get("total_value", 0),
                "address": self.config.external_api_config.ethereum_address
            }

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise ValueError(f"Account info retrieval failed: {e}")
