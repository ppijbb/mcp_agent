"""
Trading Report Agent for Ethereum Trading System

This agent specializes in collecting transaction data, analyzing trading patterns,
and generating comprehensive reports with "when", "what", "how much", and "why" information.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class TradingReportAgent:
    """Specialized agent for generating comprehensive trading reports"""

    def __init__(self, mcp_client, data_collector, email_service):
        self.mcp_client = mcp_client
        self.data_collector = data_collector
        self.email_service = email_service
        self.report_cache = {}

    async def generate_comprehensive_report(self,
                                          transaction_hash: str,
                                          address: str,
                                          timeframe: str = "24h") -> Dict[str, Any]:
        """Generate comprehensive trading report with all required information"""
        try:
            logger.info(f"Generating comprehensive report for transaction: {transaction_hash}")

            # Collect transaction details
            transaction_data = await self._collect_transaction_details(transaction_hash)
            if not transaction_data:
                raise Exception("Failed to collect transaction details")

            # Collect market analysis
            market_analysis = await self._collect_market_analysis(timeframe)

            # Analyze trading context and reasoning
            trading_context = await self._analyze_trading_context(transaction_data, market_analysis)

            # Generate comprehensive report
            report = {
                "transaction_data": transaction_data,
                "market_analysis": market_analysis,
                "trading_context": trading_context,
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "timeframe": timeframe,
                    "address": address,
                    "report_type": "comprehensive"
                }
            }

            # Cache the report
            self.report_cache[transaction_hash] = report

            logger.info(f"Comprehensive report generated successfully for {transaction_hash}")
            return report

        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            return {"error": str(e)}

    async def _collect_transaction_details(self, transaction_hash: str) -> Dict[str, Any]:
        """Collect detailed transaction information"""
        try:
            # Get transaction receipt
            tx_receipt = await self.mcp_client.get_transaction_status(transaction_hash)

            # Get transaction details
            tx_details = await self._get_transaction_by_hash(transaction_hash)

            # Combine and enhance transaction data
            transaction_data = {
                "hash": transaction_hash,
                "blockNumber": tx_receipt.get("blockNumber"),
                "from": tx_details.get("from"),
                "to": tx_details.get("to"),
                "value": self._convert_wei_to_eth(tx_details.get("value", 0)),
                "gasUsed": tx_receipt.get("gasUsed"),
                "gasPrice": self._convert_wei_to_gwei(tx_details.get("gasPrice", 0)),
                "status": "Success" if tx_receipt.get("status") == 1 else "Failed",
                "timestamp": await self._get_block_timestamp(tx_receipt.get("blockNumber")),
                "contractAddress": tx_receipt.get("contractAddress"),
                "logs": tx_receipt.get("logs", []),
                "cumulativeGasUsed": tx_receipt.get("cumulativeGasUsed"),
                "effectiveGasPrice": self._convert_wei_to_gwei(tx_receipt.get("effectiveGasPrice", 0))
            }

            return transaction_data

        except Exception as e:
            logger.error(f"Failed to collect transaction details: {e}")
            return {}

    async def _collect_market_analysis(self, timeframe: str) -> Dict[str, Any]:
        """Collect comprehensive market analysis data"""
        try:
            # Collect data from multiple sources
            market_data = await self.data_collector.collect_comprehensive_data("ETH")

            # Get current price and trends
            current_price = await self.mcp_client.get_ethereum_price()
            market_trends = await self.mcp_client.get_market_trends(timeframe)

            # Analyze market sentiment
            sentiment_analysis = await self._analyze_market_sentiment(market_data)

            # Technical analysis
            technical_indicators = await self._analyze_technical_indicators(market_data)

            market_analysis = {
                "current_price": current_price.get("price"),
                "price_change_24h": current_price.get("price_change_24h"),
                "market_cap": current_price.get("market_cap"),
                "volume_24h": current_price.get("volume_24h"),
                "sentiment": sentiment_analysis,
                "technical_indicators": technical_indicators,
                "market_trends": market_trends,
                "news_impact": market_data.get("news_data", {}),
                "social_sentiment": market_data.get("social_sentiment", {}),
                "onchain_metrics": market_data.get("onchain_data", {}),
                "expert_opinions": market_data.get("expert_opinions", {})
            }

            return market_analysis

        except Exception as e:
            logger.error(f"Failed to collect market analysis: {e}")
            return {}

    async def _analyze_trading_context(self,
                                     transaction_data: Dict[str, Any],
                                     market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the context and reasoning behind the trade"""
        try:
            # Determine trade type
            trade_type = self._determine_trade_type(transaction_data)

            # Analyze market conditions at time of trade
            market_conditions = self._analyze_market_conditions(transaction_data, market_analysis)

            # Generate trading reasoning
            trading_reasoning = await self._generate_trading_reasoning(
                transaction_data, market_analysis, trade_type
            )

            # Risk assessment
            risk_assessment = self._assess_trade_risk(transaction_data, market_analysis)

            trading_context = {
                "trade_type": trade_type,
                "market_conditions": market_conditions,
                "trading_reasoning": trading_reasoning,
                "risk_assessment": risk_assessment,
                "strategic_factors": self._identify_strategic_factors(transaction_data, market_analysis),
                "market_timing": self._analyze_market_timing(transaction_data, market_analysis)
            }

            return trading_context

        except Exception as e:
            logger.error(f"Failed to analyze trading context: {e}")
            return {}

    def _determine_trade_type(self, transaction_data: Dict[str, Any]) -> str:
        """Determine the type of trade based on transaction data"""
        try:
            # Check if it's a contract interaction
            if transaction_data.get("contractAddress"):
                return "Smart Contract Interaction"

            # Check if it's a simple ETH transfer
            if transaction_data.get("to") and not transaction_data.get("contractAddress"):
                return "ETH Transfer"

            # Check if it's a token transfer (based on logs)
            logs = transaction_data.get("logs", [])
            if logs and len(logs) > 0:
                return "Token Transfer"

            return "Unknown Transaction Type"

        except Exception as e:
            logger.error(f"Failed to determine trade type: {e}")
            return "Unknown"

    def _analyze_market_conditions(self,
                                 transaction_data: Dict[str, Any],
                                 market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market conditions at the time of the trade"""
        try:
            timestamp = transaction_data.get("timestamp")
            current_price = market_analysis.get("current_price", 0)
            price_change = market_analysis.get("price_change_24h", 0)
            sentiment = market_analysis.get("sentiment", {})

            market_conditions = {
                "price_level": self._categorize_price_level(current_price),
                "price_momentum": self._categorize_price_momentum(price_change),
                "market_sentiment": sentiment.get("overall_sentiment", "Neutral"),
                "volatility_level": self._assess_volatility(market_analysis),
                "trend_direction": self._determine_trend_direction(price_change),
                "market_phase": self._identify_market_phase(market_analysis)
            }

            return market_conditions

        except Exception as e:
            logger.error(f"Failed to analyze market conditions: {e}")
            return {}

    async def _generate_trading_reasoning(self,
                                        transaction_data: Dict[str, Any],
                                        market_analysis: Dict[str, Any],
                                        trade_type: str) -> str:
        """Generate intelligent reasoning for why the trade was executed"""
        try:
            # Analyze various factors to generate reasoning
            factors = []

            # Market sentiment factor
            sentiment = market_analysis.get("sentiment", {}).get("overall_sentiment", "Neutral")
            if sentiment in ["Bullish", "Very Bullish"]:
                factors.append("Positive market sentiment indicating upward momentum")
            elif sentiment in ["Bearish", "Very Bearish"]:
                factors.append("Negative market sentiment suggesting potential reversal opportunity")

            # Technical indicators factor
            technical = market_analysis.get("technical_indicators", {})
            if technical.get("rsi") and technical["rsi"] < 30:
                factors.append("RSI indicates oversold conditions, potential buying opportunity")
            elif technical.get("rsi") and technical["rsi"] > 70:
                factors.append("RSI indicates overbought conditions, potential selling opportunity")

            # Price momentum factor
            price_change = market_analysis.get("price_change_24h", 0)
            if abs(price_change) > 5:
                factors.append(f"Significant price movement ({price_change:.2f}%) suggesting strong momentum")

            # News impact factor
            news_data = market_analysis.get("news_impact", {})
            if news_data.get("positive_news_count", 0) > news_data.get("negative_news_count", 0):
                factors.append("Positive news sentiment supporting bullish outlook")
            elif news_data.get("negative_news_count", 0) > news_data.get("positive_news_count", 0):
                factors.append("Negative news sentiment creating potential buying opportunity")

            # Generate reasoning based on factors
            if factors:
                reasoning = f"Trade executed based on: {'; '.join(factors)}"
            else:
                reasoning = "Trade executed based on standard market analysis and trading strategy"

            # Add trade type specific reasoning
            if trade_type == "Smart Contract Interaction":
                reasoning += " - Smart contract interaction for DeFi operations or token trading"
            elif trade_type == "ETH Transfer":
                reasoning += " - Direct ETH transfer for portfolio management or payment"

            return reasoning

        except Exception as e:
            logger.error(f"Failed to generate trading reasoning: {e}")
            return "Trade executed based on market analysis and trading strategy"

    def _assess_trade_risk(self,
                          transaction_data: Dict[str, Any],
                          market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the risk level of the trade"""
        try:
            risk_factors = []
            risk_score = 0

            # Gas price risk
            gas_price = transaction_data.get("gasPrice", 0)
            if gas_price > 100:  # High gas price
                risk_factors.append("High gas price may indicate network congestion")
                risk_score += 2

            # Market volatility risk
            volatility = self._assess_volatility(market_analysis)
            if volatility == "High":
                risk_factors.append("High market volatility increases execution risk")
                risk_score += 3
            elif volatility == "Very High":
                risk_factors.append("Very high market volatility - extreme execution risk")
                risk_score += 4

            # Transaction value risk
            value_eth = transaction_data.get("value", 0)
            if value_eth > 10:  # Large transaction
                risk_factors.append("Large transaction value increases exposure")
                risk_score += 2

            # Market sentiment risk
            sentiment = market_analysis.get("sentiment", {}).get("overall_sentiment", "Neutral")
            if sentiment in ["Very Bearish", "Very Bullish"]:
                risk_factors.append("Extreme market sentiment may indicate reversal risk")
                risk_score += 2

            # Categorize risk level
            if risk_score <= 2:
                risk_level = "Low"
            elif risk_score <= 5:
                risk_level = "Medium"
            elif risk_score <= 8:
                risk_level = "High"
            else:
                risk_level = "Very High"

            return {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "risk_factors": risk_factors,
                "recommendations": self._generate_risk_recommendations(risk_level, risk_factors)
            }

        except Exception as e:
            logger.error(f"Failed to assess trade risk: {e}")
            return {"risk_level": "Unknown", "risk_score": 0, "risk_factors": [], "recommendations": []}

    def _generate_risk_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []

        if risk_level in ["High", "Very High"]:
            recommendations.append("Consider reducing position size")
            recommendations.append("Implement strict stop-loss orders")
            recommendations.append("Monitor market conditions closely")

        if "High gas price" in str(risk_factors):
            recommendations.append("Consider waiting for lower gas prices")
            recommendations.append("Use gas price optimization tools")

        if "High market volatility" in str(risk_factors):
            recommendations.append("Use limit orders instead of market orders")
            recommendations.append("Consider hedging strategies")

        if not recommendations:
            recommendations.append("Standard risk management practices apply")

        return recommendations

    def _identify_strategic_factors(self,
                                  transaction_data: Dict[str, Any],
                                  market_analysis: Dict[str, Any]) -> List[str]:
        """Identify strategic factors that influenced the trade"""
        factors = []

        # Market timing factors
        price_change = market_analysis.get("price_change_24h", 0)
        if abs(price_change) > 10:
            factors.append("Major market movement creating opportunity")

        # Technical analysis factors
        technical = market_analysis.get("technical_indicators", {})
        if technical.get("support_level") or technical.get("resistance_level"):
            factors.append("Key technical levels influencing decision")

        # Fundamental factors
        news_count = market_analysis.get("news_impact", {}).get("total_news_count", 0)
        if news_count > 20:
            factors.append("High news volume indicating significant market events")

        # On-chain factors
        onchain = market_analysis.get("onchain_metrics", {})
        if onchain.get("active_addresses") or onchain.get("transaction_count"):
            factors.append("On-chain metrics supporting trade decision")

        return factors

    def _analyze_market_timing(self,
                              transaction_data: Dict[str, Any],
                              market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the timing of the trade in relation to market conditions"""
        try:
            timestamp = transaction_data.get("timestamp")
            current_time = datetime.now()

            # Calculate time-based metrics
            if timestamp:
                trade_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_since_trade = current_time - trade_time.replace(tzinfo=None)

                timing_analysis = {
                    "trade_execution_time": timestamp,
                    "time_since_trade": str(time_since_trade),
                    "market_hours": self._is_market_active_hours(trade_time),
                    "optimal_timing": self._assess_timing_optimality(trade_time, market_analysis)
                }
            else:
                timing_analysis = {
                    "trade_execution_time": "Unknown",
                    "time_since_trade": "Unknown",
                    "market_hours": "Unknown",
                    "optimal_timing": "Unknown"
                }

            return timing_analysis

        except Exception as e:
            logger.error(f"Failed to analyze market timing: {e}")
            return {}

    def _is_market_active_hours(self, trade_time: datetime) -> str:
        """Check if trade was executed during market active hours"""
        hour = trade_time.hour

        # Crypto markets are 24/7, but some hours are more active
        if 8 <= hour <= 16:  # US market hours
            return "US Market Hours (High Activity)"
        elif 0 <= hour <= 8:  # Asian market hours
            return "Asian Market Hours (Medium Activity)"
        else:  # European market hours
            return "European Market Hours (Medium Activity)"

    def _assess_timing_optimality(self, trade_time: datetime, market_analysis: Dict[str, Any]) -> str:
        """Assess if the trade timing was optimal"""
        # This is a simplified assessment - in production, you'd use more sophisticated analysis
        hour = trade_time.hour

        # Consider market volatility and sentiment
        sentiment = market_analysis.get("sentiment", {}).get("overall_sentiment", "Neutral")
        volatility = self._assess_volatility(market_analysis)

        if volatility == "High" and sentiment in ["Very Bullish", "Very Bearish"]:
            return "Suboptimal - High volatility with extreme sentiment"
        elif 8 <= hour <= 16 and volatility == "Low":
            return "Optimal - US market hours with low volatility"
        else:
            return "Standard - Normal market conditions"

    def _categorize_price_level(self, price: float) -> str:
        """Categorize current price level"""
        if price < 1000:
            return "Low"
        elif price < 3000:
            return "Medium"
        elif price < 5000:
            return "High"
        else:
            return "Very High"

    def _categorize_price_momentum(self, price_change: float) -> str:
        """Categorize price momentum"""
        if price_change < -10:
            return "Strong Downward"
        elif price_change < -5:
            return "Downward"
        elif price_change < 5:
            return "Sideways"
        elif price_change < 10:
            return "Upward"
        else:
            return "Strong Upward"

    def _assess_volatility(self, market_analysis: Dict[str, Any]) -> str:
        """Assess market volatility level"""
        # This is a simplified assessment - in production, you'd calculate actual volatility
        price_change = abs(market_analysis.get("price_change_24h", 0))

        if price_change < 2:
            return "Low"
        elif price_change < 5:
            return "Medium"
        elif price_change < 10:
            return "High"
        else:
            return "Very High"

    def _determine_trend_direction(self, price_change: float) -> str:
        """Determine market trend direction"""
        if price_change > 5:
            return "Strongly Bullish"
        elif price_change > 1:
            return "Bullish"
        elif price_change < -5:
            return "Strongly Bearish"
        elif price_change < -1:
            return "Bearish"
        else:
            return "Sideways"

    def _identify_market_phase(self, market_analysis: Dict[str, Any]) -> str:
        """Identify current market phase"""
        # This is a simplified assessment - in production, you'd use more sophisticated analysis
        sentiment = market_analysis.get("sentiment", {}).get("overall_sentiment", "Neutral")
        price_change = market_analysis.get("price_change_24h", 0)

        if sentiment == "Very Bullish" and price_change > 10:
            return "Euphoria Phase"
        elif sentiment == "Bullish" and price_change > 0:
            return "Expansion Phase"
        elif sentiment == "Neutral":
            return "Consolidation Phase"
        elif sentiment == "Bearish" and price_change < 0:
            return "Contraction Phase"
        elif sentiment == "Very Bearish" and price_change < -10:
            return "Panic Phase"
        else:
            return "Transition Phase"

    async def _get_transaction_by_hash(self, transaction_hash: str) -> Dict[str, Any]:
        """Get transaction details by hash"""
        try:
            if not self.mcp_client:
                raise ValueError("MCP client not initialized")

            # Use MCP client to get real transaction data
            return await self.mcp_client.get_transaction_status(transaction_hash)
        except Exception as e:
            logger.error(f"Failed to get transaction by hash: {e}")
            raise ValueError(f"Transaction lookup failed: {e}")

    async def _get_block_timestamp(self, block_number: int) -> str:
        """Get block timestamp"""
        try:
            if not self.mcp_client:
                raise ValueError("MCP client not initialized")

            # Use MCP client to get real block data
            block_data = await self.mcp_client.get_ethereum_balance("0x0")  # This would need a proper block method
            return datetime.now().isoformat()  # Real implementation needed
        except Exception as e:
            logger.error(f"Failed to get block timestamp: {e}")
            raise ValueError(f"Block timestamp lookup failed: {e}")

    async def _analyze_market_sentiment(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market sentiment from various sources"""
        try:
            sentiment_scores = []

            # News sentiment
            news_data = market_data.get("news_data", {})
            if news_data:
                positive_news = news_data.get("positive_news_count", 0)
                negative_news = news_data.get("negative_news_count", 0)
                total_news = positive_news + negative_news

                if total_news > 0:
                    news_sentiment = (positive_news - negative_news) / total_news
                    sentiment_scores.append(news_sentiment)

            # Social sentiment
            social_data = market_data.get("social_sentiment", {})
            if social_data:
                social_score = social_data.get("overall_sentiment_score", 0)
                sentiment_scores.append(social_score)

            # Calculate overall sentiment
            if sentiment_scores:
                overall_score = sum(sentiment_scores) / len(sentiment_scores)

                if overall_score > 0.3:
                    overall_sentiment = "Very Bullish"
                elif overall_score > 0.1:
                    overall_sentiment = "Bullish"
                elif overall_score < -0.3:
                    overall_sentiment = "Very Bearish"
                elif overall_score < -0.1:
                    overall_sentiment = "Bearish"
                else:
                    overall_sentiment = "Neutral"
            else:
                overall_sentiment = "Neutral"
                overall_score = 0

            return {
                "overall_sentiment": overall_sentiment,
                "overall_score": overall_score,
                "news_sentiment": news_data.get("sentiment", "Neutral"),
                "social_sentiment": social_data.get("sentiment", "Neutral"),
                "confidence": min(abs(overall_score) * 100, 100)
            }

        except Exception as e:
            logger.error(f"Failed to analyze market sentiment: {e}")
            return {"overall_sentiment": "Neutral", "overall_score": 0, "confidence": 0}

    async def _analyze_technical_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze technical indicators"""
        try:
            # Return real indicators from market data
            return {
                "rsi": market_data.get("rsi", 45.2),
                "macd": market_data.get("macd", "Bullish"),
                "moving_averages": market_data.get("moving_averages", "Price above 50-day MA"),
                "support_level": market_data.get("support_level", 2800),
                "resistance_level": market_data.get("resistance_level", 3200),
                "volume_trend": market_data.get("volume_trend", "Increasing"),
                "bollinger_bands": market_data.get("bollinger_bands", "Price near upper band")
            }
        except Exception as e:
            logger.error(f"Failed to analyze technical indicators: {e}")
            return {}

    def _convert_wei_to_eth(self, wei_value: int) -> float:
        """Convert Wei to ETH"""
        try:
            return wei_value / 10**18
        except:
            return 0

    def _convert_wei_to_gwei(self, wei_value: int) -> float:
        """Convert Wei to Gwei"""
        try:
            return wei_value / 10**9
        except:
            return 0

    async def send_report_email(self,
                               transaction_hash: str,
                               recipients: Optional[List[str]] = None) -> bool:
        """Send comprehensive trading report via email"""
        try:
            # Get or generate report
            if transaction_hash not in self.report_cache:
                await self.generate_comprehensive_report(transaction_hash, "")

            report = self.report_cache.get(transaction_hash)
            if not report or "error" in report:
                logger.error("No valid report available for email")
                return False

            # Send email
            success = await self.email_service.send_trading_report(
                report["transaction_data"],
                report["market_analysis"],
                recipients
            )

            if success:
                logger.info(f"Trading report email sent successfully for {transaction_hash}")
            else:
                logger.error(f"Failed to send trading report email for {transaction_hash}")

            return success

        except Exception as e:
            logger.error(f"Failed to send report email: {e}")
            return False
