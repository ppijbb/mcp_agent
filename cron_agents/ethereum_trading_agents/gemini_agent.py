"""
Gemini 2.5 Flash AI Agent for Ethereum Trading
"""

import google.genai as genai
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)

class GeminiAgent:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """Initialize Gemini agent"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.chat = self.model.start_chat(history=[])
            logger.info(f"Gemini agent initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini agent: {e}")
            raise
    
    async def analyze_market_data(self, market_data: Dict, technical_indicators: Dict, 
                                historical_trends: List[Dict]) -> Dict[str, Any]:
        """Analyze market data and generate trading insights"""
        try:
            # Prepare context for analysis
            context = self._prepare_analysis_context(market_data, technical_indicators, historical_trends)
            
            # Generate analysis prompt
            prompt = self._generate_analysis_prompt(context)
            
            # Get AI response
            response = await self._get_ai_response(prompt)
            
            # Parse and validate response
            analysis = self._parse_analysis_response(response)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "confidence_score": analysis.get("confidence_score", 0.0),
                "recommended_action": analysis.get("recommended_action", "hold"),
                "reasoning": analysis.get("reasoning", ""),
                "risk_assessment": analysis.get("risk_assessment", "medium")
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze market data: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error_message": str(e),
                "recommended_action": "hold"
            }
    
    async def generate_trading_decision(self, market_analysis: Dict, risk_profile: Dict,
                                      account_balance: float) -> Dict[str, Any]:
        """Generate trading decision based on analysis and risk profile"""
        try:
            # Prepare decision context
            context = self._prepare_decision_context(market_analysis, risk_profile, account_balance)
            
            # Generate decision prompt
            prompt = self._generate_decision_prompt(context)
            
            # Get AI response
            response = await self._get_ai_response(prompt)
            
            # Parse and validate decision
            decision = self._parse_decision_response(response)
            
            # Validate decision against risk limits
            validated_decision = self._validate_decision(decision, risk_profile, account_balance)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "decision": validated_decision,
                "execution_ready": validated_decision.get("action") != "hold"
            }
            
        except Exception as e:
            logger.error(f"Failed to generate trading decision: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error_message": str(e),
                "decision": {"action": "hold", "reason": "Error in decision generation"}
            }
    
    async def search_market_insights(self, query: str, market_context: Dict) -> Dict[str, Any]:
        """Search for market insights and news analysis"""
        try:
            # Prepare search context
            context = self._prepare_search_context(query, market_context)
            
            # Generate search prompt
            prompt = self._generate_search_prompt(context)
            
            # Get AI response
            response = await self._get_ai_response(prompt)
            
            # Parse search results
            insights = self._parse_search_response(response)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "insights": insights,
                "relevance_score": insights.get("relevance_score", 0.0),
                "sentiment": insights.get("sentiment", "neutral")
            }
            
        except Exception as e:
            logger.error(f"Failed to search market insights: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error_message": str(e),
                "insights": {"summary": "", "relevance_score": 0.0, "sentiment": "neutral"}
            }
    
    def _prepare_analysis_context(self, market_data: Dict, technical_indicators: Dict, 
                                 historical_trends: List[Dict]) -> str:
        """Prepare context for market analysis"""
        context = f"""
        Market Data Analysis Context:
        
        Current Market Data:
        - Price: ${market_data.get('price_usd', 0):.2f}
        - 24h Change: {market_data.get('price_change_24h_percent', 0):.2f}%
        - Volume: ${market_data.get('volume_24h', 0):,.0f}
        - Trend: {market_data.get('trend', 'unknown')}
        
        Technical Indicators:
        - RSI: {technical_indicators.get('rsi', 0):.2f}
        - SMA 7: ${technical_indicators.get('sma_7', 0):.2f}
        - SMA 30: ${technical_indicators.get('sma_30', 0):.2f}
        - Signal: {technical_indicators.get('signal', 'hold')}
        
        Historical Trends (Last 24h):
        {json.dumps(historical_trends[-5:], indent=2) if historical_trends else 'No historical data'}
        
        Analyze the current market conditions and provide trading insights.
        """
        return context
    
    def _generate_analysis_prompt(self, context: str) -> str:
        """Generate prompt for market analysis"""
        return f"""
        {context}
        
        Based on the above market data, provide a comprehensive analysis including:
        1. Market sentiment assessment
        2. Technical analysis summary
        3. Risk factors identification
        4. Trading opportunity evaluation
        5. Confidence score (0-100)
        6. Recommended action (buy/sell/hold)
        7. Detailed reasoning
        
        Format your response as JSON with the following structure:
        {{
            "market_sentiment": "bullish/bearish/neutral",
            "technical_summary": "brief technical analysis",
            "risk_factors": ["risk1", "risk2"],
            "opportunity_score": 0-100,
            "confidence_score": 0-100,
            "recommended_action": "buy/sell/hold",
            "reasoning": "detailed explanation",
            "risk_assessment": "low/medium/high"
        }}
        """
    
    def _prepare_decision_context(self, market_analysis: Dict, risk_profile: Dict, 
                                 account_balance: float) -> str:
        """Prepare context for trading decision"""
        context = f"""
        Trading Decision Context:
        
        Market Analysis:
        {json.dumps(market_analysis, indent=2)}
        
        Risk Profile:
        - Max Daily Loss: {risk_profile.get('max_daily_loss_eth', 0)} ETH
        - Max Trade Amount: {risk_profile.get('max_trade_amount_eth', 0)} ETH
        - Risk Tolerance: {risk_profile.get('risk_tolerance', 'medium')}
        
        Account Balance: {account_balance:.4f} ETH
        
        Generate a trading decision based on the analysis and risk constraints.
        """
        return context
    
    def _generate_decision_prompt(self, context: str) -> str:
        """Generate prompt for trading decision"""
        return f"""
        {context}
        
        Based on the market analysis and risk profile, provide a trading decision:
        
        Format your response as JSON with the following structure:
        {{
            "action": "buy/sell/hold",
            "amount_eth": 0.0,
            "target_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "reason": "detailed explanation",
            "risk_level": "low/medium/high",
            "expected_return": "percentage estimate"
        }}
        
        Important constraints:
        - Only recommend buy if confidence > 70
        - Amount must be within risk limits
        - Include proper risk management levels
        """
    
    def _prepare_search_context(self, query: str, market_context: Dict) -> str:
        """Prepare context for market insights search"""
        return f"""
        Market Insights Search Context:
        
        Query: {query}
        
        Current Market Context:
        - Price: ${market_context.get('price_usd', 0):.2f}
        - Trend: {market_context.get('trend', 'unknown')}
        - Volume: {market_context.get('volume_trend', 'unknown')}
        
        Search for relevant insights and analyze their impact on trading decisions.
        """
    
    def _generate_search_prompt(self, context: str) -> str:
        """Generate prompt for market insights search"""
        return f"""
        {context}
        
        Based on the search query and market context, provide insights analysis:
        
        Format your response as JSON with the following structure:
        {{
            "summary": "key insights summary",
            "relevance_score": 0-100,
            "sentiment": "positive/negative/neutral",
            "trading_implications": "how this affects trading",
            "confidence": "high/medium/low"
        }}
        """
    
    async def _get_ai_response(self, prompt: str) -> str:
        """Get response from Gemini AI"""
        try:
            response = await self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Failed to get AI response: {e}")
            raise
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse AI analysis response"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback parsing
                return self._fallback_parse_analysis(response)
                
        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            return self._fallback_parse_analysis(response)
    
    def _parse_decision_response(self, response: str) -> Dict[str, Any]:
        """Parse AI decision response"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback parsing
                return self._fallback_parse_decision(response)
                
        except Exception as e:
            logger.error(f"Failed to parse decision response: {e}")
            return self._fallback_parse_decision(response)
    
    def _parse_search_response(self, response: str) -> Dict[str, Any]:
        """Parse AI search response"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback parsing
                return self._fallback_parse_search(response)
                
        except Exception as e:
            logger.error(f"Failed to parse search response: {e}")
            return self._fallback_parse_search(response)
    
    def _fallback_parse_analysis(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for analysis response"""
        return {
            "market_sentiment": "neutral",
            "technical_summary": "Unable to parse technical analysis",
            "risk_factors": ["parsing_error"],
            "opportunity_score": 50,
            "confidence_score": 30,
            "recommended_action": "hold",
            "reasoning": f"Fallback due to parsing error. Raw response: {response[:200]}...",
            "risk_assessment": "high"
        }
    
    def _fallback_parse_decision(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for decision response"""
        return {
            "action": "hold",
            "amount_eth": 0.0,
            "target_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "reason": f"Fallback due to parsing error. Raw response: {response[:200]}...",
            "risk_level": "high",
            "expected_return": "0%"
        }
    
    def _fallback_parse_search(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for search response"""
        return {
            "summary": f"Fallback response due to parsing error: {response[:200]}...",
            "relevance_score": 0,
            "sentiment": "neutral",
            "trading_implications": "Unable to determine implications",
            "confidence": "low"
        }
    
    def _validate_decision(self, decision: Dict, risk_profile: Dict, account_balance: float) -> Dict:
        """Validate trading decision against risk limits"""
        try:
            action = decision.get("action", "hold")
            amount = float(decision.get("amount_eth", 0))
            
            # Validate amount
            if action in ["buy", "sell"] and amount > 0:
                max_amount = min(risk_profile.get("max_trade_amount_eth", 0), account_balance)
                if amount > max_amount:
                    decision["amount_eth"] = max_amount
                    decision["reason"] += f" Amount adjusted to {max_amount} ETH due to risk limits."
            
            # Ensure stop loss and take profit are set
            if action in ["buy", "sell"] and amount > 0:
                if not decision.get("stop_loss") or not decision.get("take_profit"):
                    decision["action"] = "hold"
                    decision["reason"] = "Risk management levels not properly set."
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to validate decision: {e}")
            return {"action": "hold", "reason": f"Validation error: {str(e)}"}
