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
    """2025년 10월 기준 최신 Agentic Flow 구현"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """Initialize autonomous Gemini agent with agentic capabilities"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.chat = self.model.start_chat(history=[])
            
            # Agentic state management
            self.agentic_state = {
                "autonomous_mode": True,
                "learning_enabled": True,
                "decision_history": [],
                "market_context": {},
                "risk_tolerance": "dynamic",
                "strategy_adaptation": True
            }
            
            # Autonomous decision framework
            self.decision_framework = {
                "analysis_depth": "comprehensive",
                "reasoning_chain": "multi_step",
                "self_reflection": True,
                "adaptive_thinking": True
            }
            
            logger.info(f"Autonomous Gemini agent initialized with agentic capabilities: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize autonomous Gemini agent: {e}")
            raise
    
    async def analyze_market_data(self, market_data: Dict, technical_indicators: Dict, 
                                historical_trends: List[Dict]) -> Dict[str, Any]:
        """Autonomous market analysis with agentic reasoning"""
        try:
            # Update agentic state with current market context
            self.agentic_state["market_context"] = {
                "current_data": market_data,
                "technical_indicators": technical_indicators,
                "historical_trends": historical_trends,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Autonomous multi-step analysis
            analysis_steps = await self._execute_autonomous_analysis()
            
            # Self-reflection and validation
            validated_analysis = await self._self_reflect_on_analysis(analysis_steps)
            
            # Update decision history for learning
            self._update_decision_history(validated_analysis)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "analysis": validated_analysis,
                "agentic_confidence": validated_analysis.get("agentic_confidence", 0.0),
                "autonomous_recommendation": validated_analysis.get("autonomous_recommendation", "hold"),
                "reasoning_chain": validated_analysis.get("reasoning_chain", []),
                "risk_assessment": validated_analysis.get("risk_assessment", "medium"),
                "learning_insights": validated_analysis.get("learning_insights", [])
            }
            
        except Exception as e:
            logger.error(f"Autonomous market analysis failed: {e}")
            raise ValueError(f"Autonomous analysis failed: {e}")
    
    async def generate_trading_decision(self, market_analysis: Dict, risk_profile: Dict,
                                      account_balance: float) -> Dict[str, Any]:
        """Autonomous trading decision with agentic reasoning"""
        try:
            # Update agentic state with decision context
            self.agentic_state["decision_context"] = {
                "market_analysis": market_analysis,
                "risk_profile": risk_profile,
                "account_balance": account_balance,
                "decision_timestamp": datetime.now().isoformat()
            }
            
            # Autonomous decision-making process
            decision_process = await self._execute_autonomous_decision_making()
            
            # Multi-perspective validation
            validated_decision = await self._multi_perspective_validation(decision_process)
            
            # Adaptive risk adjustment
            final_decision = await self._adaptive_risk_adjustment(validated_decision)
            
            # Update learning from decision
            self._learn_from_decision(final_decision)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "autonomous_decision": final_decision,
                "execution_ready": final_decision.get("action") != "hold",
                "agentic_confidence": final_decision.get("agentic_confidence", 0.0),
                "reasoning_chain": final_decision.get("reasoning_chain", []),
                "adaptive_factors": final_decision.get("adaptive_factors", {})
            }
            
        except Exception as e:
            logger.error(f"Autonomous trading decision failed: {e}")
            raise ValueError(f"Autonomous decision making failed: {e}")
    
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
                raise ValueError("No valid JSON found in analysis response")
                
        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            raise ValueError(f"Analysis parsing failed: {e}")
    
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
                raise ValueError("No valid JSON found in decision response")
                
        except Exception as e:
            logger.error(f"Failed to parse decision response: {e}")
            raise ValueError(f"Decision parsing failed: {e}")
    
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
                raise ValueError("No valid JSON found in search response")
                
        except Exception as e:
            logger.error(f"Failed to parse search response: {e}")
            raise ValueError(f"Search parsing failed: {e}")
    
    
    
    
    async def _execute_autonomous_analysis(self) -> Dict[str, Any]:
        """Execute autonomous multi-step market analysis"""
        try:
            context = self.agentic_state["market_context"]
            
            # Step 1: Technical Analysis
            technical_analysis = await self._autonomous_technical_analysis(context)
            
            # Step 2: Fundamental Analysis
            fundamental_analysis = await self._autonomous_fundamental_analysis(context)
            
            # Step 3: Sentiment Analysis
            sentiment_analysis = await self._autonomous_sentiment_analysis(context)
            
            # Step 4: Pattern Recognition
            pattern_analysis = await self._autonomous_pattern_recognition(context)
            
            # Step 5: Risk Assessment
            risk_analysis = await self._autonomous_risk_assessment(context)
            
            return {
                "technical": technical_analysis,
                "fundamental": fundamental_analysis,
                "sentiment": sentiment_analysis,
                "patterns": pattern_analysis,
                "risk": risk_analysis,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Autonomous analysis execution failed: {e}")
            raise ValueError(f"Autonomous analysis failed: {e}")
    
    async def _self_reflect_on_analysis(self, analysis_steps: Dict[str, Any]) -> Dict[str, Any]:
        """Self-reflection and validation of analysis"""
        try:
            # Create self-reflection prompt
            reflection_prompt = f"""
            As an autonomous trading agent, perform self-reflection on your analysis:
            
            Analysis Steps: {analysis_steps}
            
            Please:
            1. Evaluate the quality and consistency of your analysis
            2. Identify any potential biases or gaps
            3. Assess confidence levels for each component
            4. Provide a synthesized autonomous recommendation
            5. Generate learning insights for future improvements
            
            Respond in JSON format with:
            - agentic_confidence: 0-100
            - autonomous_recommendation: buy/sell/hold
            - reasoning_chain: [list of reasoning steps]
            - risk_assessment: low/medium/high
            - learning_insights: [list of insights]
            - potential_biases: [list of identified biases]
            """
            
            response = await self._get_ai_response(reflection_prompt)
            return self._parse_analysis_response(response)
            
        except Exception as e:
            logger.error(f"Self-reflection failed: {e}")
            raise ValueError(f"Self-reflection failed: {e}")
    
    async def _execute_autonomous_decision_making(self) -> Dict[str, Any]:
        """Execute autonomous decision-making process"""
        try:
            context = self.agentic_state["decision_context"]
            
            # Create autonomous decision prompt
            decision_prompt = f"""
            As an autonomous trading agent, make an independent trading decision:
            
            Market Analysis: {context['market_analysis']}
            Risk Profile: {context['risk_profile']}
            Account Balance: {context['account_balance']}
            
            Consider:
            1. Current market conditions and trends
            2. Risk-reward ratios
            3. Portfolio diversification
            4. Market volatility
            5. Your learning from previous decisions
            
            Make an autonomous decision with:
            - action: buy/sell/hold
            - amount_eth: specific amount
            - target_price: target price
            - stop_loss: stop loss price
            - take_profit: take profit price
            - reasoning: detailed reasoning
            - agentic_confidence: 0-100
            - adaptive_factors: factors influencing decision
            
            Respond in JSON format.
            """
            
            response = await self._get_ai_response(decision_prompt)
            return self._parse_decision_response(response)
            
        except Exception as e:
            logger.error(f"Autonomous decision making failed: {e}")
            raise ValueError(f"Autonomous decision making failed: {e}")
    
    def _update_decision_history(self, analysis: Dict[str, Any]) -> None:
        """Update decision history for learning"""
        try:
            decision_record = {
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "agentic_state": self.agentic_state.copy(),
                "learning_insights": analysis.get("learning_insights", [])
            }
            
            self.agentic_state["decision_history"].append(decision_record)
            
            # Keep only last 100 decisions for memory efficiency
            if len(self.agentic_state["decision_history"]) > 100:
                self.agentic_state["decision_history"] = self.agentic_state["decision_history"][-100:]
                
        except Exception as e:
            logger.error(f"Failed to update decision history: {e}")
    
    def _learn_from_decision(self, decision: Dict[str, Any]) -> None:
        """Learn from decision for future improvements"""
        try:
            # Extract learning insights
            learning_insights = decision.get("learning_insights", [])
            adaptive_factors = decision.get("adaptive_factors", {})
            
            # Update agentic state based on learning
            if "strategy_adaptation" in adaptive_factors:
                self.agentic_state["strategy_adaptation"] = True
                
            if "risk_tolerance_adjustment" in adaptive_factors:
                self.agentic_state["risk_tolerance"] = adaptive_factors["risk_tolerance_adjustment"]
                
            logger.info(f"Agent learned from decision: {len(learning_insights)} insights")
            
        except Exception as e:
            logger.error(f"Failed to learn from decision: {e}")
    
    async def _autonomous_technical_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous technical analysis"""
        try:
            prompt = f"""
            Perform autonomous technical analysis:
            
            Market Data: {context['current_data']}
            Technical Indicators: {context['technical_indicators']}
            Historical Trends: {context['historical_trends']}
            
            Analyze:
            1. Price trends and momentum
            2. Support and resistance levels
            3. Volume patterns
            4. Technical indicator signals
            5. Chart patterns
            
            Provide autonomous technical assessment in JSON format.
            """
            
            response = await self._get_ai_response(prompt)
            return self._parse_analysis_response(response)
            
        except Exception as e:
            logger.error(f"Autonomous technical analysis failed: {e}")
            raise ValueError(f"Technical analysis failed: {e}")
    
    async def _autonomous_fundamental_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous fundamental analysis"""
        try:
            prompt = f"""
            Perform autonomous fundamental analysis:
            
            Market Data: {context['current_data']}
            
            Analyze:
            1. Market fundamentals
            2. Economic indicators
            3. Network metrics
            4. Adoption trends
            5. Regulatory factors
            
            Provide autonomous fundamental assessment in JSON format.
            """
            
            response = await self._get_ai_response(prompt)
            return self._parse_analysis_response(response)
            
        except Exception as e:
            logger.error(f"Autonomous fundamental analysis failed: {e}")
            raise ValueError(f"Fundamental analysis failed: {e}")
    
    async def _autonomous_sentiment_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous sentiment analysis"""
        try:
            prompt = f"""
            Perform autonomous sentiment analysis:
            
            Market Data: {context['current_data']}
            
            Analyze:
            1. Market sentiment indicators
            2. Social media sentiment
            3. News sentiment
            4. Fear and greed indicators
            5. Community sentiment
            
            Provide autonomous sentiment assessment in JSON format.
            """
            
            response = await self._get_ai_response(prompt)
            return self._parse_analysis_response(response)
            
        except Exception as e:
            logger.error(f"Autonomous sentiment analysis failed: {e}")
            raise ValueError(f"Sentiment analysis failed: {e}")
    
    async def _autonomous_pattern_recognition(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous pattern recognition"""
        try:
            prompt = f"""
            Perform autonomous pattern recognition:
            
            Market Data: {context['current_data']}
            Historical Trends: {context['historical_trends']}
            
            Identify:
            1. Chart patterns
            2. Price action patterns
            3. Volume patterns
            4. Market cycle patterns
            5. Anomaly detection
            
            Provide autonomous pattern recognition in JSON format.
            """
            
            response = await self._get_ai_response(prompt)
            return self._parse_analysis_response(response)
            
        except Exception as e:
            logger.error(f"Autonomous pattern recognition failed: {e}")
            raise ValueError(f"Pattern recognition failed: {e}")
    
    async def _autonomous_risk_assessment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous risk assessment"""
        try:
            prompt = f"""
            Perform autonomous risk assessment:
            
            Market Data: {context['current_data']}
            Technical Indicators: {context['technical_indicators']}
            
            Assess:
            1. Market volatility risk
            2. Liquidity risk
            3. Concentration risk
            4. Systemic risk
            5. Operational risk
            
            Provide autonomous risk assessment in JSON format.
            """
            
            response = await self._get_ai_response(prompt)
            return self._parse_analysis_response(response)
            
        except Exception as e:
            logger.error(f"Autonomous risk assessment failed: {e}")
            raise ValueError(f"Risk assessment failed: {e}")
    
    async def _multi_perspective_validation(self, decision_process: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-perspective validation of decision"""
        try:
            prompt = f"""
            As an autonomous agent, validate your decision from multiple perspectives:
            
            Decision Process: {decision_process}
            
            Validate from:
            1. Conservative perspective
            2. Aggressive perspective
            3. Risk management perspective
            4. Market timing perspective
            5. Portfolio optimization perspective
            
            Provide validated decision in JSON format.
            """
            
            response = await self._get_ai_response(prompt)
            return self._parse_decision_response(response)
            
        except Exception as e:
            logger.error(f"Multi-perspective validation failed: {e}")
            raise ValueError(f"Validation failed: {e}")
    
    async def _adaptive_risk_adjustment(self, validated_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptive risk adjustment based on learning"""
        try:
            # Get learning insights from decision history
            recent_decisions = self.agentic_state["decision_history"][-10:] if len(self.agentic_state["decision_history"]) >= 10 else []
            
            prompt = f"""
            As an autonomous agent, perform adaptive risk adjustment:
            
            Validated Decision: {validated_decision}
            Recent Decision History: {recent_decisions}
            Current Risk Tolerance: {self.agentic_state['risk_tolerance']}
            
            Adjust risk parameters based on:
            1. Recent performance
            2. Market volatility changes
            3. Learning from past decisions
            4. Current market conditions
            
            Provide final adaptive decision in JSON format.
            """
            
            response = await self._get_ai_response(prompt)
            return self._parse_decision_response(response)
            
        except Exception as e:
            logger.error(f"Adaptive risk adjustment failed: {e}")
            raise ValueError(f"Risk adjustment failed: {e}")
