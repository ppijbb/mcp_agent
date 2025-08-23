"""
Main Trading Agent for Ethereum
Coordinates market analysis, decision making, and trade execution
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from .gemini_agent import GeminiAgent
from ..utils.mcp_client import MCPClient
from ..utils.database import TradingDatabase
from ..utils.config import Config

logger = logging.getLogger(__name__)

class TradingAgent:
    def __init__(self, agent_name: str):
        """Initialize trading agent"""
        self.agent_name = agent_name
        self.config = Config()
        self.database = TradingDatabase()
        
        # Initialize components
        self.gemini_agent = GeminiAgent(
            api_key=self.config.GEMINI_API_KEY,
            model_name=self.config.GEMINI_MODEL
        )
        
        self.mcp_client = MCPClient(
            ethereum_trading_url=self.config.MCP_ETHEREUM_TRADING_URL,
            market_data_url=self.config.MCP_MARKET_DATA_URL
        )
        
        # Load last execution data
        self.last_execution_data = self.database.get_last_execution_data(agent_name)
        
        logger.info(f"Trading agent {agent_name} initialized")
    
    async def execute_trading_cycle(self) -> Dict[str, Any]:
        """Execute complete trading cycle"""
        execution_id = None
        
        try:
            # Record execution start
            execution_id = self.database.record_agent_execution(
                agent_name=self.agent_name,
                status="running"
            )
            
            logger.info(f"Starting trading cycle {execution_id} for agent {self.agent_name}")
            
            # Step 1: Collect market data
            market_data = await self._collect_market_data()
            if market_data["status"] != "success":
                raise Exception(f"Failed to collect market data: {market_data.get('message', 'Unknown error')}")
            
            # Step 2: Get account information
            account_info = await self._get_account_info()
            if account_info["status"] != "success":
                raise Exception(f"Failed to get account info: {account_info.get('message', 'Unknown error')}")
            
            # Step 3: Analyze market conditions
            market_analysis = await self._analyze_market_conditions(market_data)
            if market_analysis["status"] != "success":
                raise Exception(f"Failed to analyze market: {market_analysis.get('error_message', 'Unknown error')}")
            
            # Step 4: Generate trading decision
            trading_decision = await self._generate_trading_decision(
                market_analysis, account_info
            )
            if trading_decision["status"] != "success":
                raise Exception(f"Failed to generate decision: {trading_decision.get('error_message', 'Unknown error')}")
            
            # Step 5: Execute trade if decision is actionable
            execution_result = await self._execute_trade(trading_decision, account_info)
            
            # Step 6: Record results
            self._record_execution_results(
                execution_id, market_data, market_analysis, 
                trading_decision, execution_result
            )
            
            # Update execution status
            self.database.record_agent_execution(
                agent_name=self.agent_name,
                status="success",
                output_data={
                    "market_data": market_data,
                    "market_analysis": market_analysis,
                    "trading_decision": trading_decision,
                    "execution_result": execution_result
                }
            )
            
            return {
                "status": "success",
                "execution_id": execution_id,
                "timestamp": datetime.now().isoformat(),
                "market_analysis": market_analysis,
                "trading_decision": trading_decision,
                "execution_result": execution_result
            }
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Trading cycle failed: {error_message}")
            
            # Record error
            if execution_id:
                self.database.record_agent_execution(
                    agent_name=self.agent_name,
                    status="error",
                    error_message=error_message
                )
            
            return {
                "status": "error",
                "execution_id": execution_id,
                "timestamp": datetime.now().isoformat(),
                "error_message": error_message
            }
    
    async def _collect_market_data(self) -> Dict[str, Any]:
        """Collect comprehensive market data"""
        try:
            async with self.mcp_client:
                # Get batch market data
                market_data = await self.mcp_client.batch_market_data()
                
                if market_data["status"] != "success":
                    return market_data
                
                # Extract and format data
                ethereum_price = market_data["ethereum_price"]
                market_trends = market_data["market_trends"]
                technical_indicators = market_data["technical_indicators"]
                
                # Validate data quality
                if (ethereum_price["status"] != "success" or 
                    market_trends["status"] != "success" or 
                    technical_indicators["status"] != "success"):
                    return {
                        "status": "error",
                        "message": "One or more data sources failed"
                    }
                
                return {
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "ethereum_price": ethereum_price,
                    "market_trends": market_trends,
                    "technical_indicators": technical_indicators
                }
                
        except Exception as e:
            logger.error(f"Failed to collect market data: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_account_info(self) -> Dict[str, Any]:
        """Get account information and balance"""
        try:
            async with self.mcp_client:
                # Get account balance
                balance = await self.mcp_client.get_ethereum_balance(self.config.ETHEREUM_ADDRESS)
                
                if balance["status"] != "success":
                    return balance
                
                # Get gas price for cost estimation
                gas_price = await self.mcp_client.get_gas_price()
                
                return {
                    "status": "success",
                    "address": self.config.ETHEREUM_ADDRESS,
                    "balance_eth": float(balance.get("balance_eth", 0)),
                    "balance_wei": balance.get("balance_wei", "0"),
                    "gas_price_gwei": float(gas_price.get("gas_price_gwei", 0)) if gas_price["status"] == "success" else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _analyze_market_conditions(self, market_data: Dict) -> Dict[str, Any]:
        """Analyze market conditions using AI"""
        try:
            # Prepare data for analysis
            ethereum_price = market_data["ethereum_price"]
            market_trends = market_data["market_trends"]
            technical_indicators = market_data["technical_indicators"]
            
            # Get historical trends from database
            historical_trends = self.database.get_market_trends(hours=24)
            
            # Prepare market data for AI analysis
            market_data_for_analysis = {
                "price_usd": ethereum_price.get("primary_price_usd", 0),
                "price_change_24h_percent": market_trends.get("price_change_24h_percent", 0),
                "volume_24h": market_trends.get("volume_24h", 0),
                "trend": market_trends.get("trend", "unknown"),
                "volume_trend": market_trends.get("volume_trend", "unknown")
            }
            
            # Prepare technical indicators for AI analysis
            tech_indicators = {
                "rsi": technical_indicators.get("rsi", 0),
                "sma_7": technical_indicators.get("sma_7", 0),
                "sma_30": technical_indicators.get("sma_30", 0),
                "signal": technical_indicators.get("signal", "hold")
            }
            
            # Get AI analysis
            analysis = await self.gemini_agent.analyze_market_data(
                market_data_for_analysis,
                tech_indicators,
                historical_trends
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze market conditions: {e}")
            return {"status": "error", "error_message": str(e)}
    
    async def _generate_trading_decision(self, market_analysis: Dict, account_info: Dict) -> Dict[str, Any]:
        """Generate trading decision using AI"""
        try:
            # Prepare risk profile
            risk_profile = {
                "max_daily_loss_eth": self.config.MAX_DAILY_LOSS_ETH,
                "max_trade_amount_eth": self.config.MAX_TRADE_AMOUNT_ETH,
                "risk_tolerance": "medium"  # Could be configurable
            }
            
            # Get AI decision
            decision = await self.gemini_agent.generate_trading_decision(
                market_analysis["analysis"],
                risk_profile,
                account_info["balance_eth"]
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to generate trading decision: {e}")
            return {"status": "error", "error_message": str(e)}
    
    async def _execute_trade(self, trading_decision: Dict, account_info: Dict) -> Dict[str, Any]:
        """Execute trade based on decision"""
        try:
            decision = trading_decision["decision"]
            action = decision.get("action", "hold")
            
            if action == "hold":
                return {
                    "status": "success",
                    "action": "hold",
                    "reason": decision.get("reason", "No action required"),
                    "timestamp": datetime.now().isoformat()
                }
            
            # Validate decision is executable
            if not self._validate_trade_decision(decision, account_info):
                return {
                    "status": "error",
                    "action": "hold",
                    "reason": "Trade decision validation failed",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Execute trade via MCP
            async with self.mcp_client:
                if action == "buy":
                    # For demo purposes, we'll simulate a buy order
                    # In production, this would execute the actual trade
                    execution_result = {
                        "status": "success",
                        "action": "buy",
                        "amount_eth": decision["amount_eth"],
                        "target_price": decision["target_price"],
                        "stop_loss": decision["stop_loss"],
                        "take_profit": decision["take_profit"],
                        "reason": decision["reason"],
                        "timestamp": datetime.now().isoformat(),
                        "note": "Trade simulated - not executed on blockchain"
                    }
                elif action == "sell":
                    # Simulate sell order
                    execution_result = {
                        "status": "success",
                        "action": "sell",
                        "amount_eth": decision["amount_eth"],
                        "target_price": decision["target_price"],
                        "stop_loss": decision["stop_loss"],
                        "take_profit": decision["take_profit"],
                        "reason": decision["reason"],
                        "timestamp": datetime.now().isoformat(),
                        "note": "Trade simulated - not executed on blockchain"
                    }
                else:
                    execution_result = {
                        "status": "error",
                        "action": "unknown",
                        "reason": f"Unknown action: {action}",
                        "timestamp": datetime.now().isoformat()
                    }
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            return {
                "status": "error",
                "action": "hold",
                "reason": f"Trade execution failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_trade_decision(self, decision: Dict, account_info: Dict) -> bool:
        """Validate trade decision against constraints"""
        try:
            action = decision.get("action")
            amount = float(decision.get("amount_eth", 0))
            
            # Check if action is valid
            if action not in ["buy", "sell"]:
                return False
            
            # Check amount constraints
            if amount <= 0 or amount > self.config.MAX_TRADE_AMOUNT_ETH:
                return False
            
            # Check account balance for buy orders
            if action == "buy" and amount > account_info["balance_eth"]:
                return False
            
            # Check risk management levels
            if not decision.get("stop_loss") or not decision.get("take_profit"):
                return False
            
            # Check daily trade limits
            daily_summary = self.database.get_daily_trading_summary()
            if daily_summary["total_trades"] >= self.config.MAX_DAILY_TRADES:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Trade decision validation failed: {e}")
            return False
    
    def _record_execution_results(self, execution_id: int, market_data: Dict, 
                                 market_analysis: Dict, trading_decision: Dict, 
                                 execution_result: Dict):
        """Record all execution results to database"""
        try:
            # Record trading decision
            self.database.record_trading_decision(
                execution_id=execution_id,
                decision_type=trading_decision["decision"].get("action", "hold"),
                decision_data=trading_decision["decision"],
                market_conditions=market_data,
                reasoning=trading_decision["decision"].get("reason", "")
            )
            
            # Record market snapshot
            if market_data["ethereum_price"]["status"] == "success":
                ethereum_price = market_data["ethereum_price"]
                market_trends = market_data["market_trends"]
                technical_indicators = market_data["technical_indicators"]
                
                self.database.record_market_snapshot(
                    execution_id=execution_id,
                    price_usd=ethereum_price.get("primary_price_usd", 0),
                    price_change_24h=market_trends.get("price_change_24h_percent", 0),
                    volume_24h=market_trends.get("volume_24h", 0),
                    technical_indicators=technical_indicators
                )
            
            logger.info(f"Execution results recorded for execution {execution_id}")
            
        except Exception as e:
            logger.error(f"Failed to record execution results: {e}")
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        try:
            # Check MCP server health
            async with self.mcp_client:
                health_status = await self.mcp_client.health_check()
            
            # Get last execution data
            last_execution = self.database.get_last_execution_data(self.agent_name)
            
            # Get daily summary
            daily_summary = self.database.get_daily_trading_summary()
            
            return {
                "status": "success",
                "agent_name": self.agent_name,
                "timestamp": datetime.now().isoformat(),
                "mcp_health": health_status,
                "last_execution": last_execution,
                "daily_summary": daily_summary,
                "config": {
                    "max_daily_trades": self.config.MAX_DAILY_TRADES,
                    "max_daily_loss_eth": self.config.MAX_DAILY_LOSS_ETH,
                    "max_trade_amount_eth": self.config.MAX_TRADE_AMOUNT_ETH
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent status: {e}")
            return {
                "status": "error",
                "agent_name": self.agent_name,
                "timestamp": datetime.now().isoformat(),
                "error_message": str(e)
            }
