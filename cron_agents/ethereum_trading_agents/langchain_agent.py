#!/usr/bin/env python3
"""
Enhanced Trading Agent using LangChain 0.3.0 features
Leverages new LangChain capabilities for better agent orchestration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Annotated
from datetime import datetime
import json

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langgraph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field, field_validator

from .gemini_agent import GeminiAgent
from .mcp_client import MCPClient
from .database import TradingDatabase
from .config import Config

logger = logging.getLogger(__name__)

class MarketData(BaseModel):
    """Market data model using Pydantic 2"""
    price_usd: float = Field(..., description="Current Ethereum price in USD")
    price_change_24h_percent: float = Field(..., description="24h price change percentage")
    volume_24h: float = Field(..., description="24h trading volume")
    trend: str = Field(..., description="Market trend (bullish/bearish/neutral)")
    volume_trend: str = Field(..., description="Volume trend (increasing/decreasing)")
    
    @field_validator('trend')
    @classmethod
    def validate_trend(cls, v: str) -> str:
        valid_trends = ['bullish', 'bearish', 'neutral']
        if v not in valid_trends:
            raise ValueError(f'Trend must be one of {valid_trends}')
        return v

class TradingDecision(BaseModel):
    """Trading decision model using Pydantic 2"""
    action: str = Field(..., description="Trading action (buy/sell/hold)")
    amount_eth: float = Field(..., description="Amount of ETH to trade")
    target_price: float = Field(..., description="Target price for the trade")
    stop_loss: float = Field(..., description="Stop loss price")
    take_profit: float = Field(..., description="Take profit price")
    reason: str = Field(..., description="Reasoning for the decision")
    risk_level: str = Field(..., description="Risk level (low/medium/high)")
    expected_return: str = Field(..., description="Expected return percentage")
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid_actions = ['buy', 'sell', 'hold']
        if v not in valid_actions:
            raise ValueError(f'Action must be one of {valid_actions}')
        return v
    
    @field_validator('amount_eth')
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v

class LangChainTradingAgent:
    """Enhanced trading agent using LangChain 0.3.0 and LangGraph features"""
    
    def __init__(self, agent_name: str):
        """Initialize LangChain trading agent"""
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
        
        # Initialize LangChain and LangGraph components
        self._setup_langchain_components()
        self._setup_langgraph_workflow()
        
        logger.info(f"LangChain + LangGraph trading agent {agent_name} initialized")
    
    def _setup_langchain_components(self):
        """Setup LangChain 0.3.0 components"""
        try:
            # Create tools for the agent
            self.tools = [
                self._create_market_analysis_tool(),
                self._create_trading_decision_tool(),
                self._create_risk_assessment_tool(),
                self._create_market_research_tool()
            ]
            
            # Create prompt template
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt()),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Create agent
            self.agent = create_openai_tools_agent(
                llm=self.gemini_agent.model,
                tools=self.tools,
                prompt=self.prompt
            )
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True
            )
            
            # Create output parser
            self.output_parser = JsonOutputParser()
            
            logger.info("LangChain components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup LangChain components: {e}")
            raise
    
    def _setup_langgraph_workflow(self):
        """Setup LangGraph workflow for enhanced MCP processing"""
        try:
            # Create tool executor for MCP operations
            self.tool_executor = ToolExecutor(self.tools)
            
            # Define workflow state
            class WorkflowState(BaseModel):
                """State for LangGraph workflow"""
                messages: List[HumanMessage] = Field(default_factory=list)
                market_data: Optional[Dict] = None
                analysis_result: Optional[Dict] = None
                trading_decision: Optional[Dict] = None
                execution_result: Optional[Dict] = None
                mcp_operations: List[Dict] = Field(default_factory=list)
                errors: List[str] = Field(default_factory=list)
            
            self.workflow_state = WorkflowState
            
            # Create state graph
            self.workflow = StateGraph(WorkflowState)
            
            # Add nodes for MCP-enhanced workflow
            self.workflow.add_node("collect_market_data", self._collect_market_data_node)
            self.workflow.add_node("analyze_market", self._analyze_market_node)
            self.workflow.add_node("assess_risk", self._assess_risk_node)
            self.workflow.add_node("make_decision", self._make_decision_node)
            self.workflow.add_node("execute_trade", self._execute_trade_node)
            self.workflow.add_node("update_database", self._update_database_node)
            
            # Define workflow edges
            self.workflow.add_edge("collect_market_data", "analyze_market")
            self.workflow.add_edge("analyze_market", "assess_risk")
            self.workflow.add_edge("assess_risk", "make_decision")
            self.workflow.add_edge("make_decision", "execute_trade")
            self.workflow.add_edge("execute_trade", "update_database")
            
            # Add conditional edges for error handling
            self.workflow.add_conditional_edges(
                "collect_market_data",
                self._should_continue_workflow,
                {
                    "continue": "analyze_market",
                    "error": "update_database"
                }
            )
            
            # Compile workflow
            self.compiled_workflow = self.workflow.compile(
                checkpointer=MemorySaver()
            )
            
            logger.info("LangGraph workflow initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup LangGraph workflow: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent"""
        return f"""
        You are an expert Ethereum trading agent named {self.agent_name}.
        
        Your responsibilities:
        1. Analyze market data and technical indicators
        2. Make informed trading decisions based on market conditions
        3. Assess and manage risk appropriately
        4. Research market insights and news
        
        Current configuration:
        - Max trade amount: {self.config.MAX_TRADE_AMOUNT_ETH} ETH
        - Max daily trades: {self.config.MAX_DAILY_TRADES}
        - Max daily loss: {self.config.MAX_DAILY_LOSS_ETH} ETH
        - Stop loss: {self.config.STOP_LOSS_PERCENT}%
        - Take profit: {self.config.TAKE_PROFIT_PERCENT}%
        
        Always prioritize risk management and never exceed configured limits.
        Provide detailed reasoning for all decisions.
        """
    
    @tool
    def _create_market_analysis_tool(self) -> BaseTool:
        """Create market analysis tool"""
        return BaseTool(
            name="market_analysis",
            description="Analyze current market conditions and provide insights",
            func=self._analyze_market_conditions,
            args_schema=MarketData
        )
    
    @tool
    def _create_trading_decision_tool(self) -> BaseTool:
        """Create trading decision tool"""
        return BaseTool(
            name="trading_decision",
            description="Generate trading decision based on market analysis",
            func=self._generate_trading_decision,
            args_schema=TradingDecision
        )
    
    @tool
    def _create_risk_assessment_tool(self) -> BaseTool:
        """Create risk assessment tool"""
        return BaseTool(
            name="risk_assessment",
            description="Assess risk level for potential trades",
            func=self._assess_risk,
            args_schema=Dict
        )
    
    @tool
    def _create_market_research_tool(self) -> BaseTool:
        """Create market research tool"""
        return BaseTool(
            name="market_research",
            description="Research market insights and news",
            func=self._research_market,
            args_schema=Dict
        )
    
    async def _analyze_market_conditions(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze market conditions using LangChain tools"""
        try:
            # Get historical trends
            historical_trends = self.database.get_market_trends(hours=24)
            
            # Prepare data for AI analysis
            analysis_data = {
                "price_usd": market_data.price_usd,
                "price_change_24h_percent": market_data.price_change_24h_percent,
                "volume_24h": market_data.volume_24h,
                "trend": market_data.trend,
                "volume_trend": market_data.volume_trend,
                "historical_trends": historical_trends
            }
            
            # Use Gemini agent for analysis
            analysis = await self.gemini_agent.analyze_market_data(
                analysis_data,
                {},  # technical_indicators will be fetched separately
                historical_trends
            )
            
            return {
                "status": "success",
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze market conditions: {e}")
            return {"status": "error", "error_message": str(e)}
    
    async def _generate_trading_decision(self, decision_data: TradingDecision) -> Dict[str, Any]:
        """Generate trading decision using LangChain tools"""
        try:
            # Validate decision against risk limits
            validated_decision = self._validate_decision(decision_data)
            
            return {
                "status": "success",
                "decision": validated_decision.dict(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate trading decision: {e}")
            return {"status": "error", "error_message": str(e)}
    
    async def _assess_risk(self, trade_params: Dict) -> Dict[str, Any]:
        """Assess risk for potential trades"""
        try:
            # Get daily trading summary
            daily_summary = self.database.get_daily_trading_summary()
            
            # Calculate risk metrics
            risk_metrics = {
                "daily_trades_remaining": self.config.MAX_DAILY_TRADES - daily_summary["total_trades"],
                "daily_loss_remaining": self.config.MAX_DAILY_LOSS_ETH,
                "position_size_risk": "low" if trade_params.get("amount_eth", 0) <= self.config.MAX_TRADE_AMOUNT_ETH * 0.5 else "medium",
                "overall_risk": "low"
            }
            
            # Adjust overall risk based on metrics
            if risk_metrics["daily_trades_remaining"] <= 2:
                risk_metrics["overall_risk"] = "high"
            elif risk_metrics["daily_trades_remaining"] <= 5:
                risk_metrics["overall_risk"] = "medium"
            
            return {
                "status": "success",
                "risk_metrics": risk_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to assess risk: {e}")
            return {"status": "error", "error_message": str(e)}
    
    async def _research_market(self, query_params: Dict) -> Dict[str, Any]:
        """Research market insights and news"""
        try:
            query = query_params.get("query", "ethereum market trends")
            
            # Use Gemini agent for market research
            insights = await self.gemini_agent.search_market_insights(
                query, 
                {"price_usd": 0, "trend": "unknown", "volume_trend": "unknown"}
            )
            
            return {
                "status": "success",
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to research market: {e}")
            return {"status": "error", "error_message": str(e)}
    
    def _validate_decision(self, decision: TradingDecision) -> TradingDecision:
        """Validate trading decision against constraints"""
        try:
            # Check amount constraints
            if decision.amount_eth > self.config.MAX_TRADE_AMOUNT_ETH:
                decision.amount_eth = self.config.MAX_TRADE_AMOUNT_ETH
                decision.reason += f" Amount adjusted to {self.config.MAX_TRADE_AMOUNT_ETH} ETH due to risk limits."
            
            # Ensure stop loss and take profit are set
            if decision.action in ["buy", "sell"] and decision.amount_eth > 0:
                if not decision.stop_loss or not decision.take_profit:
                    decision.action = "hold"
                    decision.reason = "Risk management levels not properly set."
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to validate decision: {e}")
            return TradingDecision(
                action="hold",
                amount_eth=0.0,
                target_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                reason=f"Validation error: {str(e)}",
                risk_level="high",
                expected_return="0%"
            )
    
    async def execute_trading_cycle(self) -> Dict[str, Any]:
        """Execute complete trading cycle using LangGraph workflow"""
        execution_id = None
        
        try:
            # Record execution start
            execution_id = self.database.record_agent_execution(
                agent_name=self.agent_name,
                status="running"
            )
            
            logger.info(f"Starting LangGraph trading cycle {execution_id} for agent {self.agent_name}")
            
            # Initialize workflow state
            initial_state = self.workflow_state(
                messages=[HumanMessage(content="Execute trading cycle")],
                mcp_operations=[]
            )
            
            # Execute LangGraph workflow
            workflow_result = await self.compiled_workflow.ainvoke(initial_state)
            
            # Extract results from workflow
            market_data = workflow_result.get("market_data", {})
            analysis_result = workflow_result.get("analysis_result", {})
            trading_decision = workflow_result.get("trading_decision", {})
            execution_result = workflow_result.get("execution_result", {})
            mcp_operations = workflow_result.get("mcp_operations", [])
            
            # Record results
            self._record_execution_results(
                execution_id, market_data, analysis_result, trading_decision, execution_result
            )
            
            # Update execution status
            self.database.record_agent_execution(
                agent_name=self.agent_name,
                status="success",
                output_data={
                    "market_data": market_data,
                    "analysis_result": analysis_result,
                    "trading_decision": trading_decision,
                    "execution_result": execution_result,
                    "mcp_operations": mcp_operations
                }
            )
            
            return {
                "status": "success",
                "execution_id": execution_id,
                "timestamp": datetime.now().isoformat(),
                "workflow_result": workflow_result,
                "mcp_operations_count": len(mcp_operations)
            }
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"LangGraph trading cycle failed: {error_message}")
            
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
    
    async def _execute_langchain_agent(self, market_data: Dict, account_info: Dict) -> Dict[str, Any]:
        """Execute LangChain agent for analysis and decision making"""
        try:
            # Prepare input for the agent
            agent_input = {
                "input": f"""
                Analyze the current market conditions and make a trading decision.
                
                Market Data:
                - Price: ${market_data['ethereum_price'].get('primary_price_usd', 0):.2f}
                - 24h Change: {market_data['market_trends'].get('price_change_24h_percent', 0):.2f}%
                - Volume: ${market_data['market_trends'].get('volume_24h', 0):,.0f}
                - Trend: {market_data['market_trends'].get('trend', 'unknown')}
                
                Account Info:
                - Balance: {account_info['balance_eth']:.4f} ETH
                - Gas Price: {account_info['gas_price_gwei']:.2f} Gwei
                
                Use the available tools to analyze the market and make a decision.
                """,
                "chat_history": []
            }
            
            # Execute the agent
            result = await self.agent_executor.ainvoke(agent_input)
            
            return {
                "status": "success",
                "agent_output": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to execute LangChain agent: {e}")
            return {"status": "error", "error_message": str(e)}
    
    async def _execute_trade(self, agent_result: Dict, account_info: Dict) -> Dict[str, Any]:
        """Execute trade based on agent decision"""
        try:
            # For now, simulate trade execution
            # In production, this would use the actual trading MCP server
            
            return {
                "status": "success",
                "action": "hold",
                "reason": "Trade execution simulated - LangChain agent analysis completed",
                "timestamp": datetime.now().isoformat(),
                "note": "Trade simulated - not executed on blockchain"
            }
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            return {
                "status": "error",
                "action": "hold",
                "reason": f"Trade execution failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _record_execution_results(self, execution_id: int, market_data: Dict, 
                                 analysis_result: Dict, trading_decision: Dict, execution_result: Dict):
        """Record all execution results to database"""
        try:
            # Record market snapshot
            if market_data and market_data.get("ethereum_price", {}).get("status") == "success":
                ethereum_price = market_data["ethereum_price"]
                market_trends = market_data["market_trends"]
                technical_indicators = market_data.get("technical_indicators", {})
                
                self.database.record_market_snapshot(
                    execution_id=execution_id,
                    price_usd=ethereum_price.get("primary_price_usd", 0),
                    price_change_24h=market_trends.get("price_change_24h_percent", 0),
                    volume_24h=market_trends.get("volume_24h", 0),
                    technical_indicators=technical_indicators
                )
            
            # Record trading decision if available
            if trading_decision and trading_decision.get("status") == "success":
                decision_data = trading_decision.get("decision", {})
                self.database.record_trading_decision(
                    execution_id=execution_id,
                    decision_type=decision_data.get("action", "hold"),
                    decision_data=decision_data,
                    market_conditions=market_data,
                    reasoning=decision_data.get("reason", "")
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
                "agent_type": "LangChain Enhanced",
                "timestamp": datetime.now().isoformat(),
                "mcp_health": health_status,
                "last_execution": last_execution,
                "daily_summary": daily_summary,
                "langchain_components": {
                    "tools_count": len(self.tools),
                    "agent_initialized": self.agent is not None,
                    "executor_ready": self.agent_executor is not None
                },
                "langgraph_components": {
                    "workflow_initialized": self.workflow is not None,
                    "workflow_compiled": self.compiled_workflow is not None,
                    "tool_executor_ready": self.tool_executor is not None
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
    
    # LangGraph workflow nodes
    async def _collect_market_data_node(self, state: "WorkflowState") -> "WorkflowState":
        """LangGraph node: Collect market data via MCP"""
        try:
            async with self.mcp_client:
                market_data = await self.mcp_client.batch_market_data()
                
                if market_data["status"] == "success":
                    state.market_data = market_data
                    state.mcp_operations.append({
                        "operation": "collect_market_data",
                        "status": "success",
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    state.errors.append(f"Failed to collect market data: {market_data.get('message', 'Unknown error')}")
                    state.mcp_operations.append({
                        "operation": "collect_market_data",
                        "status": "error",
                        "error": market_data.get('message', 'Unknown error'),
                        "timestamp": datetime.now().isoformat()
                    })
            
            return state
            
        except Exception as e:
            state.errors.append(f"Market data collection error: {str(e)}")
            return state
    
    async def _analyze_market_node(self, state: "WorkflowState") -> "WorkflowState":
        """LangGraph node: Analyze market conditions"""
        try:
            if state.market_data and state.market_data["status"] == "success":
                # Get historical trends
                historical_trends = self.database.get_market_trends(hours=24)
                
                # Prepare data for analysis
                ethereum_price = state.market_data["ethereum_price"]
                market_trends = state.market_data["market_trends"]
                
                market_data_for_analysis = {
                    "price_usd": ethereum_price.get("primary_price_usd", 0),
                    "price_change_24h_percent": market_trends.get("price_change_24h_percent", 0),
                    "volume_24h": market_trends.get("volume_24h", 0),
                    "trend": market_trends.get("trend", "unknown"),
                    "volume_trend": market_trends.get("volume_trend", "unknown")
                }
                
                # Use Gemini agent for analysis
                analysis = await self.gemini_agent.analyze_market_data(
                    market_data_for_analysis,
                    {},  # technical_indicators
                    historical_trends
                )
                
                state.analysis_result = analysis
                state.mcp_operations.append({
                    "operation": "analyze_market",
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                state.errors.append("No market data available for analysis")
            
            return state
            
        except Exception as e:
            state.errors.append(f"Market analysis error: {str(e)}")
            return state
    
    async def _assess_risk_node(self, state: "WorkflowState") -> "WorkflowState":
        """LangGraph node: Assess risk for potential trades"""
        try:
            # Get daily trading summary
            daily_summary = self.database.get_daily_trading_summary()
            
            # Calculate risk metrics
            risk_metrics = {
                "daily_trades_remaining": self.config.MAX_DAILY_TRADES - daily_summary["total_trades"],
                "daily_loss_remaining": self.config.MAX_DAILY_LOSS_ETH,
                "overall_risk": "low"
            }
            
            # Adjust overall risk based on metrics
            if risk_metrics["daily_trades_remaining"] <= 2:
                risk_metrics["overall_risk"] = "high"
            elif risk_metrics["daily_trades_remaining"] <= 5:
                risk_metrics["overall_risk"] = "medium"
            
            state.mcp_operations.append({
                "operation": "assess_risk",
                "status": "success",
                "risk_metrics": risk_metrics,
                "timestamp": datetime.now().isoformat()
            })
            
            return state
            
        except Exception as e:
            state.errors.append(f"Risk assessment error: {str(e)}")
            return state
    
    async def _make_decision_node(self, state: "WorkflowState") -> "WorkflowState":
        """LangGraph node: Make trading decision"""
        try:
            if state.analysis_result and state.analysis_result["status"] == "success":
                # Prepare risk profile
                risk_profile = {
                    "max_daily_loss_eth": self.config.MAX_DAILY_LOSS_ETH,
                    "max_trade_amount_eth": self.config.MAX_TRADE_AMOUNT_ETH,
                    "risk_tolerance": "medium"
                }
                
                # Get account info for decision making
                async with self.mcp_client:
                    account_info = await self.mcp_client.get_ethereum_balance(self.config.ETHEREUM_ADDRESS)
                
                if account_info["status"] == "success":
                    # Generate trading decision
                    decision = await self.gemini_agent.generate_trading_decision(
                        state.analysis_result["analysis"],
                        risk_profile,
                        float(account_info.get("balance_eth", 0))
                    )
                    
                    state.trading_decision = decision
                    state.mcp_operations.append({
                        "operation": "make_decision",
                        "status": "success",
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    state.errors.append("Failed to get account info for decision making")
            else:
                state.errors.append("No analysis result available for decision making")
            
            return state
            
        except Exception as e:
            state.errors.append(f"Decision making error: {str(e)}")
            return state
    
    async def _execute_trade_node(self, state: "WorkflowState") -> "WorkflowState":
        """LangGraph node: Execute trade decision"""
        try:
            if state.trading_decision and state.trading_decision["status"] == "success":
                decision = state.trading_decision["decision"]
                action = decision.get("action", "hold")
                
                if action != "hold":
                    # Simulate trade execution
                    execution_result = {
                        "status": "success",
                        "action": action,
                        "amount_eth": decision.get("amount_eth", 0),
                        "target_price": decision.get("target_price", 0),
                        "stop_loss": decision.get("stop_loss", 0),
                        "take_profit": decision.get("take_profit", 0),
                        "reason": decision.get("reason", ""),
                        "timestamp": datetime.now().isoformat(),
                        "note": "Trade simulated - LangGraph workflow execution"
                    }
                else:
                    execution_result = {
                        "status": "success",
                        "action": "hold",
                        "reason": decision.get("reason", "No action required"),
                        "timestamp": datetime.now().isoformat()
                    }
                
                state.execution_result = execution_result
                state.mcp_operations.append({
                    "operation": "execute_trade",
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                state.errors.append("No valid trading decision available for execution")
            
            return state
            
        except Exception as e:
            state.errors.append(f"Trade execution error: {str(e)}")
            return state
    
    async def _update_database_node(self, state: "WorkflowState") -> "WorkflowState":
        """LangGraph node: Update database with results"""
        try:
            # This node will be called after trade execution
            # Database updates are handled in the main execution method
            state.mcp_operations.append({
                "operation": "update_database",
                "status": "success",
                "timestamp": datetime.now().isoformat()
            })
            
            return state
            
        except Exception as e:
            state.errors.append(f"Database update error: {str(e)}")
            return state
    
    def _should_continue_workflow(self, state: "WorkflowState") -> str:
        """Conditional edge function for workflow control"""
        if state.errors:
            return "error"
        return "continue"
