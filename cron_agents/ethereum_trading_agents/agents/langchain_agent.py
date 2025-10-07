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

from typing import Dict, List, Optional, Any, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langgraph.graph import StateGraph
# ToolExecutor is not available in this version of langgraph
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field, field_validator

from agents.gemini_agent import GeminiAgent
from utils.mcp_client import MCPClient
from utils.database import TradingDatabase
from utils.config import Config

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

class TradingAgentChain:
    """Enhanced trading agent using LangChain 0.3.0 and LangGraph features"""
    
    class WorkflowState(TypedDict, total=False):
        """State for LangGraph workflow with enhanced validation"""
        messages: List[HumanMessage]
        market_data: Optional[Dict[str, Any]]
        analysis_result: Optional[Dict[str, Any]]
        trading_decision: Optional[Dict[str, Any]]
        execution_result: Optional[Dict[str, Any]]
        mcp_operations: List[Dict[str, Any]]
        errors: List[str]
        workflow_step: str
        retry_count: int
    
    # Create Pydantic model for validation
    class WorkflowStateValidator(BaseModel):
        """Pydantic validator for WorkflowState"""
        messages: List[HumanMessage] = Field(default_factory=list)
        market_data: Optional[Dict[str, Any]] = None
        analysis_result: Optional[Dict[str, Any]] = None
        trading_decision: Optional[Dict[str, Any]] = None
        execution_result: Optional[Dict[str, Any]] = None
        mcp_operations: List[Dict[str, Any]] = Field(default_factory=list)
        errors: List[str] = Field(default_factory=list)
        workflow_step: str = Field(default="initialized")
        retry_count: int = Field(default=0, ge=0, le=3)  # Max 3 retries
        
        @field_validator('workflow_step')
        @classmethod
        def validate_workflow_step(cls, v):
            valid_steps = ['initialized', 'collecting', 'analyzing', 'assessing', 'deciding', 'executing', 'updating', 'completed', 'error']
            if v not in valid_steps:
                raise ValueError(f'Invalid workflow step: {v}')
            return v

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
        """Setup LangChain 0.3.0 components with enhanced error handling"""
        try:
            # Validate required components before setup
            if not self.gemini_agent or not self.gemini_agent.model:
                raise ValueError("Gemini agent not properly initialized")
            
            # Create tools for the agent with validation
            tools = []
            tool_creation_methods = [
                self._create_market_analysis_tool,
                self._create_trading_decision_tool,
                self._create_risk_assessment_tool,
                self._create_market_research_tool
            ]
            
            for tool_method in tool_creation_methods:
                try:
                    tool = tool_method()
                    if tool and hasattr(tool, 'name') and hasattr(tool, 'func'):
                        tools.append(tool)
                        logger.debug(f"Tool {tool.name} created successfully")
                    else:
                        logger.warning(f"Tool creation method {tool_method.__name__} returned invalid tool")
                except Exception as tool_error:
                    logger.error(f"Failed to create tool {tool_method.__name__}: {tool_error}")
                    # Continue with other tools instead of failing completely
            
            if not tools:
                raise ValueError("No valid tools could be created")
            
            self.tools = tools
            logger.info(f"Created {len(tools)} valid tools")
            
            # Create prompt template with validation
            system_prompt = self._get_system_prompt()
            if not system_prompt or len(system_prompt.strip()) < 10:
                raise ValueError("System prompt is invalid or too short")
            
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Create agent with validation
            self.agent = create_openai_tools_agent(
                llm=self.gemini_agent.model,
                tools=self.tools,
                prompt=self.prompt
            )
            
            if not self.agent:
                raise ValueError("Failed to create agent")
            
            # Create agent executor with enhanced error handling
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,  # Prevent infinite loops
                early_stopping_method="generate"  # Stop on generation errors
            )
            
            # Create output parser
            self.output_parser = JsonOutputParser()
            
            # Validate all components
            self._validate_langchain_components()
            
            logger.info("LangChain components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup LangChain components: {e}")
            raise
    
    def _validate_langchain_components(self):
        """Validate all LangChain components are properly initialized"""
        validation_errors = []
        
        if not self.tools:
            validation_errors.append("No tools available")
        
        if not self.prompt:
            validation_errors.append("No prompt template")
        
        if not self.agent:
            validation_errors.append("No agent created")
        
        if not self.agent_executor:
            validation_errors.append("No agent executor")
        
        if not self.output_parser:
            validation_errors.append("No output parser")
        
        if validation_errors:
            error_msg = f"LangChain component validation failed: {', '.join(validation_errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _setup_langgraph_workflow(self):
        """Setup LangGraph workflow for enhanced MCP processing with stability improvements"""
        try:
            # Validate tools before creating tool executor
            if not self.tools:
                raise ValueError("Cannot create workflow without valid tools")
            
            # Create tool executor for MCP operations
            self.tool_executor = ToolExecutor(self.tools)
            
            # Define workflow state with TypedDict for better type safety

            self.workflow_state = self.WorkflowState
            
            # Create state graph
            self.workflow = StateGraph(self.WorkflowState)
            
            # Validate node methods exist before adding
            node_methods = {
                "collect_market_data": self._collect_market_data_node,
                "analyze_market": self._analyze_market_node,
                "assess_risk": self._assess_risk_node,
                "make_decision": self._make_decision_node,
                "execute_trade": self._execute_trade_node,
                "update_database": self._update_database_node
            }
            
            # Add nodes with validation
            for node_name, node_method in node_methods.items():
                if node_method and callable(node_method):
                    self.workflow.add_node(node_name, node_method)
                    logger.debug(f"Added workflow node: {node_name}")
                else:
                    raise ValueError(f"Invalid node method for {node_name}")
            
            # Define workflow edges with validation
            edges = [
                ("collect_market_data", "analyze_market"),
                ("analyze_market", "assess_risk"),
                ("assess_risk", "make_decision"),
                ("make_decision", "execute_trade"),
                ("execute_trade", "update_database")
            ]
            
            for from_node, to_node in edges:
                try:
                    self.workflow.add_edge(from_node, to_node)
                    logger.debug(f"Added workflow edge: {from_node} -> {to_node}")
                except Exception as edge_error:
                    logger.error(f"Failed to add edge {from_node} -> {to_node}: {edge_error}")
                    raise
            
            # Add conditional edges for error handling with validation
            if hasattr(self, '_should_continue_workflow') and callable(self._should_continue_workflow):
                self.workflow.add_conditional_edges(
                    "collect_market_data",
                    self._should_continue_workflow,
                    {
                        "continue": "analyze_market",
                        "error": "update_database"
                    }
                )
                logger.debug("Added conditional edges for error handling")
            else:
                logger.warning("Conditional edge method not available, using linear workflow")
            
            # Compile workflow with error handling
            try:
                self.compiled_workflow = self.workflow.compile(
                    checkpointer=MemorySaver()
                )
                logger.info("LangGraph workflow compiled successfully")
            except Exception as compile_error:
                logger.error(f"Failed to compile workflow: {compile_error}")
                raise
            
            # Validate compiled workflow
            if not self.compiled_workflow:
                raise ValueError("Workflow compilation failed - no compiled workflow available")
            
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
        """Execute complete trading cycle using LangGraph workflow with enhanced stability"""
        execution_id = None
        
        try:
            # Validate workflow before execution
            if not self.compiled_workflow:
                raise ValueError("LangGraph workflow not properly compiled")
            
            # Record execution start
            execution_id = self.database.record_agent_execution(
                agent_name=self.agent_name,
                status="running"
            )
            
            logger.info(f"Starting LangGraph trading cycle {execution_id} for agent {self.agent_name}")
            
            # Initialize workflow state with validation
            try:
                initial_state: TradingAgentChain.WorkflowState = {
                    "messages": [HumanMessage(content="Execute trading cycle")],
                    "mcp_operations": [],
                    "workflow_step": "initialized",
                    "retry_count": 0
                }
                
                # Validate state using Pydantic validator
                self.workflow_state_validator = TradingAgentChain.WorkflowStateValidator(**initial_state)
                
            except Exception as state_error:
                raise ValueError(f"Failed to initialize workflow state: {state_error}")
            
            # Execute LangGraph workflow with timeout and retry logic
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Executing workflow attempt {attempt + 1}/{max_retries}")
                    
                    # Execute workflow with timeout
                    workflow_result = await asyncio.wait_for(
                        self.compiled_workflow.ainvoke(initial_state),
                        timeout=300  # 5 minutes timeout
                    )
                    
                    # Validate workflow result
                    if not workflow_result:
                        raise ValueError("Workflow returned empty result")
                    
                    # Extract and validate results from workflow
                    market_data = workflow_result.get("market_data")
                    analysis_result = workflow_result.get("analysis_result")
                    trading_decision = workflow_result.get("trading_decision")
                    execution_result = workflow_result.get("execution_result")
                    mcp_operations = workflow_result.get("mcp_operations", [])
                    errors = workflow_result.get("errors", [])
                    
                    # Check for critical errors
                    if errors and len(errors) > 2:  # More than 2 errors is critical
                        raise ValueError(f"Critical workflow errors: {errors[:3]}")  # Show first 3 errors
                    
                    # Validate critical data
                    if not market_data or market_data.get("status") != "success":
                        raise ValueError("Market data collection failed")
                    
                    if not analysis_result or analysis_result.get("status") != "success":
                        raise ValueError("Market analysis failed")
                    
                    logger.info(f"Workflow execution successful on attempt {attempt + 1}")
                    break
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Workflow execution timed out on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        raise ValueError("Workflow execution timed out after all retries")
                        
                except Exception as workflow_error:
                    logger.warning(f"Workflow execution failed on attempt {attempt + 1}: {workflow_error}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        raise workflow_error
            
            # Record results with validation
            try:
                self._record_execution_results(
                    execution_id, market_data, analysis_result, trading_decision, execution_result
                )
            except Exception as record_error:
                logger.error(f"Failed to record execution results: {record_error}")
                # Don't fail the entire execution for recording errors
            
            # Update execution status
            self.database.record_agent_execution(
                agent_name=self.agent_name,
                status="success",
                output_data={
                    "market_data": market_data,
                    "analysis_result": analysis_result,
                    "trading_decision": trading_decision,
                    "execution_result": execution_result,
                    "mcp_operations": mcp_operations,
                    "execution_attempts": attempt + 1
                }
            )
            
            return {
                "status": "success",
                "execution_id": execution_id,
                "timestamp": datetime.now().isoformat(),
                "workflow_result": workflow_result,
                "mcp_operations_count": len(mcp_operations),
                "execution_attempts": attempt + 1
            }
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"LangGraph trading cycle failed: {error_message}")
            
            # Record error with detailed information
            if execution_id:
                try:
                    self.database.record_agent_execution(
                        agent_name=self.agent_name,
                        status="error",
                        error_message=error_message,
                        output_data={
                            "error_type": type(e).__name__,
                            "error_details": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                except Exception as db_error:
                    logger.error(f"Failed to record error to database: {db_error}")
            
            return {
                "status": "error",
                "execution_id": execution_id,
                "timestamp": datetime.now().isoformat(),
                "error_message": error_message,
                "error_type": type(e).__name__
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
    async def _collect_market_data_node(self, state: WorkflowState) -> WorkflowState:
        """LangGraph node: Collect market data via MCP with enhanced stability"""
        try:
            # Update workflow step
            state["workflow_step"] = "collecting"
            
            # Validate MCP client
            if not self.mcp_client:
                raise ValueError("MCP client not available")
            
            # Collect market data with timeout and validation
            try:
                async with self.mcp_client:
                    market_data = await asyncio.wait_for(
                        self.mcp_client.batch_market_data(),
                        timeout=60  # 1 minute timeout for market data collection
                    )
                
                # Validate market data structure
                if not market_data or not isinstance(market_data, dict):
                    raise ValueError("Invalid market data structure received")
                
                if market_data.get("status") == "success":
                    # Validate required data fields
                    required_fields = ["ethereum_price", "market_trends", "technical_indicators"]
                    missing_fields = [field for field in required_fields if field not in market_data]
                    
                    if missing_fields:
                        raise ValueError(f"Missing required market data fields: {missing_fields}")
                    
                    # Validate data quality
                    ethereum_price = market_data.get("ethereum_price", {})
                    if ethereum_price.get("status") != "success":
                        raise ValueError("Ethereum price data collection failed")
                    
                    state["market_data"] = market_data
                    state["mcp_operations"].append({
                        "operation": "collect_market_data",
                        "status": "success",
                        "timestamp": datetime.now().isoformat(),
                        "data_quality": "validated"
                    })
                    
                    logger.info("Market data collected and validated successfully")
                    
                else:
                    error_msg = market_data.get('message', 'Unknown error')
                    state["errors"].append(f"Failed to collect market data: {error_msg}")
                    state["mcp_operations"].append({
                        "operation": "collect_market_data",
                        "status": "error",
                        "error": error_msg,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Don't proceed with invalid market data
                    raise ValueError(f"Market data collection failed: {error_msg}")
                    
            except asyncio.TimeoutError:
                timeout_error = "Market data collection timed out"
                state["errors"].append(timeout_error)
                state["mcp_operations"].append({
                    "operation": "collect_market_data",
                    "status": "timeout",
                    "error": timeout_error,
                    "timestamp": datetime.now().isoformat()
                })
                raise ValueError(timeout_error)
                
        except Exception as e:
            error_msg = f"Market data collection error: {str(e)}"
            state["errors"].append(error_msg)
            state["mcp_operations"].append({
                "operation": "collect_market_data",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            logger.error(error_msg)
            
            # Update workflow step to error
            state["workflow_step"] = "error"
            
        return state
    
    async def _analyze_market_node(self, state: WorkflowState) -> WorkflowState:
        """LangGraph node: Analyze market conditions"""
        try:
            if state.get("market_data") and state["market_data"]["status"] == "success":
                # Get historical trends
                historical_trends = self.database.get_market_trends(hours=24)
                
                # Prepare data for analysis
                ethereum_price = state["market_data"]["ethereum_price"]
                market_trends = state["market_data"]["market_trends"]
                
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
                
                state["analysis_result"] = analysis
                state["mcp_operations"].append({
                    "operation": "analyze_market",
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                state["errors"].append("No market data available for analysis")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Market analysis error: {str(e)}")
            return state
    
    async def _assess_risk_node(self, state: WorkflowState) -> WorkflowState:
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
    
    async def _make_decision_node(self, state: WorkflowState) -> WorkflowState:
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
    
    async def _execute_trade_node(self, state: WorkflowState) -> WorkflowState:
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
    
    async def _update_database_node(self, state: WorkflowState) -> WorkflowState:
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
    
    def _should_continue_workflow(self, state: WorkflowState) -> str:
        """Conditional edge function for workflow control with enhanced error handling"""
        try:
            # Check for critical errors
            if state.get("errors"):
                # Count different types of errors
                error_types = {}
                for error in state["errors"]:
                    error_type = type(error).__name__ if hasattr(error, '__class__') else 'unknown'
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                
                # Log error summary
                logger.warning(f"Workflow errors detected: {error_types}")
                
                # Check if we should retry
                if state.get("retry_count", 0) < 3:
                    logger.info(f"Attempting retry {state.get('retry_count', 0) + 1}/3")
                    state["retry_count"] = state.get("retry_count", 0) + 1
                    state["workflow_step"] = "retrying"
                    return "continue"
                else:
                    logger.error("Maximum retry attempts reached, stopping workflow")
                    state["workflow_step"] = "error"
                    return "error"
            
            # Check workflow step progression
            workflow_step = state.get("workflow_step", "unknown")
            if workflow_step == "error":
                return "error"
            elif workflow_step == "completed":
                return "completed"
            elif workflow_step in ["initialized", "collecting", "analyzing", "assessing", "deciding", "executing", "updating"]:
                return "continue"
            else:
                logger.warning(f"Unknown workflow step: {workflow_step}, continuing")
                return "continue"
                
        except Exception as e:
            logger.error(f"Error in workflow control logic: {e}")
            state["workflow_step"] = "error"
            return "error"
