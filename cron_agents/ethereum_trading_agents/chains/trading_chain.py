"""
Ethereum Trading Chain with LangGraph Integration

Enhanced trading workflow using LangGraph for complex state management:
1. Market Data Analysis
2. Risk Assessment & Management
3. Trading Signal Generation
4. Position Sizing
5. Execution Decision
6. Real-time Monitoring
"""

import datetime
from typing import Dict, Any, List, Annotated, Optional, Union, Literal
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
# Using MemorySaver instead of SqliteSaver for now
from langgraph.checkpoint.memory import MemorySaver
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

# Import advanced algorithms
from ..algorithms.integrated_trading import IntegratedTradingAlgorithm, IntegratedTradingConfig
from ..algorithms.amm_algorithm import AMMAlgorithm, AMMConfig
from ..algorithms.ai_prediction import PracticalPredictionAlgorithm, PredictionConfig
from ..algorithms.parallel_evm import ParallelEVMAlgorithm, ParallelEVMConfig
from ..algorithms.advanced_risk import AdvancedRiskAlgorithm, RiskConfig

logger = logging.getLogger(__name__)

# Strict type definitions for LangGraph - dict-based only
from typing import TypedDict

class MarketData(TypedDict):
    price: float
    volume: float
    volatility: float
    change_24h: float
    timestamp: str
    source: str

class RiskMetrics(TypedDict):
    position_risk: float
    market_risk: float
    liquidity_risk: float
    concentration_risk: float
    overall_risk_score: float

class TradingSignal(TypedDict):
    type: Literal["entry", "exit", "hold"]
    action: Literal["buy", "sell", "hold"]
    confidence: float
    reasoning: str
    timestamp: str

class PositionSizing(TypedDict):
    suggested_size: float
    max_size: float
    risk_adjustment: float
    kelly_ratio: float

class ExecutionDecision(TypedDict):
    execute: bool
    action: Optional[Literal["buy", "sell"]]
    size: Optional[float]
    reasoning: str
    risk_score: float

class MonitoringAlert(TypedDict):
    type: str
    message: str
    severity: Literal["info", "warning", "critical"]
    timestamp: str

class PortfolioPosition(TypedDict):
    asset: str
    amount: float
    value: float
    entry_price: float
    current_price: float
    pnl: float

class PortfolioStatus(TypedDict):
    total_value: float
    cash_balance: float
    positions: List[PortfolioPosition]
    daily_pnl: float
    total_pnl: float
    last_updated: str

class WorkflowStep(Enum):
    MARKET_ANALYSIS = "market_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    SIGNAL_GENERATION = "signal_generation"
    POSITION_SIZING = "position_sizing"
    EXECUTION_DECISION = "execution_decision"
    MONITORING = "monitoring"
    ERROR_RECOVERY = "error_recovery"

class TradingState(TypedDict):
    """Strict state definition for LangGraph trading workflow - NO Dict[str, Any]"""
    market_data: MarketData
    historical_data: List[MarketData]
    portfolio_status: PortfolioStatus
    risk_metrics: RiskMetrics
    trading_signals: List[TradingSignal]
    position_sizing: PositionSizing
    execution_decision: ExecutionDecision
    monitoring_alerts: List[MonitoringAlert]
    current_step: WorkflowStep
    error_messages: List[str]
    retry_count: int
    max_retries: int

class RiskManager:
    """Advanced risk management using LangGraph state management"""
    
    def __init__(self, config):
        self.config = config
        self.risk_thresholds = {
            'max_position_size': 0.1,  # 10% of portfolio
            'max_daily_loss': 0.05,   # 5% daily loss limit
            'max_drawdown': 0.15,     # 15% max drawdown
            'volatility_threshold': 0.3,  # 30% volatility limit
            'correlation_threshold': 0.7  # 70% correlation limit
        }
    
    def assess_risk(self, state: TradingState) -> Dict[str, Any]:
        """Comprehensive risk assessment"""
        risk_metrics = {
            'position_risk': self._calculate_position_risk(state),
            'market_risk': self._calculate_market_risk(state),
            'liquidity_risk': self._calculate_liquidity_risk(state),
            'concentration_risk': self._calculate_concentration_risk(state),
            'overall_risk_score': 0.0
        }
        
        # Calculate overall risk score
        risk_metrics['overall_risk_score'] = sum([
            risk_metrics['position_risk'],
            risk_metrics['market_risk'],
            risk_metrics['liquidity_risk'],
            risk_metrics['concentration_risk']
        ]) / 4
        
        return risk_metrics
    
    def _calculate_position_risk(self, state: TradingState) -> float:
        """Calculate position-based risk"""
        portfolio_value = state['portfolio_status'].get('total_value', 1.0)
        position_size = state['position_sizing'].get('suggested_size', 0.0)
        
        if portfolio_value > 0:
            position_ratio = position_size / portfolio_value
            return min(position_ratio / self.risk_thresholds['max_position_size'], 1.0)
        return 0.0
    
    def _calculate_market_risk(self, state: TradingState) -> float:
        """Calculate market-based risk"""
        market_data = state['market_data']
        volatility = market_data.get('volatility', 0.0)
        return min(volatility / self.risk_thresholds['volatility_threshold'], 1.0)
    
    def _calculate_liquidity_risk(self, state: TradingState) -> float:
        """Calculate liquidity risk"""
        market_data = state['market_data']
        volume = market_data.get('volume_24h', 0.0)
        avg_volume = market_data.get('avg_volume_30d', volume)
        
        if avg_volume > 0:
            liquidity_ratio = volume / avg_volume
            return max(0.0, 1.0 - liquidity_ratio)
        return 0.5
    
    def _calculate_concentration_risk(self, state: TradingState) -> float:
        """Calculate portfolio concentration risk"""
        portfolio = state['portfolio_status']
        positions = portfolio.get('positions', [])
        
        if not positions:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index
        total_value = sum(pos.get('value', 0) for pos in positions)
        if total_value == 0:
            return 0.0
        
        hhi = sum((pos.get('value', 0) / total_value) ** 2 for pos in positions)
        return min(hhi, 1.0)

class TradingChain:
    """Main trading chain with LangGraph integration for complex state management"""
    
    def __init__(self, llm, trading_agent, analysis_agent, config):
        self.llm = llm
        self.trading_agent = trading_agent
        self.analysis_agent = analysis_agent
        self.config = config
        self.risk_manager = RiskManager(config)
        self.memory = MemorySaver()
        
        # Initialize advanced algorithms
        self.integrated_trading = IntegratedTradingAlgorithm(IntegratedTradingConfig())
        self.amm_algorithm = AMMAlgorithm(AMMConfig())
        self.ai_prediction = PracticalPredictionAlgorithm(PredictionConfig())
        self.parallel_evm = ParallelEVMAlgorithm(ParallelEVMConfig())
        self.advanced_risk = AdvancedRiskAlgorithm(RiskConfig())
        
        self._setup_langgraph_workflow()
    
    def _setup_langgraph_workflow(self):
        """Setup LangGraph-based trading workflow with conditional edges and error recovery"""
        # Create the state graph
        workflow = StateGraph(TradingState)
        
        # Add nodes for each step in the trading process
        workflow.add_node("market_analysis", self._market_analysis_node)
        workflow.add_node("advanced_analysis", self._advanced_analysis_node)
        workflow.add_node("risk_assessment", self._risk_assessment_node)
        workflow.add_node("signal_generation", self._signal_generation_node)
        workflow.add_node("position_sizing", self._position_sizing_node)
        workflow.add_node("execution_decision", self._execution_decision_node)
        workflow.add_node("monitoring", self._monitoring_node)
        workflow.add_node("error_recovery", self._error_recovery_node)
        
        # Define the workflow edges with conditional logic
        workflow.set_entry_point("market_analysis")
        
        # Conditional edges based on success/failure
        workflow.add_conditional_edges(
            "market_analysis",
            self._should_proceed_to_advanced,
            {
                "proceed": "advanced_analysis",
                "retry": "error_recovery",
                "fail": END
            }
        )
        
        workflow.add_conditional_edges(
            "advanced_analysis",
            self._should_proceed_to_risk,
            {
                "proceed": "risk_assessment",
                "retry": "error_recovery",
                "fail": END
            }
        )
        
        workflow.add_conditional_edges(
            "risk_assessment",
            self._should_proceed_to_signals,
            {
                "proceed": "signal_generation",
                "retry": "error_recovery",
                "fail": END
            }
        )
        
        workflow.add_conditional_edges(
            "signal_generation",
            self._should_proceed_to_sizing,
            {
                "proceed": "position_sizing",
                "retry": "error_recovery",
                "fail": END
            }
        )
        
        workflow.add_conditional_edges(
            "position_sizing",
            self._should_proceed_to_execution,
            {
                "proceed": "execution_decision",
                "retry": "error_recovery",
                "fail": END
            }
        )
        
        workflow.add_conditional_edges(
            "execution_decision",
            self._should_proceed_to_monitoring,
            {
                "proceed": "monitoring",
                "retry": "error_recovery",
                "fail": END
            }
        )
        
        workflow.add_edge("monitoring", END)
        
        # Error recovery can retry or fail
        workflow.add_conditional_edges(
            "error_recovery",
            self._should_retry_or_fail,
            {
                "retry": "market_analysis",
                "fail": END
            }
        )
        
        # Compile the workflow with memory checkpointing - NO FALLBACKS
        try:
            self.workflow = workflow.compile(
                checkpointer=MemorySaver()
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize memory checkpointing: {str(e)}")
    
    def _should_proceed_to_advanced(self, state: TradingState) -> str:
        """Determine if we should proceed to advanced analysis"""
        if state["error_messages"]:
            return "retry" if state["retry_count"] < state["max_retries"] else "fail"
        return "proceed"
    
    def _should_proceed_to_risk(self, state: TradingState) -> str:
        """Determine if advanced analysis was successful"""
        if state["error_messages"]:
            return "retry" if state["retry_count"] < state["max_retries"] else "fail"
        return "proceed"
    
    def _should_proceed_to_signals(self, state: TradingState) -> str:
        """Determine if risk assessment was successful"""
        if state["error_messages"] or state["risk_metrics"]["overall_risk_score"] > 0.9:
            return "retry" if state["retry_count"] < state["max_retries"] else "fail"
        return "proceed"
    
    def _should_proceed_to_sizing(self, state: TradingState) -> str:
        """Determine if signal generation was successful"""
        if state["error_messages"] or not state["trading_signals"]:
            return "retry" if state["retry_count"] < state["max_retries"] else "fail"
        return "proceed"
    
    def _should_proceed_to_execution(self, state: TradingState) -> str:
        """Determine if position sizing was successful"""
        if state["error_messages"] or state["position_sizing"]["suggested_size"] <= 0:
            return "retry" if state["retry_count"] < state["max_retries"] else "fail"
        return "proceed"
    
    def _should_proceed_to_monitoring(self, state: TradingState) -> str:
        """Determine if execution decision was successful"""
        if state["error_messages"]:
            return "retry" if state["retry_count"] < state["max_retries"] else "fail"
        return "proceed"
    
    def _should_retry_or_fail(self, state: TradingState) -> str:
        """Determine if error recovery should retry or fail"""
        if state["retry_count"] >= state["max_retries"]:
            return "fail"
        return "retry"
    
    async def _error_recovery_node(self, state: TradingState) -> TradingState:
        """Error recovery node with retry logic"""
        try:
            logger.warning(f"Error recovery triggered. Retry count: {state['retry_count']}")
            
            # Clear previous errors
            state["error_messages"] = []
            state["retry_count"] += 1
            
            # Add recovery alert (TypedDict is dict-based)
            recovery_alert: MonitoringAlert = {
                "type": "error_recovery",
                "message": f"Attempting recovery (attempt {state['retry_count']})",
                "severity": "warning",
                "timestamp": datetime.now().isoformat()
            }
            state["monitoring_alerts"].append(recovery_alert)
            
            # Reset current step to beginning
            state["current_step"] = WorkflowStep.MARKET_ANALYSIS
            
            logger.info("Error recovery completed, retrying from market analysis")
            return state
            
        except Exception as e:
            logger.error(f"Error recovery failed: {e}")
            state["error_messages"].append(f"Error recovery failed: {str(e)}")
            return state
    
    async def _market_analysis_node(self, state: TradingState) -> TradingState:
        """Market analysis node using LangChain with strict typing"""
        try:
            logger.info("Executing market analysis...")
            
            # Create market analysis prompt
            market_analysis_prompt = PromptTemplate(
                input_variables=["market_data", "historical_data"],
                template="""
                Analyze the current market data and historical trends:
                
                Current Market Data: {market_data}
                Historical Data: {historical_data}
                
                Provide a comprehensive market analysis including:
                1. Market sentiment (bullish/bearish/neutral)
                2. Key support/resistance levels
                3. Volume analysis
                4. Trend direction
                5. Risk factors
                6. Volatility assessment
                
                Analysis:
                """
            )
            
            # Execute market analysis
            analysis_chain = market_analysis_prompt | self.llm | StrOutputParser()
            analysis_result = await analysis_chain.ainvoke({
                "market_data": str(state["market_data"]),
                "historical_data": str(state["historical_data"])
            })
            
            # Update state with proper typing
            state["current_step"] = WorkflowStep.MARKET_ANALYSIS
            
            # Update market data with analysis (TypedDict is dict-based)
            state["market_data"]["analysis"] = analysis_result
            
            logger.info("Market analysis completed")
            return state
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            state["error_messages"].append(f"Market analysis error: {str(e)}")
            return state
    
    def _advanced_analysis_node(self, state: TradingState) -> TradingState:
        """Advanced analysis using integrated trading algorithms"""
        try:
            logger.info("Starting advanced analysis with integrated algorithms")
            
            # Prepare data for advanced algorithms
            market_data = state["market_data"]
            portfolio_data = {
                "returns": [d.get("price_change", 0) for d in state["historical_data"]],
                "values": [d.get("price", 0) for d in state["historical_data"]],
                "position_returns": {},
                "position_weights": {},
                "position_sizes": {},
                "market_volumes": {"ETH": market_data.get("volume", 0)},
                "historical_data": state["historical_data"]
            }
            
            # Run integrated trading analysis
            trading_result = asyncio.run(
                self.integrated_trading.run_integrated_trading_cycle(market_data, portfolio_data)
            )
            
            # Update state with advanced analysis results
            state["current_step"] = "advanced_analysis"
            
            # Add advanced analysis results to market data
            if "advanced_analysis" not in state["market_data"]:
                state["market_data"]["advanced_analysis"] = {}
            
            state["market_data"]["advanced_analysis"].update({
                "strategy_used": trading_result["strategy_used"],
                "market_condition": trading_result["market_condition"],
                "ai_prediction": trading_result.get("ai_prediction", {}),
                "amm_analysis": trading_result.get("amm_analysis", {}),
                "risk_metrics": trading_result.get("risk_metrics", {}),
                "confidence_score": trading_result.get("confidence_score", 0.5)
            })
            
            logger.info(f"Advanced analysis completed: {trading_result['strategy_used']}")
            
        except Exception as e:
            logger.error(f"Advanced analysis failed: {e}")
            state["error_messages"].append(f"Advanced analysis error: {str(e)}")
            state["current_step"] = "advanced_analysis"
        
        return state
    
    async def _risk_assessment_node(self, state: TradingState) -> TradingState:
        """Risk assessment node using RiskManager"""
        try:
            logger.info("Executing risk assessment...")
            
            # Perform comprehensive risk assessment
            risk_metrics = self.risk_manager.assess_risk(state)
            
            # Add risk-based alerts
            alerts = []
            if risk_metrics['overall_risk_score'] > 0.8:
                alerts.append({
                    "type": "high_risk",
                    "message": "Overall risk score exceeds threshold",
                    "severity": "critical"
                })
            
            if risk_metrics['position_risk'] > 0.7:
                alerts.append({
                    "type": "position_risk",
                    "message": "Position size exceeds risk threshold",
                    "severity": "warning"
                })
            
            # Update state
            state["current_step"] = "risk_assessment"
            state["risk_metrics"] = risk_metrics
            state["monitoring_alerts"].extend(alerts)
            
            logger.info("Risk assessment completed")
            return state
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            state["error_messages"].append(f"Risk assessment error: {str(e)}")
            return state
    
    async def _signal_generation_node(self, state: TradingState) -> TradingState:
        """Trading signal generation node"""
        try:
            logger.info("Generating trading signals...")
            
            # Create signal generation prompt
            signal_prompt = PromptTemplate(
                input_variables=["market_data", "risk_metrics"],
                template="""
                Based on the market analysis and risk assessment, generate trading signals:
                
                Market Analysis: {market_data}
                Risk Metrics: {risk_metrics}
                
                Generate signals for:
                1. Entry signals (buy/sell/hold)
                2. Exit signals (stop_loss/take_profit)
                3. Position management
                4. Risk adjustments
                
                Consider the risk metrics in your recommendations.
                
                Signals:
                """
            )
            
            # Generate trading signals
            signal_chain = signal_prompt | self.llm | StrOutputParser()
            signals_result = await signal_chain.ainvoke({
                "market_data": state["market_data"],
                "risk_metrics": state["risk_metrics"]
            })
            
            # Parse signals (simplified for now)
            signals = [{
                "type": "entry",
                "action": "buy" if "buy" in signals_result.lower() else "sell" if "sell" in signals_result.lower() else "hold",
                "confidence": 0.8,
                "reasoning": signals_result
            }]
            
            # Update state
            state["current_step"] = "signal_generation"
            state["trading_signals"] = signals
            
            logger.info("Trading signals generated")
            return state
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            state["error_messages"].append(f"Signal generation error: {str(e)}")
            return state
    
    async def _position_sizing_node(self, state: TradingState) -> TradingState:
        """Position sizing node based on risk metrics"""
        try:
            logger.info("Calculating position sizing...")
            
            # Calculate position size based on risk metrics
            portfolio_value = state["portfolio_status"].get("total_value", 1.0)
            risk_score = state["risk_metrics"].get("overall_risk_score", 0.5)
            
            # Kelly Criterion-based position sizing
            base_position_size = portfolio_value * 0.02  # 2% base position
            risk_adjusted_size = base_position_size * (1 - risk_score)
            
            # Apply additional risk controls
            max_position = portfolio_value * self.risk_manager.risk_thresholds['max_position_size']
            final_size = min(risk_adjusted_size, max_position)
            
            position_sizing = {
                "suggested_size": final_size,
                "max_size": max_position,
                "risk_adjustment": 1 - risk_score,
                "kelly_ratio": 0.02
            }
            
            # Update state
            state["current_step"] = "position_sizing"
            state["position_sizing"] = position_sizing
            
            logger.info("Position sizing calculated")
            return state
            
        except Exception as e:
            logger.error(f"Position sizing failed: {e}")
            state["error_messages"].append(f"Position sizing error: {str(e)}")
            return state
    
    async def _execution_decision_node(self, state: TradingState) -> TradingState:
        """Final execution decision node"""
        try:
            logger.info("Making execution decision...")
            
            # Combine all information for final decision
            signals = state["trading_signals"]
            position_sizing = state["position_sizing"]
            risk_metrics = state["risk_metrics"]
            
            # Make execution decision
            if signals and position_sizing["suggested_size"] > 0:
                execution_decision = {
                    "execute": True,
                    "action": signals[0]["action"],
                    "size": position_sizing["suggested_size"],
                    "reasoning": signals[0]["reasoning"],
                    "risk_score": risk_metrics["overall_risk_score"]
                }
            else:
                execution_decision = {
                    "execute": False,
                    "reason": "No valid signals or position size too small",
                    "risk_score": risk_metrics["overall_risk_score"]
                }
            
            # Update state
            state["current_step"] = "execution_decision"
            state["execution_decision"] = execution_decision
            
            logger.info("Execution decision made")
            return state
            
        except Exception as e:
            logger.error(f"Execution decision failed: {e}")
            state["error_messages"].append(f"Execution decision error: {str(e)}")
            return state
    
    async def _monitoring_node(self, state: TradingState) -> TradingState:
        """Monitoring and alerting node"""
        try:
            logger.info("Setting up monitoring...")
            
            # Set up real-time monitoring based on execution decision
            if state["execution_decision"].get("execute", False):
                # Add monitoring alerts for executed trades
                state["monitoring_alerts"].append({
                    "type": "trade_executed",
                    "message": f"Trade executed: {state['execution_decision']['action']} {state['execution_decision']['size']}",
                    "severity": "info"
                })
            
            # Update state
            state["current_step"] = "monitoring"
            
            logger.info("Monitoring setup completed")
            return state
            
        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
            state["error_messages"].append(f"Monitoring error: {str(e)}")
            return state
    
    async def execute_trading_workflow(
        self, 
        market_data: MarketData,
        historical_data: List[MarketData],
        portfolio_status: PortfolioStatus
    ) -> Dict[str, Any]:
        """Execute the complete LangGraph-based trading workflow with strict typing"""
        try:
            # Initialize state with proper types (TypedDict is dict-based)
            initial_state: TradingState = {
                "market_data": market_data,
                "historical_data": historical_data,
                "portfolio_status": portfolio_status,
                "risk_metrics": {
                    "position_risk": 0.0,
                    "market_risk": 0.0,
                    "liquidity_risk": 0.0,
                    "concentration_risk": 0.0,
                    "overall_risk_score": 0.0
                },
                "trading_signals": [],
                "position_sizing": {
                    "suggested_size": 0.0,
                    "max_size": 0.0,
                    "risk_adjustment": 0.0,
                    "kelly_ratio": 0.0
                },
                "execution_decision": {
                    "execute": False,
                    "action": None,
                    "size": None,
                    "reasoning": "",
                    "risk_score": 0.0
                },
                "monitoring_alerts": [],
                "current_step": WorkflowStep.MARKET_ANALYSIS,
                "error_messages": [],
                "retry_count": 0,
                "max_retries": 3
            }
            
            # Execute the workflow
            config = {"configurable": {"thread_id": "trading_workflow"}}
            final_state = await self.workflow.ainvoke(initial_state, config=config)
            
            # Return results with proper typing
            return {
                "success": True,
                "final_state": final_state,
                "execution_decision": final_state["execution_decision"],
                "risk_metrics": final_state["risk_metrics"],
                "monitoring_alerts": final_state["monitoring_alerts"],
                "error_messages": final_state["error_messages"]
            }
            
        except Exception as e:
            logger.error(f"Trading workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_decision": {
                    "execute": False,
                    "action": None,
                    "size": None,
                    "reasoning": f"Workflow error: {str(e)}",
                    "risk_score": 1.0
                }
            }
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get information about the LangGraph-based trading chain"""
        return {
            "chain_type": "LangGraphTradingChain",
            "components": [
                "Market Analysis Node",
                "Risk Assessment Node",
                "Signal Generation Node",
                "Position Sizing Node",
                "Execution Decision Node",
                "Monitoring Node"
            ],
            "features": [
                "State-based workflow management",
                "Real-time risk assessment",
                "Dynamic position sizing",
                "Comprehensive monitoring",
                "Error handling and recovery"
            ],
            "input_variables": [
                "market_data",
                "historical_data", 
                "portfolio_status"
            ],
            "output_format": "structured_trading_decision_with_risk_metrics"
        }
