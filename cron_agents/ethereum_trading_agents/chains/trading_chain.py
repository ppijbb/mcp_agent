"""
Ethereum Trading Chain

This chain orchestrates the complete trading workflow using LangChain:
1. Market Data Analysis
2. Trading Signal Generation
3. Risk Assessment
4. Execution Decision
"""

from typing import Dict, Any, List
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig

class TradingChain:
    """Main trading chain that orchestrates the complete trading workflow"""
    
    def __init__(self, llm, trading_agent, analysis_agent):
        self.llm = llm
        self.trading_agent = trading_agent
        self.analysis_agent = analysis_agent
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup the trading workflow chain"""
        
        # Market Analysis Chain
        market_analysis_prompt = PromptTemplate(
            input_variables=["market_data", "historical_data"],
            template="""
            Analyze the current market data and historical trends:
            
            Current Market Data: {market_data}
            Historical Data: {historical_data}
            
            Provide a comprehensive market analysis including:
            1. Market sentiment
            2. Key support/resistance levels
            3. Volume analysis
            4. Trend direction
            5. Risk factors
            
            Analysis:
            """
        )
        
        self.market_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=market_analysis_prompt,
            output_parser=StrOutputParser()
        )
        
        # Trading Signal Chain
        trading_signal_prompt = PromptTemplate(
            input_variables=["market_analysis", "trading_strategy"],
            template="""
            Based on the market analysis, generate trading signals:
            
            Market Analysis: {market_analysis}
            Trading Strategy: {trading_strategy}
            
            Generate:
            1. Entry signals (BUY/SELL)
            2. Entry price levels
            3. Stop loss levels
            4. Take profit targets
            5. Position sizing recommendations
            6. Risk assessment
            
            Trading Signals:
            """
        )
        
        self.trading_signal_chain = LLMChain(
            llm=self.llm,
            prompt=trading_signal_prompt,
            output_parser=StrOutputParser()
        )
        
        # Risk Assessment Chain
        risk_assessment_prompt = PromptTemplate(
            input_variables=["trading_signals", "portfolio_status"],
            template="""
            Assess the risk of the proposed trading signals:
            
            Trading Signals: {trading_signals}
            Portfolio Status: {portfolio_status}
            
            Evaluate:
            1. Portfolio risk exposure
            2. Correlation with existing positions
            3. Maximum drawdown potential
            4. Risk-adjusted return expectations
            5. Position sizing limits
            6. Risk mitigation strategies
            
            Risk Assessment:
            """
        )
        
        self.risk_assessment_chain = LLMChain(
            llm=self.llm,
            prompt=risk_assessment_prompt,
            output_parser=StrOutputParser()
        )
        
        # Execution Decision Chain
        execution_prompt = PromptTemplate(
            input_variables=["risk_assessment", "market_conditions"],
            template="""
            Make final execution decisions based on risk assessment:
            
            Risk Assessment: {risk_assessment}
            Market Conditions: {market_conditions}
            
            Decision:
            1. Execute trade (YES/NO)
            2. Modified entry levels
            3. Adjusted position size
            4. Additional risk controls
            5. Execution timing
            6. Monitoring requirements
            
            Execution Decision:
            """
        )
        
        self.execution_chain = LLMChain(
            llm=self.llm,
            prompt=execution_prompt,
            output_parser=StrOutputParser()
        )
    
    async def execute_trading_workflow(
        self, 
        market_data: Dict[str, Any],
        historical_data: Dict[str, Any],
        trading_strategy: str,
        portfolio_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the complete trading workflow"""
        
        try:
            # Step 1: Market Analysis
            market_analysis = await self.market_analysis_chain.arun({
                "market_data": str(market_data),
                "historical_data": str(historical_data)
            })
            
            # Step 2: Trading Signal Generation
            trading_signals = await self.trading_signal_chain.arun({
                "market_analysis": market_analysis,
                "trading_strategy": trading_strategy
            })
            
            # Step 3: Risk Assessment
            risk_assessment = await self.risk_assessment_chain.arun({
                "trading_signals": trading_signals,
                "portfolio_status": str(portfolio_status)
            })
            
            # Step 4: Execution Decision
            execution_decision = await self.execution_chain.arun({
                "risk_assessment": risk_assessment,
                "market_conditions": str(market_data)
            })
            
            return {
                "market_analysis": market_analysis,
                "trading_signals": trading_signals,
                "risk_assessment": risk_assessment,
                "execution_decision": execution_decision,
                "timestamp": self._get_timestamp(),
                "status": "completed"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed",
                "timestamp": self._get_timestamp()
            }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get information about the trading chain"""
        return {
            "chain_type": "TradingChain",
            "components": [
                "Market Analysis Chain",
                "Trading Signal Chain", 
                "Risk Assessment Chain",
                "Execution Decision Chain"
            ],
            "input_variables": [
                "market_data",
                "historical_data", 
                "trading_strategy",
                "portfolio_status"
            ],
            "output_format": "structured_trading_decision"
        }
