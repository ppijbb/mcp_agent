"""
Ethereum Trading Prompts

This module contains prompt templates for various trading operations:
1. Market Analysis Prompts
2. Trading Decision Prompts
3. Risk Management Prompts
4. Portfolio Management Prompts
"""

from langchain.prompts import PromptTemplate

# Market Analysis Prompts
MARKET_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["market_data", "timeframe", "analysis_type"],
    template="""
    You are an expert cryptocurrency market analyst. Analyze the following market data:
    
    Market Data: {market_data}
    Timeframe: {timeframe}
    Analysis Type: {analysis_type}
    
    Provide a comprehensive analysis including:
    1. Current market sentiment (bullish/bearish/neutral)
    2. Key support and resistance levels
    3. Volume analysis and trends
    4. Technical indicators interpretation
    5. Risk factors and opportunities
    6. Short-term and medium-term outlook
    
    Your analysis should be:
    - Data-driven and objective
    - Specific with price levels
    - Actionable for traders
    - Risk-aware and balanced
    
    Analysis:
    """
)

TECHNICAL_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["price_data", "indicators", "timeframe"],
    template="""
    Perform detailed technical analysis on the following data:
    
    Price Data: {price_data}
    Technical Indicators: {indicators}
    Timeframe: {timeframe}
    
    Analyze:
    1. Price trends and patterns
    2. Moving averages and crossovers
    3. RSI, MACD, and Bollinger Bands
    4. Support and resistance levels
    5. Volume-price relationships
    6. Chart patterns and formations
    
    Provide specific price levels and actionable insights.
    
    Technical Analysis:
    """
)

# Trading Decision Prompts
TRADING_SIGNAL_PROMPT = PromptTemplate(
    input_variables=["market_analysis", "trading_strategy", "risk_tolerance"],
    template="""
    Based on the market analysis, generate trading signals:
    
    Market Analysis: {market_analysis}
    Trading Strategy: {trading_strategy}
    Risk Tolerance: {risk_tolerance}
    
    Generate:
    1. Entry signals (BUY/SELL/HOLD)
    2. Entry price levels
    3. Stop loss levels
    4. Take profit targets
    5. Position sizing recommendations
    6. Risk-reward ratios
    
    Ensure all recommendations align with the specified risk tolerance.
    
    Trading Signals:
    """
)

ENTRY_DECISION_PROMPT = PromptTemplate(
    input_variables=["trading_signals", "current_position", "market_conditions"],
    template="""
    Make entry decisions based on trading signals:
    
    Trading Signals: {trading_signals}
    Current Position: {current_position}
    Market Conditions: {market_conditions}
    
    Decision:
    1. Execute entry (YES/NO)
    2. Entry timing (immediate/wait for pullback)
    3. Entry method (market/limit order)
    4. Position size adjustment
    5. Additional risk controls
    
    Trading Decision:
    """
)

# Risk Management Prompts
RISK_ASSESSMENT_PROMPT = PromptTemplate(
    input_variables=["trading_plan", "portfolio_status", "market_volatility"],
    template="""
    Assess the risk of the proposed trading plan:
    
    Trading Plan: {trading_plan}
    Portfolio Status: {portfolio_status}
    Market Volatility: {market_volatility}
    
    Risk Assessment:
    1. Portfolio risk exposure
    2. Correlation with existing positions
    3. Maximum potential loss
    4. Risk-adjusted return expectations
    5. Position sizing limits
    6. Risk mitigation strategies
    
    Provide specific risk metrics and recommendations.
    
    Risk Assessment:
    """
)

POSITION_SIZING_PROMPT = PromptTemplate(
    input_variables=["account_size", "risk_per_trade", "stop_loss_distance"],
    template="""
    Calculate optimal position size based on risk parameters:
    
    Account Size: {account_size}
    Risk Per Trade: {risk_per_trade}
    Stop Loss Distance: {stop_loss_distance}
    
    Calculate:
    1. Maximum position size
    2. Risk-adjusted position size
    3. Number of contracts/shares
    4. Dollar risk per position
    5. Portfolio impact assessment
    
    Position Sizing Calculation:
    """
)

# Portfolio Management Prompts
PORTFOLIO_REVIEW_PROMPT = PromptTemplate(
    input_variables=["portfolio_positions", "performance_metrics", "market_outlook"],
    template="""
    Review and optimize the current portfolio:
    
    Portfolio Positions: {portfolio_positions}
    Performance Metrics: {performance_metrics}
    Market Outlook: {market_outlook}
    
    Portfolio Review:
    1. Position performance analysis
    2. Risk exposure assessment
    3. Diversification analysis
    4. Rebalancing recommendations
    5. Exit strategy for underperformers
    6. New opportunity identification
    
    Portfolio Recommendations:
    """
)

REBALANCING_PROMPT = PromptTemplate(
    input_variables=["current_allocation", "target_allocation", "market_conditions"],
    template="""
    Generate portfolio rebalancing recommendations:
    
    Current Allocation: {current_allocation}
    Target Allocation: {target_allocation}
    Market Conditions: {market_conditions}
    
    Rebalancing Plan:
    1. Required position adjustments
    2. Priority of changes
    3. Implementation timing
    4. Transaction costs consideration
    5. Tax implications
    6. Risk management during rebalancing
    
    Rebalancing Recommendations:
    """
)

# Market Research Prompts
TOKEN_RESEARCH_PROMPT = PromptTemplate(
    input_variables=["token_info", "market_data", "ecosystem_analysis"],
    template="""
    Research and analyze a cryptocurrency token:
    
    Token Information: {token_info}
    Market Data: {market_data}
    Ecosystem Analysis: {ecosystem_analysis}
    
    Research Analysis:
    1. Token utility and use cases
    2. Technology and innovation assessment
    3. Team and development activity
    4. Market adoption and growth
    5. Competitive landscape
    6. Investment thesis and risks
    
    Provide a comprehensive investment analysis.
    
    Token Research Report:
    """
)

SENTIMENT_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["social_media_data", "news_data", "community_activity"],
    template="""
    Analyze market sentiment from various sources:
    
    Social Media Data: {social_media_data}
    News Data: {news_data}
    Community Activity: {community_activity}
    
    Sentiment Analysis:
    1. Overall market sentiment
    2. Social media sentiment trends
    3. News impact assessment
    4. Community engagement analysis
    5. Influencer sentiment
    6. Sentiment change drivers
    
    Sentiment Analysis Results:
    """
)

# Performance Tracking Prompts
PERFORMANCE_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["trading_history", "performance_metrics", "benchmark_data"],
    template="""
    Analyze trading performance and identify areas for improvement:
    
    Trading History: {trading_history}
    Performance Metrics: {performance_metrics}
    Benchmark Data: {benchmark_data}
    
    Performance Analysis:
    1. Win rate and profit factor
    2. Risk-adjusted returns
    3. Drawdown analysis
    4. Trade duration patterns
    5. Entry/exit timing analysis
    6. Strategy effectiveness assessment
    
    Performance Insights and Recommendations:
    """
)

# Custom Prompt Builder
def create_custom_prompt(
    template: str,
    input_variables: list,
    system_message: str = None
) -> PromptTemplate:
    """
    Create a custom prompt template with optional system message
    
    Args:
        template: The prompt template string
        input_variables: List of input variable names
        system_message: Optional system message to prepend
    
    Returns:
        PromptTemplate object
    """
    if system_message:
        full_template = f"System: {system_message}\n\n{template}"
    else:
        full_template = template
    
    return PromptTemplate(
        input_variables=input_variables,
        template=full_template
    )

# Prompt Collection
TRADING_PROMPTS = {
    "market_analysis": MARKET_ANALYSIS_PROMPT,
    "technical_analysis": TECHNICAL_ANALYSIS_PROMPT,
    "trading_signal": TRADING_SIGNAL_PROMPT,
    "entry_decision": ENTRY_DECISION_PROMPT,
    "risk_assessment": RISK_ASSESSMENT_PROMPT,
    "position_sizing": POSITION_SIZING_PROMPT,
    "portfolio_review": PORTFOLIO_REVIEW_PROMPT,
    "rebalancing": REBALANCING_PROMPT,
    "token_research": TOKEN_RESEARCH_PROMPT,
    "sentiment_analysis": SENTIMENT_ANALYSIS_PROMPT,
    "performance_analysis": PERFORMANCE_ANALYSIS_PROMPT
}

def get_prompt(prompt_name: str) -> PromptTemplate:
    """
    Get a prompt template by name
    
    Args:
        prompt_name: Name of the prompt template
    
    Returns:
        PromptTemplate object
    
    Raises:
        KeyError: If prompt name doesn't exist
    """
    if prompt_name not in TRADING_PROMPTS:
        raise KeyError(f"Prompt '{prompt_name}' not found. Available prompts: {list(TRADING_PROMPTS.keys())}")
    
    return TRADING_PROMPTS[prompt_name]

def list_available_prompts() -> list:
    """
    Get list of available prompt names
    
    Returns:
        List of prompt names
    """
    return list(TRADING_PROMPTS.keys())
