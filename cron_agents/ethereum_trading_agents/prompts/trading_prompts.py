"""
Ethereum Trading Prompts - Enhanced Strategic Version

This module contains structured and strategic prompt templates for various trading operations:
1. Market Analysis Prompts with structured data
2. Strategic Trading Decision Prompts
3. Risk Management Prompts with quantitative parameters
4. Portfolio Management Prompts
5. Information Gathering and Research Prompts
"""

from langchain.prompts import PromptTemplate
from typing import Dict, Any, List
import json

# ============================================================================
# STRUCTURED TRADING STRATEGY PROMPTS
# ============================================================================

STRUCTURED_TRADING_STRATEGY_PROMPT = PromptTemplate(
    input_variables=["strategy_type", "asset_pair", "market_conditions", "risk_profile"],
    template="""
    You are an expert cryptocurrency trading strategist. Create a structured trading strategy based on the following parameters:
    
    Strategy Type: {strategy_type}
    Asset Pair: {asset_pair}
    Market Conditions: {market_conditions}
    Risk Profile: {risk_profile}
    
    Provide a structured response in the following JSON format:
    {{
        "strategy_name": "Descriptive strategy name",
        "entry_conditions": {{
            "technical_indicators": {{
                "moving_averages": {{
                    "short_term": "period and condition",
                    "long_term": "period and condition",
                    "crossover": "bullish/bearish"
                }},
                "rsi": {{
                    "period": "14",
                    "oversold_threshold": "30",
                    "overbought_threshold": "70"
                }},
                "macd": {{
                    "signal_line": "condition",
                    "histogram": "condition"
                }},
                "bollinger_bands": {{
                    "position": "upper/middle/lower",
                    "volatility": "high/medium/low"
                }}
            }},
            "fundamental_factors": {{
                "market_sentiment": "bullish/bearish/neutral",
                "volume_analysis": "above/below average",
                "news_impact": "positive/negative/neutral"
            }},
            "entry_price_levels": {{
                "primary": "exact price",
                "secondary": "alternative price",
                "stop_loss": "price level"
            }}
        }},
        "exit_conditions": {{
            "profit_targets": {{
                "conservative": "percentage",
                "moderate": "percentage",
                "aggressive": "percentage"
            }},
            "stop_loss": {{
                "initial": "percentage",
                "trailing": "percentage"
            }}
        }},
        "risk_management": {{
            "position_sizing": "percentage of account",
            "max_loss_per_trade": "percentage",
            "max_portfolio_risk": "percentage",
            "correlation_limits": "with existing positions"
        }},
        "execution_plan": {{
            "entry_timing": "immediate/limit order/wait for pullback",
            "order_type": "market/limit/stop",
            "scaling": "yes/no with conditions"
        }}
    }}
    
    Ensure all recommendations align with the specified risk profile and current market conditions.
    """
)

# ============================================================================
# ENHANCED MARKET ANALYSIS PROMPTS
# ============================================================================

COMPREHENSIVE_MARKET_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["market_data", "timeframe", "analysis_depth", "data_sources"],
    template="""
    You are an expert cryptocurrency market analyst with access to multiple data sources.
    Perform a comprehensive market analysis using the following structured approach:
    
    Market Data: {market_data}
    Timeframe: {timeframe}
    Analysis Depth: {analysis_depth}
    Data Sources: {data_sources}
    
    Provide analysis in this structured format:
    
    ## TECHNICAL ANALYSIS
    1. Price Action Patterns:
       - Support/Resistance levels
       - Trend direction and strength
       - Chart patterns (head & shoulders, triangles, etc.)
       - Volume analysis
    
    2. Technical Indicators:
       - Moving averages (20, 50, 200 day)
       - RSI, MACD, Bollinger Bands
       - Stochastic oscillator
       - Williams %R
    
    ## FUNDAMENTAL ANALYSIS
    1. Token Metrics:
       - Market cap and circulating supply
       - Token utility and adoption
       - Developer activity and GitHub metrics
    
    2. Network Analysis:
       - Transaction volume and fees
       - Active addresses and wallets
       - DeFi TVL and protocol usage
    
    ## SENTIMENT ANALYSIS
    1. Social Media Sentiment:
       - Twitter/X sentiment trends
       - Reddit community sentiment
       - Telegram/Discord activity
    
    2. News and Media:
       - Recent news impact
       - Regulatory developments
       - Institutional adoption news
    
    ## RISK ASSESSMENT
    1. Market Risks:
       - Volatility levels
       - Liquidity concerns
       - Correlation with other assets
    
    2. External Risks:
       - Regulatory changes
       - Security concerns
       - Market manipulation risks
    
    ## RECOMMENDATIONS
    1. Short-term outlook (1-7 days)
    2. Medium-term outlook (1-4 weeks)
    3. Key levels to watch
    4. Risk factors to monitor
    
    Provide specific price levels, percentages, and actionable insights.
    """
)

# ============================================================================
# STRATEGIC TRADING DECISION PROMPTS
# ============================================================================

STRATEGIC_ENTRY_DECISION_PROMPT = PromptTemplate(
    input_variables=["market_analysis", "trading_strategy", "portfolio_status", "risk_parameters"],
    template="""
    As a senior trading strategist, make a strategic entry decision based on comprehensive analysis:
    
    Market Analysis: {market_analysis}
    Trading Strategy: {trading_strategy}
    Portfolio Status: {portfolio_status}
    Risk Parameters: {risk_parameters}
    
    ## DECISION FRAMEWORK
    
    1. ENTRY DECISION (YES/NO/WAIT):
       - Decision: [Your decision]
       - Confidence Level: [1-10 scale]
       - Reasoning: [Detailed explanation]
    
    2. ENTRY EXECUTION:
       - Entry Method: [Market/Limit/Stop order]
       - Entry Price: [Specific price level]
       - Entry Timing: [Immediate/Wait for pullback/Specific time]
       - Position Size: [Percentage of account]
    
    3. RISK CONTROLS:
       - Stop Loss: [Exact price level]
       - Take Profit: [Multiple targets with percentages]
       - Trailing Stop: [Yes/No with conditions]
       - Position Scaling: [Entry in parts or all at once]
    
    4. MONITORING REQUIREMENTS:
       - Key Levels to Watch: [Specific prices]
       - Time-based Exits: [If not profitable by X time]
       - News Events: [Upcoming events that could impact]
    
    5. ALTERNATIVE SCENARIOS:
       - If entry fails: [Plan B]
       - If market changes: [Adaptation strategy]
       - If news breaks: [Response plan]
    
    Provide specific numbers, percentages, and actionable steps.
    """
)

# ============================================================================
# ENHANCED RISK MANAGEMENT PROMPTS
# ============================================================================

QUANTITATIVE_RISK_ASSESSMENT_PROMPT = PromptTemplate(
    input_variables=["trading_plan", "portfolio_status", "market_volatility", "risk_tolerance"],
    template="""
    Conduct a quantitative risk assessment for the proposed trading plan:
    
    Trading Plan: {trading_plan}
    Portfolio Status: {portfolio_status}
    Market Volatility: {market_volatility}
    Risk Tolerance: {risk_tolerance}
    
    ## RISK METRICS CALCULATION
    
    1. POSITION RISK:
       - Dollar Risk per Trade: [Calculate exact amount]
       - Portfolio Risk Exposure: [Percentage of total portfolio]
       - Maximum Drawdown Potential: [Worst-case scenario]
       - Risk-Reward Ratio: [Calculate ratio]
    
    2. PORTFOLIO RISK:
       - Correlation with Existing Positions: [Calculate correlation]
       - Portfolio Beta: [Market sensitivity]
       - VaR (Value at Risk): [95% confidence level]
       - Expected Shortfall: [Average loss beyond VaR]
    
    3. RISK MITIGATION:
       - Position Sizing Limits: [Maximum position size]
       - Stop Loss Adjustments: [Dynamic stop loss levels]
       - Hedging Strategies: [Correlation-based hedges]
       - Diversification Impact: [Portfolio balance]
    
    4. MONITORING AND ALERTS:
       - Risk Thresholds: [Specific levels to trigger alerts]
       - Rebalancing Triggers: [When to adjust positions]
       - Emergency Exits: [Conditions for immediate exit]
    
    Provide specific numbers, formulas used, and risk management recommendations.
    """
)

# ============================================================================
# INFORMATION GATHERING AND RESEARCH PROMPTS
# ============================================================================

COMPREHENSIVE_RESEARCH_PROMPT = PromptTemplate(
    input_variables=["research_topic", "data_sources", "analysis_depth", "output_format"],
    template="""
    Conduct comprehensive research on the specified topic using multiple data sources:
    
    Research Topic: {research_topic}
    Data Sources: {data_sources}
    Analysis Depth: {analysis_depth}
    Output Format: {output_format}
    
    ## RESEARCH METHODOLOGY
    
    1. PRIMARY SOURCES:
       - Official Documentation: [Token/Protocol docs]
       - GitHub Repositories: [Code analysis and activity]
       - Official Announcements: [Team communications]
       - Regulatory Filings: [Legal documents]
    
    2. SECONDARY SOURCES:
       - Financial News: [Reuters, Bloomberg, CoinDesk]
       - Social Media: [Twitter, Reddit, Telegram]
       - Expert Opinions: [Analyst reports, podcasts]
       - Academic Research: [Papers, studies]
    
    3. DATA ANALYSIS:
       - On-chain Metrics: [Blockchain data analysis]
       - Market Data: [Price, volume, order book]
       - Social Metrics: [Sentiment, engagement]
       - Network Metrics: [Transactions, addresses]
    
    4. EXPERT INSIGHTS:
       - Industry Leaders: [CEO, CTO statements]
       - Technical Analysts: [Chart analysis]
       - Fundamental Analysts: [Valuation models]
       - Risk Managers: [Risk assessment]
    
    5. COMPETITIVE ANALYSIS:
       - Direct Competitors: [Similar projects]
       - Market Position: [Market share analysis]
       - Competitive Advantages: [Unique features]
       - Market Trends: [Industry direction]
    
    Provide structured analysis with specific data points, sources, and actionable insights.
    """
)

# ============================================================================
# ENHANCED PORTFOLIO MANAGEMENT PROMPTS
# ============================================================================

STRATEGIC_PORTFOLIO_REVIEW_PROMPT = PromptTemplate(
    input_variables=["portfolio_positions", "performance_metrics", "market_outlook", "risk_parameters"],
    template="""
    Conduct a strategic portfolio review and optimization analysis:
    
    Portfolio Positions: {portfolio_positions}
    Performance Metrics: {performance_metrics}
    Market Outlook: {market_outlook}
    Risk Parameters: {risk_parameters}
    
    ## PORTFOLIO ANALYSIS
    
    1. PERFORMANCE ASSESSMENT:
       - Total Return: [Calculate percentage]
       - Risk-Adjusted Return: [Sharpe ratio, Sortino ratio]
       - Drawdown Analysis: [Maximum and current drawdown]
       - Sector Performance: [Performance by category]
    
    2. RISK EXPOSURE:
       - Portfolio Beta: [Market sensitivity]
       - Concentration Risk: [Largest positions]
       - Correlation Matrix: [Position relationships]
       - Volatility Analysis: [Portfolio volatility]
    
    3. OPTIMIZATION RECOMMENDATIONS:
       - Rebalancing Actions: [Specific trades needed]
       - Risk Reduction: [Positions to reduce]
       - Opportunity Seizing: [Positions to increase]
       - New Opportunities: [Assets to consider]
    
    4. EXECUTION PLAN:
       - Priority Actions: [Order of execution]
       - Timing: [When to execute]
       - Order Types: [Market/Limit orders]
       - Monitoring: [What to watch]
    
    Provide specific recommendations with numbers, percentages, and execution steps.
    """
)

# ============================================================================
# NEWS AND SENTIMENT ANALYSIS PROMPTS
# ============================================================================

REAL_TIME_SENTIMENT_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["news_data", "social_media", "market_data", "timeframe"],
    template="""
    Analyze real-time market sentiment from multiple sources:
    
    News Data: {news_data}
    Social Media: {social_media}
    Market Data: {market_data}
    Timeframe: {timeframe}
    
    ## SENTIMENT ANALYSIS FRAMEWORK
    
    1. NEWS IMPACT ASSESSMENT:
       - Breaking News: [Recent developments]
       - News Sentiment: [Positive/Negative/Neutral]
       - Market Impact: [Price/volume reaction]
       - Credibility Score: [Source reliability]
    
    2. SOCIAL MEDIA SENTIMENT:
       - Twitter/X Sentiment: [Trending topics, sentiment]
       - Reddit Sentiment: [Community discussions]
       - Telegram Sentiment: [Group activity]
       - Influencer Opinions: [Key opinion leaders]
    
    3. MARKET REACTION:
       - Immediate Response: [Price/volume changes]
       - Pattern Recognition: [Similar past events]
       - Volatility Impact: [Market stability]
       - Liquidity Changes: [Trading activity]
    
    4. FORWARD-LOOKING ANALYSIS:
       - Short-term Impact: [Next 24-48 hours]
       - Medium-term Impact: [Next week]
       - Long-term Implications: [Next month]
       - Risk Factors: [What could go wrong]
    
    5. TRADING IMPLICATIONS:
       - Entry Opportunities: [When to enter]
       - Exit Strategies: [When to exit]
       - Risk Management: [Stop loss levels]
       - Position Sizing: [How much to trade]
    
    Provide real-time insights with specific recommendations and risk warnings.
    """
)

# ============================================================================
# TECHNICAL ANALYSIS ENHANCEMENT PROMPTS
# ============================================================================

ADVANCED_TECHNICAL_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["price_data", "volume_data", "indicators", "timeframe", "analysis_type"],
    template="""
    Perform advanced technical analysis using multiple timeframes and indicators:
    
    Price Data: {price_data}
    Volume Data: {volume_data}
    Technical Indicators: {indicators}
    Timeframe: {timeframe}
    Analysis Type: {analysis_type}
    
    ## ADVANCED TECHNICAL ANALYSIS
    
    1. MULTI-TIMEFRAME ANALYSIS:
       - Higher Timeframe Trend: [Daily/Weekly trend]
       - Lower Timeframe Entry: [4H/1H entry points]
       - Timeframe Alignment: [Trend consistency]
       - Divergence Analysis: [Price vs indicator]
    
    2. ADVANCED PATTERN RECOGNITION:
       - Harmonic Patterns: [Gartley, Bat, Butterfly]
       - Elliott Wave: [Wave structure analysis]
       - Fibonacci Retracements: [Key levels]
       - Support/Resistance: [Dynamic levels]
    
    3. VOLUME ANALYSIS:
       - Volume Profile: [High/low volume areas]
       - Volume Divergence: [Price vs volume]
       - Accumulation/Distribution: [Smart money flow]
       - Breakout Confirmation: [Volume validation]
    
    4. INDICATOR CONFLUENCE:
       - Multiple Indicator Signals: [Converging signals]
       - Signal Strength: [Strong/Weak signals]
       - False Signal Filtering: [Noise reduction]
       - Confirmation Requirements: [Entry validation]
    
    5. RISK MANAGEMENT:
       - Key Levels: [Support/resistance]
       - Stop Loss Placement: [Optimal levels]
       - Position Sizing: [Risk-based sizing]
       - Exit Strategies: [Multiple targets]
    
    Provide specific price levels, percentages, and risk management parameters.
    """
)

# ============================================================================
# CUSTOM PROMPT BUILDER WITH ENHANCED FEATURES
# ============================================================================

def create_enhanced_strategic_prompt(
    template: str,
    input_variables: list,
    system_message: str = None,
    strategy_framework: str = None,
    risk_parameters: Dict[str, Any] = None
) -> PromptTemplate:
    """
    Create an enhanced strategic prompt template with advanced features
    
    Args:
        template: The prompt template string
        input_variables: List of input variable names
        system_message: Optional system message to prepend
        strategy_framework: Optional strategic framework to include
        risk_parameters: Optional risk management parameters
    
    Returns:
        PromptTemplate object
    """
    enhanced_template = template
    
    if system_message:
        enhanced_template = f"System: {system_message}\n\n{enhanced_template}"
    
    if strategy_framework:
        enhanced_template += f"\n\nStrategic Framework:\n{strategy_framework}"
    
    if risk_parameters:
        risk_section = "\n\nRisk Management Parameters:\n"
        for key, value in risk_parameters.items():
            risk_section += f"- {key}: {value}\n"
        enhanced_template += risk_section
    
    return PromptTemplate(
        input_variables=input_variables,
        template=enhanced_template
    )

# ============================================================================
# ENHANCED PROMPT COLLECTION
# ============================================================================

ENHANCED_TRADING_PROMPTS = {
    "structured_strategy": STRUCTURED_TRADING_STRATEGY_PROMPT,
    "comprehensive_analysis": COMPREHENSIVE_MARKET_ANALYSIS_PROMPT,
    "strategic_entry": STRATEGIC_ENTRY_DECISION_PROMPT,
    "quantitative_risk": QUANTITATIVE_RISK_ASSESSMENT_PROMPT,
    "comprehensive_research": COMPREHENSIVE_RESEARCH_PROMPT,
    "strategic_portfolio": STRATEGIC_PORTFOLIO_REVIEW_PROMPT,
    "real_time_sentiment": REAL_TIME_SENTIMENT_ANALYSIS_PROMPT,
    "advanced_technical": ADVANCED_TECHNICAL_ANALYSIS_PROMPT
}

def get_prompt(prompt_name: str) -> PromptTemplate:
    """
    Get an enhanced prompt template by name
    
    Args:
        prompt_name: Name of the enhanced prompt template
    
    Returns:
        PromptTemplate object
    
    Raises:
        KeyError: If prompt name doesn't exist
    """
    if prompt_name not in ENHANCED_TRADING_PROMPTS:
        raise KeyError(f"Enhanced prompt '{prompt_name}' not found. Available prompts: {list(ENHANCED_TRADING_PROMPTS.keys())}")
    
    return ENHANCED_TRADING_PROMPTS[prompt_name]

def list_available_prompts() -> list:
    """
    Get list of available enhanced prompt names
    
    Returns:
        List of enhanced prompt names
    """
    return list(ENHANCED_TRADING_PROMPTS.keys())

# ============================================================================
# INFORMATION SOURCES AND DATA COLLECTION
# ============================================================================

INFORMATION_SOURCES = {
    "news_apis": [
        "NewsAPI.org",
        "Alpha Vantage News",
        "CryptoCompare News",
        "CoinGecko News"
    ],
    "social_media": [
        "Twitter/X API",
        "Reddit API",
        "Telegram Channels",
        "Discord Servers"
    ],
    "technical_data": [
        "CoinGecko API",
        "CoinMarketCap API",
        "Binance API",
        "Coinbase API"
    ],
    "on_chain_data": [
        "Etherscan API",
        "Glassnode API",
        "Santiment API",
        "Messari API"
    ],
    "expert_opinions": [
        "Bloomberg Terminal",
        "Reuters",
        "CoinDesk",
        "The Block"
    ]
}

def get_information_sources() -> Dict[str, List[str]]:
    """
    Get available information sources for data collection
    
    Returns:
        Dictionary of information source categories
    """
    return INFORMATION_SOURCES
