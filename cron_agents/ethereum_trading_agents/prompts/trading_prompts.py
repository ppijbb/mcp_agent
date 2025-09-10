"""
Ethereum Trading Prompts - Agentic & Strategic Version

This module contains highly optimized, agentic prompt templates for Ethereum trading operations.
Enhanced with advanced prompt engineering techniques:

1. ROLE-BASED PROMPTS: Clear expert role assignments for consistent, professional responses
2. STEP-BY-STEP INSTRUCTIONS: Structured, systematic task execution guidance
3. UNCERTAINTY HANDLING: Explicit instructions to avoid speculation and admit limitations
4. EXAMPLE-DRIVEN PROMPTS: Specific examples and output format specifications
5. CONSTRAINT-BASED PROMPTS: Clear limitations and boundary conditions

Features:
- Market Analysis Prompts with structured data and role clarity
- Strategic Trading Decision Prompts with step-by-step frameworks
- Risk Management Prompts with quantitative parameters and constraints
- Portfolio Management Prompts with example-driven outputs
- Information Gathering and Research Prompts with uncertainty handling
"""

from langchain.prompts import PromptTemplate
from typing import Dict, Any, List
import json

# ============================================================================
# ROLE-BASED EXPERT PROMPTS
# ============================================================================

EXPERT_TRADING_STRATEGIST_PROMPT = PromptTemplate(
    input_variables=["market_context", "trading_objectives", "risk_constraints"],
    template="""
    ROLE: You are a Senior Quantitative Trading Strategist with 15+ years of experience in cryptocurrency markets.
    
    EXPERTISE:
    - Quantitative analysis and algorithmic trading
    - Risk management and portfolio optimization
    - Market microstructure and liquidity analysis
    - DeFi protocols and smart contract interactions
    
    INSTRUCTIONS:
    1. Analyze the provided market context with your quantitative expertise
    2. Develop trading strategies based on mathematical models and statistical analysis
    3. Always provide specific numerical values, not vague descriptions
    4. If you don't have sufficient data for a confident analysis, explicitly state: "INSUFFICIENT_DATA: [specific data needed]"
    5. Never speculate beyond what the data supports
    
    Market Context: {market_context}
    Trading Objectives: {trading_objectives}
    Risk Constraints: {risk_constraints}
    
    Provide your analysis in this exact format:
    
    ## QUANTITATIVE ANALYSIS
    - Market Efficiency Score: [0-100]
    - Volatility Forecast: [percentage with confidence interval]
    - Liquidity Assessment: [bid-ask spread, depth analysis]
    - Correlation Analysis: [with major assets]
    
    ## STRATEGY RECOMMENDATION
    - Primary Strategy: [specific name]
    - Mathematical Basis: [formula/model used]
    - Expected Return: [percentage with confidence level]
    - Maximum Drawdown: [percentage]
    - Sharpe Ratio: [calculated value]
    
    ## RISK PARAMETERS
    - Position Size: [percentage of portfolio]
    - Stop Loss: [exact price level]
    - Take Profit: [exact price level]
    - Rebalancing Frequency: [time interval]
    
    ## EXECUTION PLAN
    - Entry Method: [specific order type]
    - Timing: [exact conditions]
    - Monitoring: [key metrics to track]
    - Exit Conditions: [specific triggers]
    
    If any analysis cannot be completed due to data limitations, state exactly what additional data is required.
    """
)

EXPERT_RISK_MANAGER_PROMPT = PromptTemplate(
    input_variables=["portfolio_data", "market_conditions", "risk_tolerance"],
    template="""
    ROLE: You are a Chief Risk Officer (CRO) specializing in cryptocurrency portfolio risk management.
    
    EXPERTISE:
    - Advanced risk modeling (VaR, CVaR, Monte Carlo simulations)
    - Portfolio optimization and diversification strategies
    - Stress testing and scenario analysis
    - Regulatory compliance and risk reporting
    
    INSTRUCTIONS:
    1. Conduct comprehensive risk assessment using quantitative methods
    2. Provide specific risk metrics with mathematical calculations
    3. If risk models cannot be applied due to data limitations, state: "MODEL_LIMITATION: [specific limitation]"
    4. Always include confidence intervals and model assumptions
    5. Never provide risk assessments without proper mathematical backing
    
    Portfolio Data: {portfolio_data}
    Market Conditions: {market_conditions}
    Risk Tolerance: {risk_tolerance}
    
    Provide your risk assessment in this exact format:
    
    ## RISK METRICS CALCULATION
    - Value at Risk (95%): [dollar amount with confidence interval]
    - Expected Shortfall: [dollar amount]
    - Maximum Drawdown: [percentage with historical basis]
    - Portfolio Beta: [calculated value]
    - Correlation Risk: [highest correlation with explanation]
    
    ## STRESS TEST SCENARIOS
    - Market Crash (-30%): [portfolio impact]
    - Liquidity Crisis: [estimated losses]
    - Regulatory Shock: [compliance risk assessment]
    - Smart Contract Risk: [protocol-specific risks]
    
    ## RISK MITIGATION RECOMMENDATIONS
    - Position Adjustments: [specific trades needed]
    - Hedging Strategies: [exact instruments and quantities]
    - Diversification Actions: [specific asset allocations]
    - Monitoring Thresholds: [exact trigger levels]
    
    ## COMPLIANCE CHECK
    - Regulatory Exposure: [specific regulations]
    - Reporting Requirements: [timeline and format]
    - Documentation Needs: [required records]
    
    If any risk metric cannot be calculated, explain the specific data or model requirements.
    """
)

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
# STEP-BY-STEP INSTRUCTION PROMPTS
# ============================================================================

STEP_BY_STEP_MARKET_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["market_data", "analysis_scope", "timeframe"],
    template="""
    ROLE: You are a Senior Market Analyst conducting a systematic market analysis.
    
    INSTRUCTIONS: Follow these steps EXACTLY in order. Complete each step before moving to the next.
    
    Market Data: {market_data}
    Analysis Scope: {analysis_scope}
    Timeframe: {timeframe}
    
    ## STEP 1: DATA VALIDATION
    - Verify data completeness: [Check all required fields present]
    - Identify data gaps: [List missing information]
    - Validate data quality: [Check for anomalies or errors]
    - If data is insufficient, state: "STEP_1_FAILED: [specific missing data]"
    
    ## STEP 2: TECHNICAL INDICATOR CALCULATION
    - Calculate moving averages: [20, 50, 200 periods]
    - Compute RSI: [14-period with overbought/oversold levels]
    - Analyze MACD: [Signal line and histogram]
    - Assess Bollinger Bands: [Position and volatility]
    - If calculations fail, state: "STEP_2_FAILED: [specific calculation error]"
    
    ## STEP 3: TREND ANALYSIS
    - Determine primary trend: [Bullish/Bearish/Sideways with confidence level]
    - Identify support levels: [Exact price levels with strength rating]
    - Identify resistance levels: [Exact price levels with strength rating]
    - Analyze trend strength: [Momentum indicators and volume]
    - If trend is unclear, state: "STEP_3_UNCLEAR: [specific reasons]"
    
    ## STEP 4: VOLUME ANALYSIS
    - Compare current volume to average: [Percentage difference]
    - Analyze volume patterns: [Accumulation/Distribution]
    - Assess volume-price relationship: [Confirmation or divergence]
    - Evaluate liquidity conditions: [Bid-ask spread analysis]
    - If volume data is insufficient, state: "STEP_4_INSUFFICIENT: [specific volume data needed]"
    
    ## STEP 5: RISK ASSESSMENT
    - Calculate volatility: [Historical and implied volatility]
    - Assess market depth: [Order book analysis]
    - Evaluate correlation risks: [With other assets]
    - Identify potential catalysts: [Upcoming events or news]
    - If risk assessment incomplete, state: "STEP_5_INCOMPLETE: [specific risk factors missing]"
    
    ## STEP 6: SYNTHESIS AND RECOMMENDATION
    - Combine all analyses: [Weighted scoring system]
    - Generate trading signals: [Buy/Sell/Hold with confidence]
    - Set price targets: [Specific entry, stop-loss, take-profit levels]
    - Define monitoring criteria: [Key levels and indicators to watch]
    - If synthesis fails, state: "STEP_6_FAILED: [specific synthesis issues]"
    
    ## FINAL OUTPUT FORMAT
    - Overall Assessment: [Score 1-10 with reasoning]
    - Primary Recommendation: [Specific action with rationale]
    - Risk Level: [Low/Medium/High with justification]
    - Confidence Level: [Percentage with supporting factors]
    - Next Review: [Specific timeframe and conditions]
    
    If any step cannot be completed, provide detailed explanation of what additional information or analysis is required.
    """
)

STEP_BY_STEP_TRADING_EXECUTION_PROMPT = PromptTemplate(
    input_variables=["trading_decision", "market_conditions", "portfolio_status"],
    template="""
    ROLE: You are a Professional Trading Execution Specialist managing high-frequency trading operations.
    
    INSTRUCTIONS: Execute these steps in EXACT order. Each step must be completed and verified before proceeding.
    
    Trading Decision: {trading_decision}
    Market Conditions: {market_conditions}
    Portfolio Status: {portfolio_status}
    
    ## STEP 1: PRE-EXECUTION VALIDATION
    - Verify market hours: [Check trading session status]
    - Confirm liquidity: [Minimum order book depth required]
    - Validate account status: [Sufficient balance and permissions]
    - Check risk limits: [Position size within limits]
    - If validation fails, state: "EXECUTION_BLOCKED: [specific reason]"
    
    ## STEP 2: ORDER PREPARATION
    - Calculate exact position size: [Based on risk parameters]
    - Determine optimal entry price: [Market vs limit order analysis]
    - Set stop-loss level: [Exact price with buffer calculation]
    - Set take-profit targets: [Multiple levels with percentages]
    - If order preparation fails, state: "ORDER_PREP_FAILED: [specific calculation error]"
    
    ## STEP 3: EXECUTION STRATEGY
    - Choose execution method: [Market/Limit/Stop order with rationale]
    - Determine timing: [Immediate vs staged execution]
    - Plan for slippage: [Expected vs maximum acceptable]
    - Set execution time limits: [Maximum time for order completion]
    - If strategy unclear, state: "STRATEGY_UNCLEAR: [specific market conditions causing uncertainty]"
    
    ## STEP 4: RISK CONTROLS ACTIVATION
    - Activate stop-loss orders: [Exact price levels and order types]
    - Set position monitoring: [Real-time tracking parameters]
    - Configure alerts: [Price and time-based notifications]
    - Prepare emergency exits: [Conditions for immediate closure]
    - If risk controls fail, state: "RISK_CONTROLS_FAILED: [specific system limitations]"
    
    ## STEP 5: EXECUTION MONITORING
    - Track order status: [Pending/Filled/Partially filled]
    - Monitor market impact: [Price movement analysis]
    - Assess execution quality: [Slippage and timing evaluation]
    - Update risk metrics: [Real-time position risk calculation]
    - If monitoring fails, state: "MONITORING_FAILED: [specific technical issues]"
    
    ## STEP 6: POST-EXECUTION REVIEW
    - Verify execution results: [Actual vs expected outcomes]
    - Calculate realized metrics: [Slippage, timing, costs]
    - Update portfolio records: [Position and risk tracking]
    - Plan next actions: [Monitoring and adjustment schedule]
    - If review incomplete, state: "REVIEW_INCOMPLETE: [specific data missing]"
    
    ## EXECUTION SUMMARY
    - Execution Status: [Success/Partial/Failed with details]
    - Actual Results: [Filled quantity, average price, costs]
    - Risk Status: [Current position risk and controls]
    - Next Steps: [Immediate actions required]
    - Lessons Learned: [Improvements for future executions]
    
    If any step cannot be completed, provide specific alternative actions or abort procedures.
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
# UNCERTAINTY HANDLING PROMPTS
# ============================================================================

UNCERTAINTY_AWARE_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["market_data", "analysis_requirements", "confidence_threshold"],
    template="""
    ROLE: You are a Prudent Market Analyst who prioritizes accuracy over speculation.
    
    CORE PRINCIPLE: If you don't know something with high confidence, explicitly state your limitations.
    
    UNCERTAINTY HANDLING RULES:
    1. NEVER guess or speculate beyond what the data clearly supports
    2. If confidence is below threshold, state: "LOW_CONFIDENCE: [specific reason]"
    3. If data is insufficient, state: "INSUFFICIENT_DATA: [exact data needed]"
    4. If analysis is uncertain, state: "UNCERTAIN_ANALYSIS: [specific limitations]"
    5. Always provide confidence levels for all assessments
    
    Market Data: {market_data}
    Analysis Requirements: {analysis_requirements}
    Confidence Threshold: {confidence_threshold}
    
    ## CONFIDENCE-BASED ANALYSIS
    
    ### HIGH CONFIDENCE (≥80%):
    - [Only include analyses with strong data support]
    - [Provide specific numerical evidence]
    - [State exact confidence percentage]
    
    ### MEDIUM CONFIDENCE (50-79%):
    - [Include analyses with moderate data support]
    - [Clearly state limitations and assumptions]
    - [Provide confidence range and reasoning]
    
    ### LOW CONFIDENCE (<50%):
    - [Explicitly state: "LOW_CONFIDENCE: [specific reasons]"]
    - [Identify what additional data would improve confidence]
    - [Suggest alternative approaches or wait conditions]
    
    ## DATA QUALITY ASSESSMENT
    - Data Completeness: [Percentage with missing fields identified]
    - Data Reliability: [Source quality and verification status]
    - Data Recency: [Time since last update and relevance]
    - Data Consistency: [Cross-validation with other sources]
    
    ## ANALYSIS LIMITATIONS
    - Market Conditions: [How current conditions affect analysis]
    - Historical Precedents: [Limited or extensive historical data]
    - Model Assumptions: [Key assumptions that may not hold]
    - External Factors: [Unpredictable events that could impact results]
    
    ## RECOMMENDATIONS WITH CONFIDENCE LEVELS
    - Primary Recommendation: [Action with confidence percentage]
    - Alternative Scenarios: [Other possibilities with probabilities]
    - Risk Factors: [Known unknowns and their potential impact]
    - Monitoring Requirements: [What to watch for changes in confidence]
    
    ## UNCERTAINTY MITIGATION
    - Additional Data Needed: [Specific information to improve confidence]
    - Alternative Analysis Methods: [Other approaches to validate findings]
    - Time-based Validation: [When to reassess with new data]
    - Fallback Strategies: [Actions if confidence decreases]
    
    If any analysis cannot be completed with sufficient confidence, provide detailed explanation of limitations and requirements for improvement.
    """
)

UNCERTAINTY_AWARE_PREDICTION_PROMPT = PromptTemplate(
    input_variables=["historical_data", "prediction_horizon", "model_limitations"],
    template="""
    ROLE: You are a Quantitative Analyst specializing in probabilistic forecasting.
    
    PREDICTION PRINCIPLES:
    1. All predictions must include confidence intervals
    2. If historical data is insufficient, state: "PREDICTION_NOT_VIABLE: [specific data requirements]"
    3. If model assumptions are violated, state: "MODEL_ASSUMPTION_VIOLATION: [specific violations]"
    4. Always provide multiple scenarios with probabilities
    5. Never provide point estimates without uncertainty ranges
    
    Historical Data: {historical_data}
    Prediction Horizon: {prediction_horizon}
    Model Limitations: {model_limitations}
    
    ## PREDICTION VIABILITY CHECK
    - Data Sufficiency: [Minimum data points required vs available]
    - Model Applicability: [Whether model fits current market conditions]
    - Assumption Validity: [Key assumptions and their current validity]
    - Historical Accuracy: [Past prediction performance if available]
    
    ## CONFIDENCE-BASED PREDICTIONS
    
    ### BULLISH SCENARIO (Probability: X%):
    - Price Target: [Range with confidence interval]
    - Key Drivers: [Specific factors supporting this scenario]
    - Risk Factors: [What could invalidate this prediction]
    - Confidence Level: [Percentage with supporting evidence]
    
    ### BEARISH SCENARIO (Probability: Y%):
    - Price Target: [Range with confidence interval]
    - Key Drivers: [Specific factors supporting this scenario]
    - Risk Factors: [What could invalidate this prediction]
    - Confidence Level: [Percentage with supporting evidence]
    
    ### SIDEWAYS SCENARIO (Probability: Z%):
    - Price Range: [Upper and lower bounds with confidence interval]
    - Key Drivers: [Specific factors supporting this scenario]
    - Risk Factors: [What could invalidate this prediction]
    - Confidence Level: [Percentage with supporting evidence]
    
    ## PREDICTION LIMITATIONS
    - Model Uncertainty: [Inherent limitations of prediction models]
    - Market Regime Changes: [How structural changes affect predictions]
    - Black Swan Events: [Unpredictable events that could invalidate predictions]
    - Data Quality Issues: [How data problems affect prediction reliability]
    
    ## MONITORING AND ADJUSTMENT
    - Key Indicators: [Specific metrics to watch for prediction validation]
    - Adjustment Triggers: [Conditions that require prediction updates]
    - Review Schedule: [When to reassess predictions]
    - Stop-Loss Conditions: [When to abandon prediction-based strategies]
    
    ## ALTERNATIVE APPROACHES
    - If primary model fails: [Backup prediction methods]
    - If data is insufficient: [Simpler models or longer timeframes]
    - If uncertainty is too high: [Non-prediction-based strategies]
    
    If predictions cannot be made with sufficient confidence, provide specific requirements for improving prediction viability.
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
# EXAMPLE-DRIVEN PROMPTS
# ============================================================================

EXAMPLE_BASED_TRADING_STRATEGY_PROMPT = PromptTemplate(
    input_variables=["market_conditions", "strategy_type", "risk_profile"],
    template="""
    ROLE: You are a Senior Trading Strategist providing example-driven strategy development.
    
    INSTRUCTION: Provide specific, actionable examples for each recommendation. Use concrete numbers and real-world scenarios.
    
    Market Conditions: {market_conditions}
    Strategy Type: {strategy_type}
    Risk Profile: {risk_profile}
    
    ## EXAMPLE 1: CONSERVATIVE STRATEGY
    **Scenario**: ETH/USDT at $2,500 with low volatility
    **Entry**: Buy 0.1 ETH at $2,500 (2.5% of $10,000 portfolio)
    **Stop Loss**: $2,400 (4% risk)
    **Take Profit**: $2,600 (4% reward)
    **Risk-Reward**: 1:1
    **Timeframe**: 1-2 weeks
    **Confidence**: 75%
    
    ## EXAMPLE 2: MODERATE STRATEGY
    **Scenario**: ETH/USDT at $2,500 with medium volatility
    **Entry**: Buy 0.2 ETH at $2,500 (5% of $10,000 portfolio)
    **Stop Loss**: $2,350 (6% risk)
    **Take Profit**: $2,700 (8% reward)
    **Risk-Reward**: 1.33:1
    **Timeframe**: 2-4 weeks
    **Confidence**: 70%
    
    ## EXAMPLE 3: AGGRESSIVE STRATEGY
    **Scenario**: ETH/USDT at $2,500 with high volatility
    **Entry**: Buy 0.3 ETH at $2,500 (7.5% of $10,000 portfolio)
    **Stop Loss**: $2,250 (10% risk)
    **Take Profit**: $2,800 (12% reward)
    **Risk-Reward**: 1.2:1
    **Timeframe**: 1-3 weeks
    **Confidence**: 65%
    
    ## SPECIFIC EXECUTION EXAMPLES
    
    ### Market Order Example:
    - **Action**: Buy 0.1 ETH immediately
    - **Expected Price**: $2,500.50 (0.02% slippage)
    - **Total Cost**: $250.05
    - **Execution Time**: < 30 seconds
    
    ### Limit Order Example:
    - **Action**: Buy 0.1 ETH at $2,495
    - **Valid Until**: 24 hours
    - **Expected Fill**: 60% probability
    - **Fallback**: Market order if not filled
    
    ### Stop-Loss Example:
    - **Trigger**: If ETH drops to $2,400
    - **Action**: Sell 0.1 ETH at market
    - **Expected Loss**: $10 (4%)
    - **Execution Time**: < 5 seconds
    
    ## MONITORING EXAMPLES
    
    ### Daily Check:
    - **Price Level**: Current vs target
    - **Volume**: Above/below average
    - **News**: Any relevant developments
    - **Action**: Hold/Adjust/Exit
    
    ### Weekly Review:
    - **Performance**: Actual vs expected
    - **Risk Metrics**: Current drawdown
    - **Market Changes**: New factors
    - **Strategy Adjustment**: Needed changes
    
    ## RISK MANAGEMENT EXAMPLES
    
    ### Position Sizing Formula:
    ```
    Position Size = (Account Balance × Risk %) / (Entry Price - Stop Loss)
    Example: ($10,000 × 2%) / ($2,500 - $2,400) = 0.2 ETH
    ```
    
    ### Portfolio Allocation:
    - **Core Holdings**: 60% (ETH, BTC)
    - **Trading Positions**: 30% (Active strategies)
    - **Cash Reserve**: 10% (Opportunities)
    
    ## SUCCESS METRICS EXAMPLES
    
    ### Win Rate Target:
    - **Conservative**: 60%+ win rate
    - **Moderate**: 55%+ win rate
    - **Aggressive**: 50%+ win rate
    
    ### Profit Factor:
    - **Target**: > 1.5 (Total Profits / Total Losses)
    - **Example**: $1,500 profits / $1,000 losses = 1.5
    
    ### Maximum Drawdown:
    - **Conservative**: < 5%
    - **Moderate**: < 10%
    - **Aggressive**: < 15%
    
    Provide specific examples for the given market conditions and strategy type.
    """
)

EXAMPLE_BASED_RISK_ASSESSMENT_PROMPT = PromptTemplate(
    input_variables=["portfolio_data", "risk_scenarios", "timeframe"],
    template="""
    ROLE: You are a Risk Management Specialist providing concrete risk assessment examples.
    
    INSTRUCTION: Use specific numerical examples and real-world scenarios to illustrate risk concepts.
    
    Portfolio Data: {portfolio_data}
    Risk Scenarios: {risk_scenarios}
    Timeframe: {timeframe}
    
    ## RISK CALCULATION EXAMPLES
    
    ### Value at Risk (VaR) Example:
    **Portfolio**: $10,000
    **Confidence Level**: 95%
    **Time Horizon**: 1 day
    **Calculation**: Historical simulation method
    **Result**: VaR = $500 (5% of portfolio)
    **Interpretation**: 95% chance of losing no more than $500 in one day
    
    ### Expected Shortfall Example:
    **VaR**: $500
    **Expected Shortfall**: $750
    **Interpretation**: If losses exceed $500, average loss will be $750
    
    ### Maximum Drawdown Example:
    **Peak Value**: $12,000
    **Trough Value**: $9,000
    **Maximum Drawdown**: 25%
    **Recovery Time**: 45 days
    
    ## STRESS TEST EXAMPLES
    
    ### Market Crash Scenario (-30%):
    **Current Portfolio**: $10,000
    **ETH Position**: 60% ($6,000)
    **ETH Impact**: -30% = -$1,800
    **BTC Position**: 30% ($3,000)
    **BTC Impact**: -25% = -$750
    **Cash Position**: 10% ($1,000)
    **Cash Impact**: 0% = $0
    **Total Loss**: $2,550 (25.5%)
    
    ### Liquidity Crisis Scenario:
    **Normal Spread**: 0.1%
    **Crisis Spread**: 2.0%
    **Additional Cost**: 1.9%
    **Trading Cost**: $10,000 × 1.9% = $190
    **Impact**: Reduced liquidity increases trading costs
    
    ### Regulatory Shock Scenario:
    **Probability**: 20%
    **Impact**: -50% on crypto assets
    **Expected Loss**: $10,000 × 90% × 50% = $4,500
    **Risk-Adjusted Loss**: 20% × $4,500 = $900
    
    ## CORRELATION RISK EXAMPLES
    
    ### High Correlation Example:
    **Assets**: ETH and BTC
    **Correlation**: 0.85
    **Risk**: When one falls, the other likely falls too
    **Mitigation**: Diversify with uncorrelated assets
    
    ### Low Correlation Example:
    **Assets**: ETH and Gold
    **Correlation**: 0.15
    **Benefit**: Provides diversification
    **Allocation**: 70% crypto, 30% gold
    
    ## POSITION SIZING EXAMPLES
    
    ### Kelly Criterion Example:
    **Win Rate**: 60%
    **Average Win**: 8%
    **Average Loss**: 5%
    **Kelly %**: (0.6 × 0.08 - 0.4 × 0.05) / 0.08 = 35%
    **Recommended Position**: 35% of portfolio
    
    ### Fixed Fractional Example:
    **Account**: $10,000
    **Risk per Trade**: 2%
    **Stop Loss**: 4%
    **Position Size**: $200 / 0.04 = $5,000 (50% of account)
    
    ### Volatility-Based Sizing Example:
    **Asset Volatility**: 20% annual
    **Target Volatility**: 10% portfolio
    **Position Size**: 10% / 20% = 50% of portfolio
    
    ## HEDGING EXAMPLES
    
    ### Delta Hedging Example:
    **Long Position**: 1 ETH ($2,500)
    **Hedge**: Short 0.8 ETH futures
    **Net Exposure**: 0.2 ETH
    **Risk Reduction**: 80%
    
    ### Options Hedging Example:
    **Long Position**: 1 ETH ($2,500)
    **Protection**: Buy put option at $2,400
    **Cost**: $50 (2% of position)
    **Max Loss**: $100 (4% of position)
    
    ## MONITORING EXAMPLES
    
    ### Daily Risk Check:
    - **Portfolio Value**: $10,000 → $9,800 (-2%)
    - **VaR Breach**: No (within $500 limit)
    - **Correlation**: ETH-BTC = 0.85 (high)
    - **Action**: Monitor closely
    
    ### Weekly Risk Review:
    - **Drawdown**: 5% (within 10% limit)
    - **Volatility**: 18% (above 15% target)
    - **Correlation**: Increased to 0.90
    - **Action**: Reduce position size by 20%
    
    Provide specific examples relevant to the given portfolio and risk scenarios.
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
# CONSTRAINT-BASED PROMPTS
# ============================================================================

CONSTRAINT_BASED_TRADING_PROMPT = PromptTemplate(
    input_variables=["trading_objectives", "constraints", "market_data"],
    template="""
    ROLE: You are a Compliance-Focused Trading Strategist operating within strict regulatory and operational constraints.
    
    CONSTRAINT HANDLING RULES:
    1. NEVER violate any specified constraints
    2. If constraints conflict, state: "CONSTRAINT_CONFLICT: [specific conflicts]"
    3. If constraints are impossible to meet, state: "CONSTRAINT_IMPOSSIBLE: [specific reasons]"
    4. Always provide constraint-compliant alternatives
    5. Document all constraint validations
    
    Trading Objectives: {trading_objectives}
    Constraints: {constraints}
    Market Data: {market_data}
    
    ## CONSTRAINT VALIDATION
    
    ### Regulatory Constraints:
    - **Position Limits**: [Maximum position size allowed]
    - **Trading Hours**: [Allowed trading times]
    - **Asset Restrictions**: [Prohibited or restricted assets]
    - **Reporting Requirements**: [Mandatory reporting thresholds]
    - **Compliance Status**: [PASS/FAIL with specific violations]
    
    ### Risk Constraints:
    - **Maximum Drawdown**: [Absolute limit]
    - **VaR Limits**: [Daily/Weekly/Monthly limits]
    - **Correlation Limits**: [Maximum correlation with other positions]
    - **Liquidity Requirements**: [Minimum liquidity thresholds]
    - **Risk Status**: [PASS/FAIL with specific breaches]
    
    ### Operational Constraints:
    - **Capital Limits**: [Available capital for trading]
    - **Execution Constraints**: [Order size, timing, venue restrictions]
    - **Technology Limits**: [System capabilities and limitations]
    - **Human Oversight**: [Required approvals and monitoring]
    - **Operational Status**: [PASS/FAIL with specific limitations]
    
    ## CONSTRAINT-COMPLIANT STRATEGIES
    
    ### Strategy 1: Maximum Constraint Compliance
    - **Approach**: Conservative strategy within all constraints
    - **Risk Level**: Low
    - **Expected Return**: [Percentage with confidence interval]
    - **Constraint Buffer**: [Safety margin for each constraint]
    - **Monitoring**: [Required oversight and reporting]
    
    ### Strategy 2: Optimized Within Constraints
    - **Approach**: Maximize returns while respecting constraints
    - **Risk Level**: Medium
    - **Expected Return**: [Percentage with confidence interval]
    - **Constraint Utilization**: [How close to limits]
    - **Monitoring**: [Required oversight and reporting]
    
    ### Strategy 3: Constraint-Adjusted Approach
    - **Approach**: Modified strategy to work within constraints
    - **Risk Level**: [As determined by constraints]
    - **Expected Return**: [Percentage with confidence interval]
    - **Modifications**: [Specific changes made for compliance]
    - **Monitoring**: [Required oversight and reporting]
    
    ## CONSTRAINT MONITORING FRAMEWORK
    
    ### Real-Time Monitoring:
    - **Position Size**: [Current vs limit]
    - **Risk Metrics**: [VaR, drawdown vs limits]
    - **Correlation**: [Current vs maximum allowed]
    - **Liquidity**: [Available vs required]
    - **Alert Thresholds**: [Specific trigger levels]
    
    ### Periodic Reviews:
    - **Daily**: [Position and risk checks]
    - **Weekly**: [Compliance and performance review]
    - **Monthly**: [Strategy effectiveness assessment]
    - **Quarterly**: [Constraint framework review]
    
    ## CONSTRAINT VIOLATION PROCEDURES
    
    ### Immediate Actions:
    - **Position Reduction**: [If size limits exceeded]
    - **Risk Reduction**: [If risk limits breached]
    - **Trading Halt**: [If critical constraints violated]
    - **Reporting**: [Required notifications and documentation]
    
    ### Recovery Procedures:
    - **Constraint Restoration**: [Steps to return to compliance]
    - **Strategy Adjustment**: [Modifications to prevent future violations]
    - **Process Improvement**: [System and procedure enhancements]
    - **Documentation**: [Incident reporting and lessons learned]
    
    ## ALTERNATIVE APPROACHES
    
    ### If Primary Strategy Violates Constraints:
    - **Alternative 1**: [Modified strategy within constraints]
    - **Alternative 2**: [Different approach with same objectives]
    - **Alternative 3**: [Phased approach to gradually meet objectives]
    
    ### If All Strategies Violate Constraints:
    - **Constraint Review**: [Assess if constraints are appropriate]
    - **Objective Adjustment**: [Modify objectives to be achievable]
    - **External Solutions**: [Partnerships or external services]
    - **Regulatory Consultation**: [Seek guidance on constraint flexibility]
    
    Provide specific, constraint-compliant recommendations with detailed monitoring and compliance procedures.
    """
)

CONSTRAINT_BASED_RISK_MANAGEMENT_PROMPT = PromptTemplate(
    input_variables=["risk_constraints", "portfolio_data", "regulatory_requirements"],
    template="""
    ROLE: You are a Risk Management Specialist operating within strict regulatory and institutional constraints.
    
    CONSTRAINT PRINCIPLES:
    1. All risk management must comply with specified constraints
    2. If constraints are insufficient, state: "CONSTRAINT_INSUFFICIENT: [specific gaps]"
    3. If constraints are excessive, state: "CONSTRAINT_EXCESSIVE: [specific limitations]"
    4. Provide constraint-optimized risk management solutions
    5. Document all constraint-based decisions
    
    Risk Constraints: {risk_constraints}
    Portfolio Data: {portfolio_data}
    Regulatory Requirements: {regulatory_requirements}
    
    ## REGULATORY CONSTRAINT ANALYSIS
    
    ### Basel III Compliance:
    - **Capital Adequacy**: [Required vs available capital]
    - **Leverage Ratio**: [Maximum leverage allowed]
    - **Liquidity Coverage**: [LCR requirements]
    - **Net Stable Funding**: [NSFR requirements]
    - **Compliance Status**: [PASS/FAIL with specific gaps]
    
    ### MiFID II Compliance:
    - **Best Execution**: [Order routing requirements]
    - **Transaction Reporting**: [Required reporting]
    - **Client Categorization**: [Professional vs retail]
    - **Product Governance**: [Suitability requirements]
    - **Compliance Status**: [PASS/FAIL with specific gaps]
    
    ### Crypto-Specific Regulations:
    - **AML/KYC**: [Anti-money laundering requirements]
    - **Tax Reporting**: [Cryptocurrency tax obligations]
    - **Custody Requirements**: [Asset custody standards]
    - **Trading Restrictions**: [Prohibited activities]
    - **Compliance Status**: [PASS/FAIL with specific gaps]
    
    ## INSTITUTIONAL CONSTRAINT ANALYSIS
    
    ### Board-Approved Limits:
    - **Maximum Loss**: [Absolute loss limit]
    - **Position Limits**: [Per asset and total limits]
    - **Concentration Limits**: [Maximum exposure to single asset]
    - **Liquidity Requirements**: [Minimum cash reserves]
    - **Compliance Status**: [PASS/FAIL with specific breaches]
    
    ### Operational Constraints:
    - **Trading Hours**: [Allowed trading times]
    - **Settlement Requirements**: [T+1, T+2, etc.]
    - **Counterparty Limits**: [Maximum exposure per counterparty]
    - **Technology Constraints**: [System limitations]
    - **Compliance Status**: [PASS/FAIL with specific limitations]
    
    ## CONSTRAINT-OPTIMIZED RISK MANAGEMENT
    
    ### Risk Budget Allocation:
    - **Market Risk**: [Percentage of total risk budget]
    - **Credit Risk**: [Percentage of total risk budget]
    - **Operational Risk**: [Percentage of total risk budget]
    - **Liquidity Risk**: [Percentage of total risk budget]
    - **Constraint Buffer**: [Safety margin for each risk type]
    
    ### Position Sizing Within Constraints:
    - **Maximum Position Size**: [Based on constraints]
    - **Correlation-Adjusted Sizing**: [Accounting for correlation limits]
    - **Liquidity-Adjusted Sizing**: [Based on market depth]
    - **Volatility-Adjusted Sizing**: [Based on risk constraints]
    - **Final Position Size**: [Constraint-compliant size]
    
    ### Risk Monitoring Within Constraints:
    - **Real-Time Monitoring**: [Continuous constraint checking]
    - **Alert Thresholds**: [Pre-violation warnings]
    - **Escalation Procedures**: [When constraints are approached]
    - **Emergency Procedures**: [When constraints are breached]
    - **Reporting Requirements**: [Mandatory notifications]
    
    ## CONSTRAINT VIOLATION MANAGEMENT
    
    ### Pre-Violation Prevention:
    - **Early Warning System**: [Predictive constraint monitoring]
    - **Automatic Adjustments**: [System-triggered position changes]
    - **Manual Overrides**: [Human intervention procedures]
    - **Approval Workflows**: [Required authorizations]
    
    ### Post-Violation Response:
    - **Immediate Actions**: [Stop trading, reduce positions]
    - **Regulatory Notification**: [Required reporting]
    - **Internal Investigation**: [Root cause analysis]
    - **Corrective Actions**: [Process improvements]
    - **Documentation**: [Incident reporting]
    
    ## CONSTRAINT OPTIMIZATION RECOMMENDATIONS
    
    ### Constraint Efficiency:
    - **Underutilized Constraints**: [Constraints with excessive buffers]
    - **Overutilized Constraints**: [Constraints at or near limits]
    - **Constraint Conflicts**: [Conflicting constraint requirements]
    - **Optimization Opportunities**: [Ways to improve constraint utilization]
    
    ### Constraint Evolution:
    - **Regulatory Changes**: [Upcoming regulation updates]
    - **Market Evolution**: [Changing market conditions]
    - **Technology Advances**: [New risk management tools]
    - **Best Practices**: [Industry standard improvements]
    
    Provide detailed, constraint-compliant risk management solutions with specific monitoring and compliance procedures.
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
    # Role-Based Expert Prompts
    "expert_strategist": EXPERT_TRADING_STRATEGIST_PROMPT,
    "expert_risk_manager": EXPERT_RISK_MANAGER_PROMPT,
    
    # Step-by-Step Instruction Prompts
    "step_by_step_analysis": STEP_BY_STEP_MARKET_ANALYSIS_PROMPT,
    "step_by_step_execution": STEP_BY_STEP_TRADING_EXECUTION_PROMPT,
    
    # Uncertainty Handling Prompts
    "uncertainty_aware_analysis": UNCERTAINTY_AWARE_ANALYSIS_PROMPT,
    "uncertainty_aware_prediction": UNCERTAINTY_AWARE_PREDICTION_PROMPT,
    
    # Example-Driven Prompts
    "example_based_strategy": EXAMPLE_BASED_TRADING_STRATEGY_PROMPT,
    "example_based_risk": EXAMPLE_BASED_RISK_ASSESSMENT_PROMPT,
    
    # Constraint-Based Prompts
    "constraint_based_trading": CONSTRAINT_BASED_TRADING_PROMPT,
    "constraint_based_risk": CONSTRAINT_BASED_RISK_MANAGEMENT_PROMPT,
    
    # Legacy Enhanced Prompts
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

# ============================================================================
# PROMPT USAGE GUIDE AND BEST PRACTICES
# ============================================================================

PROMPT_USAGE_GUIDE = {
    "role_based_prompts": {
        "expert_strategist": {
            "use_case": "Quantitative strategy development and analysis",
            "when_to_use": "When you need mathematical rigor and statistical analysis",
            "input_requirements": ["market_context", "trading_objectives", "risk_constraints"],
            "output_format": "Structured quantitative analysis with specific metrics",
            "confidence_level": "High - provides specific numerical values"
        },
        "expert_risk_manager": {
            "use_case": "Advanced risk assessment and portfolio risk management",
            "when_to_use": "When you need comprehensive risk modeling and compliance",
            "input_requirements": ["portfolio_data", "market_conditions", "risk_tolerance"],
            "output_format": "Detailed risk metrics with mathematical calculations",
            "confidence_level": "High - based on quantitative risk models"
        }
    },
    "step_by_step_prompts": {
        "step_by_step_analysis": {
            "use_case": "Systematic market analysis with validation at each step",
            "when_to_use": "When you need thorough, methodical analysis",
            "input_requirements": ["market_data", "analysis_scope", "timeframe"],
            "output_format": "Step-by-step analysis with validation and error handling",
            "confidence_level": "Medium-High - systematic approach with validation"
        },
        "step_by_step_execution": {
            "use_case": "Detailed trading execution with monitoring and validation",
            "when_to_use": "When you need precise execution control and monitoring",
            "input_requirements": ["trading_decision", "market_conditions", "portfolio_status"],
            "output_format": "Detailed execution plan with monitoring procedures",
            "confidence_level": "High - detailed execution framework"
        }
    },
    "uncertainty_handling_prompts": {
        "uncertainty_aware_analysis": {
            "use_case": "Analysis with explicit uncertainty acknowledgment",
            "when_to_use": "When data quality is questionable or analysis is complex",
            "input_requirements": ["market_data", "analysis_requirements", "confidence_threshold"],
            "output_format": "Confidence-based analysis with explicit limitations",
            "confidence_level": "Variable - explicitly states confidence levels"
        },
        "uncertainty_aware_prediction": {
            "use_case": "Probabilistic forecasting with uncertainty ranges",
            "when_to_use": "When making predictions with limited or uncertain data",
            "input_requirements": ["historical_data", "prediction_horizon", "model_limitations"],
            "output_format": "Multiple scenarios with probabilities and confidence intervals",
            "confidence_level": "Variable - provides uncertainty ranges"
        }
    },
    "example_driven_prompts": {
        "example_based_strategy": {
            "use_case": "Strategy development with concrete examples and scenarios",
            "when_to_use": "When you need specific, actionable examples",
            "input_requirements": ["market_conditions", "strategy_type", "risk_profile"],
            "output_format": "Detailed examples with specific numbers and scenarios",
            "confidence_level": "High - provides concrete, actionable examples"
        },
        "example_based_risk": {
            "use_case": "Risk assessment with numerical examples and calculations",
            "when_to_use": "When you need to understand risk concepts through examples",
            "input_requirements": ["portfolio_data", "risk_scenarios", "timeframe"],
            "output_format": "Detailed risk examples with calculations and scenarios",
            "confidence_level": "High - provides concrete risk examples"
        }
    },
    "constraint_based_prompts": {
        "constraint_based_trading": {
            "use_case": "Trading within regulatory and operational constraints",
            "when_to_use": "When compliance and constraints are critical",
            "input_requirements": ["trading_objectives", "constraints", "market_data"],
            "output_format": "Constraint-compliant strategies with monitoring procedures",
            "confidence_level": "High - ensures compliance with all constraints"
        },
        "constraint_based_risk": {
            "use_case": "Risk management within regulatory and institutional limits",
            "when_to_use": "When operating under strict risk management constraints",
            "input_requirements": ["risk_constraints", "portfolio_data", "regulatory_requirements"],
            "output_format": "Constraint-optimized risk management solutions",
            "confidence_level": "High - ensures compliance with risk constraints"
        }
    }
}

def get_prompt_usage_guide() -> Dict[str, Any]:
    """
    Get comprehensive usage guide for all enhanced prompts
    
    Returns:
        Dictionary containing usage guidelines for each prompt category
    """
    return PROMPT_USAGE_GUIDE

def recommend_prompt_for_use_case(use_case: str, requirements: List[str]) -> str:
    """
    Recommend the best prompt for a specific use case
    
    Args:
        use_case: Description of what you want to accomplish
        requirements: List of specific requirements (e.g., ["quantitative", "uncertainty", "examples"])
    
    Returns:
        Recommended prompt name
    """
    recommendations = {
        "quantitative_analysis": "expert_strategist",
        "risk_management": "expert_risk_manager",
        "systematic_analysis": "step_by_step_analysis",
        "execution_control": "step_by_step_execution",
        "uncertainty_handling": "uncertainty_aware_analysis",
        "probabilistic_forecasting": "uncertainty_aware_prediction",
        "concrete_examples": "example_based_strategy",
        "risk_examples": "example_based_risk",
        "regulatory_compliance": "constraint_based_trading",
        "risk_constraints": "constraint_based_risk"
    }
    
    # Simple keyword matching for now
    for keyword, prompt in recommendations.items():
        if keyword in use_case.lower():
            return prompt
    
    # Default recommendation
    return "expert_strategist"

def validate_prompt_inputs(prompt_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate inputs for a specific prompt
    
    Args:
        prompt_name: Name of the prompt
        inputs: Input dictionary to validate
    
    Returns:
        Validation result with missing or invalid inputs
    """
    if prompt_name not in ENHANCED_TRADING_PROMPTS:
        return {"valid": False, "error": f"Prompt '{prompt_name}' not found"}
    
    prompt = ENHANCED_TRADING_PROMPTS[prompt_name]
    required_vars = prompt.input_variables
    
    missing_vars = [var for var in required_vars if var not in inputs]
    extra_vars = [var for var in inputs.keys() if var not in required_vars]
    
    return {
        "valid": len(missing_vars) == 0,
        "missing_variables": missing_vars,
        "extra_variables": extra_vars,
        "required_variables": required_vars
    }
