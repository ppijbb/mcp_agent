"""
Dynamic Personal Finance Health Agent - Enterprise Edition
엔터프라이즈급 동적 개인 금융 건강 에이전트

Clean, modular architecture with separated concerns:
- Models: Pydantic data structures
- Providers: Data source implementations  
- Agent: Core business logic

Features:
- Dynamic product discovery
- Real-time market data integration
- Multi-provider data aggregation
- Enterprise-grade scalability and reliability
"""

import asyncio
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import get_settings
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)

# Configuration
OUTPUT_DIR = "dynamic_finance_reports"
COMPANY_NAME = "Dynamic Finance Advisor"
TARGET_MARKET = "Korean Personal Finance Market"

app = MCPApp(
    name="dynamic_finance_health_system",
    settings=get_settings("configs/mcp_agent.config.yaml"),
    human_input_callback=None
)


async def main():
    """
    Dynamic Personal Finance Health Agent System
    
    Provides comprehensive dynamic personal finance management:
    1. Real-time market data analysis and insights
    2. Dynamic product discovery across multiple categories
    3. Personalized portfolio optimization
    4. Risk assessment and management
    5. Korean market-specific financial advice
    6. Multi-provider data aggregation and analysis
    """
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    async with app.run() as finance_app:
        context = finance_app.context
        logger = finance_app.logger
        
        # Configure servers
        if "filesystem" in context.config.mcp.servers:
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            logger.info("Filesystem server configured")
        
        # --- DYNAMIC FINANCE HEALTH AGENTS ---
        
        # Real-time Market Data Analyzer
        market_analyzer = Agent(
            name="korean_market_analyzer",
            instruction=f"""You are a Korean financial market specialist providing real-time market analysis for {COMPANY_NAME}.
            
            Analyze Korean financial markets across multiple asset classes:
            
            1. Korean Stock Market Analysis:
               - KOSPI and KOSDAQ index movements and trends
               - Top performing and declining stocks (005930.KS Samsung, 000660.KS SK Hynix, etc.)
               - Sector analysis (technology, finance, manufacturing)
               - Foreign investor flow analysis
               - Market sentiment and volatility assessment
               - Economic indicators impact on stock prices
            
            2. Korean Bond and Interest Rate Analysis:
               - Bank of Korea base rate trends and outlook
               - Government bond yields (3-year, 10-year)
               - Corporate bond spreads and credit risk
               - Inflation expectations and real interest rates
               - Currency impact on bond markets
               - Fixed deposit and savings account rates comparison
            
            3. Currency and FX Analysis:
               - USD/KRW exchange rate trends and forecasts
               - Impact of US Federal Reserve policies
               - Korean current account and trade balance effects
               - Safe haven flows and market volatility
               - FX hedging strategies for Korean investors
               - Impact on import/export companies
            
            4. Real Estate Market Insights:
               - Seoul and major cities apartment price trends
               - Government housing policies and regulations
               - Mortgage rates and lending conditions
               - REITs and real estate investment options
               - Regional market variations and opportunities
               - Rental yield analysis and investment viability
            
            5. Cryptocurrency Market Analysis:
               - Korean crypto exchange (Upbit, Bithumb) data
               - Bitcoin, Ethereum, and major altcoin trends
               - Korean crypto regulations and policy impact
               - Institutional adoption in Korea
               - Risk assessment for crypto investments
               - Correlation with traditional asset classes
            
            Provide actionable insights with specific recommendations for Korean investors.
            Include risk warnings and market outlook with confidence levels.
            """,
            server_names=["fetch", "g-search"],
        )
        
        # Dynamic Product Discovery Agent
        product_discovery = Agent(
            name="dynamic_product_discovery",
            instruction=f"""You are a financial product discovery specialist for Korean market serving {COMPANY_NAME}.
            
            Discover and analyze suitable financial products across categories:
            
            1. Korean Stock Products:
               - Individual stocks analysis (Samsung, LG, SK, Hyundai group companies)
               - Korean ETFs (KODEX, TIGER, ARIRANG series)
               - Sector-specific funds and thematic investments
               - ESG and sustainability-focused products
               - Dividend-focused stocks and funds for income generation
               - Growth vs value investment opportunities
            
            2. Fixed Income and Savings Products:
               - High-yield savings accounts from major banks (KB, Shinhan, Hana)
               - Time deposits and CDs with competitive rates
               - Corporate bonds from reliable Korean companies
               - Government bonds (KTBs) and municipal bonds
               - Structured deposits with principal protection
               - P2P lending platforms and alternative fixed income
            
            3. Real Estate Investment Products:
               - Korean REITs listed on KRX
               - Real estate funds and property investment trusts
               - Crowdfunding real estate platforms
               - Direct property investment opportunities
               - Real estate development projects
               - Commercial vs residential property analysis
            
            4. Cryptocurrency and Digital Assets:
               - Major cryptocurrencies available on Korean exchanges
               - Korean crypto funds and investment products
               - Staking and DeFi opportunities in Korean market
               - NFT and digital asset investment options
               - Blockchain technology company stocks
               - Regulatory-compliant crypto investment vehicles
            
            5. Alternative Investment Products:
               - Commodity ETFs and precious metals
               - Foreign market access through Korean brokers
               - Private equity and venture capital funds
               - Art, wine, and collectibles investment platforms
               - Peer-to-peer lending and crowdfunding
               - Insurance-linked investment products
            
            For each product category, analyze:
            - Expected returns and historical performance
            - Risk levels and volatility measures
            - Liquidity and accessibility for retail investors
            - Minimum investment amounts and fees
            - Tax implications and optimization strategies
            - Suitability for different investor profiles
            
            Provide specific product recommendations with reasoning and risk assessment.
            """,
            server_names=["g-search", "fetch"],
        )
        
        # Portfolio Optimization Agent
        portfolio_optimizer = Agent(
            name="portfolio_optimization_specialist",
            instruction=f"""You are a portfolio optimization expert creating personalized investment strategies for {COMPANY_NAME}.
            
            Create optimized portfolios based on user profiles and market conditions:
            
            1. Risk-Based Portfolio Construction:
               - Conservative portfolios (60% bonds, 30% stocks, 10% alternatives)
               - Moderate portfolios (40% bonds, 50% stocks, 10% alternatives)
               - Aggressive portfolios (20% bonds, 70% stocks, 10% alternatives)
               - Age-based allocation adjustments (100 minus age rule variations)
               - Risk tolerance assessment and portfolio matching
               - Volatility targeting and risk budgeting
            
            2. Korean Market-Specific Allocation:
               - Domestic vs international diversification ratios
               - Korean large-cap vs small-cap allocation
               - KOSPI vs KOSDAQ exposure optimization
               - Currency hedging strategies for foreign investments
               - Sector rotation based on Korean economic cycles
               - Government policy impact on asset allocation
            
            3. Dynamic Rebalancing Strategies:
               - Threshold-based rebalancing (5%, 10%, 20% rules)
               - Time-based rebalancing (monthly, quarterly, annually)
               - Volatility-based rebalancing during market stress
               - Tax-efficient rebalancing strategies
               - Transaction cost optimization
               - Dollar-cost averaging implementation
            
            4. Goal-Based Investment Planning:
               - Retirement planning and pension optimization
               - Education funding and children's future planning
               - Home purchase and real estate investment goals
               - Emergency fund optimization and liquidity management
               - Wealth preservation and estate planning
               - Short-term vs long-term goal prioritization
            
            5. Performance Monitoring and Optimization:
               - Benchmark comparison and alpha generation
               - Risk-adjusted return metrics (Sharpe, Sortino ratios)
               - Drawdown analysis and recovery strategies
               - Tax efficiency and after-tax return optimization
               - Fee minimization and cost-effective investing
               - ESG integration and sustainable investing options
            
            6. Market Condition Adaptations:
               - Bull market strategies and profit-taking rules
               - Bear market protection and defensive positioning
               - High inflation environment adaptations
               - Interest rate cycle positioning
               - Currency volatility hedging strategies
               - Geopolitical risk management
            
            Provide specific portfolio allocations with target percentages.
            Include rebalancing triggers and performance monitoring guidelines.
            Explain the rationale behind each allocation decision.
            """,
            server_names=["filesystem", "g-search"],
        )
        
        # Risk Assessment and Management Agent
        risk_assessment = Agent(
            name="comprehensive_risk_assessor",
            instruction=f"""You are a financial risk management specialist providing comprehensive risk analysis for {COMPANY_NAME}.
            
            Assess and manage various types of investment risks:
            
            1. Market Risk Assessment:
               - Systematic risk from market-wide factors
               - Specific risk from individual investments
               - Volatility analysis and VaR calculations
               - Correlation analysis between asset classes
               - Beta analysis and market sensitivity
               - Tail risk and extreme event scenarios
            
            2. Korean Market-Specific Risks:
               - Geopolitical tensions and North Korea risk
               - China economic dependency and trade risks
               - Chaebol concentration and corporate governance
               - Currency risk from USD/KRW volatility
               - Regulatory changes and government policy shifts
               - Demographic changes and aging population impact
            
            3. Liquidity Risk Management:
               - Asset liquidity assessment and ranking
               - Market depth analysis for Korean securities
               - Funding liquidity and cash flow management
               - Emergency liquidity planning and stress testing
               - Illiquid investment position sizing
               - Liquidity premium analysis and compensation
            
            4. Credit Risk Evaluation:
               - Corporate bond credit ratings and spreads
               - Bank deposit insurance coverage analysis
               - Counterparty risk in financial transactions
               - P2P lending and alternative credit risks
               - Real estate investment credit risks
               - Government and sovereign credit assessment
            
            5. Operational and Implementation Risks:
               - Broker and platform reliability assessment
               - Technology and cyber security risks
               - Execution risk and slippage analysis
               - Custody and safekeeping risks
               - Tax compliance and regulatory risks
               - Human error and behavioral biases
            
            6. Risk Mitigation Strategies:
               - Diversification across assets, sectors, and geographies
               - Hedging strategies using derivatives
               - Insurance products for investment protection
               - Stop-loss and position sizing rules
               - Regular stress testing and scenario analysis
               - Dynamic risk adjustment based on market conditions
            
            For each risk category, provide:
            - Risk identification and measurement methods
            - Impact assessment and probability estimates
            - Specific mitigation strategies and recommendations
            - Monitoring indicators and early warning signals
            - Contingency plans for adverse scenarios
            - Cost-benefit analysis of risk management measures
            
            Ensure all risk assessments include Korean regulatory and market context.
            """,
            server_names=["g-search", "fetch"],
        )
        
        # --- ORCHESTRATOR SETUP ---
        
        orchestrator = Orchestrator(
            llm_factory=GoogleAugmentedLLM,
            available_agents=[
                market_analyzer,
                product_discovery, 
                portfolio_optimizer,
                risk_assessment,
            ],
            plan_type="full",
        )
        
        # --- EVALUATOR SETUP ---
        
        evaluator = EvaluatorOptimizerLLM(
            llm_factory=GoogleAugmentedLLM,
            context=context,
            evaluation_criteria=[
                "Market analysis accuracy and timeliness",
                "Product recommendation relevance and suitability", 
                "Portfolio optimization effectiveness",
                "Risk assessment comprehensiveness",
                "Korean market context appropriateness",
                "Actionable insights and practical recommendations",
            ],
            target_quality=QualityRating.EXCELLENT,
        )
        
        # --- MAIN EXECUTION ---
        
        logger.info("🚀 Starting Dynamic Finance Health Analysis System")
        
        # Sample user profile for analysis
        user_profile = {
            "age": 35,
            "monthly_income": 5000000,  # 5M KRW
            "monthly_expenses": 3000000,  # 3M KRW  
            "current_savings": 50000000,  # 50M KRW
            "risk_tolerance": "moderate",
            "investment_goals": ["retirement", "real_estate_purchase"],
            "investment_horizon": "10-15 years",
            "preferred_categories": ["stocks", "real_estate", "savings"],
            "crypto_interest": "low",
            "current_investments": {
                "savings_account": 30000000,
                "stocks": 15000000, 
                "real_estate": 0,
                "crypto": 0
            }
        }
        
        # Main analysis prompt
        analysis_prompt = f"""
        Conduct a comprehensive dynamic personal finance health analysis for a Korean investor with the following profile:
        
        {json.dumps(user_profile, indent=2, ensure_ascii=False)}
        
        **Required Analysis Components:**
        
        1. **Current Market Environment Assessment**
           - Korean market conditions and outlook
           - Interest rate environment and trends
           - Currency and inflation considerations
           - Sector opportunities and risks
           
        2. **Personalized Product Discovery**
           - Suitable investment products for this profile
           - Expected returns and risk levels
           - Korean market-specific opportunities
           - Product comparison and recommendations
        
        3. **Optimized Portfolio Construction**  
           - Asset allocation recommendations
           - Specific product selections with rationale
           - Rebalancing strategy and guidelines
           - Performance targets and benchmarks
           
        4. **Comprehensive Risk Assessment**
           - Risk factors specific to this portfolio
           - Mitigation strategies and hedging options
           - Stress testing and scenario analysis
           - Monitoring and adjustment guidelines
           
        5. **Implementation Roadmap**
           - Step-by-step investment implementation plan
           - Timeline and priority ordering
           - Cost optimization and tax efficiency
           - Ongoing monitoring and review schedule
        
        **Output Requirements:**
        - Provide specific numerical recommendations (allocation percentages, amounts)
        - Include Korean won amounts and local market context
        - Explain rationale behind each recommendation
        - Address potential risks and mitigation strategies
        - Create actionable next steps for the investor
        
        Save the complete analysis as a detailed report in the filesystem.
        """
        
        try:
            # Execute comprehensive analysis
            logger.info("🔍 Starting comprehensive finance analysis...")
            
            result = await orchestrator.run(
                task=analysis_prompt,
                additional_context={
                    "user_profile": user_profile,
                    "analysis_date": datetime.now().isoformat(),
                    "market_focus": "Korean financial markets",
                    "currency": "Korean Won (KRW)"
                }
            )
            
            # Evaluate result quality
            logger.info("📊 Evaluating analysis quality...")
            
            evaluation = await evaluator.evaluate_and_optimize(
                task=analysis_prompt,
                result=result,
                max_iterations=2
            )
            
            # Save comprehensive report
            report_filename = f"{OUTPUT_DIR}/dynamic_finance_analysis_{timestamp}.md"
            
            report_content = f"""# Dynamic Personal Finance Health Analysis Report
            
## Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## Generated by: {COMPANY_NAME}
## Target Market: {TARGET_MARKET}

---

## Executive Summary

This comprehensive analysis provides personalized financial recommendations for a Korean investor based on current market conditions and individual financial profile.

### User Profile Summary
```json
{json.dumps(user_profile, indent=2, ensure_ascii=False)}
```

---

## Detailed Analysis Results

{evaluation.optimized_result if evaluation.optimized_result else result}

---

## Quality Assessment

**Evaluation Score:** {evaluation.final_quality.value}/5
**Iterations Completed:** {evaluation.iterations_completed}

### Evaluation Details:
{evaluation.evaluation_history[-1].feedback if evaluation.evaluation_history else "No detailed feedback available"}

---

## Implementation Checklist

- [ ] Review current market conditions
- [ ] Assess risk tolerance alignment  
- [ ] Research recommended products
- [ ] Plan implementation timeline
- [ ] Set up monitoring schedule
- [ ] Review and adjust quarterly

---

*Report generated by Dynamic Finance Health System*
*Next review recommended: {(datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')}*
"""
            
            # Save to filesystem
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"✅ Analysis complete! Report saved to: {report_filename}")
            logger.info(f"📈 Quality Score: {evaluation.final_quality.value}/5")
            
            # Print summary for user
            print(f"\n🎯 Dynamic Finance Health Analysis Complete!")
            print(f"📁 Report Location: {report_filename}")
            print(f"⭐ Quality Rating: {evaluation.final_quality.value}/5")
            print(f"🔄 Analysis Iterations: {evaluation.iterations_completed}")
            
            return {
                "status": "success",
                "report_file": report_filename,
                "quality_score": evaluation.final_quality.value,
                "user_profile": user_profile,
                "analysis_timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            print(f"❌ Error during analysis: {str(e)}")
            raise


if __name__ == "__main__":
    print("🚀 Starting Dynamic Finance Health Agent System...")
    print("💰 Specializing in Korean Personal Finance Market")
    print("📊 Providing comprehensive financial analysis and recommendations")
    print("-" * 60)
    
    try:
        result = asyncio.run(main())
        print(f"\n✅ System execution completed successfully!")
        print(f"📈 Analysis result: {result}")
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {str(e)}")
        raise 