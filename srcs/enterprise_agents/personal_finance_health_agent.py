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
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)

# Configuration
OUTPUT_DIR = "personal_finance_reports"
COMPANY_NAME = "Personal Finance Health System"
TARGET_MARKET = "Korean Personal Finance Market"

app = MCPApp(
    name="personal_finance_health_system",
    settings=get_settings("configs/mcp_agent.config.yaml"),
    human_input_callback=None
)


async def main():
    """
    Personal Finance Health & Auto Investment Agent System
    
    Comprehensive personal finance management for Korean market:
    1. Personal finance health diagnosis with real-time insights
    2. Investment portfolio recommendations and optimization
    3. Korean market-specific financial analysis and policy adaptation
    4. Risk assessment and management strategies
    5. Actionable financial planning and budgeting advice
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
        
        # --- PERSONAL FINANCE HEALTH AGENTS ---
        
        # Financial Health Analyzer Agent
        financial_health_analyzer = Agent(
            name="financial_health_analyzer",
            instruction=f"""You are a personal financial health specialist providing comprehensive financial assessments for Korean individuals.
            
            Analyze personal financial health across multiple dimensions:
            
            1. Income and Cash Flow Analysis:
               - Monthly income stability and growth potential
               - Income diversification and sources analysis
               - Cash flow patterns and seasonal variations
               - Debt-to-income ratio assessment
               - Savings rate calculation and optimization
               - Emergency fund adequacy evaluation
            
            2. Expense Management and Budgeting:
               - Expense categorization and analysis (housing, food, transportation, etc.)
               - Korean household expense benchmarking
               - Unnecessary expense identification and reduction strategies
               - Budget optimization for different life stages
               - Lifestyle inflation assessment and control
               - Cost-cutting strategies without sacrificing quality of life
            
            3. Asset and Liability Assessment:
               - Current asset allocation and diversification
               - Asset quality and growth potential evaluation
               - Liability structure and optimization opportunities
               - Net worth calculation and trend analysis
               - Asset-liability matching and optimization
               - Liquidity position and emergency preparedness
            
            4. Financial Health Scoring:
               - Comprehensive financial health score (0-100)
               - Component scores (income, expenses, assets, debts, savings)
               - Benchmarking against Korean demographics
               - Improvement recommendations and prioritization
               - Progress tracking and milestone setting
               - Financial health trend analysis
            
            5. Korean Market Context:
               - Korean household financial behavior patterns
               - Cultural factors affecting financial decisions
               - Government financial policies and their impact
               - Korean financial institutions and services
               - Local economic conditions and their effects
               - Age and life stage considerations for Korean context
            
            Provide specific, actionable recommendations with Korean won amounts.
            Include risk warnings and realistic timelines for improvement.
            """,
            server_names=["filesystem", "g-search"],
        )
        
        # Investment Advisory Agent
        investment_advisor = Agent(
            name="investment_advisor",
            instruction=f"""You are an investment advisory specialist focusing on Korean personal investors.
            
            Provide personalized investment recommendations and portfolio management:
            
            1. Investment Strategy Development:
               - Risk tolerance assessment and profiling
               - Investment goals clarification and prioritization
               - Time horizon analysis and strategy matching
               - Investment style identification (conservative, moderate, aggressive)
               - Life stage-appropriate investment strategies
               - Goal-based investment planning (retirement, education, housing)
            
            2. Korean Market Investment Opportunities:
               - Korean stock market analysis (KOSPI, KOSDAQ)
               - Blue-chip Korean stocks for stable growth
               - Korean ETFs and index funds for diversification
               - Government bonds and corporate bonds
               - Real estate investment trusts (REITs)
               - Alternative investments suitable for Korean market
            
            3. Portfolio Construction and Optimization:
               - Asset allocation based on risk profile and goals
               - Diversification across asset classes and sectors
               - Korean vs international investment balance
               - Cost-effective investment vehicle selection
               - Tax-efficient portfolio structuring
               - Rebalancing strategies and implementation
            
            4. Investment Product Selection:
               - Korean mutual funds and ETFs comparison
               - Bank investment products evaluation
               - Securities company services and platforms
               - Robo-advisor services in Korean market
               - Direct stock investment guidance
               - Fixed-income product recommendations
            
            5. Performance Monitoring and Adjustment:
               - Portfolio performance tracking and reporting
               - Benchmark comparison and analysis
               - Risk-adjusted return evaluation
               - Market condition adaptations
               - Investment strategy refinements
               - Tax-loss harvesting opportunities
            
            Provide specific investment recommendations with Korean won amounts.
            Include expected returns, risks, and investment timelines.
            """,
            server_names=["g-search", "fetch"],
        )
        
        # Korean Market Specialist Agent
        korean_market_specialist = Agent(
            name="korean_market_specialist",
            instruction=f"""You are a Korean financial market specialist providing market insights and policy analysis.
            
            Analyze Korean financial markets and policy impacts:
            
            1. Korean Economic Environment:
               - Bank of Korea monetary policy and interest rate trends
               - Korean economic indicators and their market impact
               - Inflation trends and purchasing power analysis
               - Currency (KRW) volatility and international factors
               - Trade balance and current account impact
               - Geopolitical factors affecting Korean markets
            
            2. Korean Financial Market Analysis:
               - Stock market trends and sector rotation
               - Bond market conditions and yield curve analysis
               - Real estate market trends and regional variations
               - Cryptocurrency regulations and market impact
               - Foreign investment flows and their effects
               - Market volatility and risk assessment
            
            3. Government Policy Impact:
               - Tax policy changes and investment implications
               - Real estate regulations and market effects
               - Financial sector regulations and reforms
               - Retirement and pension system changes
               - Healthcare and social security policy impacts
               - Small business and entrepreneurship support policies
            
            4. Korean Financial Services Landscape:
               - Banking sector consolidation and competition
               - Fintech innovation and digital transformation
               - Insurance market trends and product innovation
               - Securities industry developments
               - Payment system evolution and digital currencies
               - Regulatory changes and compliance requirements
            
            5. Market Timing and Opportunities:
               - Market cycle analysis and positioning
               - Sector rotation opportunities
               - Seasonal patterns and calendar effects
               - Event-driven investment opportunities
               - Policy-driven market movements
               - Risk-on vs risk-off market environments
            
            Provide Korean market-specific insights with actionable implications.
            Include policy change impacts and strategic recommendations.
            """,
            server_names=["g-search", "fetch"],
        )
        
        # Risk Management Agent
        risk_manager = Agent(
            name="risk_manager",
            instruction=f"""You are a personal finance risk management specialist for Korean investors.
            
            Assess and manage financial risks comprehensively:
            
            1. Personal Financial Risk Assessment:
               - Income stability and job security risks
               - Health and disability insurance adequacy
               - Emergency fund sufficiency analysis
               - Debt service capacity and stress testing
               - Lifestyle inflation and spending risks
               - Life stage transition risks and planning
            
            2. Investment Risk Management:
               - Portfolio risk assessment and measurement
               - Concentration risk identification and mitigation
               - Market risk and volatility management
               - Liquidity risk assessment and planning
               - Currency risk for international investments
               - Inflation risk and purchasing power protection
            
            3. Korean Market-Specific Risks:
               - Geopolitical risks and North Korea factors
               - Regulatory changes and policy risks
               - Economic dependency on China and US
               - Demographic changes and aging population
               - Real estate market risks and bubbles
               - Technology and cyber security risks
            
            4. Insurance and Protection Planning:
               - Life insurance needs analysis and optimization
               - Health insurance gap analysis and supplements
               - Disability insurance adequacy assessment
               - Property and casualty insurance review
               - Long-term care insurance planning
               - Liability protection strategies
            
            5. Risk Mitigation Strategies:
               - Diversification across assets and geographies
               - Hedging strategies for major risks
               - Insurance optimization and cost management
               - Emergency planning and contingency funds
               - Risk monitoring and early warning systems
               - Stress testing and scenario planning
            
            Provide specific risk mitigation recommendations with cost-benefit analysis.
            Include insurance recommendations and emergency planning guidance.
            """,
            server_names=["g-search", "filesystem"],
        )
        
        # --- ORCHESTRATOR SETUP ---
        
        orchestrator = Orchestrator(
            llm_factory=OpenAIAugmentedLLM,
            available_agents=[
                financial_health_analyzer,
                investment_advisor,
                korean_market_specialist,
                risk_manager,
            ],
            plan_type="full",
        )
        
        # --- EVALUATOR SETUP ---
        
        evaluator = EvaluatorOptimizerLLM(
            llm_factory=OpenAIAugmentedLLM,
            context=context,
            evaluation_criteria=[
                "Financial health assessment accuracy and completeness",
                "Investment recommendations suitability and practicality",
                "Korean market context appropriateness and relevance",
                "Risk management comprehensiveness and effectiveness",
                "Actionable advice with specific KRW amounts",
                "Realistic timelines and achievable goals",
            ],
            target_quality=QualityRating.EXCELLENT,
        )
        
        # --- MAIN EXECUTION ---
        
        logger.info("🚀 Starting Personal Finance Health Analysis System")
        
        # Sample comprehensive user profile
        user_profile = {
            "personal_info": {
                "age": 32,
                "marital_status": "married",
                "dependents": 1,
                "occupation": "software_engineer",
                "location": "Seoul"
            },
            "financial_data": {
                "monthly_income": 6000000,  # 6M KRW
                "monthly_expenses": {
                    "housing": 1500000,     # rent/mortgage
                    "food": 800000,         # food and dining
                    "transportation": 300000, # public transport, car
                    "utilities": 200000,    # electricity, gas, internet
                    "insurance": 300000,    # health, life insurance
                    "entertainment": 400000, # hobbies, dining out
                    "education": 200000,    # books, courses
                    "miscellaneous": 300000  # other expenses
                },
                "current_assets": {
                    "savings_account": 50000000,  # 50M KRW
                    "checking_account": 5000000,   # 5M KRW
                    "stocks": 20000000,           # 20M KRW
                    "funds": 10000000,            # 10M KRW
                    "real_estate": 0,             # no real estate
                    "cryptocurrency": 2000000,    # 2M KRW
                    "pension": 15000000           # 15M KRW
                },
                "debts": {
                    "credit_card": 3000000,       # 3M KRW
                    "student_loan": 10000000,     # 10M KRW
                    "mortgage": 0,                # no mortgage
                    "other_loans": 0
                }
            },
            "investment_preferences": {
                "risk_tolerance": "moderate",
                "investment_horizon": "15-20 years",
                "investment_goals": [
                    "retirement_planning",
                    "children_education",
                    "home_purchase",
                    "wealth_building"
                ],
                "preferred_investments": [
                    "korean_stocks",
                    "korean_etfs", 
                    "savings_accounts",
                    "real_estate"
                ],
                "crypto_interest": "minimal"
            },
            "financial_goals": {
                "short_term": "Build emergency fund, pay off credit card debt",
                "medium_term": "Save for home down payment, invest for growth",
                "long_term": "Retirement planning, children's education funding"
            }
        }
        
        # Comprehensive analysis prompt
        analysis_prompt = f"""
        Conduct a comprehensive personal finance health analysis for a Korean individual with the following detailed profile:
        
        {json.dumps(user_profile, indent=2, ensure_ascii=False)}
        
        **Required Analysis Components:**
        
        1. **Financial Health Assessment**
           - Overall financial health score (0-100) with component breakdown
           - Income stability and growth potential analysis
           - Expense efficiency and budget optimization recommendations
           - Asset allocation quality and diversification assessment
           - Debt management and payoff strategies
           - Emergency fund adequacy and recommendations
        
        2. **Investment Strategy and Recommendations**
           - Personalized investment strategy based on risk profile and goals
           - Specific Korean market investment opportunities
           - Portfolio construction with exact allocation percentages
           - Investment product recommendations with Korean won amounts
           - Expected returns and risk assessments
           - Implementation timeline and priority ordering
        
        3. **Korean Market Context and Opportunities**
           - Current Korean market conditions and outlook
           - Government policy impacts on personal finance
           - Korean-specific investment opportunities and risks
           - Local market trends and their implications
           - Currency and inflation considerations
           - Regulatory environment and tax implications
        
        4. **Risk Management and Protection**
           - Personal financial risk assessment and mitigation
           - Insurance needs analysis and recommendations
           - Market risk management for investment portfolio
           - Emergency planning and contingency strategies
           - Korean market-specific risk factors
           - Long-term financial security planning
        
        5. **Action Plan and Implementation**
           - Immediate action items with specific timelines
           - Medium-term strategic initiatives
           - Long-term wealth building strategies
           - Monthly and quarterly review schedules
           - Progress tracking and adjustment mechanisms
           - Milestone celebrations and motivation strategies
        
        **Output Requirements:**
        - Provide specific numerical recommendations (amounts in KRW)
        - Include realistic timelines and achievable milestones
        - Explain rationale behind each recommendation
        - Address potential obstacles and solutions
        - Create clear, actionable next steps
        - Format as a comprehensive financial plan
        
        Save the complete analysis as a detailed personal finance report.
        """
        
        try:
            # Execute comprehensive analysis
            logger.info("🔍 Starting comprehensive personal finance analysis...")
            
            result = await orchestrator.run(
                task=analysis_prompt,
                additional_context={
                    "user_profile": user_profile,
                    "analysis_date": datetime.now().isoformat(),
                    "market_focus": "Korean personal finance market",
                    "currency": "Korean Won (KRW)",
                    "cultural_context": "Korean financial culture and practices"
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
            report_filename = f"{OUTPUT_DIR}/personal_finance_health_report_{timestamp}.md"
            
            # Calculate key financial metrics for summary
            monthly_income = user_profile["financial_data"]["monthly_income"]
            total_expenses = sum(user_profile["financial_data"]["monthly_expenses"].values())
            monthly_surplus = monthly_income - total_expenses
            savings_rate = (monthly_surplus / monthly_income) * 100
            total_assets = sum(user_profile["financial_data"]["current_assets"].values())
            total_debts = sum(user_profile["financial_data"]["debts"].values())
            net_worth = total_assets - total_debts
            
            report_content = f"""# Personal Finance Health Analysis Report
            
## Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## Generated by: {COMPANY_NAME}
## Target Market: {TARGET_MARKET}

---

## Executive Summary

This comprehensive personal finance analysis provides tailored recommendations for a {user_profile["personal_info"]["age"]}-year-old Korean individual based on current financial situation and goals.

### Key Financial Metrics
- **Monthly Income**: ₩{monthly_income:,}
- **Monthly Expenses**: ₩{total_expenses:,}
- **Monthly Surplus**: ₩{monthly_surplus:,}
- **Savings Rate**: {savings_rate:.1f}%
- **Total Assets**: ₩{total_assets:,}
- **Total Debts**: ₩{total_debts:,}
- **Net Worth**: ₩{net_worth:,}

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

### Immediate Actions (This Month)
- [ ] Review and optimize monthly budget
- [ ] Set up automatic savings transfers
- [ ] Research recommended investment products
- [ ] Update insurance coverage as needed
- [ ] Create emergency fund target

### Short-term Goals (3-6 Months)
- [ ] Implement investment strategy
- [ ] Optimize debt repayment plan
- [ ] Establish regular portfolio reviews
- [ ] Track progress against financial goals
- [ ] Build emergency fund to target level

### Medium-term Goals (1-2 Years)
- [ ] Achieve target asset allocation
- [ ] Evaluate and adjust investment strategy
- [ ] Consider real estate investment options
- [ ] Plan for major life changes
- [ ] Optimize tax efficiency

### Long-term Goals (5+ Years)
- [ ] Achieve financial independence milestones
- [ ] Maximize retirement savings
- [ ] Evaluate estate planning needs
- [ ] Consider advanced investment strategies
- [ ] Plan for children's education funding

---

## Next Review Schedule

- **Monthly**: Budget review and expense tracking
- **Quarterly**: Investment portfolio performance review
- **Annually**: Comprehensive financial health assessment
- **As needed**: Major life change adaptations

---

*Report generated by Personal Finance Health System*
*Next comprehensive review recommended: {(datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')}*
"""
            
            # Save to filesystem
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"✅ Analysis complete! Report saved to: {report_filename}")
            logger.info(f"📈 Quality Score: {evaluation.final_quality.value}/5")
            
            # Print summary for user
            print(f"\n🎯 Personal Finance Health Analysis Complete!")
            print(f"📁 Report Location: {report_filename}")
            print(f"⭐ Quality Rating: {evaluation.final_quality.value}/5")
            print(f"🔄 Analysis Iterations: {evaluation.iterations_completed}")
            print(f"💰 Monthly Surplus: ₩{monthly_surplus:,}")
            print(f"📊 Savings Rate: {savings_rate:.1f}%")
            print(f"🏦 Net Worth: ₩{net_worth:,}")
            
            return {
                "status": "success",
                "report_file": report_filename,
                "quality_score": evaluation.final_quality.value,
                "financial_metrics": {
                    "monthly_surplus": monthly_surplus,
                    "savings_rate": savings_rate,
                    "net_worth": net_worth,
                    "total_assets": total_assets,
                    "total_debts": total_debts
                },
                "user_profile": user_profile,
                "analysis_timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            print(f"❌ Error during analysis: {str(e)}")
            raise


if __name__ == "__main__":
    print("🚀 Starting Personal Finance Health System...")
    print("💰 Comprehensive Korean Personal Finance Analysis")
    print("📊 Providing personalized financial health assessment and recommendations")
    print("-" * 70)
    
    try:
        result = asyncio.run(main())
        print(f"\n✅ System execution completed successfully!")
        print(f"📈 Analysis result summary: {result}")
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {str(e)}")
        raise
