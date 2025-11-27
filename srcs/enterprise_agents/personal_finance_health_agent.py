import asyncio
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)
from srcs.common.utils import setup_agent_app, save_report


# Configuration
OUTPUT_DIR = "personal_finance_reports"
COMPANY_NAME = "Personal Finance Health System"
TARGET_MARKET = "Korean Personal Finance Market"

class PersonalFinanceAgent:
    """
    Personal Finance Health & Auto Investment Agent System
    """
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        self.app = setup_agent_app("personal_finance_health_system")

    async def run_analysis(self, user_data: dict) -> dict:
        """
        Runs the full personal finance analysis based on user data.
        
        Args:
            user_data: A dictionary containing the user's financial information.
        
        Returns:
            A dictionary containing the analysis results.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        async with self.app.run() as finance_app:
            context = finance_app.context
            logger = finance_app.logger

            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
                logger.info("Filesystem server configured")
            
            agents = self._create_finance_agents()

            orchestrator = Orchestrator(
                llm_factory=GoogleAugmentedLLM,
                available_agents=list(agents.values()),
                plan_type="full",
            )
            
            task = self._create_task(user_data, timestamp)

            logger.info(f"Starting personal finance analysis for user.")
            result_str = await orchestrator.generate_str(
                message=task,
                request_params=RequestParams(model="gemini-2.5-flash-lite")
            )
            logger.info("Personal finance analysis completed successfully.")
            
            try:
                # Assuming the result is a JSON string
                result_data = json.loads(result_str)
                return result_data
            except json.JSONDecodeError:
                # If not JSON, return the raw string in a structured dict
                return {"analysis_report_text": result_str}

    def _create_task(self, user_data: dict, timestamp: str) -> str:
        """Creates the main task for the orchestrator."""
        user_data_str = json.dumps(user_data, indent=2, ensure_ascii=False)
        return f"""
        **Objective: Comprehensive Personal Financial Health Analysis and Action Plan**

        **User Financial Profile:**
        ```json
        {user_data_str}
        ```

        **Execution Plan:**
        1.  **Financial Health Diagnosis (financial_health_analyzer):**
            -   Analyze the user's income, expenses, assets, and liabilities.
            -   Calculate key financial ratios (DTI, savings rate).
            -   Provide a financial health score and a detailed diagnosis report.

        2.  **Investment Portfolio Analysis (investment_advisor):**
            -   Evaluate the current investment portfolio.
            -   Recommend an optimal asset allocation strategy based on the user's risk tolerance.
            -   Suggest specific Korean market investment products (stocks, ETFs, funds).

        3.  **Risk Assessment (risk_manager):**
            -   Identify potential financial risks (job loss, health issues).
            -   Recommend appropriate insurance coverage.
            -   Propose strategies to build an emergency fund.

        4.  **Market and Policy Context (korean_market_specialist):**
            -   Analyze how current Korean market trends and government policies affect the user's financial situation.
            -   Provide insights on real estate, taxes, and interest rates.

        5.  **Synthesize Final Report (financial_planner):**
            -   Combine all analyses into a single, integrated report.
            -   Create a concrete, actionable 3-month financial action plan.
            -   The final output should be a single JSON object containing 'diagnosis', 'investment_plan', 'risk_management', and 'action_plan' keys.
        """

    def _create_finance_agents(self) -> dict:
        """Creates and returns a dictionary of all finance agents."""
        # This function contains the agent definitions from the original main()
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
            server_names=[],  # MCP 서버 validation 에러 방지
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
            server_names=[],  # MCP 서버 validation 에러 방지
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
        
        # Final Report Synthesizer Agent
        financial_planner = Agent(
            name="financial_planner",
            instruction="""You are a master financial planner. Synthesize the inputs from all other agents into a single, cohesive, and actionable financial report. The final output must be a single JSON object.""",
            server_names=[],
        )
        
        return {
            "financial_health_analyzer": financial_health_analyzer,
            "investment_advisor": investment_advisor,
            "korean_market_specialist": korean_market_specialist,
            "risk_manager": risk_manager,
            "financial_planner": financial_planner,
        }

async def main():
    """CLI runner for demonstration."""
    print("🚀 Personal Finance Health Agent CLI Demo")
    
    # Dummy user data for CLI execution
    dummy_user_data = {
        "income_monthly": 5000000,
        "expenses_monthly": {
            "housing": 1500000,
            "food": 800000,
            "transport": 300000,
            "entertainment": 400000,
            "other": 500000
        },
        "assets": {
            "cash": 20000000,
            "stocks": 50000000,
            "real_estate": 300000000
        },
        "liabilities": {
            "mortgage": 200000000,
            "car_loan": 10000000
        },
        "risk_tolerance": "Medium"
    }
    
    agent = PersonalFinanceAgent()
    result = await agent.run_analysis(dummy_user_data)
    
    print("\n--- ANALYSIS COMPLETE ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
