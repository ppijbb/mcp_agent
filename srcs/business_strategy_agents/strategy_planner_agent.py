"""
Strategy Planner Agent - Real MCPAgent Implementation
----------------------------------------------------
Converted from fake BaseAgent to real MCPAgent using mcp_agent library.
Creates comprehensive business strategies based on data and trends.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import get_settings
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
import aiohttp
import json

# Helper function to create the HTTP client session
def get_http_session():
    return aiohttp.ClientSession()

class StrategyPlannerMCPAgent:
    """Real MCPAgent for Business Strategy Planning"""
    
    def __init__(self,
                 google_drive_mcp_url: str = "http://localhost:3001",
                 data_sourcing_mcp_url: str = "http://localhost:3005"):
        self.google_drive_mcp_url = google_drive_mcp_url
        self.data_sourcing_mcp_url = data_sourcing_mcp_url
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        return Agent(
            name="strategy_planner",
            instruction="You are a strategic business planner. Analyze the provided market and business data to generate a comprehensive strategy.",
            server_names=["data_sourcing_mcp", "fetch"] # Replaced g-search with the new data_sourcing_mcp
        )

    async def analyze_market_and_business(self, industry: str, company_info: str, competitors: List[str]) -> Dict[str, Any]:
        """
        Analyzes market trends and competitor financials using the Data Sourcing MCP
        to generate a strategic business plan.
        """
        llm = OpenAIAugmentedLLM()
        
        # This prompt is now much more powerful as it directs the agent to use structured data endpoints.
        prompt = f"""
        As a Strategy Planner, your task is to create a detailed business strategy report.
        Use the 'data_sourcing_mcp' to get reliable data for your analysis.

        **Inputs:**
        - **Industry:** {industry}
        - **Company Information:** {company_info}
        - **Competitors (Tickers):** {', '.join(competitors)}

        **Analysis Steps:**
        1.  **Market Trend Analysis:**
            - Call the `/market-trends` endpoint with the industry "{industry}".
            - Analyze the key trends, growth potential, and timeframes.
        
        2.  **Competitor Financial Analysis:**
            - For each competitor ticker in [{', '.join(competitors)}], call the `/company-financials` endpoint.
            - Summarize each competitor's financial health (market cap, revenue) and strategic summary.

        3.  **Strategic Plan Formulation:**
            - Based on the market trends and competitor analysis, formulate a strategic plan for the company.

        **Output Format (JSON):**
        Please structure your response in the following JSON format:

        {{
          "market_analysis": {{
            "industry": "{industry}",
            "key_trends": [
              {{ "topic": "...", "growth": "...", "implication": "..." }}
            ],
            "market_summary": "..."
          }},
          "competitor_analysis": [
            {{
              "ticker": "...",
              "company_name": "...",
              "financial_summary": {{...}},
              "strategic_position": "..."
            }}
          ],
          "strategic_recommendations": {{
            "positioning_statement": "For [target customers], [company name] is the [category] that [key benefit]...",
            "strategic_goals": [
              "1. Goal A (e.g., Achieve 20% market share in AI-driven automation).",
              "2. Goal B (e.g., Launch a vertical SaaS solution for the healthcare sector)."
            ],
            "key_initiatives": [
              "1. Initiative X (e.g., R&D investment in Generative AI features).",
              "2. Initiative Y (e.g., Strategic partnerships with healthcare providers)."
            ],
            "risk_analysis": {{
              "market_risks": ["..."],
              "competitive_risks": ["..."]
            }}
          }}
        }}
        """
        
        # The underlying MCP Agent framework will handle the tool calls based on the prompt.
        # We just need to make a single generate call.
        response_str = await llm.generate_str(
            message=prompt,
            agent=self.agent, # Pass the agent to enable tool use
            request_params=RequestParams(
                model="gemini-2.5-pro-vision-preview-06-07",
                temperature=0.2,
                response_format={"type": "json_object"}
            )
        )
        
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from LLM response:\n{response_str}")
            return {"error": "Invalid JSON response from strategy planner"}

    async def run_strategy_planning(self, business_context: Dict[str, Any], objectives: List[str]) -> Dict[str, Any]:
        """Run comprehensive business strategy planning and upload the result to Google Drive."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_name = f"business_strategy_plan_{timestamp}.md"
        
        async with self.app.run() as planner_app:
            context = planner_app.context
            logger = planner_app.logger
            
            # Configure MCP servers
            await self._configure_mcp_servers(context, logger)
            
            # Define specialized agents
            agents = await self._create_strategy_agents(business_context, objectives)
            
            # Create orchestrator
            orchestrator = Orchestrator(
                llm_factory=OpenAIAugmentedLLM,
                available_agents=agents,
                plan_type="full"
            )
            
            # Execute strategy planning task
            task = await self._create_strategy_planning_task(business_context, objectives)
            
            logger.info(f"Starting strategy planning for objectives: {objectives}")
            
            try:
                report_content = await orchestrator.generate_str(
                    message=task,
                    request_params=RequestParams(model="gemini-2.5-flash-lite-preview-06-07")
                )

                # Upload the final report to Google Drive
                upload_url = f"{self.google_drive_mcp_url}/upload"
                payload = {"fileName": output_file_name, "content": report_content}

                async with get_http_session() as session:
                    async with session.post(upload_url, json=payload) as response:
                        response.raise_for_status()
                        upload_result = await response.json()

                if not upload_result.get("success"):
                    raise Exception(f"MCP upload failed: {upload_result.get('message')}")

                logger.info(f"Successfully uploaded report to Google Drive. File ID: {upload_result.get('fileId')}")

                return {
                    "success": True,
                    "output_file": upload_result.get("fileId"), # Returning File ID
                    "file_url": f"https://docs.google.com/document/d/{upload_result.get('fileId')}",
                    "business_context": business_context,
                    "objectives": objectives,
                    "timestamp": timestamp,
                    "result": report_content[:200] + "..." # Snippet of the result
                }
                
            except Exception as e:
                logger.error(f"Strategy planning failed: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e),
                    "objectives": objectives,
                    "timestamp": timestamp
                }
    
    async def _configure_mcp_servers(self, context, logger):
        """Configure required MCP servers"""
        
        # Configure filesystem server - no longer needed for output, but might be used by tools
        if "filesystem" in context.config.mcp.servers:
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            logger.info("Filesystem server configured for tool access.")
        
        # Check for required servers
        required_servers = ["g-search", "fetch"]
        missing_servers = []
        
        for server in required_servers:
            if server not in context.config.mcp.servers:
                missing_servers.append(server)
        
        if missing_servers:
            logger.warning(f"Missing MCP servers: {missing_servers}")
    
    async def _create_strategy_agents(self, business_context: Dict[str, Any], objectives: List[str]) -> List[Agent]:
        """Create specialized strategy planning agents"""
        
        context_str = str(business_context)
        objectives_str = ", ".join(objectives)
        
        # Market Research Strategist
        market_researcher = Agent(
            name="market_research_strategist",
            instruction=f"""You are a strategic market research expert.
            
            Business Context: {context_str}
            Strategic Objectives: {objectives_str}
            
            Conduct comprehensive market research to inform strategy:
            
            1. Market Analysis:
               - Market size, growth rates, and segmentation
               - Competitive landscape and positioning
               - Customer needs and behavior patterns
               - Pricing strategies and value propositions
            
            2. Industry Dynamics:
               - Industry structure and key players
               - Value chain analysis and profit pools
               - Regulatory environment and compliance
               - Technology trends and disruption risks
            
            3. Opportunity Assessment:
               - Market entry and expansion opportunities
               - Partnership and acquisition targets
               - Blue ocean and whitespace identification
               - First-mover advantages and timing
            
            Research deliverables:
            - Detailed market sizing and forecasts
            - Competitive intelligence and SWOT analysis
            - Customer journey mapping and pain points
            - Strategic opportunity rankings with rationale
            
            Focus on actionable insights that directly support strategic decision-making.
            Provide credible sources and data validation.""",
            server_names=["g-search", "fetch"]
        )
        
        # Financial Strategy Analyst
        financial_analyst = Agent(
            name="financial_strategy_analyst",
            instruction=f"""You are a strategic financial planning expert.
            
            Business Context: {context_str}
            Strategic Objectives: {objectives_str}
            
            Develop comprehensive financial strategy framework:
            
            1. Financial Analysis:
               - Revenue model optimization and diversification
               - Cost structure analysis and efficiency opportunities
               - Capital requirements and funding strategies
               - Profitability projections and margin improvement
            
            2. Investment Strategy:
               - Capital allocation priorities and ROI analysis
               - Investment timing and phasing strategies
               - Risk assessment and mitigation approaches
               - Portfolio optimization and resource allocation
            
            3. Financial Planning:
               - Multi-scenario financial modeling
               - Cash flow projections and working capital needs
               - Valuation implications and value creation
               - Exit strategies and monetization paths
            
            Financial deliverables:
               - 3-5 year financial projections with scenarios
               - Investment prioritization with ROI analysis
               - Funding strategy and capital structure optimization
               - Key financial metrics and performance indicators
            
            Ensure all financial analysis supports strategic objectives with clear value creation logic.""",
            server_names=["g-search", "fetch"]
        )
        
        # Operations Strategy Designer
        operations_strategist = Agent(
            name="operations_strategy_designer",
            instruction=f"""You are an operations strategy and execution expert.
            
            Business Context: {context_str}
            Strategic Objectives: {objectives_str}
            
            Design comprehensive operations strategy:
            
            1. Operational Excellence:
               - Process optimization and automation opportunities
               - Quality management and performance standards
               - Supply chain and vendor management strategies
               - Technology integration and digital transformation
            
            2. Capability Building:
               - Core competency development and enhancement
               - Talent acquisition and development plans
               - Organizational design and structure optimization
               - Culture and change management strategies
            
            3. Execution Planning:
               - Implementation roadmap with milestones
               - Resource allocation and capacity planning
               - Risk management and contingency planning
               - Performance measurement and KPI frameworks
            
            Operations deliverables:
               - Detailed implementation roadmap with timelines
               - Resource requirements and capability assessments
               - Process design and workflow optimization
               - Performance monitoring and success metrics
            
            Focus on practical, executable strategies that ensure successful implementation.""",
            server_names=["fetch"]
        )
        
        # Risk Management Strategist
        risk_strategist = Agent(
            name="risk_management_strategist",
            instruction=f"""You are a strategic risk management expert.
            
            Business Context: {context_str}
            Strategic Objectives: {objectives_str}
            
            Develop comprehensive risk management framework:
            
            1. Risk Identification:
               - Strategic risks and competitive threats
               - Market risks and economic uncertainties
               - Operational risks and execution challenges
               - Regulatory and compliance risks
            
            2. Risk Assessment:
               - Probability and impact analysis
               - Risk interdependencies and correlation effects
               - Scenario planning and stress testing
               - Early warning indicators and monitoring
            
            3. Risk Mitigation:
               - Prevention and reduction strategies
               - Risk transfer and insurance options
               - Contingency planning and crisis management
               - Portfolio diversification and hedging
            
            Risk management deliverables:
               - Comprehensive risk register with assessments
               - Risk mitigation strategies and action plans
               - Scenario analysis with response strategies
               - Risk monitoring framework and dashboards
            
            Ensure risk management supports strategic objectives while protecting value creation.""",
            server_names=["fetch"]
        )
        
        # Strategy Integration Architect
        strategy_architect = Agent(
            name="strategy_integration_architect",
            instruction=f"""You are a strategic integration and synthesis expert.
            
            Business Context: {context_str}
            Strategic Objectives: {objectives_str}
            
            Create integrated comprehensive business strategy:
            
            1. Strategy Synthesis:
               - Integrate market, financial, operations, and risk strategies
               - Resolve conflicts and optimize trade-offs
               - Align all elements with strategic objectives
               - Ensure coherence and internal consistency
            
            2. Strategic Architecture:
               - Define strategic pillars and priorities
               - Create strategic roadmap with phases
               - Design governance and decision frameworks
               - Establish strategic metrics and dashboards
            
            3. Implementation Strategy:
               - Change management and transformation plan
               - Communication and stakeholder engagement
               - Resource mobilization and team formation
               - Success measurement and course correction
            
            Final strategy document structure:
               1. Executive Summary and Strategic Vision
               2. Strategic Context and Market Analysis
               3. Strategic Objectives and Success Metrics
               4. Strategic Pillars and Key Initiatives
               5. Financial Plan and Investment Strategy
               6. Implementation Roadmap and Milestones
               7. Risk Management and Contingency Plans
               8. Governance and Performance Management
               9. Appendices with Supporting Analysis
            
            Create a comprehensive, executable strategy document that serves as the master plan.
            The final content will be returned by the orchestrator, not saved by the agent.
            """,
            server_names=["filesystem"] # May use fs for temp operations
        )
        
        return [market_researcher, financial_analyst, operations_strategist, risk_strategist, strategy_architect]
    
    async def _create_strategy_planning_task(self, business_context: Dict[str, Any], objectives: List[str]) -> str:
        """Create comprehensive strategy planning task"""
        
        context_str = str(business_context)
        objectives_str = ", ".join(objectives)
        
        task = f"""Execute comprehensive business strategy planning process.
        
        Business Context: {context_str}
        Strategic Objectives: {objectives_str}
        
        Mission: Create a comprehensive, integrated business strategy that achieves 
        strategic objectives while managing risks and optimizing resource allocation.
        
        Strategy Development Workflow:
        
        1. MARKET RESEARCH & ANALYSIS (market_research_strategist):
           - Conduct comprehensive market and competitive analysis
           - Identify strategic opportunities and market positioning
           - Analyze customer needs and value proposition optimization
           - Provide market intelligence for strategic decision-making
        
        2. FINANCIAL STRATEGY DEVELOPMENT (financial_strategy_analyst):
           - Develop financial models and investment strategies
           - Analyze funding requirements and capital allocation
           - Create ROI projections and value creation plans
           - Design financial performance metrics and targets
        
        3. OPERATIONS STRATEGY DESIGN (operations_strategy_designer):
           - Design operational excellence and capability building plans
           - Create implementation roadmap with resource requirements
           - Develop process optimization and technology strategies
           - Define organizational design and talent strategies
        
        4. RISK MANAGEMENT PLANNING (risk_management_strategist):
           - Identify and assess strategic and operational risks
           - Develop risk mitigation and contingency strategies
           - Create scenario planning and stress testing frameworks
           - Design risk monitoring and early warning systems
        
        5. STRATEGY INTEGRATION & SYNTHESIS (strategy_integration_architect):
           - Integrate all strategic elements into coherent master plan
           - Resolve conflicts and optimize strategic trade-offs
           - Create implementation governance and change management
           - Finalize comprehensive strategy document and return it as the final output.
        
        Success Criteria:
        - Comprehensive strategy addressing all key dimensions
        - Clear alignment with stated business objectives
        - Detailed implementation roadmap with timelines
        - Robust risk management and contingency planning
        - Executive-ready strategy document with supporting analysis
        
        Deliver a complete strategic blueprint for successful execution."""
        
        return task


# Factory function for easy instantiation
async def create_strategy_planner(google_drive_mcp_url: str = "http://localhost:3001") -> StrategyPlannerMCPAgent:
    """Create and return a StrategyPlannerMCPAgent instance"""
    return StrategyPlannerMCPAgent(google_drive_mcp_url=google_drive_mcp_url)


# Main execution function
async def run_strategy_planning(business_context: Dict[str, Any], objectives: List[str], 
                              google_drive_mcp_url: str = "http://localhost:3001") -> Dict[str, Any]:
    """Run strategy planning with specified parameters"""
    
    planner_agent = await create_strategy_planner(google_drive_mcp_url)
    return await planner_agent.run_strategy_planning(business_context, objectives)


# CLI execution
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python strategy_planner_agent.py 'business_context' 'objective1,objective2'")
        print("Example: python strategy_planner_agent.py 'Tech startup in AI space' 'market_expansion,revenue_growth,operational_efficiency'")
        sys.exit(1)
    
    business_context = {"description": sys.argv[1]}
    objectives = [obj.strip() for obj in sys.argv[2].split(',')]
    
    # Run the strategy planning
    result = asyncio.run(run_strategy_planning(business_context, objectives))
    
    if result["success"]:
        print(f"✅ Strategy planning completed successfully!")
        print(f"📄 Strategy plan saved to: {result['file_url']}")
        print(f"🎯 Objectives: {', '.join(result['objectives'])}")
        print(f"🏢 Business context: {result['business_context']}")
    else:
        print(f"❌ Strategy planning failed: {result['error']}") 