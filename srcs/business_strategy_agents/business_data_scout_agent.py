"""
Business Data Scout Agent - Real MCPAgent Implementation
--------------------------------------------------------
Converted from fake BaseAgent to real MCPAgent using mcp_agent library.
Collects and analyzes business data from multiple sources.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator, QualityRating
from mcp_agent.workflows.llm.evaluator_optimizer_llm import EvaluatorOptimizerLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from srcs.common.utils import setup_agent_app, save_report
from srcs.core.agent.base import BaseAgent
from mcp_agent.agents.agent import Agent as MCP_Agent

# --- INSTRUCTIONS DEFINED LOCALLY ---
BUSINESS_DATA_SCOUT_AGENT_INSTRUCTION = """
You are a Business Data Scout. Your mission is to gather the latest and most relevant business data based on the given keywords and regions.
You will use search tools to find news articles, market reports, social media trends, and community discussions.
Focus on factual, up-to-date information.

Keywords: {keywords}
Regions: {regions}
"""

BUSINESS_DATA_SCOUT_EVALUATOR_INSTRUCTION = """
You are a Quality Evaluator for business intelligence.
Evaluate the gathered data based on relevance, accuracy, and timeliness.
Provide a rating (EXCELLENT, GOOD, FAIR, POOR) and justify your assessment.
"""
# --- END LOCAL INSTRUCTIONS ---


class BusinessDataScoutAgent(BaseAgent):
    """
    Business Data Scout Agent, refactored to inherit from BaseAgent.
    Collects and evaluates business-related data from various sources.
    """
    
    def __init__(self):
        super().__init__(
            name="BusinessDataScoutAgent",
            instruction="Collects, analyzes, and evaluates business data based on given keywords and regions.",
            server_names=["g-search", "fetch", "filesystem"]
        )
        self.output_dir = "business_strategy_reports"

    async def run_workflow(self, keywords: List[str], regions: List[str] | None = None):
        """
        The core workflow for scouting and collecting business data.
        """
        async with self.app.run() as app_context:
            self.logger.info(f"Starting data scouting for keywords: {keywords}")
            
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"business_data_scout_report_{timestamp}.md"
            output_path = os.path.join(self.output_dir, output_file)

            # 1. Define specialized sub-agents
            agents = self._create_specialized_agents(keywords, regions, output_path, app_context.llm_factory)
            
            # 2. Get an orchestrator to manage them
            orchestrator = self.get_orchestrator(agents)

            # 3. Define the main task
            task = self._create_data_collection_task(keywords, regions, output_path)

            # 4. Run the orchestrator
            final_report = await orchestrator.run(task)
            
            self.logger.info(f"Data scouting complete. Report saved to {output_path}")
            return {"report_path": output_path, "content": final_report}

    def _create_specialized_agents(self, keywords, regions, output_path, llm_factory):
        """Create specialized agents for data collection"""
        
        keyword_str = ", ".join(keywords)
        region_str = ", ".join(regions) if regions else "global"
        
        # News & Media Data Collector
        news_collector = MCP_Agent(
            name="news_data_collector",
            instruction=f"""You are an expert news and media data collector.
            
            Collect comprehensive news and media data for these keywords: {keyword_str}
            Focus on regions: {region_str}
            
            Execute these search strategies:
            1. Recent news articles (last 30 days)
            2. Industry reports and publications
            3. Press releases and announcements
            4. Market analysis and commentary
            
            For each source, extract:
            - Title and publication date
            - Source credibility and reach
            - Key insights and business implications
            - Sentiment analysis (positive/negative/neutral)
            - Trending topics and emerging themes
            
            Organize findings by relevance and business impact.
            Provide source URLs for verification.""",
            server_names=["g-search", "fetch"],
            llm_factory=llm_factory,
        )
        
        # Social Media & Trends Collector
        social_collector = MCP_Agent(
            name="social_trends_collector", 
            instruction=f"""You are a social media and trends analysis expert.
            
            Analyze social media trends and community discussions for: {keyword_str}
            Target regions: {region_str}
            
            Research areas:
            1. Social media sentiment and engagement
            2. Trending hashtags and discussions
            3. Influencer opinions and thought leadership
            4. Community feedback and user reviews
            5. Viral content and memes related to topics
            
            Extract valuable insights:
            - Engagement metrics and reach
            - Sentiment trends over time
            - Key opinion leaders and influencers
            - Emerging topics and conversations
            - Consumer behavior patterns
            
            Focus on business-relevant social signals that indicate market opportunities.""",
            server_names=["g-search", "fetch"],
            llm_factory=llm_factory,
        )
        
        # Market Intelligence Collector
        market_collector = MCP_Agent(
            name="market_intelligence_collector",
            instruction=f"""You are a market intelligence specialist.
            
            Gather competitive and market intelligence for: {keyword_str}
            Geographic focus: {region_str}
            
            Intelligence areas:
            1. Competitor analysis and positioning
            2. Market size and growth projections
            3. Investment and funding activities
            4. Regulatory changes and compliance
            5. Technology trends and innovations
            
            Collect specific data:
            - Market size estimates and growth rates
            - Key players and competitive landscape
            - Investment rounds and valuations
            - Regulatory developments
            - Technology adoption trends
            
            Provide actionable market insights with supporting data and sources.""",
            server_names=["g-search", "fetch"],
            llm_factory=llm_factory,
        )
        
        # Data Quality Evaluator
        data_evaluator = MCP_Agent(
            name="data_quality_evaluator",
            instruction=f"""You are a data quality assessment expert.
            
            Evaluate the collected business data for: {keyword_str}
            
            Assessment criteria:
            1. Source credibility and authority
            2. Data freshness and timeliness
            3. Completeness and coverage
            4. Accuracy and fact-checking
            5. Business relevance and actionability
            
            For each data source, provide:
            - Credibility score (1-10)
            - Freshness assessment
            - Completeness rating
            - Business relevance score
            - Recommendations for improvement
            
            Flag any inconsistencies or questionable data points.
            Prioritize high-quality, actionable business intelligence.""",
            server_names=["fetch"],
            llm_factory=llm_factory,
        )
        
        # Report Synthesizer
        report_synthesizer = MCP_Agent(
            name="business_data_synthesizer",
            instruction=f"""You are a business intelligence report synthesizer.
            
            Create a comprehensive business data report for: {keyword_str}
            Geographic scope: {region_str}
            
            Report structure:
            1. Executive Summary
            2. Market Overview and Size
            3. Key Trends and Opportunities
            4. Competitive Landscape
            5. Social and Media Sentiment
            6. Risk Assessment
            7. Strategic Recommendations
            8. Data Sources and References
            
            Synthesis requirements:
            - Integrate all collected data sources
            - Identify patterns and correlations
            - Provide actionable business insights
            - Include data quality assessments
            - Present clear visualizable metrics
            - Maintain professional formatting
            
            Save the comprehensive report to: {output_path}
            Format as clean markdown with proper sections and citations.""",
            server_names=["filesystem"],
            llm_factory=llm_factory,
        )
        
        return [news_collector, social_collector, market_collector, data_evaluator, report_synthesizer]
    
    def _create_data_collection_task(self, keywords: List[str], regions: List[str], output_path: str) -> str:
        """Create comprehensive data collection task"""
        
        keyword_str = ", ".join(keywords)
        region_str = ", ".join(regions) if regions else "global markets"
        
        task = f"""Execute comprehensive business data collection and analysis for: {keyword_str}
        
        Geographic Focus: {region_str}
        
        Mission: Collect high-quality business intelligence data from multiple sources and synthesize 
        into actionable strategic insights.
        
        Execution Plan:
        
        1. NEWS & MEDIA COLLECTION (news_data_collector):
           - Gather recent news, industry reports, and media coverage
           - Focus on business implications and market signals
           - Analyze sentiment and emerging themes
        
        2. SOCIAL & TRENDS ANALYSIS (social_trends_collector):
           - Research social media sentiment and engagement
           - Identify trending topics and community discussions
           - Map influencer opinions and thought leadership
        
        3. MARKET INTELLIGENCE (market_intelligence_collector):
           - Analyze competitive landscape and positioning
           - Research market size, growth, and investment activities
           - Track regulatory and technology developments
        
        4. DATA QUALITY ASSESSMENT (data_quality_evaluator):
           - Evaluate source credibility and data accuracy
           - Assess completeness and business relevance
           - Flag inconsistencies and provide quality scores
        
        5. COMPREHENSIVE SYNTHESIS (business_data_synthesizer):
           - Integrate all collected data into unified report
           - Identify key patterns and business opportunities
           - Provide strategic recommendations and action items
           - Save final report to: {output_path}
        
        Success Criteria:
        - High-quality data from credible sources
        - Clear business insights and opportunities
        - Actionable strategic recommendations
        - Professional report with proper citations
        - Comprehensive coverage of all key aspects
        
        Deliver a complete business intelligence report that enables strategic decision-making."""
        
        return task


# Factory function for easy instantiation
async def create_business_data_scout(output_dir: str = "business_strategy_reports") -> BusinessDataScoutAgent:
    """Create and return a BusinessDataScoutMCPAgent instance"""
    return BusinessDataScoutAgent()


# Main execution function
async def run_business_data_scout(
    keywords: List[str],
    regions: Optional[List[str]] = None,
    output_dir: str = "business_strategy_reports",
) -> Dict[str, Any]:
    """Asynchronously run the Business Data Scout MCPAgent."""
    os.makedirs(output_dir, exist_ok=True)

    llm_factory = lambda: OpenAIAugmentedLLM(model="gemini-2.0-flash-lite-001")

    async with MCPApp(settings=get_settings("configs/mcp_agent.config.yaml")).run() as app:
        # Create agents
        scout = Agent(
            name="business_data_scout",
            instruction=BUSINESS_DATA_SCOUT_AGENT_INSTRUCTION.format(
                keywords=", ".join(keywords),
                regions=", ".join(regions) if regions else "Global",
            ),
            server_names=["g-search", "fetch"],
            llm_factory=llm_factory,
        )

        evaluator = Agent(
            name="quality_evaluator",
            instruction=BUSINESS_DATA_SCOUT_EVALUATOR_INSTRUCTION,
            llm_factory=llm_factory,
        )

        # Create orchestrator
        orchestrator = Orchestrator(
            available_agents=[scout, evaluator],
            plan_type="full",
            llm_factory=llm_factory,
        )

        # Define task
        task = (
            f"Conduct a comprehensive business data scout for keywords '{', '.join(keywords)}' "
            f"in regions '{', '.join(regions) if regions else 'Global'}'. "
            "Deliver a structured report with key findings, data sources, and strategic insights."
        )

        try:
            # Run orchestrator
            result = await orchestrator.generate_str(
                message=task,
                request_params=RequestParams(model="gemini-2.0-flash-lite-001"),
            )

            # Save result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{output_dir}/business_data_scout_report_{timestamp}.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)

            return {
                "success": True,
                "output_file": output_file,
                "analysis_summary": "Business data scout completed successfully.",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


# CLI execution
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python business_data_scout_agent.py 'keyword1,keyword2' [region1,region2]")
        print("Example: python business_data_scout_agent.py 'AI,fintech' 'North America,Europe'")
        sys.exit(1)
    
    keywords = [k.strip() for k in sys.argv[1].split(',')]
    regions = [r.strip() for r in sys.argv[2].split(',')] if len(sys.argv) > 2 else None
    
    # Run the business data scout
    result = asyncio.run(run_business_data_scout(keywords, regions))
    
    if result["success"]:
        print(f"✅ Business data collection completed successfully!")
        print(f"📄 Report saved to: {result['output_file']}")
        print(f"🔍 Keywords analyzed: {', '.join(result['keywords'])}")
        print(f"🌍 Regions covered: {', '.join(result['regions'])}")
    else:
        print(f"❌ Business data collection failed: {result['error']}") 