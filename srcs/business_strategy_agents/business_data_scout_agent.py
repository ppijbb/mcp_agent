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
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import get_settings
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM


class BusinessDataScoutMCPAgent:
    """Real MCPAgent for Business Data Scouting"""
    
    def __init__(self, output_dir: str = "business_strategy_reports"):
        self.output_dir = output_dir
        self.app = MCPApp(
            name="business_data_scout",
            settings=get_settings("configs/mcp_agent.config.yaml"),
            human_input_callback=None
        )
        
    async def run_data_collection(self, keywords: List[str], regions: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive business data collection"""
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"business_data_scout_report_{timestamp}.md"
        output_path = os.path.join(self.output_dir, output_file)
        
        async with self.app.run() as scout_app:
            context = scout_app.context
            logger = scout_app.logger
            
            # Configure MCP servers
            await self._configure_mcp_servers(context, logger)
            
            # Define specialized agents
            agents = await self._create_specialized_agents(keywords, regions, output_path)
            
            # Create orchestrator
            orchestrator = Orchestrator(
                llm_factory=OpenAIAugmentedLLM,
                available_agents=agents,
                plan_type="full"
            )
            
            # Execute data collection task
            task = await self._create_data_collection_task(keywords, regions, output_path)
            
            logger.info(f"Starting business data collection for keywords: {keywords}")
            
            try:
                result = await orchestrator.generate_str(
                    message=task,
                    request_params=RequestParams(model="gpt-4o-mini")
                )
                
                return {
                    "success": True,
                    "output_file": output_path,
                    "keywords": keywords,
                    "regions": regions or ["global"],
                    "timestamp": timestamp,
                    "result": result
                }
                
            except Exception as e:
                logger.error(f"Business data collection failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "keywords": keywords,
                    "timestamp": timestamp
                }
    
    async def _configure_mcp_servers(self, context, logger):
        """Configure required MCP servers"""
        
        # Configure filesystem server
        if "filesystem" in context.config.mcp.servers:
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            logger.info("Filesystem server configured")
        
        # Check for required servers
        required_servers = ["g-search", "fetch"]
        missing_servers = []
        
        for server in required_servers:
            if server not in context.config.mcp.servers:
                missing_servers.append(server)
        
        if missing_servers:
            logger.warning(f"Missing MCP servers: {missing_servers}")
            logger.info("Install missing servers for full functionality")
    
    async def _create_specialized_agents(self, keywords: List[str], regions: List[str], output_path: str) -> List[Agent]:
        """Create specialized agents for data collection"""
        
        keyword_str = ", ".join(keywords)
        region_str = ", ".join(regions) if regions else "global"
        
        # News & Media Data Collector
        news_collector = Agent(
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
            server_names=["g-search", "fetch"]
        )
        
        # Social Media & Trends Collector
        social_collector = Agent(
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
            server_names=["g-search", "fetch"]
        )
        
        # Market Intelligence Collector
        market_collector = Agent(
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
            server_names=["g-search", "fetch"]
        )
        
        # Data Quality Evaluator
        data_evaluator = Agent(
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
            server_names=["fetch"]
        )
        
        # Report Synthesizer
        report_synthesizer = Agent(
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
            server_names=["filesystem"]
        )
        
        return [news_collector, social_collector, market_collector, data_evaluator, report_synthesizer]
    
    async def _create_data_collection_task(self, keywords: List[str], regions: List[str], output_path: str) -> str:
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
async def create_business_data_scout(output_dir: str = "business_strategy_reports") -> BusinessDataScoutMCPAgent:
    """Create and return a BusinessDataScoutMCPAgent instance"""
    return BusinessDataScoutMCPAgent(output_dir=output_dir)


# Main execution function
async def run_business_data_scout(keywords: List[str], regions: List[str] = None, 
                                 output_dir: str = "business_strategy_reports") -> Dict[str, Any]:
    """Run business data scout with specified parameters"""
    
    scout_agent = await create_business_data_scout(output_dir)
    return await scout_agent.run_data_collection(keywords, regions)


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