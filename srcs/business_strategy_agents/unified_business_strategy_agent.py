"""
Unified Business Strategy Agent - Real MCPAgent Implementation
------------------------------------------------------------
Integrates all converted business strategy agents into a unified workflow.
Replaces the fake BaseAgent architecture with real MCPAgent implementation.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator, QualityRating
from mcp_agent.workflows.llm.evaluator_optimizer_llm import EvaluatorOptimizerLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from srcs.common.utils import setup_agent_app
import aiohttp
import json

# Helper function to create the HTTP client session
def get_http_session():
    return aiohttp.ClientSession()


class UnifiedBusinessStrategyMCPAgent:
    """
    Real MCPAgent that provides comprehensive business intelligence 
    and strategy development capabilities.
    """
    
    def __init__(self):
        self.app = setup_agent_app("unified_strategy_agent")

    async def run_unified_strategy(self, keywords: List[str], business_context: Dict[str, Any], objectives: List[str]):
        """Run the end-to-end unified business strategy workflow."""
        async with self.app.run() as unified_app:
            context = unified_app.context
            logger = unified_app.logger
            
            # Configure MCP servers
            await self._configure_mcp_servers(context, logger)
            
            # Define specialized agents
            agents = await self._create_unified_agents(
                keywords, business_context, objectives
            )
            
            # Create orchestrator
            orchestrator = Orchestrator(
                llm_factory=OpenAIAugmentedLLM,
                available_agents=agents,
                plan_type="full"
            )
            
            # Execute unified task
            task = await self._create_unified_task(
                keywords, business_context, objectives
            )
            
            logger.info("Starting unified business strategy analysis...")
            
            try:
                report_content = await orchestrator.generate_str(
                    message=task,
                    request_params=RequestParams(model="gemini-2.5-flash-lite-preview-06-07", temperature=0.2)
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

                logger.info(f"Successfully uploaded unified report to Google Drive. File ID: {upload_result.get('fileId')}")
                
                return {
                    "success": True,
                    "output_file": upload_result.get("fileId"),
                    "file_url": f"https://docs.google.com/document/d/{upload_result.get('fileId')}",
                    "keywords": keywords,
                    "timestamp": timestamp,
                    "result": report_content[:200] + "..."
                }
                
            except Exception as e:
                logger.error(f"Unified business strategy analysis failed: {e}", exc_info=True)
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
            logger.info("Filesystem server configured for tool access.")
        
        # Check for required servers
        required_servers = ["g-search", "fetch"]
        missing_servers = []
        
        for server in required_servers:
            if server not in context.config.mcp.servers:
                missing_servers.append(server)
        
        if missing_servers:
            logger.warning(f"Missing MCP servers: {missing_servers}")
            logger.info("Install missing servers: npm install -g g-search-mcp")
    
    async def _create_unified_agents(self, 
                                   keywords: List[str],
                                   business_context: Dict[str, Any],
                                   objectives: List[str]) -> List[Agent]:
        """Create all specialized agents for unified workflow"""
        
        keyword_str = ", ".join(keywords)
        region_str = ", ".join(regions) if regions else "global"
        context_str = str(business_context) if business_context else "General business"
        objectives_str = ", ".join(objectives) if objectives else "Growth and optimization"
        
        # === DATA COLLECTION AGENTS ===
        
        # Market Intelligence Collector
        market_intel_collector = Agent(
            name="market_intelligence_collector",
            instruction=f"""You are a comprehensive market intelligence expert.
            
            Keywords: {keyword_str}
            Regions: {region_str}
            Time Horizon: {time_horizon}
            
            Collect comprehensive market intelligence:
            
            1. Market Analysis:
               - Market size, growth rates, and segmentation
               - Competitive landscape and key players
               - Customer behavior and preferences
               - Pricing trends and value propositions
            
            2. Industry Intelligence:
               - Industry trends and disruption patterns
               - Technology adoption and innovation cycles
               - Regulatory changes and compliance requirements
               - Investment flows and funding activities
            
            3. Competitive Intelligence:
               - Competitor strategies and positioning
               - Product launches and market moves
               - Partnership and acquisition activities
               - Strengths, weaknesses, and vulnerabilities
            
            Data requirements:
            - Recent news and industry reports (last 90 days)
            - Market research and analyst reports
            - Company financial data and metrics
            - Technology and innovation developments
            
            Provide comprehensive intelligence with credible sources and URLs.""",
            server_names=["g-search", "fetch"]
        )
        
        # Social & Sentiment Analyzer
        social_sentiment_analyzer = Agent(
            name="social_sentiment_analyzer",
            instruction=f"""You are a social media and sentiment analysis expert.
            
            Keywords: {keyword_str}
            Regions: {region_str}
            
            Analyze social sentiment and community insights:
            
            1. Social Media Analysis:
               - Platform-specific sentiment trends
               - Viral content and engagement patterns
               - Influencer opinions and thought leadership
               - Community discussions and feedback
            
            2. Sentiment Intelligence:
               - Brand sentiment and reputation analysis
               - Product/service sentiment tracking
               - Crisis indicators and risk signals
               - Positive sentiment drivers and amplifiers
            
            3. Community Insights:
               - User-generated content and reviews
               - Feature requests and pain points
               - Adoption barriers and success factors
               - Demographic and psychographic patterns
            
            Focus on actionable insights that inform strategy and positioning.
            Quantify sentiment scores and engagement metrics where possible.""",
            server_names=["g-search", "fetch"]
        )
        
        # === ANALYSIS AGENTS ===
        
        # Trend Pattern Analyzer
        trend_pattern_analyzer = Agent(
            name="trend_pattern_analyzer",
            instruction=f"""You are a strategic trend and pattern analysis expert.
            
            Focus Areas: {keyword_str}
            Time Horizon: {time_horizon}
            
            Analyze trends and identify strategic patterns:
            
            1. Trend Analysis:
               - Emerging trends and growth trajectories
               - Technology adoption curves and cycles
               - Consumer behavior evolution
               - Market disruption patterns
            
            2. Pattern Recognition:
               - Cross-industry convergences and synergies
               - Cyclical patterns and seasonal effects
               - Weak signals and early indicators
               - Correlation analysis and causation mapping
            
            3. Strategic Implications:
               - Trend durability and sustainability assessment
               - Competitive advantage opportunities
               - Threat identification and risk assessment
               - Timing and sequencing strategies
            
            Provide quantitative analysis with supporting data and projections.
            Focus on patterns that create strategic opportunities.""",
            server_names=["fetch"]
        )
        
        # Opportunity Detection Engine
        opportunity_detector = Agent(
            name="opportunity_detection_engine",
            instruction=f"""You are a strategic opportunity identification expert.
            
            Business Context: {context_str}
            Strategic Objectives: {objectives_str}
            Market Focus: {keyword_str}
            
            Identify and prioritize strategic opportunities:
            
            1. Opportunity Identification:
               - Market gaps and unmet needs
               - Technology convergence opportunities
               - Demographic shift advantages
               - Regulatory change benefits
               - Competitive weakness exploitation
            
            2. Opportunity Assessment:
               - Market size and growth potential
               - Revenue model viability
               - Competitive barriers and moats
               - Resource requirements and capabilities
               - Risk-reward profile analysis
            
            3. Prioritization Framework:
               - Strategic fit and alignment
               - Implementation feasibility
               - Time to market and first-mover advantage
               - ROI potential and payback period
               - Scalability and expansion potential
            
            Provide detailed opportunity briefs with:
               - Clear value proposition
               - Go-to-market strategy outline
               - Success metrics and milestones
               - Risk mitigation approaches
            
            Focus on opportunities that align with strategic objectives.""",
            server_names=["fetch"]
        )
        
        # === STRATEGY DEVELOPMENT AGENTS ===
        
        # Strategic Planning Architect
        strategic_planner = Agent(
            name="strategic_planning_architect",
            instruction=f"""You are a comprehensive strategic planning expert.
            
            Business Context: {context_str}
            Strategic Objectives: {objectives_str}
            Time Horizon: {time_horizon}
            
            Develop comprehensive business strategy:
            
            1. Strategic Framework:
               - Vision, mission, and strategic objectives alignment
               - Strategic pillars and key initiatives
               - Competitive positioning and differentiation
               - Value proposition and business model design
            
            2. Strategic Planning:
               - Multi-phase strategic roadmap
               - Resource allocation and investment priorities
               - Capability building and organizational design
               - Partnership and ecosystem strategies
            
            3. Implementation Strategy:
               - Change management and transformation plan
               - Governance and decision-making frameworks
               - Performance measurement and KPIs
               - Risk management and contingency planning
            
            Create detailed strategic plans with:
               - Clear strategic choices and rationale
               - Phased implementation with milestones
               - Resource requirements and budget implications
               - Success metrics and performance indicators
            
            Ensure strategy is comprehensive, coherent, and executable.""",
            server_names=["fetch"]
        )
        
        # === QUALITY CONTROL ===
        
        # Quality Evaluator
        quality_evaluator = Agent(
            name="comprehensive_quality_evaluator",
            instruction=f"""You are a comprehensive business analysis quality evaluator.
            
            Evaluate the quality of business strategy analysis for: {keyword_str}
            
            Quality assessment criteria:
            
            1. Data Quality (EXCELLENT = multiple authoritative sources, recent data):
               - Source credibility and authority
               - Data freshness and relevance
               - Completeness and coverage
               - Accuracy and verification
            
            2. Analysis Depth (EXCELLENT = comprehensive, multi-dimensional):
               - Analytical rigor and methodology
               - Insight quality and actionability
               - Pattern recognition and correlations
               - Strategic implications clarity
            
            3. Strategic Value (EXCELLENT = clear competitive advantage):
               - Strategic relevance and alignment
               - Opportunity identification quality
               - Implementation feasibility
               - Competitive advantage potential
            
            4. Completeness (EXCELLENT = all required elements):
               - Comprehensive coverage of key areas
               - Integration across different analyses
               - Risk assessment and mitigation
               - Actionable recommendations
            
            Rating scale:
            - EXCELLENT: Exceptional quality, comprehensive and actionable
            - GOOD: High quality, reliable with good strategic value
            - FAIR: Adequate quality, some gaps but usable
            - POOR: Low quality, incomplete or unreliable
            
            Provide detailed feedback and improvement recommendations.
            Only EXCELLENT rated analysis should proceed to final report.""",
        )
        
        # Create quality-controlled market intelligence
        quality_market_intel = EvaluatorOptimizerLLM(
            optimizer=market_intel_collector,
            evaluator=quality_evaluator,
            llm_factory=OpenAIAugmentedLLM,
            min_rating=QualityRating.EXCELLENT,
        )
        
        # === SYNTHESIS AGENT ===
        
        # Master Strategy Synthesizer
        master_synthesizer = Agent(
            name="master_strategy_synthesizer",
            instruction=f"""You are the master business strategy synthesizer.
            
            Integration Parameters:
            - Keywords: {keyword_str}
            - Business Context: {context_str}
            - Objectives: {objectives_str}
            - Regions: {region_str}
            - Time Horizon: {time_horizon}
            
            Create comprehensive unified business strategy report:
            
            REPORT STRUCTURE:
            
            1. EXECUTIVE SUMMARY
               - Strategic overview and key recommendations
               - Top opportunities and strategic priorities
               - Critical success factors and timelines
               - Investment requirements and ROI projections
            
            2. MARKET INTELLIGENCE SYNTHESIS
               - Market analysis and competitive landscape
               - Industry trends and disruption patterns
               - Customer insights and behavioral patterns
               - Social sentiment and community feedback
            
            3. STRATEGIC TREND ANALYSIS
               - Key trends and pattern recognition
               - Strategic implications and opportunities
               - Timing and sequencing recommendations
               - Trend-based competitive advantages
            
            4. OPPORTUNITY PORTFOLIO
               - Prioritized opportunity matrix
               - Detailed opportunity assessments
               - Go-to-market strategies
               - Resource requirements and timelines
            
            5. COMPREHENSIVE STRATEGY PLAN
               - Strategic framework and positioning
               - Multi-phase implementation roadmap
               - Capability building requirements
               - Partnership and ecosystem strategies
            
            6. RISK MANAGEMENT FRAMEWORK
               - Strategic and operational risks
               - Mitigation strategies and contingencies
               - Scenario planning and stress testing
               - Early warning systems and monitoring
            
            7. IMPLEMENTATION BLUEPRINT
               - Detailed action plans with milestones
               - Resource allocation and budgeting
               - Governance and decision frameworks
               - Performance measurement and KPIs
            
            8. APPENDICES
               - Data sources and methodology
               - Supporting analysis and calculations
               - Risk registers and mitigation plans
               - Success metrics and dashboards
            
            SYNTHESIS REQUIREMENTS:
               - Integrate all analyses into coherent strategy
               - Resolve conflicts and optimize trade-offs
               - Ensure alignment with business objectives
               - Provide clear, actionable recommendations
               - Include quantitative projections where possible
               - Professional formatting with executive summary
            
            Save the comprehensive strategy report to: {output_file_name}
            
            This report replaces the entire fake BaseAgent architecture with a 
            real MCPAgent-based comprehensive business strategy system.""",
            server_names=["filesystem"] # May use fs for temp operations
        )
        
        return [
            quality_market_intel,
            social_sentiment_analyzer,
            trend_pattern_analyzer,
            opportunity_detector,
            strategic_planner,
            master_synthesizer
        ]
    
    async def _create_unified_task(self,
                                 keywords: List[str],
                                 business_context: Dict[str, Any],
                                 objectives: List[str],
                                 regions: List[str],
                                 time_horizon: str) -> str:
        """Create comprehensive unified business strategy task"""
        
        keyword_str = ", ".join(keywords)
        region_str = ", ".join(regions) if regions else "global markets"
        context_str = str(business_context) if business_context else "General business environment"
        objectives_str = ", ".join(objectives) if objectives else "Growth and competitive advantage"
        time_horizon_str = time_horizon.replace("_", " ")

        task = f"""Execute a comprehensive, unified business strategy analysis.
        
        Primary Keywords: {keyword_str}
        Business Context: {context_str}
        Strategic Objectives: {objectives_str}
        Regions: {region_str}
        Time Horizon: {time_horizon_str}
        
        Workflow:
        
        1.  Data Collection (quality_market_intel, social_sentiment_analyzer):
            - Gather high-quality market data, competitor intelligence, and social sentiment.
            
        2.  Analysis & Synthesis (trend_pattern_analyzer, opportunity_detector):
            - Analyze collected data to identify key trends, patterns, and strategic opportunities.
            
        3.  Strategy Formulation (strategic_planning_architect):
            - Develop a detailed strategic plan based on the synthesized insights.
            
        4.  Final Report Generation (master_synthesizer):
            - Integrate all findings into a single, comprehensive, executive-ready report.
            - Return the complete report as the final output.
            
        Ensure all steps are executed in a coordinated manner to produce a coherent and actionable strategic plan.
        """
        return task


# Factory and execution functions
async def create_unified_business_strategy(google_drive_mcp_url: str = "http://localhost:3001") -> UnifiedBusinessStrategyMCPAgent:
    """Create and return a UnifiedBusinessStrategyMCPAgent instance"""
    return UnifiedBusinessStrategyMCPAgent(google_drive_mcp_url=google_drive_mcp_url)


async def run_unified_business_strategy(keywords: List[str],
                                      business_context: Dict[str, Any] = None,
                                      objectives: List[str] = None,
                                      regions: List[str] = None,
                                      time_horizon: str = "12_months",
                                      google_drive_mcp_url: str = "http://localhost:3001") -> Dict[str, Any]:
    """Run unified business strategy analysis"""
    
    strategy_agent = await create_unified_business_strategy(google_drive_mcp_url)
    return await strategy_agent.run_comprehensive_analysis(
        keywords, business_context, objectives, regions, time_horizon
    )


# CLI execution
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python unified_business_strategy_agent.py 'keyword1,keyword2' [context] [objectives] [regions] [time_horizon]")
        print("Example: python unified_business_strategy_agent.py 'AI,fintech' 'Tech startup' 'growth,expansion' 'North America,Europe' '12_months'")
        sys.exit(1)
    
    keywords = [k.strip() for k in sys.argv[1].split(',')]
    business_context = {"description": sys.argv[2]} if len(sys.argv) > 2 else None
    objectives = [o.strip() for o in sys.argv[3].split(',')] if len(sys.argv) > 3 else None
    regions = [r.strip() for r in sys.argv[4].split(',')] if len(sys.argv) > 4 else None
    time_horizon = sys.argv[5] if len(sys.argv) > 5 else "12_months"
    
    # Run the unified business strategy analysis
    print(f"Unified Business Strategy analysis for: {keywords}")
    
    result = asyncio.run(run_unified_business_strategy(
        keywords=keywords,
        business_context={"description": "A tech startup in the AI space."},
        objectives=["Increase market share by 20%", "Launch one new product line"],
        regions=["North America", "Europe"],
        time_horizon="24_months",
        google_drive_mcp_url="http://localhost:3001"
    ))
    
    if result["success"]:
        print(f"\n✅ Unified analysis completed successfully!")
        print(f"📄 Report uploaded. File URL: {result.get('file_url')}")
    else:
        print(f"\n❌ Unified analysis failed: {result['error']}") 