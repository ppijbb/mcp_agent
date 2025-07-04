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
from mcp_agent.config import get_settings
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)


class UnifiedBusinessStrategyMCPAgent:
    """
    Real MCPAgent that replaces the entire fake business_strategy_agents architecture.
    Provides comprehensive business intelligence and strategy development capabilities.
    """
    
    def __init__(self, output_dir: str = "business_strategy_reports"):
        self.output_dir = output_dir
        self.app = MCPApp(
            name="unified_business_strategy",
            settings=get_settings("configs/mcp_agent.config.yaml"),
            human_input_callback=None
        )
        
    async def run_comprehensive_analysis(
        self, 
        keywords: List[str],
        business_context: Dict[str, Any] = None,
        objectives: List[str] = None,
        regions: List[str] = None,
        time_horizon: str = "12_months"
    ) -> Dict[str, Any]:
        """
        Run comprehensive business strategy analysis combining:
        - Data collection and market intelligence
        - Trend analysis and pattern recognition
        - Strategic planning and opportunity identification
        - Risk assessment and implementation roadmap
        """
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"unified_business_strategy_{timestamp}.md"
        output_path = os.path.join(self.output_dir, output_file)
        
        async with self.app.run() as strategy_app:
            context = strategy_app.context
            logger = strategy_app.logger
            
            # Configure MCP servers
            await self._configure_mcp_servers(context, logger)
            
            # Create all specialized agents
            agents = await self._create_unified_agents(
                keywords, business_context, objectives, regions, time_horizon, output_path
            )
            
            # Create orchestrator
            orchestrator = Orchestrator(
                llm_factory=OpenAIAugmentedLLM,
                available_agents=agents,
                plan_type="full"
            )
            
            # Execute comprehensive analysis
            task = await self._create_unified_task(
                keywords, business_context, objectives, regions, time_horizon, output_path
            )
            
            logger.info(f"Starting unified business strategy analysis for: {keywords}")
            
            try:
                result = await orchestrator.generate_str(
                    message=task,
                    request_params=RequestParams(model="gemini-2.5-flash-lite-preview-06-07")
                )
                
                return {
                    "success": True,
                    "output_file": output_path,
                    "keywords": keywords,
                    "business_context": business_context,
                    "objectives": objectives,
                    "regions": regions or ["global"],
                    "time_horizon": time_horizon,
                    "timestamp": timestamp,
                    "result": result
                }
                
            except Exception as e:
                logger.error(f"Unified business strategy analysis failed: {e}")
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
            logger.info("Install missing servers: npm install -g g-search-mcp")
    
    async def _create_unified_agents(self, 
                                   keywords: List[str],
                                   business_context: Dict[str, Any],
                                   objectives: List[str],
                                   regions: List[str],
                                   time_horizon: str,
                                   output_path: str) -> List[Agent]:
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
            
            Save the comprehensive strategy report to: {output_path}
            
            This report replaces the entire fake BaseAgent architecture with a 
            real MCPAgent-based comprehensive business strategy system.""",
            server_names=["filesystem"]
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
                                 time_horizon: str,
                                 output_path: str) -> str:
        """Create comprehensive unified business strategy task"""
        
        keyword_str = ", ".join(keywords)
        region_str = ", ".join(regions) if regions else "global markets"
        context_str = str(business_context) if business_context else "General business environment"
        objectives_str = ", ".join(objectives) if objectives else "Growth and competitive advantage"
        
        task = f"""Execute comprehensive unified business strategy analysis and planning.
        
        ANALYSIS PARAMETERS:
        - Keywords/Focus: {keyword_str}
        - Business Context: {context_str}
        - Strategic Objectives: {objectives_str}
        - Geographic Scope: {region_str}
        - Time Horizon: {time_horizon}
        
        MISSION: Replace the fake BaseAgent architecture with a real MCPAgent-based 
        comprehensive business strategy system that delivers actionable strategic intelligence.
        
        UNIFIED WORKFLOW EXECUTION:
        
        PHASE 1: COMPREHENSIVE INTELLIGENCE GATHERING
        1. MARKET INTELLIGENCE (quality_market_intel):
           - Collect high-quality market and competitive intelligence
           - Quality control ensures only EXCELLENT rated data proceeds
           - Focus on recent, credible, and actionable insights
           - Comprehensive coverage of market dynamics and competition
        
        2. SOCIAL SENTIMENT ANALYSIS (social_sentiment_analyzer):
           - Analyze social media trends and community sentiment
           - Identify reputation risks and opportunity signals
           - Map customer feedback and feature requests
           - Quantify sentiment scores and engagement patterns
        
        PHASE 2: STRATEGIC ANALYSIS & PATTERN RECOGNITION
        3. TREND PATTERN ANALYSIS (trend_pattern_analyzer):
           - Identify strategic trends and pattern correlations
           - Analyze technology adoption curves and market cycles
           - Detect weak signals and emerging opportunities
           - Assess trend durability and strategic implications
        
        4. OPPORTUNITY DETECTION (opportunity_detector):
           - Identify and prioritize strategic opportunities
           - Assess market potential and competitive positioning
           - Develop opportunity briefs with implementation pathways
           - Align opportunities with strategic objectives
        
        PHASE 3: STRATEGY DEVELOPMENT & PLANNING
        5. STRATEGIC PLANNING (strategic_planner):
           - Develop comprehensive strategic framework
           - Create multi-phase implementation roadmap
           - Design capability building and resource strategies
           - Establish governance and performance frameworks
        
        PHASE 4: SYNTHESIS & MASTER STRATEGY
        6. MASTER SYNTHESIS (master_strategy_synthesizer):
           - Integrate all analyses into unified strategy
           - Resolve conflicts and optimize strategic trade-offs
           - Create comprehensive implementation blueprint
           - Generate executive-ready strategy document
           - Save complete strategy to: {output_path}
        
        SUCCESS CRITERIA:
        - Comprehensive strategy addressing all key dimensions
        - High-quality data and analysis throughout
        - Clear strategic recommendations with implementation plans
        - Quantitative projections and ROI analysis
        - Risk assessment and mitigation strategies
        - Executive-ready deliverable with supporting analysis
        
        DELIVERABLE: Complete unified business strategy that replaces the fake 
        BaseAgent system with a real MCPAgent-based strategic intelligence platform.
        
        This represents the complete transformation from fake to real MCPAgent architecture."""
        
        return task


# Factory and execution functions
async def create_unified_business_strategy(output_dir: str = "business_strategy_reports") -> UnifiedBusinessStrategyMCPAgent:
    """Create and return a UnifiedBusinessStrategyMCPAgent instance"""
    return UnifiedBusinessStrategyMCPAgent(output_dir=output_dir)


async def run_unified_business_strategy(keywords: List[str],
                                      business_context: Dict[str, Any] = None,
                                      objectives: List[str] = None,
                                      regions: List[str] = None,
                                      time_horizon: str = "12_months",
                                      output_dir: str = "business_strategy_reports") -> Dict[str, Any]:
    """Run unified business strategy analysis"""
    
    strategy_agent = await create_unified_business_strategy(output_dir)
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
    result = asyncio.run(run_unified_business_strategy(
        keywords, business_context, objectives, regions, time_horizon
    ))
    
    if result["success"]:
        print(f"✅ Unified business strategy analysis completed successfully!")
        print(f"📄 Strategy report saved to: {result['output_file']}")
        print(f"🔍 Keywords: {', '.join(result['keywords'])}")
        print(f"🎯 Objectives: {', '.join(result['objectives']) if result['objectives'] else 'Default growth objectives'}")
        print(f"🌍 Regions: {', '.join(result['regions'])}")
        print(f"⏰ Time horizon: {result['time_horizon']}")
        print("\n🎉 Successfully converted from fake BaseAgent to real MCPAgent architecture!")
    else:
        print(f"❌ Unified business strategy analysis failed: {result['error']}") 