"""
Trend Analyzer Agent - Real MCPAgent Implementation
--------------------------------------------------
Converted from fake BaseAgent to real MCPAgent using mcp_agent library.
Analyzes business trends and identifies emerging opportunities.
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


class TrendAnalyzerMCPAgent:
    """Real MCPAgent for Business Trend Analysis"""
    
    def __init__(self, output_dir: str = "business_strategy_reports"):
        self.output_dir = output_dir
        self.app = MCPApp(
            name="trend_analyzer",
            settings=get_settings("configs/mcp_agent.config.yaml"),
            human_input_callback=None
        )
        
    async def run_trend_analysis(self, focus_areas: List[str], time_horizon: str = "6_months") -> Dict[str, Any]:
        """Run comprehensive trend analysis"""
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"trend_analysis_report_{timestamp}.md"
        output_path = os.path.join(self.output_dir, output_file)
        
        async with self.app.run() as analyzer_app:
            context = analyzer_app.context
            logger = analyzer_app.logger
            
            # Configure MCP servers
            await self._configure_mcp_servers(context, logger)
            
            # Define specialized agents
            agents = await self._create_trend_agents(focus_areas, time_horizon, output_path)
            
            # Create orchestrator with evaluator-optimizer
            orchestrator = Orchestrator(
                llm_factory=OpenAIAugmentedLLM,
                available_agents=agents,
                plan_type="full"
            )
            
            # Execute trend analysis task
            task = await self._create_trend_analysis_task(focus_areas, time_horizon, output_path)
            
            logger.info(f"Starting trend analysis for areas: {focus_areas}")
            
            try:
                result = await orchestrator.generate_str(
                    message=task,
                    request_params=RequestParams(model="gpt-4o-mini")
                )
                
                return {
                    "success": True,
                    "output_file": output_path,
                    "focus_areas": focus_areas,
                    "time_horizon": time_horizon,
                    "timestamp": timestamp,
                    "result": result
                }
                
            except Exception as e:
                logger.error(f"Trend analysis failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "focus_areas": focus_areas,
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
    
    async def _create_trend_agents(self, focus_areas: List[str], time_horizon: str, output_path: str) -> List[Agent]:
        """Create specialized trend analysis agents"""
        
        areas_str = ", ".join(focus_areas)
        
        # Trend Data Collector - Enhanced with quality control
        trend_collector = Agent(
            name="trend_data_collector",
            instruction=f"""You are an expert trend research analyst.
            
            Collect comprehensive trend data for these focus areas: {areas_str}
            Time horizon: {time_horizon}
            
            Research methodology:
            1. Market trend reports and industry analysis
            2. Technology adoption curves and innovation cycles  
            3. Consumer behavior shifts and demographic changes
            4. Investment patterns and funding trends
            5. Regulatory changes and policy developments
            
            For each trend, document:
            - Trend description and key characteristics
            - Growth trajectory and momentum indicators
            - Market size and adoption metrics
            - Key drivers and barriers
            - Geographic variations and regional differences
            - Timeline and development stages
            - Supporting evidence and data sources
            
            Focus on quantifiable trends with strong supporting evidence.
            Provide credible sources and verification URLs.""",
            server_names=["g-search", "fetch"]
        )
        
        # Trend Quality Evaluator
        trend_evaluator = Agent(
            name="trend_quality_evaluator",
            instruction=f"""You are a trend analysis quality evaluator.
            
            Evaluate collected trend data for: {areas_str}
            
            Quality assessment criteria:
            1. Data reliability and source credibility (EXCELLENT = multiple authoritative sources)
            2. Trend significance and market impact (EXCELLENT = substantial business implications)
            3. Evidence strength and supporting metrics (EXCELLENT = quantitative data with clear patterns)
            4. Temporal relevance for {time_horizon} horizon (EXCELLENT = high relevance to timeframe)
            5. Actionability and business applicability (EXCELLENT = clear strategic implications)
            
            Rating scale:
            - EXCELLENT: Exceptional quality, highly reliable and actionable
            - GOOD: High quality, reliable with clear business value
            - FAIR: Adequate quality, some limitations but usable
            - POOR: Low quality, unreliable or not actionable
            
            For each trend, provide:
            - Overall quality rating
            - Detailed assessment for each criterion
            - Specific recommendations for improvement
            - Flag any questionable claims or weak evidence
            
            Only EXCELLENT rated trends should be included in final analysis.""",
        )
        
        # Create quality-controlled trend collector
        quality_trend_collector = EvaluatorOptimizerLLM(
            optimizer=trend_collector,
            evaluator=trend_evaluator,
            llm_factory=OpenAIAugmentedLLM,
            min_rating=QualityRating.EXCELLENT,
        )
        
        # Pattern Recognition Analyst
        pattern_analyst = Agent(
            name="pattern_recognition_analyst",
            instruction=f"""You are a business pattern recognition specialist.
            
            Analyze trend patterns for: {areas_str}
            Time frame: {time_horizon}
            
            Pattern analysis focus:
            1. Cross-industry trend correlations and convergences
            2. Cyclical patterns and seasonal variations
            3. Disruptive vs. evolutionary trend classification
            4. Technology adoption S-curves and inflection points
            5. Market maturity indicators and lifecycle stages
            
            Advanced pattern recognition:
            - Identify trend intersections and synergies
            - Map cause-and-effect relationships
            - Detect weak signals and early indicators
            - Classify trend types (macro, micro, mega-trends)
            - Assess trend durability and sustainability
            
            Provide insights on:
            - Which trends are accelerating/decelerating
            - Trend interdependencies and network effects
            - Critical inflection points and timing
            - Pattern-based predictions and projections
            - Strategic implications of pattern combinations
            
            Support findings with quantitative analysis where possible.""",
            server_names=["fetch"]
        )
        
        # Opportunity Detector
        opportunity_detector = Agent(
            name="opportunity_detector",
            instruction=f"""You are a business opportunity detection expert.
            
            Identify strategic opportunities from trend analysis of: {areas_str}
            Planning horizon: {time_horizon}
            
            Opportunity detection framework:
            1. Market gaps and unmet needs revealed by trends
            2. Technology convergences creating new possibilities
            3. Demographic shifts opening new markets
            4. Regulatory changes enabling new business models
            5. Investment flows indicating emerging opportunities
            
            For each opportunity, analyze:
            - Market size potential and addressable market
            - Competitive landscape and entry barriers
            - Required capabilities and resource needs
            - Risk factors and mitigation strategies
            - Timeline to market and development phases
            - Revenue models and monetization paths
            
            Opportunity classification:
            - HIGH IMPACT: Large market potential, strategic significance
            - MEDIUM IMPACT: Substantial potential, good ROI prospects  
            - LOW IMPACT: Niche potential, limited scope
            
            Prioritize opportunities by:
            - Market size and growth potential
            - Competitive advantage sustainability
            - Implementation feasibility
            - Strategic fit and synergies
            - Risk-adjusted returns
            
            Provide actionable opportunity briefs with clear next steps.""",
            server_names=["fetch"]
        )
        
        # Strategic Implications Analyzer
        strategy_analyzer = Agent(
            name="strategic_implications_analyzer",
            instruction=f"""You are a strategic business implications analyst.
            
            Analyze strategic implications of trends and opportunities for: {areas_str}
            Strategic timeframe: {time_horizon}
            
            Strategic analysis dimensions:
            1. Competitive positioning implications
            2. Business model evolution requirements
            3. Capability building priorities
            4. Investment and resource allocation
            5. Partnership and ecosystem strategies
            
            Strategic framework analysis:
            - How trends reshape competitive dynamics
            - What new capabilities become critical
            - Which business models become obsolete/advantaged
            - Where to invest for maximum strategic impact
            - How to build sustainable competitive advantages
            
            Strategic recommendations:
            - Immediate actions (0-6 months)
            - Medium-term initiatives (6-18 months)
            - Long-term strategic positioning (18+ months)
            - Risk mitigation strategies
            - Success metrics and KPIs
            
            Provide clear strategic roadmap with:
            - Priority initiatives and sequencing
            - Resource requirements and ROI projections
            - Key milestones and decision points
            - Contingency plans for different scenarios
            - Implementation recommendations
            
            Focus on actionable strategic insights that drive competitive advantage.""",
            server_names=["fetch"]
        )
        
        # Comprehensive Report Generator
        report_generator = Agent(
            name="trend_report_generator",
            instruction=f"""You are a comprehensive trend analysis report writer.
            
            Create a professional trend analysis report for: {areas_str}
            Time horizon: {time_horizon}
            
            Report structure:
            1. Executive Summary
               - Key trends overview
               - Top opportunities identified
               - Strategic recommendations summary
            
            2. Trend Analysis Deep Dive
               - Major trends with supporting data
               - Pattern analysis and correlations
               - Trend trajectory projections
            
            3. Opportunity Assessment
               - Prioritized opportunity matrix
               - Market sizing and potential
               - Competitive landscape analysis
            
            4. Strategic Implications
               - Business model impacts
               - Capability requirements
               - Investment priorities
            
            5. Action Plan & Roadmap
               - Immediate actions (0-6 months)
               - Medium-term initiatives (6-18 months)
               - Long-term strategic moves (18+ months)
            
            6. Risk Assessment
               - Trend risks and uncertainties
               - Mitigation strategies
               - Scenario planning
            
            7. Monitoring & Metrics
               - Key indicators to track
               - Early warning signals
               - Success metrics
            
            8. Data Sources & References
               - All sources with URLs
               - Data quality assessments
               - Research methodology
            
            Report requirements:
            - Professional markdown formatting
            - Clear visualizable data and metrics
            - Actionable insights throughout
            - Executive-ready presentation quality
            - Comprehensive source citations
            
            Save the complete report to: {output_path}""",
            server_names=["filesystem"]
        )
        
        return [quality_trend_collector, pattern_analyst, opportunity_detector, strategy_analyzer, report_generator]
    
    async def _create_trend_analysis_task(self, focus_areas: List[str], time_horizon: str, output_path: str) -> str:
        """Create comprehensive trend analysis task"""
        
        areas_str = ", ".join(focus_areas)
        
        task = f"""Execute comprehensive business trend analysis for: {areas_str}
        
        Analysis Timeframe: {time_horizon}
        
        Mission: Conduct deep trend analysis to identify strategic business opportunities 
        and provide actionable strategic recommendations.
        
        Analysis Workflow:
        
        1. HIGH-QUALITY TREND RESEARCH (quality_trend_collector):
           - Collect comprehensive trend data from authoritative sources
           - Quality evaluation ensures only EXCELLENT rated trends proceed
           - Focus on quantifiable trends with strong supporting evidence
           - Document growth trajectories and momentum indicators
        
        2. PATTERN RECOGNITION & ANALYSIS (pattern_recognition_analyst):
           - Identify cross-industry correlations and convergences
           - Map trend intersections and synergistic effects
           - Detect weak signals and early market indicators
           - Classify trends by type, durability, and impact
        
        3. OPPORTUNITY IDENTIFICATION (opportunity_detector):
           - Extract strategic business opportunities from trend patterns
           - Assess market potential and competitive landscapes
           - Prioritize opportunities by impact and feasibility
           - Develop opportunity briefs with implementation pathways
        
        4. STRATEGIC IMPLICATIONS (strategic_implications_analyzer):
           - Analyze how trends reshape competitive dynamics
           - Identify required capabilities and business model changes
           - Develop strategic roadmap with phased recommendations
           - Provide resource allocation and investment guidance
        
        5. COMPREHENSIVE REPORTING (trend_report_generator):
           - Synthesize all analysis into executive-ready report
           - Include actionable roadmap with clear timelines
           - Provide monitoring framework and success metrics
           - Save professional report to: {output_path}
        
        Success Criteria:
        - High-quality trends with strong supporting evidence
        - Clear identification of strategic opportunities
        - Actionable strategic recommendations with timelines
        - Comprehensive monitoring and success metrics
        - Professional report suitable for executive decision-making
        
        Deliver strategic intelligence that enables competitive advantage."""
        
        return task


# Factory function for easy instantiation
async def create_trend_analyzer(output_dir: str = "business_strategy_reports") -> TrendAnalyzerMCPAgent:
    """Create and return a TrendAnalyzerMCPAgent instance"""
    return TrendAnalyzerMCPAgent(output_dir=output_dir)


# Main execution function
async def run_trend_analysis(focus_areas: List[str], time_horizon: str = "6_months", 
                           output_dir: str = "business_strategy_reports") -> Dict[str, Any]:
    """Run trend analysis with specified parameters"""
    
    analyzer_agent = await create_trend_analyzer(output_dir)
    return await analyzer_agent.run_trend_analysis(focus_areas, time_horizon)


# CLI execution
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python trend_analyzer_agent.py 'area1,area2' [time_horizon]")
        print("Example: python trend_analyzer_agent.py 'AI,fintech,sustainability' '12_months'")
        print("Time horizons: 3_months, 6_months, 12_months, 24_months")
        sys.exit(1)
    
    focus_areas = [area.strip() for area in sys.argv[1].split(',')]
    time_horizon = sys.argv[2] if len(sys.argv) > 2 else "6_months"
    
    # Validate time horizon
    valid_horizons = ["3_months", "6_months", "12_months", "24_months"]
    if time_horizon not in valid_horizons:
        print(f"Invalid time horizon: {time_horizon}")
        print(f"Valid options: {', '.join(valid_horizons)}")
        sys.exit(1)
    
    # Run the trend analysis
    result = asyncio.run(run_trend_analysis(focus_areas, time_horizon))
    
    if result["success"]:
        print(f"✅ Trend analysis completed successfully!")
        print(f"📄 Report saved to: {result['output_file']}")
        print(f"🔍 Focus areas: {', '.join(result['focus_areas'])}")
        print(f"⏰ Time horizon: {result['time_horizon']}")
    else:
        print(f"❌ Trend analysis failed: {result['error']}") 