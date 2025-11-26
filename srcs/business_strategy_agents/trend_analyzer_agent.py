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
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import QualityRating
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import EvaluatorOptimizerLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from srcs.common.utils import setup_agent_app
from srcs.core.agent.base import BaseAgent
from mcp_agent.agents.agent import Agent as MCP_Agent
import json


class TrendAnalyzerAgent(BaseAgent):
    """
    Business Trend Analyzer Agent, refactored to inherit from BaseAgent.
    Analyzes market and technology trends based on specified focus areas.
    """
    
    def __init__(self):
        super().__init__(
            name="TrendAnalyzerAgent",
            instruction="Analyzes market and technology trends to provide strategic insights.",
            server_names=["g-search", "fetch", "filesystem"]
        )
        self.output_dir = "business_strategy_reports"

    async def run_workflow(self, focus_areas: List[str], time_horizon: str):
        """
        The core workflow for analyzing business trends.
        각 에이전트가 독립적으로 판단하고 동작
        """
        async with self.app.run() as app_context:
            self.logger.info(f"Starting independent trend analysis for: {focus_areas}")
            
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"trend_analysis_report_{timestamp}.md"
            output_path = os.path.join(self.output_dir, output_file)

            try:
                # LLM factory: Gemini 모델을 사용하므로 GoogleAugmentedLLM 사용
                from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
                
                # 1. Define specialized sub-agents with independent judgment
                # Agent 생성 시 llm_factory는 람다로 전달
                def llm_factory_for_agents(**kwargs):
                    return GoogleAugmentedLLM(model="gemini-2.5-flash")
                
                agents = self._create_trend_agents(focus_areas, time_horizon, output_path, llm_factory_for_agents)
                
                # 2. Create orchestrator - 공식 예제처럼 클래스 자체를 전달
                from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
                orchestrator = Orchestrator(
                    llm_factory=GoogleAugmentedLLM,
                    available_agents=agents,
                    plan_type="full"
                )

                # 3. Define the main task with independent agent requirements
                task = self._create_analysis_task(focus_areas, time_horizon, output_path)

                # 4. Run the orchestrator - 공식 예제처럼 RequestParams는 model만 전달
                from mcp_agent.workflows.llm.augmented_llm import RequestParams
                try:
                    final_report = await orchestrator.generate_str(
                        message=task,
                        request_params=RequestParams(model="gemini-2.5-flash")
                    )
                except (BrokenPipeError, OSError) as pipe_error:
                    # EPIPE 또는 파이프 관련 에러 처리
                    error_code = getattr(pipe_error, 'errno', None)
                    if error_code == 32 or 'EPIPE' in str(pipe_error):
                        self.logger.error(f"MCP server process terminated unexpectedly (EPIPE). This may indicate the server crashed or was closed.")
                        raise RuntimeError(f"MCP server connection lost: {pipe_error}") from pipe_error
                    raise
                
                self.logger.info(f"Independent trend analysis complete. Report saved to {output_path}")
                return {"report_path": output_path, "content": final_report}
                
            except Exception as e:
                self.logger.error(f"Independent trend analysis failed: {e}", exc_info=True)
                raise

    def _create_trend_agents(self, focus_areas: List[str], time_horizon: str, output_path: str, llm_factory) -> List[Agent]:
        """Create specialized trend analysis agents"""
        
        areas_str = ", ".join(focus_areas)
        
        # Trend Data Collector - Enhanced with quality control
        trend_collector = MCP_Agent(
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
        trend_evaluator = MCP_Agent(
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
            llm_factory=llm_factory,
            min_rating=QualityRating.EXCELLENT,
        )
        
        # Pattern Recognition Analyst
        pattern_analyst = MCP_Agent(
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
        opportunity_detector = MCP_Agent(
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
        strategy_analyzer = MCP_Agent(
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
        report_generator = MCP_Agent(
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
    
    def _create_analysis_task(self, focus_areas: List[str], time_horizon: str, output_path: str) -> str:
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
async def create_trend_analyzer() -> TrendAnalyzerAgent:
    """Create and return a TrendAnalyzerMCPAgent instance"""
    return TrendAnalyzerAgent()


# Main execution function
async def run_trend_analysis(focus_areas: List[str], time_horizon: str):
    """Run the trend analyzer agent"""
    analyzer = TrendAnalyzerAgent()
    result = await analyzer.run(
        focus_areas=focus_areas,
        time_horizon=time_horizon
    )
    print(json.dumps(result, indent=2))

async def main():
    """Main function to demonstrate the agent."""
    agent = TrendAnalyzerAgent()
    result = await agent.run(
        focus_areas=["AI-driven drug discovery", "decentralized clinical trials"],
        time_horizon="next 24 months"
    )
    print(json.dumps(result, indent=2))

# CLI execution
if __name__ == "__main__":
    asyncio.run(main()) 