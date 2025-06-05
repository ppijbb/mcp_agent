#!/usr/bin/env python3
"""
Researcher Agent v2 - Refactored using Common Modules

Demonstrates how to use the common modules for cleaner, more maintainable agent code.
"""

import sys
import os
import asyncio
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define fallback constants first
DEFAULT_SERVERS = ["filesystem", "fetch"]
DEFAULT_COMPANY_NAME = "TechCorp Inc."

def get_output_dir(prefix, name):
    return f"{prefix}_{name}_reports"

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Import common modules
try:
    from srcs.common import *
    # Override with common module values if available
    if 'DEFAULT_SERVERS' in globals():
        pass  # Use the one from common
    if 'DEFAULT_COMPANY_NAME' in globals():
        pass  # Use the one from common
except ImportError:
    # Fallback direct imports
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

class ResearcherAgent:
    """Research agent for comprehensive information gathering and analysis"""
    
    def __init__(self, research_topic="AI and machine learning trends"):
        self.agent_name = "researcher_v2"
        self.research_topic = research_topic
        self.company_name = DEFAULT_COMPANY_NAME
        self.output_dir = get_output_dir("research", "reports")
        self.timestamp = get_timestamp()
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.task_description = f"""You are a research specialist focused on gathering comprehensive information.
        
        Research the topic: {research_topic}
        
        Your research should include:
        1. Current state and trends analysis
        2. Key players and organizations
        3. Recent developments and innovations
        4. Future projections and implications
        5. Actionable insights and recommendations
        
        Provide well-structured, accurate, and insightful research findings.
        """
    
    def run_research_workflow(self, topic=None, focus=None, save_to_file=False):
        """
        Run research workflow synchronously for Streamlit integration
        
        Args:
            topic: Research topic to investigate
            focus: Research focus area
            save_to_file: Whether to save results to files (default: False)
        
        Returns:
            dict: Results of the execution with actual content
        """
        if topic:
            self.research_topic = topic
            
        try:
            # Run the async main function
            result = asyncio.run(self._async_workflow(topic, focus, save_to_file))
            return {
                'success': True,
                'message': 'Research workflow completed successfully',
                'topic': self.research_topic,
                'focus': focus or 'comprehensive analysis',
                'output_dir': self.output_dir if save_to_file else None,
                'timestamp': self.timestamp,
                'content': result,  # 실제 생성된 콘텐츠
                'save_to_file': save_to_file
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error during research workflow execution: {str(e)}',
                'error': str(e),
                'topic': self.research_topic,
                'focus': focus,
                'save_to_file': save_to_file
            }
    
    async def _async_workflow(self, topic, focus, save_to_file=False):
        """Internal async workflow execution"""
        
        # Setup MCP application
        app = MCPApp(
            name=f"{self.agent_name}_system",
            settings=get_settings("configs/mcp_agent.config.yaml"),
            human_input_callback=None
        )
        
        async with app.run() as research_app:
            context = research_app.context
            logger = research_app.logger
            
            # Configure servers only if saving to file
            if save_to_file and "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
                logger.info("Filesystem server configured")
            
            # Create agents
            agents = self.create_agents()
            evaluator = self.create_evaluator()
            
            # Create orchestrator
            orchestrator = Orchestrator(
                llm_factory=OpenAIAugmentedLLM,
                available_agents=agents + [evaluator],
                plan_type="full",
            )
            
            # Update task description if focus is specified
            task = self.define_task(focus, save_to_file)
            
            # Execute the workflow
            logger.info(f"Starting {self.agent_name} workflow for topic: {self.research_topic}")
            
            result = await orchestrator.generate_str(
                message=task,
                request_params=RequestParams(model="gpt-4o-mini")
            )
            
            logger.info(f"{self.agent_name} workflow completed successfully")
            if save_to_file:
                logger.info(f"Research deliverables saved in {self.output_dir}/")
            else:
                logger.info("Results returned for display (not saved to file)")
            
            return result
    
    def create_agents(self):
        """Create research-specific agents"""
        return [
            Agent(
                name="trend_researcher",
                instruction=f"""Research current trends and developments in {self.research_topic}.
                
                Focus on:
                - Latest technological advances
                - Market trends and adoption patterns
                - Industry expert opinions and predictions
                - Statistical data and market reports
                
                Provide comprehensive trend analysis with supporting data.
                """,
                server_names=DEFAULT_SERVERS,
            ),
            Agent(
                name="competitive_researcher", 
                instruction=f"""Research key players and competitive landscape for {self.research_topic}.
                
                Analyze:
                - Leading companies and organizations
                - Market share and positioning
                - Strategic initiatives and investments
                - Competitive advantages and differentiators
                
                Provide detailed competitive intelligence report.
                """,
                server_names=DEFAULT_SERVERS,
            ),
            Agent(
                name="future_researcher",
                instruction=f"""Research future implications and projections for {self.research_topic}.
                
                Explore:
                - Future technology developments
                - Potential disruptions and opportunities
                - Long-term market projections
                - Societal and economic impacts
                
                Provide forward-looking analysis and strategic implications.
                """,
                server_names=DEFAULT_SERVERS,
            )
        ]
    
    def create_evaluator(self):
        """Create research quality evaluator"""
        return Agent(
            name="research_quality_evaluator",
            instruction="""Evaluate research quality and comprehensiveness.
            
            Assess based on:
            1. Information Accuracy (30%)
               - Factual correctness and source reliability
               - Data validity and recency
               - Bias identification and mitigation
            
            2. Comprehensiveness (25%)
               - Coverage breadth and depth
               - Multiple perspective inclusion
               - Gap identification and acknowledgment
            
            3. Analysis Quality (25%)
               - Insight depth and originality
               - Pattern recognition and synthesis
               - Actionable conclusions and recommendations
            
            4. Presentation Quality (20%)
               - Clarity and organization
               - Supporting evidence and citations
               - Executive summary effectiveness
            
            Provide EXCELLENT, GOOD, FAIR, or POOR rating with specific improvement recommendations.
            """,
            server_names=DEFAULT_SERVERS,
        )
    
    def define_task(self, focus=None, save_to_file=False):
        """Define comprehensive research task"""
        
        focus_instruction = ""
        if focus and focus != 'comprehensive analysis':
            focus_instruction = f"\n\nSpecial focus on: {focus}"
            if focus == "트렌드 분석":
                focus_instruction += "\nEmphasize current trends, emerging patterns, and future directions."
            elif focus == "경쟁 분석":
                focus_instruction += "\nEmphasize competitive landscape, market positioning, and strategic advantages."
            elif focus == "미래 전망":
                focus_instruction += "\nEmphasize future projections, potential disruptions, and strategic implications."
            elif focus == "시장 조사":
                focus_instruction += "\nEmphasize market size, growth rates, customer segments, and opportunities."
        
        task = f"""Execute comprehensive research project on: {self.research_topic}{focus_instruction}

        1. Use trend_researcher to analyze:
           - Current state and latest developments
           - Market trends and adoption patterns
           - Technology advances and innovations
           - Industry expert insights and predictions
           
        2. Use competitive_researcher to examine:
           - Key players and market leaders
           - Competitive landscape and positioning
           - Strategic initiatives and investments
           - Market share and performance metrics
           
        3. Use future_researcher to explore:
           - Future technology roadmaps
           - Potential disruptions and opportunities  
           - Long-term market projections
           - Strategic implications and recommendations
        
        """
        
        # Add file saving instructions only if save_to_file is True
        if save_to_file:
            task += f"""Compile findings into comprehensive research report saved in {self.output_dir}/:
        - trend_analysis_{self.timestamp}.md
        - competitive_landscape_{self.timestamp}.md
        - future_projections_{self.timestamp}.md
        - research_executive_summary_{self.timestamp}.md
        """
        else:
            task += """Return the complete research findings for immediate display. Do not save to files.
        Provide comprehensive, detailed results that can be displayed directly including:
        - Executive Summary with key findings
        - Trend Analysis with current developments
        - Competitive Landscape analysis
        - Future Projections and strategic recommendations
        """
        
        task += "\nProvide actionable insights and strategic recommendations for decision-makers."
        
        return task
    
    async def main(self):
        """Main execution method"""
        return await self._async_workflow(None, None)


async def main():
    """
    Researcher Agent v2 Main Function
    
    Demonstrates comprehensive research capabilities for:
    - Trend analysis and market research
    - Competitive intelligence gathering
    - Future projections and strategic insights
    - Structured output generation
    """
    
    agent = ResearcherAgent()
    await agent.main()


if __name__ == "__main__":
    asyncio.run(main()) 