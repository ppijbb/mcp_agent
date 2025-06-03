#!/usr/bin/env python3
"""
Researcher Agent v2 - Refactored using Common Modules

Demonstrates how to use the common modules for cleaner, more maintainable agent code.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import *

class ResearcherAgent(BasicAgentTemplate):
    """Research agent using common template"""
    
    def __init__(self, research_topic="AI and machine learning trends"):
        super().__init__(
            agent_name="researcher_v2",
            task_description=f"""You are a research specialist focused on gathering comprehensive information.
            
            Research the topic: {research_topic}
            
            Your research should include:
            1. Current state and trends analysis
            2. Key players and organizations
            3. Recent developments and innovations
            4. Future projections and implications
            5. Actionable insights and recommendations
            
            Provide well-structured, accurate, and insightful research findings.
            """
        )
        self.research_topic = research_topic
    
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
        )
    
    def define_task(self):
        """Define comprehensive research task"""
        return f"""Execute comprehensive research project on: {self.research_topic}

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
        
        Compile findings into comprehensive research report saved in {self.output_dir}/:
        - trend_analysis_{self.timestamp}.md
        - competitive_landscape_{self.timestamp}.md
        - future_projections_{self.timestamp}.md
        - research_executive_summary_{self.timestamp}.md
        
        Provide actionable insights and strategic recommendations for decision-makers.
        """
    
    def create_summary(self):
        """Create research-specific executive summary"""
        summary_data = {
            "title": f"Research Analysis: {self.research_topic}",
            "overview": {
                "title": "Research Overview",
                "content": f"Comprehensive research analysis on {self.research_topic} completed with trend analysis, competitive intelligence, and future projections."
            },
            "impact_metrics": {
                "Information Coverage": "95%+ comprehensive analysis",
                "Source Reliability": "High-quality, verified sources",
                "Future Insights": "5-10 year strategic projections",
                "Actionable Recommendations": "Executive-level strategic guidance"
            },
            "initiatives": {
                "Trend Analysis": "Current state and development patterns",
                "Competitive Intelligence": "Market landscape and player analysis", 
                "Future Projections": "Strategic implications and opportunities"
            },
            "action_items": [
                "Review comprehensive research findings and recommendations",
                "Evaluate strategic implications for organizational planning",
                "Consider competitive positioning and market opportunities",
                "Develop action plan based on future projections"
            ],
            "next_steps": [
                "Executive review of research findings and strategic implications",
                "Cross-functional team discussion on market opportunities",
                "Strategic planning integration based on research insights",
                "Regular monitoring of identified trends and developments"
            ]
        }
        
        return create_executive_summary(
            output_dir=self.output_dir,
            agent_name="research",
            company_name=self.company_name,
            timestamp=self.timestamp,
            **summary_data
        )
    
    def create_kpis(self):
        """Create research-specific KPI template"""
        kpi_structure = {
            "research_quality_metrics": {
                "information_coverage": {
                    "source_diversity_score": "0/100",
                    "information_recency": "0% within 6 months",
                    "factual_accuracy_rate": "0%",
                    "bias_detection_score": "0/100"
                },
                "analysis_depth": {
                    "trend_identification_count": 0,
                    "competitive_insights_count": 0,
                    "future_projections_count": 0,
                    "actionable_recommendations": 0
                },
                "research_efficiency": {
                    "research_completion_time": "0 hours",
                    "source_validation_rate": "0%", 
                    "insight_per_hour_ratio": "0",
                    "stakeholder_satisfaction": "0/10"
                }
            }
        }
        
        return create_kpi_template(
            output_dir=self.output_dir,
            agent_name="research",
            kpi_structure=kpi_structure,
            timestamp=self.timestamp
        )

async def main():
    """Main execution function"""
    # You can customize the research topic here
    research_topic = input("Enter research topic (or press Enter for default): ").strip() or "AI and machine learning trends"
    
    researcher = ResearcherAgent(research_topic)
    success = await researcher.run()
    
    if success:
        print(f"\n✅ Research completed successfully!")
        print(f"📁 Results saved in: {researcher.output_dir}/")
        print(f"🔍 Research topic: {research_topic}")
    else:
        print("\n❌ Research failed. Check logs for details.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main()) 