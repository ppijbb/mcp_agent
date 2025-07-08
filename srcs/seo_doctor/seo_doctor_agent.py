import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

# Real MCP Agent imports
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from srcs.common.utils import setup_agent_app
from srcs.core.agent.base import BaseAgent
from mcp_agent.agents.agent import Agent as MCP_Agent

# Keep existing enums and data classes as they are well-defined
class SEOEmergencyLevel(Enum):
    """SEO Emergency Level Classification"""
    CRITICAL = "🚨 응급실"
    HIGH = "⚠️ 위험"
    MEDIUM = "⚡ 주의"
    LOW = "✅ 안전"
    EXCELLENT = "🚀 완벽"

class CompetitorThreatLevel(Enum):
    """Competitor Threat Assessment"""
    DOMINATING = "👑 지배중"
    RISING = "📈 급상승"
    STABLE = "➡️ 안정"
    DECLINING = "📉 하락"
    WEAK = "😴 약함"

@dataclass
class SEOAnalysisResult:
    """Real SEO Analysis Result - No Mock Data"""
    url: str
    emergency_level: SEOEmergencyLevel
    overall_score: float  # From real Lighthouse analysis
    performance_score: float
    seo_score: float
    accessibility_score: float
    best_practices_score: float
    core_web_vitals: Dict[str, Any]
    critical_issues: List[str]
    quick_fixes: List[str]
    estimated_recovery_days: int
    competitor_analysis: List[Dict[str, Any]]
    recommendations: List[str]
    analysis_timestamp: datetime
    lighthouse_raw_data: Dict[str, Any]  # Full Lighthouse report

@dataclass
class SEOPrescription:
    """SEO Treatment Prescription"""
    prescription_id: str
    patient_url: str
    emergency_treatment: List[str]
    weekly_medicine: List[str]
    monthly_checkup: List[str]
    competitive_moves: List[str]
    expected_results: str
    follow_up_date: datetime
    implementation_priority: List[str]

class SEODoctorAgent(BaseAgent):
    """
    SEO Doctor Agent, refactored to inherit from BaseAgent.
    """
    
    def __init__(self):
        super().__init__(
            name="SEODoctorAgent",
            instruction="Performs a comprehensive SEO analysis of a given URL.",
            server_names=["puppeteer", "g-search"] # 이 에이전트가 사용할 서버 명시
        )
        self.output_dir = "seo_doctor_reports"
    
    async def run_workflow(self, url: str, keywords: List[str]):
        """
        The core SEO analysis workflow.
        """
        async with self.app.run() as app_context:
            self.logger.info(f"Starting SEO analysis workflow for: {url}")

            # 1. Define specialized agents required for the task
            lighthouse_agent = MCP_Agent(
                name="LighthouseAuditor",
                instruction=f"Your only job is to run a Lighthouse audit for the URL: {url} using the 'puppeteer' tool. Return only the raw JSON output.",
                server_names=["puppeteer"],
                llm_factory=app_context.llm_factory,
            )
            
            competitor_agent = MCP_Agent(
                name="CompetitorResearcher",
                instruction=f"Analyze the search engine results pages (SERPs) for the keywords '{', '.join(keywords)}' to identify top competitors and their strategies. Use the 'g-search' tool.",
                server_names=["g-search"],
                llm_factory=app_context.llm_factory,
            )

            synthesis_agent = MCP_Agent(
                name="SEOSynthesisExpert",
                instruction="You are an expert SEO analyst. Synthesize the provided Lighthouse audit data and competitor research into a comprehensive, structured SEO diagnosis. Provide actionable recommendations and a final prescription.",
                llm_factory=app_context.llm_factory,
            )

            # 2. Get an orchestrator to manage the workflow
            orchestrator = self.get_orchestrator(
                agents=[lighthouse_agent, competitor_agent, synthesis_agent]
            )

            # 3. Define the main task for the orchestrator
            main_task = f"""
            Perform a complete SEO diagnosis for the website: {url}.
            The process involves three main steps:
            1. Use the `LighthouseAuditor` to get the raw performance and SEO metrics.
            2. Use the `CompetitorResearcher` to understand the competitive landscape for keywords: {', '.join(keywords)}.
            3. Pass the data from both steps to the `SEOSynthesisExpert` to create the final JSON report.
            Execute these steps in order and return the final report.
            """
            
            # 4. Run the orchestrator
            final_report_str = await orchestrator.run(main_task)
            
            try:
                # 최종 결과를 JSON으로 파싱 시도
                return json.loads(final_report_str)
            except json.JSONDecodeError:
                self.logger.warning("Final report is not valid JSON. Returning as raw text.")
                return {"raw_report": final_report_str}

async def main():
    """Main function to demonstrate the refactored agent."""
    agent = SEODoctorAgent()
    analysis_result = await agent.run(
        url="https://www.anthropic.com/", 
        keywords=["AI assistant", "Claude AI"]
    )
    print("--- SEO Analysis Result ---")
    print(json.dumps(analysis_result, indent=2))
    print("--------------------------")

if __name__ == "__main__":
    asyncio.run(main())