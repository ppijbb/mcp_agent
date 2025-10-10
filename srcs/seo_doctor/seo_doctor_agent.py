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

# SEO Doctor 모듈
from .config_loader import seo_config
from .ai_seo_analyzer import SEOAIAnalyzer
from .lighthouse_analyzer import PlaywrightLighthouseAnalyzer

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
    SEO Doctor Agent with AI-powered analysis
    Gemini 2.5 Flash를 활용한 독립적 SEO 진단
    """
    
    def __init__(self):
        # 설정에서 서버 정보 로드
        mcp_servers = seo_config.get_mcp_servers_config()
        server_names = [mcp_servers['puppeteer']['server_name'], mcp_servers['g_search']['server_name']]
        
        super().__init__(
            name="SEODoctorAgent",
            instruction="Performs comprehensive SEO analysis with AI-powered insights using Gemini 2.5 Flash.",
            server_names=server_names
        )
        
        # AI 분석기 초기화
        self.ai_analyzer = SEOAIAnalyzer()
        self.lighthouse_analyzer = PlaywrightLighthouseAnalyzer()
        
        # 설정에서 출력 디렉토리 로드
        logging_config = seo_config.get_logging_config()
        self.output_dir = logging_config.get('output_dir', 'seo_doctor_reports')
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def run_workflow(self, url: str, keywords: List[str]):
        """
        AI-powered SEO analysis workflow with independent agent judgment
        각 에이전트가 독립적으로 판단하고 동작
        """
        try:
            self.logger.info(f"Starting AI-powered SEO analysis for: {url}")
            
            # 1. Lighthouse 분석 실행 (독립적 판단)
            self.logger.info("Step 1: Running Lighthouse analysis...")
            lighthouse_data = await self.lighthouse_analyzer.analyze_website(url, strategy="mobile")
            
            if not lighthouse_data or lighthouse_data.get('error'):
                raise Exception(f"Lighthouse 분석 실패: {lighthouse_data.get('error', 'Unknown error')}")
            
            # 2. AI 기반 Lighthouse 데이터 분석 (독립적 판단)
            self.logger.info("Step 2: Analyzing Lighthouse data with AI...")
            lighthouse_ai_analysis = await self.ai_analyzer.analyze_lighthouse_data(lighthouse_data)
            
            # 3. 경쟁사 분석 (선택적, 독립적 판단)
            competitor_analyses = []
            if keywords:
                self.logger.info(f"Step 3: Analyzing competitors for keywords: {', '.join(keywords)}...")
                # 여기서는 MCP orchestrator를 통한 경쟁사 검색
                async with self.app.run() as app_context:
                    competitor_agent = MCP_Agent(
                        name="CompetitorResearcher",
                        instruction=f"Search for top competitors using keywords '{', '.join(keywords)}' and return their URLs and basic SEO info using 'g-search' tool.",
                        server_names=["g-search"],
                        llm_factory=app_context.llm_factory,
                    )
                    
                    competitor_task = f"Find top 5 competitors for keywords: {', '.join(keywords)}"
                    competitor_result = await competitor_agent.run(competitor_task)
                    
                    # AI로 경쟁사 분석
                    competitor_analyses = await self.ai_analyzer.analyze_competitors(
                        [{"url": url, "data": competitor_result}], 
                        url
                    )
            
            # 4. AI 기반 통합 처방전 생성 (독립적 판단)
            self.logger.info("Step 4: Generating AI-powered SEO prescription...")
            prescription = await self.ai_analyzer.generate_seo_prescription(
                lighthouse_ai_analysis,
                competitor_analyses,
                url
            )
            
            # 5. 최종 결과 구조화
            final_result = {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "lighthouse_data": lighthouse_data,
                "ai_analysis": lighthouse_ai_analysis,
                "competitor_analyses": competitor_analyses,
                "prescription": prescription,
                "emergency_level": lighthouse_data.get('emergency_level', '⚠️ 위험'),
                "overall_score": lighthouse_data.get('overall_score', 0),
                "recovery_days": lighthouse_data.get('recovery_days', 30)
            }
            
            self.logger.info("SEO analysis completed successfully")
            return final_result
            
        except Exception as e:
            self.logger.error(f"SEO analysis workflow error: {e}")
            raise

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