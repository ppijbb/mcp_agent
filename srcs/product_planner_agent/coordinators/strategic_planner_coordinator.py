"""Strategic Planner Coordinator
디자인 분석, PRD 작성, 비즈니스 전략, KPI, 마케팅 전략 등 전략 기획 단계를 담당합니다.
"""
from typing import Dict, Any
from mcp_agent.logging.logger import get_logger
import os

from srcs.product_planner_agent.agents.figma_analyzer_agent import FigmaAnalyzerAgent
from srcs.product_planner_agent.agents.prd_writer_agent import PRDWriterAgent
from srcs.product_planner_agent.agents.business_planner_agent import BusinessPlannerAgent
from srcs.product_planner_agent.agents.kpi_analyst_agent import KPIAnalystAgent
from srcs.product_planner_agent.agents.marketing_strategist_agent import MarketingStrategistAgent

logger = get_logger("strategic_planner_coordinator")

class StrategicPlannerCoordinator:
    """전략 기획 팀을 조율하는 Coordinator"""

    def __init__(self, orchestrator_factory=None):
        llm = orchestrator_factory() if orchestrator_factory else None
        # Figma URL은 실제 사용 시 외부 입력을 받아야 함. 여기서는 placeholder
        self.figma_analyzer = FigmaAnalyzerAgent(figma_url="https://www.figma.com/file/mock")
        self.prd_writer = PRDWriterAgent(output_path="outputs/product_planner/prd_output.md")
        self.business_planner = BusinessPlannerAgent(llm=llm)
        self.kpi_analyst = KPIAnalystAgent(llm=llm)
        self.marketing_strategist = MarketingStrategistAgent(llm=llm)

    async def create_strategic_plan(self, market_results: Dict[str, Any]) -> Dict[str, Any]:
        """시장 조사 결과를 기반으로 전략 기획 전 과정을 수행하고 결과를 종합합니다."""
        results: Dict[str, Any] = {}

        # 1) 디자인 분석
        figma_api_key = os.getenv("FIGMA_API_KEY")
        figma_file_id = os.getenv("FIGMA_FILE_ID")
        figma_node_id = os.getenv("FIGMA_NODE_ID")

        figma_analysis = await self.figma_analyzer.analyze_figma_for_prd(
            figma_api_key=figma_api_key,
            figma_file_id=figma_file_id,
            figma_node_id=figma_node_id
        )
        results["figma_analysis"] = figma_analysis

        # 2) PRD 작성
        prd = await self.prd_writer.write_prd(figma_analysis_result=figma_analysis)
        results["prd"] = prd

        # 3) 비즈니스 계획
        business_plan = await self.business_planner.create_business_plan(prd_content=prd)
        results["business_plan"] = business_plan

        # 4) KPI 정의
        kpis = await self.kpi_analyst.define_kpis(prd_content=prd, business_plan=business_plan)
        results["kpi"] = kpis

        # 5) 마케팅 전략
        marketing = await self.marketing_strategist.develop_marketing_strategy(prd_content=prd, business_plan=business_plan)
        results["marketing_strategy"] = marketing

        return results 