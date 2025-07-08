"""Strategic Planner Coordinator
디자인 분석, PRD 작성, 비즈니스 전략, KPI, 마케팅 전략 등 전략 기획 단계를 담당합니다.
"""
from typing import Dict, Any, Optional
from mcp_agent.logging.logger import get_logger
from srcs.product_planner_agent.utils import env_settings as env
import asyncio

from srcs.product_planner_agent.agents.figma_analyzer_agent import FigmaAnalyzerAgent
from srcs.product_planner_agent.agents.prd_writer_agent import PRDWriterAgent
from srcs.product_planner_agent.agents.business_planner_agent import BusinessPlannerAgent
from srcs.product_planner_agent.agents.kpi_analyst_agent import KPIAnalystAgent
from srcs.product_planner_agent.agents.marketing_strategist_agent import MarketingStrategistAgent
from srcs.product_planner_agent.agents.coordinator_agent import CoordinatorAgent
from srcs.product_planner_agent.agents.orchestrator import Orchestrator
from srcs.product_planner_agent.agents.base_coordinator import BaseCoordinator

logger = get_logger("strategic_planner_coordinator")

class StrategicPlannerCoordinator(BaseCoordinator):
    """
    Strategic Planner: High-level coordination of the product planning process.
    """

    def __init__(self, 
                 llm_provider: str = 'openai', 
                 model_name: str = 'gpt-4-turbo-preview',
                 google_drive_mcp_url: str = "http://localhost:3001",
                 figma_mcp_url: str = "http://localhost:3003",
                 notion_mcp_url: str = "http://localhost:3004"):
        super().__init__(llm_provider, model_name)
        self.google_drive_mcp_url = google_drive_mcp_url
        self.figma_mcp_url = figma_mcp_url
        self.notion_mcp_url = notion_mcp_url

    async def develop_product_plan(self, 
                                   product_concept: str, 
                                   user_persona: str,
                                   figma_file_id: Optional[str] = None,
                                   notion_page_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Develops a comprehensive product plan by orchestrating the CoordinatorAgent.
        """
        orchestrator = self.create_orchestrator()
        
        coordinator_agent = CoordinatorAgent(
            orchestrator=orchestrator,
            google_drive_mcp_url=self.google_drive_mcp_url,
            figma_mcp_url=self.figma_mcp_url,
            notion_mcp_url=self.notion_mcp_url
        )

        # The core logic now calls the refactored coordinator method
        final_result = await coordinator_agent.coordinate_prd_creation(
            product_concept=product_concept,
            user_persona=user_persona,
            figma_file_id=figma_file_id,
            notion_page_id=notion_page_id
        )
        
        return final_result

    def create_orchestrator(self) -> Orchestrator:
        # ... (method remains the same) ...
        pass 