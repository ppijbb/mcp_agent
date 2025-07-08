"""Executive Coordinator
전체 워크플로우를 총괄하며 각 Phase Coordinator를 호출합니다.
"""
from typing import Dict, Any, Optional
from mcp_agent.logging.logger import get_logger
import re
from urllib.parse import unquote

from srcs.product_planner_agent.coordinators.reporting_coordinator import ReportingCoordinator
from srcs.product_planner_agent.coordinators.market_research_coordinator import MarketResearchCoordinator
from srcs.product_planner_agent.coordinators.strategic_planner_coordinator import StrategicPlannerCoordinator
from srcs.product_planner_agent.coordinators.execution_planner_coordinator import ExecutionPlannerCoordinator

# orchestrator utils
from srcs.product_planner_agent.utils.orchestrator_factory import orchestrator_factory as _default_orch_factory
from srcs.product_planner_agent.utils.retry import async_retry

logger = get_logger("executive_coordinator")

class ExecutiveCoordinator:
    """최상위 Coordinator. run()을 호출하면 전체 제품 기획 워크플로우가 수행됩니다."""

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

    # ---------------------------------------------
    # Turn / Step Budget Helpers
    # ---------------------------------------------
    def _consume_turns(self, n: int = 1):
        """간단한 전역 턴 예산 관리. n 만큼 차감 후 잔여 턴이 0 미만이면 예외 발생."""
        self.turn_budget -= n
        if self.turn_budget < 0:
            raise RuntimeError("Turn budget exhausted: AGENT_MAX_TURNS limit reached.")

    async def run_product_planning_workflow(self, 
                                            product_concept: str, 
                                            user_persona: str,
                                            figma_file_id: Optional[str] = None,
                                            notion_page_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs the full product planning workflow, from market research to final report.
        """
        logger.info("Starting complete product planning workflow...")
        
        # In this refactored version, we are focusing on the core PRD generation
        # The market research part can be a separate preliminary step.
        
        strategic_planner = StrategicPlannerCoordinator(
            llm_provider=self.llm_provider,
            model_name=self.model_name,
            google_drive_mcp_url=self.google_drive_mcp_url,
            figma_mcp_url=self.figma_mcp_url,
            notion_mcp_url=self.notion_mcp_url
        )
        
        # Develop the core product plan (which now includes PRD generation)
        plan_results = await strategic_planner.develop_product_plan(
            product_concept=product_concept,
            user_persona=user_persona,
            figma_file_id=figma_file_id,
            notion_page_id=notion_page_id
        )
        
        logger.info("Product planning workflow completed.")
        return {
            "status": "Workflow Complete",
            "results": plan_results
        } 