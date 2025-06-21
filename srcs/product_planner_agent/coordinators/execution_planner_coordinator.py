"""Execution Planner Coordinator
개발·운영 실행 계획 단계를 담당합니다.
"""
from typing import Dict, Any
from mcp_agent.logging.logger import get_logger

from srcs.product_planner_agent.agents.project_manager_agent import ProjectManagerAgent
from srcs.product_planner_agent.agents.operations_agent import OperationsAgent

logger = get_logger("execution_planner_coordinator")

class ExecutionPlannerCoordinator:
    """실행 계획 팀을 조율하는 Coordinator"""

    def __init__(self, orchestrator_factory=None):
        llm = orchestrator_factory() if orchestrator_factory else None
        self.project_manager = ProjectManagerAgent(llm=llm)
        self.operations_agent = OperationsAgent(llm=llm)

    async def create_execution_plan(self, strategic_results: Dict[str, Any]) -> Dict[str, Any]:
        """프로젝트 관리 & 운영 계획을 생성합니다."""
        prd = strategic_results.get("prd", {})
        business_plan = strategic_results.get("business_plan", {})

        project_plan = await self.project_manager.create_project_plan(prd_content=prd, business_plan=business_plan)
        operations_plan = await self.operations_agent.plan_operations(prd_content=prd)

        return {
            "project_plan": project_plan,
            "operations_plan": operations_plan
        } 