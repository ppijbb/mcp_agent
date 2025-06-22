"""Executive Coordinator
전체 워크플로우를 총괄하며 각 Phase Coordinator를 호출합니다.
"""
from typing import Dict, Any, Optional
from mcp_agent.logging.logger import get_logger

from srcs.product_planner_agent.coordinators.reporting_coordinator import ReportingCoordinator
from srcs.product_planner_agent.coordinators.market_research_coordinator import MarketResearchCoordinator
from srcs.product_planner_agent.coordinators.strategic_planner_coordinator import StrategicPlannerCoordinator
from srcs.product_planner_agent.coordinators.execution_planner_coordinator import ExecutionPlannerCoordinator

logger = get_logger("executive_coordinator")

class ExecutiveCoordinator:
    """최상위 Coordinator. run()을 호출하면 전체 제품 기획 워크플로우가 수행됩니다."""

    def __init__(self, orchestrator_factory=None):
        self.orchestrator_factory = orchestrator_factory  # 추후 LLM·툴 공유 목적
        self.reporting = ReportingCoordinator(orchestrator_factory=orchestrator_factory)
        self.market = MarketResearchCoordinator(orchestrator_factory=orchestrator_factory)
        self.strategic = StrategicPlannerCoordinator(orchestrator_factory=orchestrator_factory)
        self.execution = ExecutionPlannerCoordinator(orchestrator_factory=orchestrator_factory)

        import os
        self.turn_budget = int(os.getenv("AGENT_MAX_TURNS", 20))

    # ---------------------------------------------
    # Turn / Step Budget Helpers
    # ---------------------------------------------
    def _consume_turns(self, n: int = 1):
        """간단한 전역 턴 예산 관리. n 만큼 차감 후 잔여 턴이 0 미만이면 예외 발생."""
        self.turn_budget -= n
        if self.turn_budget < 0:
            raise RuntimeError("Turn budget exhausted: AGENT_MAX_TURNS limit reached.")

    async def run(self, initial_prompt: str) -> Dict[str, Any]:
        """전체 플로우를 순차적으로 실행하고 결과를 종합해 반환합니다."""
        results: Dict[str, Any] = {}

        # 1. 사용자 요구사항 수집 및 초기 보고
        self._consume_turns()
        logger.info("🔹 Phase: Reporting/Conversation 시작")
        convo_results = await self.reporting.collect_initial_requirements(initial_prompt)
        results["conversation"] = convo_results

        # 2. 시장 조사
        self._consume_turns()
        logger.info("🔹 Phase: Market Research 시작")
        market_results = await self.market.perform_market_research(convo_results)
        results["market_research"] = market_results

        # 3. 전략 기획(디자인 분석, PRD, 비즈니스 플랜)
        self._consume_turns()
        logger.info("🔹 Phase: Strategic Planning 시작")
        strategic_results = await self.strategic.create_strategic_plan(market_results)
        results.update(strategic_results)

        # 4. 실행 계획(프로젝트·운영)
        self._consume_turns()
        logger.info("🔹 Phase: Execution Planning 시작")
        execution_results = await self.execution.create_execution_plan(strategic_results)
        results.update(execution_results)

        # 5. 최종 보고서 작성
        self._consume_turns()
        logger.info("🔹 Phase: Final Reporting 시작")
        final_report = await self.reporting.generate_final_report(results)
        results["final_report"] = final_report

        logger.info("✅ ExecutiveCoordinator 모든 단계 완료")
        return results 