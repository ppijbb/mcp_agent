"""Market Research Coordinator
시장 조사 및 경쟁 분석 단계를 담당합니다.
"""
from typing import Dict, Any
from mcp_agent.logging.logger import get_logger

from srcs.product_planner_agent.agents.market_research_agent import MarketResearchAgent

logger = get_logger("market_research_coordinator")

class MarketResearchCoordinator:
    """시장 조사 팀을 조율하는 Coordinator"""

    def __init__(self, orchestrator_factory=None):
        llm = orchestrator_factory() if orchestrator_factory else None
        self.market_agent = MarketResearchAgent(llm=llm)
        # 경쟁 분석 Agent는 추후 추가 가능

    async def perform_market_research(self, convo_results: Dict[str, Any]) -> Dict[str, Any]:
        """Conversation 단계에서 얻은 제품 컨텍스트 기반 시장 조사를 수행합니다."""
        product_context = {
            "user_requirements": convo_results
        }
        logger.info("📊 Market research 수행")
        return await self.market_agent.analyze_market(product_context) 