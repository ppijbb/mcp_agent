"""Reporting Coordinator
초기 요구사항 수집 및 최종 보고서 생성을 담당합니다.
"""
from typing import Dict, Any, Optional
from mcp_agent.logging.logger import get_logger

from srcs.product_planner_agent.agents.conversation_agent import ConversationAgent
from srcs.product_planner_agent.agents.notion_document_agent import NotionDocumentAgent

logger = get_logger("reporting_coordinator")

class ReportingCoordinator:
    """대화 기반 요구사항 수집 & Notion 보고서 생성을 담당하는 Coordinator"""

    def __init__(self, orchestrator_factory=None):
        # LLM 팩토리가 주어지면 각 에이전트에 주입, 아니면 None
        llm = orchestrator_factory() if orchestrator_factory else None
        self.conversation_agent = ConversationAgent(llm=llm)
        self.notion_agent = NotionDocumentAgent(llm=llm)

    async def collect_initial_requirements(self, initial_prompt: str) -> Dict[str, Any]:
        """ConversationAgent를 통해 초기 요구사항을 수집합니다."""
        logger.info("🗣️ collect_initial_requirements 실행")
        return await self.conversation_agent.collect_requirements_via_chat(initial_prompt)

    async def generate_final_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """모든 결과물을 Notion 문서로 종합합니다."""
        logger.info("📄 generate_final_report 실행")
        return await self.notion_agent.create_project_workspace(all_results) 