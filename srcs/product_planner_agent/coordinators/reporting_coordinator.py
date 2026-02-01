"""Reporting Coordinator
초기 요구사항 수집 및 최종 보고서 생성을 담당합니다.
"""
from typing import Dict, Any
from srcs.product_planner_agent.utils.logger import get_product_planner_logger

logger = get_product_planner_logger("coordinator.reporting")


class ReportingCoordinator:
    """대화 기반 요구사항 수집 & 보고서 생성을 담당하는 Coordinator"""

    def __init__(self):
        logger.info("ReportingCoordinator initialized.")

    async def collect_initial_requirements(self, initial_prompt: str) -> Dict[str, Any]:
        """초기 요구사항을 수집합니다."""
        logger.info("🗣️ collect_initial_requirements 실행")
        return {
            "status": "success",
            "requirements": initial_prompt,
            "collected_at": "2024-01-01T00:00:00Z"
        }

    async def generate_final_report(self, context: Any) -> Dict[str, Any]:
        """모든 결과물을 종합하여 최종 보고서를 생성합니다."""
        logger.info("📄 generate_final_report 실행")
        prd_draft = context.get("prd_draft", {})
        product_concept = context.get("product_concept", "제품")
        figma_creation_result = context.get("figma_creation_result", {})

        logger.info(f"Generating final report for: {product_concept}")

        # 최종 보고서 생성
        report = {
            "executive_summary": f"This report details the product plan for '{product_concept}'. The plan is based on an initial PRD and outlines the key features and vision.",
            "product_concept": product_concept,
            "prd_summary": prd_draft,
            "figma_components": figma_creation_result,
            "next_steps": [
                "Begin high-fidelity design based on PRD.",
                "Develop a detailed technical specification.",
                "Create a project plan and timeline.",
                "Implement Figma components in actual design system."
            ],
            "generated_at": "2024-01-01T00:00:00Z",
            "status": "completed"
        }

        logger.info("Final report successfully generated.")
        return report
