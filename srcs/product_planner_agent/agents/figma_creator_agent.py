"""
Figma Creator Agent
PRD 요구사항을 바탕으로 Figma에서 직접 디자인을 생성하는 Agent
"""
import asyncio
from typing import Dict, Any, List

from srcs.core.agent.base import BaseAgent
from srcs.core.errors import WorkflowError
from srcs.product_planner_agent.integrations.figma_integration import create_rectangles_on_canvas, RectangleParams
from srcs.product_planner_agent.utils.logger import get_product_planner_logger


logger = get_product_planner_logger("agent.figma_creator")


class FigmaCreatorAgent(BaseAgent):
    """Figma 디자인 생성 전문 Agent"""

    def __init__(self, **kwargs):
        super().__init__("figma_creator_agent", **kwargs)
        logger.info("FigmaCreatorAgent initialized.")

    async def run_workflow(self, context: Any) -> Dict[str, Any]:
        """
        컨텍스트에서 받은 요구사항에 따라 Figma에 사각형을 생성합니다.
        
        필수 컨텍스트:
        - figma_file_key: Figma 파일 키
        - figma_parent_node_id: 디자인을 추가할 부모 노드 ID
        - rectangles: 생성할 사각형 정보 리스트 (List[RectangleParams])
        """
        file_key = context.get("figma_file_key")
        parent_node_id = context.get("figma_parent_node_id")
        rectangles_to_create: List[RectangleParams] = context.get("rectangles")

        if not all([file_key, parent_node_id, rectangles_to_create]):
            raise WorkflowError("Figma file_key, parent_node_id, rectangles 데이터는 필수입니다.")

        logger.info(f"🎨 Figma 생성 작업을 시작합니다. 대상 파일: {file_key}")

        try:
            # 동기 함수인 create_rectangles_on_canvas를 비동기적으로 실행합니다.
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,  # 기본 스레드 풀 사용
                create_rectangles_on_canvas,
                file_key,
                parent_node_id,
                rectangles_to_create
            )
            
            logger.info("✅ Figma 생성 작업이 성공적으로 완료되었습니다.")
            
            final_result = {
                "status": "success",
                "created_nodes": result.get("nodes", {})
            }
            context.set("figma_creation_result", final_result)
            return final_result
        except Exception as e:
            logger.error(f"Figma 생성 워크플로우 실패: {e}", exc_info=True)
            raise WorkflowError(f"Figma 생성 실패: {e}") from e

    # --- 기존 정적 메소드 (AgentFactory 등에서 사용될 수 있으므로 유지) ---
    
    @staticmethod
    def get_description() -> str:
        """Agent 설명 반환"""
        return "🖌️ PRD 요구사항을 바탕으로 Figma에서 직접 디자인을 생성하는 Agent"
    
    @staticmethod
    def get_capabilities() -> list[str]:
        """Agent 주요 기능 목록 반환"""
        return [
            "Figma 사각형, 텍스트 등 기본 요소 생성",
            "Figma REST API를 통한 디자인 자동화",
            "요구사항 기반 목업 생성"
        ]
    
    @staticmethod
    def get_creation_tools() -> dict[str, list[str]]:
        """생성 도구 목록 반환 (이제는 내부 함수 호출로 대체됨)"""
        return {
            "node_creation": [
                "create_rectangles_on_canvas",
            ],
        }
    
    @staticmethod
    def get_design_process() -> list[str]:
        """디자인 프로세스 단계 반환"""
        return [
            "컨텍스트에서 디자인 요구사항 수신",
            "figma_integration을 통해 API 호출",
            "생성 결과 반환"
        ] 