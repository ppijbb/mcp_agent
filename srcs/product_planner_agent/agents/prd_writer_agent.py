"""
PRD Writer Agent
디자인 분석 결과를 바탕으로 전문적인 제품 요구사항 문서를 작성하는 Agent
"""

import json
from datetime import datetime
from typing import Any, Dict

from srcs.product_planner_agent.agents.base_agent_simple import BaseAgentSimple as BaseAgent
from srcs.product_planner_agent.utils.logger import get_product_planner_logger

logger = get_product_planner_logger(__name__)

class PRDWriterAgent(BaseAgent):
    """
    Agent responsible for drafting the PRD document based on various inputs.
    """

    def __init__(self, **kwargs):
        super().__init__("prd_writer_agent")
        logger.info("PRDWriterAgent initialized.")

    async def run_workflow(self, context: Any) -> Dict[str, Any]:
        """
        Drafts the PRD using the product brief and feedback from the context.
        """
        logger.info("🖊️ Starting PRD generation workflow...")
        
        # 컨텍스트에서 데이터 추출
        product_concept = context.get("product_concept", "제품")
        user_persona = context.get("user_persona", "사용자")
        figma_analysis = context.get("figma_analysis", {})
        
        logger.info(f"Starting PRD draft for product: '{product_concept[:50]}...'")
        
        # 간단한 PRD 템플릿 생성
        prd_data = {
            "product_name": f"{product_concept} 제품",
            "version": "1.0",
            "created_date": datetime.now().isoformat(),
            "introduction": {
                "problem_statement": f"{product_concept}와 관련된 문제점을 해결하기 위한 제품",
                "goal": f"{product_concept} 사용자 경험 향상 및 효율성 증대",
                "target_audience": user_persona
            },
            "product_requirements": {
                "user_stories": [
                    f"사용자는 {product_concept} 기능을 쉽게 사용할 수 있어야 한다",
                    f"사용자는 직관적인 인터페이스를 통해 빠르게 작업을 완료할 수 있어야 한다",
                    f"사용자는 개인화된 설정을 통해 맞춤형 경험을 할 수 있어야 한다"
                ],
                "functional_requirements": [
                    "로그인/회원가입 기능",
                    "메인 기능 접근",
                    "설정 및 프로필 관리",
                    "데이터 저장 및 동기화"
                ],
                "non_functional_requirements": [
                    "응답 시간 3초 이내",
                    "99.9% 가용성",
                    "모바일 및 웹 호환성",
                    "보안 인증 및 암호화"
                ]
            },
            "design_ux": {
                "design_mockups": "Figma 디자인 프로토타입 참조",
                "user_flow": [
                    "시작 화면",
                    "로그인/회원가입",
                    "메인 대시보드",
                    "기능 사용",
                    "설정 및 프로필"
                ]
            },
            "assumptions_constraints": {
                "assumptions": [
                    "사용자는 기본적인 디지털 리터러시를 보유",
                    "인터넷 연결이 안정적",
                    "모바일 기기 사용 가능"
                ],
                "constraints": [
                    "기술적 제약사항",
                    "예산 제약",
                    "시간 제약"
                ]
            },
            "figma_analysis": figma_analysis
        }
        
        logger.info("PRD 생성 완료")
        return prd_data 