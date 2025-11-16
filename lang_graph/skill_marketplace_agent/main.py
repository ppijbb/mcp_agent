"""
Skill Marketplace & Learning Coach Agent 메인 진입점
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from .llm.model_manager import ModelManager, ModelProvider
from .llm.fallback_handler import FallbackHandler
from .config.marketplace_config import MarketplaceConfig
from .chains.marketplace_chain import MarketplaceChain
from .utils.validators import InputValidationError

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SkillMarketplaceAgent:
    """
    Skill Marketplace & Learning Coach Agent 애플리케이션
    
    LangChain 및 LangGraph를 사용하여 양면 시장 구조의
    스킬 학습 플랫폼을 자동화합니다.
    """
    
    def __init__(self, preferred_provider: Optional[ModelProvider] = None):
        """
        SkillMarketplaceAgent 초기화
        
        Args:
            preferred_provider: LLM 선택 시 선호하는 Provider (예: ModelProvider.GROQ)
        """
        self.config = MarketplaceConfig()
        
        # 출력 및 데이터 디렉토리 생성
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.data_dir, exist_ok=True)
        logger.info(f"Output directory created: {self.config.output_dir}")
        logger.info(f"Data directory created: {self.config.data_dir}")
        
        self.model_manager = ModelManager()
        self.fallback_handler = FallbackHandler(self.model_manager)
        self.marketplace_chain = MarketplaceChain(
            config=self.config,
            model_manager=self.model_manager,
            fallback_handler=self.fallback_handler,
            preferred_provider=preferred_provider
        )
        
        logger.info("SkillMarketplaceAgent initialized.")

    async def run_marketplace_workflow(
        self,
        user_input: str,
        learner_id: str,
        learner_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Skill Marketplace 워크플로우를 실행합니다.
        
        Args:
            user_input: 사용자의 초기 요청
            learner_id: 학습자 ID
            learner_profile: 학습자 프로필 정보 (learner_id, goals, current_skills, learning_style 등)
        
        Returns:
            워크플로우 실행 결과 (최종 상태)
        """
        logger.info(f"Starting marketplace workflow for learner: {learner_id}")
        try:
            final_state = await self.marketplace_chain.run_workflow(user_input, learner_id, learner_profile)
            logger.info(f"Marketplace workflow completed for learner: {learner_id}")
            return final_state
        except InputValidationError as e:
            logger.error(f"Workflow stopped due to input validation error: {e}")
            return {"status": "failed", "error": str(e)}
        except Exception as e:
            logger.critical(f"An unexpected error occurred during workflow execution: {e}", exc_info=True)
            return {"status": "failed", "error": f"Unexpected workflow error: {e}"}


async def main():
    """메인 실행 함수"""
    # 예시 학습자 프로필 및 요청
    example_learner_profile = {
        "learner_id": "learner_001",
        "goals": ["Python 프로그래밍 마스터", "데이터 분석 스킬 향상"],
        "current_skills": {
            "Python": "beginner",
            "Data Analysis": "beginner"
        },
        "learning_style": "visual",
        "preferred_format": "one-on-one",
        "budget_range": "medium",
        "time_availability": "medium"
    }
    
    # Groq를 선호하는 Provider로 설정하여 에이전트 초기화
    agent = SkillMarketplaceAgent(preferred_provider=ModelProvider.GROQ)
    
    # 워크플로우 실행
    result = await agent.run_marketplace_workflow(
        user_input="Python 프로그래밍을 배우고 싶어요. 초보자에게 맞는 강사와 학습 경로를 추천해주세요.",
        learner_id="learner_001",
        learner_profile=example_learner_profile
    )
    
    print("\n=== Skill Marketplace Workflow Final Result ===")
    for key, value in result.items():
        if key == "final_report" and isinstance(value, dict) and "report_path" in value:
            print(f"{key}: (Report saved to: {value['report_path']})")
        else:
            print(f"{key}: {value}")
    
    if result.get("final_report") and result["final_report"].get("report_path"):
        print(f"\nFull marketplace report saved to: {result['final_report']['report_path']}")


if __name__ == "__main__":
    asyncio.run(main())

