"""
PetCare Physical AI Agent 메인 진입점
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from .llm.model_manager import ModelManager, ModelProvider
from .llm.fallback_handler import FallbackHandler
from .config.petcare_config import PetCareConfig
from .chains.petcare_chain import PetCareChain
from .utils.validators import InputValidationError

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PetCarePhysicalAIAgent:
    """
    PetCare Physical AI Agent 애플리케이션
    
    LangChain 및 LangGraph를 사용하여 Physical AI 기기와 연동된
    반려동물 맞춤형 케어 서비스를 자동화합니다.
    """
    
    def __init__(self, preferred_provider: Optional[ModelProvider] = None):
        """
        PetCarePhysicalAIAgent 초기화
        
        Args:
            preferred_provider: LLM 선택 시 선호하는 Provider (예: ModelProvider.GROQ)
        """
        self.config = PetCareConfig()
        
        # 출력 및 데이터 디렉토리 생성
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.data_dir, exist_ok=True)
        logger.info(f"Output directory created: {self.config.output_dir}")
        logger.info(f"Data directory created: {self.config.data_dir}")
        
        self.model_manager = ModelManager()
        self.fallback_handler = FallbackHandler(self.model_manager)
        self.petcare_chain = PetCareChain(
            config=self.config,
            model_manager=self.model_manager,
            fallback_handler=self.fallback_handler,
            preferred_provider=preferred_provider
        )
        
        logger.info("PetCarePhysicalAIAgent initialized.")

    async def run_pet_care_workflow(
        self,
        user_input: str,
        pet_id: str,
        pet_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        반려동물 케어 워크플로우를 실행합니다.
        
        Args:
            user_input: 사용자의 초기 요청
            pet_id: 반려동물 ID
            pet_profile: 반려동물 프로필 정보 (pet_id, name, species, breed, age 등)
        
        Returns:
            워크플로우 실행 결과 (최종 상태)
        """
        logger.info(f"Starting pet care workflow for pet: {pet_id}")
        try:
            final_state = await self.petcare_chain.run_workflow(user_input, pet_id, pet_profile)
            logger.info(f"Pet care workflow completed for pet: {pet_id}")
            return final_state
        except InputValidationError as e:
            logger.error(f"Workflow stopped due to input validation error: {e}")
            return {"status": "failed", "error": str(e)}
        except Exception as e:
            logger.critical(f"An unexpected error occurred during workflow execution: {e}", exc_info=True)
            return {"status": "failed", "error": f"Unexpected workflow error: {e}"}


async def main():
    """메인 실행 함수"""
    # 예시 반려동물 프로필 및 요청
    example_pet_profile = {
        "pet_id": "pet_001",
        "name": "뽀삐",
        "species": "dog",
        "breed": "골든 리트리버",
        "age": 24,  # 24개월 (2세)
    }
    
    # Groq를 선호하는 Provider로 설정하여 에이전트 초기화
    agent = PetCarePhysicalAIAgent(preferred_provider=ModelProvider.GROQ)
    
    # 워크플로우 실행
    result = await agent.run_pet_care_workflow(
        user_input="우리 강아지가 최근 활동량이 줄었어요. 건강 체크하고 케어 계획을 만들어주세요.",
        pet_id="pet_001",
        pet_profile=example_pet_profile
    )
    
    print("\n=== Pet Care Workflow Final Result ===")
    for key, value in result.items():
        if key == "final_report" and isinstance(value, dict) and "report_path" in value:
            print(f"{key}: (Report saved to: {value['report_path']})")
        else:
            print(f"{key}: {value}")
    
    if result.get("final_report") and result["final_report"].get("report_path"):
        print(f"\nFull pet care report saved to: {result['final_report']['report_path']}")


if __name__ == "__main__":
    asyncio.run(main())

