"""
반려동물 프로필 분석 Agent

반려동물의 종류, 성격, 건강 상태를 분석하여 맞춤형 프로필 생성
"""

import logging
import json
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper
from ..tools.pet_tools import PetTools

logger = logging.getLogger(__name__)


class PetProfileAnalyzerAgent:
    """
    반려동물 프로필 분석 Agent
    
    반려동물의 종류, 성격, 건강 상태를 분석하여 맞춤형 프로필을 생성합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "petcare_data"
    ):
        """
        PetProfileAnalyzerAgent 초기화
        
        Args:
            model_manager: ModelManager 인스턴스
            fallback_handler: FallbackHandler 인스턴스
            preferred_provider: 선호하는 Provider
            data_dir: 데이터 저장 디렉토리
        """
        self.model_manager = model_manager
        self.fallback_handler = fallback_handler
        self.preferred_provider = preferred_provider
        
        # 도구 초기화
        self.mcp_tools = MCPToolsWrapper()
        self.pet_tools = PetTools(data_dir=data_dir)
        self.tools = self.mcp_tools.get_tools() + self.pet_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        # Agent는 fallback_handler를 통해 직접 LLM 호출하므로 별도 초기화 불필요
        logger.info("Pet Profile Analyzer Agent initialized")

    async def analyze_profile(self, pet_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        반려동물 프로필을 분석합니다.
        
        Args:
            pet_info: 반려동물 기본 정보 (pet_id, name, species, breed, age 등)
        
        Returns:
            분석된 프로필 정보
        """
        input_message = f"""
        다음 반려동물 정보를 분석하여 종합적인 프로필을 생성해주세요.
        
        반려동물 정보:
        {json.dumps(pet_info, ensure_ascii=False, indent=2)}
        
        결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "pet_id": "{pet_info.get('pet_id', 'unknown')}",
            "profile": {{
                "species": "종류",
                "breed": "품종",
                "age_months": 나이,
                "personality_traits": ["특성1", "특성2"],
                "behavior_patterns": {{
                    "activity_level": "high/medium/low",
                    "social_level": "high/medium/low",
                    "sleep_pattern": "정상/불규칙"
                }},
                "health_status": "healthy/needs_attention",
                "care_requirements": {{
                    "exercise_needs": "high/medium/low",
                    "feeding_schedule": "정기적/불규칙",
                    "grooming_needs": "high/medium/low"
                }},
                "preferences": {{
                    "favorite_activities": ["활동1", "활동2"],
                    "stress_factors": ["요인1", "요인2"]
                }}
            }}
        }}
        """
        
        try:
            response = await self.fallback_handler.invoke_with_fallback(
                messages=[HumanMessage(content=input_message)],
                preferred_provider=self.preferred_provider
            )
            
            # LLM 응답에서 프로필 정보 추출 및 파싱
            logger.info(f"Profile analysis completed for pet {pet_info.get('pet_id')}")
            if hasattr(response, 'content') and response.content:
                try:
                    return json.loads(response.content)
                except (json.JSONDecodeError, ValueError):
                    logger.warning(f"Failed to parse JSON response, using default profile for pet {pet_info.get('pet_id')}")
            return {"pet_id": pet_info.get('pet_id'), "profile": {}}
        except Exception as e:
            logger.error(f"Failed to analyze profile for pet {pet_info.get('pet_id')}: {e}")
            raise

