"""
종합 반려동물 어시스턴트 Agent

모든 반려동물 관리 기능을 통합하여 종합적인 어시스턴트 제공
"""

import logging
import json
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..config.petcare_config import PetCareConfig
from ..tools.mcp_tools import MCPToolsWrapper
from ..tools.pet_tools import PetTools
from ..tools.health_tools import HealthTools
from ..tools.physical_ai_tools import PhysicalAITools

logger = logging.getLogger(__name__)


class PetAssistantAgent:
    """
    종합 반려동물 어시스턴트 Agent
    
    모든 반려동물 관리 기능을 통합하여 종합적인 어시스턴트를 제공합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "petcare_data",
        config: Optional[PetCareConfig] = None
    ):
        """
        PetAssistantAgent 초기화
        
        Args:
            model_manager: ModelManager 인스턴스
            fallback_handler: FallbackHandler 인스턴스
            preferred_provider: 선호하는 Provider
            data_dir: 데이터 저장 디렉토리
            config: PetCareConfig 인스턴스 (최신 Physical AI 기술 사용)
        """
        self.model_manager = model_manager
        self.fallback_handler = fallback_handler
        self.preferred_provider = preferred_provider
        self.config = config
        
        # 모든 도구 초기화 (최신 기술: MQTT v5, Home Assistant API)
        self.mcp_tools = MCPToolsWrapper()
        self.pet_tools = PetTools(data_dir=data_dir)
        self.health_tools = HealthTools(data_dir=data_dir)
        self.physical_ai_tools = PhysicalAITools(data_dir=data_dir, config=config)
        self.tools = (
            self.mcp_tools.get_tools() +
            self.pet_tools.get_tools() +
            self.health_tools.get_tools() +
            self.physical_ai_tools.get_tools()
        )
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        # Agent는 fallback_handler를 통해 직접 LLM 호출하므로 별도 초기화 불필요
        logger.info("Pet Assistant Agent initialized")

    async def assist_pet_care(self, user_request: str, pet_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        사용자의 반려동물 케어 요청을 처리하고 종합적인 지원을 제공합니다.
        
        Args:
            user_request: 사용자의 요청
            pet_id: 반려동물 ID
            context: 추가 컨텍스트 (선택 사항)
        
        Returns:
            종합적인 케어 지원 결과
        """
        input_message = f"""
        사용자의 반려동물 케어 요청을 처리하고 최적의 케어 경험을 제공해주세요.
        필요한 경우 Physical AI 기기와 도구를 활용하여 정보를 수집하고 작업을 수행한 후, 최종 결과를 종합하여 반환해주세요.
        
        사용자 요청: "{user_request}"
        
        반려동물 ID: {pet_id}
        
        추가 컨텍스트:
        {json.dumps(context, ensure_ascii=False, indent=2) if context else "없음"}
        
        최종 결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "pet_id": "{pet_id}",
            "summary": "요청에 대한 종합적인 요약 및 수행된 작업",
            "actions_taken": [
                {{
                    "action": "수행된 작업",
                    "device": "사용된 기기 (있는 경우)",
                    "result": "결과"
                }}
            ],
            "recommendations": ["추천사항1", "추천사항2"],
            "next_steps": ["다음 단계1", "다음 단계2"]
        }}
        """
        
        try:
            response = await self.fallback_handler.invoke_with_fallback(
                messages=[HumanMessage(content=input_message)],
                preferred_provider=self.preferred_provider
            )
            
            logger.info(f"Pet care assistance completed for pet {pet_id}")
            if hasattr(response, 'content') and response.content:
                try:
                    return json.loads(response.content)
                except (json.JSONDecodeError, ValueError):
                    logger.warning(f"Failed to parse JSON response, using default summary for pet {pet_id}")
            return {"pet_id": pet_id, "summary": "처리 완료"}
        except Exception as e:
            logger.error(f"Failed to assist pet care for pet {pet_id}: {e}")
            raise

