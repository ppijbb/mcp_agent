"""
Physical AI 기기 제어 Agent

로봇 청소기, 스마트 장난감, 자동급식기 등 Physical AI 기기 통합 제어
"""

import logging
import json
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper
from ..tools.physical_ai_tools import PhysicalAITools
from ..tools.pet_tools import PetTools

logger = logging.getLogger(__name__)


class PhysicalAIControllerAgent:
    """
    Physical AI 기기 제어 Agent
    
    반려동물의 상태와 행동에 따라 Physical AI 기기를 자동으로 제어합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "petcare_data"
    ):
        """
        PhysicalAIControllerAgent 초기화
        
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
        self.physical_ai_tools = PhysicalAITools(data_dir=data_dir)
        self.pet_tools = PetTools(data_dir=data_dir)
        self.tools = self.mcp_tools.get_tools() + self.physical_ai_tools.get_tools() + self.pet_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        # Agent는 fallback_handler를 통해 직접 LLM 호출하므로 별도 초기화 불필요
        logger.info("Physical AI Controller Agent initialized")

    async def control_devices(
        self,
        pet_id: str,
        situation: str,
        pet_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        반려동물 상태에 따라 Physical AI 기기를 제어합니다.
        
        Args:
            pet_id: 반려동물 ID
            situation: 상황 설명 (예: "배변 완료", "활동량 낮음", "식사 시간")
            pet_state: 반려동물 현재 상태 (선택 사항)
        
        Returns:
            기기 제어 결과
        """
        input_message = f"""
        반려동물 '{pet_id}'의 상황에 따라 적절한 Physical AI 기기를 제어해주세요.
        
        상황: {situation}
        
        반려동물 상태:
        {json.dumps(pet_state, ensure_ascii=False, indent=2) if pet_state else "없음 (도구를 사용하여 조회)"}
        
        결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "pet_id": "{pet_id}",
            "situation": "{situation}",
            "devices_controlled": [
                {{
                    "device_type": "robot_vacuum/smart_toy/auto_feeder/smart_environment",
                    "device_id": "기기ID",
                    "action": "실행된 액션",
                    "result": "성공/실패"
                }}
            ],
            "reasoning": "제어 이유 설명"
        }}
        """
        
        try:
            response = await self.fallback_handler.invoke_with_fallback(
                messages=[HumanMessage(content=input_message)],
                preferred_provider=self.preferred_provider
            )
            
            logger.info(f"Device control completed for pet {pet_id} in situation: {situation}")
            if hasattr(response, 'content') and response.content:
                try:
                    return json.loads(response.content)
                except (json.JSONDecodeError, ValueError):
                    logger.warning(f"Failed to parse JSON response, using default device control result for pet {pet_id}")
            return {"pet_id": pet_id, "devices_controlled": []}
        except Exception as e:
            logger.error(f"Failed to control devices for pet {pet_id}: {e}")
            raise

