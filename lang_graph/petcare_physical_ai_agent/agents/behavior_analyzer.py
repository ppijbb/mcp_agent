"""
반려동물 행동 분석 Agent

행동 패턴 분석, 이상 행동 감지, 행동 기반 인사이트 생성
"""

import logging
import json
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper
from ..tools.pet_tools import PetTools
from ..tools.health_tools import HealthTools

logger = logging.getLogger(__name__)


class BehaviorAnalyzerAgent:
    """
    행동 분석 Agent
    
    반려동물의 행동 패턴을 분석하고 이상 행동을 감지합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "petcare_data"
    ):
        """
        BehaviorAnalyzerAgent 초기화
        
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
        self.health_tools = HealthTools(data_dir=data_dir)
        self.tools = self.mcp_tools.get_tools() + self.pet_tools.get_tools() + self.health_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        # Agent는 fallback_handler를 통해 직접 LLM 호출하므로 별도 초기화 불필요
        logger.info("Behavior Analyzer Agent initialized")

    async def analyze_behavior(self, pet_id: str, behavior_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        반려동물 행동을 분석합니다.
        
        Args:
            pet_id: 반려동물 ID
            behavior_data: 행동 데이터 (선택 사항, 없으면 도구로 조회)
        
        Returns:
            행동 분석 결과
        """
        input_message = f"""
        반려동물 '{pet_id}'의 행동 패턴을 분석해주세요.
        
        행동 데이터:
        {json.dumps(behavior_data, ensure_ascii=False, indent=2) if behavior_data else "없음 (도구를 사용하여 조회)"}
        
        결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "pet_id": "{pet_id}",
            "behavior_patterns": {{
                "eating_pattern": "정상/불규칙",
                "sleeping_pattern": "정상/불규칙",
                "activity_pattern": "정상/낮음/높음",
                "social_behavior": "정상/변화"
            }},
            "anomalies": [
                {{
                    "type": "이상 타입",
                    "description": "설명",
                    "severity": "low/medium/high",
                    "recommendation": "권장사항"
                }}
            ],
            "insights": ["인사이트1", "인사이트2"],
            "recommendations": ["권장사항1", "권장사항2"]
        }}
        """
        
        try:
            response = await self.fallback_handler.invoke_with_fallback(
                messages=[HumanMessage(content=input_message)],
                preferred_provider=self.preferred_provider
            )
            
            logger.info(f"Behavior analysis completed for pet {pet_id}")
            if hasattr(response, 'content') and response.content:
                try:
                    return json.loads(response.content)
                except (json.JSONDecodeError, ValueError):
                    logger.warning(f"Failed to parse JSON response, using default behavior patterns for pet {pet_id}")
            return {"pet_id": pet_id, "behavior_patterns": {}}
        except Exception as e:
            logger.error(f"Failed to analyze behavior for pet {pet_id}: {e}")
            raise

