"""
종합 반려동물 어시스턴트 Agent

모든 반려동물 관리 기능을 통합하여 종합적인 어시스턴트 제공
"""

import logging
import json
from typing import Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
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
        data_dir: str = "petcare_data"
    ):
        """
        PetAssistantAgent 초기화
        
        Args:
            model_manager: ModelManager 인스턴스
            fallback_handler: FallbackHandler 인스턴스
            preferred_provider: 선호하는 Provider
            data_dir: 데이터 저장 디렉토리
        """
        self.model_manager = model_manager
        self.fallback_handler = fallback_handler
        self.preferred_provider = preferred_provider
        
        # 모든 도구 초기화
        self.mcp_tools = MCPToolsWrapper()
        self.pet_tools = PetTools(data_dir=data_dir)
        self.health_tools = HealthTools(data_dir=data_dir)
        self.physical_ai_tools = PhysicalAITools(data_dir=data_dir)
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
        
        # Agent 초기화
        self._initialize_agent()
    
    def _initialize_agent(self):
        """LangChain Agent 초기화"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a comprehensive Pet Care Assistant powered by Physical AI. Your role is to understand pet care needs and orchestrate various specialized agents and tools to provide the best possible pet care experience.

Your tasks include:
1. Interpreting user requests related to pet care (e.g., "my dog seems sad", "clean up after my cat", "schedule feeding time", "check my pet's health").
2. Coordinating with Physical AI devices (robot vacuums, smart toys, auto feeders, smart environment) to provide automated care.
3. Monitoring pet health and behavior patterns.
4. Creating and managing personalized care plans.
5. Providing proactive recommendations and alerts.

Always aim to provide personalized, intelligent, and automated pet care. Use Physical AI devices to enhance pet comfort and reduce owner burden. If a specific tool is needed, call it with precise arguments. If information is missing, try to infer or ask clarifying questions.
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        try:
            self.agent = create_openai_functions_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=15  # Increased for orchestration
            )
            
            logger.info("Pet Assistant Agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pet Assistant Agent: {e}")
            raise

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
            return json.loads(response.content) if hasattr(response, 'content') else {"pet_id": pet_id, "summary": "처리 완료"}
        except Exception as e:
            logger.error(f"Failed to assist pet care for pet {pet_id}: {e}")
            raise

