"""
맞춤형 케어 계획 Agent

운동량, 식사량, 놀이 시간 최적화, 스트레스 완화 활동 제안
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


class CarePlannerAgent:
    """
    맞춤형 케어 계획 Agent
    
    반려동물의 프로필과 건강 상태를 기반으로 맞춤형 케어 계획을 생성합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "petcare_data"
    ):
        """
        CarePlannerAgent 초기화
        
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
            ("system", """You are an expert pet care planner specializing in creating personalized care plans for pets.

Your task is to create comprehensive care plans that optimize pet health and well-being.

For each pet, you must:
1. Analyze pet profile, health status, and behavior patterns
2. Create optimized exercise schedules
3. Plan feeding schedules and amounts
4. Schedule playtime and enrichment activities
5. Suggest stress relief activities
6. Recommend training programs if needed
7. Integrate Physical AI devices into care plans

Use the available tools to:
- Get pet profiles and health data
- Analyze behavior patterns
- Plan device usage

Provide detailed, actionable care plans tailored to each pet's unique needs."""),
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
                max_iterations=10
            )
            
            logger.info("Care Planner Agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Care Planner Agent: {e}")
            raise

    async def create_care_plan(
        self,
        pet_id: str,
        pet_profile: Optional[Dict[str, Any]] = None,
        health_status: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        맞춤형 케어 계획을 생성합니다.
        
        Args:
            pet_id: 반려동물 ID
            pet_profile: 반려동물 프로필 (선택 사항)
            health_status: 건강 상태 (선택 사항)
        
        Returns:
            케어 계획
        """
        input_message = f"""
        반려동물 '{pet_id}'를 위한 맞춤형 케어 계획을 생성해주세요.
        
        반려동물 프로필:
        {json.dumps(pet_profile, ensure_ascii=False, indent=2) if pet_profile else "없음 (도구를 사용하여 조회)"}
        
        건강 상태:
        {json.dumps(health_status, ensure_ascii=False, indent=2) if health_status else "없음 (도구를 사용하여 조회)"}
        
        결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "pet_id": "{pet_id}",
            "care_plan": {{
                "exercise_schedule": {{
                    "daily_walks": {{"times": ["시간1", "시간2"], "duration_minutes": 30}},
                    "playtime": {{"times": ["시간1"], "duration_minutes": 20}},
                    "intensity": "high/medium/low"
                }},
                "feeding_schedule": {{
                    "times": ["시간1", "시간2"],
                    "amount_per_meal": 100,
                    "total_daily_amount": 200
                }},
                "enrichment_activities": ["활동1", "활동2"],
                "stress_relief": ["방법1", "방법2"],
                "training_programs": ["프로그램1"],
                "physical_ai_integration": {{
                    "robot_vacuum": "배변 후 자동 청소",
                    "smart_toy": "활동량 낮을 때 자동 활성화",
                    "auto_feeder": "스케줄에 맞춰 자동 급식"
                }}
            }},
            "rationale": "계획 근거 설명"
        }}
        """
        
        try:
            response = await self.fallback_handler.invoke_with_fallback(
                messages=[HumanMessage(content=input_message)],
                preferred_provider=self.preferred_provider
            )
            
            logger.info(f"Care plan created for pet {pet_id}")
            return json.loads(response.content) if hasattr(response, 'content') else {"pet_id": pet_id, "care_plan": {}}
        except Exception as e:
            logger.error(f"Failed to create care plan for pet {pet_id}: {e}")
            raise

