"""
반려동물 건강 모니터링 Agent

건강 상태 모니터링, 이상 행동 감지, 건강 데이터 분석
"""

import logging
import json
from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper
from ..tools.health_tools import HealthTools
from ..tools.pet_tools import PetTools

logger = logging.getLogger(__name__)


class HealthMonitorAgent:
    """
    건강 모니터링 Agent
    
    반려동물의 건강 상태를 모니터링하고 이상 징후를 감지합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "petcare_data"
    ):
        """
        HealthMonitorAgent 초기화
        
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
        self.health_tools = HealthTools(data_dir=data_dir)
        self.pet_tools = PetTools(data_dir=data_dir)
        self.tools = self.mcp_tools.get_tools() + self.health_tools.get_tools() + self.pet_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        # Agent 초기화
        self._initialize_agent()
    
    def _initialize_agent(self):
        """LangChain Agent 초기화"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert pet health monitor specializing in tracking pet health, detecting anomalies, and providing health insights.

Your task is to monitor pet health status and identify potential issues early.

For each pet, you must:
1. Record and track health metrics (weight, temperature, activity level, etc.)
2. Analyze behavior patterns for anomalies
3. Detect early warning signs of health issues
4. Provide health summaries and recommendations
5. Alert owners when veterinary attention may be needed

Use the available tools to:
- Record health data
- Detect anomalies in behavior
- Analyze health trends
- Get health summaries

Provide detailed health monitoring with actionable insights and recommendations."""),
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
            
            logger.info("Health Monitor Agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Health Monitor Agent: {e}")
            raise

    async def monitor_health(self, pet_id: str, health_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        반려동물 건강 상태를 모니터링합니다.
        
        Args:
            pet_id: 반려동물 ID
            health_data: 건강 데이터 (선택 사항)
        
        Returns:
            건강 모니터링 결과
        """
        input_message = f"""
        반려동물 '{pet_id}'의 건강 상태를 모니터링하고 분석해주세요.
        
        건강 데이터:
        {json.dumps(health_data, ensure_ascii=False, indent=2) if health_data else "없음 (도구를 사용하여 조회)"}
        
        결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "pet_id": "{pet_id}",
            "health_status": "healthy/needs_attention/urgent",
            "metrics": {{
                "weight": {{"current": 0, "trend": "stable/increasing/decreasing"}},
                "activity_level": "normal/low/high",
                "appetite": "normal/decreased/increased"
            }},
            "anomalies_detected": [],
            "recommendations": ["권장사항1", "권장사항2"],
            "veterinary_consultation_needed": false
        }}
        """
        
        try:
            response = await self.fallback_handler.invoke_with_fallback(
                messages=[HumanMessage(content=input_message)],
                preferred_provider=self.preferred_provider
            )
            
            logger.info(f"Health monitoring completed for pet {pet_id}")
            return json.loads(response.content) if hasattr(response, 'content') else {"pet_id": pet_id, "health_status": "unknown"}
        except Exception as e:
            logger.error(f"Failed to monitor health for pet {pet_id}: {e}")
            raise

