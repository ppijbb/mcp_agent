"""
Marketplace 오케스트레이터 Agent

종합 오케스트레이터 - 매칭, 결제, 리뷰 등 전체 워크플로우 관리
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
from ..tools.marketplace_tools import MarketplaceTools
from ..tools.learning_tools import LearningTools

logger = logging.getLogger(__name__)


class MarketplaceOrchestratorAgent:
    """
    Marketplace 오케스트레이터 Agent
    
    전체 워크플로우를 오케스트레이션하고 매칭, 결제, 리뷰 등을 관리합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "marketplace_data"
    ):
        """
        MarketplaceOrchestratorAgent 초기화
        
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
        self.marketplace_tools = MarketplaceTools(data_dir=data_dir)
        self.learning_tools = LearningTools(data_dir=data_dir)
        self.tools = (
            self.mcp_tools.get_tools() +
            self.marketplace_tools.get_tools() +
            self.learning_tools.get_tools()
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
            ("system", """You are a comprehensive Marketplace Orchestrator for a skill learning platform. Your role is to orchestrate the entire learning marketplace workflow including matching, payment processing, session management, and reviews.

Your tasks include:
1. Coordinating learner-instructor matching
2. Managing learning session creation and scheduling
3. Processing marketplace transactions and calculating commissions
4. Tracking learning progress and generating reports
5. Facilitating reviews and ratings
6. Providing overall marketplace insights

Always aim to provide seamless marketplace experience. Use all available tools to complete tasks. If information is missing, try to infer or ask clarifying questions.
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
            
            logger.info("Marketplace Orchestrator Agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Marketplace Orchestrator Agent: {e}")
            raise

    async def orchestrate_marketplace(
        self,
        user_request: str,
        learner_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Marketplace 워크플로우를 오케스트레이션합니다.
        
        Args:
            user_request: 사용자의 요청
            learner_id: 학습자 ID
            context: 추가 컨텍스트 (선택 사항)
        
        Returns:
            오케스트레이션 결과
        """
        input_message = f"""
        사용자의 Marketplace 요청을 처리하고 최적의 학습 경험을 제공해주세요.
        필요한 경우 모든 도구를 활용하여 정보를 수집하고 작업을 수행한 후, 최종 결과를 종합하여 반환해주세요.
        
        사용자 요청: "{user_request}"
        
        학습자 ID: {learner_id}
        
        추가 컨텍스트:
        {json.dumps(context, ensure_ascii=False, indent=2) if context else "없음"}
        
        최종 결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "learner_id": "{learner_id}",
            "summary": "요청에 대한 종합적인 요약 및 수행된 작업",
            "actions_taken": [
                {{
                    "action": "수행된 작업",
                    "result": "결과"
                }}
            ],
            "marketplace_data": {{
                "matched_instructors": [],
                "created_sessions": [],
                "transactions": [],
                "commission_earned": 0.0
            }},
            "recommendations": ["추천사항1", "추천사항2"],
            "next_steps": ["다음 단계1", "다음 단계2"]
        }}
        """
        
        try:
            response = await self.fallback_handler.invoke_with_fallback(
                messages=[HumanMessage(content=input_message)],
                preferred_provider=self.preferred_provider
            )
            
            logger.info(f"Marketplace orchestration completed for learner {learner_id}")
            return json.loads(response.content) if hasattr(response, 'content') else {"learner_id": learner_id, "summary": "처리 완료"}
        except Exception as e:
            logger.error(f"Failed to orchestrate marketplace for learner {learner_id}: {e}")
            raise

