"""
강사 매칭 Agent

양면 시장 핵심 - 학습자와 강사를 최적으로 매칭
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
from ..tools.marketplace_tools import MarketplaceTools
from ..tools.learning_tools import LearningTools

logger = logging.getLogger(__name__)


class InstructorMatcherAgent:
    """
    강사 매칭 Agent
    
    학습자 프로필과 강사 프로필을 분석하여 최적의 강사를 매칭합니다.
    양면 시장의 핵심 기능입니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "marketplace_data"
    ):
        """
        InstructorMatcherAgent 초기화
        
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
            ("system", """You are an expert instructor matcher specializing in matching learners with the best instructors for their needs.

Your task is to find optimal instructor-learner matches based on multiple factors.

For each matching request, you must:
1. Analyze learner profile (goals, current level, learning style, budget, time availability)
2. Search and filter available instructors
3. Calculate matching scores based on:
   - Skill alignment
   - Teaching level compatibility
   - Learning style match
   - Budget compatibility
   - Time availability
   - Instructor ratings and reviews
4. Rank instructors by match quality
5. Provide detailed match explanations

Use the available tools to:
- Search instructors
- Match instructors with learners
- Analyze learning progress

Provide detailed matching results with clear explanations for each recommendation."""),
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
            
            logger.info("Instructor Matcher Agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Instructor Matcher Agent: {e}")
            raise

    async def match_instructor(
        self,
        learner_id: str,
        skill: str,
        learner_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        학습자에게 최적의 강사를 매칭합니다.
        
        Args:
            learner_id: 학습자 ID
            skill: 학습할 스킬
            learner_profile: 학습자 프로필 (선택 사항)
        
        Returns:
            매칭된 강사 정보 및 매칭 점수
        """
        input_message = f"""
        학습자 '{learner_id}'를 위한 '{skill}' 스킬 강사를 매칭해주세요.
        
        학습자 프로필:
        {json.dumps(learner_profile, ensure_ascii=False, indent=2) if learner_profile else "없음 (도구를 사용하여 조회)"}
        
        결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "learner_id": "{learner_id}",
            "skill": "{skill}",
            "matched_instructors": [
                {{
                    "instructor_id": "강사ID",
                    "match_score": 0.95,
                    "reasons": ["이유1", "이유2"],
                    "instructor_info": {{
                        "name": "강사명",
                        "rating": 4.8,
                        "hourly_rate": 50,
                        "specialties": ["스킬1", "스킬2"]
                    }}
                }}
            ],
            "recommendation": "최종 추천 강사 및 이유"
        }}
        """
        
        try:
            response = await self.fallback_handler.invoke_with_fallback(
                messages=[HumanMessage(content=input_message)],
                preferred_provider=self.preferred_provider
            )
            
            logger.info(f"Instructor matching completed for learner {learner_id}, skill: {skill}")
            return json.loads(response.content) if hasattr(response, 'content') else {"learner_id": learner_id, "skill": skill, "matched_instructors": []}
        except Exception as e:
            logger.error(f"Failed to match instructor for learner {learner_id}: {e}")
            raise

