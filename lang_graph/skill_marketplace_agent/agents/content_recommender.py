"""
학습 컨텐츠 추천 Agent

온라인 강의, 튜토리얼 등 학습 컨텐츠 추천
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
from ..tools.learning_tools import LearningTools

logger = logging.getLogger(__name__)


class ContentRecommenderAgent:
    """
    학습 컨텐츠 추천 Agent
    
    학습자 프로필과 목표에 맞는 학습 컨텐츠를 추천합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "marketplace_data"
    ):
        """
        ContentRecommenderAgent 초기화
        
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
        self.learning_tools = LearningTools(data_dir=data_dir)
        self.tools = self.mcp_tools.get_tools() + self.learning_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        # Agent 초기화
        self._initialize_agent()
    
    def _initialize_agent(self):
        """LangChain Agent 초기화"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content recommender specializing in finding and recommending learning content (online courses, tutorials, books, videos).

Your task is to recommend relevant learning content based on learner profiles and goals.

For each learner, you must:
1. Analyze learning goals and current skill levels
2. Search for relevant content across different formats
3. Evaluate content quality and relevance
4. Match content to learning style preferences
5. Consider budget and time constraints
6. Recommend content in optimal learning sequence

Use the available tools to:
- Search for content using web search
- Track learning progress
- Recommend next steps

Provide detailed content recommendations with clear explanations and learning paths."""),
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
            
            logger.info("Content Recommender Agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Content Recommender Agent: {e}")
            raise

    async def recommend_content(
        self,
        learner_id: str,
        skill: str,
        learner_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        학습 컨텐츠를 추천합니다.
        
        Args:
            learner_id: 학습자 ID
            skill: 학습할 스킬
            learner_profile: 학습자 프로필 (선택 사항)
        
        Returns:
            추천된 컨텐츠 목록
        """
        input_message = f"""
        학습자 '{learner_id}'를 위한 '{skill}' 스킬 학습 컨텐츠를 추천해주세요.
        
        학습자 프로필:
        {json.dumps(learner_profile, ensure_ascii=False, indent=2) if learner_profile else "없음 (도구를 사용하여 조회)"}
        
        결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "learner_id": "{learner_id}",
            "skill": "{skill}",
            "recommended_content": [
                {{
                    "content_type": "online_course/tutorial/book/video",
                    "title": "컨텐츠 제목",
                    "provider": "제공자",
                    "url": "URL",
                    "rating": 4.5,
                    "duration_hours": 10,
                    "price": 49.99,
                    "description": "설명",
                    "why_recommended": "추천 이유"
                }}
            ],
            "learning_sequence": ["컨텐츠1", "컨텐츠2"],
            "estimated_total_hours": 30
        }}
        """
        
        try:
            response = await self.fallback_handler.invoke_with_fallback(
                messages=[HumanMessage(content=input_message)],
                preferred_provider=self.preferred_provider
            )
            
            logger.info(f"Content recommendation completed for learner {learner_id}, skill: {skill}")
            return json.loads(response.content) if hasattr(response, 'content') else {"learner_id": learner_id, "skill": skill, "recommended_content": []}
        except Exception as e:
            logger.error(f"Failed to recommend content for learner {learner_id}: {e}")
            raise

