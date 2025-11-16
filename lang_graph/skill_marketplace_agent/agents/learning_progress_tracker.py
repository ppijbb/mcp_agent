"""
학습 진행 추적 Agent

학습 진행 추적 및 피드백 제공
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
from ..tools.learning_tools import LearningTools

logger = logging.getLogger(__name__)


class LearningProgressTrackerAgent:
    """
    학습 진행 추적 Agent
    
    학습자의 진행 상황을 추적하고 피드백을 제공합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "marketplace_data"
    ):
        """
        LearningProgressTrackerAgent 초기화
        
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
            ("system", """You are an expert learning progress tracker specializing in monitoring and analyzing learner progress.

Your task is to track learning progress and provide actionable feedback.

For each learner, you must:
1. Record and track learning progress across sessions
2. Analyze skill improvement over time
3. Identify learning patterns and trends
4. Detect areas needing attention
5. Generate comprehensive learning reports
6. Recommend next steps based on progress

Use the available tools to:
- Record learning progress
- Track skill improvements
- Generate learning reports
- Recommend next steps

Provide detailed progress analysis with actionable insights and recommendations."""),
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
            
            logger.info("Learning Progress Tracker Agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Learning Progress Tracker Agent: {e}")
            raise

    async def track_progress(
        self,
        learner_id: str,
        skill: Optional[str] = None,
        period_days: Optional[int] = 30
    ) -> Dict[str, Any]:
        """
        학습 진행을 추적합니다.
        
        Args:
            learner_id: 학습자 ID
            skill: 추적할 스킬 (선택 사항, 없으면 전체)
            period_days: 추적 기간 (일)
        
        Returns:
            진행 추적 결과 및 피드백
        """
        input_message = f"""
        학습자 '{learner_id}'의 학습 진행을 추적하고 분석해주세요.
        
        스킬: {skill or "전체"}
        기간: 최근 {period_days}일
        
        결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "learner_id": "{learner_id}",
            "skill": "{skill or 'all'}",
            "period_days": {period_days},
            "progress_summary": {{
                "total_sessions": 10,
                "skills_learned": ["스킬1", "스킬2"],
                "average_progress": 65.5,
                "improvement_rate": 15.2
            }},
            "skill_breakdown": {{
                "skill_name": {{
                    "sessions": 5,
                    "progress": 70.0,
                    "improvement": 20.0,
                    "current_level": "intermediate"
                }}
            }},
            "insights": ["인사이트1", "인사이트2"],
            "recommendations": ["권장사항1", "권장사항2"]
        }}
        """
        
        try:
            response = await self.fallback_handler.invoke_with_fallback(
                messages=[HumanMessage(content=input_message)],
                preferred_provider=self.preferred_provider
            )
            
            logger.info(f"Progress tracking completed for learner {learner_id}")
            return json.loads(response.content) if hasattr(response, 'content') else {"learner_id": learner_id, "progress_summary": {}}
        except Exception as e:
            logger.error(f"Failed to track progress for learner {learner_id}: {e}")
            raise

