"""
학습자 프로필 분석 Agent

학습자의 목표, 현재 수준, 학습 스타일을 분석하여 맞춤형 프로필 생성
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


class LearnerProfileAnalyzerAgent:
    """
    학습자 프로필 분석 Agent
    
    학습자의 목표, 현재 수준, 학습 스타일을 분석하여 맞춤형 프로필을 생성합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "marketplace_data"
    ):
        """
        LearnerProfileAnalyzerAgent 초기화
        
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
            ("system", """You are an expert learner profile analyst specializing in understanding learning goals, current skill levels, and learning styles.

Your task is to analyze learner information and create comprehensive profiles for personalized learning paths.

For each learner, you must:
1. Analyze learning goals and objectives
2. Assess current skill levels across different domains
3. Identify learning style preferences (visual, auditory, kinesthetic, reading/writing)
4. Determine budget constraints and time availability
5. Understand preferred learning formats (one-on-one, group, online, offline)
6. Create a comprehensive profile for personalized learning recommendations

Use the available tools to:
- Track learning progress
- Analyze skill improvements
- Generate learning reports

Provide detailed, actionable profile analysis based on real learner data."""),
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
            
            logger.info("Learner Profile Analyzer Agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Learner Profile Analyzer Agent: {e}")
            raise

    async def analyze_profile(self, learner_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        학습자 프로필을 분석합니다.
        
        Args:
            learner_info: 학습자 기본 정보 (learner_id, goals, current_skills, learning_style 등)
        
        Returns:
            분석된 프로필 정보
        """
        input_message = f"""
        다음 학습자 정보를 분석하여 종합적인 프로필을 생성해주세요.
        
        학습자 정보:
        {json.dumps(learner_info, ensure_ascii=False, indent=2)}
        
        결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "learner_id": "{learner_info.get('learner_id', 'unknown')}",
            "profile": {{
                "learning_goals": ["목표1", "목표2"],
                "current_skills": {{
                    "skill_name": "beginner/intermediate/advanced"
                }},
                "learning_style": "visual/auditory/kinesthetic/reading",
                "preferred_format": "one-on-one/group/online/offline",
                "budget_range": "low/medium/high",
                "time_availability": "low/medium/high",
                "learning_pace": "slow/moderate/fast"
            }},
            "recommendations": {{
                "suitable_instructor_types": ["타입1", "타입2"],
                "recommended_learning_path": "경로 설명"
            }}
        }}
        """
        
        try:
            response = await self.fallback_handler.invoke_with_fallback(
                messages=[HumanMessage(content=input_message)],
                preferred_provider=self.preferred_provider
            )
            
            logger.info(f"Profile analysis completed for learner {learner_info.get('learner_id')}")
            return json.loads(response.content) if hasattr(response, 'content') else {"learner_id": learner_info.get('learner_id'), "profile": {}}
        except Exception as e:
            logger.error(f"Failed to analyze profile for learner {learner_info.get('learner_id')}: {e}")
            raise

