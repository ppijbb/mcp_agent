"""
스킬 경로 추천 Agent

맞춤형 스킬 학습 경로 추천
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


class SkillPathRecommenderAgent:
    """
    스킬 경로 추천 Agent
    
    학습자 프로필을 기반으로 맞춤형 스킬 학습 경로를 추천합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "marketplace_data"
    ):
        """
        SkillPathRecommenderAgent 초기화
        
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
            ("system", """You are an expert skill path recommender specializing in creating personalized learning paths for skill development.

Your task is to recommend optimal learning paths based on learner profiles and goals.

For each learner, you must:
1. Analyze learning goals and current skill levels
2. Identify prerequisite skills and dependencies
3. Create a structured learning path with milestones
4. Recommend learning resources and formats
5. Estimate time to completion for each stage
6. Suggest intermediate goals and checkpoints

Use the available tools to:
- Track learning progress
- Analyze skill improvements
- Recommend next steps

Provide detailed, actionable learning paths with clear milestones and recommendations."""),
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
            
            logger.info("Skill Path Recommender Agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Skill Path Recommender Agent: {e}")
            raise

    async def recommend_skill_path(
        self,
        learner_id: str,
        target_skill: str,
        learner_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        스킬 학습 경로를 추천합니다.
        
        Args:
            learner_id: 학습자 ID
            target_skill: 목표 스킬
            learner_profile: 학습자 프로필 (선택 사항)
        
        Returns:
            추천된 스킬 경로
        """
        input_message = f"""
        학습자 '{learner_id}'를 위한 '{target_skill}' 스킬 학습 경로를 추천해주세요.
        
        학습자 프로필:
        {json.dumps(learner_profile, ensure_ascii=False, indent=2) if learner_profile else "없음 (도구를 사용하여 조회)"}
        
        결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "learner_id": "{learner_id}",
            "target_skill": "{target_skill}",
            "learning_path": {{
                "stages": [
                    {{
                        "stage": 1,
                        "name": "기초 단계",
                        "skills": ["기초 스킬1", "기초 스킬2"],
                        "estimated_hours": 20,
                        "resources": ["리소스1", "리소스2"]
                    }},
                    {{
                        "stage": 2,
                        "name": "중급 단계",
                        "skills": ["중급 스킬1"],
                        "estimated_hours": 30,
                        "resources": ["리소스3"]
                    }}
                ],
                "total_estimated_hours": 50,
                "prerequisites": ["선수 스킬1", "선수 스킬2"]
            }},
            "recommendations": {{
                "instructor_types": ["타입1", "타입2"],
                "learning_format": "one-on-one/group/online",
                "pace": "slow/moderate/fast"
            }}
        }}
        """
        
        try:
            response = await self.fallback_handler.invoke_with_fallback(
                messages=[HumanMessage(content=input_message)],
                preferred_provider=self.preferred_provider
            )
            
            logger.info(f"Skill path recommendation completed for learner {learner_id}, skill: {target_skill}")
            return json.loads(response.content) if hasattr(response, 'content') else {"learner_id": learner_id, "target_skill": target_skill, "learning_path": {}}
        except Exception as e:
            logger.error(f"Failed to recommend skill path for learner {learner_id}: {e}")
            raise

