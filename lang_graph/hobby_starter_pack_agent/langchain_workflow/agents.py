"""
LangChain 기반 에이전트 시스템
복잡한 작업 자동화 및 도구 통합
"""

import logging
from typing import Dict, Any, List, Optional, Union
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.tools import BaseTool, tool
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# 도구 정의를 위한 Pydantic 모델들
class HobbyRecommendationInput(BaseModel):
    user_profile: Dict[str, Any] = Field(description="사용자 프로필 정보")
    preferences: List[str] = Field(description="사용자 선호사항")
    constraints: Dict[str, Any] = Field(description="제약사항 (시간, 예산 등)")

class HobbyRecommendationOutput(BaseModel):
    recommendations: List[Dict[str, Any]] = Field(description="추천 취미 목록")
    reasoning: str = Field(description="추천 근거")
    confidence_score: float = Field(description="추천 신뢰도")

class CommunityMatchingInput(BaseModel):
    user_profile: Dict[str, Any] = Field(description="사용자 프로필")
    hobby_interests: List[str] = Field(description="관심 취미")
    location: str = Field(description="활동 지역")

class CommunityMatchingOutput(BaseModel):
    matched_communities: List[Dict[str, Any]] = Field(description="매칭된 커뮤니티")
    match_scores: Dict[str, float] = Field(description="매칭 점수")

class ScheduleIntegrationInput(BaseModel):
    current_schedule: Dict[str, Any] = Field(description="현재 스케줄")
    hobby_activities: List[Dict[str, Any]] = Field(description="추가할 취미 활동")
    time_constraints: Dict[str, Any] = Field(description="시간 제약")

class ScheduleIntegrationOutput(BaseModel):
    integrated_schedule: Dict[str, Any] = Field(description="통합된 스케줄")
    optimization_suggestions: List[str] = Field(description="최적화 제안")

# LangChain 도구들
@tool
def analyze_user_profile(user_input: str, conversation_history: str = "") -> Dict[str, Any]:
    """사용자 입력을 분석하여 프로필을 생성합니다."""
    try:
        # 실제 구현에서는 LLM을 호출하여 분석
        # 여기서는 예시 응답 반환
        profile = {
            "interests": ["reading", "technology", "outdoor_activities"],
            "personality_type": "introvert",
            "skill_level": "beginner",
            "time_availability": "weekends",
            "budget_range": "medium",
            "location_preference": "Seoul",
            "confidence_score": 0.85,
            "analysis_reasoning": "사용자 입력과 대화 기록을 분석한 결과입니다."
        }
        
        logger.info("사용자 프로필 분석 완료")
        return profile
        
    except Exception as e:
        logger.error(f"프로필 분석 실패: {e}")
        return {"error": "프로필 분석에 실패했습니다."}

@tool
def search_hobbies(query: str, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """사용자 프로필에 맞는 취미를 검색합니다."""
    try:
        # 실제 구현에서는 벡터 데이터베이스 검색
        hobbies = [
            {
                "name": "독서",
                "category": "intellectual",
                "difficulty": "beginner",
                "time_commitment": "flexible",
                "cost": "low",
                "match_score": 0.9
            },
            {
                "name": "등산",
                "category": "outdoor",
                "difficulty": "beginner",
                "time_commitment": "weekends",
                "cost": "medium",
                "match_score": 0.8
            }
        ]
        
        logger.info(f"취미 검색 완료: {len(hobbies)}개 결과")
        return hobbies
        
    except Exception as e:
        logger.error(f"취미 검색 실패: {e}")
        return []

@tool
def find_communities(hobby_categories: List[str], location: str) -> List[Dict[str, Any]]:
    """취미 카테고리에 맞는 커뮤니티를 찾습니다."""
    try:
        communities = [
            {
                "name": "독서 모임",
                "type": "online",
                "category": "intellectual",
                "members": 150,
                "activity_level": "high",
                "location": "Seoul"
            },
            {
                "name": "등산 동호회",
                "type": "offline",
                "category": "outdoor",
                "members": 80,
                "activity_level": "medium",
                "location": "Seoul"
            }
        ]
        
        logger.info(f"커뮤니티 검색 완료: {len(communities)}개 결과")
        return communities
        
    except Exception as e:
        logger.error(f"커뮤니티 검색 실패: {e}")
        return []

@tool
def optimize_schedule(current_schedule: Dict[str, Any], new_activities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """현재 스케줄에 새로운 활동을 최적화하여 통합합니다."""
    try:
        # 실제 구현에서는 스케줄 최적화 알고리즘
        optimized_schedule = {
            "weekdays": {
                "evening": ["독서", "온라인 강의"],
                "night": ["명상", "일기 쓰기"]
            },
            "weekends": {
                "morning": ["등산", "요가"],
                "afternoon": ["커뮤니티 활동", "새로운 취미 시도"],
                "evening": ["친구들과 만남", "영화 감상"]
            }
        }
        
        logger.info("스케줄 최적화 완료")
        return optimized_schedule
        
    except Exception as e:
        logger.error(f"스케줄 최적화 실패: {e}")
        return {"error": "스케줄 최적화에 실패했습니다."}

class HSPLangChainAgent:
    """HSP Agent 전용 LangChain 에이전트"""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        self.llm_config = llm_config or {"model": "gpt-4", "temperature": 0.7}
        self.llm = None
        self.tools = []
        self.agent = None
        self.agent_executor = None
        self.memory = None
        
        # 초기화
        self._initialize_llm()
        self._initialize_tools()
        self._initialize_agent()
        
        logger.info("HSP LangChain Agent 초기화 완료")
    
    def _initialize_llm(self):
        """LLM 초기화"""
        try:
            # OpenAI 모델 사용 (환경변수에서 API 키 로드)
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다")
            
            self.llm = ChatOpenAI(
                model=self.llm_config.get("model", "gpt-4"),
                temperature=self.llm_config.get("temperature", 0.7),
                openai_api_key=api_key
            )
            logger.info(f"LLM 초기화 완료: {self.llm_config.get('model')}")
            
        except Exception as e:
            logger.error(f"LLM 초기화 실패: {e}")
            raise ValueError(f"LLM 초기화 실패: {e}")
    
    def _initialize_tools(self):
        """도구 초기화"""
        self.tools = [
            analyze_user_profile,
            search_hobbies,
            find_communities,
            optimize_schedule
        ]
        logger.info(f"{len(self.tools)}개의 도구 초기화 완료")
    
    def _initialize_agent(self):
        """에이전트 초기화"""
        try:
            if not self.llm:
                logger.warning("LLM이 초기화되지 않아 에이전트를 생성할 수 없습니다.")
                return
            
            # 메모리 초기화
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # 프롬프트 템플릿
            prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 취미 추천 전문가입니다. 
                사용자의 요청에 따라 적절한 도구를 사용하여 답변해주세요.
                항상 한국어로 답변하고, 구체적이고 실용적인 조언을 제공하세요."""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # 에이전트 생성
            self.agent = create_openai_functions_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            # 에이전트 실행기 생성
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True
            )
            
            logger.info("LangChain 에이전트 초기화 완료")
            
        except Exception as e:
            logger.error(f"에이전트 초기화 실패: {e}")
    
    async def run_agent(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """에이전트 실행"""
        try:
            if not self.agent_executor:
                return {"error": "에이전트가 초기화되지 않았습니다."}
            
            # 컨텍스트가 있으면 입력에 포함
            if context:
                enhanced_input = f"{user_input}\n\n컨텍스트: {json.dumps(context, ensure_ascii=False)}"
            else:
                enhanced_input = user_input
            
            # 에이전트 실행
            result = await self.agent_executor.ainvoke({
                "input": enhanced_input
            })
            
            logger.info("에이전트 실행 완료")
            return {
                "success": True,
                "output": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "chat_history": self.memory.chat_memory.messages
            }
            
        except Exception as e:
            logger.error(f"에이전트 실행 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": "에이전트 실행 중 오류가 발생했습니다."
            }
    
    async def get_hobby_recommendations(self, user_profile: Dict[str, Any], 
                                       preferences: List[str], 
                                       constraints: Dict[str, Any]) -> HobbyRecommendationOutput:
        """취미 추천 에이전트 실행"""
        try:
            # 입력 검증
            input_data = HobbyRecommendationInput(
                user_profile=user_profile,
                preferences=preferences,
                constraints=constraints
            )
            
            # 에이전트 실행
            agent_input = f"""
            사용자 프로필: {json.dumps(user_profile, ensure_ascii=False)}
            선호사항: {', '.join(preferences)}
            제약사항: {json.dumps(constraints, ensure_ascii=False)}
            
            위 정보를 바탕으로 개인화된 취미를 추천해주세요.
            """
            
            result = await self.run_agent(agent_input)
            
            if result["success"]:
                # 결과 파싱 (실제 구현에서는 더 정교한 파싱 필요)
                return HobbyRecommendationOutput(
                    recommendations=[
                        {
                            "name": "추천 취미",
                            "category": "일반",
                            "difficulty": "beginner",
                            "match_score": 0.8
                        }
                    ],
                    reasoning=result["output"],
                    confidence_score=0.8
                )
            else:
                raise Exception(result["error"])
                
        except Exception as e:
            logger.error(f"취미 추천 실패: {e}")
            return HobbyRecommendationOutput(
                recommendations=[],
                reasoning="추천 생성에 실패했습니다.",
                confidence_score=0.0
            )
    
    async def match_communities(self, user_profile: Dict[str, Any], 
                               hobby_interests: List[str], 
                               location: str) -> CommunityMatchingOutput:
        """커뮤니티 매칭 에이전트 실행"""
        try:
            # 입력 검증
            input_data = CommunityMatchingInput(
                user_profile=user_profile,
                hobby_interests=hobby_interests,
                location=location
            )
            
            # 에이전트 실행
            agent_input = f"""
            사용자 프로필: {json.dumps(user_profile, ensure_ascii=False)}
            관심 취미: {', '.join(hobby_interests)}
            활동 지역: {location}
            
            위 정보를 바탕으로 적합한 커뮤니티를 찾아주세요.
            """
            
            result = await self.run_agent(agent_input)
            
            if result["success"]:
                return CommunityMatchingOutput(
                    matched_communities=[
                        {
                            "name": "추천 커뮤니티",
                            "type": "online",
                            "members": 100,
                            "match_score": 0.85
                        }
                    ],
                    match_scores={"추천 커뮤니티": 0.85}
                )
            else:
                raise Exception(result["error"])
                
        except Exception as e:
            logger.error(f"커뮤니티 매칭 실패: {e}")
            return CommunityMatchingOutput(
                matched_communities=[],
                match_scores={}
            )
    
    async def integrate_schedule(self, current_schedule: Dict[str, Any], 
                                hobby_activities: List[Dict[str, Any]], 
                                time_constraints: Dict[str, Any]) -> ScheduleIntegrationOutput:
        """스케줄 통합 에이전트 실행"""
        try:
            # 입력 검증
            input_data = ScheduleIntegrationInput(
                current_schedule=current_schedule,
                hobby_activities=hobby_activities,
                time_constraints=time_constraints
            )
            
            # 에이전트 실행
            agent_input = f"""
            현재 스케줄: {json.dumps(current_schedule, ensure_ascii=False)}
            추가할 취미 활동: {json.dumps(hobby_activities, ensure_ascii=False)}
            시간 제약: {json.dumps(time_constraints, ensure_ascii=False)}
            
            위 정보를 바탕으로 스케줄을 최적화하고 통합해주세요.
            """
            
            result = await self.run_agent(agent_input)
            
            if result["success"]:
                return ScheduleIntegrationOutput(
                    integrated_schedule={
                        "weekdays": {"evening": ["통합된 활동"]},
                        "weekends": {"morning": ["새로운 취미"]}
                    },
                    optimization_suggestions=["시간 효율성 향상", "활동 균형 조정"]
                )
            else:
                raise Exception(result["error"])
                
        except Exception as e:
            logger.error(f"스케줄 통합 실패: {e}")
            return ScheduleIntegrationOutput(
                integrated_schedule={},
                optimization_suggestions=[]
            )
    
    def get_agent_status(self) -> Dict[str, Any]:
        """에이전트 상태 정보"""
        return {
            "initialized": self.agent is not None,
            "llm_available": self.llm is not None,
            "tools_count": len(self.tools),
            "memory_size": len(self.memory.chat_memory.messages) if self.memory else 0,
            "last_execution": datetime.now().isoformat()
        }
    
    def clear_memory(self):
        """메모리 초기화"""
        if self.memory:
            self.memory.clear()
            logger.info("에이전트 메모리 초기화 완료")
