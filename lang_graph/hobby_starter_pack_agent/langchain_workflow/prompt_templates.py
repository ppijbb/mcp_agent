"""
LangChain 기반 프롬프트 템플릿 시스템
컨텍스트 정보를 포함한 최적화된 프롬프트 생성
"""

import logging
from typing import Dict, Any, List, Optional
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
import json

logger = logging.getLogger(__name__)

class HSPPromptTemplates:
    """HSP Agent 전용 프롬프트 템플릿"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        logger.info("HSP Prompt Templates 초기화 완료")
    
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """프롬프트 템플릿 초기화"""
        
        # 1. 프로필 분석 프롬프트
        profile_analysis_template = PromptTemplate(
            input_variables=["user_input", "conversation_history", "user_context"],
            template="""
당신은 사용자 프로필 분석 전문가입니다.

=== 대화 기록 ===
{conversation_history}

=== 사용자 컨텍스트 ===
{user_context}

=== 현재 사용자 입력 ===
{user_input}

위의 정보를 바탕으로 사용자의 프로필을 분석해주세요.

다음 JSON 형식으로 응답해주세요:
{{
    "interests": ["관심사1", "관심사2", "관심사3"],
    "personality_type": "성격 유형 (introvert/extrovert/ambivert)",
    "skill_level": "경험 수준 (beginner/intermediate/advanced)",
    "time_availability": "가용 시간 (weekday/evening/weekend/flexible)",
    "budget_range": "예산 범위 (low/medium/high)",
    "location_preference": "활동 선호 지역",
    "confidence_score": 0.85,
    "analysis_reasoning": "분석 근거 설명"
}}
"""
        )
        
        # 2. 취미 추천 프롬프트
        hobby_recommendation_template = PromptTemplate(
            input_variables=["user_profile", "available_hobbies", "conversation_history"],
            template="""
당신은 개인화된 취미 추천 전문가입니다.

=== 사용자 프로필 ===
{user_profile}

=== 사용 가능한 취미 목록 ===
{available_hobbies}

=== 대화 기록 ===
{conversation_history}

사용자에게 가장 적합한 취미 5개를 추천해주세요.

다음 JSON 형식으로 응답해주세요:
{{
    "recommendations": [
        {{
            "hobby_name": "취미 이름",
            "category": "카테고리",
            "difficulty": "난이도",
            "time_commitment": "시간 투자",
            "cost": "비용",
            "match_score": 0.9,
            "reason": "추천 이유",
            "getting_started": "시작 방법"
        }}
    ],
    "overall_reasoning": "전체적인 추천 근거",
    "next_steps": ["다음 단계 제안1", "다음 단계 제안2"]
}}
"""
        )
        
        # 3. 커뮤니티 매칭 프롬프트
        community_matching_template = PromptTemplate(
            input_variables=["user_profile", "hobby_recommendations", "available_communities"],
            template="""
당신은 커뮤니티 매칭 전문가입니다.

=== 사용자 프로필 ===
{user_profile}

=== 추천된 취미 ===
{hobby_recommendations}

=== 사용 가능한 커뮤니티 ===
{available_communities}

사용자에게 가장 적합한 커뮤니티를 매칭해주세요.

다음 JSON 형식으로 응답해주세요:
{{
    "matched_communities": [
        {{
            "community_name": "커뮤니티 이름",
            "type": "온라인/오프라인",
            "members": 150,
            "activity_level": "활성도",
            "match_score": 0.85,
            "why_suitable": "적합한 이유",
            "how_to_join": "참여 방법"
        }}
    ],
    "matching_strategy": "매칭 전략 설명"
}}
"""
        )
        
        # 4. 스케줄 통합 프롬프트
        schedule_integration_template = PromptTemplate(
            input_variables=["user_profile", "hobby_recommendations", "current_schedule"],
            template="""
당신은 일정 통합 전문가입니다.

=== 사용자 프로필 ===
{user_profile}

=== 추천된 취미 ===
{hobby_recommendations}

=== 현재 스케줄 ===
{current_schedule}

취미 활동을 일상 스케줄에 효율적으로 통합하는 방안을 제시해주세요.

다음 JSON 형식으로 응답해주세요:
{{
    "integration_plan": [
        {{
            "hobby": "취미 이름",
            "suggested_time": "제안 시간",
            "frequency": "빈도",
            "duration": "소요 시간",
            "integration_tips": "통합 팁"
        }}
    ],
    "schedule_optimization": "스케줄 최적화 제안",
    "potential_conflicts": ["잠재적 충돌 사항들"],
    "conflict_resolution": "충돌 해결 방안"
}}
"""
        )
        
        # 5. 진행상황 추적 프롬프트
        progress_tracking_template = PromptTemplate(
            input_variables=["user_profile", "hobby_activities", "progress_data"],
            template="""
당신은 진행상황 추적 전문가입니다.

=== 사용자 프로필 ===
{user_profile}

=== 취미 활동 기록 ===
{hobby_activities}

=== 진행상황 데이터 ===
{progress_data}

사용자의 취미 활동 진행상황을 분석하고 인사이트를 제공해주세요.

다음 JSON 형식으로 응답해주세요:
{{
    "progress_summary": {{
        "overall_progress": 0.75,
        "completed_activities": 15,
        "total_planned": 20,
        "streak_days": 7
    }},
    "key_insights": ["주요 인사이트들"],
    "achievements": ["달성한 성과들"],
    "improvement_areas": ["개선 영역들"],
    "next_goals": ["다음 목표들"],
    "motivational_message": "격려 메시지"
}}
"""
        )
        
        return {
            "profile_analysis": profile_analysis_template,
            "hobby_recommendation": hobby_recommendation_template,
            "community_matching": community_matching_template,
            "schedule_integration": schedule_integration_template,
            "progress_tracking": progress_tracking_template
        }
    
    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """템플릿 이름으로 템플릿 조회"""
        return self.templates.get(template_name)
    
    def create_chat_prompt(self, template_name: str, **kwargs) -> ChatPromptTemplate:
        """채팅 형식의 프롬프트 생성"""
        try:
            template = self.get_template(template_name)
            if not template:
                raise ValueError(f"템플릿을 찾을 수 없습니다: {template_name}")
            
            # 시스템 메시지와 사용자 메시지로 분리
            system_message = SystemMessage(content=template.template)
            human_message = HumanMessage(content=template.format(**kwargs))
            
            return ChatPromptTemplate.from_messages([
                system_message,
                human_message
            ])
            
        except Exception as e:
            logger.error(f"채팅 프롬프트 생성 실패: {e}")
            # 기본 프롬프트 반환
            return ChatPromptTemplate.from_messages([
                SystemMessage(content="당신은 도움이 되는 AI 어시스턴트입니다."),
                HumanMessage(content=kwargs.get("user_input", "안녕하세요"))
            ])
    
    def create_memory_aware_prompt(self, template_name: str, user_id: str, 
                                  conversation_history: List[Dict], **kwargs) -> ChatPromptTemplate:
        """메모리를 고려한 프롬프트 생성"""
        try:
            template = self.get_template(template_name)
            if not template:
                raise ValueError(f"템플릿을 찾을 수 없습니다: {template_name}")
            
            # 대화 기록을 MessagesPlaceholder로 추가
            messages = [
                SystemMessage(content=template.template),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content=template.format(**kwargs))
            ]
            
            return ChatPromptTemplate.from_messages(messages)
            
        except Exception as e:
            logger.error(f"메모리 인식 프롬프트 생성 실패: {e}")
            return self.create_chat_prompt(template_name, **kwargs)
    
    def create_dynamic_prompt(self, base_template: str, dynamic_variables: Dict[str, Any]) -> str:
        """동적 변수를 포함한 프롬프트 생성"""
        try:
            # 기본 템플릿에 동적 변수 삽입
            prompt = base_template
            
            for key, value in dynamic_variables.items():
                placeholder = f"{{{key}}}"
                if placeholder in prompt:
                    if isinstance(value, (dict, list)):
                        prompt = prompt.replace(placeholder, json.dumps(value, ensure_ascii=False, indent=2))
                    else:
                        prompt = prompt.replace(placeholder, str(value))
            
            return prompt
            
        except Exception as e:
            logger.error(f"동적 프롬프트 생성 실패: {e}")
            return base_template
    
    def validate_template_variables(self, template_name: str, provided_vars: Dict[str, Any]) -> Dict[str, Any]:
        """템플릿 변수 검증 및 기본값 설정"""
        try:
            template = self.get_template(template_name)
            if not template:
                return provided_vars
            
            required_vars = template.input_variables
            validated_vars = {}
            
            for var in required_vars:
                if var in provided_vars:
                    validated_vars[var] = provided_vars[var]
                else:
                    # 기본값 설정
                    default_values = {
                        "conversation_history": "대화 기록이 없습니다.",
                        "user_context": "사용자 컨텍스트가 없습니다.",
                        "available_hobbies": "사용 가능한 취미 정보가 없습니다.",
                        "current_schedule": "현재 스케줄 정보가 없습니다.",
                        "progress_data": "진행상황 데이터가 없습니다."
                    }
                    validated_vars[var] = default_values.get(var, "")
            
            return validated_vars
            
        except Exception as e:
            logger.error(f"템플릿 변수 검증 실패: {e}")
            return provided_vars
