import json
from typing import Dict, Any, List
from datetime import datetime
import asyncio
import hashlib
import logging
from .agents import HSPAutoGenAgents

# Logger 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutoGenDecisionEngine:
    """최적화된 AutoGen 의사결정 엔진"""
    
    def __init__(self):
        self.agents = HSPAutoGenAgents()
        self.llm_cache = {}  # LLM 응답 캐시
        logger.info("AutoGenDecisionEngine 초기화 완료")
        
    async def _call_llm_with_cache(self, prompt: str, agent_type: str = "general") -> str:
        """캐시가 적용된 LLM 호출"""
        cache_key = hashlib.md5(f"{prompt}:{agent_type}".encode()).hexdigest()
        
        if cache_key in self.llm_cache:
            return self.llm_cache[cache_key]
        
        try:
            # 실제 LLM API 호출 (OpenAI/Claude 등)
            import openai
            client = openai.AsyncOpenAI()
            
            response = await client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": f"{agent_type} 전문가로서 답변해주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            result = response.choices[0].message.content
            
            # 캐시 저장 (메모리 제한 고려)
            if len(self.llm_cache) < 100:
                self.llm_cache[cache_key] = result
                
            return result
            
        except Exception as e:
            logger.error(f"LLM 호출 실패: {e}")
            # 안전한 폴백 (빈 값보다는 의미있는 기본값)
            return self._get_fallback_response(prompt, agent_type)
    
    def _get_fallback_response(self, prompt: str, agent_type: str) -> str:
        """LLM 실패시 의미있는 폴백 응답"""
        fallback_responses = {
            "profile_analyst": "일반적인 취미 활동에 관심이 있으신 것 같습니다.",
            "hobby_discoverer": "다양한 취미 옵션을 탐색해보시는 것을 추천합니다.",
            "community_matcher": "지역 커뮤니티 참여를 고려해보세요."
        }
        return fallback_responses.get(agent_type, "추가 정보가 필요합니다.")
    
    async def analyze_user_profile(self, user_input: str) -> Dict[str, Any]:
        """사용자 프로필 분석 - 병렬 처리"""
        try:
            # 병렬로 다양한 관점에서 분석
            analysis_tasks = [
                self._call_llm_with_cache(
                    f"다음 입력에서 관심사를 추출해주세요: {user_input}", 
                    "profile_analyst"
                ),
                self._call_llm_with_cache(
                    f"다음 사용자의 경험 수준을 평가해주세요: {user_input}", 
                    "skill_assessor"
                ),
                self._call_llm_with_cache(
                    f"시간 가용성을 분석해주세요: {user_input}", 
                    "schedule_analyst"
                )
            ]
            
            results = await asyncio.gather(*analysis_tasks)
            
            # 결과 구조화
            return {
                "interests": self._extract_interests(results[0]),
                "skill_level": self._extract_skill_level(results[1]),
                "time_availability": self._extract_time_availability(results[2]),
                "raw_analysis": results
            }
            
        except Exception as e:
            logger.error(f"프로필 분석 실패: {e}")
            return {
                "interests": ["general"],
                "skill_level": "beginner", 
                "time_availability": "flexible"
            }
    
    async def filter_hobbies(self, hobby_list: List[Dict], user_profile: Dict) -> List[Dict]:
        """취미 필터링 및 개인화"""
        try:
            # 사용자 프로필 기반 필터링 프롬프트
            filter_prompt = f"""
            사용자 프로필: {user_profile}
            취미 목록: {hobby_list}
            
            사용자에게 가장 적합한 취미 5개를 선별하고 각각의 이유를 설명해주세요.
            """
            
            filtered_result = await self._call_llm_with_cache(filter_prompt, "hobby_discoverer")
            
            # 결과 파싱 및 구조화
            return self._parse_hobby_recommendations(filtered_result, hobby_list)
            
        except Exception as e:
            logger.error(f"취미 필터링 실패: {e}")
            # 안전한 폴백: 첫 3개 취미 반환
            return hobby_list[:3] if hobby_list else []
    
    def _extract_interests(self, analysis_text: str) -> List[str]:
        """관심사 추출 로직"""
        # 키워드 매칭을 통한 관심사 추출
        interest_keywords = {
            "music": ["음악", "악기", "노래", "밴드"],
            "sports": ["운동", "스포츠", "헬스", "피트니스"],
            "art": ["그림", "미술", "창작", "디자인"],
            "technology": ["코딩", "프로그래밍", "IT", "개발"],
            "cooking": ["요리", "베이킹", "음식", "레시피"]
        }
        
        detected_interests = []
        for category, keywords in interest_keywords.items():
            if any(keyword in analysis_text for keyword in keywords):
                detected_interests.append(category)
        
        return detected_interests if detected_interests else ["general"]
    
    def _extract_skill_level(self, analysis_text: str) -> str:
        """경험 수준 추출"""
        if any(word in analysis_text for word in ["초보", "처음", "시작"]):
            return "beginner"
        elif any(word in analysis_text for word in ["중급", "어느정도", "경험"]):
            return "intermediate"
        elif any(word in analysis_text for word in ["고급", "전문", "숙련"]):
            return "advanced"
        return "beginner"
    
    def _extract_time_availability(self, analysis_text: str) -> str:
        """시간 가용성 추출"""
        if any(word in analysis_text for word in ["주말", "토요일", "일요일"]):
            return "weekend"
        elif any(word in analysis_text for word in ["저녁", "퇴근", "밤"]):
            return "evening"
        elif any(word in analysis_text for word in ["평일", "오전", "점심"]):
            return "weekday"
        return "flexible"
    
    def _parse_hobby_recommendations(self, llm_result: str, original_list: List[Dict]) -> List[Dict]:
        """LLM 결과에서 취미 추천 파싱"""
        try:
            # LLM 응답에서 추천된 취미명 추출
            recommended_names = []
            lines = llm_result.split('\n')
            
            for line in lines:
                # "1. 독서", "- 등산" 같은 패턴에서 취미명 추출
                if any(hobby['name'] in line for hobby in original_list):
                    for hobby in original_list:
                        if hobby['name'] in line and hobby not in recommended_names:
                            recommended_names.append(hobby)
                            break
            
            # 추천된 취미가 없으면 원본 리스트에서 상위 3개
            if not recommended_names and original_list:
                recommended_names = original_list[:3]
            
            return recommended_names
            
        except Exception as e:
            logger.error(f"취미 추천 파싱 실패: {e}")
            return original_list[:3] if original_list else []
    
    async def generate_final_recommendations(self, user_profile: Dict, hobbies: List[Dict], 
                                           communities: List[Dict]) -> List[Dict]:
        """최종 추천 생성"""
        try:
            final_prompt = f"""
            사용자 프로필, 취미 후보, 커뮤니티 정보를 종합하여 최종 추천을 생성해주세요:
            
            사용자 프로필: {user_profile}
            취미 후보: {hobbies}  
            커뮤니티: {communities}
            
            최종 추천 3개를 우선순위 순으로 제시해주세요.
            """
            
            final_result = await self._call_llm_with_cache(final_prompt, "final_coordinator")
            
            return {
                "final_recommendations": final_result,
                "recommended_hobbies": hobbies[:3],
                "matched_communities": communities[:2]
            }
            
        except Exception as e:
            logger.error(f"최종 추천 생성 실패: {e}")
            return {
                "final_recommendations": "개인화된 취미 추천을 준비했습니다.",
                "recommended_hobbies": hobbies[:3] if hobbies else [],
                "matched_communities": communities[:2] if communities else []
            }

class AgentDecisionEngine:
    """모든 판단을 에이전트가 LLM 호출로 수행하는 엔진"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.decision_history = []
    
    async def make_hobby_recommendation_decision(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """취미 추천 의사결정 - 하드코딩 없이 순수 LLM 판단"""
        
        prompt = f"""
        사용자 컨텍스트를 분석하여 적절한 취미를 추천해주세요.
        
        사용자 정보:
        {json.dumps(user_context, ensure_ascii=False, indent=2)}
        
        다음 사항들을 고려하여 추천해주세요:
        1. 사용자의 성격과 성향
        2. 현재 생활 패턴과 스케줄
        3. 관심사와 선호도
        4. 예산과 시간 제약
        5. 지역적 접근성
        
        응답은 반드시 JSON 형태로 해주세요:
        {{
            "recommendations": [
                {{
                    "hobby_name": "취미 이름",
                    "reason": "추천 이유",
                    "difficulty": "초급/중급/고급",
                    "time_commitment": "시간 투자 정도",
                    "budget_range": "예산 범위",
                    "confidence_score": 0.85
                }}
            ],
            "reasoning": "전체적인 추천 근거"
        }}
        
        만약 충분한 정보가 없다면 빈 배열을 반환해주세요.
        """
        
        try:
            # LLM 호출하여 동적 결정
            response = await self._call_llm(prompt)
            result = json.loads(response) if response else {"recommendations": [], "reasoning": ""}
            
            # 의사결정 이력 저장
            self.decision_history.append({
                "decision_type": "hobby_recommendation",
                "timestamp": datetime.now().isoformat(),
                "input_context": user_context,
                "result": result
            })
            
            return result
            
        except Exception as e:
            # 예외 시 빈 값 반환
            return {"recommendations": [], "reasoning": ""}
    
    async def analyze_schedule_compatibility(self, schedule_data: Dict[str, Any], hobby_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """스케줄 호환성 분석 - LLM 기반 동적 판단"""
        
        prompt = f"""
        사용자의 스케줄과 취미 요구사항을 분석하여 호환성을 판단해주세요.
        
        현재 스케줄:
        {json.dumps(schedule_data, ensure_ascii=False, indent=2)}
        
        취미 요구사항:
        {json.dumps(hobby_requirements, ensure_ascii=False, indent=2)}
        
        다음을 분석해주세요:
        1. 시간적 호환성
        2. 에너지 레벨 매칭
        3. 일정 충돌 가능성
        4. 최적 시간대 제안
        
        응답은 JSON 형태로:
        {{
            "compatibility_score": 0.8,
            "available_time_slots": [
                {{
                    "day": "월요일",
                    "time_range": "19:00-21:00",
                    "confidence": 0.9
                }}
            ],
            "potential_conflicts": ["리스트"],
            "optimization_suggestions": ["제안사항들"],
            "integration_strategy": "통합 전략"
        }}
        
        정보가 부족하면 빈 객체를 반환해주세요.
        """
        
        try:
            response = await self._call_llm(prompt)
            return json.loads(response) if response else {}
        except Exception:
            return {}
    
    async def evaluate_community_match(self, user_profile: Dict[str, Any], community_data: Dict[str, Any]) -> Dict[str, Any]:
        """커뮤니티 매칭 평가 - LLM 기반 동적 판단"""
        
        prompt = f"""
        사용자 프로필과 커뮤니티 정보를 비교하여 매칭 적합성을 평가해주세요.
        
        사용자 프로필:
        {json.dumps(user_profile, ensure_ascii=False, indent=2)}
        
        커뮤니티 정보:
        {json.dumps(community_data, ensure_ascii=False, indent=2)}
        
        다음 기준으로 평가:
        1. 관심사 일치도
        2. 활동 스타일 호환성
        3. 경험 레벨 적합성
        4. 지역적 접근성
        5. 커뮤니티 활성도
        
        JSON 응답:
        {{
            "match_score": 0.85,
            "strength_areas": ["강점 영역들"],
            "concern_areas": ["우려 영역들"],
            "recommendation": "추천/보류/비추천",
            "integration_tips": ["참여 팁들"]
        }}
        
        평가가 어려우면 빈 객체를 반환해주세요.
        """
        
        try:
            response = await self._call_llm(prompt)
            return json.loads(response) if response else {}
        except Exception:
            return {}
    
    async def generate_weekly_insights(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """주간 인사이트 생성 - LLM 기반 동적 생성"""
        
        prompt = f"""
        사용자의 주간 활동 데이터를 분석하여 인사이트를 생성해주세요.
        
        활동 데이터:
        {json.dumps(activity_data, ensure_ascii=False, indent=2)}
        
        다음을 포함한 인사이트 생성:
        1. 성취도 분석
        2. 패턴 발견
        3. 개선 영역
        4. 다음 주 목표 제안
        5. 동기부여 메시지
        
        JSON 응답:
        {{
            "achievement_score": 0.75,
            "key_insights": ["주요 인사이트들"],
            "progress_trends": ["진행 트렌드"],
            "improvement_areas": ["개선 영역"],
            "next_week_goals": ["다음 주 목표들"],
            "motivational_message": "격려 메시지",
            "personalized_journal": "개인화된 일지 내용"
        }}
        
        데이터가 부족하면 빈 값들로 응답해주세요.
        """
        
        try:
            response = await self._call_llm(prompt)
            return json.loads(response) if response else {
                "achievement_score": 0,
                "key_insights": [],
                "progress_trends": [],
                "improvement_areas": [],
                "next_week_goals": [],
                "motivational_message": "",
                "personalized_journal": ""
            }
        except Exception:
            return {
                "achievement_score": 0,
                "key_insights": [],
                "progress_trends": [],
                "improvement_areas": [],
                "next_week_goals": [],
                "motivational_message": "",
                "personalized_journal": ""
            }
    
    async def _call_llm(self, prompt: str) -> str:
        """LLM 호출 - 실제 구현 시 사용할 LLM API"""
        # 실제 LLM API 호출 로직
        # OpenAI, Anthropic, Google Gemini 등
        try:
            # LLM API 호출
            return ""  # 실제 응답
        except Exception:
            return ""  # 실패 시 빈 문자열 