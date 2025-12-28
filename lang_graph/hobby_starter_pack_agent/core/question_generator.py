"""
적응형 질문 생성 모듈
사용자 답변을 분석하여 다음 질문을 동적으로 생성하고 정보 수집 완성도를 평가합니다.
"""

import logging
import re
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """적응형 질문 생성 및 정보 수집 완성도 평가"""
    
    # 질문 카테고리 정의
    QUESTION_CATEGORIES = {
        "basic_info": {
            "name": "기본 정보",
            "questions": [
                "나이를 알려주세요.",
                "직업은 무엇인가요?",
                "거주 지역은 어디인가요?"
            ],
            "fields": ["age", "occupation", "location"],
            "weight": 0.2
        },
        "time_availability": {
            "name": "시간 가용성",
            "questions": [
                "평일과 주말 중 언제 시간이 나시나요?",
                "하루에 취미 활동에 투자할 수 있는 시간은 얼마나 되나요?",
                "정기적으로 활동할 수 있는 요일이 있나요?"
            ],
            "fields": ["available_days", "daily_hours", "regular_days"],
            "weight": 0.25
        },
        "interests": {
            "name": "관심사",
            "questions": [
                "현재 관심 있는 취미나 활동이 있나요?",
                "과거에 시도해본 취미가 있나요?",
                "특별히 관심 있는 분야나 주제가 있나요?"
            ],
            "fields": ["current_interests", "past_experiences", "topic_interests"],
            "weight": 0.25
        },
        "constraints": {
            "name": "제약사항",
            "questions": [
                "취미 활동에 투자할 수 있는 예산이 있나요?",
                "활동할 수 있는 공간이 제한되어 있나요?",
                "건강상 제약사항이 있나요?"
            ],
            "fields": ["budget", "space_constraints", "health_constraints"],
            "weight": 0.15
        },
        "goals": {
            "name": "목표",
            "questions": [
                "취미를 통해 달성하고 싶은 목표가 있나요? (예: 스트레스 해소, 사회성 향상, 창의성 발휘)",
                "취미 활동을 통해 얻고 싶은 것이 있나요?",
                "취미를 시작하는 이유는 무엇인가요?"
            ],
            "fields": ["primary_goals", "desired_outcomes", "motivation"],
            "weight": 0.15
        }
    }
    
    def __init__(self):
        logger.info("QuestionGenerator 초기화 완료")
    
    def analyze_user_response(self, response: str, current_category: str, 
                             collected_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 답변을 분석하여 정보를 추출하고 다음 질문을 결정합니다.
        
        Args:
            response: 사용자 답변
            current_category: 현재 질문 카테고리
            collected_preferences: 현재까지 수집된 선호도 정보
            
        Returns:
            분석 결과 딕셔너리
        """
        try:
            # 답변에서 정보 추출
            extracted_info = self._extract_information(response, current_category)
            
            # 수집된 정보 업데이트
            updated_preferences = {**collected_preferences}
            if current_category in self.QUESTION_CATEGORIES:
                category_fields = self.QUESTION_CATEGORIES[current_category]["fields"]
                for field in category_fields:
                    if field in extracted_info:
                        updated_preferences[field] = extracted_info[field]
            
            # 다음 질문 결정
            next_question_info = self._determine_next_question(updated_preferences)
            
            return {
                "extracted_info": extracted_info,
                "updated_preferences": updated_preferences,
                "next_question": next_question_info["question"],
                "next_category": next_question_info["category"],
                "completeness_score": self._calculate_completeness_score(updated_preferences)
            }
            
        except Exception as e:
            logger.error(f"사용자 답변 분석 실패: {e}")
            return {
                "extracted_info": {},
                "updated_preferences": collected_preferences,
                "next_question": "죄송합니다. 다시 한 번 답변해주실 수 있나요?",
                "next_category": current_category,
                "completeness_score": self._calculate_completeness_score(collected_preferences)
            }
    
    def _extract_information(self, response: str, category: str) -> Dict[str, Any]:
        """답변에서 구조화된 정보 추출"""
        extracted = {}
        
        if category == "basic_info":
            # 나이 추출
            age_match = re.search(r'\d+', response)
            if age_match:
                extracted["age"] = int(age_match.group())
            
            # 직업 키워드 매칭
            occupation_keywords = {
                "학생": "student",
                "직장인": "office_worker",
                "교사": "teacher",
                "의사": "doctor",
                "엔지니어": "engineer",
                "디자이너": "designer"
            }
            for keyword, value in occupation_keywords.items():
                if keyword in response:
                    extracted["occupation"] = value
                    break
            
            # 지역 추출 (간단한 키워드 매칭)
            location_keywords = ["서울", "부산", "대구", "인천", "광주", "대전", "울산"]
            for location in location_keywords:
                if location in response:
                    extracted["location"] = location
                    break
        
        elif category == "time_availability":
            if "평일" in response or "주중" in response:
                extracted["available_days"] = "weekdays"
            elif "주말" in response:
                extracted["available_days"] = "weekends"
            elif "매일" in response or "언제든" in response:
                extracted["available_days"] = "anytime"
            
            # 시간 추출
            time_match = re.search(r'(\d+)\s*시간', response)
            if time_match:
                extracted["daily_hours"] = float(time_match.group(1))
        
        elif category == "interests":
            # 관심사 키워드 추출 (간단한 예시)
            interest_keywords = [
                "독서", "운동", "요리", "그림", "음악", "사진", "여행",
                "게임", "영화", "드라마", "등산", "자전거", "수영"
            ]
            found_interests = [kw for kw in interest_keywords if kw in response]
            if found_interests:
                extracted["current_interests"] = found_interests
        
        elif category == "constraints":
            # 예산 추출
            budget_match = re.search(r'(\d+)\s*만원|(\d+)\s*원', response)
            if budget_match:
                amount = budget_match.group(1) or budget_match.group(2)
                if amount:
                    extracted["budget"] = int(amount)
            
            if "공간" in response or "장소" in response:
                extracted["space_constraints"] = True
        
        elif category == "goals":
            goal_keywords = {
                "스트레스": "stress_relief",
                "사회성": "socialization",
                "창의성": "creativity",
                "건강": "health",
                "취미": "hobby"
            }
            found_goals = []
            for keyword, value in goal_keywords.items():
                if keyword in response:
                    found_goals.append(value)
            if found_goals:
                extracted["primary_goals"] = found_goals
        
        return extracted
    
    def _determine_next_question(self, collected_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """수집된 정보를 바탕으로 다음 질문 결정"""
        # 각 카테고리별 정보 수집 상태 확인
        category_status = {}
        for category, config in self.QUESTION_CATEGORIES.items():
            collected_fields = sum(1 for field in config["fields"] 
                                 if field in collected_preferences)
            category_status[category] = {
                "collected": collected_fields,
                "total": len(config["fields"]),
                "completeness": collected_fields / len(config["fields"]) if config["fields"] else 0
            }
        
        # 가장 완성도가 낮은 카테고리 선택
        min_completeness = min(cat["completeness"] for cat in category_status.values())
        next_category = min(category_status.items(), 
                          key=lambda x: x[1]["completeness"])[0]
        
        # 해당 카테고리의 수집되지 않은 필드 찾기
        category_config = self.QUESTION_CATEGORIES[next_category]
        missing_fields = [field for field in category_config["fields"]
                         if field not in collected_preferences]
        
        # 질문 선택 (간단한 로직: 첫 번째 미수집 필드에 해당하는 질문)
        question_index = 0
        if missing_fields:
            # 필드 인덱스에 해당하는 질문 선택
            field_index = category_config["fields"].index(missing_fields[0])
            question_index = min(field_index, len(category_config["questions"]) - 1)
        
        question = category_config["questions"][question_index]
        
        return {
            "question": question,
            "category": next_category,
            "category_name": category_config["name"]
        }
    
    def _calculate_completeness_score(self, collected_preferences: Dict[str, Any]) -> float:
        """정보 수집 완성도 점수 계산 (0.0 ~ 1.0)"""
        total_score = 0.0
        total_weight = 0.0
        
        for category, config in self.QUESTION_CATEGORIES.items():
            collected_fields = sum(1 for field in config["fields"]
                                 if field in collected_preferences and 
                                 collected_preferences[field] is not None)
            category_completeness = collected_fields / len(config["fields"]) if config["fields"] else 0
            
            category_score = category_completeness * config["weight"]
            total_score += category_score
            total_weight += config["weight"]
        
        # 정규화
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        return min(1.0, max(0.0, final_score))
    
    def generate_initial_question(self, user_input: str) -> Dict[str, Any]:
        """초기 사용자 입력을 바탕으로 첫 질문 생성"""
        # 사용자 입력에서 이미 제공된 정보 추출
        initial_preferences = {}
        
        # 기본 정보 카테고리부터 시작
        first_category = "basic_info"
        category_config = self.QUESTION_CATEGORIES[first_category]
        first_question = category_config["questions"][0]
        
        return {
            "question": first_question,
            "category": first_category,
            "category_name": category_config["name"],
            "collected_preferences": initial_preferences
        }
    
    def should_continue_conversation(self, completeness_score: float, 
                                     min_score: float = 0.7) -> bool:
        """대화를 계속할지 결정"""
        return completeness_score < min_score

