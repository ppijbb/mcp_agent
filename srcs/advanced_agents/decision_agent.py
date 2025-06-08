"""
Decision Agent - 모바일 인터액션 기반 자동 결정 시스템

사용자의 모든 모바일 인터액션을 감지하고, 상황을 분석한 후
사용자 맞춤형 결정을 자동으로 내려주는 AI Agent

주요 기능:
1. 모바일 인터액션 감지 및 분류
2. 개입 여부 판단 로직
3. 사용자 프로필 기반 상황 분석
4. 맞춤형 결정 생성 및 실행
5. 학습 및 최적화
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from anthropic import Anthropic
import requests

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractionType(Enum):
    """모바일 인터액션 타입"""
    APP_OPEN = "app_open"
    NOTIFICATION = "notification"
    PURCHASE = "purchase"
    CALL = "call"
    MESSAGE = "message"
    NAVIGATION = "navigation"
    SEARCH = "search"
    SOCIAL_MEDIA = "social_media"
    SETTINGS = "settings"
    PAYMENT = "payment"
    BOOKING = "booking"
    SHOPPING = "shopping"
    FOOD_ORDER = "food_order"
    UNKNOWN = "unknown"

class DecisionPriority(Enum):
    """결정 우선순위"""
    CRITICAL = "critical"      # 즉시 개입 필요
    HIGH = "high"             # 5초 내 개입
    MEDIUM = "medium"         # 30초 내 개입
    LOW = "low"              # 5분 내 개입
    MONITOR = "monitor"       # 관찰만 함

@dataclass
class UserProfile:
    """사용자 프로필"""
    user_id: str
    name: str
    age: int
    preferences: Dict[str, Any]
    behavior_patterns: Dict[str, Any]
    decision_history: List[Dict[str, Any]]
    financial_profile: Dict[str, Any]
    risk_tolerance: str  # conservative, moderate, aggressive
    values: Dict[str, float]  # 개인 가치관 점수
    goals: List[str]
    constraints: Dict[str, Any]
    
    def to_dict(self):
        return asdict(self)

@dataclass
class MobileInteraction:
    """모바일 인터액션 데이터"""
    timestamp: datetime
    interaction_type: InteractionType
    app_name: str
    context: Dict[str, Any]
    user_location: Optional[Tuple[float, float]]
    device_state: Dict[str, Any]
    urgency_score: float
    metadata: Dict[str, Any]

@dataclass
class DecisionContext:
    """결정 컨텍스트"""
    interaction: MobileInteraction
    user_profile: UserProfile
    current_state: Dict[str, Any]
    historical_data: List[Dict[str, Any]]
    external_factors: Dict[str, Any]
    time_constraints: Optional[datetime]

@dataclass
class Decision:
    """생성된 결정"""
    decision_id: str
    timestamp: datetime
    decision_type: str
    recommendation: str
    confidence_score: float
    reasoning: str
    alternatives: List[str]
    expected_outcome: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    auto_execute: bool
    execution_plan: Optional[Dict[str, Any]]

class DecisionAgent:
    """모바일 인터액션 기반 자동 결정 에이전트"""
    
    def __init__(self, anthropic_api_key: str):
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)
        self.user_profiles: Dict[str, UserProfile] = {}
        self.interaction_buffer: List[MobileInteraction] = []
        self.decision_history: List[Decision] = []
        self.is_monitoring = False
        
        # 개입 임계값 설정
        self.intervention_thresholds = {
            InteractionType.PURCHASE: 0.7,
            InteractionType.PAYMENT: 0.9,
            InteractionType.BOOKING: 0.8,
            InteractionType.CALL: 0.6,
            InteractionType.MESSAGE: 0.4,
            InteractionType.NAVIGATION: 0.5,
            InteractionType.FOOD_ORDER: 0.6,
            InteractionType.SHOPPING: 0.7,
            InteractionType.APP_OPEN: 0.3,
        }
        
        logger.info("Decision Agent 초기화 완료")

    async def start_monitoring(self, user_id: str):
        """모바일 인터액션 모니터링 시작"""
        self.is_monitoring = True
        logger.info(f"사용자 {user_id}의 모바일 인터액션 모니터링 시작")
        
        while self.is_monitoring:
            try:
                # 실제 구현에서는 모바일 OS API나 ADB 연동
                interaction = await self._detect_interaction()
                
                if interaction:
                    await self._process_interaction(interaction, user_id)
                    
                await asyncio.sleep(0.1)  # 100ms 간격으로 체크
                
            except Exception as e:
                logger.error(f"모니터링 중 오류 발생: {e}")
                await asyncio.sleep(1)

    async def _detect_interaction(self) -> Optional[MobileInteraction]:
        """모바일 인터액션 감지 (모의 구현)"""
        # 실제로는 Android AccessibilityService나 iOS ScreenTime API 사용
        # 여기서는 시뮬레이션을 위한 더미 데이터
        
        if np.random.random() < 0.1:  # 10% 확률로 인터액션 발생
            interaction_types = list(InteractionType)
            interaction_type = np.random.choice(interaction_types)
            
            return MobileInteraction(
                timestamp=datetime.now(),
                interaction_type=interaction_type,
                app_name=self._get_app_name(interaction_type),
                context=self._generate_context(interaction_type),
                user_location=(37.5665, 126.9780),  # 서울시청
                device_state={"battery": 85, "network": "WiFi"},
                urgency_score=np.random.random(),
                metadata={}
            )
        return None

    def _get_app_name(self, interaction_type: InteractionType) -> str:
        """인터액션 타입에 따른 앱 이름 생성"""
        app_mapping = {
            InteractionType.PURCHASE: "쿠팡",
            InteractionType.PAYMENT: "토스",
            InteractionType.BOOKING: "야놀자",
            InteractionType.CALL: "전화",
            InteractionType.MESSAGE: "카카오톡",
            InteractionType.NAVIGATION: "네이버 지도",
            InteractionType.FOOD_ORDER: "배달의민족",
            InteractionType.SHOPPING: "네이버 쇼핑",
            InteractionType.SOCIAL_MEDIA: "인스타그램",
        }
        return app_mapping.get(interaction_type, "알 수 없음")

    def _generate_context(self, interaction_type: InteractionType) -> Dict[str, Any]:
        """인터액션 타입에 따른 컨텍스트 생성"""
        if interaction_type == InteractionType.PURCHASE:
            return {
                "product": "무선 이어폰",
                "price": 150000,
                "discount": 0.15,
                "seller_rating": 4.5,
                "reviews_count": 1250
            }
        elif interaction_type == InteractionType.FOOD_ORDER:
            return {
                "restaurant": "맛있는 치킨집",
                "menu": "후라이드 치킨",
                "price": 18000,
                "delivery_time": 25,
                "rating": 4.2
            }
        elif interaction_type == InteractionType.BOOKING:
            return {
                "hotel": "서울 호텔",
                "check_in": "2024-02-15",
                "check_out": "2024-02-17",
                "price": 120000,
                "rating": 4.1
            }
        return {}

    async def _process_interaction(self, interaction: MobileInteraction, user_id: str):
        """인터액션 처리"""
        try:
            # 1. 개입 여부 판단
            should_intervene = await self._should_intervene(interaction, user_id)
            
            if not should_intervene:
                logger.info(f"개입하지 않음: {interaction.interaction_type.value}")
                return
            
            logger.info(f"개입 결정: {interaction.interaction_type.value} in {interaction.app_name}")
            
            # 2. 사용자 프로필 로드
            user_profile = await self._get_user_profile(user_id)
            
            # 3. 결정 컨텍스트 구성
            context = await self._build_decision_context(interaction, user_profile)
            
            # 4. 결정 생성
            decision = await self._generate_decision(context)
            
            # 5. 결정 실행 또는 추천
            await self._execute_or_recommend_decision(decision, user_id)
            
            # 6. 결정 이력 저장
            self.decision_history.append(decision)
            
        except Exception as e:
            logger.error(f"인터액션 처리 중 오류: {e}")

    async def _should_intervene(self, interaction: MobileInteraction, user_id: str) -> bool:
        """개입 여부 판단"""
        # 1. 기본 임계값 체크
        threshold = self.intervention_thresholds.get(interaction.interaction_type, 0.5)
        
        # 2. 긴급도 점수 계산
        urgency_factors = {
            "high_value": interaction.context.get("price", 0) > 100000,
            "time_sensitive": interaction.interaction_type in [
                InteractionType.CALL, InteractionType.PAYMENT
            ],
            "financial_impact": interaction.interaction_type in [
                InteractionType.PURCHASE, InteractionType.BOOKING, InteractionType.PAYMENT
            ],
            "irreversible": interaction.interaction_type in [
                InteractionType.PAYMENT, InteractionType.CALL
            ]
        }
        
        urgency_score = sum(urgency_factors.values()) / len(urgency_factors)
        
        # 3. 사용자별 개입 패턴 고려
        user_profile = await self._get_user_profile(user_id)
        if user_profile:
            personal_threshold = user_profile.preferences.get("intervention_threshold", threshold)
            threshold = (threshold + personal_threshold) / 2
        
        # 4. 최종 판단
        final_score = (urgency_score + interaction.urgency_score) / 2
        should_intervene = final_score >= threshold
        
        logger.info(f"개입 판정: {should_intervene} (점수: {final_score:.2f}, 임계값: {threshold:.2f})")
        return should_intervene

    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """사용자 프로필 로드 또는 생성"""
        if user_id not in self.user_profiles:
            # 기본 프로필 생성 (실제로는 DB에서 로드)
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                name="사용자",
                age=30,
                preferences={
                    "intervention_threshold": 0.6,
                    "auto_execute_threshold": 0.8,
                    "budget_daily": 50000,
                    "budget_monthly": 1500000,
                    "preferred_brands": ["삼성", "애플", "네이버"],
                    "dietary_restrictions": [],
                    "travel_preferences": ["호텔", "가성비"]
                },
                behavior_patterns={
                    "active_hours": (9, 22),
                    "frequent_apps": ["카카오톡", "네이버", "유튜브"],
                    "purchase_patterns": {"electronics": 0.3, "food": 0.4, "travel": 0.2, "other": 0.1},
                    "decision_speed": "moderate"
                },
                decision_history=[],
                financial_profile={
                    "monthly_income": 4000000,
                    "savings_rate": 0.3,
                    "investment_portfolio": {"stocks": 0.6, "bonds": 0.3, "cash": 0.1}
                },
                risk_tolerance="moderate",
                values={
                    "efficiency": 0.8,
                    "cost_saving": 0.7,
                    "quality": 0.9,
                    "convenience": 0.8,
                    "health": 0.9,
                    "relationships": 0.9,
                    "career": 0.8,
                    "learning": 0.7
                },
                goals=["건강 관리", "재정 최적화", "시간 절약", "업무 효율성"],
                constraints={
                    "time": {"daily_free_time": 3},
                    "budget": {"discretionary": 500000},
                    "health": {"allergies": []}
                }
            )
        
        return self.user_profiles[user_id]

    async def _build_decision_context(self, interaction: MobileInteraction, user_profile: UserProfile) -> DecisionContext:
        """결정 컨텍스트 구성"""
        # 현재 상태 정보 수집
        current_state = {
            "time": interaction.timestamp,
            "location": interaction.user_location,
            "device_state": interaction.device_state,
            "recent_activity": self._get_recent_activity(user_profile.user_id),
            "current_mood": self._estimate_mood(interaction),
            "budget_status": self._check_budget_status(user_profile),
        }
        
        # 외부 요인 수집
        external_factors = {
            "weather": await self._get_weather_info(interaction.user_location),
            "traffic": await self._get_traffic_info(interaction.user_location),
            "market_conditions": await self._get_market_info(),
            "social_trends": await self._get_trend_info(),
        }
        
        return DecisionContext(
            interaction=interaction,
            user_profile=user_profile,
            current_state=current_state,
            historical_data=self._get_historical_data(interaction.interaction_type, user_profile),
            external_factors=external_factors,
            time_constraints=None
        )

    def _get_recent_activity(self, user_id: str) -> List[Dict[str, Any]]:
        """최근 활동 내역 조회"""
        # 최근 1시간 내 인터액션들
        recent_interactions = [
            interaction for interaction in self.interaction_buffer
            if (datetime.now() - interaction.timestamp).seconds < 3600
        ]
        return [{"type": i.interaction_type.value, "app": i.app_name, "time": i.timestamp} 
                for i in recent_interactions]

    def _estimate_mood(self, interaction: MobileInteraction) -> str:
        """사용자 기분 추정"""
        hour = interaction.timestamp.hour
        interaction_type = interaction.interaction_type
        
        if hour < 6 or hour > 23:
            return "tired"
        elif interaction_type in [InteractionType.SHOPPING, InteractionType.FOOD_ORDER]:
            return "relaxed"
        elif interaction_type in [InteractionType.CALL, InteractionType.MESSAGE]:
            return "social"
        else:
            return "neutral"

    def _check_budget_status(self, user_profile: UserProfile) -> Dict[str, Any]:
        """예산 상태 확인"""
        # 실제로는 가계부 앱이나 은행 API 연동
        return {
            "daily_spent": 35000,
            "daily_limit": user_profile.preferences.get("budget_daily", 50000),
            "monthly_spent": 850000,
            "monthly_limit": user_profile.preferences.get("budget_monthly", 1500000),
            "available": True
        }

    async def _get_weather_info(self, location: Optional[Tuple[float, float]]) -> Dict[str, Any]:
        """날씨 정보 조회"""
        # 실제로는 날씨 API 호출
        return {
            "temperature": 15,
            "condition": "cloudy",
            "humidity": 60,
            "precipitation": 0
        }

    async def _get_traffic_info(self, location: Optional[Tuple[float, float]]) -> Dict[str, Any]:
        """교통 정보 조회"""
        return {
            "congestion_level": "moderate",
            "travel_time_factor": 1.2
        }

    async def _get_market_info(self) -> Dict[str, Any]:
        """시장 정보 조회"""
        return {
            "stock_market": "stable",
            "currency_rate": {"USD": 1300, "JPY": 9.8},
            "oil_price": 80.5
        }

    async def _get_trend_info(self) -> Dict[str, Any]:
        """트렌드 정보 조회"""
        return {
            "popular_products": ["무선 이어폰", "스마트 워치"],
            "seasonal_trends": ["겨울 의류", "난방 용품"],
            "social_buzz": ["환경 친화", "건강 관리"]
        }

    def _get_historical_data(self, interaction_type: InteractionType, user_profile: UserProfile) -> List[Dict[str, Any]]:
        """과거 데이터 조회"""
        # 유사한 상황에서의 과거 결정들
        similar_decisions = [
            decision for decision in user_profile.decision_history
            if decision.get("interaction_type") == interaction_type.value
        ]
        return similar_decisions[-10:]  # 최근 10개

    async def _generate_decision(self, context: DecisionContext) -> Decision:
        """AI를 사용한 결정 생성"""
        
        # 프롬프트 구성
        prompt = self._build_decision_prompt(context)
        
        try:
            # Anthropic Claude를 사용한 결정 생성
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # 응답 파싱
            decision_data = self._parse_decision_response(response.content[0].text)
            
            # Decision 객체 생성
            decision = Decision(
                decision_id=f"dec_{int(time.time())}",
                timestamp=datetime.now(),
                decision_type=context.interaction.interaction_type.value,
                recommendation=decision_data["recommendation"],
                confidence_score=decision_data["confidence_score"],
                reasoning=decision_data["reasoning"],
                alternatives=decision_data["alternatives"],
                expected_outcome=decision_data["expected_outcome"],
                risk_assessment=decision_data["risk_assessment"],
                auto_execute=decision_data["auto_execute"],
                execution_plan=decision_data.get("execution_plan", None)
            )
            
            logger.info(f"결정 생성 완료: {decision.recommendation}")
            return decision
            
        except Exception as e:
            logger.error(f"결정 생성 중 오류: {e}")
            # 기본 결정 반환
            return self._create_default_decision(context)

    def _build_decision_prompt(self, context: DecisionContext) -> str:
        """결정 생성을 위한 프롬프트 구성"""
        interaction = context.interaction
        user_profile = context.user_profile
        
        prompt = f"""
당신은 사용자를 대신해 결정을 내리는 개인 AI 어시스턴트입니다.

**현재 상황:**
- 앱: {interaction.app_name}
- 인터액션: {interaction.interaction_type.value}
- 시간: {interaction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- 컨텍스트: {json.dumps(interaction.context, ensure_ascii=False, indent=2)}

**사용자 프로필:**
- 나이: {user_profile.age}세
- 위험 성향: {user_profile.risk_tolerance}
- 주요 가치관: {', '.join([f'{k}({v})' for k, v in user_profile.values.items() if v > 0.7])}
- 목표: {', '.join(user_profile.goals)}
- 선호도: {json.dumps(user_profile.preferences, ensure_ascii=False, indent=2)}

**현재 상태:**
- 예산 상태: {json.dumps(context.current_state['budget_status'], ensure_ascii=False)}
- 최근 활동: {json.dumps(context.current_state['recent_activity'], ensure_ascii=False)}
- 기분 상태: {context.current_state['current_mood']}

**외부 요인:**
- 날씨: {json.dumps(context.external_factors['weather'], ensure_ascii=False)}
- 트렌드: {json.dumps(context.external_factors['social_trends'], ensure_ascii=False)}

사용자의 입장에서 이 상황을 분석하고 최적의 결정을 내려주세요.

다음 JSON 형식으로 응답해주세요:
{{
    "recommendation": "구체적인 추천 행동",
    "confidence_score": 0.0-1.0,
    "reasoning": "결정 근거 설명",
    "alternatives": ["대안1", "대안2", "대안3"],
    "expected_outcome": {{"benefit": "예상 이익", "risk": "예상 위험"}},
    "risk_assessment": {{"level": "low/medium/high", "factors": ["위험요소1", "위험요소2"]}},
    "auto_execute": true/false,
    "execution_plan": {{"steps": ["단계1", "단계2"], "timeline": "실행 일정"}}
}}
"""
        return prompt

    def _parse_decision_response(self, response_text: str) -> Dict[str, Any]:
        """AI 응답 파싱"""
        try:
            # JSON 부분 추출
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            json_str = response_text[start_idx:end_idx]
            
            decision_data = json.loads(json_str)
            
            # 기본값 설정
            decision_data.setdefault("confidence_score", 0.5)
            decision_data.setdefault("alternatives", [])
            decision_data.setdefault("expected_outcome", {})
            decision_data.setdefault("risk_assessment", {"level": "medium", "factors": []})
            decision_data.setdefault("auto_execute", False)
            
            return decision_data
            
        except Exception as e:
            logger.error(f"응답 파싱 오류: {e}")
            return {
                "recommendation": "추가 정보가 필요합니다.",
                "confidence_score": 0.3,
                "reasoning": "응답 파싱 중 오류가 발생했습니다.",
                "alternatives": [],
                "expected_outcome": {},
                "risk_assessment": {"level": "high", "factors": ["파싱 오류"]},
                "auto_execute": False
            }

    def _create_default_decision(self, context: DecisionContext) -> Decision:
        """기본 결정 생성"""
        interaction = context.interaction
        
        return Decision(
            decision_id=f"default_{int(time.time())}",
            timestamp=datetime.now(),
            decision_type=interaction.interaction_type.value,
            recommendation="현재 상황에서는 신중하게 검토 후 결정하시기 바랍니다.",
            confidence_score=0.5,
            reasoning="충분한 정보가 없어 보수적인 접근을 권장합니다.",
            alternatives=["더 많은 정보 수집", "전문가 상담", "나중에 다시 검토"],
            expected_outcome={"benefit": "위험 최소화", "risk": "기회 손실 가능"},
            risk_assessment={"level": "medium", "factors": ["정보 부족"]},
            auto_execute=False,
            execution_plan=None
        )

    async def _execute_or_recommend_decision(self, decision: Decision, user_id: str):
        """결정 실행 또는 추천"""
        
        if decision.auto_execute and decision.confidence_score > 0.8:
            # 자동 실행
            logger.info(f"자동 실행: {decision.recommendation}")
            await self._execute_decision(decision, user_id)
        else:
            # 사용자에게 추천
            logger.info(f"추천 전송: {decision.recommendation}")
            await self._send_recommendation(decision, user_id)

    async def _execute_decision(self, decision: Decision, user_id: str):
        """결정 자동 실행"""
        try:
            if decision.execution_plan:
                steps = decision.execution_plan.get("steps", [])
                for step in steps:
                    logger.info(f"실행 단계: {step}")
                    # 실제 실행 로직 (API 호출, 앱 조작 등)
                    await asyncio.sleep(0.5)  # 시뮬레이션
                    
            logger.info(f"결정 실행 완료: {decision.recommendation}")
            
        except Exception as e:
            logger.error(f"결정 실행 중 오류: {e}")

    async def _send_recommendation(self, decision: Decision, user_id: str):
        """사용자에게 추천 전송"""
        
        notification = {
            "title": "🤖 Decision Agent 추천",
            "message": decision.recommendation,
            "confidence": f"신뢰도: {decision.confidence_score:.0%}",
            "reasoning": decision.reasoning,
            "alternatives": decision.alternatives,
            "timestamp": decision.timestamp.isoformat(),
            "decision_id": decision.decision_id
        }
        
        # 실제로는 푸시 알림, SMS, 이메일 등으로 전송
        logger.info(f"추천 알림: {notification}")
        
        # 시뮬레이션: 콘솔에 출력
        print("\n" + "="*60)
        print(f"🤖 DECISION AGENT 추천")
        print("="*60)
        print(f"📱 {decision.decision_type.upper()}")
        print(f"💡 추천: {decision.recommendation}")
        print(f"🎯 신뢰도: {decision.confidence_score:.0%}")
        print(f"📝 근거: {decision.reasoning}")
        if decision.alternatives:
            print(f"🔄 대안: {', '.join(decision.alternatives)}")
        print("="*60)

    def get_decision_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """결정 이력 조회"""
        recent_decisions = self.decision_history[-limit:]
        return [
            {
                "id": d.decision_id,
                "timestamp": d.timestamp.isoformat(),
                "type": d.decision_type,
                "recommendation": d.recommendation,
                "confidence": d.confidence_score,
                "executed": d.auto_execute
            }
            for d in recent_decisions
        ]

    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """사용자 선호도 업데이트"""
        if user_id in self.user_profiles:
            self.user_profiles[user_id].preferences.update(preferences)
            logger.info(f"사용자 {user_id} 설정 업데이트: {preferences}")

    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        logger.info("Decision Agent 모니터링 중지")

# 사용 예시
async def main():
    """Decision Agent 데모"""
    
    # API 키 설정 (실제로는 환경변수에서 로드)
    api_key = "your-anthropic-api-key"  # 실제 키로 교체 필요
    
    # Decision Agent 초기화
    agent = DecisionAgent(api_key)
    
    print("🤖 Decision Agent 시작")
    print("모바일 인터액션 모니터링을 시작합니다...")
    
    try:
        # 모니터링 시작
        await agent.start_monitoring("user_001")
        
    except KeyboardInterrupt:
        print("\n모니터링을 중지합니다.")
        agent.stop_monitoring()
    
    # 결정 이력 출력
    history = agent.get_decision_history("user_001")
    if history:
        print("\n📊 최근 결정 이력:")
        for decision in history:
            print(f"- {decision['timestamp']}: {decision['recommendation']}")

if __name__ == "__main__":
    asyncio.run(main())