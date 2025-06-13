"""
Decision Agent 데모 및 테스트 스크립트

실제 Anthropic API 키 없이도 Decision Agent의 핵심 기능을 
시뮬레이션하여 테스트할 수 있습니다.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any
import sys
import os

# 상위 디렉토리를 패스에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_agents.decision_agent import (
    DecisionAgent, InteractionType, MobileInteraction, 
    Decision, UserProfile
)

class MockDecisionAgent(DecisionAgent):
    """⚠️ DEPRECATED: Use decision_agent_mcp_agent.py for real MCP implementation
    
    🚨 CRITICAL ISSUE: This class contains MOCK DATA and should not be used in production.
    
    Based on real-world MCP implementation patterns from:
    https://medium.com/@govindarajpriyanthan/from-theory-to-practice-building-a-multi-agent-research-system-with-mcp-part-2-811b0163e87c
    
    USE srcs/advanced_agents/decision_agent_mcp_agent.py INSTEAD for real ReAct decision making.
    
    ❌ PROBLEMS WITH THIS CLASS:
    - Line 47-77: Pre-defined mock decisions (fake data)
    - Line 113: Mock confidence scores
    - Line 235: Simulated sleep delays
    - No real market research
    - No actual risk assessment
    - No ReAct reasoning pattern
    """
    
    def __init__(self):
        print("⚠️ WARNING: MockDecisionAgent is DEPRECATED!")
        print("🚨 Use srcs/advanced_agents/decision_agent_mcp_agent.py for real MCP implementation")
        print("📖 Based on: https://medium.com/@govindarajpriyanthan/from-theory-to-practice-building-a-multi-agent-research-system-with-mcp-part-2-811b0163e87c")
        
        # API 키 없이 초기화
        self.user_profiles: Dict[str, UserProfile] = {}
        self.interaction_buffer: list = []
        self.decision_history: list = []
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
        
        print("🤖 Mock Decision Agent 초기화 완료")

    async def _generate_decision(self, context) -> Decision:
        """Mock 결정 생성 (실제 AI 호출 없이)"""
        
        interaction = context.interaction
        user_profile = context.user_profile
        
        # 인터액션 타입별 미리 정의된 결정들
        mock_decisions = {
            InteractionType.PURCHASE: {
                "recommendation": f"{interaction.context.get('product', '상품')} 구매를 권장하지 않습니다. 더 나은 대안을 찾아보세요.",
                "confidence_score": 0.8,
                "reasoning": f"가격 {interaction.context.get('price', 0):,}원이 예산 대비 높고, 리뷰 분석 결과 더 좋은 대안이 있습니다.",
                "alternatives": ["다른 브랜드 제품 검토", "할인 기간 대기", "중고 제품 고려"],
                "auto_execute": False
            },
            InteractionType.FOOD_ORDER: {
                "recommendation": f"{interaction.context.get('restaurant', '음식점')}에서 주문하세요. 좋은 선택입니다!",
                "confidence_score": 0.9,
                "reasoning": f"평점 {interaction.context.get('rating', 0)}점, 배달시간 {interaction.context.get('delivery_time', 0)}분으로 적당하며 건강한 식단에 도움됩니다.",
                "alternatives": ["다른 메뉴 선택", "직접 요리", "근처 식당 방문"],
                "auto_execute": True
            },
            InteractionType.BOOKING: {
                "recommendation": f"{interaction.context.get('hotel', '호텔')} 예약을 진행하세요.",
                "confidence_score": 0.7,
                "reasoning": f"가격 대비 좋은 위치와 시설을 제공하며, 여행 일정에 적합합니다.",
                "alternatives": ["다른 숙소 비교", "예약 시기 조정", "에어비앤비 고려"],
                "auto_execute": False
            },
            InteractionType.CALL: {
                "recommendation": "지금 전화하는 것이 좋겠습니다.",
                "confidence_score": 0.85,
                "reasoning": "상대방이 통화 가능한 시간대이고, 중요한 용건으로 보입니다.",
                "alternatives": ["문자 메시지 먼저 전송", "나중에 통화", "이메일로 연락"],
                "auto_execute": False
            },
            InteractionType.MESSAGE: {
                "recommendation": "메시지를 보내세요. 적절한 시점입니다.",
                "confidence_score": 0.75,
                "reasoning": "상대방과의 관계와 대화 맥락을 고려할 때 지금이 좋은 타이밍입니다.",
                "alternatives": ["나중에 전송", "전화로 대체", "더 신중하게 작성"],
                "auto_execute": True
            }
        }
        
        # 기본 결정 데이터 가져오기
        decision_data = mock_decisions.get(interaction.interaction_type, {
            "recommendation": "신중하게 검토해보세요.",
            "confidence_score": 0.6,
            "reasoning": "충분한 정보를 수집한 후 결정하는 것이 좋겠습니다.",
            "alternatives": ["더 많은 정보 수집", "전문가 상담", "시간을 두고 고민"],
            "auto_execute": False
        })
        
        # 사용자 프로필 반영
        if user_profile.risk_tolerance == "conservative":
            decision_data["confidence_score"] *= 0.8
            decision_data["auto_execute"] = False
        elif user_profile.risk_tolerance == "aggressive":
            decision_data["confidence_score"] *= 1.2
            decision_data["auto_execute"] = decision_data["confidence_score"] > 0.7

        # Decision 객체 생성
        decision = Decision(
            decision_id=f"mock_{int(time.time())}",
            timestamp=datetime.now(),
            decision_type=interaction.interaction_type.value,
            recommendation=decision_data["recommendation"],
            confidence_score=min(decision_data["confidence_score"], 1.0),
            reasoning=decision_data["reasoning"],
            alternatives=decision_data["alternatives"],
            expected_outcome={
                "benefit": "시간 절약, 최적 결정",
                "risk": "예상치 못한 결과 가능성"
            },
            risk_assessment={
                "level": "low" if decision_data["confidence_score"] > 0.8 else "medium",
                "factors": ["정보 부족", "시장 변동성"]
            },
            auto_execute=decision_data["auto_execute"],
            execution_plan={
                "steps": ["상황 재검토", "실행", "결과 모니터링"],
                "timeline": "즉시"
            } if decision_data["auto_execute"] else None
        )
        
        return decision

def create_sample_interactions():
    """샘플 인터액션 생성"""
    interactions = []
    
    # 구매 인터액션
    interactions.append(MobileInteraction(
        timestamp=datetime.now(),
        interaction_type=InteractionType.PURCHASE,
        app_name="쿠팡",
        context={
            "product": "애플 에어팟 프로",
            "price": 329000,
            "discount": 0.10,
            "seller_rating": 4.8,
            "reviews_count": 2847
        },
        user_location=(37.5665, 126.9780),
        device_state={"battery": 75, "network": "WiFi"},
        urgency_score=0.8,
        metadata={}
    ))
    
    # 음식 주문 인터액션
    interactions.append(MobileInteraction(
        timestamp=datetime.now(),
        interaction_type=InteractionType.FOOD_ORDER,
        app_name="배달의민족",
        context={
            "restaurant": "건강한 샐러드",
            "menu": "아보카도 치킨 샐러드",
            "price": 12500,
            "delivery_time": 20,
            "rating": 4.6
        },
        user_location=(37.5665, 126.9780),
        device_state={"battery": 75, "network": "WiFi"},
        urgency_score=0.6,
        metadata={}
    ))
    
    # 호텔 예약 인터액션
    interactions.append(MobileInteraction(
        timestamp=datetime.now(),
        interaction_type=InteractionType.BOOKING,
        app_name="야놀자",
        context={
            "hotel": "제주 오션뷰 리조트",
            "check_in": "2024-03-15",
            "check_out": "2024-03-17",
            "price": 280000,
            "rating": 4.3
        },
        user_location=(37.5665, 126.9780),
        device_state={"battery": 60, "network": "4G"},
        urgency_score=0.7,
        metadata={}
    ))
    
    # 전화 인터액션
    interactions.append(MobileInteraction(
        timestamp=datetime.now(),
        interaction_type=InteractionType.CALL,
        app_name="전화",
        context={
            "contact": "김부장",
            "call_type": "업무",
            "last_contact": "2일 전",
            "importance": "high"
        },
        user_location=(37.5665, 126.9780),
        device_state={"battery": 85, "network": "WiFi"},
        urgency_score=0.9,
        metadata={}
    ))
    
    return interactions

async def run_demo():
    """Decision Agent 데모 실행"""
    
    print("🚀 Decision Agent 데모 시작")
    print("=" * 60)
    
    # Mock Agent 생성
    agent = MockDecisionAgent()
    
    # 샘플 인터액션들 생성
    interactions = create_sample_interactions()
    
    print(f"📱 {len(interactions)}개의 모바일 인터액션을 처리합니다...\n")
    
    for i, interaction in enumerate(interactions, 1):
        print(f"[{i}/{len(interactions)}] 처리 중: {interaction.interaction_type.value}")
        print(f"앱: {interaction.app_name}")
        print(f"컨텍스트: {json.dumps(interaction.context, ensure_ascii=False, indent=2)}")
        
        # 인터액션 처리
        await agent._process_interaction(interaction, "demo_user")
        
        print("\n" + "-" * 50 + "\n")
        await asyncio.sleep(1)  # 데모 효과를 위한 지연
    
    # 결정 이력 출력
    print("📊 생성된 결정 이력:")
    history = agent.get_decision_history("demo_user")
    
    for decision in history:
        print(f"- {decision['type']}: {decision['recommendation']}")
        print(f"  신뢰도: {decision['confidence']:.0%}")
        print()

def interactive_demo():
    """인터랙티브 데모"""
    
    print("🎮 인터랙티브 Decision Agent 데모")
    print("=" * 60)
    
    agent = MockDecisionAgent()
    
    scenarios = {
        "1": {
            "name": "온라인 쇼핑",
            "interaction": MobileInteraction(
                timestamp=datetime.now(),
                interaction_type=InteractionType.PURCHASE,
                app_name="네이버 쇼핑",
                context={
                    "product": "무선 청소기",
                    "price": 450000,
                    "discount": 0.20,
                    "seller_rating": 4.2,
                    "reviews_count": 892
                },
                user_location=(37.5665, 126.9780),
                device_state={"battery": 60, "network": "WiFi"},
                urgency_score=0.7,
                metadata={}
            )
        },
        "2": {
            "name": "음식 배달 주문",
            "interaction": MobileInteraction(
                timestamp=datetime.now(),
                interaction_type=InteractionType.FOOD_ORDER,
                app_name="요기요",
                context={
                    "restaurant": "맛있는 피자집",
                    "menu": "페퍼로니 피자",
                    "price": 25000,
                    "delivery_time": 35,
                    "rating": 4.1
                },
                user_location=(37.5665, 126.9780),
                device_state={"battery": 40, "network": "4G"},
                urgency_score=0.8,
                metadata={}
            )
        },
        "3": {
            "name": "중요한 전화",
            "interaction": MobileInteraction(
                timestamp=datetime.now(),
                interaction_type=InteractionType.CALL,
                app_name="전화",
                context={
                    "contact": "엄마",
                    "call_type": "개인",
                    "last_contact": "1주일 전",
                    "importance": "medium"
                },
                user_location=(37.5665, 126.9780),
                device_state={"battery": 90, "network": "WiFi"},
                urgency_score=0.6,
                metadata={}
            )
        }
    }
    
    while True:
        print("\n시나리오를 선택하세요:")
        for key, scenario in scenarios.items():
            print(f"{key}. {scenario['name']}")
        print("0. 종료")
        
        choice = input("\n선택: ").strip()
        
        if choice == "0":
            print("데모를 종료합니다.")
            break
        elif choice in scenarios:
            scenario = scenarios[choice]
            print(f"\n🎯 시나리오: {scenario['name']}")
            
            asyncio.run(agent._process_interaction(
                scenario['interaction'], 
                "interactive_user"
            ))
        else:
            print("잘못된 선택입니다.")

def main():
    """메인 함수"""
    print("Decision Agent 데모 모드를 선택하세요:")
    print("1. 자동 데모 (모든 시나리오 순차 실행)")
    print("2. 인터랙티브 데모 (시나리오 선택)")
    
    choice = input("선택 (1 또는 2): ").strip()
    
    if choice == "1":
        asyncio.run(run_demo())
    elif choice == "2":
        interactive_demo()
    else:
        print("잘못된 선택입니다. 자동 데모를 실행합니다.")
        asyncio.run(run_demo())

if __name__ == "__main__":
    main() 