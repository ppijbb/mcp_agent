import streamlit as st
import time
import json
import pandas as pd
import plotly.express as px
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 필수 imports 추가
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from srcs.advanced_agents.decision_agent_demo import (
        MockDecisionAgent, 
        create_sample_interactions, 
        InteractionType
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="Decision Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """메인 함수"""
    st.title("🤖 Decision Agent")
    st.markdown("### 모바일 인터액션 AI 결정 시스템")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 사용자 프로필 설정
        st.subheader("👤 사용자 프로필")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("나이", 18, 80, 30)
            budget = st.number_input("월 예산 (원)", min_value=0, value=2000000, step=100000)
        
        with col2:
            risk_tolerance = st.select_slider(
                "위험 허용도", 
                options=["보수적", "중간", "적극적"],
                value="중간"
            )
            priority = st.selectbox(
                "우선순위",
                ["절약", "편의성", "품질", "시간"]
            )
        
        # 결정 임계값 설정
        st.subheader("🎯 결정 임계값")
        intervention_threshold = st.slider(
            "개입 임계값", 
            0.0, 1.0, 0.7, 0.1,
            help="이 값 이상의 긴급도에서만 AI가 개입합니다"
        )
        
        auto_execute_threshold = st.slider(
            "자동 실행 임계값", 
            0.0, 1.0, 0.9, 0.1,
            help="이 값 이상의 신뢰도에서 자동으로 실행합니다"
        )
        
        # 알림 설정
        st.subheader("🔔 알림 설정")
        enable_notifications = st.checkbox("알림 활성화", value=True)
        notification_types = st.multiselect(
            "알림 유형",
            ["구매", "결제", "예약", "통화", "메시지"],
            default=["구매", "결제", "예약"]
        )
    
    # 메인 탭
    tab1, tab2, tab3, tab4 = st.tabs([
        "📱 실시간 모니터링", 
        "📊 결정 이력", 
        "🎯 시나리오 테스트",
        "⚙️ 시스템 분석"
    ])
    
    with tab1:
        display_realtime_monitoring()
    
    with tab2:
        display_decision_history()
    
    with tab3:
        display_scenario_testing()
    
    with tab4:
        display_system_analysis()

def display_realtime_monitoring():
    """실시간 모니터링 탭"""
    
    st.markdown("### 📱 실시간 모바일 인터액션 모니터링")
    
    # 컨트롤 버튼
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔍 모니터링 시작", type="primary"):
            st.session_state['monitoring'] = True
            st.rerun()
    
    with col2:
        if st.button("⏹️ 모니터링 중지"):
            st.session_state['monitoring'] = False
            st.rerun()
    
    with col3:
        if st.session_state.get('monitoring', False):
            st.success("🟢 모니터링 활성화")
        else:
            st.info("⚪ 모니터링 비활성화")
    
    # 모니터링 상태에 따른 표시
    if st.session_state.get('monitoring', False):
        # 모의 실시간 업데이트
        st.info("🔍 모바일 인터액션 감지 중...")
        
        # 가상의 인터액션 표시
        interactions = create_sample_interactions()[:2]
        
        for interaction in interactions:
            with st.expander(f"📱 {interaction.app_name} - {interaction.interaction_type.value}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.json(interaction.context, expanded=False)
                
                with col2:
                    st.markdown(f"""
                    **⏰ 시간:** {interaction.timestamp.strftime('%H:%M:%S')}  
                    **🔋 배터리:** {interaction.device_state['battery']}%  
                    **📶 네트워크:** {interaction.device_state['network']}  
                    **🚨 긴급도:** {interaction.urgency_score:.1f}/1.0
                    """)
                    
                    if st.button(f"🤖 AI 결정 요청", key=f"decide_{interaction.timestamp}"):
                        with st.spinner("AI가 결정을 생성 중..."):
                            # Mock 결정 생성
                            agent = MockDecisionAgent()
                            user_profile = asyncio.run(agent._get_user_profile("demo_user"))
                            context = asyncio.run(agent._build_decision_context(interaction, user_profile))
                            decision = asyncio.run(agent._generate_decision(context))
                            
                            # 결정 표시
                            st.success(f"💡 **추천:** {decision.recommendation}")
                            st.info(f"🎯 **신뢰도:** {decision.confidence_score:.0%}")
                            st.write(f"📝 **근거:** {decision.reasoning}")
                            
                            if decision.alternatives:
                                st.write(f"🔄 **대안:** {', '.join(decision.alternatives)}")
    else:
        st.info("모니터링을 시작하여 실시간 인터액션을 확인하세요.")

def display_decision_history():
    """결정 이력 탭"""
    
    st.markdown("### 📊 AI 결정 이력 분석")
    
    # 샘플 결정 데이터 생성
    if 'decision_history' not in st.session_state:
        st.session_state.decision_history = generate_sample_decision_history()
    
    history = st.session_state.decision_history
    
    if history:
        # 통계 요약
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📋 총 결정", len(history))
        
        with col2:
            auto_count = sum(1 for d in history if d['auto_execute'])
            st.metric("⚡ 자동 실행", f"{auto_count}/{len(history)}")
        
        with col3:
            avg_confidence = sum(d['confidence'] for d in history) / len(history)
            st.metric("🎯 평균 신뢰도", f"{avg_confidence:.0%}")
        
        with col4:
            purchase_count = sum(1 for d in history if d['type'] == 'purchase')
            st.metric("🛒 구매 관련", purchase_count)
        
        # 결정 유형별 분포
        st.markdown("#### 📈 결정 유형별 분포")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 결정 유형 파이 차트
            type_counts = {}
            for decision in history:
                type_counts[decision['type']] = type_counts.get(decision['type'], 0) + 1
            
            fig_pie = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                title="결정 유형별 분포"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # 신뢰도 분포 히스토그램
            confidences = [d['confidence'] for d in history]
            fig_hist = px.histogram(
                x=confidences,
                nbins=10,
                title="신뢰도 분포",
                labels={'x': '신뢰도', 'y': '빈도'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # 시간대별 결정 패턴
        st.markdown("#### ⏰ 시간대별 결정 패턴")
        
        df = pd.DataFrame(history)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_counts = df.groupby('hour').size().reset_index(name='count')
        
        fig_line = px.line(
            hourly_counts,
            x='hour',
            y='count',
            title="시간대별 결정 빈도",
            labels={'hour': '시간', 'count': '결정 수'}
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        # 상세 결정 이력
        st.markdown("#### 📋 상세 결정 이력")
        
        for i, decision in enumerate(reversed(history[-10:]), 1):
            with st.expander(f"{i}. {decision['type']} - {decision['timestamp']}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**💡 추천:** {decision['recommendation']}")
                    st.write(f"**📝 근거:** {decision.get('reasoning', '근거 없음')}")
                    if decision.get('alternatives'):
                        st.write(f"**🔄 대안:** {', '.join(decision['alternatives'])}")
                
                with col2:
                    st.metric("신뢰도", f"{decision['confidence']:.0%}")
                    st.write(f"**⚡ 자동실행:** {'예' if decision['auto_execute'] else '아니오'}")
    else:
        st.info("아직 결정 이력이 없습니다. 모니터링을 시작해보세요!")

def display_scenario_testing():
    """시나리오 테스트 탭"""
    
    st.markdown("### 🎯 Decision Agent 시나리오 테스트")
    
    # 시나리오 선택
    scenarios = {
        "온라인 쇼핑": {
            "description": "고가의 전자제품 구매 상황",
            "interaction_type": InteractionType.PURCHASE,
            "context": {
                "product": "맥북 프로 16인치",
                "price": 3500000,
                "discount": 0.05,
                "seller_rating": 4.9,
                "reviews_count": 1547
            }
        },
        "음식 배달": {
            "description": "늦은 밤 음식 주문 상황",
            "interaction_type": InteractionType.FOOD_ORDER,
            "context": {
                "restaurant": "24시 치킨집",
                "menu": "후라이드 치킨 + 맥주",
                "price": 35000,
                "delivery_time": 40,
                "rating": 3.8
            }
        },
        "호텔 예약": {
            "description": "해외 출장 호텔 예약",
            "context": {
                "hotel": "서울 비즈니스 호텔",
                "check_in": "2024-03-20",
                "check_out": "2024-03-22",
                "price": 450000,
                "rating": 4.6
            }
        },
        "중요한 전화": {
            "description": "상사로부터의 긴급 전화",
            "interaction_type": InteractionType.CALL,
            "context": {
                "contact": "이사장님",
                "call_type": "업무",
                "last_contact": "1개월 전",
                "importance": "critical"
            }
        }
    }
    
    selected_scenario = st.selectbox(
        "🎭 테스트 시나리오 선택",
        list(scenarios.keys()),
        format_func=lambda x: f"{x} - {scenarios[x]['description']}"
    )
    
    scenario = scenarios[selected_scenario]
    
    # 시나리오 상세 정보
    st.markdown(f"#### 📋 시나리오: {selected_scenario}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**📝 시나리오 상세:**")
        st.json(scenario['context'])
    
    with col2:
        st.markdown("**🎯 시나리오 정보:**")
        st.write(f"**📱 유형:** {scenario.get('interaction_type', 'N/A')}")
        st.write(f"**📄 설명:** {scenario['description']}")
        
        if st.button("🚀 시나리오 실행", type="primary"):
            with st.spinner("AI 결정 생성 중..."):
                # 시나리오 실행
                agent = MockDecisionAgent()
                
                # 가상 인터액션 생성
                from srcs.advanced_agents.decision_agent_demo import MobileInteraction
                
                interaction = MobileInteraction(
                    timestamp=datetime.now(),
                    app_name=selected_scenario,
                    interaction_type=scenario.get('interaction_type', InteractionType.PURCHASE),
                    context=scenario['context'],
                    device_state={'battery': 85, 'network': 'WiFi'},
                    location={'lat': 37.5665, 'lon': 126.9780},
                    urgency_score=0.8
                )
                
                # 결정 생성
                user_profile = asyncio.run(agent._get_user_profile("demo_user"))
                context = asyncio.run(agent._build_decision_context(interaction, user_profile))
                decision = asyncio.run(agent._generate_decision(context))
                
                # 결과 표시
                st.success(f"✅ 시나리오 실행 완료!")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.markdown("**🤖 AI 결정:**")
                    st.info(f"💡 **추천:** {decision.recommendation}")
                    st.write(f"📝 **근거:** {decision.reasoning}")
                    if decision.alternatives:
                        st.write(f"🔄 **대안:** {', '.join(decision.alternatives)}")
                
                with col4:
                    st.markdown("**📊 결정 메트릭:**")
                    st.metric("신뢰도", f"{decision.confidence_score:.0%}")
                    st.metric("자동 실행", "예" if decision.auto_execute else "아니오")
                    st.metric("긴급도", f"{interaction.urgency_score:.1f}/1.0")

def display_system_analysis():
    """시스템 분석 탭"""
    
    st.markdown("### ⚙️ Decision Agent 시스템 분석")
    
    # 시스템 상태
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🟢 시스템 상태", "정상")
    
    with col2:
        st.metric("📊 처리 속도", "1.2초")
    
    with col3:
        st.metric("🧠 AI 모델", "Claude-3.5")
    
    with col4:
        st.metric("📈 정확도", "87.3%")
    
    # 성능 지표
    st.markdown("#### 📈 시스템 성능 지표")
    
    # 가상 성능 데이터
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    performance_data = {
        'date': dates,
        'accuracy': [0.85 + (i % 7) * 0.02 for i in range(len(dates))],
        'response_time': [1.0 + (i % 5) * 0.1 for i in range(len(dates))],
        'decisions_count': [20 + (i % 10) * 5 for i in range(len(dates))]
    }
    
    df = pd.DataFrame(performance_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 정확도 추이
        fig_accuracy = px.line(
            df, x='date', y='accuracy',
            title='AI 결정 정확도 추이',
            labels={'accuracy': '정확도', 'date': '날짜'}
        )
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    with col2:
        # 응답 시간 추이
        fig_response = px.line(
            df, x='date', y='response_time',
            title='평균 응답 시간 추이',
            labels={'response_time': '응답시간(초)', 'date': '날짜'}
        )
        st.plotly_chart(fig_response, use_container_width=True)
    
    # 일별 결정 수
    fig_decisions = px.bar(
        df, x='date', y='decisions_count',
        title='일별 AI 결정 수',
        labels={'decisions_count': '결정 수', 'date': '날짜'}
    )
    st.plotly_chart(fig_decisions, use_container_width=True)
    
    # 시스템 설정
    st.markdown("#### ⚙️ 시스템 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🤖 AI 모델 설정:**")
        st.write("- 모델: Claude-3.5-Sonnet")
        st.write("- 최대 토큰: 4096")
        st.write("- 온도: 0.3")
        st.write("- 최대 재시도: 3")
    
    with col2:
        st.markdown("**📊 데이터 설정:**")
        st.write("- 저장 기간: 30일")
        st.write("- 백업 주기: 매일")
        st.write("- 데이터 암호화: AES-256")
        st.write("- 익명화: 활성화")

def generate_sample_decision_history():
    """샘플 결정 이력 생성"""
    
    import random
    
    decisions = []
    decision_types = ['purchase', 'food_order', 'booking', 'call', 'message']
    
    for i in range(50):
        decision = {
            'timestamp': (datetime.now() - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d %H:%M:%S'),
            'type': random.choice(decision_types),
            'recommendation': f"샘플 추천 {i+1}",
            'reasoning': f"샘플 근거 {i+1}",
            'confidence': random.uniform(0.6, 0.95),
            'auto_execute': random.choice([True, False]),
            'alternatives': [f"대안 {j+1}" for j in range(random.randint(0, 3))]
        }
        decisions.append(decision)
    
    return decisions

if __name__ == "__main__":
    main() 