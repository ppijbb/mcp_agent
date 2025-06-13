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

# 중앙 설정 시스템 import
from configs.settings import get_reports_path

try:
    from srcs.advanced_agents.decision_agent import (
        DecisionAgent, 
        InteractionType,
        MobileInteraction
    )
    DECISION_AGENT_AVAILABLE = True
except ImportError as e:
    st.error(f"Decision Agent를 사용하려면 필요한 의존성을 설치해야 합니다: {e}")
    st.error("시스템 관리자에게 문의하여 Decision Agent 모듈을 설정하세요.")
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="Decision Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_risk_tolerance_options():
    """위험 허용도 옵션 동적 로딩"""
    # TODO: 실제 사용자 프로필 시스템에서 로드
    return ["보수적", "중간", "적극적"]

def load_priority_options():
    """우선순위 옵션 동적 로딩"""
    # TODO: 실제 시스템 설정에서 로드
    return ["절약", "편의성", "품질", "시간"]

def load_notification_types():
    """알림 유형 동적 로딩"""
    # TODO: 실제 시스템에서 지원하는 알림 유형 로드
    return ["구매", "결제", "예약", "통화", "메시지"]

def load_user_profile_defaults():
    """사용자 프로필 기본값 동적 로딩"""
    # TODO: 실제 사용자 데이터베이스에서 로드
    return {
        "age_min": 18,
        "age_max": 80,
        "budget_min": 0,
        "budget_step": 100000
    }

def load_decision_scenarios():
    """결정 시나리오 동적 로딩"""
    # TODO: 실제 시나리오 데이터베이스에서 로드
    return {}

def get_real_decision_history():
    """실제 결정 이력 조회"""
    # TODO: 실제 데이터베이스에서 결정 이력 조회
    raise NotImplementedError("실제 결정 이력 조회 기능이 구현되지 않았습니다.")

def get_real_system_metrics():
    """실제 시스템 메트릭 조회"""
    # TODO: 실제 시스템 모니터링에서 메트릭 조회
    raise NotImplementedError("실제 시스템 메트릭 조회 기능이 구현되지 않았습니다.")

def get_real_mobile_interactions():
    """실제 모바일 인터액션 조회"""
    # TODO: 실제 모바일 모니터링 시스템에서 인터액션 조회
    raise NotImplementedError("실제 모바일 인터액션 조회 기능이 구현되지 않았습니다.")

def main():
    """메인 함수"""
    st.title("🤖 Decision Agent")
    st.markdown("### 모바일 인터액션 AI 결정 시스템")
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    # 파일 저장 옵션 추가
    st.markdown("### ⚙️ 출력 옵션")
    save_to_file = st.checkbox(
        "결정 결과를 파일로 저장", 
        value=False,
        help=f"체크하면 {get_reports_path('decision_agent')} 디렉토리에 결정 결과를 파일로 저장합니다"
    )
    
    if save_to_file:
        st.info(f"📁 결정 결과가 {get_reports_path('decision_agent')} 디렉토리에 저장됩니다.")
    
    st.markdown("---")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 사용자 프로필 설정
        st.subheader("👤 사용자 프로필")
        
        profile_defaults = load_user_profile_defaults()
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider(
                "나이", 
                profile_defaults["age_min"], 
                profile_defaults["age_max"], 
                value=None,
                help="사용자의 나이를 입력하세요"
            )
            budget = st.number_input(
                "월 예산 (원)", 
                min_value=profile_defaults["budget_min"], 
                value=None, 
                step=profile_defaults["budget_step"],
                help="월 예산을 입력하세요"
            )
        
        with col2:
            risk_tolerance_options = load_risk_tolerance_options()
            risk_tolerance = st.select_slider(
                "위험 허용도", 
                options=risk_tolerance_options,
                value=None,
                help="투자 위험 허용도를 선택하세요"
            )
            
            priority_options = load_priority_options()
            priority = st.selectbox(
                "우선순위",
                priority_options,
                index=None,
                placeholder="우선순위를 선택하세요"
            )
        
        # 결정 임계값 설정
        st.subheader("🎯 결정 임계값")
        intervention_threshold = st.slider(
            "개입 임계값", 
            0.0, 1.0, value=None, step=0.1,
            help="이 값 이상의 긴급도에서만 AI가 개입합니다"
        )
        
        auto_execute_threshold = st.slider(
            "자동 실행 임계값", 
            0.0, 1.0, value=None, step=0.1,
            help="이 값 이상의 신뢰도에서 자동으로 실행합니다"
        )
        
        # 알림 설정
        st.subheader("🔔 알림 설정")
        enable_notifications = st.checkbox("알림 활성화", value=False)
        
        notification_types_options = load_notification_types()
        notification_types = st.multiselect(
            "알림 유형",
            notification_types_options,
            help="받고 싶은 알림 유형을 선택하세요"
        )
    
    # 메인 탭
    tab1, tab2, tab3, tab4 = st.tabs([
        "📱 실시간 모니터링", 
        "📊 결정 이력", 
        "🎯 시나리오 테스트",
        "⚙️ 시스템 분석"
    ])
    
    with tab1:
        display_realtime_monitoring(save_to_file)
    
    with tab2:
        display_decision_history(save_to_file)
    
    with tab3:
        display_scenario_testing(save_to_file)
    
    with tab4:
        display_system_analysis()

def display_realtime_monitoring(save_to_file=False):
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
        st.info("🔍 모바일 인터액션 감지 중...")
        
        try:
            # 실제 인터액션 조회
            interactions = get_real_mobile_interactions()
            
            if not interactions:
                st.info("현재 감지된 모바일 인터액션이 없습니다.")
                return
        
            for interaction in interactions[:2]:  # 최근 2개만 표시
                with st.expander(f"📱 {interaction.app_name} - {interaction.interaction_type.value}", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.json(interaction.context, expanded=False)
                    
                    with col2:
                        st.markdown(f"""
                        **⏰ 시간:** {interaction.timestamp.strftime('%H:%M:%S')}  
                        **🔋 배터리:** {interaction.device_state.get('battery', 'N/A')}%  
                        **📶 네트워크:** {interaction.device_state.get('network', 'N/A')}  
                        **🚨 긴급도:** {interaction.urgency_score:.1f}/1.0
                        """)
                        
                        if st.button(f"🤖 AI 결정 요청", key=f"decide_{interaction.timestamp}"):
                            with st.spinner("AI가 결정을 생성 중..."):
                                # 실제 결정 에이전트 호출
                                agent = DecisionAgent()
                                decision = agent.make_decision(interaction)
                                
                                if not decision:
                                    st.error("AI 결정 생성에 실패했습니다.")
                                    return
                            
                                # 결정 표시
                                st.success(f"💡 **추천:** {decision.recommendation}")
                                st.info(f"🎯 **신뢰도:** {decision.confidence_score:.0%}")
                                st.write(f"📝 **근거:** {decision.reasoning}")
                                
                                if decision.alternatives:
                                    st.write(f"🔄 **대안:** {', '.join(decision.alternatives)}")
                                
                                # 텍스트 출력 생성
                                decision_text = format_decision_result(interaction, decision)
                                
                                # 텍스트 결과 표시
                                st.markdown("#### 📄 결정 결과 텍스트")
                                st.text_area(
                                    "결정 내용",
                                    value=decision_text,
                                    height=150,
                                    disabled=True,
                                    key=f"decision_text_{interaction.timestamp}"
                                )
                                
                                # 파일 저장 처리
                                if save_to_file:
                                    file_saved, output_path = save_decision_to_file(interaction, decision, decision_text)
                                    if file_saved:
                                        st.success(f"💾 결정이 파일로 저장되었습니다: {output_path}")
                                    else:
                                        st.error("파일 저장 중 오류가 발생했습니다.")
        
        except NotImplementedError as e:
            st.error(f"실시간 모니터링 기능이 구현되지 않았습니다: {e}")
        except Exception as e:
            st.error(f"모니터링 중 오류가 발생했습니다: {e}")
    else:
        st.info("모니터링을 시작하여 실시간 인터액션을 확인하세요.")

def display_decision_history(save_to_file=False):
    """결정 이력 탭"""
    
    st.markdown("### 📊 AI 결정 이력 분석")
    
    try:
        # 실제 결정 이력 조회
        history = get_real_decision_history()
        
        if not history:
            st.info("아직 결정 이력이 없습니다. 모니터링을 시작해보세요!")
            return
        
        # 통계 요약
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📋 총 결정", len(history))
        
        with col2:
            auto_count = sum(1 for d in history if d.get('auto_execute', False))
            st.metric("⚡ 자동 실행", f"{auto_count}/{len(history)}")
        
        with col3:
            confidences = [d.get('confidence', 0) for d in history if d.get('confidence')]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            st.metric("🎯 평균 신뢰도", f"{avg_confidence:.0%}")
        
        with col4:
            purchase_count = sum(1 for d in history if d.get('type') == 'purchase')
            st.metric("🛒 구매 관련", purchase_count)
        
        # 결정 유형별 분포 차트
        display_decision_analytics(history)
        
        # 상세 결정 이력
        display_detailed_history(history)
    
    except NotImplementedError as e:
        st.error(f"결정 이력 조회 기능이 구현되지 않았습니다: {e}")
    except Exception as e:
        st.error(f"결정 이력 조회 중 오류가 발생했습니다: {e}")

def display_scenario_testing(save_to_file=False):
    """시나리오 테스트 탭"""
    
    st.markdown("### 🎯 Decision Agent 시나리오 테스트")
    
    try:
        # 실제 시나리오 로딩
        scenarios = load_decision_scenarios()
        
        if not scenarios:
            st.warning("현재 사용 가능한 테스트 시나리오가 없습니다.")
            st.info("시스템 관리자에게 문의하여 시나리오를 설정하세요.")
            return
        
        selected_scenario = st.selectbox(
            "🎭 테스트 시나리오 선택",
            list(scenarios.keys()),
            index=None,
            placeholder="시나리오를 선택하세요",
            format_func=lambda x: f"{x} - {scenarios[x].get('description', '')}"
        )
        
        if not selected_scenario:
            st.info("테스트할 시나리오를 선택하세요.")
            return
        
        scenario = scenarios[selected_scenario]
        
        # 시나리오 실행
        execute_scenario_test(scenario, selected_scenario, save_to_file)
    
    except NotImplementedError as e:
        st.error(f"시나리오 테스트 기능이 구현되지 않았습니다: {e}")
    except Exception as e:
        st.error(f"시나리오 테스트 중 오류가 발생했습니다: {e}")

def display_system_analysis():
    """시스템 분석 탭"""
    
    st.markdown("### ⚙️ Decision Agent 시스템 분석")
    
    try:
        # 실제 시스템 메트릭 조회
        metrics = get_real_system_metrics()
        
        if not metrics:
            st.error("시스템 메트릭을 조회할 수 없습니다.")
            return
        
        # 시스템 상태 표시
        display_system_status(metrics)
        
        # 성능 지표 표시
        display_performance_metrics(metrics)
        
        # 시스템 설정 표시
        display_system_configuration(metrics)
    
    except NotImplementedError as e:
        st.error(f"시스템 분석 기능이 구현되지 않았습니다: {e}")
    except Exception as e:
        st.error(f"시스템 분석 중 오류가 발생했습니다: {e}")

def display_decision_analytics(history):
    """결정 분석 차트 표시"""
    if not history:
        return
    
    st.markdown("#### 📈 결정 유형별 분포")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 결정 유형 파이 차트
        type_counts = {}
        for decision in history:
            decision_type = decision.get('type', 'unknown')
            type_counts[decision_type] = type_counts.get(decision_type, 0) + 1
            
        if type_counts:
            fig_pie = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                title="결정 유형별 분포"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # 신뢰도 분포 히스토그램
        confidences = [d.get('confidence', 0) for d in history if d.get('confidence') is not None]
        if confidences:
            fig_hist = px.histogram(
                x=confidences,
                nbins=10,
                title="신뢰도 분포",
                labels={'x': '신뢰도', 'y': '빈도'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

def display_detailed_history(history):
    """상세 결정 이력 표시"""
    st.markdown("#### 📋 상세 결정 이력")
    
    for i, decision in enumerate(reversed(history[-10:]), 1):
        timestamp = decision.get('timestamp', 'N/A')
        decision_type = decision.get('type', 'unknown')
    
        with st.expander(f"{i}. {decision_type} - {timestamp}", expanded=False):
            col1, col2 = st.columns([3, 1])
                
            with col1:
                st.write(f"**💡 추천:** {decision.get('recommendation', 'N/A')}")
                st.write(f"**📝 근거:** {decision.get('reasoning', '근거 없음')}")
                alternatives = decision.get('alternatives', [])
                if alternatives:
                    st.write(f"**🔄 대안:** {', '.join(alternatives)}")
                    
            with col2:
                confidence = decision.get('confidence', 0)
                st.metric("신뢰도", f"{confidence:.0%}")
                auto_execute = decision.get('auto_execute', False)
                st.write(f"**⚡ 자동실행:** {'예' if auto_execute else '아니오'}")

def execute_scenario_test(scenario, scenario_name, save_to_file):
    """시나리오 테스트 실행"""
    st.markdown(f"#### 📋 시나리오: {scenario_name}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**📝 시나리오 상세:**")
        st.json(scenario.get('context', {}))
    
    with col2:
        st.markdown("**🎯 시나리오 정보:**")
        st.write(f"**📱 유형:** {scenario.get('interaction_type', 'N/A')}")
        st.write(f"**📄 설명:** {scenario.get('description', 'N/A')}")
        
        if st.button("🚀 시나리오 실행", type="primary"):
            with st.spinner("AI 결정 생성 중..."):
                # 실제 시나리오 실행
                agent = DecisionAgent()
                result = agent.test_scenario(scenario)
                
                if not result:
                    st.error("시나리오 테스트에 실패했습니다.")
                    return
                
                # 결과 표시
                display_scenario_results(result)

def display_scenario_results(result):
    """시나리오 테스트 결과 표시"""
    st.success("✅ 시나리오 실행 완료!")
                    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**🤖 AI 결정:**")
        st.info(f"💡 **추천:** {result.get('recommendation', 'N/A')}")
        st.write(f"📝 **근거:** {result.get('reasoning', 'N/A')}")
        alternatives = result.get('alternatives', [])
        if alternatives:
            st.write(f"🔄 **대안:** {', '.join(alternatives)}")
                    
    with col4:
        st.markdown("**📊 결정 메트릭:**")
        confidence = result.get('confidence_score', 0)
        st.metric("신뢰도", f"{confidence:.0%}")
        auto_execute = result.get('auto_execute', False)
        st.metric("자동 실행", "예" if auto_execute else "아니오")
        urgency = result.get('urgency_score', 0)
        st.metric("긴급도", f"{urgency:.1f}/1.0")

def display_system_status(metrics):
    """시스템 상태 표시"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = metrics.get('system_status', 'unknown')
        st.metric("🟢 시스템 상태", status)
    
    with col2:
        processing_speed = metrics.get('processing_speed', 'N/A')
        st.metric("📊 처리 속도", processing_speed)
    
    with col3:
        ai_model = metrics.get('ai_model', 'N/A')
        st.metric("🧠 AI 모델", ai_model)
    
    with col4:
        accuracy = metrics.get('accuracy', 'N/A')
        st.metric("📈 정확도", accuracy)
    
def display_performance_metrics(metrics):
    """성능 지표 표시"""
    st.markdown("#### 📈 시스템 성능 지표")
    
    performance_data = metrics.get('performance_data', {})
    if not performance_data:
        st.warning("성능 데이터를 사용할 수 없습니다.")
        return
    
    # 성능 차트 표시 로직
    # TODO: 실제 성능 데이터를 기반으로 차트 생성

def display_system_configuration(metrics):
    """시스템 설정 표시"""
    st.markdown("#### ⚙️ 시스템 설정")
    
    config = metrics.get('configuration', {})
    if not config:
        st.warning("시스템 설정 정보를 사용할 수 없습니다.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🤖 AI 모델 설정:**")
        ai_config = config.get('ai_model', {})
        for key, value in ai_config.items():
            st.write(f"- {key}: {value}")
    
    with col2:
        st.markdown("**📊 데이터 설정:**")
        data_config = config.get('data', {})
        for key, value in data_config.items():
            st.write(f"- {key}: {value}")

def format_decision_result(interaction, decision):
    """Decision Agent 결과 포맷팅"""
    
    text_output = f"""
🤖 AI 결정 결과

📱 인터액션 정보:
- 앱: {interaction.app_name}
- 유형: {interaction.interaction_type.value}
- 시간: {interaction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- 긴급도: {interaction.urgency_score:.2f}/1.0

🎯 AI 추천 결정:
- 추천 액션: {decision.recommendation}
- 신뢰도: {decision.confidence_score:.0%}
- 자동 실행: {'예' if decision.auto_execute else '아니오'}

📝 결정 근거:
{decision.reasoning}

🔄 대안 옵션:"""
    
    if decision.alternatives:
        for i, alt in enumerate(decision.alternatives, 1):
            text_output += f"\n{i}. {alt}"
    else:
        text_output += "\n- 추가 대안 없음"
    
    # 컨텍스트 정보 추가
    text_output += f"""

📊 디바이스 상태:
- 배터리: {interaction.device_state.get('battery', 'N/A')}%
- 네트워크: {interaction.device_state.get('network', 'N/A')}
- 위치: {interaction.device_state.get('location', 'N/A')}

⚡ 실행 결과:
- 결정 생성 시간: {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- 결정 ID: {decision.decision_id}
"""
    
    return text_output.strip()

def save_decision_to_file(interaction, decision, decision_text):
    """Decision Agent 결정을 파일로 저장"""
    
    try:
        import os
        from datetime import datetime
        
        output_dir = get_reports_path('decision_agent')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"decision_result_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Decision Agent 결정 보고서\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(decision_text)
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("*본 보고서는 Decision Agent에 의해 자동 생성되었습니다.*\n")
        
        return True, filepath
        
    except Exception as e:
        print(f"파일 저장 중 오류: {e}")
        return False, None

if __name__ == "__main__":
    main() 