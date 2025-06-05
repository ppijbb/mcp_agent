"""
SEO Doctor - 모바일 친화적 Streamlit 앱

🏥 사이트 응급처치 + 🕵️ 경쟁사 스파이 = MAU 10만+ 목표
"""

import streamlit as st
import asyncio
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Any

# SEO Doctor Agent 임포트
from .seo_doctor_agent import (
    get_seo_doctor, 
    run_seo_emergency_service,
    SEOEmergencyLevel,
    CompetitorThreatLevel
)

# 모바일 최적화 CSS
MOBILE_CSS = """
<style>
/* 모바일 우선 설계 */
.main > div {
    padding-top: 1rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

/* 큰 버튼 스타일 */
.stButton > button {
    height: 3rem;
    width: 100%;
    font-size: 1.2rem;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    margin: 0.5rem 0;
}

/* 응급 상황별 색상 */
.critical-btn {
    background: linear-gradient(45deg, #ff4757, #ff3838) !important;
    color: white !important;
}

.emergency-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.competitor-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

.prescription-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
}

/* 모바일 텍스트 크기 */
.metric-big {
    font-size: 2rem !important;
    font-weight: bold !important;
}

/* 터치 친화적 스페이싱 */
.touch-friendly {
    min-height: 44px;
    padding: 12px;
    margin: 8px 0;
}

/* 스와이프 힌트 */
.swipe-hint {
    position: relative;
    overflow-x: auto;
    white-space: nowrap;
    padding: 1rem;
}

/* 진동 애니메이션 */
@keyframes vibrate {
    0% { transform: translateX(0); }
    25% { transform: translateX(-2px); }
    50% { transform: translateX(2px); }
    75% { transform: translateX(-2px); }
    100% { transform: translateX(0); }
}

.vibrate {
    animation: vibrate 0.3s ease-in-out;
}

/* 로딩 스피너 */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}
</style>
"""

def init_mobile_app():
    """모바일 앱 초기화"""
    try:
        st.set_page_config(
            page_title="🏥 SEO Doctor",
            page_icon="🏥",
            layout="centered",  # 모바일에 최적화된 중앙 정렬
            initial_sidebar_state="collapsed"  # 사이드바 숨김
        )
    except Exception:
        # 이미 page config가 설정된 경우 무시
        pass
    
    # 모바일 CSS 적용
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
    
    # 세션 상태 초기화
    if 'diagnosis_history' not in st.session_state:
        st.session_state.diagnosis_history = []
    if 'current_diagnosis' not in st.session_state:
        st.session_state.current_diagnosis = None
    if 'emergency_mode' not in st.session_state:
        st.session_state.emergency_mode = False

def render_mobile_header():
    """모바일 헤더"""
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1>🏥 SEO Doctor</h1>
        <p style="font-size: 1.1rem; color: #666;">
            사이트 응급처치 + 경쟁사 스파이 전문의
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_emergency_button():
    """📱 응급 진단 버튼 (가장 크고 눈에 띄게)"""
    
    st.markdown("""
    <div class="emergency-card">
        <h2>🚨 응급 진단</h2>
        <p>사이트 트래픽 급락? 3분 내 원인 분석!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # URL 입력
    url = st.text_input(
        "🌐 사이트 URL을 입력하세요", 
        placeholder="https://example.com",
        help="진단받을 웹사이트 주소를 입력해주세요"
    )
    
    # 응급 진단 버튼 (큰 버튼)
    if st.button("🚨 응급 진단 시작", type="primary", use_container_width=True):
        if not url:
            st.error("URL을 입력해주세요!")
            return
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # 응급 모드 활성화
        st.session_state.emergency_mode = True
        run_emergency_diagnosis(url)

def run_emergency_diagnosis(url: str):
    """응급 진단 실행"""
    
    # 진행 상황 표시
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🏥 진단 중...")
        
        # 모바일 친화적 진행 표시
        progress_steps = [
            "🔍 사이트 접속 중...",
            "📊 SEO 건강도 스캔...", 
            "🚨 응급 상황 평가...",
            "💊 처방전 작성 중...",
            "✅ 진단 완료!"
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 단계별 진행
        for i, step in enumerate(progress_steps):
            progress_bar.progress((i + 1) / len(progress_steps))
            status_text.text(step)
            time.sleep(1.2)  # 사용자가 볼 수 있도록 충분한 시간
        
        # 실제 진단 실행
        try:
            diagnosis_result = asyncio.run(run_seo_emergency_service(url))
            st.session_state.current_diagnosis = diagnosis_result
            
            # 성공 시 결과 표시
            progress_container.empty()
            display_diagnosis_results(diagnosis_result)
            
        except Exception as e:
            st.error(f"진단 중 오류가 발생했습니다: {e}")
        finally:
            progress_container.empty()

def display_diagnosis_results(diagnosis_result: Dict[str, Any]):
    """📱 모바일 최적화된 진단 결과 표시"""
    
    diagnosis = diagnosis_result['diagnosis']
    
    # 응급 상황 알림
    emergency_level = diagnosis.emergency_level
    
    if emergency_level == SEOEmergencyLevel.CRITICAL:
        st.markdown("""
        <div style="
            background: linear-gradient(45deg, #ff4757, #ff3838);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem 0;
            animation: vibrate 0.5s ease-in-out 3;
        ">
            <h2>🚨 응급 상황!</h2>
            <p style="font-size: 1.2rem;">즉시 치료가 필요합니다!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 진동 효과 (실제 모바일에서는 Haptic Feedback)
        st.balloons()  # 시각적 효과
        
    elif emergency_level == SEOEmergencyLevel.HIGH:
        st.warning("⚠️ 위험: 빠른 조치가 필요합니다!")
    elif emergency_level == SEOEmergencyLevel.EXCELLENT:
        st.success("🚀 완벽: 사이트가 매우 건강합니다!")
    
    # 메인 점수 (큰 숫자로 표시)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🏥 건강도", 
            f"{diagnosis.overall_score:.0f}/100",
            delta=None
        )
    
    with col2:
        st.metric(
            "⏰ 회복 예상", 
            f"{diagnosis.estimated_recovery_days}일",
            delta=None
        )
    
    with col3:
        st.metric(
            "📈 트래픽 예측", 
            diagnosis.traffic_prediction.split()[0],
            delta=None
        )
    
    # 탭으로 정보 구분 (모바일에서 스와이프하기 쉽게)
    tab1, tab2, tab3 = st.tabs(["🚨 응급처치", "🕵️ 경쟁사 정보", "💊 처방전"])
    
    with tab1:
        render_emergency_treatment(diagnosis)
    
    with tab2:
        if diagnosis_result.get('competitor_intelligence'):
            render_competitor_intel(diagnosis_result['competitor_intelligence'])
        else:
            st.info("경쟁사 분석을 원하시면 하단에서 경쟁사 URL을 추가해주세요!")
    
    with tab3:
        render_prescription(diagnosis_result['prescription'])

def render_emergency_treatment(diagnosis):
    """응급 처치 방법 (모바일 최적화)"""
    
    st.markdown("### 🚨 즉시 해야 할 것들")
    
    for i, fix in enumerate(diagnosis.quick_fixes, 1):
        st.markdown(f"""
        <div class="touch-friendly" style="
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        ">
            <strong>{i}. {fix}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # 주요 문제점
    if diagnosis.critical_issues:
        st.markdown("### 🔍 발견된 문제점")
        
        for issue in diagnosis.critical_issues:
            st.markdown(f"- {issue}")

def render_competitor_intel(competitor_intel: List[Dict]):
    """경쟁사 인텔리전스 (모바일 카드 형태)"""
    
    st.markdown("### 🕵️ 경쟁사 분석 결과")
    
    for intel in competitor_intel:
        threat_emoji = {
            CompetitorThreatLevel.DOMINATING: "👑",
            CompetitorThreatLevel.RISING: "📈", 
            CompetitorThreatLevel.STABLE: "➡️",
            CompetitorThreatLevel.DECLINING: "📉",
            CompetitorThreatLevel.WEAK: "😴"
        }
        
        emoji = threat_emoji.get(intel.threat_level, "❓")
        
        st.markdown(f"""
        <div class="competitor-card">
            <h4>{emoji} {intel.competitor_url}</h4>
            <p><strong>위협 수준:</strong> {intel.threat_level.value}</p>
            <p><strong>전략:</strong> {intel.content_strategy}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 확장 가능한 상세 정보
        with st.expander(f"📊 {intel.competitor_url} 상세 분석"):
            
            st.write("**🎯 훔칠 만한 전술:**")
            for tactic in intel.steal_worthy_tactics:
                st.write(f"- {tactic}")
            
            st.write("**🔍 약점:**")
            for vuln in intel.vulnerabilities:
                st.write(f"- {vuln}")
            
            st.write("**📝 콘텐츠 갭:**")
            for gap in intel.content_gaps:
                st.write(f"- {gap}")

def render_prescription(prescription):
    """처방전 (체크리스트 형태)"""
    
    st.markdown("""
    <div class="prescription-card">
        <h3>💊 SEO Doctor 처방전</h3>
        <p>처방전 ID: {}</p>
    </div>
    """.format(prescription.prescription_id), unsafe_allow_html=True)
    
    # 응급 처치 (체크박스로)
    st.markdown("### 🚨 응급 처치 (즉시 실행)")
    
    for i, treatment in enumerate(prescription.emergency_treatment):
        checked = st.checkbox(f"{treatment}", key=f"emergency_{i}")
        if checked:
            st.success("✅ 완료!")
    
    # 주간/월간 처방
    with st.expander("📅 주간 처방"):
        for med in prescription.weekly_medicine:
            st.write(f"- {med}")
    
    with st.expander("🗓️ 월간 체크업"):
        for checkup in prescription.monthly_checkup:
            st.write(f"- {checkup}")
    
    # 예상 결과
    st.info(f"**💡 예상 결과:** {prescription.expected_results}")

def render_competitor_analyzer():
    """경쟁사 분석기 (별도 섹션)"""
    
    st.markdown("---")
    st.markdown("### 🕵️ 경쟁사 스파이 추가 분석")
    
    # 현재 진단이 있을 때만 표시
    if st.session_state.current_diagnosis:
        current_url = st.session_state.current_diagnosis['patient_url']
        
        st.write(f"**현재 분석 중인 사이트:** {current_url}")
        
        # 경쟁사 URL 입력
        competitor_urls = st.text_area(
            "경쟁사 URL 입력 (한 줄에 하나씩)",
            placeholder="https://competitor1.com\nhttps://competitor2.com",
            help="경쟁사 웹사이트 주소를 한 줄에 하나씩 입력해주세요"
        )
        
        if st.button("🕵️ 경쟁사 스파이 시작", use_container_width=True):
            if competitor_urls.strip():
                urls = [url.strip() for url in competitor_urls.split('\n') if url.strip()]
                
                # 경쟁사 분석 실행
                with st.spinner("🕵️ 경쟁사를 몰래 분석하는 중..."):
                    try:
                        analysis_result = asyncio.run(
                            run_seo_emergency_service(current_url, urls)
                        )
                        
                        # 결과 업데이트
                        st.session_state.current_diagnosis = analysis_result
                        
                        st.success("🎯 경쟁사 분석 완료!")
                        st.rerun()  # 페이지 새로고침으로 결과 표시
                        
                    except Exception as e:
                        st.error(f"경쟁사 분석 중 오류: {e}")
            else:
                st.error("경쟁사 URL을 입력해주세요!")
    else:
        st.info("먼저 사이트 진단을 받아주세요!")

def render_quick_actions():
    """빠른 액션 버튼들"""
    
    st.markdown("---")
    st.markdown("### ⚡ 빠른 액션")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 새로운 진단", use_container_width=True):
            # 상태 초기화
            st.session_state.current_diagnosis = None
            st.session_state.emergency_mode = False
            st.rerun()
    
    with col2:
        if st.button("📊 진단 기록", use_container_width=True):
            render_diagnosis_history()

def render_diagnosis_history():
    """진단 기록"""
    
    if st.session_state.diagnosis_history:
        st.markdown("### 📋 진단 기록")
        
        for i, record in enumerate(reversed(st.session_state.diagnosis_history[-5:])):
            with st.expander(f"{i+1}. {record['url']} - {record['date']}"):
                st.write(f"**건강도:** {record['score']}/100")
                st.write(f"**상태:** {record['level']}")
    else:
        st.info("아직 진단 기록이 없습니다.")

def render_success_stories():
    """성공 사례 (사회적 증명)"""
    
    st.markdown("---")
    st.markdown("### 🎉 SEO Doctor 성공 사례")
    
    success_stories = [
        {
            "company": "온라인 쇼핑몰 A",
            "before": "트래픽 90% 감소",
            "after": "3개월 만에 120% 회복",
            "testimonial": "SEO Doctor 덕분에 사업을 살렸어요!"
        },
        {
            "company": "로컬 레스토랑 B", 
            "before": "구글 검색 3페이지",
            "after": "지역 검색 1위 달성",
            "testimonial": "예약이 3배 늘었습니다!"
        },
        {
            "company": "IT 스타트업 C",
            "before": "경쟁사에 밀려 침체",
            "after": "업계 키워드 상위 5위",
            "testimonial": "투자 유치에도 도움이 됐어요!"
        }
    ]
    
    for story in success_stories:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        ">
            <h4>🏆 {story['company']}</h4>
            <p><strong>Before:</strong> {story['before']}</p>
            <p><strong>After:</strong> {story['after']}</p>
            <p><em>"{story['testimonial']}"</em></p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """메인 앱"""
    
    # 모바일 앱 초기화
    init_mobile_app()
    
    # 헤더
    render_mobile_header()
    
    # 메인 기능: 응급 진단
    render_emergency_button()
    
    # 진단 결과가 있으면 표시
    if st.session_state.current_diagnosis:
        st.markdown("---")
        display_diagnosis_results(st.session_state.current_diagnosis)
        
        # 경쟁사 분석 추가
        render_competitor_analyzer()
    
    # 빠른 액션
    render_quick_actions()
    
    # 성공 사례
    render_success_stories()
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 2rem 0;">
        🏥 SEO Doctor v1.0 | 
        24시간 응급실 운영 중 📱
        <br>
        만든이: AI Doctor Team 
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 