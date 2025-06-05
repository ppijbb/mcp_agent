"""
🔒 Cybersecurity Agent Page

사이버 보안 인프라 관리 및 위협 탐지
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import random
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 페이지 설정
st.set_page_config(
    page_title="🔒 Cybersecurity Agent",
    page_icon="🔒",
    layout="wide"
)

def main():
    """Cybersecurity Agent 메인 페이지"""
    
    # 헤더
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #ff4757, #ff3838);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    ">
        <h1>🔒 Cybersecurity Agent</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            AI 기반 사이버 보안 인프라 관리 및 위협 탐지 시스템
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    st.markdown("---")
    
    # 실시간 보안 대시보드
    render_security_dashboard()
    
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚨 위협 탐지", 
        "🛡️ 보안 점검", 
        "📊 보안 분석",
        "⚙️ 설정"
    ])
    
    with tab1:
        render_threat_detection()
    
    with tab2:
        render_security_check()
    
    with tab3:
        render_security_analysis()
    
    with tab4:
        render_security_settings()

def render_security_dashboard():
    """실시간 보안 대시보드"""
    
    st.markdown("### 🔒 실시간 보안 현황")
    
    # 보안 지표
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        threat_level = random.choice(["낮음", "보통", "높음", "위험"])
        color = {"낮음": "green", "보통": "blue", "높음": "orange", "위험": "red"}[threat_level]
        st.markdown(f"""
        <div style="
            background: {color};
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        ">
            <h3>위협 수준</h3>
            <h2>{threat_level}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("🚨 탐지된 위협", f"{random.randint(0, 15)}개", f"{random.randint(-5, 3):+d}")
    
    with col3:
        st.metric("🛡️ 차단된 공격", f"{random.randint(50, 200)}개", f"{random.randint(10, 50):+d}")
    
    with col4:
        st.metric("📊 보안 점수", f"{random.randint(75, 98)}/100", f"{random.randint(-2, 5):+d}")
    
    # 실시간 위협 맵
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌍 실시간 위협 지도")
        
        # 가상 위협 데이터
        threat_data = pd.DataFrame({
            '국가': ['중국', '러시아', '미국', '북한', '이란', '브라질'],
            '위협수': [random.randint(10, 50) for _ in range(6)],
            '위협유형': ['DDoS', 'Malware', 'Phishing', 'APT', 'Ransomware', 'Botnet']
        })
        
        fig = px.bar(threat_data, x='국가', y='위협수', color='위협유형', 
                    title='국가별 위협 현황')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 시간별 트래픽")
        
        # 시간별 트래픽 데이터
        hours = list(range(24))
        normal_traffic = [random.randint(100, 500) for _ in hours]
        suspicious_traffic = [random.randint(0, 50) for _ in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=normal_traffic, name='정상 트래픽', 
                               line=dict(color='green')))
        fig.add_trace(go.Scatter(x=hours, y=suspicious_traffic, name='의심 트래픽', 
                               line=dict(color='red')))
        fig.update_layout(title='24시간 트래픽 모니터링', xaxis_title='시간', yaxis_title='요청 수')
        st.plotly_chart(fig, use_container_width=True)

def render_threat_detection():
    """위협 탐지 섹션"""
    
    st.markdown("### 🚨 실시간 위협 탐지")
    
    # 최근 탐지된 위협들
    threats = [
        {"시간": "2024-11-15 14:23", "유형": "DDoS", "심각도": "높음", "출발지": "203.123.45.67", "상태": "차단됨"},
        {"시간": "2024-11-15 14:18", "유형": "Malware", "심각도": "중간", "출발지": "192.168.1.100", "상태": "격리됨"},
        {"시간": "2024-11-15 14:15", "유형": "Phishing", "심각도": "낮음", "출발지": "suspicious@fake.com", "상태": "모니터링"},
        {"시간": "2024-11-15 14:10", "유형": "Brute Force", "심각도": "높음", "출발지": "45.67.89.123", "상태": "차단됨"},
        {"시간": "2024-11-15 14:05", "유형": "SQL Injection", "심각도": "중간", "출발지": "web-scanner.com", "상태": "차단됨"}
    ]
    
    threat_df = pd.DataFrame(threats)
    
    # 위협 필터링
    col1, col2, col3 = st.columns(3)
    
    with col1:
        severity_filter = st.selectbox("심각도 필터", ["전체", "높음", "중간", "낮음"])
    
    with col2:
        threat_type_filter = st.selectbox("위협 유형", ["전체", "DDoS", "Malware", "Phishing", "Brute Force", "SQL Injection"])
    
    with col3:
        status_filter = st.selectbox("상태 필터", ["전체", "차단됨", "격리됨", "모니터링"])
    
    # 필터 적용
    filtered_df = threat_df.copy()
    if severity_filter != "전체":
        filtered_df = filtered_df[filtered_df['심각도'] == severity_filter]
    if threat_type_filter != "전체":
        filtered_df = filtered_df[filtered_df['유형'] == threat_type_filter]
    if status_filter != "전체":
        filtered_df = filtered_df[filtered_df['상태'] == status_filter]
    
    # 위협 목록 표시
    st.dataframe(filtered_df, use_container_width=True)
    
    # 자동 대응 설정
    st.markdown("---")
    st.markdown("#### ⚙️ 자동 대응 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_block = st.checkbox("자동 차단 활성화", value=True)
        auto_quarantine = st.checkbox("자동 격리 활성화", value=True)
        
    with col2:
        notification_email = st.checkbox("이메일 알림", value=True)
        notification_sms = st.checkbox("SMS 알림", value=False)
    
    if st.button("🔧 설정 저장", use_container_width=True):
        st.success("자동 대응 설정이 저장되었습니다!")

def render_security_check():
    """보안 점검 섹션"""
    
    st.markdown("### 🛡️ 종합 보안 점검")
    
    # 점검 실행
    if st.button("🔍 보안 점검 시작", use_container_width=True):
        
        # 진행 바
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        checks = [
            "방화벽 상태 확인",
            "안티바이러스 업데이트 확인", 
            "시스템 패치 상태 점검",
            "사용자 권한 검토",
            "네트워크 보안 스캔",
            "데이터베이스 보안 점검"
        ]
        
        results = []
        
        for i, check in enumerate(checks):
            progress_bar.progress((i + 1) / len(checks))
            status_text.text(f"진행 중: {check}")
            
            # 가상 결과 생성
            status = random.choice(["정상", "주의", "위험"])
            score = random.randint(60, 100) if status == "정상" else random.randint(30, 80)
            
            results.append({
                "점검 항목": check,
                "상태": status,
                "점수": score,
                "권장사항": get_recommendation(check, status)
            })
            
            import time
            time.sleep(0.5)
        
        # 결과 표시
        progress_bar.empty()
        status_text.empty()
        
        st.markdown("#### 📋 점검 결과")
        
        results_df = pd.DataFrame(results)
        
        # 상태별 색상 적용
        def color_status(val):
            if val == "정상":
                return "background-color: #d4edda; color: #155724"
            elif val == "주의":
                return "background-color: #fff3cd; color: #856404"
            else:
                return "background-color: #f8d7da; color: #721c24"
        
        styled_df = results_df.style.applymap(color_status, subset=['상태'])
        st.dataframe(styled_df, use_container_width=True)
        
        # 종합 점수
        avg_score = results_df['점수'].mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("종합 보안 점수", f"{avg_score:.0f}/100")
        
        with col2:
            normal_count = len(results_df[results_df['상태'] == '정상'])
            st.metric("정상 항목", f"{normal_count}/{len(results)}")
        
        with col3:
            risk_count = len(results_df[results_df['상태'] == '위험'])
            st.metric("위험 항목", f"{risk_count}/{len(results)}")

def get_recommendation(check_item, status):
    """점검 항목별 권장사항"""
    
    recommendations = {
        "방화벽 상태 확인": {
            "정상": "방화벽이 정상 작동 중입니다.",
            "주의": "방화벽 규칙을 업데이트하세요.",
            "위험": "방화벽을 즉시 활성화하세요."
        },
        "안티바이러스 업데이트 확인": {
            "정상": "최신 바이러스 정의 파일이 적용되었습니다.",
            "주의": "바이러스 정의 파일을 업데이트하세요.",
            "위험": "안티바이러스를 즉시 업데이트하세요."
        },
        "시스템 패치 상태 점검": {
            "정상": "모든 보안 패치가 적용되었습니다.",
            "주의": "일부 패치가 누락되었습니다.",
            "위험": "중요 보안 패치를 즉시 적용하세요."
        }
    }
    
    return recommendations.get(check_item, {}).get(status, "추가 검토가 필요합니다.")

def render_security_analysis():
    """보안 분석 섹션"""
    
    st.markdown("### 📊 보안 분석 리포트")
    
    # 월별 보안 동향
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 월별 위협 동향")
        
        months = ['1월', '2월', '3월', '4월', '5월', '6월']
        threats = [random.randint(50, 200) for _ in months]
        blocked = [random.randint(40, 180) for _ in months]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=months, y=threats, name='탐지된 위협', marker_color='red'))
        fig.add_trace(go.Bar(x=months, y=blocked, name='차단된 위협', marker_color='green'))
        fig.update_layout(title='월별 위협 탐지 및 차단 현황')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 위협 유형별 분포")
        
        threat_types = ['DDoS', 'Malware', 'Phishing', 'Brute Force', 'SQL Injection']
        threat_counts = [random.randint(10, 50) for _ in threat_types]
        
        fig = px.pie(values=threat_counts, names=threat_types, title='위협 유형별 분포')
        st.plotly_chart(fig, use_container_width=True)
    
    # 보안 권장사항
    st.markdown("---")
    st.markdown("#### 💡 보안 강화 권장사항")
    
    recommendations = [
        "🔐 다단계 인증(MFA) 도입으로 계정 보안 강화",
        "🛡️ 제로 트러스트 보안 모델 적용 검토",
        "📚 직원 보안 교육 프로그램 정기 실시",
        "🔄 정기적인 보안 감사 및 취약점 점검",
        "💾 중요 데이터 백업 및 복구 계획 수립",
        "🚨 보안 사고 대응 절차 문서화"
    ]
    
    for rec in recommendations:
        st.write(f"- {rec}")

def render_security_settings():
    """보안 설정 섹션"""
    
    st.markdown("### ⚙️ 보안 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 일반 설정")
        
        scan_frequency = st.selectbox("스캔 주기", ["실시간", "1시간", "6시간", "24시간"])
        log_retention = st.slider("로그 보관 기간 (일)", 7, 365, 90)
        alert_threshold = st.slider("알림 임계값", 1, 10, 5)
        
        st.markdown("#### 📧 알림 설정")
        
        email_alerts = st.checkbox("이메일 알림", value=True)
        sms_alerts = st.checkbox("SMS 알림", value=False)
        slack_alerts = st.checkbox("Slack 알림", value=True)
        
        if email_alerts:
            email_address = st.text_input("알림 이메일", "admin@company.com")
        
    with col2:
        st.markdown("#### 🛡️ 보안 정책")
        
        password_policy = st.selectbox("비밀번호 정책", ["기본", "강화", "최고"])
        session_timeout = st.slider("세션 타임아웃 (분)", 15, 480, 60)
        failed_login_limit = st.slider("로그인 실패 제한", 3, 10, 5)
        
        st.markdown("#### 🚨 자동 대응")
        
        auto_block_ip = st.checkbox("의심 IP 자동 차단", value=True)
        auto_quarantine = st.checkbox("악성 파일 자동 격리", value=True)
        auto_patch = st.checkbox("자동 보안 패치", value=False)
    
    # 설정 저장
    if st.button("💾 설정 저장", use_container_width=True):
        st.success("보안 설정이 저장되었습니다!")
        
        # 설정 요약 표시
        with st.expander("📋 저장된 설정 요약"):
            st.write(f"- 스캔 주기: {scan_frequency}")
            st.write(f"- 로그 보관: {log_retention}일")
            st.write(f"- 알림 임계값: {alert_threshold}")
            st.write(f"- 비밀번호 정책: {password_policy}")
            st.write(f"- 세션 타임아웃: {session_timeout}분")

if __name__ == "__main__":
    main() 