"""
🏥 SEO Doctor Page

사이트 응급처치 + 경쟁사 스파이 AI
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# SEO Doctor 모듈 임포트
try:
    from srcs.seo_doctor.seo_doctor_app import main as seo_main
    from srcs.seo_doctor.seo_doctor_app import *
    SEO_DOCTOR_AVAILABLE = True
except ImportError as e:
    SEO_DOCTOR_AVAILABLE = False
    import_error = str(e)

# 페이지 설정 (SEO Doctor 자체가 page config를 설정하므로 생략)

def main():
    """SEO Doctor 메인 페이지"""
    
    # 헤더
    st.markdown("""
    <div style="
        background: linear-gradient(45deg, #ff4757, #ff3838);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    ">
        <h1>🏥 SEO Doctor</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            사이트 응급처치 + 경쟁사 스파이 전문의
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 다크모드 대응 CSS
    st.markdown("""
    <style>
        .stButton > button {
            background: linear-gradient(135deg, #ff4757, #ff3838) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #ff3838, #ff2f2f) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    st.markdown("---")
    
    # SEO Doctor 실행
    if SEO_DOCTOR_AVAILABLE:
        try:
            # 기존 SEO Doctor의 main 함수 실행
            seo_main()
            
        except Exception as e:
            st.error(f"SEO Doctor 실행 중 오류가 발생했습니다: {e}")
            
            # 대체 인터페이스 제공
            show_fallback_interface()
                
    else:
        st.error("SEO Doctor를 불러올 수 없습니다.")
        st.error(f"오류 내용: {import_error}")
        
        # 대체 UI 제공
        show_fallback_interface()

def show_fallback_interface():
    """대체 인터페이스"""
    
    st.markdown("### 🏥 SEO Doctor 소개")
    
    # 주요 기능 소개
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        ">
            <h3>🚨 응급 진단</h3>
            <p><strong>3분 내</strong> 사이트 문제점 발견</p>
            <ul>
                <li>SEO 건강도 점수 (0-100)</li>
                <li>응급 상황 레벨 판정</li>
                <li>즉시 처방전 제공</li>
                <li>회복 예상 시간</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        ">
            <h3>🕵️ 경쟁사 스파이</h3>
            <p><strong>비밀 정보</strong> 몰래 분석</p>
            <ul>
                <li>경쟁사 SEO 전략 분석</li>
                <li>콘텐츠 갭 발견</li>
                <li>훔칠 만한 전술 추출</li>
                <li>약점 공략 방법</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 바이럴 요소 소개
    st.markdown("---")
    st.markdown("### 🚀 바이럴 기능")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📊 점수 시스템
        - 사이트 건강도 점수
        - 경쟁사와 비교
        - 소셜 공유 유도
        """)
    
    with col2:
        st.markdown("""
        #### 🏆 리더보드
        - 업계 최고 점수
        - 순위 경쟁 심리
        - 개선 동기 부여
        """)
    
    with col3:
        st.markdown("""
        #### 💊 처방전
        - 전문의 진단서
        - 단계별 치료법
        - 성공 사례 공유
        """)
    
    # 데모 진단 시뮬레이션
    st.markdown("---")
    st.markdown("### 🎮 데모 체험")
    
    # URL 입력
    demo_url = st.text_input(
        "🌐 사이트 URL을 입력해보세요 (데모용)",
        placeholder="https://example.com",
        help="실제 분석은 아니지만 인터페이스를 체험할 수 있습니다"
    )
    
    if st.button("🚨 응급 진단 시작 (데모)", use_container_width=True):
        if demo_url:
            show_demo_diagnosis(demo_url)
        else:
            st.error("URL을 입력해주세요!")

def show_demo_diagnosis(url):
    """데모 진단 결과"""
    
    import random
    import time
    
    # 진행 바 시뮬레이션
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🏥 진단 중...")
        
        progress_steps = [
            "🔍 사이트 접속 중...",
            "📊 SEO 건강도 스캔...", 
            "🚨 응급 상황 평가...",
            "💊 처방전 작성 중...",
            "✅ 진단 완료!"
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, step in enumerate(progress_steps):
            progress_bar.progress((i + 1) / len(progress_steps))
            status_text.text(step)
            time.sleep(0.8)
    
    # 진행 바 제거
    progress_container.empty()
    
    # 가상 진단 결과
    score = random.randint(30, 95)
    
    if score >= 80:
        level = "🚀 완벽"
        color = "#28a745"
    elif score >= 60:
        level = "✅ 안전"
        color = "#17a2b8"
    elif score >= 40:
        level = "⚠️ 위험"
        color = "#ffc107"
    else:
        level = "🚨 응급실"
        color = "#dc3545"
    
    # 결과 카드
    st.markdown(f"""
    <div style="
        background: {color};
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    ">
        <h2>{level}</h2>
        <h1 style="font-size: 3rem; margin: 0;">{score}/100</h1>
        <p style="font-size: 1.2rem;">SEO 건강도 점수</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 상세 결과
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("⏰ 회복 예상", f"{random.randint(7, 90)}일")
    
    with col2:
        st.metric("🔍 발견된 문제", f"{random.randint(2, 8)}개")
    
    with col3:
        st.metric("📈 개선 가능성", f"+{random.randint(10, 40)}%")
    
    # 처방전
    st.markdown("---")
    st.markdown("### 💊 처방전")
    
    emergency_fixes = [
        "🚨 즉시: robots.txt 확인 및 수정",
        "⚡ 1시간 내: 404 에러 페이지 수정",
        "🔧 오늘 내: 페이지 속도 최적화",
        "📝 이번 주: 중복 콘텐츠 제거"
    ]
    
    for i, fix in enumerate(emergency_fixes[:3], 1):
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        ">
            <strong>{i}. {fix}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # 공유 버튼들
    st.markdown("---")
    st.markdown("### 📱 결과 공유")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 결과 복사", use_container_width=True):
            st.success("클립보드에 복사되었습니다!")
    
    with col2:
        if st.button("💬 카카오톡 공유", use_container_width=True):
            st.info("카카오톡 공유 기능")
    
    with col3:
        if st.button("📊 상세 분석", use_container_width=True):
            st.info("프리미엄 기능입니다!")
    
    # 경쟁사 분석 추가
    st.markdown("---")
    st.markdown("### 🕵️ 경쟁사 스파이 추가 분석")
    
    competitor_urls = st.text_area(
        "경쟁사 URL 입력 (한 줄에 하나씩)",
        placeholder="https://competitor1.com\nhttps://competitor2.com",
        help="경쟁사 웹사이트 주소를 입력하면 비교 분석합니다"
    )
    
    if st.button("🕵️ 경쟁사 스파이 시작 (데모)", use_container_width=True):
        if competitor_urls.strip():
            show_competitor_demo()
        else:
            st.error("경쟁사 URL을 입력해주세요!")

def show_competitor_demo():
    """경쟁사 분석 데모"""
    
    import random
    
    st.markdown("#### 🏆 경쟁사 분석 결과")
    
    competitors = [
        {"name": "경쟁사 A", "score": random.randint(60, 95), "threat": "👑 지배중"},
        {"name": "경쟁사 B", "score": random.randint(40, 80), "threat": "📈 급상승"},
        {"name": "경쟁사 C", "score": random.randint(30, 70), "threat": "➡️ 안정"}
    ]
    
    for comp in competitors:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        ">
            <h4>{comp['threat']} {comp['name']}</h4>
            <p><strong>SEO 점수:</strong> {comp['score']}/100</p>
            <p><strong>위협 수준:</strong> {comp['threat']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 훔칠 만한 전술
    st.markdown("#### 🎯 훔칠 만한 전술")
    
    tactics = [
        "🎯 FAQ 섹션으로 롱테일 키워드 공략",
        "📊 인포그래픽으로 복잡한 정보 시각화",
        "🔗 관련 업체들과 상호 링크 교환",
        "📱 모바일 우선 콘텐츠 제작"
    ]
    
    for tactic in tactics:
        st.write(f"- {tactic}")

# 수동 설치 가이드
with st.expander("🔧 SEO Doctor 수동 실행 가이드"):
    st.markdown("""
    ### SEO Doctor 설정 및 실행
    
    1. **디렉토리 이동**:
    ```bash
    cd srcs/seo_doctor
    ```
    
    2. **필요한 패키지 설치**:
    ```bash
    pip install streamlit plotly pandas asyncio
    ```
    
    3. **SEO Doctor 실행**:
    ```bash
    streamlit run seo_doctor_app.py --server.port 8502
    ```
    
    4. **런처 사용** (추천):
    ```bash
    python ../../seo_doctor_launcher.py
    ```
    
    ### 🎯 주요 특징
    - **모바일 최적화**: 터치 친화적 UI
    - **3분 진단**: 빠른 결과 제공
    - **바이럴 요소**: 점수 공유, 경쟁 심리
    - **실시간 분석**: 즉시 처방전 생성
    """)

if __name__ == "__main__":
    main() 