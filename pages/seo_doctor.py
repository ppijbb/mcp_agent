"""
🏥 SEO Doctor Page

사이트 응급처치 + 경쟁사 스파이 AI
"""

import streamlit as st
import sys
from pathlib import Path
import time
import asyncio
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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

# 실제 Lighthouse 분석기 임포트
try:
    from srcs.seo_doctor.lighthouse_analyzer import analyze_website_with_lighthouse
    LIGHTHOUSE_AVAILABLE = True
except ImportError:
    LIGHTHOUSE_AVAILABLE = False

# 페이지 설정
try:
    st.set_page_config(
        page_title="🏥 SEO Doctor", 
        page_icon="🏥",
        layout="wide"
    )
except Exception:
    pass

def main():
    """SEO Doctor 메인 페이지"""
    
    # 헤더
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    ">
        <h1>🏥 SEO Doctor</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            AI 기반 실시간 SEO 진단 및 처방 서비스
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    # 파일 저장 옵션 추가
    save_to_file = st.checkbox(
        "SEO 분석 결과를 파일로 저장", 
        value=False,
        help="체크하면 seo_doctor_reports/ 디렉토리에 분석 결과를 파일로 저장합니다"
    )
    
    st.markdown("---")
    
    # Lighthouse 사용 가능 여부 확인
    if not LIGHTHOUSE_AVAILABLE:
        st.error("⚠️ Lighthouse 분석기를 불러올 수 없습니다.")
        st.info("Node.js, Lighthouse, Chrome을 설치해주세요.")
        
        with st.expander("🔧 설치 가이드"):
            st.markdown("""
            ### Lighthouse 환경 설정
            
            1. **Node.js 설치**:
            ```bash
            # Ubuntu/Debian
            curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
            sudo apt-get install -y nodejs
            
            # macOS
            brew install node
            ```
            
            2. **Lighthouse 설치**:
            ```bash
            npm install -g lighthouse chrome-launcher
            ```
            
            3. **Chrome 설치** (헤드리스 모드용):
            ```bash
            # Ubuntu/Debian
            wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
            sudo apt-get install google-chrome-stable
            ```
            """)
        
        # 폴백으로 기본 인터페이스 제공
        render_fallback_interface()
        return
    else:
        st.success("🤖 Lighthouse 실시간 분석기가 준비되었습니다!")

    # 실제 분석 인터페이스
    render_real_seo_analysis()

def render_real_seo_analysis():
    """실제 Lighthouse 분석 인터페이스"""
    
    st.markdown("### 🚨 실시간 SEO 응급 진단")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # URL 입력
        url = st.text_input(
            "🌐 분석할 웹사이트 URL", 
            placeholder="https://example.com",
            help="실시간으로 웹사이트를 분석합니다"
        )
        
        # 분석 옵션
        strategy = st.selectbox(
            "📱 분석 환경",
            ["mobile", "desktop"],
            help="모바일 또는 데스크탑 환경에서 분석"
        )
    
    with col2:
        st.markdown("#### 🎯 실시간 분석 특징")
        st.markdown("""
        - ✅ **Google Lighthouse** 엔진 사용
        - 🚀 **Core Web Vitals** 측정
        - 🔍 **SEO 점수** 실시간 계산
        - ♿ **접근성** 진단
        - 🛡️ **Best Practices** 검사
        """)

    # 분석 시작 버튼
    if st.button("🚨 실시간 SEO 진단 시작", type="primary", use_container_width=True):
        if not url:
            st.error("URL을 입력해주세요!")
            return
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # 실제 분석 수행
        run_real_lighthouse_analysis(url, strategy)

def run_real_lighthouse_analysis(url: str, strategy: str):
    """실제 Lighthouse 분석 수행"""
    
    # 진행 상황 표시
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🔬 Lighthouse 분석 진행 중...")
        
        progress_steps = [
            "🚀 Chrome 브라우저 실행 중...",
            "📊 웹사이트 로딩 및 분석...", 
            "🔍 Core Web Vitals 측정...",
            "🎯 SEO 요소 검사...",
            "♿ 접근성 진단...",
            "📋 분석 결과 생성...",
            "✅ 진단 완료!"
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 실제 분석 수행 (비동기)
        try:
            for i, step in enumerate(progress_steps[:-1]):
                progress_bar.progress((i + 1) / len(progress_steps))
                status_text.text(step)
                time.sleep(1)  # UI 표시용 딜레이
            
            # 실제 Lighthouse 분석 실행
            status_text.text("🔬 Lighthouse 엔진 실행 중... (30-60초 소요)")
            
            # asyncio를 사용하여 분석 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysis_result = loop.run_until_complete(
                analyze_website_with_lighthouse(url, strategy)
            )
            loop.close()
            
            # 마지막 단계
            progress_bar.progress(1.0)
            status_text.text(progress_steps[-1])
            time.sleep(1)
            
        except Exception as e:
            st.error(f"분석 중 오류 발생: {str(e)}")
            return
    
    # 진행 바 제거
    progress_container.empty()
    
    # 분석 결과 표시
    if "error" in analysis_result:
        st.error(f"❌ 분석 실패: {analysis_result['error']}")
        st.info("URL을 확인하거나 잠시 후 다시 시도해주세요.")
        return
    
    display_real_analysis_results(analysis_result, strategy)

def display_real_analysis_results(result: dict, strategy: str):
    """실제 분석 결과 표시"""
    
    # 기본 정보 추출
    overall_score = result.get('overall_score', 0)
    scores = result.get('scores', {})
    metrics = result.get('metrics', {})
    issues = result.get('issues', [])
    recovery_days = result.get('recovery_days', 0)
    emergency_level = result.get('emergency_level', '⚠️ 분석 중')
    improvement_potential = result.get('improvement_potential', 0)
    
    # 응급 레벨에 따른 색상 결정
    if overall_score >= 85:
        color = "#28a745"
    elif overall_score >= 70:
        color = "#17a2b8"
    elif overall_score >= 55:
        color = "#ffc107"
    else:
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
        <h2>{emergency_level}</h2>
        <h1 style="font-size: 3rem; margin: 0;">{overall_score}/100</h1>
        <p style="font-size: 1.2rem;">실시간 SEO 건강도 점수 ({strategy.upper()})</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 상세 점수
    st.markdown("### 📊 카테고리별 상세 점수")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚀 성능", f"{scores.get('performance', 0)}/100")
    
    with col2:
        st.metric("🔍 SEO", f"{scores.get('seo', 0)}/100")
    
    with col3:
        st.metric("♿ 접근성", f"{scores.get('accessibility', 0)}/100")
    
    with col4:
        st.metric("🛡️ Best Practices", f"{scores.get('best_practices', 0)}/100")
    
    # Core Web Vitals 메트릭
    if metrics:
        st.markdown("### ⚡ Core Web Vitals")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("⏰ LCP", metrics.get('lcp', 'N/A'))
        
        with col2:
            st.metric("🎨 FCP", metrics.get('fcp', 'N/A'))
        
        with col3:
            st.metric("📏 CLS", metrics.get('cls', 'N/A'))
    
    # 실시간 예측 메트릭
    st.markdown("### 📈 AI 예측 분석")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("⏰ 회복 예상", f"{recovery_days}일")
    
    with col2:
        st.metric("🔍 발견된 문제", f"{len(issues)}개")
    
    with col3:
        st.metric("📈 개선 가능성", f"+{improvement_potential}%")
    
    # 발견된 문제점들
    if issues:
        st.markdown("### 🚨 발견된 주요 문제점")
        
        for issue in issues:
            st.warning(issue)
    else:
        st.success("🎉 주요 문제점이 발견되지 않았습니다!")
    
    # 차트 시각화
    render_score_visualization(scores)
    
    # 상세 분석 보고서
    with st.expander("📋 상세 Lighthouse 보고서"):
        st.json(result.get('raw_lighthouse_result', {}))

def render_score_visualization(scores: dict):
    """점수 시각화 차트"""
    
    if not scores:
        return
    
    st.markdown("### 📊 점수 시각화")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 레이더 차트
        categories = list(scores.keys())
        values = list(scores.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='현재 점수'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            title="카테고리별 점수 분포"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 바 차트
        fig = px.bar(
            x=categories,
            y=values,
            title="카테고리별 점수",
            color=values,
            color_continuous_scale="RdYlGn"
        )
        
        fig.update_layout(
            yaxis_range=[0, 100],
            xaxis_title="카테고리",
            yaxis_title="점수"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_fallback_interface():
    """Lighthouse 사용 불가능시 폴백 인터페이스"""
    
    st.markdown("### 🔧 시스템 점검 모드")
    st.info("현재 Lighthouse 엔진이 설정되지 않아 시스템 점검 모드로 실행됩니다.")
    
    # 기본 입력 폼은 유지
    url = st.text_input("🌐 웹사이트 URL", placeholder="https://example.com")
    
    if st.button("🔍 기본 점검 시작", use_container_width=True):
        if url:
            st.warning("⚠️ 현재 기본 점검 모드입니다. 정확한 분석을 위해 Lighthouse를 설치해주세요.")
        else:
            st.error("URL을 입력해주세요!")

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