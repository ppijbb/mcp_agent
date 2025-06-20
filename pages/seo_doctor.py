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

# 중앙 설정 임포트
from configs.settings import get_reports_path

# 🚨 CRITICAL UPDATE: Use Real MCP Agent instead of Mock
# Based on: https://medium.com/@matteo28/how-i-solved-a-real-world-customer-problem-with-the-model-context-protocol-mcp-328da5ac76fe

# Real SEO Doctor MCP Agent import
try:
    # P1-4: Import from the correct agent file and only what's needed
    from srcs.seo_doctor.seo_doctor_agent import (
        run_emergency_seo_diagnosis,
        SEOAnalysisResult,
        SEOEmergencyLevel
    )
    SEO_AGENT_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Real SEO Doctor MCP Agent를 불러올 수 없습니다: {e}")
    st.info("srcs/seo_doctor/seo_doctor_agent.py 파일을 확인하고 필요한 의존성을 설치해주세요.")
    SEO_AGENT_AVAILABLE = False

# ✅ P2: Lighthouse fallback system removed - Using real MCP Agent only
# ✅ P1-4: 모든 함수는 이제 srcs.seo_doctor.seo_doctor_agent에서 import됩니다.

def validate_seo_result(result):
    """SEO 분석 결과 검증"""
    if not result:
        raise Exception("SEO 분석에서 유효한 결과를 반환하지 않았습니다")
    return result

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
    
    # 파일 저장 옵션 제거 - 에이전트가 항상 저장함
    st.info(f"ℹ️ 분석 결과는 자동으로 {get_reports_path('seo_doctor')}/ 디렉토리에 저장됩니다.")
    
    st.markdown("---")
    
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
            value=None,
            placeholder="https://example.com",
            help="실시간으로 웹사이트를 분석합니다"
        )
        
        # 분석 옵션 제거 - 에이전트가 기본값(모바일)으로 처리
        st.markdown("<p style='font-size: 0.9rem; color: #888;'>* 현재 모든 분석은 모바일 환경을 기준으로 수행됩니다.</p>", unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🎯 실시간 분석 특징")
        st.markdown("""
        - ✅ **Google Lighthouse** 엔진 사용
        - 🚀 **Core Web Vitals** 측정
        - 🔍 **SEO 점수** 실시간 계산
        - ♿ **접근성** 진단
        - 🛡️ **Best Practices** 검사
        """)

    # 필수 입력 검증
    if not url:
        st.warning("분석할 웹사이트 URL을 입력해주세요.")
    else:
        # 분석 시작 버튼
        if st.button("🚨 실시간 SEO 진단 시작", type="primary", use_container_width=True):
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # 실제 분석 수행
            run_real_lighthouse_analysis(url)

def run_real_lighthouse_analysis(url: str):
    """🚨 REAL MCP Agent Analysis - No More Mock Data"""
    
    # Check if real MCP Agent is available
    if not SEO_AGENT_AVAILABLE:
        st.error("🚨 Real SEO Doctor MCP Agent가 사용 불가능합니다!")
        st.info("srcs/seo_doctor/seo_doctor_agent.py를 확인하고 필요한 의존성을 설치해주세요.")
        return
    
    # 진행 상황 표시
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🏥 Real MCP Agent Emergency Diagnosis")
        st.markdown("**Based on real-world MCP implementation patterns**")
        
        progress_steps = [
            "🚀 Initializing MCP Agent...",
            "🔧 Configuring MCP Servers (g-search, fetch, lighthouse)...",
            "📊 Real website analysis in progress...", 
            "🔍 Core Web Vitals measurement...",
            "🎯 SEO factors examination...",
            "♿ Accessibility diagnosis...",
            "🕵️ Competitor intelligence gathering...",
            "📋 Generating prescription...",
            "✅ Emergency diagnosis complete!"
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 실제 MCP Agent 분석 수행
        try:
            # 🚨 CRITICAL: Use Real MCP Agent instead of mock
            status_text.text("🏥 Running Real MCP Agent Emergency Diagnosis...")
            progress_bar.progress(0.8)
            
            # Execute real SEO analysis
            # The result from this function should be used.
            analysis_result = asyncio.run(run_emergency_seo_diagnosis(
                url=url,
                include_competitors=True,
                output_dir=get_reports_path('seo_doctor')
            ))
            
            # Final steps
            for i, step in enumerate(progress_steps[-2:], len(progress_steps)-2):
                progress_bar.progress((i + 1) / len(progress_steps))
                status_text.text(step)
                time.sleep(0.5)

            # 결과 검증
            validate_seo_result(analysis_result)
            
            # 마지막 단계
            progress_bar.progress(1.0)
            status_text.text(progress_steps[-1])
            time.sleep(1)
            
        except Exception as e:
            st.error(f"분석 중 오류 발생: {str(e)}")
            return
    
    # 진행 바 제거
    progress_container.empty()
    
    # SEOAnalysisResult 객체를 dict로 변환하여 결과 표시에 사용
    if isinstance(analysis_result, SEOAnalysisResult):
        display_data = {
            "url": analysis_result.url,
            "emergency_level": analysis_result.emergency_level.value,
            "overall_score": analysis_result.overall_score,
            "performance_score": analysis_result.performance_score,
            "seo_score": analysis_result.seo_score,
            "accessibility_score": analysis_result.accessibility_score,
            "best_practices_score": analysis_result.best_practices_score,
            "core_web_vitals": analysis_result.core_web_vitals,
            "critical_issues": analysis_result.critical_issues,
            "quick_fixes": analysis_result.quick_fixes,
            "estimated_recovery_days": analysis_result.estimated_recovery_days,
            "competitor_analysis": analysis_result.competitor_analysis,
            "recommendations": analysis_result.recommendations,
            "analysis_timestamp": analysis_result.analysis_timestamp.isoformat(),
            "lighthouse_raw_data": analysis_result.lighthouse_raw_data
        }
        display_real_analysis_results(display_data, url)
    elif "error" in analysis_result:
        st.error(f"❌ 분석 실패: {analysis_result['error']}")
        return
    else:
        st.error("❌ 알 수 없는 분석 결과 형식입니다.")

def display_real_analysis_results(result: dict, url: str):
    """실제 분석 결과 표시"""
    
    # 기본 정보 추출
    overall_score = result.get('overall_score', 0)
    scores = {
        "performance": result.get('performance_score', 0),
        "seo": result.get('seo_score', 0),
        "accessibility": result.get('accessibility_score', 0),
        "best_practices": result.get('best_practices_score', 0)
    }
    metrics = result.get('core_web_vitals', {})
    issues = result.get('critical_issues', [])
    recovery_days = result.get('estimated_recovery_days', 0)
    emergency_level = result.get('emergency_level', '⚠️ 분석 중')
    
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
        <p style="font-size: 1.2rem;">실시간 SEO 건강도 점수 (모바일 기준)</p>
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("⏰ 회복 예상", f"{recovery_days}일")
    
    with col2:
        st.metric("🔍 발견된 문제", f"{len(issues)}개")
    
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
        st.json(result.get('lighthouse_raw_data', {}))

# ✅ P1-4: generate_seo_report_content 함수는 srcs.seo_doctor.seo_doctor_mcp_agent에서 import

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

if __name__ == "__main__":
    main() 