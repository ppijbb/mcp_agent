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
import json
import os
import streamlit_process_manager as spm
from streamlit_process_manager.process import Process
import plotly.express as px
import plotly.graph_objects as go

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 중앙 설정 임포트
from configs.settings import get_reports_path
REPORTS_PATH = get_reports_path('seo_doctor')

# SEO_AGENT_AVAILABLE 체크는 유지하여 에이전트 존재 여부 확인
try:
    from srcs.seo_doctor.seo_doctor_agent import run_emergency_seo_diagnosis
    SEO_AGENT_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Real SEO Doctor MCP Agent를 불러올 수 없습니다: {e}")
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
    
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    st.info(f"ℹ️ 분석 결과는 자동으로 {REPORTS_PATH}/ 디렉토리에 저장됩니다.")
    st.markdown("---")
    
    if SEO_AGENT_AVAILABLE:
        st.success("🤖 Lighthouse 실시간 분석기가 준비되었습니다!")
        render_seo_analysis_interface()
    else:
        st.error("SEO Doctor 에이전트를 찾을 수 없습니다. srcs/seo_doctor 폴더를 확인해주세요.")

def render_seo_analysis_interface():
    """SEO 분석 인터페이스 (프로세스 모니터링)"""
    
    st.markdown("### 🚨 실시간 SEO 응급 진단")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        with st.form("seo_form"):
            url = st.text_input(
                "🌐 분석할 웹사이트 URL", 
                placeholder="https://example.com"
            )
            include_competitors = st.checkbox("🕵️ 경쟁사 분석 포함", value=True)
            
            submitted = st.form_submit_button("🚨 실시간 SEO 진단 시작", type="primary", use_container_width=True)

            if submitted:
                if not url:
                    st.error("분석할 웹사이트 URL을 입력해주세요.")
                    return

                final_url = url if url.startswith(('http://', 'https://')) else 'https://' + url

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 결과 파일 경로들을 session_state에 저장
                result_filename = f"seo_result_{timestamp}.json"
                st.session_state['seo_result_path'] = os.path.join(REPORTS_PATH, result_filename)
                
                log_filename = f"seo_agent_output_{timestamp}.log"
                st.session_state['seo_log_path'] = os.path.join(REPORTS_PATH, log_filename)

                command = [
                    "python", "-u",
                    "srcs/seo_doctor/run_seo_doctor.py",
                    "--url", final_url,
                    "--output-dir", REPORTS_PATH,
                    "--result-json-path", st.session_state['seo_result_path']
                ]
                if not include_competitors:
                    command.append("--no-competitors")
                
                st.session_state['seo_doctor_command'] = command
                st.session_state['seo_doctor_url'] = final_url
                
    with col2:
        if 'seo_doctor_command' in st.session_state:
            st.info("🔄 SEO Doctor 실행 중...")
            
            process = Process(
                st.session_state['seo_doctor_command'],
                output_file=st.session_state['seo_log_path']
            ).start()
            
            spm.st_process_monitor(
                process,
                label="SEO 진단 분석"
            ).loop_until_finished()
            
            st.success(f"✅ 분석 프로세스가 완료되었습니다. 전체 로그는 {st.session_state['seo_log_path']}에 저장됩니다.")
            
            # 결과 파일 읽기 및 표시
            try:
                with open(st.session_state['seo_result_path'], 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                display_real_analysis_results(result_data, st.session_state['seo_doctor_url'])
            except FileNotFoundError:
                st.error("결과 파일을 찾을 수 없습니다. 에이전트 실행 중 오류가 발생했을 수 있습니다.")
            except Exception as e:
                st.error(f"결과를 표시하는 중 오류가 발생했습니다: {e}")

            # 실행 후 상태 초기화
            for key in ['seo_doctor_command', 'seo_log_path', 'seo_result_path', 'seo_doctor_url']:
                if key in st.session_state:
                    del st.session_state[key]
        else:
            st.markdown("""
            #### 🎯 실시간 분석 특징
            - ✅ **Google Lighthouse** 엔진 사용
            - 🚀 **Core Web Vitals** 측정
            - 🔍 **SEO 점수** 실시간 계산
            - ♿ **접근성** 진단
            - 🛡️ **Best Practices** 검사
            """)

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
    
    if metrics:
        st.markdown("### ⚡ Core Web Vitals")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("⏰ LCP", metrics.get('lcp', 'N/A'))
        with col2: st.metric("🎨 FCP", metrics.get('fcp', 'N/A'))
        with col3: st.metric("📏 CLS", metrics.get('cls', 'N/A'))
    
    if issues:
        st.markdown("### 🚨 발견된 주요 문제점")
        for issue in issues:
            st.warning(issue)
    
    render_score_visualization(scores)
    
    with st.expander("📋 상세 Lighthouse 보고서 (JSON)"):
        st.json(result.get('lighthouse_raw_data', {}))

def render_score_visualization(scores: dict):
    """점수 시각화 차트"""
    if not scores: return
    
    st.markdown("### 📊 점수 시각화")
    col1, col2 = st.columns(2)
    
    with col1:
        categories = list(scores.keys())
        values = list(scores.values())
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='현재 점수'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="카테고리별 점수 분포")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(x=list(scores.keys()), y=list(scores.values()), title="카테고리별 점수", color=list(scores.values()), color_continuous_scale="RdYlGn")
        fig.update_layout(yaxis_range=[0, 100], xaxis_title="카테고리", yaxis_title="점수")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 