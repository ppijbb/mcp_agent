"""
🎯 Business Strategy Agent Page

비즈니스 전략 수립과 시장 분석을 위한 AI 어시스턴트
"""

import streamlit as st
import sys
from pathlib import Path
import os
from datetime import datetime
import json
import streamlit_process_manager as spm
from streamlit_process_manager.process import Process

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 설정 파일에서 경로 가져오기
try:
    from configs.settings import get_reports_path
    REPORTS_PATH = get_reports_path('business_strategy')
except ImportError:
    st.error("❌ 설정 파일을 찾을 수 없습니다. configs/settings.py를 확인해주세요.")
    st.stop()

# 공통 스타일 및 유틸리티 임포트
from srcs.common.styles import get_common_styles, get_page_header
from srcs.common.page_utils import setup_page, render_home_button, create_agent_page
from srcs.common.ui_utils import run_agent_process
from srcs.business_strategy_agents.run_business_strategy_agents import BusinessStrategyRunner

# 페이지 설정
setup_page("🎯 Business Strategy Agent", "🎯")

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 분석 결과 요약")
    
    if not result_data or "summary" not in result_data:
        st.warning("분석 결과를 찾을 수 없습니다.")
        return

    summary = result_data.get("summary", {})
    results = result_data.get("results", {})

    st.metric("총 실행 시간", f"{summary.get('execution_time', 0):.2f}초")
    
    st.markdown("#### 📄 생성된 보고서 목록")
    for agent_name, result in results.items():
        if result.get("success") and "output_file" in result:
            st.success(f"**{agent_name.replace('_', ' ').title()}**: `{result['output_file']}`")
        else:
            st.error(f"**{agent_name.replace('_', ' ').title()}**: 실패 - {result.get('error', '알 수 없는 오류')}")

    # Display content of the unified strategy report if it exists
    unified_report_path = results.get("unified_strategy", {}).get("output_file")
    if unified_report_path and os.path.exists(unified_report_path):
        with st.expander("📈 통합 전략 보고서 보기", expanded=True):
            with open(unified_report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            st.markdown(report_content)

def main():
    create_agent_page(
        "📈 Business Strategy Agent",
        "AI 기반 비즈니스 전략 수립 및 시장 분석 플랫폼",
        "pages/business_strategy.py"
    )

    result_placeholder = st.empty()

    with st.form("business_strategy_form"):
        st.subheader("📝 분석 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            keywords_input = st.text_input("🔍 핵심 키워드 (쉼표로 구분)", "AI, fintech, sustainability")
            business_context_input = st.text_area("🏢 비즈니스 맥락", "AI 스타트업, 핀테크 회사 등")
        with col2:
            objectives_input = st.text_input("🎯 목표 (쉼표로 구분)", "growth, expansion, efficiency")
            regions_input = st.text_input("🌍 타겟 지역 (쉼표로 구분)", "North America, Europe, Asia")

        st.subheader("⚙️ 고급 설정")
        col3, col4 = st.columns(2)
        with col3:
            time_horizon = st.selectbox("⏰ 분석 기간", ["3_months", "6_months", "12_months", "24_months"], index=2)
        with col4:
            analysis_mode = st.selectbox("🔄 분석 모드", ["unified", "individual", "both"], index=0)
        
        submitted = st.form_submit_button("🚀 비즈니스 전략 분석 시작", use_container_width=True)

    if submitted:
        if not keywords_input.strip():
            st.warning("핵심 키워드를 입력해주세요.")
        else:
            reports_path = get_reports_path('business_strategy')
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # runner.run_full_suite에 맞는 config 객체 생성
            config = {
                'keywords': [k.strip() for k in keywords_input.split(',')],
                'business_context': {"description": business_context_input} if business_context_input.strip() else None,
                'objectives': [o.strip() for o in objectives_input.split(',')] if objectives_input.strip() else None,
                'regions': [r.strip() for r in regions_input.split(',')] if regions_input.strip() else None,
                'time_horizon': time_horizon,
                'mode': analysis_mode
            }

            py_executable = sys.executable
            command = [
                py_executable, "-m", "srcs.common.generic_agent_runner",
                "--module-path", "srcs.business_strategy_agents.run_business_strategy_agents",
                "--class-name", "BusinessStrategyRunner",
                "--method-name", "run_full_suite",
                "--config-json", json.dumps(config, ensure_ascii=False),
                "--result-json-path", str(result_json_path)
            ]

            result = run_agent_process(
                placeholder=result_placeholder, 
                command=command, 
                process_key_prefix="business_strategy"
            )

            if result and "data" in result:
                display_results(result["data"])

if __name__ == "__main__":
    main() 