"""
💡 Product Innovation Accelerator Agent Page

제품 혁신 가속화 AI
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.page_utils import create_agent_page
from srcs.common.ui_utils import run_agent_process
from configs.settings import get_reports_path

try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def main():
    create_agent_page(
        agent_name="Product Innovation Accelerator Agent",
        page_icon="💡",
        page_type="innovation",
        title="Product Innovation Accelerator Agent",
        subtitle="제품 혁신 아이디어 생성 및 개발 가속화",
        module_path="srcs.enterprise_agents.product_innovation_accelerator_agent"
    )

    result_placeholder = st.empty()

    with st.form("innovation_form"):
        st.subheader("📝 제품 혁신 분석 설정")
        
        product_domain = st.text_input("제품 도메인", placeholder="예: AI 기반 헬스케어")
        
        innovation_focus = st.selectbox(
            "혁신 초점",
            options=["market_opportunity", "technology_trend", "user_needs", "competitive_analysis"],
            format_func=lambda x: {
                "market_opportunity": "시장 기회",
                "technology_trend": "기술 트렌드",
                "user_needs": "사용자 니즈",
                "competitive_analysis": "경쟁사 분석"
            }.get(x, x)
        )
        
        submitted = st.form_submit_button("🚀 혁신 분석 시작", use_container_width=True)

    if submitted:
        if not product_domain.strip():
            st.warning("제품 도메인을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('innovation'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"innovation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            py_executable = sys.executable
            command = [
                py_executable, "-m", "srcs.common.generic_agent_runner",
                "--module-path", "srcs.enterprise_agents.product_innovation_accelerator_agent",
                "--class-name", "ProductInnovationAcceleratorAgent",
                "--method-name", "analyze_innovation",
                "--config-json", json.dumps({
                    "product_domain": product_domain,
                    "innovation_focus": innovation_focus
                }, ensure_ascii=False),
                "--result-json-path", str(result_json_path)
            ]

            result = run_agent_process(
                placeholder=result_placeholder,
                command=command,
                process_key_prefix="logs/innovation"
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Innovation 결과")
    latest_result = result_reader.get_latest_result("innovation_agent", "innovation_analysis")
    if latest_result:
        with st.expander("💡 최신 제품 혁신 분석 결과", expanded=False):
            st.json(latest_result)

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 제품 혁신 분석 결과")
    if result_data:
        st.json(result_data)

if __name__ == "__main__":
    main()

