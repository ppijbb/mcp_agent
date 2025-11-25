"""
🏢 Hybrid Workplace Optimizer Agent Page

하이브리드 근무 환경 최적화 AI
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
        agent_name="Hybrid Workplace Optimizer Agent",
        page_icon="🏢",
        page_type="workplace",
        title="Hybrid Workplace Optimizer Agent",
        subtitle="하이브리드 근무 환경 최적화 및 생산성 향상",
        module_path="srcs.enterprise_agents.hybrid_workplace_optimizer_agent"
    )

    result_placeholder = st.empty()

    with st.form("workplace_form"):
        st.subheader("📝 근무 환경 분석 설정")
        
        company_name = st.text_input("회사명", value="TechCorp Inc.")
        
        analysis_focus = st.multiselect(
            "분석 초점",
            options=["space_utilization", "productivity", "collaboration", "wellness", "cost_optimization"],
            default=["productivity", "space_utilization"]
        )
        
        submitted = st.form_submit_button("🚀 근무 환경 분석 시작", use_container_width=True)

    if submitted:
        if not company_name.strip():
            st.warning("회사명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('workplace'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"workplace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            py_executable = sys.executable
            command = [
                py_executable, "-m", "srcs.common.generic_agent_runner",
                "--module-path", "srcs.enterprise_agents.hybrid_workplace_optimizer_agent",
                "--class-name", "HybridWorkplaceOptimizerAgent",
                "--method-name", "analyze_workplace",
                "--config-json", json.dumps({
                    "company_name": company_name,
                    "analysis_focus": analysis_focus
                }, ensure_ascii=False),
                "--result-json-path", str(result_json_path)
            ]

            result = run_agent_process(
                placeholder=result_placeholder,
                command=command,
                process_key_prefix="logs/workplace"
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Workplace 결과")
    latest_result = result_reader.get_latest_result("workplace_agent", "workplace_analysis")
    if latest_result:
        with st.expander("🏢 최신 근무 환경 분석 결과", expanded=False):
            st.json(latest_result)

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 근무 환경 분석 결과")
    if result_data:
        st.json(result_data)

if __name__ == "__main__":
    main()

