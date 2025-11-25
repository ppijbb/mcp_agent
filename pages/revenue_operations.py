"""
📈 Revenue Operations Intelligence Agent Page

매출 운영 인텔리전스 AI
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
        agent_name="Revenue Operations Intelligence Agent",
        page_icon="📈",
        page_type="revenue",
        title="Revenue Operations Intelligence Agent",
        subtitle="매출 예측, 파이프라인 분석 및 최적화",
        module_path="srcs.enterprise_agents.revenue_operations_intelligence_agent"
    )

    result_placeholder = st.empty()

    with st.form("revenue_form"):
        st.subheader("📝 매출 운영 분석 설정")
        
        company_name = st.text_input("회사명", value="TechCorp Inc.")
        
        analysis_type = st.selectbox(
            "분석 유형",
            options=["revenue_forecast", "pipeline_analysis", "conversion_optimization", "comprehensive"],
            format_func=lambda x: {
                "revenue_forecast": "매출 예측",
                "pipeline_analysis": "파이프라인 분석",
                "conversion_optimization": "전환 최적화",
                "comprehensive": "종합 분석"
            }.get(x, x)
        )
        
        submitted = st.form_submit_button("🚀 매출 분석 시작", use_container_width=True)

    if submitted:
        if not company_name.strip():
            st.warning("회사명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('revenue'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"revenue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            py_executable = sys.executable
            command = [
                py_executable, "-m", "srcs.common.generic_agent_runner",
                "--module-path", "srcs.enterprise_agents.revenue_operations_intelligence_agent",
                "--class-name", "RevenueOperationsIntelligenceAgent",
                "--method-name", "analyze_revenue",
                "--config-json", json.dumps({
                    "company_name": company_name,
                    "analysis_type": analysis_type
                }, ensure_ascii=False),
                "--result-json-path", str(result_json_path)
            ]

            result = run_agent_process(
                placeholder=result_placeholder,
                command=command,
                process_key_prefix="logs/revenue"
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Revenue Operations 결과")
    latest_result = result_reader.get_latest_result("revenue_agent", "revenue_analysis")
    if latest_result:
        with st.expander("📈 최신 매출 운영 분석 결과", expanded=False):
            st.json(latest_result)

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 매출 운영 분석 결과")
    if result_data:
        st.json(result_data)

if __name__ == "__main__":
    main()

