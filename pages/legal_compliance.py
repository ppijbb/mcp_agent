"""
⚖️ Legal Compliance Agent Page

법률 준수 및 규정 관리 AI
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
        agent_name="Legal Compliance Agent",
        page_icon="⚖️",
        page_type="legal",
        title="Legal Compliance Agent",
        subtitle="법률 준수 검토, 규정 관리 및 리스크 분석",
        module_path="srcs.enterprise_agents.legal_compliance_agent"
    )

    result_placeholder = st.empty()

    with st.form("legal_form"):
        st.subheader("📝 법률 준수 검토 설정")
        
        company_name = st.text_input("회사명", value="TechCorp Inc.")
        
        compliance_areas = st.multiselect(
            "준수 영역",
            options=["GDPR", "CCPA", "HIPAA", "SOX", "PCI-DSS", "ISO27001"],
            default=["GDPR", "CCPA"]
        )
        
        submitted = st.form_submit_button("🚀 법률 준수 검토 시작", use_container_width=True)

    if submitted:
        if not company_name.strip():
            st.warning("회사명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('legal'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"legal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            py_executable = sys.executable
            command = [
                py_executable, "-m", "srcs.enterprise_agents.legal_compliance_agent",
                "--company-name", company_name,
                "--compliance-areas", json.dumps(compliance_areas, ensure_ascii=False),
                "--result-json-path", str(result_json_path)
            ]

            result = run_agent_process(
                placeholder=result_placeholder,
                command=command,
                process_key_prefix="logs/legal"
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Legal Compliance 결과")
    latest_result = result_reader.get_latest_result("legal_agent", "compliance_review")
    if latest_result:
        with st.expander("⚖️ 최신 법률 준수 검토 결과", expanded=False):
            st.json(latest_result)

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 법률 준수 검토 결과")
    if result_data:
        st.json(result_data)

if __name__ == "__main__":
    main()

