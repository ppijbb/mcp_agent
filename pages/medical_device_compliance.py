"""
🏥 Medical Device Compliance Agent Page

LangGraph 기반 의료기기 규정 준수 Agent
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.page_utils import create_agent_page
from srcs.common.streamlit_a2a_runner import run_agent_via_a2a
from configs.settings import get_reports_path

try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def main():
    create_agent_page(
        agent_name="Medical Device Compliance Agent",
        page_icon="🏥",
        page_type="medical_compliance",
        title="Medical Device Compliance Agent",
        subtitle="LangGraph 기반 의료기기 규정 준수 검토 시스템",
        module_path="lang_graph.medical_device_compliance_agent"
    )

    result_placeholder = st.empty()

    with st.form("medical_compliance_form"):
        st.subheader("📝 의료기기 규정 준수 검토")
        
        device_description = st.text_area(
            "의료기기 설명",
            placeholder="예: 심박수 모니터링 웨어러블 디바이스",
            height=150
        )
        
        regulatory_region = st.selectbox(
            "규제 지역",
            options=["FDA", "CE", "PMDA", "NMPA"],
            help="의료기기 규제 기관 선택"
        )
        
        submitted = st.form_submit_button("🚀 규정 준수 검토 시작", width='stretch')

    if submitted:
        if not device_description.strip():
            st.warning("의료기기 설명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('medical_compliance'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"medical_compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            agent_metadata = {
                "agent_id": "medical_device_compliance_agent",
                "agent_name": "Medical Device Compliance Agent",
                "entry_point": "lang_graph.medical_device_compliance_agent",
                "agent_type": "langgraph_agent",
                "capabilities": ["medical_device_compliance", "regulatory_analysis", "fda_compliance", "ce_compliance"],
                "description": "LangGraph 기반 의료기기 규정 준수 검토 시스템"
            }

            input_data = {
                "device": device_description,
                "region": regulatory_region,
                "messages": [{"role": "user", "content": f"Device: {device_description}, Region: {regulatory_region}"}],
                "result_json_path": str(result_json_path)
            }

            result = run_agent_via_a2a(
                placeholder=result_placeholder,
                agent_metadata=agent_metadata,
                input_data=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Medical Device Compliance 결과")
    latest_result = result_reader.get_latest_result("medical_compliance_agent", "compliance_review")
    if latest_result:
        with st.expander("🏥 최신 규정 준수 검토 결과", expanded=False):

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 규정 준수 검토 결과")
    if result_data:

if __name__ == "__main__":
    main()

