"""
🛡️ Customer White Hacking Agent Page

고객 관점 보안 테스트 AI
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
        agent_name="Customer White Hacking Agent",
        page_icon="🛡️",
        page_type="white_hacking",
        title="Customer White Hacking Agent",
        subtitle="고객 관점에서의 보안 취약점 테스트 및 분석",
        module_path="srcs.enterprise_agents.customer_white_hacking_agent"
    )

    result_placeholder = st.empty()

    with st.form("white_hacking_form"):
        st.subheader("📝 보안 테스트 설정")
        
        target_url = st.text_input("테스트 대상 URL", placeholder="https://example.com")
        
        test_scenarios = st.multiselect(
            "테스트 시나리오",
            options=["authentication", "authorization", "input_validation", "session_management"],
            default=["authentication", "input_validation"]
        )
        
        submitted = st.form_submit_button("🚀 보안 테스트 시작", width='stretch')

    if submitted:
        if not target_url.strip():
            st.warning("테스트 대상 URL을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('white_hacking'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"white_hacking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            agent_metadata = {
                "agent_id": "customer_white_hacking_agent",
                "agent_name": "Customer White Hacking Agent",
                "entry_point": "srcs.common.generic_agent_runner",
                "agent_type": "mcp_agent",
                "capabilities": ["security_testing", "vulnerability_analysis", "penetration_testing"],
                "description": "고객 관점에서의 보안 취약점 테스트 및 분석"
            }

            input_data = {
                "module_path": "srcs.enterprise_agents.customer_white_hacking_agent",
                "class_name": "CustomerWhiteHackingAgent",
                "method_name": "run_security_test",
                "config": {
                    "target_url": target_url,
                    "test_scenarios": test_scenarios
                },
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
    st.markdown("## 📊 최신 White Hacking 결과")
    latest_result = result_reader.get_latest_result("white_hacking_agent", "security_test")
    if latest_result:
        with st.expander("🛡️ 최신 보안 테스트 결과", expanded=False):
            st.json(latest_result)

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 보안 테스트 결과")
    if result_data:
        st.json(result_data)

if __name__ == "__main__":
    main()

