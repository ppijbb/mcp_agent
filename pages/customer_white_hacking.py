"""
🛡️ Customer White Hacking Agent Page

고객 관점 보안 테스트 AI
표준 A2A 패턴 적용
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.standard_a2a_page_helper import (
    execute_standard_agent_via_a2a,
    process_standard_agent_result
)
from srcs.common.agent_interface import AgentType
from srcs.common.page_utils import create_agent_page
from configs.settings import get_reports_path

try:
    from srcs.utils.result_reader import result_reader
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 보안 테스트 결과")
    if result_data:
        st.json(result_data)

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
        
        submitted = st.form_submit_button("🚀 보안 테스트 시작", use_container_width=True)

    if submitted:
        if not target_url.strip():
            st.warning("테스트 대상 URL을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('white_hacking'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"white_hacking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 표준화된 방식으로 agent 실행 (클래스 기반)
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="customer_white_hacking_agent",
                agent_name="Customer White Hacking Agent",
                entry_point="srcs.enterprise_agents.customer_white_hacking_agent",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["security_testing", "vulnerability_analysis", "penetration_testing"],
                description="고객 관점에서의 보안 취약점 테스트 및 분석",
                input_params={
                    "target_url": target_url,
                    "test_scenarios": test_scenarios
                },
                class_name="CustomerWhiteHackingAgent",
                method_name="run_security_test",
                result_json_path=result_json_path
            )

            # 결과 처리
            processed = process_standard_agent_result(result, "customer_white_hacking_agent")
            if processed["success"] and processed["has_data"]:
                display_results(processed["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 White Hacking 결과")
    latest_result = result_reader.get_latest_result("white_hacking_agent", "security_test")
    if latest_result:
        with st.expander("🛡️ 최신 보안 테스트 결과", expanded=False):
            st.json(latest_result)
    else:
        st.info("💡 아직 Customer White Hacking Agent의 결과가 없습니다. 위에서 보안 테스트를 실행해보세요.")

if __name__ == "__main__":
    main()

