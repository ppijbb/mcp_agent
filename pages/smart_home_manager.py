"""
🏡 Smart Home Manager Agent Page

LangGraph 기반 스마트 홈 관리 Agent
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.page_utils import create_agent_page
from srcs.common.standard_a2a_page_helper import execute_standard_agent_via_a2a
from srcs.common.agent_interface import AgentType
from configs.settings import get_reports_path

try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def main():
    create_agent_page(
        agent_name="Smart Home Manager Agent",
        page_icon="🏡",
        page_type="smart_home",
        title="Smart Home Manager Agent",
        subtitle="LangGraph 기반 스마트 홈 자동화 및 관리 시스템",
        module_path="lang_graph.smart_home_manager"
    )

    result_placeholder = st.empty()

    with st.form("smart_home_form"):
        st.subheader("📝 스마트 홈 제어 요청")
        
        home_command = st.text_area(
            "홈 제어 명령",
            placeholder="예: 저녁 7시에 조명을 켜고 온도를 22도로 설정",
            height=150
        )
        
        device_type = st.multiselect(
            "제어할 디바이스",
            options=["lighting", "temperature", "security", "entertainment"],
            default=["lighting", "temperature"]
        )
        
        submitted = st.form_submit_button("🚀 홈 제어 실행", width='stretch')

    if submitted:
        if not home_command.strip():
            st.warning("홈 제어 명령을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('smart_home'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"smart_home_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

                        # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                
                "agent_id": "smart_home_manager_agent",
                "agent_name": "Smart Home Manager Agent",
                "entry_point": "lang_graph.smart_home_manager",
                agent_type=AgentType.LANGGRAPH_AGENT,
                "capabilities": ["home_automation", "device_control", "smart_home_management"],
                "description": "LangGraph 기반 스마트 홈 자동화 및 관리 시스템"
            ,
                input_params=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Smart Home Manager 결과")
    latest_result = result_reader.get_latest_result("smart_home_agent", "home_control")
    if latest_result:
        with st.expander("🏡 최신 홈 제어 결과", expanded=False):

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 홈 제어 결과")
    if result_data:

if __name__ == "__main__":
    main()

