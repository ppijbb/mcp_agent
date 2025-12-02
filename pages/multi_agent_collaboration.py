"""
🤝 Multi-Agent Collaboration Page

LangGraph 기반 다중 Agent 협업 시스템
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
        agent_name="Multi-Agent Collaboration",
        page_icon="🤝",
        page_type="multi_agent",
        title="Multi-Agent Collaboration",
        subtitle="LangGraph 기반 다중 Agent 협업 및 통신 시스템",
        module_path="lang_graph.multi_agent_collaboration"
    )

    result_placeholder = st.empty()

    with st.form("multi_agent_form"):
        st.subheader("📝 협업 작업 설정")
        
        collaboration_task = st.text_area(
            "협업 작업 설명",
            placeholder="예: 여러 agent가 협력하여 복잡한 문제 해결",
            height=150
        )
        
        agent_count = st.slider(
            "Agent 수",
            min_value=2,
            max_value=10,
            value=3
        )
        
        submitted = st.form_submit_button("🚀 협업 시작", width='stretch')

    if submitted:
        if not collaboration_task.strip():
            st.warning("협업 작업 설명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('multi_agent_collaboration'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"multi_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

                        # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                
                "agent_id": "multi_agent_collaboration",
                "agent_name": "Multi-Agent Collaboration",
                "entry_point": "lang_graph.multi_agent_collaboration",
                agent_type=AgentType.LANGGRAPH_AGENT,
                "capabilities": ["multi_agent_collaboration", "task_decomposition", "coordination"],
                "description": "LangGraph 기반 다중 Agent 협업 및 통신 시스템"
            ,
                input_params=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Multi-Agent Collaboration 결과")
    latest_result = result_reader.get_latest_result("multi_agent_collaboration", "collaboration_execution")
    if latest_result:
        with st.expander("🤝 최신 협업 실행 결과", expanded=False):

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 협업 실행 결과")
    if result_data:

if __name__ == "__main__":
    main()

