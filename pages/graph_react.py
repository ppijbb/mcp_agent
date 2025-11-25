"""
🕸️ Graph ReAct Agent Page

Graph 기반 ReAct Agent
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
        agent_name="Graph ReAct Agent",
        page_icon="🕸️",
        page_type="graph_react",
        title="Graph ReAct Agent",
        subtitle="Graph 데이터베이스 기반 추론 및 행동 Agent",
        module_path="srcs.advanced_agents.graph_react_agent"
    )

    result_placeholder = st.empty()

    with st.form("graph_react_form"):
        st.subheader("📝 Graph ReAct 작업 설정")
        
        query = st.text_area(
            "질의",
            placeholder="예: 특정 패턴을 가진 코드를 찾아서 리팩토링해줘",
            height=150
        )
        
        graph_path = st.text_input("Graph 경로 (선택)", placeholder="기본 그래프 사용")
        
        submitted = st.form_submit_button("🚀 Graph ReAct 실행", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("질의를 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('graph_react'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"graph_react_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            py_executable = sys.executable
            command = [
                py_executable, "-m", "srcs.advanced_agents.graph_react_agent",
                "--query", query,
                "--result-json-path", str(result_json_path)
            ]
            if graph_path.strip():
                command.extend(["--graph-path", graph_path])

            result = run_agent_process(
                placeholder=result_placeholder,
                command=command,
                process_key_prefix="logs/graph_react"
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Graph ReAct 결과")
    latest_result = result_reader.get_latest_result("graph_react_agent", "graph_react_execution")
    if latest_result:
        with st.expander("🕸️ 최신 Graph ReAct 실행 결과", expanded=False):
            st.json(latest_result)

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 Graph ReAct 실행 결과")
    if result_data:
        st.json(result_data)

if __name__ == "__main__":
    main()

