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
        
        submitted = st.form_submit_button("🚀 Graph ReAct 실행", width='stretch')

    if submitted:
        if not query.strip():
            st.warning("질의를 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('graph_react'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"graph_react_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="graph_react_agent",
                agent_name="Graph ReAct Agent",
                entry_point="srcs.advanced_agents.graph_react_agent",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["graph_reasoning", "code_analysis", "refactoring"],
                description="Graph 데이터베이스 기반 추론 및 행동 Agent",
                input_params={
                    "query": query,
                    "graph_path": graph_path if graph_path.strip() else None,
                    "result_json_path": str(result_json_path)
                },
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Graph ReAct 결과")
    latest_result = result_reader.get_latest_result("graph_react_agent", "graph_react_execution")
    if latest_result:
        with st.expander("🕸️ 최신 Graph ReAct 실행 결과", expanded=False):
            display_results(latest_result)

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 Graph ReAct 실행 결과")

    if not result_data:
        st.warning("실행 결과를 찾을 수 없습니다.")
        return

    # result_data가 중첩된 구조일 수 있음
    actual_data = result_data.get('data', result_data)

    # 기본 결과 표시
    if isinstance(actual_data, dict):
        if 'execution_result' in actual_data:
            st.markdown("### ⚙️ 실행 결과")
            st.write(actual_data['execution_result'])

        if 'actions_taken' in actual_data:
            st.markdown("### 🎯 수행된 액션")
            actions = actual_data['actions_taken']
            if isinstance(actions, list):
                for i, action in enumerate(actions, 1):
                    st.write(f"{i}. {action}")
            else:
                st.write(actions)

        if 'nodes_processed' in actual_data:
            st.metric("처리된 노드 수", actual_data['nodes_processed'])

        if 'edges_traversed' in actual_data:
            st.metric("탐색된 엣지 수", actual_data['edges_traversed'])

        # 전체 결과 JSON 표시
        with st.expander("📄 전체 결과 (JSON)", expanded=False):
            st.json(actual_data)
    else:
        # 문자열이나 다른 형식의 결과
        st.write(str(actual_data))

if __name__ == "__main__":
    main()

