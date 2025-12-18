"""
🕸️ GraphRAG Agent Page

LangGraph 기반 지식 그래프 관리 Agent
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
        agent_name="GraphRAG Agent",
        page_icon="🕸️",
        page_type="graphrag",
        title="GraphRAG Agent",
        subtitle="LangGraph 기반 지식 그래프 생성 및 질의응답 시스템",
        module_path="lang_graph.graphrag_agent"
    )

    result_placeholder = st.empty()

    with st.form("graphrag_form"):
        st.subheader("📝 GraphRAG 작업 설정")
        
        command = st.text_area(
            "자연어 명령",
            placeholder="예: Apple을 그래프에 추가해줘",
            height=150
        )
        
        mode = st.selectbox(
            "모드",
            options=["standalone", "interactive"],
            format_func=lambda x: {
                "standalone": "Standalone 모드",
                "interactive": "Interactive 모드"
            }.get(x, x)
        )
        
        submitted = st.form_submit_button("🚀 GraphRAG 실행", use_container_width=True)

    if submitted:
        if not command.strip():
            st.warning("자연어 명령을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('graphrag_agent'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"graphrag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 입력 파라미터 준비
            input_data = {
                "command": command,
                "mode": mode,
                "messages": [{"role": "user", "content": command}],
                "result_json_path": str(result_json_path)
            }

            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="graphrag_agent",
                agent_name="GraphRAG Agent",
                entry_point="lang_graph.graphrag_agent",
                agent_type=AgentType.LANGGRAPH_AGENT,
                capabilities=["graph_creation", "graph_query", "graph_visualization", "knowledge_management"],
                description="LangGraph 기반 지식 그래프 생성 및 질의응답 시스템",
                input_params=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and result.get("success") and result.get("data"):
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 GraphRAG 결과")
    latest_result = result_reader.get_latest_result("graphrag_agent", "graphrag_execution")
    if latest_result:
        with st.expander("🕸️ 최신 GraphRAG 실행 결과", expanded=False):
            display_results(latest_result)
    else:
        st.info("💡 아직 GraphRAG Agent의 결과가 없습니다. 위에서 GraphRAG 작업을 실행해보세요.")

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 GraphRAG 실행 결과")
    if result_data:
        if isinstance(result_data, dict):
            if 'graph_data' in result_data:
                st.markdown("### 🕸️ 그래프 데이터")
                st.json(result_data['graph_data'])
            if 'query_result' in result_data:
                st.markdown("### 💬 질의 결과")
                st.write(result_data['query_result'])
            if 'nodes_added' in result_data:
                st.metric("추가된 노드 수", result_data['nodes_added'])
            if 'edges_added' in result_data:
                st.metric("추가된 엣지 수", result_data['edges_added'])
            st.json(result_data)
        else:
            st.write(str(result_data))
    else:
        st.warning("결과 데이터가 없습니다.")

if __name__ == "__main__":
    main()

