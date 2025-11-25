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
from srcs.common.streamlit_a2a_runner import run_agent_via_a2a
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

            agent_metadata = {
                "agent_id": "graphrag_agent",
                "agent_name": "GraphRAG Agent",
                "entry_point": "lang_graph.graphrag_agent",
                "agent_type": "langgraph_agent",
                "capabilities": ["graph_creation", "graph_query", "graph_visualization", "knowledge_management"],
                "description": "LangGraph 기반 지식 그래프 생성 및 질의응답 시스템"
            }

            input_data = {
                "command": command,
                "mode": "standalone",
                "messages": [{"role": "user", "content": command}],
                "result_json_path": str(result_json_path)
            }

            result = run_agent_via_a2a(
                placeholder=result_placeholder,
                agent_metadata=agent_metadata,
                input_data=input_data,
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
            st.json(latest_result)

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 GraphRAG 실행 결과")
    if result_data:
        st.json(result_data)

if __name__ == "__main__":
    main()

