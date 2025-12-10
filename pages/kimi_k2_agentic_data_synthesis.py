"""
🧪 Kimi K2 Agentic Data Synthesis Page

LangGraph 기반 데이터 합성 Agent
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
        agent_name="Kimi K2 Agentic Data Synthesis",
        page_icon="🧪",
        page_type="kimi_k2",
        title="Kimi K2 Agentic Data Synthesis",
        subtitle="LangGraph 기반 에이전트식 데이터 합성 시스템",
        module_path="lang_graph.synthetic_data_genertor"
    )

    result_placeholder = st.empty()

    with st.form("kimi_k2_form"):
        st.subheader("📝 데이터 합성 요청")
        
        synthesis_task = st.text_area(
            "합성 작업 설명",
            placeholder="예: 특정 패턴을 가진 시계열 데이터 생성",
            height=150
        )
        
        data_type = st.selectbox(
            "데이터 타입",
            options=["time_series", "tabular", "text", "image"],
            format_func=lambda x: {
                "time_series": "시계열 데이터",
                "tabular": "표 형식 데이터",
                "text": "텍스트 데이터",
                "image": "이미지 데이터"
            }.get(x, x)
        )
        
        submitted = st.form_submit_button("🚀 데이터 합성 시작", width='stretch')

    if submitted:
        if not synthesis_task.strip():
            st.warning("합성 작업 설명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('kimi_k2'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"kimi_k2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 입력 파라미터 준비
            input_data = {
                "synthesis_task": synthesis_task,
                "data_type": data_type,
                "messages": [{"role": "user", "content": synthesis_task}],
                "result_json_path": str(result_json_path)
            }

            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="kimi_k2_agentic_data_synthesis",
                agent_name="Kimi K2 Agentic Data Synthesis",
                entry_point="lang_graph.synthetic_data_genertor",
                agent_type=AgentType.LANGGRAPH_AGENT,
                capabilities=["data_synthesis", "time_series_generation", "tabular_data_generation"],
                description="LangGraph 기반 에이전트식 데이터 합성 시스템",
                input_params=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Kimi K2 결과")
    latest_result = result_reader.get_latest_result("kimi_k2_agent", "data_synthesis")
    if latest_result:
        with st.expander("🧪 최신 데이터 합성 결과", expanded=False):
            display_results(latest_result)
    else:
        st.info("💡 아직 Kimi K2 Agent의 결과가 없습니다. 위에서 데이터 합성을 실행해보세요.")

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 데이터 합성 결과")
    if result_data:
        if isinstance(result_data, dict):
            if 'synthesized_data' in result_data:
                st.markdown("### 📊 합성된 데이터")
                st.json(result_data['synthesized_data'])
            if 'synthesis_metadata' in result_data:
                st.markdown("### 📋 합성 메타데이터")
                st.json(result_data['synthesis_metadata'])
            st.json(result_data)
        else:
            st.write(str(result_data))
    else:
        st.warning("결과 데이터가 없습니다.")

if __name__ == "__main__":
    main()

