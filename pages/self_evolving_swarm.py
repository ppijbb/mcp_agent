"""
🧬 Self Evolving Swarm Agent Page

자기 진화형 Swarm Agent 시스템
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
    st.subheader("📊 진화 실행 결과")
    if result_data:
        st.json(result_data)

def main():
    create_agent_page(
        agent_name="Self Evolving Swarm Agent",
        page_icon="🧬",
        page_type="evolving_swarm",
        title="Self Evolving Swarm Agent",
        subtitle="자기 진화형 multi-agent 시스템",
        module_path="srcs.advanced_agents.self_evolving_swarm"
    )

    result_placeholder = st.empty()

    with st.form("evolving_swarm_form"):
        st.subheader("📝 진화 작업 설정")
        
        task_description = st.text_area(
            "작업 설명",
            placeholder="예: 복잡한 문제를 해결하기 위해 agent들이 스스로 진화하며 협업",
            height=150
        )
        
        evolution_steps = st.slider(
            "진화 단계 수",
            min_value=1,
            max_value=10,
            value=5
        )
        
        submitted = st.form_submit_button("🚀 진화 시작", use_container_width=True)

    if submitted:
        if not task_description.strip():
            st.warning("작업 설명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('evolving_swarm'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"evolving_swarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="self_evolving_swarm_agent",
                agent_name="Self Evolving Swarm Agent",
                entry_point="srcs.advanced_agents.self_evolving_swarm",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["self_evolution", "multi_agent_collaboration", "adaptive_learning"],
                description="자기 진화형 multi-agent 시스템",
                input_params={
                    "task": task_description,
                    "evolution_steps": evolution_steps
                },
                result_json_path=result_json_path
            )

            # 결과 처리
            processed = process_standard_agent_result(result, "self_evolving_swarm_agent")
            if processed["success"] and processed["has_data"]:
                display_results(processed["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Self Evolving Swarm 결과")
    latest_result = result_reader.get_latest_result("evolving_swarm_agent", "evolution_execution")
    if latest_result:
        with st.expander("🧬 최신 진화 실행 결과", expanded=False):
            st.json(latest_result)
    else:
        st.info("💡 아직 Self Evolving Swarm Agent의 결과가 없습니다. 위에서 진화 작업을 실행해보세요.")

if __name__ == "__main__":
    main()

