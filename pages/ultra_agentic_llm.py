"""
🚀 Ultra Agentic LLM Agent Page

초 Agentic LLM Agent 시스템
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
        agent_name="Ultra Agentic LLM Agent",
        page_icon="🚀",
        page_type="ultra_agentic",
        title="Ultra Agentic LLM Agent",
        subtitle="LLM 중심의 초 Agentic 시스템 - 자율 의사결정, 계획, 학습",
        module_path="srcs.advanced_agents.ultra_agentic_llm_agent"
    )

    result_placeholder = st.empty()

    with st.form("ultra_agentic_form"):
        st.subheader("📝 Ultra Agentic 작업 설정")
        
        goal = st.text_area(
            "목표",
            placeholder="예: 복잡한 문제를 해결하기 위해 스스로 계획을 수립하고 실행",
            height=150
        )
        
        agent_id = st.text_input("Agent ID", value="ultra_agent_001")
        
        submitted = st.form_submit_button("🚀 Ultra Agentic 실행", use_container_width=True)

    if submitted:
        if not goal.strip():
            st.warning("목표를 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('ultra_agentic'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"ultra_agentic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            from srcs.common.standard_a2a_page_helper import (
                execute_standard_agent_via_a2a,
                process_standard_agent_result
            )
            from srcs.common.agent_interface import AgentType

            # 표준화된 방식으로 agent 실행 (클래스 기반)
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="ultra_agentic_llm_agent",
                agent_name="Ultra Agentic LLM Agent",
                entry_point="srcs.advanced_agents.ultra_agentic_llm_agent",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["autonomous_planning", "self_reflection", "goal_driven_execution", "multi_agent_collaboration"],
                description="LLM 중심의 초 Agentic 시스템 - 자율 의사결정, 계획, 학습",
                input_params={
                    "agent_id": agent_id,
                    "goal": goal
                },
                class_name="UltraAgenticLLMAgent",
                method_name="run",
                result_json_path=result_json_path
            )

            # 결과 처리
            processed = process_standard_agent_result(result, "ultra_agentic_llm_agent")
            if processed["success"] and processed["has_data"]:
                display_results(processed["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Ultra Agentic LLM 결과")
    latest_result = result_reader.get_latest_result("ultra_agentic_agent", "ultra_agentic_execution")
    if latest_result:
        with st.expander("🚀 최신 Ultra Agentic 실행 결과", expanded=False):
            display_results(latest_result)
    else:
        st.info("💡 아직 Ultra Agentic LLM Agent의 결과가 없습니다. 위에서 Ultra Agentic 작업을 실행해보세요.")

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 Ultra Agentic 실행 결과")
    if result_data:
        if isinstance(result_data, dict):
            if 'execution_result' in result_data:
                st.markdown("### ⚙️ 실행 결과")
                st.write(result_data['execution_result'])
            if 'plan' in result_data:
                st.markdown("### 📋 계획")
                st.write(result_data['plan'])
            if 'reflection' in result_data:
                st.markdown("### 🤔 반성 및 학습")
                st.write(result_data['reflection'])
            st.json(result_data)
        else:
            st.write(str(result_data))
    else:
        st.warning("결과 데이터가 없습니다.")

if __name__ == "__main__":
    main()

