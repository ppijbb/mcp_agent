"""
🤖 AIOps Orchestrator Agent Page

AI 기반 IT 운영 자동화
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
        agent_name="AIOps Orchestrator Agent",
        page_icon="🤖",
        page_type="aiops",
        title="AIOps Orchestrator Agent",
        subtitle="AI 기반 IT 운영 자동화 및 모니터링",
        module_path="srcs.enterprise_agents.aiops_orchestrator_agent"
    )

    result_placeholder = st.empty()

    with st.form("aiops_form"):
        st.subheader("📝 AIOps 작업 설정")
        
        task_description = st.text_area(
            "작업 설명",
            placeholder="예: 프로덕션 서버의 성능 모니터링 및 최적화",
            height=100
        )
        
        # 시뮬레이션 모드 토글
        simulation_mode = st.checkbox(
            "시뮬레이션 모드 활성화",
            value=True,
            help="시뮬레이션 모드가 활성화되면 인프라 메트릭 시뮬레이터를 사용하여 시스템 메트릭을 생성합니다."
        )
        
        if simulation_mode:
            st.info("🔬 시뮬레이션 모드: 인프라 메트릭 시뮬레이터를 사용합니다.")
        
        submitted = st.form_submit_button("🚀 AIOps 작업 실행", width='stretch')

    if submitted:
        if not task_description.strip():
            st.warning("작업 설명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('aiops'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"aiops_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            agent_metadata = {
                "agent_id": "aiops_orchestrator_agent",
                "agent_name": "AIOps Orchestrator Agent",
                "entry_point": "srcs.common.generic_agent_runner",
                "agent_type": "mcp_agent",
                "capabilities": ["it_operations", "performance_monitoring", "automation", "infrastructure_management"],
                "description": "AI 기반 IT 운영 자동화 및 모니터링"
            }

            input_data = {
                "module_path": "srcs.enterprise_agents.aiops_orchestrator_agent",
                "class_name": "AIOpsOrchestratorAgent",
                "method_name": "execute_task",
                "config": {"task": task_description, "simulation_mode": simulation_mode},
                "result_json_path": str(result_json_path)
            }

            result = run_agent_via_a2a(
                placeholder=result_placeholder,
                agent_metadata=agent_metadata,
                input_data=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 AIOps 결과")
    latest_result = result_reader.get_latest_result("aiops_orchestrator_agent", "aiops_task")
    if latest_result:
        with st.expander("🤖 최신 AIOps 작업 결과", expanded=False):
            st.json(latest_result)

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 AIOps 작업 결과")
    if result_data:
        st.json(result_data)

if __name__ == "__main__":
    main()

