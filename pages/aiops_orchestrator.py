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
from srcs.common.ui_utils import run_agent_process
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
        
        submitted = st.form_submit_button("🚀 AIOps 작업 실행", use_container_width=True)

    if submitted:
        if not task_description.strip():
            st.warning("작업 설명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('aiops'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"aiops_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            py_executable = sys.executable
            command = [
                py_executable, "-m", "srcs.common.generic_agent_runner",
                "--module-path", "srcs.enterprise_agents.aiops_orchestrator_agent",
                "--class-name", "AIOpsOrchestratorAgent",
                "--method-name", "execute_task",
                "--config-json", json.dumps({"task": task_description}, ensure_ascii=False),
                "--result-json-path", str(result_json_path)
            ]

            result = run_agent_process(
                placeholder=result_placeholder,
                command=command,
                process_key_prefix="logs/aiops"
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

