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
        
        submitted = st.form_submit_button("🚀 AIOps 작업 실행", use_container_width=True)

    if submitted:
        if not task_description.strip():
            st.warning("작업 설명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('aiops'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"aiops_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 입력 파라미터 준비
            input_data = {
                "task_description": task_description,
                "simulation_mode": simulation_mode,
                "result_json_path": str(result_json_path)
            }

            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="aiops_orchestrator_agent",
                agent_name="AIOps Orchestrator Agent",
                entry_point="srcs.common.generic_agent_runner",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["it_operations", "performance_monitoring", "automation", "infrastructure_management"],
                description="AI 기반 IT 운영 자동화 및 모니터링",
                input_params=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 AIOps 결과")
    latest_result = result_reader.get_latest_result("aiops_orchestrator_agent", "aiops_task")
    if latest_result:
        display_results(latest_result)

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 AIOps 작업 결과")
    if result_data:
        # JSON이 아닌 실제 결과 내용 표시
        if isinstance(result_data, dict):
            # result 필드가 있으면 그것을 표시
            if "result" in result_data:
                result_text = result_data["result"]
                if isinstance(result_text, str):
                    st.markdown(result_text)
                else:
                    st.write(result_text)
            # success 필드 표시
            if "success" in result_data:
                if result_data["success"]:
                    st.success("✅ 작업이 성공적으로 완료되었습니다.")
                else:
                    st.error(f"❌ 작업 실패: {result_data.get('error', '알 수 없는 오류')}")
            # alert 정보 표시
            if "alert_id" in result_data:
                st.info(f"**Alert ID**: {result_data.get('alert_id', 'N/A')} | **Node**: {result_data.get('node', 'N/A')}")
        elif isinstance(result_data, str):
            st.markdown(result_data)
        else:
            st.write(result_data)

if __name__ == "__main__":
    main()

