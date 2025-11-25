"""
🐝 Swarm Agent Page

Multi-agent 협업 시스템
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.page_utils import create_agent_page
from srcs.common.streamlit_a2a_runner import run_agent_via_a2a
from configs.settings import get_reports_path

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def main():
    create_agent_page(
        agent_name="Swarm Agent",
        page_icon="🐝",
        page_type="swarm",
        title="Swarm Agent",
        subtitle="Multi-agent 협업을 통한 복잡한 작업 처리",
        module_path="srcs.basic_agents.swarm"
    )

    result_placeholder = st.empty()

    with st.form("swarm_form"):
        st.subheader("📝 Swarm 작업 설정")
        
        task_description = st.text_area(
            "작업 설명",
            placeholder="예: 고객 지원 케이스 처리 - 항공편 변경 요청",
            height=150,
            help="Swarm agent들이 협업하여 처리할 작업을 설명하세요"
        )
        
        agent_count = st.slider(
            "사용할 Agent 수",
            min_value=2,
            max_value=10,
            value=5,
            help="협업할 agent의 수"
        )
        
        submitted = st.form_submit_button("🚀 Swarm 실행", use_container_width=True)

    if submitted:
        if not task_description.strip():
            st.warning("작업 설명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('swarm'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"swarm_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            agent_metadata = {
                "agent_id": "swarm_agent",
                "agent_name": "Swarm Agent",
                "entry_point": "srcs.basic_agents.swarm",
                "agent_type": "mcp_agent",
                "capabilities": ["multi_agent_collaboration", "task_decomposition", "parallel_execution"],
                "description": "Multi-agent 협업을 통한 복잡한 작업 처리"
            }

            input_data = {
                "task": task_description,
                "agent_count": agent_count,
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

    # 최신 Swarm 결과 확인
    st.markdown("---")
    st.markdown("## 📊 최신 Swarm 결과")
    
    latest_swarm_result = result_reader.get_latest_result("swarm_agent", "swarm_execution")
    
    if latest_swarm_result:
        with st.expander("🐝 최신 Swarm 실행 결과", expanded=False):
            st.subheader("🤖 최근 Swarm 실행 결과")
            
            if isinstance(latest_swarm_result, dict):
                task = latest_swarm_result.get('task', 'N/A')
                st.success(f"**작업: {task}**")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Agent 수", latest_swarm_result.get('agent_count', 0))
                col2.metric("완료된 단계", latest_swarm_result.get('completed_steps', 0))
                col3.metric("상태", "완료" if latest_swarm_result.get('success', False) else "실패")
                
                if latest_swarm_result.get('result'):
                    st.subheader("📋 실행 결과")
                    st.write(latest_swarm_result['result'])
            else:
                st.json(latest_swarm_result)
    else:
        st.info("💡 아직 Swarm Agent의 결과가 없습니다. 위에서 Swarm 작업을 실행해보세요.")

def display_results(result_data):
    """결과 표시"""
    st.markdown("---")
    st.subheader("📊 Swarm 실행 결과")
    
    if not result_data:
        st.warning("실행 결과를 찾을 수 없습니다.")
        return
    
    st.success(f"**작업**: {result_data.get('task', 'N/A')}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Agent 수", result_data.get('agent_count', 0))
    col2.metric("완료된 단계", result_data.get('completed_steps', 0))
    col3.metric("상태", "완료" if result_data.get('success', False) else "실패")
    
    if result_data.get('result'):
        st.subheader("📋 실행 결과")
        st.write(result_data['result'])
    
    if result_data.get('agent_logs'):
        st.subheader("🐝 Agent 로그")
        with st.expander("상세 로그", expanded=False):
            for log in result_data['agent_logs']:
                st.write(f"• {log}")

if __name__ == "__main__":
    main()

