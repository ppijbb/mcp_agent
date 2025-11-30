"""
🐝 Swarm Agent Page

Multi-agent 협업 시스템
표준 A2A 패턴 적용
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.standard_a2a_page_template import create_standard_a2a_page
from srcs.common.agent_interface import AgentType

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

def main():
    # 표준화된 A2A Page 생성
    create_standard_a2a_page(
        agent_id="swarm_agent",
        agent_name="Swarm Agent",
        page_icon="🐝",
        page_type="swarm",
        title="Swarm Agent",
        subtitle="Multi-agent 협업을 통한 복잡한 작업 처리",
        entry_point="srcs.basic_agents.swarm",
        agent_type=AgentType.MCP_AGENT,
        capabilities=["multi_agent_collaboration", "task_decomposition", "parallel_execution"],
        description="Multi-agent 협업을 통한 복잡한 작업 처리",
        form_fields=[
            {
                "type": "text_area",
                "key": "task",
                "label": "작업 설명",
                "default": "",
                "height": 150,
                "help": "Swarm agent들이 협업하여 처리할 작업을 설명하세요",
                "required": True
            },
            {
                "type": "slider",
                "key": "agent_count",
                "label": "사용할 Agent 수",
                "min_value": 2,
                "max_value": 10,
                "default": 5,
                "help": "협업할 agent의 수"
            }
        ],
        display_results_func=display_results,
        result_category="swarm_execution"
    )

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

