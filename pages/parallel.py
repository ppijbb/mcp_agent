"""
⚡ Parallel Agent Page

병렬 처리 Agent 시스템
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
    st.subheader("📊 병렬 실행 결과")
    
    if not result_data:
        st.warning("실행 결과를 찾을 수 없습니다.")
        return
    
    tasks = result_data.get('tasks', [])
    st.success(f"**총 작업 수: {len(tasks)}**")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("완료된 작업", result_data.get('completed_count', 0))
    col2.metric("실패한 작업", result_data.get('failed_count', 0))
    col3.metric("실행 시간", f"{result_data.get('execution_time', 0):.2f}초")
    
    if result_data.get('results'):
        st.subheader("📋 작업별 결과")
        for i, task_result in enumerate(result_data['results'], 1):
            with st.expander(f"작업 {i}: {task_result.get('task', 'N/A')}", expanded=False):
                st.write(f"**상태**: {'✅ 성공' if task_result.get('success') else '❌ 실패'}")
                if task_result.get('result'):
                    st.write(f"**결과**: {task_result['result']}")
                if task_result.get('error'):
                    st.error(f"**오류**: {task_result['error']}")

def main():
    from srcs.common.standard_a2a_page_helper import (
        execute_standard_agent_via_a2a,
        process_standard_agent_result
    )
    from configs.settings import get_reports_path
    from datetime import datetime
    
    # 페이지 기본 설정
    from srcs.common.page_utils import create_agent_page
    create_agent_page(
        agent_name="Parallel Agent",
        page_icon="⚡",
        page_type="parallel",
        title="Parallel Agent",
        subtitle="병렬 처리로 여러 작업을 동시에 실행",
        module_path="srcs.basic_agents.parallel"
    )
    
    result_placeholder = st.empty()
    
    with st.form("parallel_form"):
        st.subheader("📝 병렬 작업 설정")
        
        tasks_input = st.text_area(
            "병렬로 실행할 작업들 (한 줄에 하나씩)",
            placeholder="작업 1\n작업 2\n작업 3",
            height=150,
            help="각 줄에 하나의 작업을 입력하세요"
        )
        
        max_workers = st.slider(
            "최대 동시 실행 수",
            min_value=1,
            max_value=10,
            value=3,
            help="동시에 실행할 최대 작업 수"
        )
        
        submitted = st.form_submit_button("🚀 병렬 실행", use_container_width=True)
    
    if submitted:
        if not tasks_input.strip():
            st.warning("작업을 입력해주세요.")
        else:
            tasks = [task.strip() for task in tasks_input.split('\n') if task.strip()]
            
            reports_path = Path(get_reports_path('parallel'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"parallel_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="parallel_agent",
                agent_name="Parallel Agent",
                entry_point="srcs.basic_agents.parallel",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["parallel_execution", "task_distribution", "concurrent_processing"],
                description="병렬 처리로 여러 작업을 동시에 실행",
                input_params={
                    "tasks": tasks,
                    "max_workers": max_workers
                },
                result_json_path=result_json_path
            )
            
            # 결과 처리
            processed = process_standard_agent_result(result, "parallel_agent")
            if processed["success"] and processed["has_data"]:
                display_results(processed["data"])
    
    # 최신 결과 확인
    from srcs.utils.result_reader import result_reader
    st.markdown("---")
    st.markdown("## 📊 최신 Parallel 결과")
    
    latest_parallel_result = result_reader.get_latest_result("parallel_agent", "parallel_execution")
    
    if latest_parallel_result:
        with st.expander("⚡ 최신 병렬 실행 결과", expanded=False):
            st.subheader("🤖 최근 병렬 실행 결과")
            
            if isinstance(latest_parallel_result, dict):
                tasks = latest_parallel_result.get('tasks', [])
                st.success(f"**총 작업 수: {len(tasks)}**")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("완료된 작업", latest_parallel_result.get('completed_count', 0))
                col2.metric("실패한 작업", latest_parallel_result.get('failed_count', 0))
                col3.metric("실행 시간", f"{latest_parallel_result.get('execution_time', 0):.2f}초")
                
                if latest_parallel_result.get('results'):
                    st.subheader("📋 작업별 결과")
                    for i, task_result in enumerate(latest_parallel_result['results'], 1):
                        with st.expander(f"작업 {i}: {task_result.get('task', 'N/A')}", expanded=False):
                            st.write(f"**상태**: {'✅ 성공' if task_result.get('success') else '❌ 실패'}")
                            if task_result.get('result'):
                                st.write(f"**결과**: {task_result['result']}")
            else:
                st.write(latest_parallel_result)
    else:
        st.info("💡 아직 Parallel Agent의 결과가 없습니다. 위에서 병렬 작업을 실행해보세요.")

def display_results(result_data):
    """결과 표시"""
    st.markdown("---")
    st.subheader("📊 병렬 실행 결과")
    
    if not result_data:
        st.warning("실행 결과를 찾을 수 없습니다.")
        return
    
    tasks = result_data.get('tasks', [])
    st.success(f"**총 작업 수: {len(tasks)}**")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("완료된 작업", result_data.get('completed_count', 0))
    col2.metric("실패한 작업", result_data.get('failed_count', 0))
    col3.metric("실행 시간", f"{result_data.get('execution_time', 0):.2f}초")
    
    if result_data.get('results'):
        st.subheader("📋 작업별 결과")
        for i, task_result in enumerate(result_data['results'], 1):
            with st.expander(f"작업 {i}: {task_result.get('task', 'N/A')}", expanded=False):
                st.write(f"**상태**: {'✅ 성공' if task_result.get('success') else '❌ 실패'}")
                if task_result.get('result'):
                    st.write(f"**결과**: {task_result['result']}")
                if task_result.get('error'):
                    st.error(f"**오류**: {task_result['error']}")

if __name__ == "__main__":
    main()

