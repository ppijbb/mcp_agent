"""
🚀 DevOps Assistant Agent Page

개발자 생산성 자동화 AI
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
from srcs.common.ui_utils import run_agent_process
from configs.settings import get_reports_path

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def main():
    create_agent_page(
        agent_name="DevOps Assistant Agent",
        page_icon="🚀",
        page_type="devops",
        title="DevOps Assistant Agent",
        subtitle="GitHub 코드 리뷰, CI/CD 모니터링, 이슈 분석 등 개발자 생산성 자동화",
        module_path="srcs.enterprise_agents.devops_assistant_agent"
    )

    result_placeholder = st.empty()

    with st.form("devops_form"):
        st.subheader("📝 DevOps 작업 선택")
        
        task_type = st.selectbox(
            "작업 유형",
            options=[
                "code_review",
                "deployment_check",
                "issue_analysis",
                "team_standup",
                "performance_analysis"
            ],
            format_func=lambda x: {
                "code_review": "🔍 코드 리뷰",
                "deployment_check": "🚀 배포 상태 확인",
                "issue_analysis": "🎯 이슈 분석",
                "team_standup": "👥 팀 스탠드업",
                "performance_analysis": "📊 성능 분석"
            }.get(x, x)
        )
        
        col1, col2 = st.columns(2)
        with col1:
            owner = st.text_input("GitHub 소유자", value="microsoft")
        with col2:
            repo = st.text_input("저장소 이름", value="vscode")
        
        if task_type == "code_review":
            pull_number = st.number_input("PR 번호", min_value=1, value=42)
        
        submitted = st.form_submit_button("🚀 DevOps 작업 실행", use_container_width=True)

    if submitted:
        reports_path = Path(get_reports_path('devops_assistant'))
        reports_path.mkdir(parents=True, exist_ok=True)
        result_json_path = reports_path / f"devops_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        config = {
            "task_type": task_type,
            "owner": owner,
            "repo": repo,
        }
        if task_type == "code_review":
            config["pull_number"] = int(pull_number)

        py_executable = sys.executable
        command = [
            py_executable, "-m", "srcs.common.generic_agent_runner",
            "--module-path", "srcs.enterprise_agents.devops_assistant_agent",
            "--class-name", "DevOpsAssistantMCPAgent",
            "--method-name", f"run_{task_type}",
            "--config-json", json.dumps(config, ensure_ascii=False),
            "--result-json-path", str(result_json_path)
        ]

        result = run_agent_process(
            placeholder=result_placeholder,
            command=command,
            process_key_prefix="logs/devops_assistant"
        )

        if result and "data" in result:
            display_results(result["data"])

    # 최신 DevOps Assistant 결과 확인
    st.markdown("---")
    st.markdown("## 📊 최신 DevOps Assistant 결과")
    
    latest_devops_result = result_reader.get_latest_result("devops_assistant_agent", "devops_task")
    
    if latest_devops_result:
        with st.expander("🚀 최신 DevOps 작업 결과", expanded=False):
            st.subheader("🤖 최근 DevOps 작업 결과")
            
            if isinstance(latest_devops_result, dict):
                task_type = latest_devops_result.get('task_type', 'N/A')
                st.success(f"**작업 유형: {task_type}**")
                
                col1, col2 = st.columns(2)
                col1.metric("상태", latest_devops_result.get('status', 'N/A'))
                col2.metric("처리 시간", f"{latest_devops_result.get('processing_time', 0):.2f}초")
                
                if latest_devops_result.get('result_data'):
                    st.subheader("📋 작업 결과")
                    st.json(latest_devops_result['result_data'])
                
                if latest_devops_result.get('recommendations'):
                    st.subheader("💡 권장사항")
                    for rec in latest_devops_result['recommendations']:
                        st.write(f"• {rec}")
            else:
                st.json(latest_devops_result)
    else:
        st.info("💡 아직 DevOps Assistant Agent의 결과가 없습니다. 위에서 DevOps 작업을 실행해보세요.")

def display_results(result_data):
    """결과 표시"""
    st.markdown("---")
    st.subheader("📊 DevOps 작업 결과")
    
    if not result_data:
        st.warning("작업 결과를 찾을 수 없습니다.")
        return
    
    task_type = result_data.get('task_type', 'N/A')
    st.success(f"**작업 유형: {task_type}**")
    
    col1, col2 = st.columns(2)
    col1.metric("상태", result_data.get('status', 'N/A'))
    col2.metric("처리 시간", f"{result_data.get('processing_time', 0):.2f}초")
    
    if result_data.get('result_data'):
        st.subheader("📋 작업 결과")
        st.json(result_data['result_data'])
    
    if result_data.get('recommendations'):
        st.subheader("💡 권장사항")
        for rec in result_data['recommendations']:
            st.write(f"• {rec}")

if __name__ == "__main__":
    main()

