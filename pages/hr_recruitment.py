"""
👥 HR Recruitment Agent Page

인재 채용 및 관리 최적화 AI
"""

import streamlit as st
import sys
from pathlib import Path
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import streamlit_process_manager as spm


from srcs.common.page_utils import create_agent_page
from srcs.common.ui_utils import run_agent_process
from configs.settings import get_reports_path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 중앙 설정 시스템 import
from configs.settings import get_reports_path

# HR Recruitment Agent 임포트 시도
try:
    from srcs.enterprise_agents.hr_recruitment_agent import HRRecruitmentAgent
except ImportError as e:
    st.error(f"HR Recruitment Agent를 사용하려면 필요한 의존성을 설치해야 합니다: {e}")
    st.error("시스템 관리자에게 문의하여 HR Recruitment Agent 모듈을 설정하세요.")
    st.stop()

def get_workflow_options():
    """사용 가능한 워크플로우 목록을 반환"""
    return [
        "job_description",
        "resume_screening",
        "interview_questions",
        "reference_check"
    ]

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 채용 분석 결과")

    if not result_data:
        st.warning("분석 결과를 찾을 수 없습니다.")
        return

    content = result_data.get('content', '')
    with st.expander("상세 보고서 보기", expanded=True):
        st.markdown(content)
    
    st.download_button(
        label="📥 보고서 다운로드 (.md)",
        data=content,
        file_name=f"recruitment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True
    )

def main():
    create_agent_page(
        agent_name="HR Recruitment Agent",
        page_icon="👥",
        page_type="hr",
        title="HR Recruitment Agent",
        subtitle="AI 기반 채용 프로세스 자동화 및 최적화 솔루션",
        module_path="srcs.enterprise_agents.hr_recruitment_agent"
    )
    result_placeholder = st.empty()

    with st.form("recruitment_form"):
        st.subheader("📝 채용 설정")
        
        position = st.text_input("채용 포지션", value="Senior Software Engineer")
        company = st.text_input("회사명", value="TechCorp Inc.")
        
        workflows = st.multiselect(
            "실행할 워크플로우 선택",
            options=get_workflow_options(),
            default=get_workflow_options() # 기본으로 모두 선택
        )
        
        submitted = st.form_submit_button("🚀 채용 프로세스 시작", use_container_width=True)

    if submitted:
        if not position.strip() or not company.strip():
            st.warning("포지션과 회사명을 모두 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('recruitment'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"recruitment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            config = {
                'position': position,
                'company': company,
                'workflows': workflows,
                'save_to_file': False # UI 모드에서는 파일 저장을 비활성화
            }

            py_executable = sys.executable
            command = [
                py_executable, "-m", "srcs.common.generic_agent_runner",
                "--module-path", "srcs.enterprise_agents.hr_recruitment_agent",
                "--class-name", "HRRecruitmentAgent",
                "--method-name", "run_recruitment_workflow",
                "--config-json", json.dumps(config, ensure_ascii=False),
                "--result-json-path", str(result_json_path)
            ]

            result = run_agent_process(
                placeholder=result_placeholder, 
                command=command, 
                process_key_prefix="hr_recruitment"
            )

            if result and "data" in result:
                display_results(result["data"])

if __name__ == "__main__":
    main() 