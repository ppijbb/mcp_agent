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


from srcs.common.page_utils import create_agent_page
from srcs.common.standard_a2a_page_helper import execute_standard_agent_via_a2a
from srcs.common.agent_interface import AgentType

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 중앙 설정 시스템 import
from configs.settings import get_reports_path

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

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
        width='stretch'
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
        
        submitted = st.form_submit_button("🚀 채용 프로세스 시작", width='stretch')

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

            # 입력 파라미터 준비
            input_data = {
                "position": position,
                "company": company,
                "workflows": workflows,
                "result_json_path": str(result_json_path)
            }

            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="hr_recruitment_agent",
                agent_name="HR Recruitment Agent",
                entry_point="srcs.common.generic_agent_runner",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["job_description", "resume_screening", "interview_questions", "reference_check"],
                description="인재 채용 및 관리 최적화",
                input_params=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

    # 최신 HR Recruitment Agent 결과 확인
    st.markdown("---")
    st.markdown("## 📊 최신 HR Recruitment Agent 결과")
    
    latest_recruitment_result = result_reader.get_latest_result("hr_recruitment_agent", "recruitment_analysis")
    
    if latest_recruitment_result:
        with st.expander("👥 최신 채용 분석 결과", expanded=False):
            st.subheader("🤖 최근 채용 분석 결과")
            
            if isinstance(latest_recruitment_result, dict):
                # 채용 정보 표시
                position = latest_recruitment_result.get('position', 'N/A')
                company = latest_recruitment_result.get('company', 'N/A')
                
                st.success(f"**포지션: {position}**")
                st.info(f"**회사: {company}**")
                
                # 채용 분석 결과 요약
                col1, col2, col3 = st.columns(3)
                col1.metric("실행된 워크플로우", len(latest_recruitment_result.get('workflows', [])))
                col2.metric("분석 상태", "완료" if latest_recruitment_result.get('success', False) else "실패")
                col3.metric("보고서 길이", f"{len(latest_recruitment_result.get('content', ''))} 문자")
                
                # 실행된 워크플로우 표시
                workflows = latest_recruitment_result.get('workflows', [])
                if workflows:
                    st.subheader("🔄 실행된 워크플로우")
                    for workflow in workflows:
                        st.write(f"• {workflow.replace('_', ' ').title()}")
                
                # 보고서 내용 표시
                content = latest_recruitment_result.get('content', '')
                if content:
                    st.subheader("📄 채용 분석 보고서")
                    with st.expander("보고서 내용", expanded=False):
                        st.markdown(content)
                    
                    # 다운로드 버튼
                    st.download_button(
                        label="📥 채용 보고서 다운로드 (.md)",
                        data=content,
                        file_name=f"recruitment_report_{position.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        width='stretch'
                    )
                
                # 메타데이터 표시
                if 'timestamp' in latest_recruitment_result:
                    st.caption(f"⏰ 분석 시간: {latest_recruitment_result['timestamp']}")
            else:
                st.write("결과 데이터 형식이 예상과 다릅니다.")
    else:
        st.info("💡 아직 HR Recruitment Agent의 결과가 없습니다. 위에서 채용 분석을 실행해보세요.")

if __name__ == "__main__":
    main() 