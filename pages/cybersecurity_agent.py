import streamlit as st
from pathlib import Path
import sys
import json
from datetime import datetime
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.page_utils import create_agent_page
from srcs.common.ui_utils import run_agent_process
from srcs.core.config.loader import settings
from srcs.enterprise_agents.cybersecurity_infrastructure_agent import (
    CybersecurityAgent,
    load_assessment_types,
    load_compliance_frameworks
)

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 분석 결과")

    if not result_data:
        st.warning("분석 결과를 찾을 수 없습니다.")
        return

    st.info(f"**회사명**: {result_data.get('company_name', 'N/A')}")
    st.info(f"**평가 유형**: {result_data.get('assessment_type', 'N/A')}")
    
    content = result_data.get('content', '')
    with st.expander("상세 보고서 보기", expanded=True):
        st.markdown(content)
    
    st.download_button(
        label="📥 보고서 다운로드 (.md)",
        data=content,
        file_name=f"cybersecurity_report_{result_data.get('assessment_type', 'report')}.md",
        mime="text/markdown",
        use_container_width=True
    )


def main():
    create_agent_page(
        agent_name="Cybersecurity Agent",
        page_icon="🛡️",
        page_type="cybersecurity",
        title="Cybersecurity Agent",
        subtitle="사이버 위협으로부터 조직을 보호하기 위한 AI 기반 보안 솔루션",
        module_path="srcs.enterprise_agents.cybersecurity_infrastructure_agent"
    )

    result_placeholder = st.empty()

    with st.form("cybersecurity_form"):
        st.subheader("📝 보안 평가 설정")
        
        company_name = st.text_input("회사명", value="TechCorp Inc.")
        assessment_type = st.selectbox("평가 유형 선택", options=load_assessment_types())
        frameworks = st.multiselect(
            "컴플라이언스 프레임워크 선택",
            options=load_compliance_frameworks(),
            default=["ISO 27001 (Information Security Management)", "GDPR (General Data Protection Regulation)"]
        )
        
        submitted = st.form_submit_button("🚀 보안 평가 시작", use_container_width=True)

    if submitted:
        if not company_name.strip():
            st.warning("회사명을 입력해주세요.")
        else:
            reports_path = settings.get_reports_path('cybersecurity')
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"cybersecurity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            config = {
                'company_name': company_name,
                'assessment_type': assessment_type,
                'frameworks': frameworks,
                'save_to_file': False # UI 모드에서는 파일 저장을 비활성화
            }

            py_executable = sys.executable
            command = [
                py_executable, "-m", "srcs.common.generic_agent_runner",
                "--module-path", "srcs.enterprise_agents.cybersecurity_infrastructure_agent",
                "--class-name", "CybersecurityAgent",
                "--method-name", "run_cybersecurity_workflow",
                "--config-json", json.dumps(config, ensure_ascii=False),
                "--result-json-path", str(result_json_path)
            ]

            result = run_agent_process(
                placeholder=result_placeholder, 
                command=command, 
                process_key_prefix="logs/cybersecurity"
            )

            if result and "data" in result:
                display_results(result["data"])

    # 최신 Cybersecurity Agent 결과 확인
    st.markdown("---")
    st.markdown("## 📊 최신 Cybersecurity Agent 결과")
    
    latest_cybersecurity_result = result_reader.get_latest_result("cybersecurity_agent", "security_assessment")
    
    if latest_cybersecurity_result:
        with st.expander("🛡️ 최신 보안 평가 결과", expanded=False):
            st.subheader("🔒 최근 보안 평가 결과")
            
            if isinstance(latest_cybersecurity_result, dict):
                # 회사 정보 표시
                company_name = latest_cybersecurity_result.get('company_name', 'N/A')
                assessment_type = latest_cybersecurity_result.get('assessment_type', 'N/A')
                
                st.success(f"**회사: {company_name}**")
                st.info(f"**평가 유형: {assessment_type}**")
                
                # 평가 결과 요약
                col1, col2, col3 = st.columns(3)
                col1.metric("평가 상태", "완료" if latest_cybersecurity_result.get('success', False) else "실패")
                col2.metric("컴플라이언스 프레임워크", len(latest_cybersecurity_result.get('frameworks', [])))
                col3.metric("보고서 길이", f"{len(latest_cybersecurity_result.get('content', ''))} 문자")
                
                # 보고서 내용 표시
                content = latest_cybersecurity_result.get('content', '')
                if content:
                    st.subheader("📋 보안 평가 보고서")
                    with st.expander("상세 보고서", expanded=False):
                        st.markdown(content)
                    
                    # 다운로드 버튼
                    st.download_button(
                        label="📥 보고서 다운로드 (.md)",
                        data=content,
                        file_name=f"cybersecurity_report_{assessment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                # 메타데이터 표시
                if 'timestamp' in latest_cybersecurity_result:
                    st.caption(f"⏰ 평가 시간: {latest_cybersecurity_result['timestamp']}")
                
                # 컴플라이언스 프레임워크 표시
                frameworks = latest_cybersecurity_result.get('frameworks', [])
                if frameworks:
                    st.subheader("📋 적용된 컴플라이언스 프레임워크")
                    for framework in frameworks:
                        st.write(f"• {framework}")
            else:
                st.json(latest_cybersecurity_result)
    else:
        st.info("💡 아직 Cybersecurity Agent의 결과가 없습니다. 위에서 보안 평가를 실행해보세요.")

if __name__ == "__main__":
    main() 