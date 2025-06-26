"""
🔒 Cybersecurity Agent Page

사이버 보안 인프라 관리 에이전트 연결
"""

import streamlit as st
import sys
from pathlib import Path
import json
import os
from datetime import datetime
import streamlit_process_manager as spm
from streamlit_process_manager.process import Process

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 설정 파일에서 경로 가져오기
try:
    from configs.settings import get_reports_path
    REPORTS_PATH = get_reports_path('cybersecurity')
except ImportError:
    st.error("❌ 설정 파일을 찾을 수 없습니다. configs/settings.py를 확인해주세요.")
    st.stop()

# ✅ P2-1: Import real implementations from Cybersecurity Agent
try:
    from srcs.enterprise_agents.cybersecurity_infrastructure_agent import (
        CybersecurityAgent,
        load_assessment_types,
        load_compliance_frameworks
    )
except ImportError as e:
    st.error(f"❌ Cybersecurity Infrastructure Agent를 불러올 수 없습니다: {e}")
    st.error("**시스템 요구사항**: CybersecurityAgent가 필수입니다.")
    st.info("에이전트 모듈을 설치하고 다시 시도해주세요.")
    st.stop()

# 페이지 설정
try:
    st.set_page_config(
        page_title="🔒 Cybersecurity Agent",
        page_icon="🔒",
        layout="wide"
    )
except Exception:
    pass

def main():
    """Cybersecurity Agent 메인 페이지"""
    
    # 헤더
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #ff4757, #ff3838);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    ">
        <h1>🔒 Cybersecurity Infrastructure Agent</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            AI 기반 사이버 보안 인프라 관리 및 위협 분석 시스템
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    st.markdown("---")
    
    st.success("🤖 Cybersecurity Infrastructure Agent가 성공적으로 연결되었습니다!")
    
    render_cybersecurity_agent_interface()

def render_cybersecurity_agent_interface():
    """Cybersecurity Agent 실행 인터페이스 (프로세스 모니터링)"""
    
    st.markdown("### 🚀 Cybersecurity Agent 실행")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        with st.form("cybersecurity_form"):
            st.markdown("#### 🎯 보안 평가 설정")
            
            company_name = st.text_input(
                "회사명", 
                placeholder="보안 평가를 수행할 회사명을 입력하세요",
                help="보안 평가를 수행할 회사명을 입력하세요"
            )
            
            assessment_types = load_assessment_types()
            assessment_type = st.selectbox(
                "평가 유형",
                assessment_types if assessment_types else ["전체 보안 평가"]
            )
            
            st.markdown("#### 📋 컴플라이언스 프레임워크")
            
            available_frameworks = load_compliance_frameworks()
            frameworks = st.multiselect(
                "적용할 프레임워크",
                available_frameworks if available_frameworks else ["ISO 27001"],
                help="적용할 컴플라이언스 프레임워크를 선택하세요"
            )
            
            save_to_file = st.checkbox(
                "파일로 저장", 
                value=False,
                help=f"체크하면 {REPORTS_PATH} 디렉토리에 파일로 저장합니다"
            )

            submitted = st.form_submit_button("🚀 Cybersecurity Agent 실행", type="primary", use_container_width=True)

            if submitted:
                if not company_name or not frameworks:
                    st.error("회사명을 입력하고 최소 하나의 프레임워크를 선택해주세요.")
                    return
                
                # 프로세스 실행 명령어 생성
                command = [
                    "python", "-u",
                    "srcs/enterprise_agents/run_cybersecurity_agent.py",
                    "--company-name", company_name,
                    "--assessment-type", assessment_type,
                    "--frameworks", json.dumps(frameworks),
                ]
                if save_to_file:
                    command.append("--save-to-file")

                # 결과 파일 경로 설정
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(REPORTS_PATH, f"cyber_agent_output_{timestamp}.log")
                os.makedirs(REPORTS_PATH, exist_ok=True)

                st.session_state['cybersecurity_command'] = command
                st.session_state['cybersecurity_output_file'] = output_file
    
    with col2:
        if 'cybersecurity_command' in st.session_state:
            st.info("🔄 Cybersecurity Agent 실행 중...")
            
            process = Process(
                st.session_state['cybersecurity_command'],
                output_file=st.session_state['cybersecurity_output_file']
            ).start()
            
            spm.st_process_monitor(
                process,
                label="사이버 보안 분석"
            ).loop_until_finished()
            
            st.success(f"✅ 분석 프로세스가 완료되었습니다. 전체 로그는 {st.session_state['cybersecurity_output_file']}에 저장됩니다.")
            
            # 실행 후 상태 초기화
            del st.session_state['cybersecurity_command']
            del st.session_state['cybersecurity_output_file']
        else:
            st.markdown("""
            #### 🤖 Cybersecurity Agent 정보
            
            **실행되는 프로세스:**
            1. **보안 취약점 평가** - 네트워크, 웹앱, 데이터베이스 보안 스캔
            2. **컴플라이언스 감사** - 선택된 프레임워크 기준 준수 여부 평가
            3. **사고 대응 계획** - 위협 인텔리전스 및 디지털 포렌식
            4. **인프라 보안 설계** - 제로 트러스트 및 네트워크 보안 아키텍처
            5. **클라우드 보안** - 멀티클라우드 거버넌스 및 컨테이너 보안
            6. **데이터 보호** - 암호화, DLP, 백업 및 재해 복구
            
            **생성되는 보안 결과:**
            - 🛡️ **보안 평가 보고서**: 취약점 분석 및 위험 점수
            - 📋 **컴플라이언스 감사**: 프레임워크별 준수 상태
            - 🚨 **사고 대응 계획**: 포괄적 대응 절차
            - 🏗️ **인프라 보안 아키텍처**: 제로 트러스트 설계
            - ☁️ **클라우드 보안 프레임워크**: 멀티클라우드 거버넌스
            - 🔐 **데이터 보호 프로그램**: 엔터프라이즈 데이터 보안 제어
            
            **출력 옵션:**
            - 🖥️ **화면 표시**: 즉시 결과 확인 (기본값)
            - 💾 **파일 저장**: {REPORTS_PATH} 디렉토리에 저장
            """.format(REPORTS_PATH=REPORTS_PATH))

if __name__ == "__main__":
    main() 