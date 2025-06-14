"""
🔒 Cybersecurity Agent Page

사이버 보안 인프라 관리 에이전트 연결
"""

import streamlit as st
import sys
from pathlib import Path

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
    
    # 에이전트 인터페이스
    render_cybersecurity_agent_interface()

def render_cybersecurity_agent_interface():
    """Cybersecurity Agent 실행 인터페이스"""
    
    st.markdown("### 🚀 Cybersecurity Agent 실행")
    
    # 에이전트 초기화
    try:
        # 설정 입력
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 🎯 보안 평가 설정")
            
            company_name = st.text_input(
                "회사명", 
                placeholder="보안 평가를 수행할 회사명을 입력하세요",
                help="보안 평가를 수행할 회사명을 입력하세요"
            )
            
            # 동적으로 로드되어야 할 평가 유형들
            assessment_types = load_assessment_types()
            assessment_type = st.selectbox(
                "평가 유형",
                assessment_types if assessment_types else ["전체 보안 평가"]
            )
            
            st.markdown("#### 📋 컴플라이언스 프레임워크")
            
            # 동적으로 로드되어야 할 프레임워크들
            available_frameworks = load_compliance_frameworks()
            frameworks = st.multiselect(
                "적용할 프레임워크",
                available_frameworks if available_frameworks else ["ISO 27001"],
                help="적용할 컴플라이언스 프레임워크를 선택하세요"
            )
            
            # 파일 저장 옵션
            save_to_file = st.checkbox(
                "파일로 저장", 
                value=False,
                help=f"체크하면 {REPORTS_PATH} 디렉토리에 파일로 저장합니다"
            )
            
            if st.button("🚀 Cybersecurity Agent 실행", type="primary", use_container_width=True):
                if company_name and frameworks:
                    execute_cybersecurity_agent(company_name, assessment_type, frameworks, save_to_file)
                else:
                    st.error("회사명을 입력하고 최소 하나의 프레임워크를 선택해주세요.")
        
        with col2:
            if 'cybersecurity_execution_result' in st.session_state:
                result = st.session_state['cybersecurity_execution_result']
                
                if result['success']:
                    st.success("✅ Cybersecurity Agent 실행 완료!")
                    
                    # 실제 에이전트 결과 정보 표시
                    display_cybersecurity_results(result)
                    
                    # 생성된 콘텐츠 표시
                    if 'content' in result and result['content']:
                        st.markdown("#### 📄 생성된 보안 평가 결과")
                        
                        # 콘텐츠를 보기 좋게 표시
                        content = result['content']
                        
                        # 텍스트가 너무 길면 확장 가능한 형태로 표시
                        if len(content) > 2000:
                            with st.expander("📋 전체 보안 평가 결과 보기", expanded=True):
                                st.markdown(content)
                        else:
                            st.markdown(content)
                        
                        # 콘텐츠 다운로드 버튼
                        st.download_button(
                            label="📥 보안 평가 결과 다운로드",
                            data=content,
                            file_name=f"cybersecurity_assessment_{result['company_name'].replace(' ', '_').lower()}_{result['assessment_type'].replace(' ', '_')}.md",
                            mime="text/markdown"
                        )
                    
                    # 상세 결과 (디버그용)
                    with st.expander("🔍 상세 실행 정보"):
                        st.json({
                            'success': result['success'],
                            'message': result['message'],
                            'company_name': result['company_name'],
                            'assessment_type': result['assessment_type'],
                            'frameworks': result['frameworks'],
                            'save_to_file': result['save_to_file'],
                            'output_dir': result.get('output_dir'),
                            'content_length': len(result.get('content', '')) if result.get('content') else 0
                        })
                        
                else:
                    st.error("❌ 실행 중 오류 발생")
                    st.error(f"**오류**: {result['message']}")
                    
                    with st.expander("🔍 오류 상세"):
                        st.code(result.get('error', 'Unknown error'))
        
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
                
    except Exception as e:
        st.error(f"❌ Agent 초기화 실패: {e}")
        st.error("CybersecurityAgent 구현을 확인해주세요.")
        st.stop()

# ✅ P2-1: load_assessment_types and load_compliance_frameworks are now imported from srcs.enterprise_agents.cybersecurity_infrastructure_agent

def display_cybersecurity_results(result):
    """실제 사이버보안 에이전트 결과 표시"""
    
    st.markdown("#### 📊 실행 결과")
    
    # 기본 정보 표시
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**평가 대상**: {result['company_name']}")
        st.info(f"**평가 유형**: {result['assessment_type']}")
    
    with col2:
        st.info(f"**적용 프레임워크**: {', '.join(result['frameworks'])}")
        if result['save_to_file'] and result.get('output_dir'):
            st.info(f"**출력 디렉토리**: {result['output_dir']}")
    
    # 메시지 표시
    if result.get('message'):
        st.success(f"**결과**: {result['message']}")

def execute_cybersecurity_agent(company_name, assessment_type, frameworks, save_to_file):
    """Cybersecurity Agent 실행"""
    
    try:
        with st.spinner("🔄 Cybersecurity Agent를 실행하는 중..."):
            # 에이전트 초기화
            if 'cybersecurity_agent' not in st.session_state:
                st.session_state.cybersecurity_agent = CybersecurityAgent()
            
            agent = st.session_state.cybersecurity_agent
            
            # 실제 에이전트 실행 - 폴백 없음
            result = agent.run_cybersecurity_workflow(
                company_name=company_name,
                assessment_type=assessment_type,
                frameworks=frameworks,
                save_to_file=save_to_file
            )
            
            if not result:
                raise Exception("에이전트가 유효한 결과를 반환하지 않았습니다.")
            
            st.session_state['cybersecurity_execution_result'] = result
            st.rerun()
            
    except Exception as e:
        st.session_state['cybersecurity_execution_result'] = {
            'success': False,
            'message': f'Cybersecurity Agent 실행 실패: {str(e)}',
            'error': str(e),
            'company_name': company_name,
            'assessment_type': assessment_type,
            'frameworks': frameworks,
            'save_to_file': save_to_file
        }
        st.rerun()

if __name__ == "__main__":
    main() 