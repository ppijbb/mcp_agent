"""
👥 HR Recruitment Agent Page

인재 채용 및 관리 최적화 AI
"""

import streamlit as st
import sys
from pathlib import Path

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

def load_workflow_options():
    """워크플로우 옵션 동적 로딩"""
    # TODO: 실제 시스템에서 지원하는 워크플로우 로드
    return [
        "채용공고 생성", 
        "이력서 스크리닝 가이드", 
        "면접 질문 세트", 
        "레퍼런스 체크 프로세스",
        "오퍼레터 템플릿",
        "온보딩 프로그램"
    ]

def load_default_workflows():
    """기본 워크플로우 동적 로딩"""
    # TODO: 실제 사용자 설정에서 기본 워크플로우 로드
    return []

def get_user_company_info():
    """사용자 회사 정보 조회"""
    # TODO: 실제 사용자 프로필에서 회사 정보 로드
    return {
        "company_name": None,
        "default_positions": []
    }

def main():
    """HR Recruitment Agent 메인 페이지"""
    
    # 헤더
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    ">
        <h1>👥 HR Recruitment Agent</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            AI 기반 인재 채용 및 관리 최적화 시스템
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    # 파일 저장 옵션 추가
    save_to_file = st.checkbox(
        "채용 결과를 파일로 저장", 
        value=False,
        help=f"체크하면 {get_reports_path('hr_recruitment')} 디렉토리에 결과를 파일로 저장합니다"
    )
    
    st.markdown("---")
    
    st.success("🤖 HR Recruitment Agent가 성공적으로 연결되었습니다!")
        
    # 에이전트 실행 인터페이스 제공
    render_hr_agent_interface(save_to_file)

def render_hr_agent_interface(save_to_file=False):
    """HR Agent 실행 인터페이스"""
    
    st.markdown("### 🚀 HR Recruitment Agent 실행")
    
    # 에이전트 초기화
    try:
        if 'hr_agent' not in st.session_state:
            st.session_state.hr_agent = HRRecruitmentAgent()
        
        agent = st.session_state.hr_agent
        
        # 실행 설정
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### ⚙️ 채용 프로젝트 설정")
            
            # 사용자 회사 정보 로딩
            company_info = get_user_company_info()
            
            position_name = st.text_input(
                "채용 포지션", 
                value=None,
                placeholder="채용하려는 직책명을 입력하세요",
                help="채용하려는 직책명을 입력하세요"
            )
            
            company_name = st.text_input(
                "회사명", 
                value=company_info.get("company_name"),
                placeholder="회사명을 입력하세요",
                help="회사명을 입력하세요"
            )
            
            # 워크플로우 옵션 동적 로딩
            workflow_options = load_workflow_options()
            default_workflows = load_default_workflows()
            
            workflow_scope = st.multiselect(
                "실행할 워크플로우",
                workflow_options,
                default=default_workflows,
                help="실행할 채용 워크플로우를 선택하세요"
            )
            
            # 필수 입력값 검증
            if all([position_name, company_name, workflow_scope]):
                if st.button("🚀 HR Agent 실행", type="primary", use_container_width=True):
                    execute_hr_agent(agent, position_name, company_name, workflow_scope, save_to_file)
            else:
                st.warning("모든 필수 정보를 입력해주세요.")
                if st.button("🚀 HR Agent 실행", type="primary", use_container_width=True, disabled=True):
                    pass
        
        with col2:
            if 'hr_execution_result' in st.session_state:
                result = st.session_state['hr_execution_result']
                
                if result.get('success', False):
                    st.success("✅ HR Recruitment Agent 실행 완료!")
                    
                    # 결과 검증
                    if not result:
                        st.error("HR Agent 실행 결과를 받을 수 없습니다.")
                        return
                    
                    # 결과 정보 표시
                    display_hr_results(result, position_name if 'position_name' in locals() else 'unknown')
                        
                else:
                    st.error("❌ 실행 중 오류 발생")
                    st.error(f"**오류**: {result.get('message', 'Unknown error')}")
                    
                    with st.expander("🔍 오류 상세"):
                        st.code(result.get('error', 'Unknown error'))
                        
            else:
                st.markdown("""
                #### 🤖 Agent 실행 정보
                
                **실행되는 프로세스:**
                1. **MCP App 초기화** - MCP 프레임워크 연결
                2. **다중 에이전트 생성** - 채용 전문 AI 에이전트들
                3. **오케스트레이터 실행** - 통합 워크플로우 관리
                4. **콘텐츠 생성** - 실시간 결과 표시 또는 파일 저장
                
                **생성되는 콘텐츠:**
                - 📝 채용공고 (Job Description)
                - 📋 이력서 스크리닝 가이드
                - ❓ 면접 질문 세트
                - 📞 레퍼런스 체크 프로세스
                - 📄 오퍼레터 템플릿
                - 🎯 온보딩 프로그램
                
                **출력 옵션:**
                - 🖥️ **화면 표시**: 즉시 결과 확인 (기본값)
                - 💾 **파일 저장**: 동적 경로에 저장
                """)
                
    except Exception as e:
        st.error(f"Agent 초기화 중 오류: {e}")
        st.info("에이전트 클래스를 확인해주세요.")

def display_hr_results(result, position_name):
    """HR 실행 결과 표시"""
    
    st.markdown("#### 📊 실행 결과")
    
    # 기본 정보 표시
    if 'message' in result:
        st.info(f"**메시지**: {result['message']}")
    
    if result.get('save_to_file') and result.get('output_dir'):
        st.info(f"**출력 디렉토리**: {result['output_dir']}")
    
    if 'workflows_executed' in result:
        st.info(f"**실행된 워크플로우**: {', '.join(result['workflows_executed'])}")
    
    # 생성된 콘텐츠 표시
    if 'content' in result and result['content']:
        st.markdown("#### 📄 생성된 콘텐츠")
        
        content = result['content']
        
        # 콘텐츠 길이에 따른 표시 방식
        if len(content) > 1000:
            with st.expander("📋 전체 콘텐츠 보기", expanded=True):
                st.markdown(content)
        else:
            st.markdown(content)
        
        # 콘텐츠 다운로드 버튼
        safe_filename = position_name.replace(' ', '_').lower() if position_name else 'hr_result'
        st.download_button(
            label="📥 콘텐츠 다운로드",
            data=content,
            file_name=f"hr_recruitment_result_{safe_filename}.md",
            mime="text/markdown"
        )
    
    # 상세 결과 (디버그용)
    with st.expander("🔍 상세 실행 정보"):
        debug_info = {
            'success': result.get('success', False),
            'message': result.get('message', 'N/A'),
            'workflows_executed': result.get('workflows_executed', []),
            'save_to_file': result.get('save_to_file', False),
            'output_dir': result.get('output_dir', 'N/A'),
            'content_length': len(result.get('content', '')) if result.get('content') else 0
        }
        st.json(debug_info)

def execute_hr_agent(agent, position, company, workflows, save_to_file):
    """HR Agent 실행"""
    
    try:
        with st.spinner("🔄 HR Recruitment Agent를 실행하는 중..."):
            # 에이전트 실행
            result = agent.run_recruitment_workflow(
                position=position,
                company=company,
                workflows=workflows,
                save_to_file=save_to_file
            )
            
            # 결과 검증
            if not result:
                st.session_state['hr_execution_result'] = {
                    'success': False,
                    'message': 'HR Agent가 결과를 반환하지 않았습니다.',
                    'error': 'Empty result from agent',
                    'save_to_file': save_to_file
                }
            else:
                st.session_state['hr_execution_result'] = result
            
            st.rerun()
            
    except Exception as e:
        st.session_state['hr_execution_result'] = {
            'success': False,
            'message': f'Agent 실행 중 오류 발생: {str(e)}',
            'error': str(e),
            'save_to_file': save_to_file
        }
        st.rerun()

if __name__ == "__main__":
    main() 