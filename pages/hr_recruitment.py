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

# HR Recruitment Agent 임포트 시도
try:
    from srcs.enterprise_agents.hr_recruitment_agent import HRRecruitmentAgent
    HR_AGENT_AVAILABLE = True
except ImportError as e:
    HR_AGENT_AVAILABLE = False
    import_error = str(e)

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
    
    st.markdown("---")
    
    # Agent 연동 상태 확인
    if not HR_AGENT_AVAILABLE:
        st.error(f"⚠️ HR Recruitment Agent를 불러올 수 없습니다: {import_error}")
        st.info("에이전트 모듈을 확인하고 필요한 의존성을 설치해주세요.")
        
        with st.expander("🔧 설치 가이드"):
            st.markdown("""
            ### HR Recruitment Agent 설정
            
            1. **필요한 패키지 설치**:
            ```bash
            pip install openai transformers pandas nltk asyncio
            ```
            
            2. **환경 변수 설정**:
            ```bash
            export OPENAI_API_KEY="your-api-key"
            ```
            
            3. **에이전트 모듈 확인**:
            ```bash
            ls srcs/enterprise_agents/hr_recruitment_agent.py
            ```
            
            4. **MCP Agent 설정**:
            ```bash
            # MCP Agent 설정 파일 확인
            ls configs/mcp_agent.config.yaml
            ```
            """)
        
        # 에이전트 소개만 제공
        render_agent_info()
        return
    else:
        st.success("🤖 HR Recruitment Agent가 성공적으로 연결되었습니다!")
        
        # 에이전트 실행 인터페이스 제공
        render_hr_agent_interface()

def render_agent_info():
    """에이전트 기능 소개"""
    
    st.markdown("### 👥 HR Recruitment Agent 소개")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📋 주요 기능
        - **채용공고 생성**: AI 기반 맞춤형 채용공고 작성
        - **이력서 스크리닝**: 자동 이력서 평가 및 순위
        - **면접 질문 생성**: 기술/인성 면접 질문 자동 생성
        - **레퍼런스 체크**: 체계적인 경력 검증 프로세스
        - **오퍼레터 생성**: 법적 컴플라이언스 준수 채용 제안서
        - **온보딩 프로그램**: 신입사원 통합 교육 계획
        """)
    
    with col2:
        st.markdown("""
        #### ⚙️ 기술 특징
        - **MCP 프레임워크**: 다중 에이전트 시스템
        - **오케스트레이터**: 통합 워크플로우 관리
        - **품질 관리**: EvaluatorOptimizerLLM 적용
        - **파일 출력**: recruitment_reports/ 디렉토리
        - **비동기 처리**: asyncio 기반 실행
        """)

def render_hr_agent_interface():
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
            
            position_name = st.text_input(
                "채용 포지션", 
                value="Senior Software Engineer",
                help="채용하려는 직책명을 입력하세요"
            )
            
            company_name = st.text_input(
                "회사명", 
                value="TechCorp Inc.",
                help="회사명을 입력하세요"
            )
            
            workflow_scope = st.multiselect(
                "실행할 워크플로우",
                [
                    "채용공고 생성", 
                    "이력서 스크리닝 가이드", 
                    "면접 질문 세트", 
                    "레퍼런스 체크 프로세스",
                    "오퍼레터 템플릿",
                    "온보딩 프로그램"
                ],
                default=["채용공고 생성", "면접 질문 세트", "온보딩 프로그램"]
            )
            
            # 파일 저장 옵션
            save_to_file = st.checkbox(
                "파일로 저장", 
                value=False,
                help="체크하면 recruitment_reports/ 디렉토리에 파일로 저장합니다"
            )
            
            if st.button("🚀 HR Agent 실행", type="primary", use_container_width=True):
                if position_name and company_name and workflow_scope:
                    execute_hr_agent(agent, position_name, company_name, workflow_scope, save_to_file)
                else:
                    st.error("모든 필수 정보를 입력해주세요.")
        
        with col2:
            if 'hr_execution_result' in st.session_state:
                result = st.session_state['hr_execution_result']
                
                if result['success']:
                    st.success("✅ HR Recruitment Agent 실행 완료!")
                    
                    # 결과 정보 표시
                    st.markdown("#### 📊 실행 결과")
                    st.info(f"**메시지**: {result['message']}")
                    if result['save_to_file'] and result['output_dir']:
                        st.info(f"**출력 디렉토리**: {result['output_dir']}")
                    st.info(f"**실행된 워크플로우**: {', '.join(result['workflows_executed'])}")
                    
                    # 생성된 콘텐츠 표시
                    if 'content' in result and result['content']:
                        st.markdown("#### 📄 생성된 콘텐츠")
                        
                        # 콘텐츠를 보기 좋게 표시
                        content = result['content']
                        
                        # 텍스트가 너무 길면 확장 가능한 형태로 표시
                        if len(content) > 1000:
                            with st.expander("📋 전체 콘텐츠 보기", expanded=True):
                                st.markdown(content)
                        else:
                            st.markdown(content)
                        
                        # 콘텐츠 다운로드 버튼
                        st.download_button(
                            label="📥 콘텐츠 다운로드",
                            data=content,
                            file_name=f"hr_recruitment_result_{position_name.replace(' ', '_').lower()}.md",
                            mime="text/markdown"
                        )
                    
                    # 상세 결과 (디버그용)
                    with st.expander("🔍 상세 실행 정보"):
                        st.json({
                            'success': result['success'],
                            'message': result['message'],
                            'workflows_executed': result['workflows_executed'],
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
                - 💾 **파일 저장**: recruitment_reports/ 디렉토리에 저장
                """)
                
    except Exception as e:
        st.error(f"Agent 초기화 중 오류: {e}")
        st.info("에이전트 클래스를 확인해주세요.")

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