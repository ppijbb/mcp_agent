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

# Cybersecurity Infrastructure Agent 임포트
try:
    from srcs.enterprise_agents.cybersecurity_infrastructure_agent import CybersecurityAgent
    CYBERSECURITY_AGENT_AVAILABLE = True
except ImportError as e:
    CYBERSECURITY_AGENT_AVAILABLE = False
    import_error = str(e)

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
    
    # Agent 연동 상태 확인
    if not CYBERSECURITY_AGENT_AVAILABLE:
        st.error(f"⚠️ Cybersecurity Infrastructure Agent를 불러올 수 없습니다: {import_error}")
        st.info("에이전트 모듈을 확인하고 필요한 의존성을 설치해주세요.")
        
        with st.expander("🔧 설치 가이드"):
            st.markdown("""
            ### Cybersecurity Infrastructure Agent 설정
            
            1. **필요한 패키지 설치**:
            ```bash
            pip install openai asyncio
            ```
            
            2. **환경 변수 설정**:
            ```bash
            export OPENAI_API_KEY="your-api-key"
            ```
            
            3. **MCP Agent 설정**:
            ```bash
            # MCP Agent 설정 파일 확인
            ls configs/mcp_agent.config.yaml
            ```
            """)
        
        # 에이전트 소개
        render_agent_info()
        return
    else:
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
                value="TechCorp Inc.",
                help="보안 평가를 수행할 회사명을 입력하세요"
            )
            
            assessment_type = st.selectbox(
                "평가 유형",
                ["전체 보안 평가", "취약점 스캔만", "컴플라이언스 감사만", "사고 대응 계획만"]
            )
            
            st.markdown("#### 📋 컴플라이언스 프레임워크")
            
            frameworks = st.multiselect(
                "적용할 프레임워크",
                ["SOX", "ISO 27001", "NIST", "GDPR", "HIPAA"],
                default=["ISO 27001", "NIST", "GDPR"]
            )
            
            # 파일 저장 옵션
            save_to_file = st.checkbox(
                "파일로 저장", 
                value=False,
                help="체크하면 cybersecurity_infrastructure_reports/ 디렉토리에 파일로 저장합니다"
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
                    
                    # 결과 정보 표시
                    st.markdown("#### 📊 실행 결과")
                    st.info(f"**메시지**: {result['message']}")
                    st.info(f"**평가 대상**: {result['company_name']}")
                    st.info(f"**평가 유형**: {result['assessment_type']}")
                    if result['save_to_file'] and result['output_dir']:
                        st.info(f"**출력 디렉토리**: {result['output_dir']}")
                    st.info(f"**적용 프레임워크**: {', '.join(result['frameworks'])}")
                    
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
                            file_name=f"cybersecurity_assessment_{company_name.replace(' ', '_').lower()}_{assessment_type.replace(' ', '_')}.md",
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
                - 💾 **파일 저장**: cybersecurity_infrastructure_reports/ 디렉토리에 저장
                """)
                
    except Exception as e:
        st.error(f"Agent 초기화 중 오류: {e}")
        st.info("에이전트 클래스를 확인해주세요.")

def execute_cybersecurity_agent(company_name, assessment_type, frameworks, save_to_file):
    """Cybersecurity Agent 실행"""
    
    try:
        with st.spinner("🔄 Cybersecurity Agent를 실행하는 중..."):
            # 에이전트 초기화
            if 'cybersecurity_agent' not in st.session_state:
                st.session_state.cybersecurity_agent = CybersecurityAgent()
            
            agent = st.session_state.cybersecurity_agent
            
            # 실제 에이전트 실행
            result = agent.run_cybersecurity_workflow(
                company_name=company_name,
                assessment_type=assessment_type,
                frameworks=frameworks,
                save_to_file=save_to_file
            )
            
            st.session_state['cybersecurity_execution_result'] = result
            st.rerun()
            
    except Exception as e:
        st.session_state['cybersecurity_execution_result'] = {
            'success': False,
            'message': f'Agent 실행 중 오류 발생: {str(e)}',
            'error': str(e),
            'company_name': company_name,
            'assessment_type': assessment_type,
            'frameworks': frameworks,
            'save_to_file': save_to_file
        }
        st.rerun()

def render_agent_info():
    """에이전트 기능 소개"""
    
    st.markdown("### 🔒 Cybersecurity Agent 소개")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🛡️ 보안 평가 기능
        - **취약점 스캔**: 네트워크, 웹앱, 데이터베이스 보안 검사
        - **침투 테스트**: 외부/내부 네트워크 및 소셜 엔지니어링
        - **위험 평가**: 자산 분류, 위협 모델링, CVSS 점수
        - **보안 제어**: 접근 제어, 암호화, 모니터링 평가
        - **보안 개선**: 패치 관리, 보안 통제 강화 계획
        """)
    
    with col2:
        st.markdown("""
        #### 📋 컴플라이언스 감사
        - **SOX**: IT 통제 및 변경 관리
        - **ISO 27001**: 정보보안 관리 체계
        - **NIST**: 사이버보안 프레임워크
        - **GDPR**: 기술적 보호 조치
        - **HIPAA**: 관리적/물리적/기술적 보호조치
        """)
    
    st.markdown("#### 🎯 특화 기능")
    special_features = [
        "🚨 **사고 대응**: 인시던트 대응 계획 및 위협 인텔리전스",
        "🏗️ **제로 트러스트**: 네트워크 보안 아키텍처 설계",
        "☁️ **클라우드 보안**: 멀티클라우드 거버넌스 및 컨테이너 보안",
        "🔐 **데이터 보호**: 암호화, DLP, 백업 및 재해 복구",
        "📊 **보안 대시보드**: KPI 추적 및 성과 측정",
        "💼 **경영진 리포트**: 예산 고려사항 및 로드맵 제공"
    ]
    
    for feature in special_features:
        st.markdown(f"- {feature}")

if __name__ == "__main__":
    main() 