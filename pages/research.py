"""
🔍 Research Agent Page

정보 검색 및 분석 AI
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 공통 유틸리티 임포트
from srcs.common.page_utils import create_agent_page

# Research Agent 임포트 시도
try:
    from srcs.basic_agents.researcher_v2 import ResearcherAgent
    RESEARCH_AGENT_AVAILABLE = True
except ImportError as e:
    RESEARCH_AGENT_AVAILABLE = False
    import_error = str(e)

def main():
    """Research Agent 메인 페이지"""
    
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
        <h1>🔍 Research Agent</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            AI 기반 정보 검색 및 분석 시스템
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    st.markdown("---")
    
    # Agent 연동 상태 확인
    if not RESEARCH_AGENT_AVAILABLE:
        st.error(f"⚠️ Research Agent를 불러올 수 없습니다: {import_error}")
        st.info("에이전트 모듈을 확인하고 필요한 의존성을 설치해주세요.")
        
        with st.expander("🔧 설치 가이드"):
            st.markdown("""
            ### Research Agent v2 설정
            
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
        st.success("🤖 Research Agent v2가 성공적으로 연결되었습니다!")
        
        # 에이전트 인터페이스
        render_research_agent_interface()

def render_research_agent_interface():
    """Research Agent 실행 인터페이스"""
    
    st.markdown("### 🚀 Research Agent 실행")
    
    # 에이전트 초기화
    try:
        # 연구 주제 입력
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 🎯 연구 설정")
            
            research_topic = st.text_input(
                "연구 주제",
                value="AI and machine learning trends",
                help="조사하고 싶은 주제를 입력하세요"
            )
            
            research_focus = st.selectbox(
                "연구 초점",
                ["종합 분석", "트렌드 분석", "경쟁 분석", "미래 전망", "시장 조사"]
            )
            
            # 파일 저장 옵션
            save_to_file = st.checkbox(
                "파일로 저장", 
                value=False,
                help="체크하면 research_reports/ 디렉토리에 파일로 저장합니다"
            )
            
            if st.button("🚀 Research Agent 실행", type="primary", use_container_width=True):
                execute_research_agent(research_topic, research_focus, save_to_file)
        
        with col2:
            if 'research_execution_result' in st.session_state:
                result = st.session_state['research_execution_result']
                
                if result['success']:
                    st.success("✅ Research Agent 실행 완료!")
                    
                    # 결과 정보 표시
                    st.markdown("#### 📊 실행 결과")
                    st.info(f"**메시지**: {result['message']}")
                    st.info(f"**연구 주제**: {result['topic']}")
                    if result['save_to_file'] and result['output_dir']:
                        st.info(f"**출력 디렉토리**: {result['output_dir']}")
                    st.info(f"**연구 초점**: {result['focus']}")
                    
                    # 생성된 콘텐츠 표시
                    if 'content' in result and result['content']:
                        st.markdown("#### 📄 생성된 연구 결과")
                        
                        # 콘텐츠를 보기 좋게 표시
                        content = result['content']
                        
                        # 텍스트가 너무 길면 확장 가능한 형태로 표시
                        if len(content) > 1500:
                            with st.expander("📋 전체 연구 결과 보기", expanded=True):
                                st.markdown(content)
                        else:
                            st.markdown(content)
                        
                        # 콘텐츠 다운로드 버튼
                        st.download_button(
                            label="📥 연구 결과 다운로드",
                            data=content,
                            file_name=f"research_result_{research_topic.replace(' ', '_').lower()}_{result['focus'].replace(' ', '_')}.md",
                            mime="text/markdown"
                        )
                    
                    # 상세 결과 (디버그용)
                    with st.expander("🔍 상세 실행 정보"):
                        st.json({
                            'success': result['success'],
                            'message': result['message'],
                            'topic': result['topic'],
                            'focus': result['focus'],
                            'save_to_file': result['save_to_file'],
                            'output_dir': result.get('output_dir'),
                            'timestamp': result.get('timestamp'),
                            'content_length': len(result.get('content', '')) if result.get('content') else 0
                        })
                        
                else:
                    st.error("❌ 실행 중 오류 발생")
                    st.error(f"**오류**: {result['message']}")
                    
                    with st.expander("🔍 오류 상세"):
                        st.code(result.get('error', 'Unknown error'))
        
            else:
                st.markdown("""
                #### 🤖 Research Agent 정보
                
                **실행되는 프로세스:**
                1. **다중 에이전트 생성** - 전문 연구 AI 에이전트들
                2. **MCP App 초기화** - MCP 프레임워크 연결
                3. **오케스트레이터 실행** - 통합 워크플로우 관리
                4. **연구 수행** - 포괄적 정보 수집 및 분석
                
                **생성되는 연구 결과:**
                - 📈 **트렌드 분석**: 현재 동향 및 발전 패턴
                - 🏢 **경쟁 분석**: 주요 업체 및 시장 현황
                - 🔮 **미래 전망**: 전략적 시사점 및 기회
                - 📋 **종합 보고서**: 실행 요약 및 권고사항
                
                **출력 옵션:**
                - 🖥️ **화면 표시**: 즉시 결과 확인 (기본값)
                - 💾 **파일 저장**: research_reports/ 디렉토리에 저장
                """)
                
    except Exception as e:
        st.error(f"Agent 초기화 중 오류: {e}")
        st.info("에이전트 클래스를 확인해주세요.")

def execute_research_agent(topic, focus, save_to_file):
    """Research Agent 실행"""
    
    try:
        with st.spinner("🔄 Research Agent를 실행하는 중..."):
            # 에이전트 초기화
            if 'research_agent' not in st.session_state:
                st.session_state.research_agent = ResearcherAgent(research_topic=topic)
            
            agent = st.session_state.research_agent
            
            # 실제 에이전트 실행
            result = agent.run_research_workflow(
                topic=topic,
                focus=focus,
                save_to_file=save_to_file
            )
            
            st.session_state['research_execution_result'] = result
            st.rerun()
            
    except Exception as e:
        st.session_state['research_execution_result'] = {
            'success': False,
            'message': f'Agent 실행 중 오류 발생: {str(e)}',
            'error': str(e),
            'topic': topic,
            'focus': focus,
            'save_to_file': save_to_file
        }
        st.rerun()

def render_agent_info():
    """에이전트 기능 소개"""
    
    st.markdown("### 🔍 Research Agent 소개")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📊 주요 기능
        - **종합 정보 수집**: 다양한 소스에서 정보 수집
        - **트렌드 분석**: 최신 동향 및 패턴 분석
        - **경쟁 분석**: 시장 참여자 및 경쟁 현황
        - **미래 전망**: 전략적 시사점 및 예측
        - **보고서 생성**: 구조화된 연구 보고서 작성
        """)
    
    with col2:
        st.markdown("""
        #### ✨ 고급 기능
        - **다중 에이전트**: 전문화된 연구 에이전트들
        - **품질 평가**: EvaluatorOptimizer 적용
        - **실시간 분석**: 최신 정보 기반 분석
        - **구조화 출력**: 마크다운 형식 보고서
        - **KPI 추적**: 연구 품질 지표 모니터링
        """)
    
    st.markdown("#### 🎯 사용 사례")
    use_cases = [
        "기술 트렌드 조사 및 분석",
        "시장 동향 및 경쟁 분석",
        "신규 사업 기회 탐색",
        "학술 연구 지원",
        "전략 기획 정보 수집"
    ]
    
    for use_case in use_cases:
        st.markdown(f"- {use_case}")

if __name__ == "__main__":
    main() 