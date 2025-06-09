"""
📝 RAG Agent Page

문서 기반 질의응답 및 지식 관리 AI
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# RAG Agent 임포트 시도
try:
    from srcs.basic_agents.rag_agent import main as rag_main, initialize_collection, MCPApp
    RAG_AGENT_AVAILABLE = True
except ImportError as e:
    RAG_AGENT_AVAILABLE = False
    import_error = str(e)

def main():
    """RAG Agent 메인 페이지"""
    
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
        <h1>📝 RAG Agent</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            문서 기반 질의응답 및 지식 관리 시스템
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    # 파일 저장 옵션 추가
    save_to_file = st.checkbox(
        "대화 결과를 파일로 저장", 
        value=False,
        help="체크하면 rag_agent_reports/ 디렉토리에 대화 내용을 파일로 저장합니다"
    )
    
    st.markdown("---")
    
    # Agent 연동 상태 확인
    if not RAG_AGENT_AVAILABLE:
        st.error(f"⚠️ RAG Agent를 불러올 수 없습니다: {import_error}")
        st.info("에이전트 모듈을 확인하고 필요한 의존성을 설치해주세요.")
        
        with st.expander("🔧 설치 가이드"):
            st.markdown("""
            ### RAG Agent 설정
            
            1. **필요한 패키지 설치**:
            ```bash
            pip install qdrant-client openai streamlit asyncio
            ```
            
            2. **Qdrant 서버 실행**:
            ```bash
            # Docker로 Qdrant 실행
            docker run -p 6333:6333 qdrant/qdrant
            ```
            
            3. **환경 변수 설정**:
            ```bash
            export OPENAI_API_KEY="your-api-key"
            ```
            """)
        
        # 에이전트 소개
        render_agent_info()
        return
    else:
        st.success("🤖 RAG Agent가 연결되었습니다!")
        
        # Qdrant 서버 연결 확인
        check_qdrant_connection()
        
        # RAG Agent 실행
        render_rag_interface()

def check_qdrant_connection():
    """Qdrant 서버 연결 상태 확인"""
    
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient("http://localhost:6333")
        
        # 연결 테스트
        collections = client.get_collections()
        st.success("✅ Qdrant 서버에 연결되었습니다!")
        
        # 컬렉션 정보 표시
        if hasattr(collections, 'collections') and collections.collections:
            st.info(f"사용 가능한 컬렉션: {len(collections.collections)}개")
        
    except Exception as e:
        st.warning("⚠️ Qdrant 서버에 연결할 수 없습니다.")
        st.error(f"오류: {e}")
        
        st.markdown("### 🔧 Qdrant 서버 설정")
        st.code("""
# Qdrant 서버 실행 (Docker)
docker run -p 6333:6333 qdrant/qdrant

# 또는 로컬 설치
pip install qdrant-client
        """)

def render_rag_interface():
    """RAG Agent 실행 인터페이스"""
    
    st.markdown("### 🤖 RAG Agent 실행")
    
    # 초기화 버튼
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🔄 컬렉션 초기화", use_container_width=True):
            try:
                with st.spinner("Qdrant 컬렉션을 초기화하는 중..."):
                    initialize_collection()
                st.success("✅ 컬렉션이 초기화되었습니다!")
            except Exception as e:
                st.error(f"컬렉션 초기화 실패: {e}")
    
    with col2:
        st.info("컬렉션을 먼저 초기화한 후 채팅을 시작하세요.")
    
    st.markdown("---")
    
    # RAG 챗봇 실행
    try:
        # Streamlit 환경에서 안전한 실행
        render_rag_chatbot()
        
    except Exception as e:
        st.error(f"RAG Agent 실행 중 오류: {e}")
        st.info("다음 사항을 확인해주세요:")
        st.markdown("""
        - Qdrant 서버가 실행 중인지 확인
        - OpenAI API 키가 설정되어 있는지 확인
        - 컬렉션이 초기화되었는지 확인
        """)

def render_rag_chatbot():
    """RAG 챗봇 인터페이스 (Streamlit 호환)"""
    
    st.markdown("### 💬 RAG Chatbot")
    st.caption("🚀 Model Context Protocol 관련 질문을 해보세요!")
    
    # 샘플 질문
    with st.expander("💡 샘플 질문들"):
        sample_questions = [
            "Model Context Protocol이 무엇인가요?",
            "MCP의 주요 장점은 무엇인가요?",
            "Claude Desktop에서 MCP를 어떻게 사용하나요?",
            "Block과 Apollo는 MCP를 어떻게 활용하고 있나요?"
        ]
        
        for question in sample_questions:
            if st.button(f"📝 {question}", key=f"sample_{hash(question)}"):
                st.session_state.selected_question = question
    
    # 메시지 히스토리 초기화
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = [
            {"role": "assistant", "content": "안녕하세요! Model Context Protocol에 대해 무엇이든 물어보세요. 🤖"}
        ]
    
    # 기존 메시지 표시
    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # 사용자 입력
    user_input = st.chat_input("MCP에 대해 궁금한 것을 물어보세요...")
    
    # 샘플 질문 선택 처리
    if hasattr(st.session_state, 'selected_question'):
        user_input = st.session_state.selected_question
        delattr(st.session_state, 'selected_question')
    
    if user_input:
        # 사용자 메시지 추가
        st.session_state.rag_messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("RAG 시스템이 응답을 생성하는 중..."):
                try:
                    # 실제 RAG 검색 및 응답 생성은 여기서 구현
                    # 현재는 시뮬레이션
                    response = generate_rag_response(user_input)
                    st.write(response)
                    
                    # 응답을 히스토리에 추가
                    st.session_state.rag_messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"응답 생성 중 오류가 발생했습니다: {e}"
                    st.error(error_msg)
                    st.session_state.rag_messages.append({"role": "assistant", "content": error_msg})

def generate_rag_response(question):
    """RAG 기반 응답 생성 (시뮬레이션)"""
    
    # MCP 관련 기본 응답들
    responses = {
        "model context protocol": """
        **Model Context Protocol (MCP)**는 AI 어시스턴트를 데이터가 있는 시스템에 연결하는 새로운 표준입니다.
        
        🎯 **주요 특징:**
        - 콘텐츠 저장소, 비즈니스 도구, 개발 환경과의 연결
        - 프론티어 모델의 더 나은, 관련성 높은 응답 생성
        - 범용적이고 개방적인 표준 제공
        
        📊 **구조:**
        - 개발자가 MCP 서버를 통해 데이터 노출
        - AI 애플리케이션(MCP 클라이언트)이 서버에 연결
        - 안전하고 양방향 연결 제공
        """,
        
        "장점": """
        **MCP의 주요 장점:**
        
        🔗 **연결성:**
        - 파편화된 통합을 단일 프로토콜로 대체
        - 데이터 사일로와 레거시 시스템으로부터 AI 해방
        
        🛠️ **개발 효율성:**
        - 각 데이터 소스별 커스텀 구현 불필요
        - 표준 프로토콜 기반 개발
        
        📈 **확장성:**
        - 진정으로 연결된 시스템의 확장 가능
        - 지속 가능한 아키텍처 제공
        """,
        
        "claude": """
        **Claude Desktop에서의 MCP 지원:**
        
        🖥️ **로컬 MCP 서버 지원:**
        - Claude Desktop 앱에서 직접 MCP 서버 연결
        - 모든 Claude.ai 플랜에서 지원
        
        🏢 **Claude for Work:**
        - 고객이 로컬에서 MCP 서버 테스트 가능
        - 내부 시스템 및 데이터셋과 연결
        - 곧 원격 프로덕션 MCP 서버 지원 예정
        
        🔧 **사전 구축된 서버:**
        - Google Drive, Slack, GitHub, Git, Postgres, Puppeteer 등
        """,
        
        "block apollo": """
        **Block과 Apollo의 MCP 활용:**
        
        🏗️ **Block:**
        - CTO Dhanji R. Prasanna: "오픈 소스는 개발 모델 이상의 의미"
        - MCP를 통한 에이전트 시스템 구축
        - 기계적 부담 제거로 창의적 작업에 집중
        
        🚀 **Apollo:**
        - MCP를 시스템에 통합
        - 개발 도구 회사들과 협력
        
        🔧 **개발 도구 회사들:**
        - Zed, Replit, Codeium, Sourcegraph
        - AI 에이전트의 관련 정보 검색 향상
        - 더 정교하고 기능적인 코드 생성
        """
    }
    
    # 질문에서 키워드 매칭
    question_lower = question.lower()
    
    for keyword, response in responses.items():
        if keyword in question_lower:
            return response
    
    # 기본 응답
    return f"""
    질문: "{question}"에 대한 구체적인 정보를 찾지 못했습니다.
    
    하지만 Model Context Protocol에 대한 일반적인 정보를 제공해드릴 수 있습니다:
    
    🤖 **MCP는 AI 시스템과 데이터 소스를 연결하는 개방형 표준입니다.**
    
    더 구체적인 질문을 해주시면 관련 정보를 찾아서 답변드리겠습니다!
    
    💡 **추천 질문:**
    - "MCP의 주요 특징은?"
    - "Claude에서 MCP 사용법은?"
    - "MCP의 장점은 무엇인가요?"
    """

def render_agent_info():
    """에이전트 기능 소개"""
    
    st.markdown("### 📝 RAG Agent 소개")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📊 주요 기능
        - **문서 기반 QA**: 업로드한 문서를 기반으로 정확한 답변
        - **지식 베이스 구축**: 자동 문서 인덱싱 및 검색
        - **다양한 파일 지원**: PDF, DOCX, TXT, HTML 등
        - **의미적 검색**: 키워드가 아닌 의미 기반 검색
        - **출처 추적**: 답변의 근거 문서 및 페이지 제공
        """)
    
    with col2:
        st.markdown("""
        #### ✨ 스페셜 기능
        - **실시간 학습**: 새로운 문서 자동 반영
        - **다중 언어**: 한국어, 영어 등 다국어 지원
        - **개인화**: 사용자별 맞춤 지식 베이스
        - **버전 관리**: 문서 변경 이력 추적
        - **API 연동**: 외부 시스템과의 연계
        """)
    
    st.markdown("#### 🎯 사용 사례")
    use_cases = [
        "기업 내부 문서 검색 시스템",
        "고객 지원 챗봇", 
        "연구 논문 분석 도구",
        "정책 및 규정 문의 시스템"
    ]
    
    for use_case in use_cases:
        st.markdown(f"- {use_case}")

if __name__ == "__main__":
    main() 