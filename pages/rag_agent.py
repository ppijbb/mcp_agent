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

# 중앙 설정 임포트
from configs.settings import get_reports_path

# RAG Agent 임포트 시도
try:
    from srcs.basic_agents.rag_agent import (
        main as rag_main, 
        initialize_collection, 
        MCPApp,
        load_collection_types,
        load_document_formats,
        get_qdrant_status,
        get_available_collections,
        save_rag_conversation,
        generate_rag_response
    )
except ImportError as e:
    st.error(f"⚠️ RAG Agent를 불러올 수 없습니다: {e}")
    st.info("에이전트 모듈을 확인하고 필요한 의존성을 설치해주세요.")
    st.stop()

def validate_rag_result(result):
    """RAG 결과 검증"""
    if not result:
        raise Exception("RAG 시스템에서 유효한 결과를 반환하지 않았습니다")
    return result

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
        help=f"체크하면 {get_reports_path('rag_agent')}/ 디렉토리에 대화 내용을 파일로 저장합니다"
    )
    
    st.markdown("---")
    
    st.success("🤖 RAG Agent가 연결되었습니다!")
        
    # Qdrant 서버 연결 확인
    check_qdrant_connection()
        
    # RAG Agent 실행
    render_rag_interface()

def check_qdrant_connection():
    """Qdrant 서버 연결 상태 확인"""
    
    try:
        status = get_qdrant_status()
        validate_rag_result(status)
        st.success("✅ Qdrant 서버에 연결되었습니다!")
        
        # 컬렉션 정보 표시
        collections = get_available_collections()
        if collections:
            st.info(f"사용 가능한 컬렉션: {len(collections)}개")
        
    except Exception as e:
        st.error(f"Qdrant 서버 연결 실패: {e}")
        st.stop()

def render_rag_interface():
    """RAG Agent 실행 인터페이스"""
    
    st.markdown("### 🤖 RAG Agent 실행")
    
    # 초기화 버튼
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🔄 컬렉션 초기화", use_container_width=True):
            try:
                with st.spinner("Qdrant 컬렉션을 초기화하는 중..."):
                    result = initialize_collection()
                    validate_rag_result(result)
                st.success("✅ 컬렉션이 초기화되었습니다!")
            except Exception as e:
                st.error(f"컬렉션 초기화 실패: {e}")
    
    with col2:
        st.info("컬렉션을 먼저 초기화한 후 채팅을 시작하세요.")
    
    st.markdown("---")
    
    # RAG 챗봇 실행
    try:
        render_rag_chatbot()
        
    except Exception as e:
        st.error(f"RAG Agent 실행 중 오류: {e}")

def render_rag_chatbot():
    """RAG 챗봇 인터페이스"""
    
    st.markdown("### 💬 RAG Chatbot")
    st.caption("🚀 문서 기반 질의응답을 시작하세요!")
    
    # ✅ P2: Sample questions fallback system removed - Using real RAG Agent dynamic questions
    st.info("💡 문서가 로드된 후 관련 샘플 질문들이 자동으로 생성됩니다.")
    
    # 메시지 히스토리 초기화
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = [
            {"role": "assistant", "content": "안녕하세요! 문서 기반 질의응답 시스템입니다. 궁금한 것을 물어보세요! 🤖"}
        ]
    
    # 기존 메시지 표시
    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # 사용자 입력
    user_input = st.chat_input("문서에 대해 궁금한 것을 물어보세요...")
    
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
                    response = generate_rag_response(user_input)
                    validate_rag_result(response)
                    st.write(response)
                    
                    # 응답을 히스토리에 추가
                    st.session_state.rag_messages.append({"role": "assistant", "content": response})
                    
                    # 파일 저장 옵션이 활성화된 경우
                    if st.session_state.get('save_to_file', False):
                        filename = f"rag_conversation_{len(st.session_state.rag_messages)}.txt"
                        save_rag_conversation(st.session_state.rag_messages, filename)
                    
                except Exception as e:
                    error_msg = f"응답 생성 중 오류가 발생했습니다: {e}"
                    st.error(error_msg)
                    st.session_state.rag_messages.append({"role": "assistant", "content": error_msg})

# ✅ P2: Removed load_sample_questions fallback function
# ✅ P1-2: generate_rag_response 함수는 srcs.basic_agents.rag_agent에서 import

if __name__ == "__main__":
    main() 