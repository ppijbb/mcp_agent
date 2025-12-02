"""
📝 RAG Agent Page

문서 기반 질의응답 및 지식 관리 AI
"""

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
from srcs.common.standard_a2a_page_helper import execute_standard_agent_via_a2a
from srcs.common.agent_interface import AgentType
from configs.settings import get_reports_path
from srcs.basic_agents.rag_agent import get_qdrant_status

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def main():
    create_agent_page(
        agent_name="RAG Agent",
        page_icon="📝",
        page_type="rag",
        title="RAG Agent",
        subtitle="Qdrant 벡터 데이터베이스와 연동하여 질문에 답변하는 RAG 챗봇",
        module_path="srcs.basic_agents.run_rag_agent"
    )

    # Qdrant 서버 상태 확인
    q_status = get_qdrant_status()
    if q_status.get("status") != "connected":
        st.error("Qdrant 연결 실패")
        st.error(q_status.get('error'))
        st.stop()
    
    # 세션 상태에 대화 기록 초기화
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = [
            {"role": "assistant", "content": "Model Context Protocol(MCP)에 대해 무엇이 궁금하신가요?"}
        ]

    # 대화 기록 표시
    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("MCP에 대해 질문해보세요..."):
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # UI에 즉시 로딩 스피너 표시
        with st.chat_message("assistant"):
            result_placeholder = st.empty()
            with result_placeholder.container():
                with st.spinner("답변 생성 중..."):
                    reports_path = Path(get_reports_path('rag'))
                    reports_path.mkdir(parents=True, exist_ok=True)
                    result_json_path = reports_path / f"rag_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

                    # 이전 대화 기록 (마지막 응답 제외)
                    history = [msg for msg in st.session_state.rag_messages if msg['role'] != 'assistant']

                                # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                
                        "agent_id": "rag_agent",
                        "agent_name": "RAG Agent",
                        "entry_point": "srcs.basic_agents.run_rag_agent",
                        agent_type=AgentType.MCP_AGENT,
                        "capabilities": ["document_qa", "information_retrieval", "context_aware_answering"],
                        "description": "문서 기반 질의응답 및 정보 추출"
                    ,
                input_params=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            ),
                        agent_metadata=agent_metadata,
                        input_data=input_data,
                        result_json_path=result_json_path,
                        use_a2a=True
                    )
            
            response_text = "죄송합니다, 답변을 생성하는 데 실패했습니다."
            if result and "data" in result and "response" in result["data"]:
                response_text = result["data"]["response"]
            elif result and "error" in result:
                response_text = f"오류 발생: {result['error']}"

            # 최종 응답을 placeholder에 표시
            result_placeholder.markdown(response_text)
            st.session_state.rag_messages.append({"role": "assistant", "content": response_text})

    # 결과 확인 섹션 추가
    st.divider()
    
    # 최신 RAG 결과 확인
    latest_rag_result = result_reader.get_latest_result("rag_agent", "rag_query")
    
    if latest_rag_result:
        with st.expander("📊 최신 RAG 결과 확인", expanded=False):
            st.subheader("🤖 최근 질의응답 결과")
            
            if isinstance(latest_rag_result, dict):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**질문:**")
                    st.write(latest_rag_result.get('query', 'N/A'))
                
                with col2:
                    st.write("**답변:**")
                    st.write(latest_rag_result.get('response', 'N/A'))
                
                # 메타데이터 표시
                if 'collection_name' in latest_rag_result:
                    st.info(f"📚 사용된 컬렉션: {latest_rag_result['collection_name']}")
                
                if 'timestamp' in latest_rag_result:
                    st.caption(f"⏰ 생성 시간: {latest_rag_result['timestamp']}")
            else:

if __name__ == "__main__":
    main() 