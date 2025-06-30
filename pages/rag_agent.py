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

from srcs.common.page_utils import create_agent_page
from srcs.common.ui_utils import run_agent_process
from configs.settings import get_reports_path
from srcs.basic_agents.rag_agent import get_qdrant_status

def main():
    create_agent_page(
        "💬 RAG Agent",
        "Qdrant 벡터 데이터베이스와 연동하여 질문에 답변하는 RAG 챗봇",
        "pages/rag_agent.py"
    )

    # Qdrant 서버 상태 확인
    q_status = get_qdrant_status()
    if q_status.get("status") == "connected":
        st.sidebar.success(f"Qdrant 연결됨 ({q_status.get('collections_count')}개 컬렉션)")
    else:
        st.sidebar.error("Qdrant 연결 실패")
        with st.sidebar.expander("에러 상세"):
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
            
            reports_path = get_reports_path('rag')
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"rag_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 이전 대화 기록 (마지막 응답 제외)
            history = [msg for msg in st.session_state.rag_messages if msg['role'] != 'assistant']

            py_executable = sys.executable
            command = [
                py_executable, "-m", "srcs.basic_agents.run_rag_agent",
                "--query", prompt,
                "--history", json.dumps(history),
                "--result-json-path", str(result_json_path)
            ]
            
            # run_agent_process는 자체적으로 spinner를 표시하지만, 
            # 여기서는 chat_message 컨텍스트 내에서 결과를 바로 표시하기 위해
            # placeholder를 사용합니다.
            result = run_agent_process(
                placeholder=result_placeholder,
                command=command,
                process_key_prefix="rag_agent"
            )
            
            response_text = "죄송합니다, 답변을 생성하는 데 실패했습니다."
            if result and "data" in result and "response" in result["data"]:
                response_text = result["data"]["response"]
            elif result and "error" in result:
                response_text = f"오류 발생: {result['error']}"

            # 최종 응답을 placeholder에 표시
            result_placeholder.markdown(response_text)
            st.session_state.rag_messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main() 