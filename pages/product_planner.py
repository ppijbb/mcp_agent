import streamlit as st
import asyncio
from pathlib import Path
import sys
import json
from datetime import datetime
import os

from srcs.common.page_utils import create_agent_page
from srcs.common.ui_utils import run_agent_process
from configs.settings import get_reports_path

# Product Planner Agent는 자체 환경변수 로더를 사용
from srcs.product_planner_agent.utils import env_settings as env
from srcs.product_planner_agent.product_planner_agent import ProductPlannerAgent

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 제품 기획 분석 결과")

    if not result_data:
        st.warning("분석 결과를 찾을 수 없습니다.")
        return

    final_report = result_data.get('final_report', {})
    if not final_report:
        st.info("최종 보고서가 생성되지 않았습니다. 상세 로그를 확인해주세요.")
    else:
        st.success("✅ 최종 보고서가 성공적으로 생성되었습니다.")
        # 파일 경로가 있다면 링크 제공
        if 'file_path' in final_report:
            st.markdown(f"**보고서 위치**: `{final_report['file_path']}`")
        # 보고서 내용 표시
        with st.expander("📄 최종 보고서 내용 보기", expanded=True):
            st.markdown(final_report.get('content', '내용 없음'))

    with st.expander("상세 분석 결과 보기 (JSON)"):
        st.json(result_data)

async def run_full_workflow(user_input):
    # 초기 입력으로 agent state를 세팅
    response = await st.session_state.agent.process_message(user_input)
    st.session_state.chat_history.append({"role": "assistant", "content": response["message"]})
    # 단계별 자동 진행
    while response.get("state") != "complete":
        response = await st.session_state.agent.process_message("")  # 빈 입력으로 다음 단계 진행
        st.session_state.chat_history.append({"role": "assistant", "content": response["message"]})
        if response.get("state") == "complete" and "final_report" in response:
            display_results(response["final_report"])
    st.session_state.agent_state = st.session_state.agent.get_state()
    return response

async def main():
    create_agent_page(
        agent_name="Product Planner Agent",
        page_icon="🚀",
        page_type="product",
        title="Product Planner Agent",
        subtitle="Figma 디자인을 분석하여 시장 조사, 전략, 실행 계획까지 한번에 수립합니다.",
        module_path="srcs.product_planner_agent.run_product_planner"
    )
    # 세션별 agent/state 관리
    if "agent" not in st.session_state:
        st.session_state.agent = ProductPlannerAgent()
        st.session_state.agent_state = st.session_state.agent.get_state()
        st.session_state.chat_history = []
    else:
        # state 복원
        if "agent_state" in st.session_state:
            st.session_state.agent.set_state(st.session_state.agent_state)

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 전체 실행 입력 폼
    user_input = st.text_area("제품 기획에 대해 말씀해주세요...", "", key="planner_input")
    if st.button("한 번에 전체 실행 🚀"):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            with st.spinner("전체 워크플로우 실행 중..."):
                response = await run_full_workflow(user_input)
                st.markdown(response["message"])
                if response.get("state") == "complete" and "final_report" in response:
                    display_results(response["final_report"])
        st.session_state.chat_history.append({"role": "assistant", "content": response["message"]})
        st.session_state.agent_state = st.session_state.agent.get_state()

# Streamlit 1.25+ async 지원, 구버전 fallback
try:
    st.run(main)
except AttributeError:
    asyncio.run(main()) 