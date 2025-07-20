import streamlit as st
import asyncio
from pathlib import Path
import sys
import json
from datetime import datetime
import os
import time

from srcs.common.page_utils import create_agent_page
# from srcs.common.ui_utils import run_agent_process  # streamlit_process_manager 의존성 제거
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

def get_step_progress(step_name):
    """단계별 진행률 계산"""
    step_progress = {
        "init": 0,
        "figma_analysis": 20,
        "prd_drafting": 40,
        "figma_creation": 60,
        "report_generation": 80,
        "save_report": 90,
        "complete": 100
    }
    return step_progress.get(step_name, 0)

def get_step_icon(step_name):
    """단계별 아이콘"""
    step_icons = {
        "init": "🚀",
        "figma_analysis": "🎨",
        "prd_drafting": "📝",
        "figma_creation": "🔧",
        "report_generation": "📊",
        "save_report": "💾",
        "complete": "✅"
    }
    return step_icons.get(step_name, "⚙️")

def get_step_description(step_name):
    """단계별 설명"""
    step_descriptions = {
        "init": "초기화 중...",
        "figma_analysis": "Figma 디자인 분석 중...",
        "prd_drafting": "PRD 문서 작성 중...",
        "figma_creation": "Figma 컴포넌트 생성 중...",
        "report_generation": "최종 보고서 생성 중...",
        "save_report": "보고서 저장 중...",
        "complete": "완료!"
    }
    return step_descriptions.get(step_name, "처리 중...")

async def run_full_workflow(user_input, progress_bar, status_text, step_container):
    """전체 워크플로우 실행 with 진행률 표시"""
    try:
        # 초기 입력으로 agent state를 세팅
        current_step = "init"
        progress_bar.progress(get_step_progress(current_step))
        status_text.text(f"{get_step_icon(current_step)} {get_step_description(current_step)}")
        
        response = await st.session_state.agent.process_message(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": response["message"]})
        
        # 단계별 자동 진행
        max_retries = 3
        retry_count = 0
        
        while response.get("state") != "complete" and retry_count < max_retries:
            current_step = response.get("state", "processing")
            progress_bar.progress(get_step_progress(current_step))
            status_text.text(f"{get_step_icon(current_step)} {get_step_description(current_step)}")
            
            # 단계별 상세 로그 표시
            with step_container:
                st.info(f"**현재 단계**: {current_step}")
                st.text(response.get("message", "처리 중..."))
            
            response = await st.session_state.agent.process_message("")  # 빈 입력으로 다음 단계 진행
            st.session_state.chat_history.append({"role": "assistant", "content": response["message"]})
            
            if response.get("state") == "complete" and "final_report" in response:
                progress_bar.progress(100)
                status_text.text("✅ 완료!")
                display_results(response["final_report"])
                break
            elif response.get("state") == "error":
                retry_count += 1
                if retry_count < max_retries:
                    st.warning(f"오류 발생. 재시도 중... ({retry_count}/{max_retries})")
                    await asyncio.sleep(2)  # 잠시 대기 후 재시도
                else:
                    st.error("최대 재시도 횟수를 초과했습니다.")
                    break
        
        st.session_state.agent_state = st.session_state.agent.get_state()
        return response
        
    except Exception as e:
        st.error(f"워크플로우 실행 중 오류가 발생했습니다: {str(e)}")
        return {"state": "error", "message": f"오류: {str(e)}"}

def create_settings_sidebar():
    """설정 사이드바"""
    st.sidebar.markdown("## ⚙️ 설정")
    
    # 자동 실행 옵션
    auto_run = st.sidebar.checkbox("자동 실행", value=True, help="입력 후 자동으로 전체 워크플로우 실행")
    
    # 진행률 표시 옵션
    show_progress = st.sidebar.checkbox("진행률 표시", value=True, help="단계별 진행률 표시")
    
    # 상세 로그 옵션
    show_detailed_logs = st.sidebar.checkbox("상세 로그", value=True, help="단계별 상세 로그 표시")
    
    # 재시도 설정
    max_retries = st.sidebar.slider("최대 재시도 횟수", 1, 5, 3, help="오류 발생 시 재시도 횟수")
    
    return {
        "auto_run": auto_run,
        "show_progress": show_progress,
        "show_detailed_logs": show_detailed_logs,
        "max_retries": max_retries
    }

def create_quick_actions():
    """빠른 액션 버튼들"""
    st.markdown("### 🚀 빠른 시작")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📱 모바일 앱 기획", help="모바일 앱 제품 기획 템플릿"):
            return "모바일 앱 제품을 기획해주세요. 사용자 경험과 기능성을 중점으로 분석해주세요."
    
    with col2:
        if st.button("🌐 웹 서비스 기획", help="웹 서비스 제품 기획 템플릿"):
            return "웹 서비스 제품을 기획해주세요. 확장성과 사용자 편의성을 고려해주세요."
    
    with col3:
        if st.button("🤖 AI 서비스 기획", help="AI 기반 서비스 제품 기획 템플릿"):
            return "AI 기반 서비스 제품을 기획해주세요. 기술적 혁신과 실용성을 균형있게 분석해주세요."
    
    return None

async def main():
    create_agent_page(
        agent_name="Product Planner Agent",
        page_icon="🚀",
        page_type="product",
        title="Product Planner Agent",
        subtitle="Figma 디자인을 분석하여 시장 조사, 전략, 실행 계획까지 한번에 수립합니다.",
        module_path="srcs.product_planner_agent.run_product_planner"
    )
    
    # 설정 사이드바
    settings = create_settings_sidebar()
    
    # 세션별 agent/state 관리
    if "agent" not in st.session_state:
        st.session_state.agent = ProductPlannerAgent()
        st.session_state.agent_state = st.session_state.agent.get_state()
        st.session_state.chat_history = []
    else:
        # state 복원
        if "agent_state" in st.session_state:
            st.session_state.agent.set_state(st.session_state.agent_state)

    # 빠른 액션 버튼들
    quick_input = create_quick_actions()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 대화 기록")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 메인 입력 섹션
    st.markdown("### 📝 제품 기획 입력")
    
    # 전체 실행 입력 폼
    user_input = st.text_area(
        "제품 기획에 대해 말씀해주세요...",
        value=quick_input if quick_input else "",
        key="planner_input",
        height=150,
        help="구체적인 제품 아이디어나 요구사항을 자세히 설명해주세요."
    )
    
    # 실행 버튼들
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🚀 전체 워크플로우 실행", type="primary", use_container_width=True):
            if not user_input.strip():
                st.warning("제품 기획 내용을 입력해주세요.")
                return
                
            # 진행률 표시 초기화
            if settings["show_progress"]:
                progress_bar = st.progress(0)
                status_text = st.empty()
                step_container = st.container()
            else:
                progress_bar = None
                status_text = None
                step_container = None
            
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.chat_message("assistant"):
                with st.spinner("전체 워크플로우 실행 중..."):
                    response = await run_full_workflow(
                        user_input, 
                        progress_bar, 
                        status_text, 
                        step_container
                    )
                    st.markdown(response["message"])
                    if response.get("state") == "complete" and "final_report" in response:
                        display_results(response["final_report"])
            
            st.session_state.chat_history.append({"role": "assistant", "content": response["message"]})
            st.session_state.agent_state = st.session_state.agent.get_state()
    
    with col2:
        if st.button("🔄 재시작", help="에이전트 상태 초기화"):
            st.session_state.agent = ProductPlannerAgent()
            st.session_state.agent_state = st.session_state.agent.get_state()
            st.session_state.chat_history = []
            st.success("에이전트가 초기화되었습니다.")
            st.rerun()
    
    with col3:
        if st.button("📊 결과만 보기", help="이전 결과 확인"):
            if "agent_state" in st.session_state:
                current_state = st.session_state.agent.get_state()
                if current_state.get("final_report"):
                    display_results(current_state.get("final_report"))
                else:
                    st.info("저장된 결과가 없습니다.")

    # 상태 정보 표시
    if settings["show_detailed_logs"] and "agent_state" in st.session_state:
        with st.expander("🔍 현재 에이전트 상태"):
            st.json(st.session_state.agent_state)

# Streamlit 1.25+ async 지원, 구버전 fallback
try:
    st.run(main)
except AttributeError:
    asyncio.run(main()) 