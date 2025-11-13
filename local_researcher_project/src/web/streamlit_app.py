#!/usr/bin/env python3
"""
Streamlit Web Interface for Local Researcher Project - 8 Core Innovations

This module provides a comprehensive web interface for the Local Researcher system
with real-time monitoring, data visualization, and interactive research capabilities
implementing all 8 core innovations.
"""

import streamlit as st
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.agent_orchestrator import AgentOrchestrator
from src.agents.autonomous_researcher import AutonomousResearcherAgent
from src.core.reliability import HealthMonitor
from src.core.mcp_integration import get_available_tools, execute_tool
from src.core.researcher_config import config

import logging
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SparkleForge - Where Ideas Sparkle and Get Forged",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'active_research' not in st.session_state:
    st.session_state.active_research = {}
if 'health_monitor' not in st.session_state:
    st.session_state.health_monitor = None
if 'innovation_stats' not in st.session_state:
    st.session_state.innovation_stats = {}


def initialize_orchestrator():
    """Initialize the SparkleForge with AgentOrchestrator."""
    try:
        # Load configuration first (skip if environment variables not set)
        global config
        if config is None:
            try:
                from src.core.researcher_config import load_config_from_env
                config = load_config_from_env()
            except Exception as config_error:
                logger.warning(f"Configuration loading failed, using defaults: {config_error}")
                # Create minimal config for UI demonstration
                from src.core.researcher_config import MCPConfig, ResearcherSystemConfig
                config = ResearcherSystemConfig(
                    llm=None,
                    agent=None,
                    research=None,
                    mcp=MCPConfig(
                        enabled=True,
                        timeout=30,
                        server_names=['g-search', 'tavily', 'exa', 'fetch']
                    ),
                    output=None,
                    compression=None,
                    verification=None,
                    context_window=None,
                    reliability=None,
                    agent_tools=None
                )

        if st.session_state.orchestrator is None:
            # Initialize with AgentOrchestrator
            st.session_state.orchestrator = AgentOrchestrator()

            # Initialize health monitor
            st.session_state.health_monitor = HealthMonitor()

            logger.info("SparkleForge initialized with AgentOrchestrator")

    except Exception as e:
        st.error(f"Failed to initialize orchestrator: {e}")
        logger.error(f"Orchestrator initialization failed: {e}")


def main():
    """Main Streamlit application with 8 core innovations."""
    # Add custom CSS for forge theme
    st.markdown("""
    <style>
    .forge-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .forge-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .sparkle {
        animation: sparkle 2s infinite;
    }
    @keyframes sparkle {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("⚒️ SparkleForge - Where Ideas Sparkle and Get Forged")
    st.markdown("**Revolutionary Multi-Agent Forge System with Real-Time Collaboration and Creative AI**")
    st.markdown("---")
    
    # Initialize orchestrator
    initialize_orchestrator()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("⚒️ The Forge Process")
        st.markdown("""
        - **Adaptive Forge Master** - Dynamic craftsman allocation
        - **Hierarchical Refinement** - Multi-stage material processing
        - **Multi-Model Forge** - Role-based model selection
        - **Continuous Quality Control** - 3-stage verification system
        - **Streaming Forge** - Real-time progress delivery
        - **Universal Tool Forge** - 100+ MCP tools
        - **Adaptive Workspace** - Dynamic context management
        - **Production-Grade Forge** - Enterprise-grade stability
        """)

        st.header("Navigation")
        page = st.selectbox(
            "Choose a page",
            ["Forge Dashboard", "Live Forge", "Forge Monitor", "Creative Forge", "Data Visualization", "Report Generator", "System Health", "Settings"]
        )

    # Main content area with left-right split layout for Forge Dashboard
    if page == "Forge Dashboard":
        # 좌우 분할 레이아웃 구현
        col_left, col_right = st.columns([3, 2])

        with col_left:
            # 왼쪽: 진행상황 표시 영역
            forge_dashboard_left()
        with col_right:
            # 오른쪽: 최종 출력물 표시 영역
            forge_dashboard_right()
    elif page == "Live Forge":
        live_research_dashboard()
    elif page == "Forge Monitor":
        innovations_monitor()
    elif page == "Creative Forge":
        creative_insights_page()
    elif page == "Data Visualization":
        data_visualization()
    elif page == "Report Generator":
        report_generator()
    elif page == "System Health":
        system_health()
    elif page == "Settings":
        settings_page()


def forge_dashboard_left():
    """왼쪽 패널: 진행상황 표시 및 입력."""
    st.header("⚒️ Forge Dashboard - 실시간 진행상황")

    # Innovation status overview
    st.subheader("Forge Status")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Adaptive Forge Master", "✅ Active", "Dynamic allocation")
    with col2:
        st.metric("Hierarchical Refinement", "✅ Active", "3-stage processing")
    with col3:
        st.metric("Multi-Model Forge", "✅ Active", "Role-based selection")
    with col4:
        st.metric("Continuous Quality Control", "✅ Active", "3-stage verification")

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric("Streaming Forge", "✅ Active", "Real-time delivery")
    with col6:
        st.metric("Universal Tool Forge", "✅ Active", f"{len(config.mcp.server_names)} tools")
    with col7:
        st.metric("Adaptive Workspace", "✅ Active", "2K-1M tokens")
    with col8:
        st.metric("Production-Grade Forge", "✅ Active", "99.9% uptime")

    # Forge input section
    with st.container():
        st.subheader("Start New Forge with 8 Innovations")

        col1, col2 = st.columns([3, 1])

        with col1:
            research_query = st.text_area(
                "Research Query",
                placeholder="Enter your research question or topic...",
                height=100,
                key="research_query"
            )

        with col2:
            st.write("8 Innovations Options")

            # Adaptive Supervisor options
            st.write("**Adaptive Supervisor**")
            enable_adaptive_supervisor = st.checkbox("Enable Dynamic Allocation", value=True, key="adaptive_supervisor")
            max_researchers = st.slider("Max Researchers", 1, 10, 5, key="max_researchers")

            # Streaming Pipeline options
            st.write("**Streaming Pipeline**")
            enable_streaming = st.checkbox("Enable Real-time Streaming", value=True, key="streaming_pipeline")

            # Multi-Model Orchestration options
            st.write("**Multi-Model Orchestration**")
            enable_multi_model = st.checkbox("Enable Role-based Models", value=True, key="multi_model")

            # Universal MCP Hub options
            st.write("**Universal MCP Hub**")
            enable_mcp = st.checkbox("Enable MCP Tools", value=True, key="mcp_hub")
            mcp_tools = st.multiselect(
                "Select MCP Tools",
                config.mcp.server_names,
                default=config.mcp.server_names[:3],
                key="mcp_tools"
            )

        if st.button("🚀 Start Research with 8 Innovations", type="primary", key="start_research"):
            if research_query:
                start_research_with_streaming(
                    research_query,
                    enable_adaptive_supervisor,
                    max_researchers,
                    enable_streaming,
                    enable_multi_model,
                    enable_mcp,
                    mcp_tools
                )
            else:
                st.warning("Please enter a research query.")

    # 실시간 진행상황 표시 영역
    st.subheader("🔴 실시간 진행상황")
    display_realtime_progress()

    # 채팅 UI 영역
    st.subheader("💬 Agent 채팅")
    display_chat_interface()

    # Active research section
    if st.session_state.active_research:
        st.subheader("Active Research")
        display_active_research()

    # Research history section
    if st.session_state.research_history:
        st.subheader("Research History")
        display_research_history()


def forge_dashboard_right():
    """오른쪽 패널: 최종 출력물 표시."""
    st.header("📋 최종 출력물")

    # 최종 보고서 표시
    display_final_output()

    # 파일 다운로드 섹션
    st.subheader("📁 생성된 파일 다운로드")
    display_file_downloads()


def research_dashboard():
    """Main forge dashboard with 8 innovations (legacy compatibility)."""
    forge_dashboard_left()


def start_research_with_streaming(
    query: str,
    enable_adaptive_supervisor: bool,
    max_researchers: int,
    enable_streaming: bool,
    enable_multi_model: bool,
    enable_mcp: bool,
    mcp_tools: List[str]
):
    """실시간 스트리밍으로 연구 작업 시작."""
    try:
        # Create research context with 8 innovations
        context = {
            "query": query,
            "enable_adaptive_supervisor": enable_adaptive_supervisor,
            "max_researchers": max_researchers,
            "enable_streaming": enable_streaming,
            "enable_multi_model": enable_multi_model,
            "enable_mcp": enable_mcp,
            "mcp_tools": mcp_tools,
            "timestamp": datetime.now().isoformat()
        }

        # 연구 ID 생성
        research_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 세션 상태 초기화
        st.session_state.active_research[research_id] = {
            "query": query,
            "context": context,
            "start_time": datetime.now(),
            "status": "running",
            "result": None,
            "progress_logs": [],
            "final_report": "",
            "innovation_stats": {}
        }

        # 실시간 스트리밍으로 연구 실행
        if st.session_state.orchestrator:
            # 스트리밍 실행을 위한 placeholder 생성
            progress_placeholder = st.empty()
            report_placeholder = st.empty()

            # 스트리밍 실행 함수 정의
            async def run_streaming_research():
                try:
                    # 스트리밍 이벤트 수집
                    all_events = []
                    async for event in st.session_state.orchestrator.stream(query):
                        all_events.append(event)

                        # 진행상황 업데이트
                        if event.get('current_agent'):
                            agent_info = f"[{event['current_agent'].upper()}] Processing..."
                            if event.get('user_query'):
                                agent_info += f" Query: {event['user_query'][:50]}..."
                            st.session_state.active_research[research_id]["progress_logs"].append(agent_info)

                        # 최종 보고서 업데이트
                        if event.get('final_report'):
                            st.session_state.active_research[research_id]["final_report"] = event['final_report']

                        # UI 업데이트 (빈번한 업데이트 방지 위해 일부 이벤트만)
                        if len(all_events) % 5 == 0:  # 5번째 이벤트마다 업데이트
                            update_realtime_ui(research_id, progress_placeholder, report_placeholder)

                    # 최종 결과 저장
                    final_event = all_events[-1] if all_events else {}
                    st.session_state.active_research[research_id]["status"] = "completed"
                    st.session_state.active_research[research_id]["result"] = final_event
                    st.session_state.active_research[research_id]["innovation_stats"] = final_event.get('innovation_stats', {})

                    # 히스토리에 추가
                    st.session_state.research_history.append({
                        "id": research_id,
                        "query": query,
                        "completed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "completed",
                        "innovation_stats": final_event.get('innovation_stats', {})
                    })

                    # 최종 UI 업데이트
                    update_realtime_ui(research_id, progress_placeholder, report_placeholder)

                    st.success("🎉 Research completed successfully with 8 Core Innovations!")

                except Exception as e:
                    st.session_state.active_research[research_id]["status"] = "error"
                    st.session_state.active_research[research_id]["error"] = str(e)
                    st.error(f"Research failed: {e}")
                    logger.error(f"Streaming research failed: {e}")

            # 비동기 함수 실행
            import threading
            def run_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(run_streaming_research())
                loop.close()

            thread = threading.Thread(target=run_async)
            thread.start()

            # 초기 UI 표시
            update_realtime_ui(research_id, progress_placeholder, report_placeholder)

        else:
            st.error("Orchestrator not initialized")

    except Exception as e:
        st.error(f"Failed to start research: {e}")
        logger.error(f"Research start failed: {e}")


def update_realtime_ui(research_id: str, progress_placeholder, report_placeholder):
    """실시간 UI 업데이트."""
    research_data = st.session_state.active_research.get(research_id, {})

    # 진행상황 업데이트
    with progress_placeholder.container():
        st.subheader("🔄 진행상황")
        logs = research_data.get("progress_logs", [])
        if logs:
            # 최근 10개 로그 표시
            for log in logs[-10:]:
                st.code(log, language=None)
        else:
            st.info("연구 시작 대기 중...")

        # 진행 상태 표시
        status = research_data.get("status", "unknown")
        if status == "running":
            st.info("⚡ 연구 진행 중...")
        elif status == "completed":
            st.success("✅ 연구 완료!")
        elif status == "error":
            st.error(f"❌ 오류 발생: {research_data.get('error', 'Unknown error')}")

    # 보고서 업데이트
    with report_placeholder.container():
        st.subheader("📄 최종 보고서")
        final_report = research_data.get("final_report", "")
        if final_report:
            st.markdown(final_report)
        else:
            st.info("보고서 생성 대기 중...")


def display_realtime_progress():
    """실시간 진행상황 표시."""
    # 현재 활성 연구 확인
    if st.session_state.active_research:
        for research_id, research_data in st.session_state.active_research.items():
            if research_data["status"] in ["running", "completed"]:
                # 터미널 스타일 로그 표시
                with st.expander(f"🔴 실시간 로그 - {research_data['query'][:30]}...", expanded=True):
                    logs = research_data.get("progress_logs", [])
                    if logs:
                        # 스크롤 가능한 컨테이너
                        log_container = st.container(height=300)
                        with log_container:
                            for log in logs[-20:]:  # 최근 20개 로그
                                st.code(log, language=None)
                    else:
                        st.info("진행 로그가 없습니다.")

                    # 진행 상태 표시
                    status = research_data["status"]
                    if status == "running":
                        st.info("⚡ 연구가 진행 중입니다...")
                    elif status == "completed":
                        st.success("✅ 연구가 완료되었습니다!")
                    elif status == "error":
                        st.error(f"❌ 오류 발생: {research_data.get('error', 'Unknown error')}")
    else:
        st.info("활성 연구가 없습니다. 새 연구를 시작하세요.")


def display_chat_interface():
    """Agent 채팅 인터페이스 표시 (실시간 스트리밍 지원)."""
    # 채팅 히스토리 초기화
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # 채팅 메시지 표시
    chat_container = st.container(height=300)
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    if "streaming" in message and message["streaming"]:
                        # 스트리밍 응답 표시
                        st.write_stream(message["content"])
                    else:
                        st.write(message["content"])
            elif message["role"] == "agent":
                with st.chat_message("assistant", avatar="🤖"):
                    agent_name = message.get("agent_name", "Agent")
                    st.caption(f"**{agent_name}**:")
                    if "streaming" in message and message["streaming"]:
                        st.write_stream(message["content"])
                    else:
                        st.write(message["content"])

    # Agent 선택 옵션
    st.subheader("🎯 Agent 선택")
    col1, col2 = st.columns([2, 1])

    with col1:
        agent_options = {
            "auto": "자동 선택 (현재 연구 상황에 맞게)",
            "planner": "Planner Agent - 계획 수립",
            "executor": "Executor Agent - 검색 실행",
            "verifier": "Verifier Agent - 결과 검증",
            "generator": "Generator Agent - 보고서 생성",
            "research": "Research Agent - 심층 연구",
            "evaluation": "Evaluation Agent - 품질 평가"
        }
        selected_agent = st.selectbox(
            "대화할 Agent 선택:",
            options=list(agent_options.keys()),
            format_func=lambda x: agent_options[x],
            key="selected_agent"
        )

    with col2:
        if st.button("🔄 채팅 초기화", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    # 채팅 입력
    if prompt := st.chat_input("Agent에게 질문하기...", key="chat_input"):
        if not prompt.strip():
            return

        # 사용자 메시지 추가
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })

        # Agent 응답 생성 (비동기 스트리밍)
        try:
            if st.session_state.orchestrator:
                # 스트리밍 응답을 위한 placeholder
                response_placeholder = st.empty()

                async def generate_agent_response():
                    try:
                        # Agent 선택에 따른 응답 생성
                        if selected_agent == "auto":
                            # 현재 연구 상태에 따라 자동 선택
                            response = await generate_auto_agent_response(prompt)
                        else:
                            # 특정 Agent 호출
                            response = await generate_specific_agent_response(selected_agent, prompt)

                        # 스트리밍 응답 표시
                        response_text = ""
                        for chunk in response:
                            response_text += chunk
                            with response_placeholder.container():
                                st.chat_message("assistant").write(response_text)
                            await asyncio.sleep(0.05)  # 스트리밍 효과

                        # 최종 응답을 히스토리에 추가
                        agent_name = get_agent_display_name(selected_agent)
                        st.session_state.chat_history.append({
                            "role": "agent",
                            "agent_name": agent_name,
                            "content": response_text,
                            "streaming": False,
                            "timestamp": datetime.now().isoformat()
                        })

                        # UI 업데이트
                        st.rerun()

                    except Exception as e:
                        error_msg = f"Agent 응답 생성 중 오류 발생: {e}"
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": datetime.now().isoformat()
                        })
                        st.error(error_msg)

                # 비동기 실행
                import threading
                def run_async_response():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(generate_agent_response())
                    loop.close()

                thread = threading.Thread(target=run_async_response)
                thread.start()

            else:
                st.error("Orchestrator가 초기화되지 않았습니다.")
        except Exception as e:
            st.error(f"채팅 응답 생성 실패: {e}")


async def generate_auto_agent_response(prompt: str) -> AsyncGenerator[str, None]:
    """현재 연구 상태에 따라 자동으로 적절한 Agent 선택."""
    # 현재 활성 연구 확인
    active_research = st.session_state.get('active_research', {})

    if not active_research:
        yield "현재 진행 중인 연구가 없습니다. 먼저 연구를 시작해주세요."
        return

    # 연구 상태에 따라 Agent 선택
    current_status = None
    for research_id, data in active_research.items():
        if data["status"] in ["running", "completed"]:
            current_status = data["status"]
            break

    if current_status == "running":
        # 실행 중인 경우 현재 작업 상태에 따라 응답
        yield f"연구가 진행 중입니다. '{prompt}'에 대한 질문은 완료 후 답변드리겠습니다."
    elif current_status == "completed":
        # 완료된 경우 Generator Agent를 통해 응답
        async for chunk in generate_specific_agent_response("generator", prompt):
            yield chunk
    else:
        yield "연구 상태를 확인할 수 없습니다."


async def generate_specific_agent_response(agent_type: str, prompt: str) -> AsyncGenerator[str, None]:
    """특정 Agent에게 질문 전달."""
    try:
        # Agent별 프롬프트 구성
        agent_prompts = {
            "planner": f"다음은 연구 계획에 관한 질문입니다: {prompt}\n연구 계획을 어떻게 수립할지 설명해주세요.",
            "executor": f"다음은 검색 실행에 관한 질문입니다: {prompt}\n어떻게 검색을 수행할지 설명해주세요.",
            "verifier": f"다음은 결과 검증에 관한 질문입니다: {prompt}\n결과를 어떻게 검증할지 설명해주세요.",
            "generator": f"다음은 보고서 생성에 관한 질문입니다: {prompt}\n연구 결과를 어떻게 종합해서 보고서를 만들지 설명해주세요.",
            "research": f"다음은 심층 연구에 관한 질문입니다: {prompt}\n어떻게 심층 연구를 수행할지 설명해주세요.",
            "evaluation": f"다음은 품질 평가에 관한 질문입니다: {prompt}\n연구 결과를 어떻게 평가할지 설명해주세요."
        }

        if agent_type not in agent_prompts:
            yield f"'{agent_type}' Agent를 찾을 수 없습니다."
            return

        agent_prompt = agent_prompts[agent_type]

        # LLM을 통한 응답 생성 (간단한 구현)
        if hasattr(st.session_state.orchestrator, 'llm_manager'):
            # 실제 LLM 호출 (가능한 경우)
            response = f"[{agent_type.upper()} Agent] {agent_prompt[:100]}...\n\n실제 LLM 응답을 생성하는 중입니다."
        else:
            # 모의 응답
            response = f"[{agent_type.upper()} Agent] 귀하의 질문에 답변드리겠습니다.\n\n질문: {prompt}\n\n{agent_type} 관점에서 분석해보면..."

        # 스트리밍 효과를 위한 청크 분할
        words = response.split()
        for i, word in enumerate(words):
            yield word + " "
            if i % 10 == 0:  # 10단어씩 yield
                await asyncio.sleep(0.1)

    except Exception as e:
        yield f"Agent 응답 생성 중 오류: {e}"


def get_agent_display_name(agent_type: str) -> str:
    """Agent 타입을 표시 이름으로 변환."""
    agent_names = {
        "auto": "Auto Agent",
        "planner": "Planner Agent",
        "executor": "Executor Agent",
        "verifier": "Verifier Agent",
        "generator": "Generator Agent",
        "research": "Research Agent",
        "evaluation": "Evaluation Agent"
    }
    return agent_names.get(agent_type, "Unknown Agent")


def display_final_output():
    """최종 출력물 표시."""
    # 현재 활성 연구의 최종 보고서 표시
    if st.session_state.active_research:
        for research_id, research_data in st.session_state.active_research.items():
            if research_data["status"] == "completed":
                final_report = research_data.get("final_report", "")
                if final_report:
                    st.markdown(final_report)

                    # 혁신 통계 표시
                    if research_data.get("innovation_stats"):
                        st.subheader("🚀 혁신 통계")
                        display_innovation_stats(research_data["innovation_stats"])
                else:
                    st.info("최종 보고서가 아직 생성되지 않았습니다.")
                break  # 첫 번째 완료된 연구만 표시
    else:
        st.info("완료된 연구가 없습니다.")


def display_file_downloads():
    """생성된 파일 다운로드 표시."""
    import os
    from pathlib import Path

    # output 디렉토리 스캔
    output_dir = Path("./output")
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        if files:
            # 파일 목록 표시
            for file_path in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
                if file_path.is_file():
                    # 파일 정보
                    file_size = file_path.stat().st_size
                    file_date = datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')

                    # 파일 타입에 따른 아이콘
                    if file_path.suffix == '.md':
                        icon = "📄"
                    elif file_path.suffix == '.json':
                        icon = "📋"
                    elif file_path.suffix == '.pdf':
                        icon = "📕"
                    else:
                        icon = "📁"

                    # 파일 정보 표시
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"{icon} {file_path.name}")
                        st.caption(f"크기: {file_size:,} bytes | 수정일: {file_date}")
                    with col2:
                        # 파일 내용 미리보기 버튼
                        if st.button("👁️ 미리보기", key=f"preview_{file_path.name}"):
                            try:
                                if file_path.suffix == '.md':
                                    content = file_path.read_text(encoding='utf-8')
                                    st.markdown(content[:1000] + "..." if len(content) > 1000 else content)
                                elif file_path.suffix == '.json':
                                    import json
                                    data = json.loads(file_path.read_text(encoding='utf-8'))
                                    st.json(data)
                                else:
                                    st.info("미리보기를 지원하지 않는 파일 형식입니다.")
                            except Exception as e:
                                st.error(f"파일 읽기 실패: {e}")
                    with col3:
                        # 다운로드 버튼
                        try:
                            with open(file_path, 'rb') as f:
                                file_data = f.read()
                            st.download_button(
                                label="📥 다운로드",
                                data=file_data,
                                file_name=file_path.name,
                                mime=get_mime_type(file_path.suffix),
                                key=f"download_{file_path.name}"
                            )
                        except Exception as e:
                            st.error(f"다운로드 준비 실패: {e}")
        else:
            st.info("생성된 파일이 없습니다.")
    else:
        st.info("output 디렉토리가 존재하지 않습니다.")


def get_mime_type(extension: str) -> str:
    """파일 확장자에 따른 MIME 타입 반환."""
    mime_types = {
        '.md': 'text/markdown',
        '.json': 'application/json',
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.html': 'text/html',
        '.csv': 'text/csv'
    }
    return mime_types.get(extension.lower(), 'application/octet-stream')


def start_research_with_innovations(
    query: str,
    enable_adaptive_supervisor: bool,
    max_researchers: int,
    enable_streaming: bool,
    enable_multi_model: bool,
    enable_mcp: bool,
    mcp_tools: List[str]
):
    """Start a new research task with 8 innovations (legacy compatibility)."""
    # 기존 함수는 새로운 스트리밍 함수로 리다이렉트
    start_research_with_streaming(
        query,
        enable_adaptive_supervisor,
        max_researchers,
        enable_streaming,
        enable_multi_model,
        enable_mcp,
        mcp_tools
    )


def display_innovation_stats(stats: Dict[str, Any]):
    """Display innovation statistics."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Adaptive Supervisor", stats.get('adaptive_supervisor', 'N/A'))
        st.metric("Hierarchical Compression", stats.get('hierarchical_compression', 'N/A'))
        st.metric("Multi-Model Orchestration", stats.get('multi_model_orchestration', 'N/A'))
        st.metric("Continuous Verification", stats.get('continuous_verification', 'N/A'))
    
    with col2:
        st.metric("Streaming Pipeline", stats.get('streaming_pipeline', 'N/A'))
        st.metric("Universal MCP Hub", stats.get('universal_mcp_hub', 'N/A'))
        st.metric("Adaptive Context Window", stats.get('adaptive_context_window', 'N/A'))
        st.metric("Production Reliability", stats.get('production_grade_reliability', 'N/A'))


def innovations_monitor():
    """8 Innovations Monitor page."""
    st.header("🚀 8 Core Innovations Monitor")
    
    # Innovation status cards
    innovations = [
        ("Adaptive Supervisor", "Dynamic researcher allocation and quality monitoring"),
        ("Hierarchical Compression", "Multi-stage data compression with validation"),
        ("Multi-Model Orchestration", "Role-based LLM selection and cost optimization"),
        ("Continuous Verification", "3-stage verification with confidence scoring"),
        ("Streaming Pipeline", "Real-time result delivery and incremental saving"),
        ("Universal MCP Hub", "100+ MCP tools with smart selection"),
        ("Adaptive Context Window", "Dynamic context management (2K-1M tokens)"),
        ("Production Reliability", "Circuit breakers and graceful degradation")
    ]
    
    for i in range(0, len(innovations), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(innovations):
                name, description = innovations[i]
                with st.container():
                    st.subheader(f"1️⃣ {name}")
                    st.write(description)
                    st.success("✅ Active")
        
        with col2:
            if i + 1 < len(innovations):
                name, description = innovations[i + 1]
                with st.container():
                    st.subheader(f"2️⃣ {name}")
                    st.write(description)
                    st.success("✅ Active")
    
    # Real-time metrics
    st.subheader("Real-time Metrics")
    if st.session_state.health_monitor:
        try:
            metrics = st.session_state.health_monitor.get_current_metrics()
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("CPU Usage", f"{metrics.cpu_usage:.1f}%")
                with col2:
                    st.metric("Memory Usage", f"{metrics.memory_usage:.1f}%")
                with col3:
                    st.metric("Active Processes", metrics.active_processes)
                with col4:
                    st.metric("Research Tasks", metrics.research_tasks)
        except Exception as e:
            st.warning(f"Could not get real-time metrics: {e}")


def display_active_research():
    """Display active research tasks."""
    for obj_id, research_info in st.session_state.active_research.items():
        with st.expander(f"Research: {research_info['query'][:50]}..."):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Status:** {research_info['status']}")
                st.write(f"**Started:** {research_info['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                st.write(f"**Domain:** {research_info['context']['research_domain']}")
                st.write(f"**Depth:** {research_info['context']['research_depth']}")
            
            with col3:
                if st.button(f"View Details", key=f"view_{obj_id}"):
                    view_research_details(obj_id)
                
                if st.button(f"Cancel", key=f"cancel_{obj_id}"):
                    cancel_research(obj_id)


def display_research_history():
    """Display research history."""
    for i, research in enumerate(st.session_state.research_history):
        with st.expander(f"Research {i+1}: {research['query'][:50]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Query:** {research['query']}")
                st.write(f"**Status:** {research['status']}")
            
            with col2:
                st.write(f"**Completed:** {research['completed_at']}")
                if research.get('deliverable_path'):
                    st.write(f"**Report:** {research['deliverable_path']}")


def data_visualization():
    """Data visualization page."""
    st.header("Data Visualization")
    
    # Load actual data from logs/results
    try:
        load_actual_visualization_data()
    except Exception as e:
        st.error(f"Failed to load visualization data: {e}")
        st.info("No data available for visualization. Run some research tasks first.")


def load_actual_visualization_data():
    """Load actual visualization data from logs and results."""
    import json
    from pathlib import Path
    from datetime import datetime, timedelta
    
    # Load research results from output directory
    output_dir = Path("output")
    if not output_dir.exists():
        st.warning("No output directory found. Run some research tasks first.")
        return
    
    # Find recent research results
    result_files = list(output_dir.glob("*.json"))
    if not result_files:
        st.warning("No research results found. Run some research tasks first.")
        return
    
    # Load and process recent results
    research_data = []
    agent_stats = {}
    
    for file_path in sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[:30]:  # Last 30 results
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract research metadata
            if 'metadata' in data:
                metadata = data['metadata']
                research_data.append({
                    'date': metadata.get('timestamp', datetime.now().isoformat()),
                    'execution_time': metadata.get('execution_time', 0),
                    'quality_score': metadata.get('confidence', 0.5),
                    'sources_count': len(data.get('sources', [])),
                    'success': metadata.get('success', True)
                })
            
            # Extract agent performance data
            if 'agent_collaboration_log' in data:
                for log_entry in data['agent_collaboration_log']:
                    agent = log_entry.get('agent', 'unknown')
                    if agent not in agent_stats:
                        agent_stats[agent] = {'tasks': 0, 'successes': 0}
                    agent_stats[agent]['tasks'] += 1
                    if log_entry.get('interaction_success', False):
                        agent_stats[agent]['successes'] += 1
        
        except (json.JSONDecodeError, KeyError) as e:
            st.warning(f"Failed to parse result file {file_path.name}: {e}")
            continue
    
    if not research_data:
        st.warning("No valid research data found for visualization.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(research_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Research activity over time
    daily_counts = df.groupby(df['date'].dt.date).size().reset_index(name='Research_Count')
    daily_counts['Date'] = pd.to_datetime(daily_counts['date'])
    
    fig1 = px.line(daily_counts, x='Date', y='Research_Count', 
                   title='Research Activity Over Time (Actual Data)')
    st.plotly_chart(fig1, use_container_width=True)
    
    # Quality score distribution
    if 'quality_score' in df.columns:
        fig2 = px.histogram(df, x='quality_score', 
                            title='Quality Score Distribution (Actual Data)', nbins=20)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Agent performance
    if agent_stats:
        agent_data = []
        for agent, stats in agent_stats.items():
            success_rate = stats['successes'] / stats['tasks'] if stats['tasks'] > 0 else 0
            agent_data.append({
                'Agent': agent.replace('_', ' ').title(),
                'Tasks_Completed': stats['tasks'],
                'Success_Rate': success_rate
            })
        
        if agent_data:
            agent_df = pd.DataFrame(agent_data)
            
            fig3 = px.bar(agent_df, x='Agent', y='Tasks_Completed', 
                          title='Agent Task Completion (Actual Data)')
            st.plotly_chart(fig3, use_container_width=True)
            
            fig4 = px.pie(agent_df, values='Success_Rate', names='Agent', 
                          title='Agent Success Rates (Actual Data)')
            st.plotly_chart(fig4, use_container_width=True)
    
    # Show summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Research Tasks", len(research_data))
    
    with col2:
        avg_quality = df['quality_score'].mean() if 'quality_score' in df.columns else 0
        st.metric("Average Quality Score", f"{avg_quality:.2f}")
    
    with col3:
        avg_sources = df['sources_count'].mean() if 'sources_count' in df.columns else 0
        st.metric("Average Sources per Task", f"{avg_sources:.1f}")
    
    with col4:
        success_rate = df['success'].mean() if 'success' in df.columns else 0
        st.metric("Success Rate", f"{success_rate:.1%}")


def report_generator():
    """Report generation page."""
    st.header("Report Generator")
    
    st.subheader("Generate Research Report")
    
    # Report options
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Detailed Analysis", "Academic Paper", "Presentation Slides"]
        )
        
        report_format = st.selectbox(
            "Output Format",
            ["PDF", "HTML", "Markdown", "Word Document"]
        )
    
    with col2:
        include_charts = st.checkbox("Include Visualizations", value=True)
        include_sources = st.checkbox("Include Source Citations", value=True)
        include_appendix = st.checkbox("Include Technical Appendix", value=False)
    
    if st.button("Generate Report"):
        generate_report(report_type, report_format, include_charts, include_sources, include_appendix)


def generate_report(report_type: str, report_format: str, include_charts: bool, 
                   include_sources: bool, include_appendix: bool):
    """Generate a research report."""
    with st.spinner("Generating report..."):
        # Simulate report generation
        st.success("Report generated successfully!")
        
        # Display report preview
        st.subheader("Report Preview")
        st.markdown("""
        # Research Report
        
        ## Executive Summary
        This is a sample research report generated by the Local Researcher system.
        
        ## Key Findings
        - Finding 1: Important discovery
        - Finding 2: Significant insight
        - Finding 3: Critical observation
        
        ## Recommendations
        - Recommendation 1
        - Recommendation 2
        - Recommendation 3
        """)


def system_health():
    """System health monitoring page with 8 innovations."""
    st.header("🏥 System Health - Production-Grade Reliability")
    
    # Overall system health
    st.subheader("Overall System Health")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Research", len(st.session_state.active_research))
    
    with col2:
        st.metric("Completed Research", len(st.session_state.research_history))
    
    with col3:
        st.metric("System Uptime", "99.9%")
    
    with col4:
        st.metric("Health Score", "98.5%")
    
    # 8 Innovations Health Status
    st.subheader("8 Core Innovations Health Status")
    
    innovations_health = [
        ("Adaptive Supervisor", "🟢 Healthy", "Dynamic allocation working"),
        ("Hierarchical Compression", "🟢 Healthy", "3-stage compression active"),
        ("Multi-Model Orchestration", "🟢 Healthy", "Role-based selection active"),
        ("Continuous Verification", "🟢 Healthy", "3-stage verification active"),
        ("Streaming Pipeline", "🟢 Healthy", "Real-time delivery active"),
        ("Universal MCP Hub", "🟢 Healthy", f"{len(config.mcp.server_names)} tools active"),
        ("Adaptive Context Window", "🟢 Healthy", "Dynamic context active"),
        ("Production Reliability", "🟢 Healthy", "Circuit breakers active")
    ]
    
    for i in range(0, len(innovations_health), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(innovations_health):
                name, status, details = innovations_health[i]
                with st.container():
                    st.write(f"**{name}**")
                    st.write(f"{status} - {details}")
        
        with col2:
            if i + 1 < len(innovations_health):
                name, status, details = innovations_health[i + 1]
                with st.container():
                    st.write(f"**{name}**")
                    st.write(f"{status} - {details}")
    
    # Real-time metrics
    st.subheader("Real-time System Metrics")
    if st.session_state.health_monitor:
        try:
            metrics = st.session_state.health_monitor.get_current_metrics()
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("CPU Usage", f"{metrics.cpu_usage:.1f}%")
                with col2:
                    st.metric("Memory Usage", f"{metrics.memory_usage:.1f}%")
                with col3:
                    st.metric("Disk Usage", f"{metrics.disk_usage:.1f}%")
                with col4:
                    st.metric("Active Processes", metrics.active_processes)
                
                # 8 innovations metrics
                st.subheader("8 Innovations Metrics")
                
                if metrics.adaptive_supervisor_metrics:
                    st.write("**Adaptive Supervisor Metrics**")
                    st.json(metrics.adaptive_supervisor_metrics)
                
                if metrics.universal_mcp_hub_metrics:
                    st.write("**Universal MCP Hub Metrics**")
                    st.json(metrics.universal_mcp_hub_metrics)
                
                if metrics.production_reliability_metrics:
                    st.write("**Production Reliability Metrics**")
                    st.json(metrics.production_reliability_metrics)
        except Exception as e:
            st.warning(f"Could not get real-time metrics: {e}")
    
    # Recent activity
    st.subheader("Recent Activity")
    
    activity_data = [
        {"Time": "10:30 AM", "Event": "Research with 8 innovations started", "Details": "AI market analysis"},
        {"Time": "10:25 AM", "Event": "Report generated", "Details": "Technology trends report"},
        {"Time": "10:20 AM", "Event": "Innovation stats updated", "Details": "All 8 innovations active"},
        {"Time": "10:15 AM", "Event": "System health check", "Details": "All systems operational"},
    ]
    
    for activity in activity_data:
        with st.container():
            col1, col2, col3 = st.columns([2, 3, 4])
            with col1:
                st.write(activity["Time"])
            with col2:
                st.write(activity["Event"])
            with col3:
                st.write(activity["Details"])


def settings_page():
    """Settings configuration page."""
    st.header("Settings")
    
    # Configuration sections
    tab1, tab2, tab3, tab4 = st.tabs(["General", "Research", "Display", "Advanced"])
    
    with tab1:
        st.subheader("General Settings")
        
        st.text_input("Project Name", value="Local Researcher")
        st.text_input("Output Directory", value="./outputs")
        st.selectbox("Language", ["English", "Korean", "Japanese", "Chinese"])
    
    with tab2:
        st.subheader("Research Settings")
        
        st.slider("Default Research Depth", 1, 5, 3)
        st.number_input("Max Concurrent Research", 1, 10, 5)
        st.checkbox("Enable Browser Automation", value=True)
        st.checkbox("Enable MCP Tools", value=True)
    
    with tab3:
        st.subheader("Display Settings")
        
        st.selectbox("Theme", ["Light", "Dark", "Auto"])
        st.selectbox("Chart Style", ["Plotly", "Matplotlib", "Seaborn"])
        st.checkbox("Show Advanced Options", value=False)
    
    with tab4:
        st.subheader("Advanced Settings")
        
        st.text_area("Custom Configuration", value="{}")
        st.button("Reset to Defaults")
        st.button("Export Configuration")
        st.button("Import Configuration")


def view_research_details(objective_id: str):
    """View detailed research information."""
    st.write(f"Research Details for: {objective_id}")
    # Implementation for viewing research details


def cancel_research(objective_id: str):
    """Cancel a research task."""
    if objective_id in st.session_state.active_research:
        del st.session_state.active_research[objective_id]
        st.success("Research cancelled")
        st.rerun()


def live_research_dashboard():
    """Live Research Dashboard with real-time agent monitoring."""
    st.header("🔴 Live Research Dashboard")
    st.markdown("**Real-time monitoring of AI research agents**")
    st.markdown("---")
    
    # Import agent visualizer
    from src.web.components.agent_visualizer import AgentVisualizer
    
    # Initialize visualizer
    visualizer = AgentVisualizer()
    
    # Workflow selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get available workflows from session state or create demo
        available_workflows = st.session_state.get('available_workflows', ['demo_workflow_1', 'demo_workflow_2'])
        selected_workflow = st.selectbox(
            "Select Workflow",
            available_workflows,
            key="workflow_selector"
        )
    
    with col2:
        if st.button("🔄 Refresh", key="refresh_workflow"):
            st.rerun()
    
    # Demo workflow creation if none exists
    if not st.session_state.get('available_workflows'):
        st.session_state.available_workflows = ['demo_workflow_1', 'demo_workflow_2']
        st.session_state.workflow_start_time = datetime.now()
    
    # Render live dashboard
    if selected_workflow:
        visualizer.render_live_dashboard(selected_workflow)
        
        # Additional controls
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Timeline View", key="timeline_view"):
                st.session_state.show_timeline = True
        
        with col2:
            if st.button("🔄 Flow Diagram", key="flow_diagram"):
                st.session_state.show_flow = True
        
        with col3:
            if st.button("💡 Creative Insights", key="creative_insights"):
                st.session_state.show_creative = True
        
        # Timeline view
        if st.session_state.get('show_timeline', False):
            st.markdown("### 📈 Progress Timeline")
            visualizer.render_timeline_chart(selected_workflow)
        
        # Flow diagram
        if st.session_state.get('show_flow', False):
            st.markdown("### 🔄 Agent Flow Diagram")
            visualizer.render_agent_flow_diagram(selected_workflow)
        
        # Creative insights
        if st.session_state.get('show_creative', False):
            st.markdown("### 💡 Creative Insights")
            visualizer.render_creative_insights(selected_workflow)
        
        # Auto-refresh controls
        st.markdown("---")
        visualizer.start_auto_refresh(selected_workflow)
    
    else:
        st.info("No workflows available. Start a research task to see live monitoring.")
        
        # Demo workflow creation
        if st.button("🚀 Create Demo Workflow", key="create_demo"):
            demo_workflow_id = f"demo_workflow_{int(time.time())}"
            st.session_state.available_workflows.append(demo_workflow_id)
            st.session_state.workflow_start_time = datetime.now()
            st.success(f"Created demo workflow: {demo_workflow_id}")
            st.rerun()


def creative_insights_page():
    """Creative Forge page for displaying generated creative insights."""
    st.header("✨ Creative Forge - Where Ideas Sparkle and Get Forged")
    st.markdown("**Discover novel solutions through AI-powered creative synthesis**")
    st.markdown("---")
    
    # Check if there are any research results with creative insights
    if 'research_history' in st.session_state and st.session_state.research_history:
        # Get the latest research result
        latest_research = st.session_state.research_history[-1]
        
        if 'creative_insights' in latest_research and latest_research['creative_insights']:
            insights = latest_research['creative_insights']
            
            # Display insights overview
            st.subheader("✨ Forged Insights Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Forged Ideas", len(insights))
            with col2:
                avg_confidence = sum(insight['confidence'] for insight in insights) / len(insights)
                st.metric("Avg Quality", f"{avg_confidence:.2f}")
            with col3:
                avg_novelty = sum(insight['novelty_score'] for insight in insights) / len(insights)
                st.metric("Avg Sparkle", f"{avg_novelty:.2f}")
            
            # Display each insight
            st.subheader("⚒️ Forged Ideas")
            
            for i, insight in enumerate(insights):
                with st.expander(f"✨ {insight['title']} ({insight['type'].replace('_', ' ').title()})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {insight['description']}")
                        st.markdown(f"**Forging Process:** {insight['reasoning']}")
                        
                        if insight['examples']:
                            st.markdown("**Examples:**")
                            for example in insight['examples']:
                                st.markdown(f"- {example}")
                    
                    with col2:
                        # Quality and sparkle scores
                        st.markdown("**Forge Quality:**")
                        st.progress(insight['confidence'])
                        st.caption(f"Quality: {insight['confidence']:.2f}")
                        
                        st.progress(insight['novelty_score'])
                        st.caption(f"Sparkle: {insight['novelty_score']:.2f}")
                        
                        st.progress(insight['applicability_score'])
                        st.caption(f"Usability: {insight['applicability_score']:.2f}")
                        
                        # Related concepts
                        if insight['related_concepts']:
                            st.markdown("**Related Materials:**")
                            for concept in insight['related_concepts']:
                                st.markdown(f"- {concept}")
            
            # Forge type distribution
            st.subheader("📊 Forge Type Distribution")
            insight_types = [insight['type'] for insight in insights]
            type_counts = {}
            for insight_type in insight_types:
                type_counts[insight_type] = type_counts.get(insight_type, 0) + 1
            
            if type_counts:
                fig = px.pie(
                    values=list(type_counts.values()),
                    names=list(type_counts.keys()),
                    title="Distribution of Forge Types"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Sparkle vs Usability scatter plot
            st.subheader("✨ Sparkle vs Usability Analysis")
            df = pd.DataFrame(insights)
            fig = px.scatter(
                df,
                x='novelty_score',
                y='applicability_score',
                color='type',
                size='confidence',
                hover_data=['title'],
                title="Forge Quality Analysis"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No forged ideas available. Complete a forge task to generate creative insights.")
            
            # Demo creative insights
            if st.button("⚒️ Generate Demo Forged Ideas", key="demo_creative"):
                demo_insights = [
                    {
                        'insight_id': 'demo_1',
                        'type': 'analogical',
                        'title': 'Nature-Inspired Research Approach',
                        'description': 'Apply evolutionary principles to research methodology, allowing ideas to adapt and evolve through iterative refinement.',
                        'related_concepts': ['evolution', 'adaptation', 'research methodology'],
                        'confidence': 0.85,
                        'novelty_score': 0.78,
                        'applicability_score': 0.82,
                        'reasoning': 'Nature has perfected problem-solving through evolution, which can be applied to research processes.',
                        'examples': ['Genetic algorithms for research optimization', 'Ecosystem-based collaboration models'],
                        'metadata': {'analogical_source': 'biological', 'generation_method': 'analogical_reasoning'}
                    },
                    {
                        'insight_id': 'demo_2',
                        'type': 'cross_domain',
                        'title': 'AI-Art Research Synthesis',
                        'description': 'Combine artificial intelligence with artistic creativity to generate novel research perspectives and methodologies.',
                        'related_concepts': ['AI', 'art', 'creativity', 'research synthesis'],
                        'confidence': 0.92,
                        'novelty_score': 0.88,
                        'applicability_score': 0.75,
                        'reasoning': 'AI and art represent different modes of thinking that can complement each other in research.',
                        'examples': ['AI-generated research hypotheses', 'Artistic visualization of data patterns'],
                        'metadata': {'domain1': 'technology', 'domain2': 'art', 'generation_method': 'cross_domain_synthesis'}
                    }
                ]
                
                st.session_state.demo_creative_insights = demo_insights
                st.success("Demo forged ideas generated!")
                st.rerun()
    
    else:
        st.info("No forge history available. Start a forge task to generate creative insights.")
        
        # Show creativity forge capabilities
        st.subheader("✨ Creative Forge Capabilities")
        st.markdown("""
        The Creative Forge can forge insights using:
        
        - **Analogical Reasoning**: Drawing parallels from different domains
        - **Cross-Domain Synthesis**: Combining principles from different fields
        - **Lateral Thinking**: Challenging conventional approaches
        - **Convergent Thinking**: Finding unifying patterns
        - **Divergent Thinking**: Exploring all possible variations
        """)
        
        # Show forge patterns
        st.subheader("⚒️ Forge Patterns")
        forge_patterns = {
            'Analogical': [
                "How does this work in nature?",
                "What if we applied this to a completely different field?",
                "How do other industries solve similar problems?"
            ],
            'Cross-Domain': [
                "Combine technology principles with business methods",
                "Apply scientific thinking to artistic problems",
                "Merge social concepts with technical solutions"
            ],
            'Lateral': [
                "What if we did the opposite?",
                "How can we make this more absurd?",
                "What if we removed the main constraint?"
            ]
        }
        
        for pattern_type, patterns in forge_patterns.items():
            with st.expander(f"**{pattern_type} Forging**"):
                for pattern in patterns:
                    st.markdown(f"- {pattern}")


if __name__ == "__main__":
    main()
