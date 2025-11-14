#!/usr/bin/env python3
"""
Streamlit Web Interface for SparkleForge - 인터랙티브 채팅 UI

좌우 분할 레이아웃:
- 왼쪽: 채팅 인터페이스 (사용자 ↔ Agent)
- 오른쪽: 실시간 연구 진행 상황 (Agent 작업 내용)
"""

import streamlit as st
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import threading
import queue
import logging
from io import StringIO

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.agent_orchestrator import AgentOrchestrator, AgentState
from src.core.reliability import HealthMonitor
from src.core.researcher_config import config

import logging
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SparkleForge",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent_activity_log' not in st.session_state:
    st.session_state.agent_activity_log = []
if 'current_research' not in st.session_state:
    st.session_state.current_research = None
if 'research_status' not in st.session_state:
    st.session_state.research_status = "idle"  # idle, running, completed
if 'streaming_queue' not in st.session_state:
    st.session_state.streaming_queue = queue.Queue()
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()
if 'update_flag' not in st.session_state:
    st.session_state.update_flag = False
if 'log_handler' not in st.session_state:
    st.session_state.log_handler = None


class StreamlitLogHandler(logging.Handler):
    """Streamlit UI에 로그를 전달하는 핸들러."""
    
    def __init__(self, queue: queue.Queue):
        super().__init__()
        self.queue = queue
        self.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    def emit(self, record):
        try:
            log_message = self.format(record)
            # 로그 레벨에 따라 Agent 추출
            agent = "system"
            if hasattr(record, 'name') and record.name:
                if 'planner' in record.name.lower():
                    agent = "planner"
                elif 'executor' in record.name.lower():
                    agent = "executor"
                elif 'verifier' in record.name.lower():
                    agent = "verifier"
                elif 'generator' in record.name.lower():
                    agent = "generator"
            
            # 로그 메시지에서 Agent 이름 추출
            if '[PLANNER]' in log_message or '[planner]' in log_message:
                agent = "planner"
            elif '[EXECUTOR]' in log_message or '[executor]' in log_message:
                agent = "executor"
            elif '[VERIFIER]' in log_message or '[verifier]' in log_message:
                agent = "verifier"
            elif '[GENERATOR]' in log_message or '[generator]' in log_message:
                agent = "generator"
            
            # 큐에 추가
            self.queue.put(("log", agent, log_message, "info"))
        except Exception:
            pass  # 로깅 실패는 무시


def initialize_orchestrator():
    """Orchestrator 초기화."""
    try:
        global config
        if config is None:
            try:
                from src.core.researcher_config import load_config_from_env
                config = load_config_from_env()
            except Exception as config_error:
                logger.warning(f"Configuration loading failed: {config_error}")
                from src.core.researcher_config import MCPConfig, ResearcherSystemConfig
                config = ResearcherSystemConfig(
                    llm=None, agent=None, research=None,
                    mcp=MCPConfig(enabled=True, timeout=30, server_names=['g-search', 'tavily', 'exa']),
                    output=None, compression=None, verification=None,
                    context_window=None, reliability=None, agent_tools=None
                )

        if st.session_state.orchestrator is None:
            # Logger 핸들러 설정
            if st.session_state.log_handler is None:
                log_handler = StreamlitLogHandler(st.session_state.streaming_queue)
                log_handler.setLevel(logging.INFO)
                # 모든 관련 logger에 핸들러 추가
                root_logger = logging.getLogger()
                root_logger.addHandler(log_handler)
                # 특정 logger에도 추가
                for logger_name in ['src.core.agent_orchestrator', 'src.core.llm_manager', 'src.core.mcp_integration']:
                    module_logger = logging.getLogger(logger_name)
                    module_logger.addHandler(log_handler)
                    module_logger.setLevel(logging.INFO)
                st.session_state.log_handler = log_handler
            
            st.session_state.orchestrator = AgentOrchestrator()
            logger.info("Orchestrator initialized")
            
    except Exception as e:
        st.error(f"초기화 실패: {e}")
        logger.error(f"Initialization failed: {e}")


def main():
    """메인 애플리케이션 - 좌우 분할 레이아웃."""
    st.title("⚒️ SparkleForge - Multi-Agent Research System")
    st.markdown("---")
    
    # Orchestrator 초기화
    initialize_orchestrator()
    
    if st.session_state.orchestrator is None:
        st.error("⚠️ Orchestrator가 초기화되지 않았습니다.")
        return
    
    # 좌우 분할 레이아웃
    col_left, col_right = st.columns([1, 1], gap="medium")
    
    with col_left:
        chat_interface()
    
    with col_right:
        activity_panel()


def chat_interface():
    """왼쪽: 채팅 인터페이스."""
    # 큐에서 업데이트 처리
    process_streaming_queue()
    
    st.header("💬 Agent와 대화하기")
    
    # 연구 시작 버튼
    with st.expander("🔍 새 연구 시작", expanded=False):
        research_query = st.text_area(
            "연구 주제",
            placeholder="예: 인공지능의 최신 동향",
            height=80,
            key="research_query_input"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 연구 시작", type="primary", use_container_width=True):
                if research_query.strip():
                    start_research(research_query)
                else:
                    st.warning("연구 주제를 입력해주세요.")
        with col2:
            if st.button("🔄 초기화", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.agent_activity_log = []
                st.session_state.current_research = None
                st.session_state.research_status = "idle"
                st.rerun()
    
    st.markdown("---")
    
    # 채팅 히스토리 표시
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
                    if msg.get("timestamp"):
                        st.caption(msg["timestamp"])
            elif msg["role"] == "agent":
                with st.chat_message("assistant", avatar="🤖"):
                    agent_name = msg.get("agent_name", "Agent")
                    st.caption(f"**{agent_name}**")
                    st.write(msg["content"])
                    if msg.get("timestamp"):
                        st.caption(msg["timestamp"])
            elif msg["role"] == "system":
                st.info(f"ℹ️ {msg['content']}")
    
    # 채팅 입력
    if prompt := st.chat_input("Agent에게 질문하거나 연구를 시작하세요..."):
        if prompt.strip():
            handle_user_message(prompt)
            # 즉시 rerun하여 UI 업데이트
            st.rerun()


def process_streaming_queue():
    """스트리밍 큐에서 업데이트 처리."""
    try:
        while not st.session_state.streaming_queue.empty():
            update = st.session_state.streaming_queue.get_nowait()
            update_type = update[0]
            
            if update_type == "log":
                _, agent, message, activity_type = update
                add_activity_log(agent, message, activity_type)
            elif update_type == "chat":
                _, role, agent_name, content = update
                st.session_state.chat_history.append({
                    "role": role,
                    "agent_name": agent_name,
                    "content": content,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            elif update_type == "status":
                _, status = update
                st.session_state.research_status = status
            elif update_type == "save":
                _, query, report, session_id = update
                save_research_result(query, report, session_id)
    except queue.Empty:
        pass
    except Exception as e:
        logger.error(f"Failed to process streaming queue: {e}")


def activity_panel():
    """오른쪽: 실시간 Agent 활동 패널."""
    st.header("🔴 실시간 Agent 활동")
    
    # 큐에서 업데이트 처리
    process_streaming_queue()
    
    # 상태 표시
    status_colors = {
        "idle": "⚪",
        "running": "🟢",
        "completed": "✅",
        "error": "🔴"
    }
    status_icon = status_colors.get(st.session_state.research_status, "⚪")
    st.markdown(f"**상태:** {status_icon} {st.session_state.research_status.upper()}")
    
    st.markdown("---")
    
    # Agent 활동 로그
    activity_container = st.container(height=550)
    with activity_container:
        if st.session_state.agent_activity_log:
            # 최근 활동부터 표시
            for activity in reversed(st.session_state.agent_activity_log[-50:]):  # 최근 50개
                agent_name = activity.get("agent", "Unknown")
                activity_type = activity.get("type", "info")
                message = activity.get("message", "")
                timestamp = activity.get("timestamp", "")
                
                # Agent별 색상
                agent_colors = {
                    "planner": "🔵",
                    "executor": "🟢",
                    "verifier": "🟡",
                    "generator": "🟣"
                }
                agent_icon = agent_colors.get(agent_name.lower(), "🤖")
                
                # 활동 타입별 스타일
                # 로그 메시지가 너무 길면 자르기
                display_message = message
                if len(display_message) > 200:
                    display_message = display_message[:200] + "..."
                
                if activity_type == "start":
                    st.success(f"{agent_icon} **[{agent_name.upper()}]** 시작: {display_message}")
                elif activity_type == "progress":
                    st.info(f"{agent_icon} **[{agent_name.upper()}]** 진행: {display_message}")
                elif activity_type == "complete":
                    st.success(f"{agent_icon} **[{agent_name.upper()}]** 완료: {display_message}")
                elif activity_type == "error":
                    st.error(f"{agent_icon} **[{agent_name.upper()}]** 오류: {display_message}")
                else:
                    # 일반 로그는 코드 블록으로 표시
                    st.code(f"[{agent_name.upper()}] {display_message}", language=None)
                
                if timestamp:
                    st.caption(timestamp)
                st.markdown("---")
        else:
            st.info("Agent 활동이 없습니다. 연구를 시작하거나 Agent와 대화해보세요.")
    
    # 자동 새로고침 (연구 진행 중이거나 큐에 업데이트가 있을 때)
    if st.session_state.research_status == "running" or not st.session_state.streaming_queue.empty():
        # 큐에 업데이트가 있으면 즉시 새로고침
        if not st.session_state.streaming_queue.empty():
            time.sleep(0.5)
            st.rerun()
        elif st.session_state.research_status == "running":
            # 업데이트가 없어도 주기적으로 확인 (2초마다)
            current_time = time.time()
            if current_time - st.session_state.last_update_time > 2:
                st.session_state.last_update_time = current_time
                time.sleep(1)
                st.rerun()


def add_activity_log(agent: str, message: str, activity_type: str = "info"):
    """Agent 활동 로그 추가."""
    try:
        log_entry = {
            "agent": agent,
            "message": message,
            "type": activity_type,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.agent_activity_log.append(log_entry)
        # 최대 100개까지만 유지
        if len(st.session_state.agent_activity_log) > 100:
            st.session_state.agent_activity_log = st.session_state.agent_activity_log[-100:]
        # 업데이트 플래그 설정
        st.session_state.update_flag = True
        st.session_state.last_update_time = time.time()
    except Exception as e:
        # 스레드에서 호출될 수 있으므로 안전하게 처리
        logger.error(f"Failed to add activity log: {e}")


def handle_user_message(prompt: str):
    """사용자 메시지 처리."""
    # 사용자 메시지 추가
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    
    # 연구 시작 명령인지 확인
    if prompt.lower().startswith("연구:") or prompt.lower().startswith("research:"):
        query = prompt.split(":", 1)[1].strip() if ":" in prompt else prompt
        start_research(query)
    else:
        # 일반 채팅 - 자동으로 적절한 Agent 선택
        handle_chat_message(prompt)
    
    st.rerun()


def start_research(query: str):
    """연구 시작."""
    try:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        st.session_state.current_research = {
            "query": query,
            "session_id": session_id,
            "start_time": datetime.now()
        }
        st.session_state.research_status = "running"
        
        # 시스템 메시지 추가
        st.session_state.chat_history.append({
            "role": "system",
            "content": f"연구 시작: {query}",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        add_activity_log("system", f"연구 시작: {query}", "start")
        
        # 비동기 실행
        def run_research():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(execute_research_stream(query, session_id))
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_research, daemon=True)
        thread.start()
        
        # 즉시 rerun하여 UI 업데이트
        st.rerun()
        
    except Exception as e:
        st.error(f"연구 시작 실패: {e}")
        logger.error(f"Research start failed: {e}")
        st.session_state.research_status = "error"
        add_activity_log("system", f"오류: {str(e)}", "error")


async def execute_research_stream(query: str, session_id: str):
    """연구 실행 (스트리밍) - 실시간 로그 업데이트."""
    try:
        orchestrator = st.session_state.orchestrator
        if not orchestrator:
            st.session_state.streaming_queue.put(("log", "system", "Orchestrator가 없습니다", "error"))
            return
        
        # 큐에 업데이트 추가 (스레드 안전)
        st.session_state.streaming_queue.put(("log", "system", f"연구 시작: {query}", "start"))
        st.session_state.streaming_queue.put(("log", "system", "Orchestrator 초기화 완료", "start"))
        
        # 스트리밍 실행
        event_count = 0
        async for state_update in orchestrator.stream(query, session_id=session_id):
            event_count += 1
            st.session_state.streaming_queue.put(("log", "system", f"이벤트 수신: {event_count}", "progress"))
            
            if isinstance(state_update, dict):
                # 각 노드의 상태 확인
                for node_name, node_state in state_update.items():
                    if isinstance(node_state, dict):
                        # Agent 식별
                        current_agent = node_state.get('current_agent') or node_name
                        
                        # 노드 이름으로 Agent 추정
                        if 'planner' in node_name.lower():
                            current_agent = "planner"
                        elif 'executor' in node_name.lower():
                            current_agent = "executor"
                        elif 'verifier' in node_name.lower():
                            current_agent = "verifier"
                        elif 'generator' in node_name.lower():
                            current_agent = "generator"
                        else:
                            current_agent = "system"
                        
                        # 진행 중인 Agent 표시
                        if node_name not in ["__start__", "__end__"]:
                            st.session_state.streaming_queue.put(("log", current_agent, f"[{node_name}] 노드 실행 중", "progress"))
                        
                        # 연구 계획 생성
                        if node_state.get('research_plan'):
                            plan = node_state['research_plan']
                            st.session_state.streaming_queue.put(("log", "planner", f"연구 계획 생성 완료 ({len(plan)}자)", "complete"))
                            # 채팅에 계획 추가
                            st.session_state.streaming_queue.put(("chat", "agent", "Planner", f"연구 계획을 수립했습니다:\n\n{plan[:500]}..."))
                        
                        # 검색 결과
                        if node_state.get('research_results'):
                            results = node_state['research_results']
                            if isinstance(results, list) and len(results) > 0:
                                st.session_state.streaming_queue.put(("log", "executor", f"{len(results)}개 검색 결과 수집 완료", "complete"))
                            elif results:
                                st.session_state.streaming_queue.put(("log", "executor", f"검색 결과 수집 완료", "complete"))
                        
                        # 검증 결과
                        if node_state.get('verified_results'):
                            verified = node_state['verified_results']
                            if isinstance(verified, list) and len(verified) > 0:
                                st.session_state.streaming_queue.put(("log", "verifier", f"{len(verified)}개 결과 검증 완료", "complete"))
                            elif verified:
                                st.session_state.streaming_queue.put(("log", "verifier", f"검증 완료", "complete"))
                        
                        # 최종 보고서
                        final_report = node_state.get('final_report')
                        if final_report:
                            st.session_state.streaming_queue.put(("log", "generator", f"최종 보고서 생성 완료 ({len(final_report)}자)", "complete"))
                            st.session_state.streaming_queue.put(("status", "completed"))
                            # 채팅에 보고서 추가
                            st.session_state.streaming_queue.put(("chat", "agent", "Generator", f"연구 보고서가 완성되었습니다:\n\n{final_report[:1000]}..."))
                            # 결과 저장
                            st.session_state.streaming_queue.put(("save", query, final_report, session_id))
                        
                        # 에러 확인
                        if node_state.get('error'):
                            error_msg = node_state['error']
                            st.session_state.streaming_queue.put(("log", current_agent, f"오류: {error_msg}", "error"))
        
        # 완료 처리
        if st.session_state.research_status == "running":
            st.session_state.streaming_queue.put(("status", "completed"))
            st.session_state.streaming_queue.put(("log", "system", f"연구 완료 (총 {event_count}개 이벤트)", "complete"))
        
    except Exception as e:
        logger.error(f"Research execution failed: {e}")
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"Error details: {error_detail}")
        st.session_state.streaming_queue.put(("status", "error"))
        st.session_state.streaming_queue.put(("log", "system", f"오류 발생: {str(e)}", "error"))
        # 채팅에 오류 메시지 추가
        st.session_state.streaming_queue.put(("chat", "system", None, f"❌ 오류 발생: {str(e)}"))


def save_research_result(query: str, report: str, session_id: str):
    """연구 결과 저장."""
    try:
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)
        
        filename = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# 연구 보고서\n\n")
            f.write(f"**주제:** {query}\n\n")
            f.write(f"**생성 시간:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**세션 ID:** {session_id}\n\n")
            f.write("---\n\n")
            f.write(report)
        
        add_activity_log("system", f"결과 저장: {filename}", "complete")
    except Exception as e:
        logger.error(f"Failed to save research result: {e}")


def handle_chat_message(prompt: str):
    """일반 채팅 메시지 처리."""
    # Agent 선택 (간단한 휴리스틱)
    agent_type = "planner"  # 기본값
    
    if any(keyword in prompt.lower() for keyword in ["검색", "찾아", "search", "find"]):
        agent_type = "executor"
    elif any(keyword in prompt.lower() for keyword in ["검증", "확인", "verify", "check"]):
        agent_type = "verifier"
    elif any(keyword in prompt.lower() for keyword in ["보고서", "생성", "report", "generate"]):
        agent_type = "generator"
    
    # 활동 로그 추가
    st.session_state.streaming_queue.put(("log", agent_type, f"질문 처리 중: {prompt[:50]}...", "progress"))
    
    # Agent 응답 생성
    def generate_response():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(get_agent_response(prompt, agent_type))
            
            # 큐에 응답 추가
            st.session_state.streaming_queue.put(("chat", "agent", agent_type.upper(), response))
            st.session_state.streaming_queue.put(("log", agent_type, "응답 생성 완료", "complete"))
        except Exception as e:
            error_msg = f"⚠️ 오류 발생: {str(e)}"
            st.session_state.streaming_queue.put(("chat", "agent", agent_type.upper(), error_msg))
            st.session_state.streaming_queue.put(("log", agent_type, f"오류: {str(e)}", "error"))
        finally:
            loop.close()
    
    thread = threading.Thread(target=generate_response, daemon=True)
    thread.start()


async def get_agent_response(prompt: str, agent_type: str) -> str:
    """Agent 응답 가져오기 - LLM을 직접 호출하여 간단한 응답 생성."""
    try:
        orchestrator = st.session_state.orchestrator
        if not orchestrator:
            return "⚠️ Orchestrator가 초기화되지 않았습니다."
        
        st.session_state.streaming_queue.put(("log", agent_type, "응답 생성 시작", "start"))
        
        # LLM을 직접 호출하여 Agent 역할에 맞는 응답 생성
        from src.core.llm_manager import execute_llm_task, TaskType
        
        # Agent별 프롬프트 구성
        agent_prompts = {
            "planner": f"""당신은 연구 계획 수립 전문가입니다. 사용자의 질문에 대해 연구 계획을 수립하는 방법을 설명해주세요.

질문: {prompt}

연구 계획 수립 방법을 단계별로 설명해주세요.""",
            "executor": f"""당신은 정보 검색 전문가입니다. 사용자의 질문에 대해 효과적인 검색 방법을 설명해주세요.

질문: {prompt}

효과적인 검색 방법과 전략을 설명해주세요.""",
            "verifier": f"""당신은 정보 검증 전문가입니다. 사용자의 질문에 대해 정보를 검증하는 방법을 설명해주세요.

질문: {prompt}

정보 검증 방법과 팩트 체크 전략을 설명해주세요.""",
            "generator": f"""당신은 보고서 작성 전문가입니다. 사용자의 질문에 대해 보고서를 작성하는 방법을 설명해주세요.

질문: {prompt}

효과적인 보고서 작성 방법과 구조를 설명해주세요."""
        }
        
        agent_prompt = agent_prompts.get(agent_type, f"질문: {prompt}\n\n이 질문에 대해 답변해주세요.")
        
        st.session_state.streaming_queue.put(("log", agent_type, "LLM 호출 중...", "progress"))
        
        # LLM 실행
        result = await execute_llm_task(
            prompt=agent_prompt,
            task_type=TaskType.PLANNING if agent_type == "planner" else TaskType.GENERATION,
            model_name=None,
            system_message=None
        )
        
        response = result.content if result.content else f"[{agent_type.upper()}] 응답을 생성하지 못했습니다."
        
        st.session_state.streaming_queue.put(("log", agent_type, "응답 생성 완료", "complete"))
        
        return response
        
    except Exception as e:
        logger.error(f"Agent response failed: {e}")
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"Error details: {error_detail}")
        return f"⚠️ 오류 발생: {str(e)}\n\n자세한 내용은 로그를 확인해주세요."


if __name__ == "__main__":
    main()
