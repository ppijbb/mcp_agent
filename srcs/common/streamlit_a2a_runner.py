"""
Streamlit용 A2A Agent 실행 헬퍼

Streamlit 페이지에서 A2A를 통해 agent를 실행하는 공통 함수
"""

import streamlit as st
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from srcs.common.standard_agent_runner import StandardAgentRunner
from srcs.common.agent_interface import AgentType, AgentMetadata
from srcs.common.a2a_integration import get_global_registry

logger = logging.getLogger(__name__)


def _detect_agent_type(entry_point: str) -> str:
    """entry_point를 분석하여 agent 타입 자동 판단"""
    if entry_point.startswith("lang_graph.") or "lang_graph/" in entry_point:
        return AgentType.LANGGRAPH_AGENT.value
    elif entry_point.startswith("cron_agents.") or "cron_agents/" in entry_point:
        return AgentType.CRON_AGENT.value
    elif entry_point.startswith("sparkleforge.") or "sparkleforge/" in entry_point:
        return AgentType.SPARKLEFORGE_AGENT.value
    else:
        return AgentType.MCP_AGENT.value


def _normalize_entry_point(entry_point: str) -> str:
    """entry_point를 정규화 (CLI 명령에서 모듈 경로 추출)"""
    # "python -m srcs.enterprise_agents.esg_carbon_neutral_agent" -> "srcs.enterprise_agents.esg_carbon_neutral_agent"
    if entry_point.startswith("python -m "):
        return entry_point.replace("python -m ", "")
    # "srcs/enterprise_agents/esg_carbon_neutral_agent.py" -> "srcs.enterprise_agents.esg_carbon_neutral_agent"
    if entry_point.endswith(".py"):
        entry_point = entry_point.replace(".py", "")
    entry_point = entry_point.replace("/", ".")
    return entry_point


def run_agent_via_a2a(
    placeholder,
    agent_metadata: Dict[str, Any],
    input_data: Dict[str, Any],
    result_json_path: Optional[Path] = None,
    use_a2a: bool = True,
    log_expander_title: str = "🤖 A2A Agent 실행 중..."
) -> Optional[Dict[str, Any]]:
    """
    A2A를 통해 agent를 실행하는 Streamlit 헬퍼 함수

    Args:
        placeholder: Streamlit placeholder 컨테이너
        agent_metadata: Agent 메타데이터 딕셔너리
            - agent_id: Agent ID (필수)
            - agent_name: Agent 이름 (필수)
            - entry_point: 실행 경로 (필수, 모듈 경로 또는 CLI 명령)
            - agent_type: Agent 타입 (선택, 자동 판단됨)
            - capabilities: Agent 능력 목록 (선택)
            - description: 설명 (선택)
        input_data: Agent에 전달할 입력 데이터
        result_json_path: 결과를 저장할 JSON 파일 경로 (선택)
        use_a2a: A2A 사용 여부 (기본값: True)
        log_expander_title: 로그 제목

    Returns:
        성공 시 결과 데이터(dict), 실패 시 None
    """
    if placeholder is None:
        st.error("결과를 표시할 UI 컨테이너가 지정되지 않았습니다.")
        return None

    # 필수 필드 검증
    agent_id = agent_metadata.get("agent_id")
    agent_name = agent_metadata.get("agent_name")
    entry_point = agent_metadata.get("entry_point")

    if not agent_id or not agent_name or not entry_point:
        st.error("❌ agent_metadata에 agent_id, agent_name, entry_point가 필요합니다.")
        return None

    # entry_point 정규화
    entry_point = _normalize_entry_point(entry_point)

    # agent_type 자동 판단 (지정되지 않은 경우)
    raw_agent_type = agent_metadata.get("agent_type")
    if not raw_agent_type:
        raw_agent_type = _detect_agent_type(entry_point)

    # AgentType enum 또는 문자열을 문자열로 정규화
    if isinstance(raw_agent_type, AgentType):
        agent_type_str = raw_agent_type.value
        agent_type_enum = raw_agent_type
    elif isinstance(raw_agent_type, str):
        try:
            agent_type_enum = AgentType(raw_agent_type)
            agent_type_str = raw_agent_type
        except ValueError:
            st.error(f"❌ 잘못된 agent_type: {raw_agent_type}")
            logger.error(f"Invalid agent_type: {raw_agent_type}")
            return None
    else:
        st.error(f"❌ agent_type이 올바른 형식이 아닙니다: {type(raw_agent_type)} - {raw_agent_type}")
        logger.error(f"Invalid agent_type type: {type(raw_agent_type)}, value: {raw_agent_type}")
        return None

    logger.info(f"Registering agent {agent_id} with type {agent_type_str} (from {raw_agent_type})")

    metadata = AgentMetadata(
        agent_id=agent_id,
        agent_name=agent_name,
        agent_type=agent_type_enum,
        description=agent_metadata.get("description", ""),
        capabilities=agent_metadata.get("capabilities", []),
        entry_point=entry_point,
    )

    # Streamlit 세션 상태에서 runner 초기화
    if "a2a_runner" not in st.session_state:
        st.session_state.a2a_runner = StandardAgentRunner()

    runner = st.session_state.a2a_runner
    registry = get_global_registry()

    with placeholder.container():
        # A2A 통신 상태 표시를 위한 UI 요소
        status_placeholder = st.empty()

        # 사이드바에 A2A 통신 로그 표시 (전체 agent 공통)
        log_key = "a2a_log_global"  # 전체 agent 공통 로그 키
        if log_key not in st.session_state:
            st.session_state[log_key] = []

        # 사이드바에 A2A 로그 섹션 생성 (한 번만 생성)
        if "a2a_log_sidebar_initialized" not in st.session_state:
            with st.sidebar:
                st.markdown("### 📡 A2A 통신 로그")
                st.session_state.a2a_log_sidebar_placeholder = st.empty()
            st.session_state.a2a_log_sidebar_initialized = True

        def update_log_ui():
            """로그 UI 업데이트 (사이드바) - 스크롤 가능"""
            # 로그 텍스트 생성
            log_text = ""
            for log_entry in st.session_state[log_key][-100:]:  # 최근 100개만 표시
                timestamp, icon, message = log_entry
                log_text += f"[{timestamp}] {icon} {message}\n"

            # placeholder를 직접 업데이트 (container() 사용하지 않음 - 중복 생성 방지)
            # st.code()는 스크롤 가능하고 읽기 전용이며 key가 필요 없음
            st.session_state.a2a_log_sidebar_placeholder.code(
                log_text if log_text else "로그가 없습니다.",
                language="text"
                )

        def log_a2a_message(message: str, level: str = "info"):
            """A2A 메시지 로그를 세션 상태에 저장"""
            timestamp = datetime.now().strftime("%H:%M:%S")
            icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(level, "ℹ️")
            st.session_state[log_key].append((timestamp, icon, message))

            # 로그가 너무 길어지면 오래된 것 제거 (최대 200개 유지)
            if len(st.session_state[log_key]) > 200:
                st.session_state[log_key] = st.session_state[log_key][-200:]

            logger.info(f"A2A [{level}]: {message}")
            # UI 업데이트 시도 (비동기에서는 제한적)
            try:
                update_log_ui()
            except Exception:
                pass  # 비동기 컨텍스트에서는 실패할 수 있음

        try:
            # Streamlit UI agent 등록 및 메시지 수신 대기
            async def register_and_run():
                from srcs.common.a2a_adapter import CommonAgentA2AWrapper
                from srcs.common.a2a_integration import get_global_broker, A2AMessage, MessagePriority
                import uuid

                streamlit_agent_id = "streamlit_ui"
                response_received = False
                response_data = None
                correlation_id = str(uuid.uuid4())

                # 상태 업데이트 함수
                def update_status(text: str):
                    try:
                        status_placeholder.info(f"🔄 {text}")
                    except Exception:
                        pass  # 비동기 컨텍스트에서는 실패할 수 있음

                def update_log(message: str, level: str = "info"):
                    log_a2a_message(message, level)

                update_status("A2A 통신 초기화 중...")
                update_log("A2A 통신 시작", "info")

                # Streamlit UI agent 등록 (페이지 전환 시 기존 agent 정리 후 재등록)
                existing_ui_agent = await registry.get_agent(streamlit_agent_id)

                # 기존 agent가 있으면 정리 (다른 event loop에 바인딩된 queue 정리)
                if existing_ui_agent:
                    old_wrapper = existing_ui_agent.get("a2a_adapter")
                    if old_wrapper:
                        try:
                            # 기존 listener 중지 (안전하게 처리)
                            try:
                                await asyncio.wait_for(old_wrapper.stop_listener(), timeout=1.0)
                            except (asyncio.TimeoutError, RuntimeError, AttributeError):
                                # Event loop가 닫혀있거나 타임아웃된 경우 무시
                                logger.debug("기존 listener 중지 중 타임아웃/에러 (예상됨)")

                            # 기존 agent 등록 해제
                            try:
                                await registry.unregister_agent(streamlit_agent_id)
                            except Exception as e:
                                logger.debug(f"Agent 등록 해제 중 에러 (무시): {e}")

                            update_log("기존 Streamlit UI Agent 정리 완료", "info")
                        except Exception as e:
                            # 모든 예외를 안전하게 처리하여 Streamlit이 크래시되지 않도록
                            logger.debug(f"기존 agent 정리 중 오류 (무시): {e}")

                # 새 Streamlit UI agent 등록 (현재 event loop에서)
                ui_wrapper = None  # 초기화
                if use_a2a:
                    update_status("Streamlit UI Agent 등록 중...")
                    update_log("Streamlit UI Agent 등록 시작", "info")

                    ui_metadata = {
                        "agent_id": streamlit_agent_id,
                        "agent_name": "Streamlit UI",
                        "agent_type": "streamlit_ui",
                        "capabilities": ["ui_display", "result_receiving"],
                        "description": "Streamlit UI for receiving agent results"
                    }
                    ui_wrapper = CommonAgentA2AWrapper(streamlit_agent_id, ui_metadata)

                    # task_response 메시지 핸들러 등록
                    async def handle_task_response(message: A2AMessage) -> Optional[Dict[str, Any]]:
                        """task_response 메시지 처리"""
                        nonlocal response_received, response_data
                        if message.correlation_id == correlation_id:
                            response_data = message.payload
                            response_received = True
                            update_log(f"✅ task_response 수신: {message.message_id} (correlation_id: {correlation_id})", "success")
                            logger.info(f"Streamlit UI received task response: {message.message_id}")
                        else:
                            update_log(f"⚠️ 다른 correlation_id의 메시지 수신: {message.correlation_id} (기대: {correlation_id})", "warning")
                        return message.payload

                    ui_wrapper.register_handler("task_response", handle_task_response)

                    # notification 메시지 핸들러 등록
                    async def handle_notification(message: A2AMessage) -> Optional[Dict[str, Any]]:
                        """notification 메시지 처리"""
                        if message.correlation_id == correlation_id:
                            payload = message.payload
                            if payload.get("type") == "log":
                                msg = payload.get("message", "")
                                if msg:
                                    update_status(msg)
                                    update_log(f"📋 {msg}", "info")
                        return None

                    ui_wrapper.register_handler("notification", handle_notification)
                    await ui_wrapper.start_listener()
                    await registry.register_agent(
                        agent_id=streamlit_agent_id,
                        agent_type="streamlit_ui",
                        metadata=ui_metadata,
                        a2a_adapter=ui_wrapper
                    )
                    update_log(f"✅ Streamlit UI Agent 등록 완료: {streamlit_agent_id}", "success")
                    logger.info(f"Streamlit UI agent registered: {streamlit_agent_id}")

                # Target agent 등록 및 A2A adapter 설정
                update_status(f"Target Agent ({agent_id}) 등록 및 A2A Adapter 설정 중...")
                update_log(f"Target Agent 확인: {agent_id}", "info")

                existing_agent = await registry.get_agent(agent_id)
                # A2A를 사용하는 경우 기존 agent가 있어도 새로 등록 (핸들러가 제대로 등록되도록)
                # 기존 agent가 있고 adapter도 있으면 먼저 unregister
                if existing_agent and use_a2a:
                    update_log(f"기존 agent {agent_id} 발견, 재등록을 위해 해제 중...", "info")
                    try:
                        await registry.unregister_agent(agent_id)
                        existing_agent = None  # 새로 등록하도록 설정
                    except Exception as e:
                        logger.warning(f"기존 agent 해제 실패 (무시): {e}")

                if not existing_agent:
                    metadata_dict = metadata.to_dict()
                    if "agent_type" in metadata_dict and isinstance(metadata_dict["agent_type"], AgentType):
                        metadata_dict["agent_type"] = metadata_dict["agent_type"].value

                    logger.debug(f"Registering agent {agent_id} with type {agent_type_str}")

                    # A2A를 사용하는 경우 adapter를 미리 생성하고 등록
                    if use_a2a:
                        update_log(f"A2A Adapter 생성 중: {agent_id}", "info")
                        agent_wrapper = CommonAgentA2AWrapper(agent_id, metadata_dict)
                        await agent_wrapper.start_listener()
                        await agent_wrapper.register_capabilities(metadata_dict.get("capabilities", []))
                        update_log(f"✅ A2A Adapter 생성 완료, Listener 시작됨: {agent_id}", "success")

                        # task_request 메시지 핸들러 등록
                        async def handle_task_request(message: A2AMessage) -> Optional[Dict[str, Any]]:
                            """task_request 메시지 처리"""
                            update_log(f"📨 task_request 수신: {message.message_id} from {message.source_agent}", "info")
                            logger.info(f"Agent {agent_id} received task request: {message.message_id}")
                            task_data = message.payload.get("task_data", {})
                            task_start_time = datetime.now()

                            update_log(f"🚀 Agent 실행 시작: {agent_id}", "info")

                            # 직접 agent 실행 (레지스트리 조회 없이)
                            # entry_point는 metadata에서 가져오기
                            entry_point = metadata_dict.get("entry_point", "")

                            from srcs.common.a2a_adapter import A2ALogHandler, current_correlation_id
                            log_handler = A2ALogHandler(agent_wrapper, correlation_id=message.correlation_id)
                            log_handler.setLevel(logging.INFO)
                            root_logger = logging.getLogger()
                            root_logger.addHandler(log_handler)

                            # ContextVar 설정
                            token = current_correlation_id.set(message.correlation_id)

                            try:
                                # class-based agent인지 확인
                                is_class_based = "module_path" in task_data and "class_name" in task_data

                                if is_class_based:
                                    # class-based agent는 _run_module_agent 직접 호출
                                    update_log(f"📦 Class-based agent 감지: {task_data.get('class_name')}", "info")
                                    exec_result = await runner._run_module_agent(entry_point, task_data)
                                else:
                                    # 실행 방식 결정
                                    execution_method = task_data.get("_execution_method")
                                    if execution_method == "cli" or entry_point.startswith("python -m") or entry_point.endswith(".py") or "/" in entry_point:
                                        exec_result = await runner._run_cli_agent(entry_point, task_data)
                                    else:
                                        exec_result = await runner._run_module_agent(entry_point, task_data)

                                execution_time = (datetime.now() - task_start_time).total_seconds()
                                exec_result.execution_time = execution_time

                            except Exception as e:
                                logger.error(f"Error executing agent task: {e}", exc_info=True)
                                execution_time = (datetime.now() - task_start_time).total_seconds()
                                from srcs.common.agent_interface import AgentExecutionResult
                                exec_result = AgentExecutionResult(
                                    success=False,
                                    error=str(e),
                                    execution_time=execution_time,
                                    metadata={"agent_id": agent_id, "message_id": message.message_id, "exception": str(e)}
                                )
                            finally:
                                root_logger.removeHandler(log_handler)
                                current_correlation_id.reset(token)

                            update_log(f"✅ Agent 실행 완료: {agent_id} (성공: {exec_result.success})", "success" if exec_result.success else "error")

                            # 결과를 A2A 메시지로 전송
                            response_payload = {
                                "success": exec_result.success,
                                "data": exec_result.data,
                                "error": exec_result.error,
                                "execution_time": exec_result.execution_time,
                                "metadata": exec_result.metadata,
                                "timestamp": exec_result.timestamp.isoformat(),
                            }

                            update_log(f"📤 task_response 전송 중: {message.source_agent}", "info")
                            await agent_wrapper.send_message(
                                target_agent=message.source_agent,
                                message_type="task_response",
                                payload=response_payload,
                                correlation_id=message.correlation_id,  # 원래 요청의 correlation_id 사용
                                priority=MessagePriority.HIGH.value
                            )

                            update_log(f"✅ task_response 전송 완료: {message.correlation_id}", "success")
                            logger.info(f"Agent {agent_id} sent task response: {message.message_id}")
                            return response_payload

                        agent_wrapper.register_handler("task_request", handle_task_request)
                        update_log("task_request 핸들러 등록 완료", "info")

                        await registry.register_agent(
                            agent_id=agent_id,
                            agent_type=agent_type_str,
                            metadata=metadata_dict,
                            a2a_adapter=agent_wrapper
                        )
                        update_log(f"✅ Agent 등록 완료 (A2A Adapter 포함): {agent_id}", "success")
                        logger.info(f"Agent registered with A2A adapter: {agent_id}")
                    else:
                        await registry.register_agent(
                            agent_id=agent_id,
                            agent_type=agent_type_str,
                            metadata=metadata_dict,
                            a2a_adapter=None
                        )
                        logger.info(f"Agent registered: {agent_id} with type {agent_type_str}")
                else:
                    # 이미 등록된 agent의 adapter 확인
                    if use_a2a and not existing_agent.get("a2a_adapter"):
                        logger.warning(f"Agent {agent_id} exists but has no A2A adapter, creating one...")
                        metadata_dict = existing_agent.get("metadata", {})
                        agent_wrapper = CommonAgentA2AWrapper(agent_id, metadata_dict)
                        await agent_wrapper.start_listener()
                        await agent_wrapper.register_capabilities(metadata_dict.get("capabilities", []))

                        # task_request 핸들러 등록
                        async def handle_task_request(message: A2AMessage) -> Optional[Dict[str, Any]]:
                            logger.info(f"Agent {agent_id} received task request: {message.message_id}")
                            task_data = message.payload.get("task_data", {})
                            task_start_time = datetime.now()

                            # 직접 agent 실행 (레지스트리 조회 없이)
                            # entry_point는 metadata에서 가져오기
                            entry_point = metadata_dict.get("entry_point", "")

                            from srcs.common.a2a_adapter import A2ALogHandler, current_correlation_id
                            log_handler = A2ALogHandler(agent_wrapper, correlation_id=message.correlation_id)
                            log_handler.setLevel(logging.INFO)
                            root_logger = logging.getLogger()
                            root_logger.addHandler(log_handler)

                            # ContextVar 설정
                            token = current_correlation_id.set(message.correlation_id)

                            try:
                                # class-based agent인지 확인
                                is_class_based = "module_path" in task_data and "class_name" in task_data

                                if is_class_based:
                                    # class-based agent는 _run_module_agent 직접 호출
                                    update_log(f"📦 Class-based agent 감지: {task_data.get('class_name')}", "info")
                                    exec_result = await runner._run_module_agent(entry_point, task_data)
                                else:
                                    # 실행 방식 결정
                                    execution_method = task_data.get("_execution_method")
                                    if execution_method == "cli" or entry_point.startswith("python -m") or entry_point.endswith(".py") or "/" in entry_point:
                                        exec_result = await runner._run_cli_agent(entry_point, task_data)
                                    else:
                                        exec_result = await runner._run_module_agent(entry_point, task_data)

                                execution_time = (datetime.now() - task_start_time).total_seconds()
                                exec_result.execution_time = execution_time

                            except Exception as e:
                                logger.error(f"Error executing agent task: {e}", exc_info=True)
                                execution_time = (datetime.now() - task_start_time).total_seconds()
                                from srcs.common.agent_interface import AgentExecutionResult
                                exec_result = AgentExecutionResult(
                                    success=False,
                                    error=str(e),
                                    execution_time=execution_time,
                                    metadata={"agent_id": agent_id, "message_id": message.message_id, "exception": str(e)}
                                )
                            finally:
                                root_logger.removeHandler(log_handler)
                                current_correlation_id.reset(token)

                            response_payload = {
                                "success": exec_result.success,
                                "data": exec_result.data,
                                "error": exec_result.error,
                                "execution_time": exec_result.execution_time,
                                "metadata": exec_result.metadata,
                                "timestamp": exec_result.timestamp.isoformat(),
                            }
                            await agent_wrapper.send_message(
                                target_agent=message.source_agent,
                                message_type="task_response",
                                payload=response_payload,
                                correlation_id=message.correlation_id,  # 원래 요청의 correlation_id 사용
                                priority=MessagePriority.HIGH.value
                            )
                            return response_payload

                        agent_wrapper.register_handler("task_request", handle_task_request)

                        await registry.register_agent(
                            agent_id=agent_id,
                            agent_type=existing_agent.get("agent_type"),
                            metadata=metadata_dict,
                            a2a_adapter=agent_wrapper
                        )

                # A2A를 통한 실행인 경우 메시지로 요청
                if use_a2a:
                    # task_request 메시지 생성 및 전송
                    broker = get_global_broker()
                    # task_data에 metadata와 agent_id 추가 (LangGraph agent 실행에 필요)
                    task_data = {k: v for k, v in input_data.items() if not k.startswith("_")}
                    task_data["_metadata"] = metadata.to_dict() if hasattr(metadata, "to_dict") else (metadata if isinstance(metadata, dict) else {})
                    task_data["_agent_id"] = agent_id

                    request_message = A2AMessage(
                        source_agent=streamlit_agent_id,
                        target_agent=agent_id,
                        message_type="task_request",
                        payload={
                            "task_data": task_data,
                            "correlation_id": correlation_id
                        },
                        correlation_id=correlation_id,
                        priority=MessagePriority.HIGH.value
                    )

                    update_status(f"📤 task_request 메시지 전송 중: {agent_id}")
                    update_log(f"📤 task_request 전송: {agent_id} (correlation_id: {correlation_id})", "info")

                    route_success = await broker.route_message(request_message)
                    if route_success:
                        update_log(f"✅ task_request 라우팅 성공: {agent_id}", "success")
                        logger.info(f"Task request sent to {agent_id} with correlation_id: {correlation_id}")
                    else:
                        update_log(f"❌ task_request 라우팅 실패: {agent_id}", "error")
                        logger.error(f"Failed to route task request to {agent_id}")

                    # 응답 대기 (최대 5분)
                    update_status(f"⏳ Agent 응답 대기 중... (correlation_id: {correlation_id[:8]}...)")
                    timeout = 300
                    check_interval = 0.5
                    elapsed = 0

                    while not response_received and elapsed < timeout:
                        await asyncio.sleep(check_interval)
                        elapsed += check_interval

                        # UI wrapper의 메시지 큐에서 응답 확인 (ui_wrapper가 있는 경우만)
                        if ui_wrapper is not None:
                            try:
                                queue = ui_wrapper._ensure_queue()
                                message = await asyncio.wait_for(queue.get(), timeout=0.1)
                                if message.message_type == "task_response" and message.correlation_id == correlation_id:
                                    response_data = message.payload
                                    response_received = True
                                    update_log(f"✅ task_response 수신: {message.message_id} (correlation_id: {correlation_id})", "success")
                                    logger.info(f"Streamlit UI received task response: {message.message_id}")
                                elif message.message_type == "notification" and message.correlation_id == correlation_id:
                                    payload = message.payload
                                    if payload.get("type") == "log":
                                        msg = payload.get("message", "")
                                        if msg:
                                            update_status(msg)
                                            update_log(f"📋 {msg}", "info")
                                else:
                                    # 다른 메시지는 다시 큐에 넣기
                                    try:
                                        await queue.put(message)
                                    except (RuntimeError, AttributeError):
                                        # Event loop 문제로 큐에 넣을 수 없는 경우 무시
                                        pass
                            except asyncio.TimeoutError:
                                pass
                            except (AttributeError, UnboundLocalError, NameError) as e:
                                # _message_queue가 없는 경우 또는 ui_wrapper가 없는 경우
                                logger.debug(f"Error accessing ui_wrapper queue: {e}")

                        if int(elapsed) % 5 == 0:  # 5초마다 상태 업데이트
                            update_log(f"⏳ 응답 대기 중... ({int(elapsed)}초 경과)", "info")

                    if not response_received:
                        update_log(f"❌ 타임아웃: Agent 응답을 받지 못함 ({timeout}초)", "error")
                        from srcs.common.agent_interface import AgentExecutionResult
                        return AgentExecutionResult(
                            success=False,
                            error="Timeout waiting for agent response via A2A",
                            metadata={"agent_id": agent_id, "correlation_id": correlation_id}
                        )

                    update_log(f"✅ Agent 응답 수신 완료: {agent_id}", "success")
                    update_status("✅ Agent 응답 처리 중...")

                    # 응답 데이터를 AgentExecutionResult로 변환
                    from srcs.common.agent_interface import AgentExecutionResult
                    result = AgentExecutionResult(
                        success=response_data.get("success", False),
                        data=response_data.get("data"),
                        error=response_data.get("error"),
                        execution_time=response_data.get("execution_time", 0),
                        metadata=response_data.get("metadata", {}),
                        timestamp=datetime.fromisoformat(response_data.get("timestamp", datetime.now().isoformat()))
                    )

                    update_log(f"✅ A2A 통신 완료: {agent_id} (성공: {result.success})", "success" if result.success else "error")

                    # 작업 완료 후 A2A adapter 종료
                    update_log(f"🛑 A2A Adapter 종료 중: {agent_id}", "info")
                    try:
                        # Target agent의 A2A adapter 종료
                        target_agent_info = await registry.get_agent(agent_id)
                        if target_agent_info and target_agent_info.get("a2a_adapter"):
                            target_adapter = target_agent_info.get("a2a_adapter")
                            await target_adapter.stop_listener()
                            update_log(f"✅ Target Agent A2A Adapter 종료 완료: {agent_id}", "success")

                        # Streamlit UI agent의 A2A adapter는 유지 (다음 요청을 위해)
                        # 필요시 여기서도 종료할 수 있음
                    except Exception as e:
                        logger.warning(f"Error stopping A2A adapter for {agent_id}: {e}")
                        update_log(f"⚠️ A2A Adapter 종료 중 오류: {str(e)}", "warning")

                    # 레지스트리에서 agent 제거 (선택사항)
                    try:
                        await registry.unregister_agent(agent_id)
                        update_log(f"✅ Agent 레지스트리에서 제거: {agent_id}", "success")
                    except Exception as e:
                        logger.warning(f"Error unregistering agent {agent_id}: {e}")

                    return result
                else:
                    # 직접 실행
                    result = await runner.run_agent(
                        agent_id=agent_id,
                        input_data=input_data,
                        use_a2a=use_a2a
                    )
                    return result

            # 비동기 함수 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(register_and_run())
            finally:
                loop.close()

            # 최종 로그 UI 업데이트
            update_log_ui()

            # 결과 처리
            if result.success:
                st.success("✅ Agent 실행이 성공적으로 완료되었습니다!")

                # 결과 데이터 준비
                result_data = {
                    "success": True,
                    "data": result.data,
                    "execution_time": result.execution_time,
                    "metadata": result.metadata,
                    "timestamp": result.timestamp.isoformat(),
                }

                # 결과를 JSON 파일로 저장
                if result_json_path:
                    result_json_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(result_json_path, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
                    logger.info(f"Result saved to: {result_json_path}")

                # 결과 표시는 각 페이지의 display_results 함수에서 처리
                # JSON 직접 출력 제거 (사용자 경험 개선)
                logger.info(f"Agent execution completed. Result data keys: {list(result.data.keys()) if isinstance(result.data, dict) else 'N/A'}")

                return result_data
            else:
                # 상세한 에러 정보 표시
                error_details = {
                    "error": result.error,
                    "metadata": result.metadata,
                    "execution_time": result.execution_time,
                }

                st.error(f"❌ Agent 실행 실패: {result.error}")

                # 메타데이터에 상세 정보가 있으면 표시
                if result.metadata:
                    with st.expander("🔍 상세 에러 정보", expanded=True):
                        st.text(f"에러: {result.error}")
                        if "step" in result.metadata:
                            st.info(f"**실패 단계**: {result.metadata['step']}")
                        if "exception" in result.metadata:
                            st.code(result.metadata["exception"], language="text")
                        # JSON 출력 대신 구조화된 텍스트로 표시
                        st.text("메타데이터:")
                        for key, value in result.metadata.items():
                            if key not in ["step", "exception"]:
                                st.text(f"  - {key}: {value}")

                # 에러 결과 저장
                error_result = {
                    "success": False,
                    "error": result.error,
                    "execution_time": result.execution_time,
                    "metadata": result.metadata,
                    "timestamp": result.timestamp.isoformat(),
                }

                if result_json_path:
                    result_json_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(result_json_path, 'w', encoding='utf-8') as f:
                        json.dump(error_result, f, indent=2, ensure_ascii=False, default=str)

                logger.error(f"Agent execution failed: {result.error}, metadata: {result.metadata}")
                return None

        except Exception as e:
            import traceback
            error_msg = f"Agent 실행 중 오류 발생: {str(e)}"
            error_traceback = traceback.format_exc()
            logger.error(f"{error_msg}\n{error_traceback}", exc_info=True)

            st.error(f"❌ {error_msg}")

            # 상세한 스택 트레이스 표시
            with st.expander("🔍 상세 에러 정보 (클릭하여 확인)", expanded=False):
                st.code(error_traceback, language="python")

            # 에러 결과 저장
            if result_json_path:
                error_result = {
                    "success": False,
                    "error": error_msg,
                    "traceback": error_traceback,
                    "timestamp": datetime.now().isoformat(),
                }
                result_json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(result_json_path, 'w', encoding='utf-8') as f:
                    json.dump(error_result, f, indent=2, ensure_ascii=False, default=str)

            return None

    return None


def get_registered_agents() -> List[Dict[str, Any]]:
    """
    등록된 agent 목록 조회 (동기 함수)

    Returns:
        등록된 agent 목록
    """
    if "a2a_runner" not in st.session_state:
        return []

    registry = get_global_registry()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        agents = loop.run_until_complete(registry.list_agents())
        return agents
    finally:
        loop.close()


def send_a2a_message(
    source_agent_id: str,
    target_agent_id: str,
    message_type: str,
    payload: Dict[str, Any]
) -> bool:
    """
    A2A 메시지 전송 (동기 함수)

    Args:
        source_agent_id: 소스 agent ID
        target_agent_id: 타겟 agent ID (빈 문자열이면 브로드캐스트)
        message_type: 메시지 타입
        payload: 메시지 페이로드

    Returns:
        전송 성공 여부
    """
    if "a2a_runner" not in st.session_state:
        st.error("A2A runner가 초기화되지 않았습니다.")
        return False

    from srcs.common.a2a_adapter import CommonAgentA2AWrapper
    from srcs.common.a2a_integration import get_global_broker, MessagePriority

    registry = get_global_registry()
    broker = get_global_broker()

    async def send():
        source_agent = await registry.get_agent(source_agent_id)
        if not source_agent:
            logger.error(f"Source agent not found: {source_agent_id}")
            return False

        # 임시 wrapper 생성하여 메시지 전송
        wrapper = CommonAgentA2AWrapper(
            agent_id=source_agent_id,
            agent_metadata=source_agent.get("metadata", {})
        )

        return await wrapper.send_message(
            target_agent=target_agent_id,
            message_type=message_type,
            payload=payload,
            priority=MessagePriority.MEDIUM.value
        )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(send())
    finally:
        loop.close()
