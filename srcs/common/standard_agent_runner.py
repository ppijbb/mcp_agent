"""
표준 Agent 실행 시스템

모든 agent 타입을 통일된 방식으로 실행하는 표준 러너
"""

import importlib
import importlib.util
importlib.invalidate_caches()

# HACK: mcp-agent 0.1.0과 mcp 1.x 간의 타입 호환성 문제 해결
import mcp.types
import types
if hasattr(mcp.types, "ElicitRequestParams") and isinstance(mcp.types.ElicitRequestParams, types.UnionType):
    mcp.types.ElicitRequestParams = mcp.types.ElicitRequestURLParams

# COMPAT: mcp-agent config cache reset for file change reflection
# mcp-agent 0.1.0+ caches settings; reset ensures latest config is loaded
try:
    import mcp_agent.config
    mcp_agent.config._settings = None
except Exception as e:
    logger.warning("Failed to reset mcp-agent config cache: %s", e)

# COMPAT: Google GenAI Safety Settings - filter JAILBREAK category
# Some prompts trigger safety filters unnecessarily; strip JAILBREAK category
try:
    from google.genai import types as genai_types
    if hasattr(genai_types, "GenerateContentConfig"):
        original_config_init = genai_types.GenerateContentConfig.__init__
        def patched_config_init(self, *args, **kwargs):
            if "safety_settings" in kwargs and kwargs["safety_settings"]:
                kwargs["safety_settings"] = [
                    s for s in kwargs["safety_settings"]
                    if "JAILBREAK" not in str(getattr(s, "category", s.get("category", "") if isinstance(s, dict) else ""))
                ]
            original_config_init(self, *args, **kwargs)
        genai_types.GenerateContentConfig.__init__ = patched_config_init
except Exception as e:
    logger.warning("Failed to patch Google GenAI safety settings: %s", e)

import asyncio
import logging
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from srcs.common.agent_interface import (
    AgentType,
    AgentExecutionResult,
)
from srcs.common.a2a_integration import get_global_registry, A2AMessage, MessagePriority
from srcs.common.a2a_adapter import CommonAgentA2AWrapper
from lang_graph.common.a2a_adapter import LangGraphAgentA2AWrapper
from cron_agents.common.a2a_adapter import CronAgentA2AWrapper
_root = Path(__file__).resolve().parent.parent.parent
_sf_adapter = _root / "primary" / "SparkleForge" / "common" / "a2a_adapter.py"
if _sf_adapter.exists():
    try:
        _spec = importlib.util.spec_from_file_location("_sf_a2a", _sf_adapter)
        _sf_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_sf_mod)
        SparkleForgeA2AWrapper = _sf_mod.SparkleForgeA2AWrapper
    except Exception as e:
        logger.warning(f"Failed to load SparkleForgeA2AWrapper: {e}")
        SparkleForgeA2AWrapper = None
else:
    SparkleForgeA2AWrapper = None

logger = logging.getLogger(__name__)


def _normalize_agent_type(agent_type: Any) -> str:
    """
    Agent 타입을 문자열로 정규화

    Args:
        agent_type: AgentType enum 또는 문자열

    Returns:
        정규화된 agent 타입 문자열
    """
    if agent_type is None:
        logger.warning("agent_type is None")
        return "unknown"
    elif isinstance(agent_type, AgentType):
        return agent_type.value
    elif isinstance(agent_type, str):
        # 문자열이지만 enum 표현식처럼 보이는 경우 처리
        if agent_type.startswith("<AgentType.") and agent_type.endswith(">"):
            # "<AgentType.MCP_AGENT: 'mcp_agent'>" 형식에서 값 추출
            try:
                # 'mcp_agent' 부분 추출
                value_start = agent_type.find("'") + 1
                value_end = agent_type.rfind("'")
                if value_start > 0 and value_end > value_start:
                    return agent_type[value_start:value_end]
            except Exception as e:
                logger.warning(f"Failed to parse agent_type string: {agent_type}, error: {e}")
        return agent_type
    else:
        logger.warning(f"Unknown agent_type type: {type(agent_type)}, value: {agent_type}")
        # enum 객체의 문자열 표현에서 값 추출 시도
        agent_type_str = str(agent_type)
        if ": '" in agent_type_str and "'" in agent_type_str:
            try:
                value_start = agent_type_str.find("'") + 1
                value_end = agent_type_str.rfind("'")
                if value_start > 0 and value_end > value_start:
                    return agent_type_str[value_start:value_end]
            except Exception as e:
                logger.warning("Failed to parse agent_type string: %s", e)
        return str(agent_type)


class StandardAgentRunner:
    """Standard Agent execution system for running all agent types in a unified way."""

    def __init__(self):
        """Initialize the StandardAgentRunner with a registry and agent cache."""
        self.registry = get_global_registry()
        self._agent_cache: Dict[str, Any] = {}

    async def run_agent(
        self,
        agent_id: str,
        input_data: Dict[str, Any],
        use_a2a: bool = False
    ) -> AgentExecutionResult:
        """
        Agent 실행

        Args:
            agent_id: Agent ID
            input_data: 입력 데이터
            use_a2a: A2A 연결 사용 여부

        Returns:
            AgentExecutionResult: 실행 결과
        """
        # 레지스트리에서 agent 정보 조회
        try:
            agent_info = await self.registry.get_agent(agent_id)
            if not agent_info:
                error_msg = f"Agent not found in registry: {agent_id}"
                logger.error(error_msg)
                return AgentExecutionResult(
                    success=False,
                    error=error_msg,
                    metadata={"agent_id": agent_id, "step": "registry_lookup"}
                )

            logger.info(f"Found agent in registry: {agent_id}, type: {agent_info.get('agent_type')}")
            logger.debug(f"Full agent_info: {agent_info}")
        except Exception as e:
            error_msg = f"Error looking up agent in registry: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return AgentExecutionResult(
                success=False,
                error=error_msg,
                metadata={"agent_id": agent_id, "step": "registry_lookup", "exception": str(e)}
            )

        # Agent 타입 정규화 - 레지스트리의 최상위 레벨에서 가져오기
        raw_agent_type = agent_info.get("agent_type")
        agent_type = _normalize_agent_type(raw_agent_type)

        # metadata도 정규화 (metadata 안에 agent_type이 enum 객체로 있을 수 있음)
        metadata = agent_info.get("metadata", {})
        if metadata and "agent_type" in metadata:
            metadata_agent_type = metadata.get("agent_type")
            if isinstance(metadata_agent_type, AgentType):
                metadata["agent_type"] = metadata_agent_type.value
                logger.debug(f"Normalized agent_type in metadata: {metadata['agent_type']}")

        logger.info(f"Running agent {agent_id} with type {agent_type} (normalized from {raw_agent_type})")

        # Agent 타입에 따라 실행
        try:
            if agent_type == AgentType.MCP_AGENT.value:
                logger.debug(f"Executing MCP agent: {agent_id}")
                return await self._run_mcp_agent(agent_id, metadata, input_data, use_a2a)
            elif agent_type == AgentType.LANGGRAPH_AGENT.value:
                logger.debug(f"Executing LangGraph agent: {agent_id}")
                return await self._run_langgraph_agent(agent_id, metadata, input_data, use_a2a)
            elif agent_type == AgentType.CRON_AGENT.value:
                logger.debug(f"Executing Cron agent: {agent_id}")
                return await self._run_cron_agent(agent_id, metadata, input_data, use_a2a)
            elif agent_type == AgentType.SPARKLEFORGE_AGENT.value:
                logger.debug(f"Executing SparkleForge agent: {agent_id}")
                return await self._run_sparkleforge_agent(agent_id, metadata, input_data, use_a2a)
            else:
                error_msg = f"Unsupported agent type: {agent_type} (raw: {raw_agent_type}, type: {type(raw_agent_type)})"
                logger.error(error_msg)
                return AgentExecutionResult(
                    success=False,
                    error=error_msg,
                    metadata={
                        "agent_id": agent_id,
                        "agent_type": agent_type,
                        "raw_agent_type": str(raw_agent_type),
                        "step": "agent_type_check"
                    }
                )
        except Exception as e:
            error_msg = f"Error executing agent {agent_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return AgentExecutionResult(
                success=False,
                error=error_msg,
                metadata={"agent_id": agent_id, "agent_type": agent_type, "step": "agent_execution", "exception": str(e)}
            )

    async def _run_mcp_agent(
        self,
        agent_id: str,
        metadata: Dict[str, Any],
        input_data: Dict[str, Any],
        use_a2a: bool
    ) -> AgentExecutionResult:
        """
        Execute an MCP Agent.

        Args:
            agent_id: The agent ID to execute
            metadata: Agent metadata including entry_point and capabilities
            input_data: Input data for the agent
            use_a2a: Whether to use A2A messaging

        Returns:
            AgentExecutionResult: The execution result
        """
        start_time = datetime.now()

        try:
            entry_point = metadata.get("entry_point")
            if not entry_point:
                return AgentExecutionResult(
                    success=False,
                    error=f"No entry point specified for agent {agent_id}",
                    metadata={"agent_id": agent_id}
                )

            # A2A 래퍼 생성 및 등록
            wrapper = None
            if use_a2a:
                wrapper = CommonAgentA2AWrapper(agent_id, metadata)
                await wrapper.start_listener()
                await wrapper.register_capabilities(metadata.get("capabilities", []))

                # A2A 메시지 핸들러 등록
                runner_instance = self  # self를 클로저에 저장
                async def handle_task_request(message: A2AMessage) -> Optional[Dict[str, Any]]:
                    """task_request 메시지 처리"""
                    logger.info(f"Agent {agent_id} received task request: {message.message_id}")
                    task_data = message.payload.get("task_data", {})
                    task_start_time = datetime.now()

                    # 실행 방식 결정
                    execution_method = task_data.get("_execution_method")

                    # class-based agent인지 확인 (module_path, class_name, method_name이 있는 경우)
                    is_class_based = "module_path" in task_data and "class_name" in task_data

                    # Agent 실행
                    from srcs.common.a2a_adapter import A2ALogHandler, current_correlation_id
                    log_handler = A2ALogHandler(wrapper, correlation_id=message.correlation_id)
                    log_handler.setLevel(logging.INFO)
                    root_logger = logging.getLogger()
                    root_logger.addHandler(log_handler)

                    # ContextVar 설정
                    token = current_correlation_id.set(message.correlation_id)

                    try:
                        if is_class_based:
                            # class-based agent는 _run_module_agent 사용
                            exec_result = await runner_instance._run_module_agent(entry_point, task_data)
                        elif execution_method == "cli" or entry_point.startswith("python -m") or entry_point.endswith(".py") or "/" in entry_point:
                            exec_result = await runner_instance._run_cli_agent(entry_point, task_data)
                        else:
                            exec_result = await runner_instance._run_module_agent(entry_point, task_data)

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
                            metadata={"agent_id": agent_id, "message_id": message.message_id}
                        )
                    finally:
                        root_logger.removeHandler(log_handler)
                        current_correlation_id.reset(token)

                    # 결과를 A2A 메시지로 전송
                    response_payload = {
                        "success": exec_result.success,
                        "data": exec_result.data,
                        "error": exec_result.error,
                        "execution_time": exec_result.execution_time,
                        "metadata": exec_result.metadata,
                        "timestamp": exec_result.timestamp.isoformat(),
                    }

                    await wrapper.send_message(
                        target_agent=message.source_agent,
                        message_type="task_response",
                        payload=response_payload,
                        correlation_id=message.correlation_id,  # 원래 요청의 correlation_id 사용
                        priority=MessagePriority.HIGH.value
                    )

                    logger.info(f"Agent {agent_id} sent task response: {message.message_id}")
                    return response_payload

                wrapper.register_handler("task_request", handle_task_request)

                # 레지스트리에 wrapper 등록 (이미 등록되어 있으면 업데이트)
                agent_info = await self.registry.get_agent(agent_id)
                if agent_info:
                    await self.registry.register_agent(
                        agent_id=agent_id,
                        agent_type=agent_info.get("agent_type"),
                        metadata=metadata,
                        a2a_adapter=wrapper
                    )

            # A2A를 통한 실행인 경우, 메시지로 요청 전송하고 응답 대기
            if use_a2a and wrapper:
                from srcs.common.a2a_integration import get_global_broker
                import uuid

                # Streamlit UI agent ID (요청자)
                source_agent_id = input_data.get("_source_agent_id", "streamlit_ui")

                # task_request 메시지 생성
                correlation_id = str(uuid.uuid4())
                request_message = A2AMessage(
                    source_agent=source_agent_id,
                    target_agent=agent_id,
                    message_type="task_request",
                    payload={
                        "task_data": {k: v for k, v in input_data.items() if not k.startswith("_")},
                        "correlation_id": correlation_id
                    },
                    correlation_id=correlation_id,
                    priority=MessagePriority.HIGH.value
                )

                # 메시지 전송
                broker = get_global_broker()
                await broker.route_message(request_message)

                # 응답 대기 (최대 5분)
                response_received = False
                response_data = None
                timeout = 300  # 5분
                check_interval = 0.5  # 0.5초마다 확인
                elapsed = 0

                while not response_received and elapsed < timeout:
                    await asyncio.sleep(check_interval)
                    elapsed += check_interval

                    # 메시지 큐에서 응답 확인
                    try:
                        message = await asyncio.wait_for(wrapper._message_queue.get(), timeout=0.1)
                        if message.message_type == "task_response" and message.correlation_id == correlation_id:
                            response_data = message.payload
                            response_received = True
                    except asyncio.TimeoutError:
                        continue

                if not response_received:
                    return AgentExecutionResult(
                        success=False,
                        error="Timeout waiting for agent response",
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        metadata={"agent_id": agent_id, "correlation_id": correlation_id}
                    )

                # 응답 데이터를 AgentExecutionResult로 변환
                return AgentExecutionResult(
                    success=response_data.get("success", False),
                    data=response_data.get("data"),
                    error=response_data.get("error"),
                    execution_time=response_data.get("execution_time", 0),
                    metadata=response_data.get("metadata", {}),
                    timestamp=datetime.fromisoformat(response_data.get("timestamp", datetime.now().isoformat()))
                )

            # A2A를 사용하지 않는 경우 직접 실행
            # 실행 방식 결정
            execution_method = input_data.get("_execution_method")

            # class-based agent인지 확인 (module_path, class_name, method_name이 있는 경우)
            is_class_based = "module_path" in input_data and "class_name" in input_data

            if is_class_based:
                # class-based agent는 _run_module_agent 사용
                result = await self._run_module_agent(entry_point, input_data)
            elif execution_method == "cli" or entry_point.startswith("python -m") or entry_point.endswith(".py") or "/" in entry_point:
                result = await self._run_cli_agent(entry_point, input_data)
            else:
                result = await self._run_module_agent(entry_point, input_data)

            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error running MCP agent {agent_id}: {e}", exc_info=True)
            return AgentExecutionResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                metadata={"agent_id": agent_id}
            )

    async def _run_langgraph_agent(
        self,
        agent_id: str,
        metadata: Dict[str, Any],
        input_data: Dict[str, Any],
        use_a2a: bool
    ) -> AgentExecutionResult:
        """
        Execute a LangGraph Agent.

        Args:
            agent_id: The agent ID to execute
            metadata: Agent metadata including entry_point and capabilities
            input_data: Input data for the agent
            use_a2a: Whether to use A2A messaging

        Returns:
            AgentExecutionResult: The execution result
        """
        start_time = datetime.now()

        try:
            entry_point = metadata.get("entry_point")
            if not entry_point:
                return AgentExecutionResult(
                    success=False,
                    error=f"No entry point specified for agent {agent_id}",
                    metadata={"agent_id": agent_id}
                )

            # LangGraph app 로드
            graph_app = await self._load_langgraph_app(entry_point)
            if not graph_app:
                return AgentExecutionResult(
                    success=False,
                    error=f"Failed to load LangGraph app for agent {agent_id}",
                    metadata={"agent_id": agent_id}
                )

            # 실행
            wrapper = LangGraphAgentA2AWrapper(agent_id, metadata, graph_app=graph_app)

            if use_a2a:
                await wrapper.start_listener()
                await wrapper.register_capabilities(metadata.get("capabilities", []))

            result_data = await wrapper.execute_graph(input_data, stream=False)

            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentExecutionResult(
                success=True,
                data=result_data if isinstance(result_data, dict) else {"result": result_data},
                execution_time=execution_time,
                metadata={"agent_id": agent_id, "agent_type": "langgraph"}
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error running LangGraph agent {agent_id}: {e}")
            return AgentExecutionResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                metadata={"agent_id": agent_id}
            )

    async def _run_cron_agent(
        self,
        agent_id: str,
        metadata: Dict[str, Any],
        input_data: Dict[str, Any],
        use_a2a: bool
    ) -> AgentExecutionResult:
        """
        Execute a Cron Agent.

        Args:
            agent_id: The agent ID to execute
            metadata: Agent metadata including cron_schedule and entry_point
            input_data: Input data for the agent
            use_a2a: Whether to use A2A messaging

        Returns:
            AgentExecutionResult: The execution result
        """
        start_time = datetime.now()

        try:
            cron_schedule = metadata.get("cron_schedule", "")

            wrapper = CronAgentA2AWrapper(
                agent_id,
                metadata,
                cron_schedule=cron_schedule,
                execute_function=None  # 실제 함수는 entry_point에서 로드
            )

            if use_a2a:
                await wrapper.start_listener()
                await wrapper.register_capabilities(metadata.get("capabilities", []))

            # Cron agent는 일반적으로 스케줄에 의해 실행되므로 즉시 실행은 선택적
            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentExecutionResult(
                success=True,
                data={"message": "Cron agent scheduled", "schedule": cron_schedule},
                execution_time=execution_time,
                metadata={"agent_id": agent_id, "agent_type": "cron"}
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error running Cron agent {agent_id}: {e}")
            return AgentExecutionResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                metadata={"agent_id": agent_id}
            )

    async def _run_sparkleforge_agent(
        self,
        agent_id: str,
        metadata: Dict[str, Any],
        input_data: Dict[str, Any],
        use_a2a: bool
    ) -> AgentExecutionResult:
        """
        Execute a SparkleForge Agent.

        Args:
            agent_id: The agent ID to execute
            metadata: Agent metadata
            input_data: Input data for the agent including query and context
            use_a2a: Whether to use A2A messaging

        Returns:
            AgentExecutionResult: The execution result
        """
        if SparkleForgeA2AWrapper is None:
            return AgentExecutionResult(
                success=False,
                error="SparkleForge not available (primary/SparkleForge not present)",
                execution_time=0.0,
                metadata={"agent_id": agent_id}
            )
        start_time = datetime.now()

        try:
            # SparkleForge orchestrator 로드
            from sparkleforge.src.core.agent_orchestrator import AgentOrchestrator
            from sparkleforge.src.core.researcher_config import load_config_from_env

            config = load_config_from_env()
            orchestrator = AgentOrchestrator(config=config)

            wrapper = SparkleForgeA2AWrapper(agent_id, metadata, orchestrator=orchestrator)

            if use_a2a:
                await wrapper.start_listener()
                await wrapper.register_capabilities(metadata.get("capabilities", []))

            query = input_data.get("query", input_data.get("task", ""))
            context = input_data.get("context", {})

            result_data = await wrapper.execute_research(query, context)

            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentExecutionResult(
                success=True,
                data=result_data if isinstance(result_data, dict) else {"result": result_data},
                execution_time=execution_time,
                metadata={"agent_id": agent_id, "agent_type": "sparkleforge"}
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error running SparkleForge agent {agent_id}: {e}")
            return AgentExecutionResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                metadata={"agent_id": agent_id}
            )

    async def _run_cli_agent(
        self,
        entry_point: str,
        input_data: Dict[str, Any]
    ) -> AgentExecutionResult:
        """
        Execute an agent via CLI (command-line interface).
        
        Converts input data to CLI arguments and executes the agent as a subprocess.
        Supports Python module execution, script files, and direct CLI commands.
        
        Args:
            entry_point: CLI command or script path to execute
            input_data: Dictionary of input data to convert to CLI arguments
            
        Returns:
            AgentExecutionResult: Execution result with success status and data
        """
        try:
            # input_data에서 CLI 인자 추출
            # input_data에 직접 CLI 인자가 있는 경우 (예: {"input_json_path": "...", "result_json_path": "..."})
            # 또는 표준 형식 (예: {"_cli_args": ["--arg1", "value1", ...]})

            cli_args = input_data.get("_cli_args", [])

            # 표준 인자가 없는 경우, input_data를 CLI 인자로 변환
            if not cli_args:
                # 일반적인 패턴: input_json_path, result_json_path 등
                args = []
                for key, value in input_data.items():
                    if key.startswith("_") or value is None:
                        continue
                    if isinstance(value, (dict, list)):
                        # 복잡한 객체는 JSON 파일로 저장
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                            json.dump(value, f)
                            args.extend([f"--{key.replace('_', '-')}", f.name])
                    else:
                        args.extend([f"--{key.replace('_', '-')}", str(value)])
                cli_args = args

            # 사용할 Python 인터프리터 결정
            python_exe = sys.executable
            env_python = os.environ.get("MCP_AGENT_PYTHON")
            if env_python and os.path.exists(env_python):
                python_exe = env_python

            # 명령어 구성
            if entry_point.startswith("python -m"):
                command = [python_exe] + entry_point.split()[1:] + cli_args
            elif entry_point.endswith(".py"):
                command = [python_exe, entry_point] + cli_args
            else:
                # 모듈 경로인 경우 python -m으로 실행
                command = [python_exe, "-m", entry_point] + cli_args

            logger.info(f"Executing CLI command: {' '.join(command)}")

            # 실행
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent.parent.parent
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                # 결과 파일 경로가 있으면 파일에서 읽기
                result_json_path = input_data.get("result_json_path")
                if result_json_path and Path(result_json_path).exists():
                    with open(result_json_path, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                elif stdout:
                    try:
                        result_data = json.loads(stdout.decode())
                    except json.JSONDecodeError:
                        result_data = {"output": stdout.decode()}
                else:
                    result_data = {"success": True, "message": "Agent executed successfully"}

                return AgentExecutionResult(
                    success=True,
                    data=result_data
                )
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"CLI agent execution failed: {error_msg}")
                return AgentExecutionResult(
                    success=False,
                    error=error_msg
                )

        except Exception as e:
            logger.error(f"Error in _run_cli_agent: {e}", exc_info=True)
            return AgentExecutionResult(
                success=False,
                error=str(e)
            )

    async def _run_module_agent(
        self,
        entry_point: str,
        input_data: Dict[str, Any]
    ) -> AgentExecutionResult:
        """
        Execute an agent via module import.
        
        Dynamically imports and executes an agent module, supporting both
        function-based and class-based agents. Automatically detects the
        agent type and handles async/sync functions appropriately.
        
        Args:
            entry_point: Module path (e.g., 'module.submodule.function')
            input_data: Dictionary of input data to pass to the agent
            
        Returns:
            AgentExecutionResult: Execution result with success status and data
        """
        try:
            import importlib

            # input_data에 module_path, class_name, method_name이 있는 경우 (복잡한 구조)
            if "module_path" in input_data and "class_name" in input_data:
                module_path = input_data["module_path"]
                class_name = input_data["class_name"]
                method_name = input_data.get("method_name", "main")

                # class_name이 None이거나 빈 문자열이면 함수 호출 방식으로 처리
                # 더 엄격한 체크: None, 빈 문자열, "None" 문자열 모두 처리
                if class_name is None:
                    is_class_name_valid = False
                elif not isinstance(class_name, str):
                    is_class_name_valid = False
                elif class_name.strip() == "":
                    is_class_name_valid = False
                elif class_name.lower() == "none":
                    is_class_name_valid = False
                else:
                    is_class_name_valid = True

                logger.debug(f"class_name={class_name}, is_class_name_valid={is_class_name_valid}")

                if not is_class_name_valid:
                    logger.info(f"Loading function-based agent: {module_path}.{method_name}")

                    # 모듈 import
                    module = importlib.import_module(module_path)

                    # 함수 가져오기
                    func = getattr(module, method_name)

                    # 함수 시그니처 확인하여 필요한 인자만 추출
                    import inspect
                    try:
                        sig = inspect.signature(func)
                        func_params = set(sig.parameters.keys())

                        # 함수가 실제로 받을 수 있는 인자만 필터링
                        exclude_keys = ["module_path", "class_name", "method_name", "init_kwargs",
                                      "result_json_path", "_execution_method", "_cli_args"]
                        func_kwargs = {k: v for k, v in input_data.items()
                                     if k in func_params and k not in exclude_keys}

                        logger.debug(f"Function {method_name} accepts parameters: {func_params}")
                        logger.debug(f"Passing arguments: {list(func_kwargs.keys())}")
                    except Exception as e:
                        logger.warning(f"Could not inspect function signature: {e}, using all input_data")
                        exclude_keys = ["module_path", "class_name", "method_name", "init_kwargs",
                                      "result_json_path", "_execution_method", "_cli_args"]
                        func_kwargs = {k: v for k, v in input_data.items()
                                     if k not in exclude_keys}

                    # 실행
                    if asyncio.iscoroutinefunction(func):
                        result = await func(**func_kwargs)
                    else:
                        result = func(**func_kwargs)
                else:
                    # 클래스 기반 호출
                    logger.info(f"Loading class-based agent: {module_path}.{class_name}.{method_name}")

                # 모듈 import
                module = importlib.import_module(module_path)

                # 클래스 가져오기
                agent_class = getattr(module, class_name)

                # 인스턴스 생성 (필요한 경우)
                # input_data에서 클래스 초기화에 필요한 인자 추출
                init_kwargs = {}
                if "init_kwargs" in input_data:
                    init_kwargs = input_data["init_kwargs"]

                # 인스턴스 생성
                if init_kwargs:
                    agent_instance = agent_class(**init_kwargs)
                else:
                    # 기본 생성자로 생성 시도
                    try:
                        agent_instance = agent_class()
                    except TypeError:
                        # 생성자가 필요한 인자를 요구하는 경우, init_kwargs에서 추출하여 생성
                        if init_kwargs:
                            agent_instance = agent_class(**init_kwargs)
                        else:
                            logger.error(f"Cannot instantiate {agent_class.__name__}: requires init arguments")
                            raise

                # 메서드 호출
                method = getattr(agent_instance, method_name)

                # 메서드 시그니처 확인하여 필요한 인자만 추출
                import inspect
                try:
                    sig = inspect.signature(method)
                    method_params = set(sig.parameters.keys())

                    # 메서드가 실제로 받을 수 있는 인자만 필터링
                    method_kwargs = {k: v for k, v in input_data.items()
                                   if k in method_params and k not in ["module_path", "class_name", "method_name", "init_kwargs"]}

                    logger.debug(f"Method {method_name} accepts parameters: {method_params}")
                    logger.debug(f"Passing arguments: {list(method_kwargs.keys())}")
                except Exception as e:
                    logger.warning(f"Could not inspect method signature: {e}, using all input_data")
                    # 시그니처 확인 실패 시 기본 제외 목록 사용
                    exclude_keys = ["module_path", "class_name", "method_name", "init_kwargs",
                                  "result_json_path", "_execution_method", "_cli_args"]
                    method_kwargs = {k: v for k, v in input_data.items()
                                   if k not in exclude_keys}

                # 실행
                if asyncio.iscoroutinefunction(method):
                    result = await method(**method_kwargs)
                else:
                    result = method(**method_kwargs)

            else:
                # 단순 함수 호출 방식 또는 LangGraph 모듈
                # 모듈 경로 파싱 (예: "srcs.basic_agents.run_rag_agent")
                if "." in entry_point:
                    parts = entry_point.rsplit(".", 1)
                    module_path = parts[0]
                    potential_function = parts[1]

                    # potential_function이 실제 함수인지 모듈 이름인지 확인
                    # 먼저 모듈을 import하고 확인
                    try:
                        module = importlib.import_module(entry_point)
                        # 모듈 자체가 import되었으면 (예: lang_graph.table_game_mate.agents.game_ui_analyzer)
                        # app 속성을 찾아봐야 함
                        if hasattr(module, "app"):
                            # LangGraph 모듈
                            graph_app = module.app
                            logger.info(f"Loading LangGraph agent from module: {entry_point}")
                            from lang_graph.common.a2a_adapter import LangGraphAgentA2AWrapper
                            # metadata는 input_data에서 추출하거나 기본값 사용
                            metadata_dict = input_data.get("_metadata", {})
                            if not isinstance(metadata_dict, dict):
                                metadata_dict = metadata_dict.to_dict() if hasattr(metadata_dict, "to_dict") else {}
                            agent_id_value = input_data.get("_agent_id", entry_point)
                            wrapper = LangGraphAgentA2AWrapper(agent_id_value, metadata_dict, graph_app=graph_app)
                            result_data = await wrapper.execute_graph(input_data, stream=False)
                            result = result_data if isinstance(result_data, dict) else {"result": result_data}
                        elif callable(getattr(module, potential_function, None)):
                            # 함수가 있는 경우
                            function_name = potential_function
                            logger.info(f"Loading function-based agent: {module_path}.{function_name}")
                            func = getattr(module, function_name)
                            if asyncio.iscoroutinefunction(func):
                                result = await func(**input_data)
                            else:
                                result = func(**input_data)
                        else:
                            # 함수도 없고 app도 없으면 에러
                            raise ValueError(f"Module {entry_point} has no 'app' attribute and no callable '{potential_function}' function")
                    except ImportError:
                        # 전체 경로로 import 실패, 모듈 경로와 함수로 분리
                        function_name = potential_function
                        logger.info(f"Loading function-based agent: {module_path}.{function_name}")
                        module = importlib.import_module(module_path)
                        func = getattr(module, function_name)
                        if asyncio.iscoroutinefunction(func):
                            result = await func(**input_data)
                        else:
                            result = func(**input_data)
                else:
                    # 단일 모듈 이름
                    module_path = entry_point
                    module = importlib.import_module(module_path)
                    if hasattr(module, "app"):
                        # LangGraph 모듈
                        graph_app = module.app
                        logger.info(f"Loading LangGraph agent from module: {module_path}")
                        from lang_graph.common.a2a_adapter import LangGraphAgentA2AWrapper
                        # metadata는 input_data에서 추출하거나 기본값 사용
                        metadata_dict = input_data.get("_metadata", {})
                        if not isinstance(metadata_dict, dict):
                            metadata_dict = metadata_dict.to_dict() if hasattr(metadata_dict, "to_dict") else {}
                        agent_id_value = input_data.get("_agent_id", module_path)
                        wrapper = LangGraphAgentA2AWrapper(agent_id_value, metadata_dict, graph_app=graph_app)
                        result_data = await wrapper.execute_graph(input_data, stream=False)
                        result = result_data if isinstance(result_data, dict) else {"result": result_data}
                    else:
                        # 함수 찾기 시도
                        function_name = "main"
                        if hasattr(module, function_name):
                            func = getattr(module, function_name)
                            if asyncio.iscoroutinefunction(func):
                                result = await func(**input_data)
                            else:
                                result = func(**input_data)
                        else:
                            raise ValueError(f"Module {module_path} has no 'app' attribute and no '{function_name}' function")

            return AgentExecutionResult(
                success=True,
                data=result if isinstance(result, dict) else {"result": result}
            )

        except Exception as e:
            error_msg = f"Error in _run_module_agent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return AgentExecutionResult(
                success=False,
                error=error_msg,
                metadata={"entry_point": entry_point, "input_data_keys": list(input_data.keys())}
            )

    async def _load_langgraph_app(self, entry_point: str) -> Optional[Any]:
        """
        Load a LangGraph application from a module or file path.
        
        Supports loading LangGraph apps from both Python module paths and direct
        file paths. Looks for 'app' or 'graph' attributes in the loaded module.
        
        Args:
            entry_point: Module path or file path to the LangGraph application
            
        Returns:
            LangGraph compiled application (app or graph) if found, None otherwise
        """
        try:
            # entry_point는 모듈 경로 또는 파일 경로
            if entry_point.endswith(".py"):
                # 파일 경로인 경우
                import importlib.util
                spec = importlib.util.spec_from_file_location("langgraph_app", entry_point)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # app 또는 graph 찾기
                if hasattr(module, "app"):
                    return module.app
                elif hasattr(module, "graph"):
                    return module.graph
                else:
                    logger.warning(f"No app or graph found in {entry_point}")
                    return None
            else:
                # 모듈 경로인 경우
                import importlib
                module = importlib.import_module(entry_point)

                if hasattr(module, "app"):
                    return module.app
                elif hasattr(module, "graph"):
                    return module.graph
                else:
                    logger.warning(f"No app or graph found in {entry_point}")
                    return None

        except Exception as e:
            logger.error(f"Error loading LangGraph app from {entry_point}: {e}")
            return None

    async def list_available_agents(self) -> List[Dict[str, Any]]:
        """사용 가능한 agent 목록 반환"""
        return await self.registry.list_agents()

    async def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Agent 정보 조회"""
        return await self.registry.get_agent(agent_id)
