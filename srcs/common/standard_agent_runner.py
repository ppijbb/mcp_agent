"""
표준 Agent 실행 시스템

모든 agent 타입을 통일된 방식으로 실행하는 표준 러너
"""

import asyncio
import logging
import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from srcs.common.agent_interface import (
    AgentType,
    AgentMetadata,
    AgentExecutionResult,
    AgentRunner,
    BaseAgent,
)
from srcs.common.a2a_integration import get_global_registry
from srcs.common.a2a_adapter import CommonAgentA2AWrapper
from lang_graph.common.a2a_adapter import LangGraphAgentA2AWrapper
from cron_agents.common.a2a_adapter import CronAgentA2AWrapper
from sparkleforge.common.a2a_adapter import SparkleForgeA2AWrapper

logger = logging.getLogger(__name__)


class StandardAgentRunner:
    """표준 Agent 실행 시스템"""
    
    def __init__(self):
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
        agent_info = await self.registry.get_agent(agent_id)
        if not agent_info:
            return AgentExecutionResult(
                success=False,
                error=f"Agent not found: {agent_id}",
                metadata={"agent_id": agent_id}
            )
        
        agent_type = agent_info.get("agent_type")
        metadata = agent_info.get("metadata", {})
        
        # Agent 타입에 따라 실행
        if agent_type == AgentType.MCP_AGENT.value:
            return await self._run_mcp_agent(agent_id, metadata, input_data, use_a2a)
        elif agent_type == AgentType.LANGGRAPH_AGENT.value:
            return await self._run_langgraph_agent(agent_id, metadata, input_data, use_a2a)
        elif agent_type == AgentType.CRON_AGENT.value:
            return await self._run_cron_agent(agent_id, metadata, input_data, use_a2a)
        elif agent_type == AgentType.SPARKLEFORGE_AGENT.value:
            return await self._run_sparkleforge_agent(agent_id, metadata, input_data, use_a2a)
        else:
            return AgentExecutionResult(
                success=False,
                error=f"Unsupported agent type: {agent_type}",
                metadata={"agent_id": agent_id, "agent_type": agent_type}
            )
    
    async def _run_mcp_agent(
        self,
        agent_id: str,
        metadata: Dict[str, Any],
        input_data: Dict[str, Any],
        use_a2a: bool
    ) -> AgentExecutionResult:
        """MCP Agent 실행"""
        start_time = datetime.now()
        
        try:
            entry_point = metadata.get("entry_point")
            if not entry_point:
                return AgentExecutionResult(
                    success=False,
                    error=f"No entry point specified for agent {agent_id}",
                    metadata={"agent_id": agent_id}
                )
            
            # CLI 실행 방식
            if entry_point.startswith("python -m") or entry_point.endswith(".py"):
                result = await self._run_cli_agent(entry_point, input_data)
            else:
                # 모듈 import 방식
                result = await self._run_module_agent(entry_point, input_data)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            # A2A 래퍼 사용
            if use_a2a:
                wrapper = CommonAgentA2AWrapper(agent_id, metadata)
                await wrapper.start_listener()
                await wrapper.register_capabilities(metadata.get("capabilities", []))
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error running MCP agent {agent_id}: {e}")
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
        """LangGraph Agent 실행"""
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
        """Cron Agent 실행"""
        start_time = datetime.now()
        
        try:
            cron_schedule = metadata.get("cron_schedule", "")
            entry_point = metadata.get("entry_point")
            
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
        """SparkleForge Agent 실행"""
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
        """CLI 방식 Agent 실행"""
        try:
            # JSON 파일로 입력 데이터 저장
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(input_data, f)
                input_file = f.name
            
            # 명령어 구성
            if entry_point.startswith("python -m"):
                command = entry_point.split() + ["--input-json", input_file]
            else:
                command = ["python", entry_point, "--input-json", input_file]
            
            # 실행
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                result_data = json.loads(stdout.decode()) if stdout else {}
                return AgentExecutionResult(
                    success=True,
                    data=result_data
                )
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                return AgentExecutionResult(
                    success=False,
                    error=error_msg
                )
                
        except Exception as e:
            return AgentExecutionResult(
                success=False,
                error=str(e)
            )
    
    async def _run_module_agent(
        self,
        entry_point: str,
        input_data: Dict[str, Any]
    ) -> AgentExecutionResult:
        """모듈 import 방식 Agent 실행"""
        try:
            # 모듈 경로 파싱 (예: "srcs.basic_agents.run_rag_agent")
            module_path, function_name = entry_point.rsplit(".", 1) if "." in entry_point else (entry_point, "main")
            
            import importlib
            module = importlib.import_module(module_path)
            func = getattr(module, function_name)
            
            # 실행
            if asyncio.iscoroutinefunction(func):
                result = await func(**input_data)
            else:
                result = func(**input_data)
            
            return AgentExecutionResult(
                success=True,
                data=result if isinstance(result, dict) else {"result": result}
            )
            
        except Exception as e:
            return AgentExecutionResult(
                success=False,
                error=str(e)
            )
    
    async def _load_langgraph_app(self, entry_point: str) -> Optional[Any]:
        """LangGraph app 로드"""
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

