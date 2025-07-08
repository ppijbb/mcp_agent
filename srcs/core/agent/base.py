import asyncio
import functools
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent as MCP_Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from srcs.core.config.loader import settings
from srcs.core.errors import APIError, MCPError, WorkflowError, CircuitBreakerOpen
from cachetools import LRUCache
from cachetools.keys import hashkey
from pybreaker import CircuitBreaker, CircuitBreakerError
import aiohttp

def async_memoize(func):
    cache = LRUCache(maxsize=128)
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        key = hashkey(*args, **kwargs)
        if key in cache:
            return cache[key]
        
        result = await func(*args, **kwargs)
        cache[key] = result
        return result

    return wrapper

class BaseAgent(ABC):
    """
    모든 MCP 에이전트의 최상위 기반 클래스.
    공통적인 설정, 로깅, MCPApp 초기화를 처리합니다.
    """
    def __init__(self, name: str, instruction: str = "", server_names: List[str] | None = None):
        """
        BaseAgent를 초기화합니다.

        Args:
            name (str): 에이전트의 고유 이름.
            instruction (str): 에이전트가 수행할 작업에 대한 자연어 설명.
            server_names (List[str] | None): 에이전트가 사용할 MCP 서버 목록.
        """
        self.name = name
        self.instruction = instruction
        self.server_names = server_names or []
        
        self.settings = settings  # 중앙 설정 객체 사용
        self.app = self._setup_app()
        self.logger = self.app.logger # MCPApp이 생성한 로거를 사용
        self._session = None
        
        failure_threshold = self.settings.get("resilience.circuit_breaker.failure_threshold", 5)
        recovery_timeout = self.settings.get("resilience.circuit_breaker.recovery_timeout", 30)
        self.circuit_breaker = CircuitBreaker(
            fail_max=failure_threshold,
            reset_timeout=recovery_timeout,
        )

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # You can configure the session here if needed
            self._session = aiohttp.ClientSession()
        return self._session

    async def close_session(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _setup_app(self) -> MCPApp:
        """
        새로운 설정 시스템을 사용하여 MCPApp을 설정합니다.
        """
        # 설정에서 MCP 서버 설정을 가져와 MCPAppConfig 형식으로 변환
        mcp_servers_config = {
            name: server.model_dump() 
            for name, server in self.settings.mcp_servers.items()
            if server.enabled and (not self.server_names or name in self.server_names)
        }
        
        app_config = {
            "name": self.name,
            "mcp": mcp_servers_config
        }
        
        return MCPApp(settings=app_config, human_input_callback=None)

    @abstractmethod
    async def run_workflow(self, *args, **kwargs) -> Any:
        """
        에이전트의 핵심 워크플로우를 실행하는 추상 메서드.
        자식 클래스에서 반드시 구현해야 합니다.
        """
        pass

    def get_orchestrator(self, agents: List[MCP_Agent]) -> Orchestrator:
        """
        주어진 에이전트들로 오케스트레이터를 생성합니다.
        """
        return Orchestrator(
            llm_factory=self.app.llm_factory,
            available_agents=agents,
            plan_type="full"
        )

    async def run(self, *args, **kwargs):
        """
        에이전트 워크플로우를 실행하고 결과를 반환합니다.
        API 오류에 대한 재시도 로직과 서킷 브레이커를 포함합니다.
        """
        self.logger.info(f"'{self.name}' 에이전트 워크플로우를 시작합니다.")
        try:
            max_retries = self.settings.get("resilience.retry_attempts", 3)
            retry_delay = self.settings.get("resilience.retry_delay_seconds", 5)
            
            for attempt in range(max_retries):
                try:
                    result = await self.circuit_breaker.call_async(self.run_workflow, *args, **kwargs)

                    self.logger.info(f"'{self.name}' 에이전트 워크플로우를 성공적으로 완료했습니다.")
                    return result
                except CircuitBreakerError as e:
                    self.logger.error(f"서킷 브레이커가 열렸습니다. '{self.name}' 워크플로우를 중단합니다.")
                    raise CircuitBreakerOpen(f"Circuit breaker is open for workflow '{self.name}'") from e
                except APIError as e:
                    self.logger.warning(f"API 오류 발생 (시도 {attempt + 1}/{max_retries}): {e}")
                    if attempt + 1 == max_retries:
                        self.logger.error("최대 재시도 횟수 도달. API 오류로 워크플로우 실패.")
                        raise WorkflowError(f"API Error after {max_retries} retries: {e}") from e
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                except MCPError as e:
                    self.logger.error(f"'{self.name}' 워크플로우 중 처리된 오류 발생: {e}", exc_info=True)
                    raise
                except Exception as e:
                    self.logger.critical(f"'{self.name}' 워크플로우 중 예기치 않은 심각한 오류 발생: {e}", exc_info=True)
                    raise WorkflowError(f"Unexpected error in workflow '{self.name}': {e}") from e
        finally:
            await self.close_session()
            self.logger.info(f"'{self.name}' 에이전트 세션을 정리했습니다.") 