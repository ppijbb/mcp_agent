import asyncio
import functools
import atexit
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent as MCP_Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from srcs.core.config.loader import settings
from srcs.core.errors import APIError, MCPError, WorkflowError, CircuitBreakerOpen
from cachetools import LRUCache
from cachetools.keys import hashkey
from pybreaker import CircuitBreaker, CircuitBreakerError
import aiohttp

# Apply schema fix for fetch-mcp and other servers with non-standard type formats
from srcs.core.agent.schema_fix import patch_transform_mcp_tool_schema

# Patch the schema transformer on module import
patch_transform_mcp_tool_schema()

# 전역 MCPApp 인스턴스 추적 (cleanup용)
_active_mcp_apps: List[MCPApp] = []
_cleanup_registered = False
_cleanup_lock = threading.Lock()

def _cleanup_mcp_apps():
    """모든 활성 MCPApp 인스턴스 정리"""
    import logging
    logger = logging.getLogger(__name__)
    for app in _active_mcp_apps[:]:  # 복사본으로 순회 (리스트 변경 가능)
        try:
            # MCPApp이 cleanup 메서드를 가지고 있는지 확인
            if hasattr(app, 'cleanup'):
                # 동기 cleanup이면 직접 호출, 비동기면 무시 (atexit에서는 async 불가)
                if not asyncio.iscoroutinefunction(app.cleanup):
                    app.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up MCPApp: {e}")

# 프로세스 종료 시 cleanup 등록 (한 번만)
def _register_cleanup():
    """cleanup 핸들러 등록 (메인 스레드에서만)"""
    global _cleanup_registered
    with _cleanup_lock:
        if not _cleanup_registered:
            # 메인 스레드인지 확인
            if threading.current_thread() is threading.main_thread():
                try:
                    atexit.register(_cleanup_mcp_apps)
                    _cleanup_registered = True
                except Exception:
                    pass  # 등록 실패해도 계속 진행


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
        
        # Pydantic 모델은 .get() 메서드가 없으므로 기본값 직접 사용
        failure_threshold = 5
        recovery_timeout = 30
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
        MCPApp을 설정합니다. mcp_agent 라이브러리의 get_settings를 사용합니다.
        """
        from mcp_agent.config import get_settings
        from pathlib import Path
        
        # 프로젝트 루트에서 설정 파일 경로 찾기
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        config_path = project_root / "mcp_agent.config.yaml"
        
        # mcp_agent 라이브러리의 표준 설정 사용
        # set_global=False로 설정하여 멀티스레드 환경에서 안전하게 사용
        settings = get_settings(str(config_path), set_global=False)
        
        app = MCPApp(
            name=self.name,
            settings=settings,
            human_input_callback=None
        )
        # 전역 리스트에 추가 (cleanup용)
        _active_mcp_apps.append(app)
        # cleanup 핸들러 등록 시도 (메인 스레드에서만)
        _register_cleanup()
        return app

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
        Fallback 지원 포함.
        """
        from srcs.common.llm import create_fallback_orchestrator_llm_factory
        
        # Fallback이 가능한 LLM factory 사용 (common 모듈)
        llm_factory = create_fallback_orchestrator_llm_factory(
            primary_model="gemini-2.5-flash",
            logger_instance=self.logger
        )
        
        return Orchestrator(
            llm_factory=llm_factory,
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
            # Pydantic 모델은 .get() 메서드가 없으므로 기본값 직접 사용
            max_retries = 3
            retry_delay = 5
            
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