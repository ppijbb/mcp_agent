"""
Base Agent Module

Provides the BaseAgent abstract class for all MCP agents with common functionality
including MCPApp initialization, circuit breaker, async session management, and
workflow execution with retry logic.

Classes:
    BaseAgent: Abstract base class for MCP agents
    
Functions:
    async_memoize: Memoization decorator for async functions
    _cleanup_mcp_apps: Clean up all active MCPApp instances on process exit
    _register_cleanup: Register cleanup handler for MCPApp instances
"""
import asyncio
import functools
import atexit
import threading
from abc import ABC, abstractmethod
from typing import Any, List

# COMPAT: mcp-agent config cache reset for file change reflection
# mcp-agent caches settings on import; reset ensures latest config is used
try:
    import mcp_agent.config
    config_settings = getattr(mcp_agent.config, '_settings', None)
    if config_settings is not None:
        mcp_agent.config._settings = None
except ImportError:
    pass  # Ignore if module structure differs

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

# Global MCPApp instance tracking (for cleanup)
_active_mcp_apps: List[MCPApp] = []
_cleanup_registered = False
_cleanup_lock = threading.Lock()


def _cleanup_mcp_apps():
    """Clean up all active MCPApp instances on process exit."""
    import logging
    logger = logging.getLogger(__name__)
    for app in _active_mcp_apps[:]:
        try:
            if hasattr(app, 'cleanup'):
                if not asyncio.iscoroutinefunction(app.cleanup):
                    app.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up MCPApp: {e}")


def _register_cleanup():
    """Register cleanup handler for MCPApp instances (only in main thread)."""
    global _cleanup_registered
    with _cleanup_lock:
        if not _cleanup_registered:
            if threading.current_thread() is threading.main_thread():
                try:
                    atexit.register(_cleanup_mcp_apps)
                    _cleanup_registered = True
                except (OSError, RuntimeError):
                    pass


def async_memoize(func):
    """
    Memoization decorator for async functions using LRU cache.

    Caches async function results with a maximum cache size of 512 entries.
    Uses cachetools.LRUCache for efficient in-memory caching.

    Args:
        func: The async function to memoize

    Returns:
        Wrapped async function with caching capability
    """
    cache = LRUCache(maxsize=512)

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
    Base class for all MCP agents.
    
    Handles common setup, logging, and MCPApp initialization.
    """
    def __init__(self, name: str, instruction: str = "", server_names: List[str] | None = None):
        """
        Initialize BaseAgent.

        Args:
            name: Unique name for the agent
            instruction: Natural language description of the agent's task
            server_names: List of MCP servers the agent will use
        """
        self.name = name
        self.instruction = instruction
        self.server_names = server_names or []

        self.settings = settings
        self.app = self._setup_app()
        self.logger = self.app.logger
        self._session = None

        ttl = self.settings.cache.ttl
        failure_threshold = max(ttl // 10, 1) if ttl else 5
        recovery_timeout = max(ttl // 20, 1) if ttl else 30
        self.circuit_breaker = CircuitBreaker(
            fail_max=failure_threshold,
            reset_timeout=recovery_timeout,
        )

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session with proper async handling."""
        if self._session is None or self._session.closed:
            # Configure session with reasonable defaults
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close_session(self):
        """Close the aiohttp session safely to prevent resource warnings."""
        if self._session and not self._session.closed:
            await self._session.close()
            await asyncio.sleep(0)
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        """
        Legacy property for backward compatibility. Use get_session() in async contexts.

        Note:
            This property raises an error in async contexts. Prefer get_session() method instead.

        Raises:
            RuntimeError: If used in async context. Use await get_session() instead.
        """
        import warnings
        warnings.warn(
            "session property is deprecated, use get_session() in async contexts",
            DeprecationWarning,
            stacklevel=2
        )
        raise RuntimeError(
            "session property cannot be used in async context. "
            "Use await get_session() instead."
        )

    def _setup_app(self) -> MCPApp:
        """
        Create and configure the MCPApp instance.

        Uses mcp_agent library's get_settings for configuration management.
        Sets up logging and registers cleanup handlers for the app instance.

        Returns:
            MCPApp: Configured MCPApp instance.
        """
        from mcp_agent.config import get_settings
        from pathlib import Path

        # Find config file path from project root
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        config_path = project_root / "mcp_agent.config.yaml"

        # Use mcp_agent library's standard settings
        # COMPAT: Force config cache reset to ensure google section loads
        import mcp_agent.config
        mcp_agent.config._settings = None
        settings = get_settings(str(config_path))

        if not settings.google or not settings.google.api_key:
             # Last resort: inject directly if still None
             import os
             if os.getenv("GOOGLE_API_KEY"):
                 from mcp_agent.config import GoogleSettings
                 settings.google = GoogleSettings(api_key=os.getenv("GOOGLE_API_KEY"))

        app = MCPApp(
            name=self.name,
            settings=settings,
            human_input_callback=None
        )
        # Add to global list (for cleanup)
        _active_mcp_apps.append(app)
        # Try to register cleanup handler (only in main thread)
        _register_cleanup()
        return app

    @abstractmethod
    async def run_workflow(self, *args, **kwargs) -> Any:
        """
        Execute the agent's core workflow.
        
        Must be implemented by child classes.
        """

    def get_orchestrator(self, agents: List[MCP_Agent]) -> Orchestrator:
        """
        Create an orchestrator with the given agents.
        
        Includes fallback support.
        """
        from srcs.common.llm import create_fallback_orchestrator_llm_factory

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
        Execute the agent workflow and return the result.
        
        Includes retry logic for API errors and circuit breaker protection.
        """
        self.logger.info(f"Starting agent workflow for '{self.name}'.")
        try:
            max_retries = 3
            retry_delay = 5

            for attempt in range(max_retries):
                try:
                    result = await self.circuit_breaker.call_async(self.run_workflow, *args, **kwargs)

                    self.logger.info(f"Agent workflow for '{self.name}' completed successfully.")

                    return result
                except CircuitBreakerError as e:
                    self.logger.error(f"Circuit breaker opened. Stopping workflow '{self.name}'.")
                    raise CircuitBreakerOpen(f"Circuit breaker is open for workflow '{self.name}'") from e
                except APIError as e:
                    self.logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt + 1 == max_retries:
                        self.logger.error("Max retries reached. Workflow failed due to API error.")
                        raise WorkflowError(f"API Error after {max_retries} retries: {e}") from e
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                except MCPError as e:
                    self.logger.error(f"Handled error in workflow '{self.name}': {e}", exc_info=True)
                    raise
                except Exception as e:
                    self.logger.critical(f"Unexpected critical error in workflow '{self.name}': {e}", exc_info=True)
                    raise WorkflowError(f"Unexpected error in workflow '{self.name}': {e}") from e
        finally:
            await self.close_session()
            self.logger.info(f"Agent session cleaned up for '{self.name}'.")
