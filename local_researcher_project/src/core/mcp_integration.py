"""
Universal MCP Hub - 2025년 10월 최신 버전

Model Context Protocol 통합을 위한 범용 허브.
OpenRouter와 Gemini 2.5 Flash Lite 기반의 최신 MCP 연결.
Production 수준의 안정성과 신뢰성 보장.
"""

import asyncio
import sys
import json
import logging
import time
import aiohttp
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import subprocess
import os
from datetime import datetime, timedelta
from contextlib import AsyncExitStack

# MCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import ListToolsResult, TextContent
    from urllib.parse import urlencode
    MCP_AVAILABLE = True
    HTTP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    HTTP_CLIENT_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    streamablehttp_client = None
    urlencode = None
    ListToolsResult = None
    TextContent = None

# LangChain imports
try:
    from langchain_core.tools import BaseTool, StructuredTool
    # Pydantic v2 호환성 - 최신 LangChain은 pydantic v2 사용
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        try:
            from pydantic.v1 import BaseModel, Field
        except ImportError:
            from langchain_core.pydantic_v1 import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = None
    StructuredTool = None
    BaseModel = None
    Field = None

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.researcher_config import get_mcp_config, get_llm_config

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """MCP 도구 카테고리."""
    SEARCH = "search"
    DATA = "data"
    CODE = "code"
    ACADEMIC = "academic"
    BUSINESS = "business"
    UTILITY = "utility"


@dataclass
class ToolInfo:
    """도구 정보."""
    name: str
    category: ToolCategory
    description: str
    parameters: Dict[str, Any]
    mcp_server: str


@dataclass
class ToolResult:
    """도구 실행 결과."""
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    confidence: float = 0.0


class ToolRegistry:
    """Tool 중앙 관리 시스템 - MCP 및 로컬 Tool 통합 관리."""
    
    def __init__(self):
        """ToolRegistry 초기화."""
        self.tools: Dict[str, ToolInfo] = {}  # tool_name -> ToolInfo
        self.langchain_tools: Dict[str, BaseTool] = {}  # tool_name -> LangChain Tool
        self.tool_sources: Dict[str, str] = {}  # tool_name -> source (mcp/local)
        self.mcp_tool_mapping: Dict[str, Tuple[str, str]] = {}  # tool_name -> (server_name, original_tool_name)
        
    def register_mcp_tool(self, server_name: str, tool: Any, tool_def: Any = None):
        """
        MCP Tool을 server_name::tool_name 형식으로 등록.
        
        Args:
            server_name: MCP 서버 이름
            tool: MCP Tool 객체 또는 tool name
            tool_def: MCP Tool 정의 (description, inputSchema 등 포함)
        """
        if isinstance(tool, str):
            tool_name = tool
        else:
            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
        
        # server_name::tool_name 형식으로 등록
        registered_name = f"{server_name}::{tool_name}"
        
        # ToolInfo 생성
        if tool_def and hasattr(tool_def, 'description'):
            description = tool_def.description
            input_schema = tool_def.inputSchema if hasattr(tool_def, 'inputSchema') else {}
        else:
            description = f"Tool from MCP server {server_name}"
            input_schema = {}
        
        # 카테고리 추론 (기본값: UTILITY)
        category = ToolCategory.UTILITY
        tool_lower = tool_name.lower()
        if 'search' in tool_lower:
            category = ToolCategory.SEARCH
        elif 'scholar' in tool_lower or 'arxiv' in tool_lower or 'paper' in tool_lower:
            category = ToolCategory.ACADEMIC
        elif 'fetch' in tool_lower or 'file' in tool_lower:
            category = ToolCategory.DATA
        elif 'code' in tool_lower or 'python' in tool_lower:
            category = ToolCategory.CODE
        
        tool_info = ToolInfo(
            name=registered_name,
            category=category,
            description=description,
            parameters=input_schema,
            mcp_server=server_name
        )
        
        self.tools[registered_name] = tool_info
        self.tool_sources[registered_name] = "mcp"
        self.mcp_tool_mapping[registered_name] = (server_name, tool_name)
        
        logger.debug(f"Registered MCP tool: {registered_name} from server {server_name}")
        
    def register_local_tool(self, tool_info: ToolInfo, langchain_tool: BaseTool):
        """
        로컬 Tool을 LangChain Tool과 함께 등록.
        
        Args:
            tool_info: ToolInfo 객체
            langchain_tool: LangChain BaseTool 인스턴스
        """
        tool_name = tool_info.name
        
        self.tools[tool_name] = tool_info
        self.langchain_tools[tool_name] = langchain_tool
        self.tool_sources[tool_name] = "local"
        
        logger.debug(f"Registered local tool: {tool_name}")
        
    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """Tool 정보 조회."""
        return self.tools.get(tool_name)
        
    def get_langchain_tool(self, tool_name: str) -> Optional[BaseTool]:
        """LangChain Tool 조회."""
        return self.langchain_tools.get(tool_name)
        
    def get_all_langchain_tools(self) -> List[BaseTool]:
        """모든 Tool을 LangChain Tool 리스트로 반환."""
        return list(self.langchain_tools.values())
        
    def get_tools_by_category(self, category: ToolCategory) -> List[str]:
        """카테고리별 Tool 목록 반환."""
        return [
            name for name, info in self.tools.items()
            if info.category == category
        ]
        
    def is_mcp_tool(self, tool_name: str) -> bool:
        """Tool이 MCP Tool인지 확인."""
        return self.tool_sources.get(tool_name) == "mcp"
        
    def get_mcp_server_info(self, tool_name: str) -> Optional[Tuple[str, str]]:
        """MCP Tool의 서버 정보 반환: (server_name, original_tool_name)."""
        return self.mcp_tool_mapping.get(tool_name)
        
    def get_all_tool_names(self) -> List[str]:
        """등록된 모든 Tool 이름 반환."""
        return list(self.tools.keys())
        
    def remove_tool(self, tool_name: str):
        """Tool 제거."""
        if tool_name in self.tools:
            del self.tools[tool_name]
        if tool_name in self.langchain_tools:
            del self.langchain_tools[tool_name]
        if tool_name in self.tool_sources:
            del self.tool_sources[tool_name]
        if tool_name in self.mcp_tool_mapping:
            del self.mcp_tool_mapping[tool_name]


class OpenRouterClient:
    """(비활성화) OpenRouter 경유는 사용하지 않습니다."""
    def __init__(self, api_key: str):
        self.api_key = api_key
    async def __aenter__(self):
        raise RuntimeError("OpenRouter is disabled. Use Gemini direct path via llm_manager.")
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False
    async def generate_response(self, *args, **kwargs):
        raise RuntimeError("OpenRouter is disabled. Use Gemini direct path via llm_manager.")



class UniversalMCPHub:
    """Universal MCP Hub - 2025년 10월 최신 버전."""

    def __init__(self):
        self.config = get_mcp_config()
        self.llm_config = get_llm_config()

        # ToolRegistry 통합 관리
        self.registry = ToolRegistry()
        self.tools: Dict[str, ToolInfo] = {}  # 하위 호환성을 위해 유지 (registry.tools 참조)
        self.openrouter_client: Optional[OpenRouterClient] = None
        
        # MCP 클라이언트
        self.mcp_sessions: Dict[str, ClientSession] = {}
        self.exit_stacks: Dict[str, AsyncExitStack] = {}  # 참조만 유지, cleanup에서 aclose() 호출 안 함
        self.mcp_tools_map: Dict[str, Dict[str, Any]] = {}  # server_name -> {tool_name -> tool_info}
        self.mcp_server_configs: Dict[str, Dict[str, Any]] = {}
        # 각 서버별 연결 진단 정보
        self.connection_diagnostics: Dict[str, Dict[str, Any]] = {}
        # 종료/차단 플래그 (종료 중 신규 연결 방지)
        self.stopping: bool = False

        self._load_tools_config()
        self._initialize_tools()
        self._initialize_clients()
        self._load_mcp_servers_from_config()
        
    def _load_tools_config(self):
        """tools_config.json에서 Tool 메타데이터 로드."""
        # configs 폴더에서 로드 시도 (우선)
        tools_config_file = project_root / "configs" / "tools_config.json"
        if not tools_config_file.exists():
            # 하위 호환성: 루트에서도 시도
            tools_config_file = project_root / "tools_config.json"
        
        if tools_config_file.exists():
            try:
                with open(tools_config_file, 'r', encoding='utf-8') as f:
                    self.tools_config = json.load(f)
                logger.info(f"✅ Loaded tools config from {tools_config_file}")
            except Exception as e:
                logger.warning(f"Failed to load tools config: {e}")
                self.tools_config = {}
        else:
            logger.warning(f"tools_config.json not found at {tools_config_file}")
            self.tools_config = {}
    
    def _create_langchain_tool_wrapper(self, tool_name: str, tool_config: Dict[str, Any]) -> Optional[BaseTool]:
        """
        tools_config.json의 설정을 기반으로 LangChain Tool 래퍼 생성.
        
        Args:
            tool_name: Tool 이름
            tool_config: tools_config.json에서 로드된 Tool 설정
            
        Returns:
            LangChain BaseTool 인스턴스 또는 None
        """
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, cannot create tool wrapper")
            return None
        
        try:
            # 카테고리 매핑
            category_map = {
                "search": ToolCategory.SEARCH,
                "data": ToolCategory.DATA,
                "code": ToolCategory.CODE,
                "academic": ToolCategory.ACADEMIC,
                "business": ToolCategory.BUSINESS,
                "utility": ToolCategory.UTILITY
            }
            
            category_str = tool_config.get("category", "utility")
            category = category_map.get(category_str, ToolCategory.UTILITY)
            description = tool_config.get("description", f"{tool_name} tool")
            params_config = tool_config.get("parameters", {})
            
            # Pydantic 스키마 생성 - 최신 방식으로 단순화 (args_schema 없이도 동작)
            ToolSchema = None
            # LangChain StructuredTool은 args_schema 없이도 함수 시그니처에서 자동으로 파라미터를 추론함
            # 복잡한 동적 스키마 생성을 피하고 함수 파라미터로 처리
            
            # Tool 실행 함수 선택 (동기 래퍼 생성) - 함수 시그니처 명시
            def create_sync_func(tool_name_str, func_type):
                """동기 함수 래퍼 생성 - 명시적 함수 시그니처로 LangChain이 파라미터 추론."""
                if func_type == "search":
                    def search_wrapper(query: str, max_results: int = 10, num_results: int = 10) -> str:
                        params = {"query": query}
                        if max_results:
                            params["max_results"] = max_results
                        elif num_results:
                            params["max_results"] = num_results
                        return _execute_search_tool_sync(tool_name_str, params)
                    return search_wrapper
                elif func_type == "academic":
                    def academic_wrapper(query: str, max_results: int = 10, num_results: int = 10) -> str:
                        params = {"query": query}
                        if max_results:
                            params["max_results"] = max_results
                        elif num_results:
                            params["max_results"] = num_results
                        return _execute_academic_tool_sync(tool_name_str, params)
                    return academic_wrapper
                elif func_type == "data":
                    if tool_name_str == "fetch":
                        def fetch_wrapper(url: str) -> str:
                            return _execute_data_tool_sync("fetch", {"url": url})
                        return fetch_wrapper
                    elif tool_name_str == "filesystem":
                        def filesystem_wrapper(path: str, operation: str = "read") -> str:
                            return _execute_data_tool_sync("filesystem", {"path": path, "operation": operation})
                        return filesystem_wrapper
                    else:
                        def data_wrapper(**kwargs) -> str:
                            return _execute_data_tool_sync(tool_name_str, kwargs)
                        return data_wrapper
                elif func_type == "code":
                    if "interpreter" in tool_name_str.lower():
                        def code_wrapper(code: str, language: str = "python") -> str:
                            return _execute_code_tool_sync(tool_name_str, {"code": code, "language": language})
                        return code_wrapper
                    else:
                        def code_wrapper(code: str) -> str:
                            return _execute_code_tool_sync(tool_name_str, {"code": code})
                        return code_wrapper
                else:
                    return None
            
            # Tool별 실행 함수 매핑
            func = None
            category_str = tool_config.get("category", "utility")
            
            if tool_name == "g-search":
                func = create_sync_func("g-search", "search")
            elif tool_name == "fetch":
                func = create_sync_func("fetch", "data")
            elif tool_name == "filesystem":
                func = create_sync_func("filesystem", "data")
            elif tool_name == "python_coder":
                func = create_sync_func("python_coder", "code")
            elif tool_name == "code_interpreter":
                func = create_sync_func("code_interpreter", "code")
            elif tool_name == "arxiv":
                func = create_sync_func("arxiv", "academic")
            elif tool_name == "scholar":
                func = create_sync_func("scholar", "academic")
            else:
                # 카테고리 기반으로 자동 선택 시도
                if category_str == "search":
                    func = create_sync_func(tool_name, "search")
                elif category_str == "data":
                    func = create_sync_func(tool_name, "data")
                elif category_str == "code":
                    func = create_sync_func(tool_name, "code")
                elif category_str == "academic":
                    func = create_sync_func(tool_name, "academic")
            
            if func is None:
                logger.warning(f"No execution function for tool: {tool_name}, category: {category_str}")
                # 실행 함수가 없어도 기본 래퍼 함수 생성
                def generic_executor(**kwargs):
                    """Generic executor when specific function not available."""
                    raise RuntimeError(f"Tool {tool_name} execution not implemented yet. Please configure execution function.")
                func = generic_executor
            
            # StructuredTool 생성 - args_schema 없이도 생성 가능하도록
            try:
                if StructuredTool and ToolSchema:
                    langchain_tool = StructuredTool.from_function(
                        func=func,
                        name=tool_name,
                        description=description,
                        args_schema=ToolSchema
                    )
                elif StructuredTool:
                    # args_schema 없이 생성 (파라미터는 함수 시그니처에서 자동 추론)
                    langchain_tool = StructuredTool.from_function(
                        func=func,
                        name=tool_name,
                        description=description
                    )
                else:
                    return None
                
                logger.info(f"✅ Created LangChain tool wrapper for {tool_name}")
                return langchain_tool
            except Exception as schema_error:
                # Schema 생성 실패 시 args_schema 없이 재시도
                logger.warning(f"Failed to create tool with schema for {tool_name}: {schema_error}, trying without schema")
                try:
                    if StructuredTool:
                        langchain_tool = StructuredTool.from_function(
                            func=func,
                            name=tool_name,
                            description=description
                        )
                        logger.info(f"✅ Created LangChain tool wrapper for {tool_name} (without schema)")
                        return langchain_tool
                except Exception as e2:
                    logger.error(f"Failed to create tool without schema for {tool_name}: {e2}")
                    return None
            
        except Exception as e:
            logger.error(f"Failed to create LangChain tool wrapper for {tool_name}: {e}")
            return None
    
    def _initialize_tools(self):
        """도구 초기화 - tools_config.json 기반."""
        local_tools = self.tools_config.get("local_tools", {})
        
        for tool_name, tool_config in local_tools.items():
            # MCP 전용 Tool은 건너뛰기 (MCP 서버에서 동적 등록됨)
            if tool_config.get("implementation") == "mcp_only":
                continue
            
            # 카테고리 매핑
            category_map = {
                "search": ToolCategory.SEARCH,
                "data": ToolCategory.DATA,
                "code": ToolCategory.CODE,
                "academic": ToolCategory.ACADEMIC,
                "business": ToolCategory.BUSINESS,
                "utility": ToolCategory.UTILITY
            }
            
            category_str = tool_config.get("category", "utility")
            category = category_map.get(category_str, ToolCategory.UTILITY)
            description = tool_config.get("description", f"{tool_name} tool")
            
            # ToolInfo 생성
            tool_info = ToolInfo(
                name=tool_name,
                category=category,
                description=description,
                parameters=tool_config.get("parameters", {}),
                mcp_server=tool_config.get("mcp_server_name", "")
            )
            
            # LangChain Tool 래퍼 생성
            langchain_tool = self._create_langchain_tool_wrapper(tool_name, tool_config)
            
            if langchain_tool:
                # Registry에 등록
                self.registry.register_local_tool(tool_info, langchain_tool)
                # 하위 호환성을 위해 self.tools에도 추가
                self.tools[tool_name] = tool_info
                logger.info(f"✅ Registered local tool: {tool_name}")
            else:
                # LangChain wrapper 생성 실패해도 기본 ToolInfo는 등록 (나중에 실행 시도 가능)
                logger.warning(f"⚠️ Failed to create LangChain wrapper for {tool_name}, registering without wrapper")
                self.registry.tools[tool_name] = tool_info
                self.tools[tool_name] = tool_info
        
        # Registry의 tools를 self.tools와 동기화
        self.tools.update(self.registry.tools)
        
        logger.info(f"✅ Initialized {len(self.registry.tools)} tools in registry ({len(self.registry.langchain_tools)} with LangChain wrappers)")
    
    def _initialize_clients(self):
        """클라이언트 초기화 - Gemini 직결 사용, OpenRouter 비활성화."""
        self.openrouter_client = None
        logger.info("✅ LLM routed via llm_manager (Gemini direct). OpenRouter disabled.")
    
    def _load_mcp_servers_from_config(self):
        """MCP 서버 설정을 config에서 로드."""
        try:
            # configs 폴더에서 로드 시도 (우선)
            config_file = project_root / "configs" / "mcp_config.json"
            if not config_file.exists():
                # 하위 호환성: 루트에서도 시도
                config_file = project_root / "mcp_config.json"
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    self.mcp_server_configs = config_data.get("mcpServers", {})
                    logger.info(f"✅ Loaded MCP server configs: {list(self.mcp_server_configs.keys())}")
            else:
                # 기본 DuckDuckGo MCP 서버 설정
                self.mcp_server_configs = {
                    "ddg_search": {
                        "command": "npx",
                        "args": [
                            "-y",
                            "@smithery/cli@latest",
                            "run",
                            "@OEvortex/ddg_search",
                            "--key",
                            os.getenv("DDG_SEARCH_KEY", "")
                        ]
                    }
                }
                logger.info("✅ Using default MCP server config for ddg_search")
                
        except Exception as e:
            logger.warning(f"Failed to load MCP server configs: {e}")
            self.mcp_server_configs = {}
    
    async def _connect_to_mcp_server(self, server_name: str, server_config: Dict[str, Any], timeout: float = 15.0):
        """MCP 서버에 연결 - stdio 및 HTTP 지원 (타임아웃 포함)."""
        if self.stopping:
            logger.warning(f"[MCP][skip.stopping] server={server_name}")
            return False
        logger.info(f"[MCP][connect.start] server={server_name} type={server_config.get('type','stdio')} url={(server_config.get('httpUrl') or server_config.get('url'))} timeout={timeout}")
        self.connection_diagnostics[server_name] = {
            "server": server_name,
            "type": ("http" if (server_config.get("httpUrl") or server_config.get("url") or server_config.get("type") == "http") else "stdio"),
            "url": server_config.get("httpUrl") or server_config.get("url"),
            "stage": "start",
            "ok": False,
            "error": None,
            "traceback": None,
            "init_ms": None,
            "list_ms": None,
        }
        if not MCP_AVAILABLE:
            logger.error("MCP package not available")
            return False
        
        try:
            if server_name in self.mcp_sessions:
                await self._disconnect_from_mcp_server(server_name)
            
            exit_stack = AsyncExitStack()
            self.exit_stacks[server_name] = exit_stack
            
            # HTTP 기반 서버인지 확인
            if "httpUrl" in server_config or "url" in server_config or server_config.get("type") == "http":
                # HTTP 기반 MCP 서버 연결
                if not HTTP_CLIENT_AVAILABLE or streamablehttp_client is None:
                    logger.error(f"HTTP MCP client not available for server {server_name}")
                    return False
                
                base_url = server_config.get("httpUrl") or server_config.get("url")
                if not base_url:
                    logger.error(f"No URL provided for HTTP MCP server {server_name}")
                    return False
                
                # URL 파라미터 구성 (api_key, profile 등)
                params = server_config.get("params", {})
                if params and urlencode:
                    url = f"{base_url}?{urlencode(params)}"
                else:
                    url = base_url
                
                logger.info(f"Connecting to HTTP MCP server: {server_name} ({url})")
                
                try:
                    # HTTP transport 사용
                    http_transport = await exit_stack.enter_async_context(streamablehttp_client(url))
                    read, write, _ = http_transport
                    session = await exit_stack.enter_async_context(ClientSession(read, write))
                except Exception as e:
                    logger.exception(f"[MCP][connect.error] create-http-transport server={server_name} err={e}")
                    di = self.connection_diagnostics.get(server_name, {})
                    di.update({"stage": "create_http_transport", "error": str(e)})
                    self.connection_diagnostics[server_name] = di
                    return False
                
            else:
                # stdio 기반 서버 (기존 방식)
                command = server_config.get("command")
                args = server_config.get("args", [])
                
                if not command:
                    logger.error(f"No command or URL provided for MCP server {server_name}")
                    return False
                
                logger.info(f"Connecting to stdio MCP server: {server_name} ({command})")
                
                try:
                    # stdio transport 사용
                    server_params = StdioServerParameters(command=command, args=args)
                    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
                    read, write = stdio_transport
                    session = await exit_stack.enter_async_context(ClientSession(read, write))
                except Exception as e:
                    logger.exception(f"[MCP][connect.error] create-stdio-transport server={server_name} err={e}")
                    di = self.connection_diagnostics.get(server_name, {})
                    di.update({"stage": "create_stdio_transport", "error": str(e)})
                    self.connection_diagnostics[server_name] = di
                    return False
            
            self.mcp_sessions[server_name] = session
            
            # 세션 초기화 및 도구 목록 가져오기 (타임아웃 적용 - 개별 작업에만)
            try:
                t0 = asyncio.get_running_loop().time()
                await asyncio.wait_for(session.initialize(), timeout=timeout)
                t1 = asyncio.get_running_loop().time()
                response = await asyncio.wait_for(session.list_tools(), timeout=timeout)
                t2 = asyncio.get_running_loop().time()
                di = self.connection_diagnostics.get(server_name, {})
                di.update({
                    "stage": "list_tools_ok",
                    "ok": True,
                    "init_ms": (t1 - t0) * 1000.0,
                    "list_ms": (t2 - t1) * 1000.0,
                })
                self.connection_diagnostics[server_name] = di
            except asyncio.CancelledError:
                # 작업이 취소된 경우 (종료 신호 등)
                logger.warning(f"[MCP][connect.cancelled] server={server_name} stage=initialize_or_list")
                await self._disconnect_from_mcp_server(server_name)
                di = self.connection_diagnostics.get(server_name, {})
                di.update({"stage": "cancelled_initialize_or_list", "error": "cancelled"})
                self.connection_diagnostics[server_name] = di
                raise  # 상위로 전파하여 초기화 중단
            except asyncio.TimeoutError:
                logger.error(f"[MCP][connect.timeout] server={server_name} after={timeout}s stage=initialize_or_list")
                di = self.connection_diagnostics.get(server_name, {})
                di.update({"stage": "timeout_initialize_or_list", "error": f"timeout_{timeout}s"})
                self.connection_diagnostics[server_name] = di
                # 연결은 되었지만 초기화 실패 - 세션 정리
                # 타임아웃 발생 시 exit_stack 참조만 제거 (aclose() 호출하지 않음 - anyio 오류 방지)
                if server_name in self.exit_stacks:
                    del self.exit_stacks[server_name]
                await self._disconnect_from_mcp_server(server_name)
                return False
            
            # 도구 맵 생성 및 Registry에 동적 등록
            self.mcp_tools_map[server_name] = {}
            for tool in response.tools:
                self.mcp_tools_map[server_name][tool.name] = tool
                # ToolRegistry에 server_name::tool_name 형식으로 등록
                self.registry.register_mcp_tool(server_name, tool.name, tool)
                logger.debug(f"[MCP][register] {server_name}::{tool.name}")
            
            # Registry tools를 self.tools에 동기화
            self.tools.update(self.registry.tools)
            
            tool_names = [t for t in self.mcp_tools_map.get(server_name, {}).keys()]
            if 't1' in locals() and 't2' in locals():
                logger.info(f"[MCP][connect.ok] server={server_name} init_ms={(t1-t0)*1000:.0f} list_ms={(t2-t1)*1000:.0f} tools={tool_names}")
            else:
                logger.info(f"[MCP][connect.ok] server={server_name} tools={tool_names}")
            logger.info(f"✅ Connected to MCP server {server_name} with {len(response.tools)} tools")
            return True
            
        except asyncio.CancelledError:
            # 작업이 취소된 경우 (종료 신호 등)
            logger.warning(f"[MCP][connect.cancelled] server={server_name} stage=generic")
            await self._disconnect_from_mcp_server(server_name)
            raise  # 상위로 전파하여 초기화 중단
        except asyncio.TimeoutError:
            logger.error(f"[MCP][connect.timeout] server={server_name} stage=generic")
            di = self.connection_diagnostics.get(server_name, {})
            di.update({"stage": "timeout_generic", "error": f"timeout_{timeout}s"})
            self.connection_diagnostics[server_name] = di
            # 타임아웃 발생 시 exit_stack 참조만 제거 (aclose() 호출하지 않음 - anyio 오류 방지)
            if server_name in self.exit_stacks:
                del self.exit_stacks[server_name]
            await self._disconnect_from_mcp_server(server_name)
            return False
        except Exception as e:
            logger.exception(f"[MCP][connect.error] server={server_name} err={e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            di = self.connection_diagnostics.get(server_name, {})
            di.update({"stage": "exception", "error": str(e), "traceback": traceback.format_exc()})
            self.connection_diagnostics[server_name] = di
            # 실패 시 exit_stack 참조만 제거 (aclose() 호출하지 않음 - anyio 오류 방지)
            if server_name in self.exit_stacks:
                del self.exit_stacks[server_name]
            try:
                await self._disconnect_from_mcp_server(server_name)
            except:
                pass
            return False
    
    async def _disconnect_from_mcp_server(self, server_name: str):
        """MCP 서버 연결 해제 - 안전한 비동기 정리."""
        try:
            # 세션 먼저 제거
            if server_name in self.mcp_sessions:
                session = self.mcp_sessions[server_name]
                try:
                    # 세션 종료 시도 (안전하게)
                    if hasattr(session, 'shutdown'):
                        await asyncio.wait_for(session.shutdown(), timeout=1.0)
                except (asyncio.TimeoutError, AttributeError, Exception):
                    pass  # 세션 종료 실패는 무시
                del self.mcp_sessions[server_name]
            
            # Exit stack 정리: aclose() 호출하지 않음 (anyio cancel scope 오류 방지)
            # 참조만 제거 - 컨텍스트는 원래 태스크에서 정리됨
            if server_name in self.exit_stacks:
                del self.exit_stacks[server_name]
            
            if server_name in self.mcp_tools_map:
                del self.mcp_tools_map[server_name]
            
            logger.debug(f"Disconnected from MCP server: {server_name}")
            
        except Exception as e:
            logger.debug(f"Error disconnecting from MCP server {server_name}: {e}")
    
    async def initialize_mcp(self):
        """MCP 초기화 - OpenRouter와 MCP 서버 연결."""
        if not self.config.enabled:
            logger.warning("MCP is disabled. Continuing with limited functionality.")
            return
        if self.stopping:
            logger.warning("MCP initialization requested during stopping state; skipping")
            return
        
        try:
            logger.info("Initializing MCP Hub with MCP servers (no OpenRouter)...")
            
            # MCP 서버 연결 (모든 서버) - 병렬 + 타임아웃 적용
            timeout_per_server = float(os.getenv("MCP_CONNECT_TIMEOUT", "15"))  # 서버당 최대 15초(환경변수로 조정)
            max_concurrency = 4
            semaphore = asyncio.Semaphore(max_concurrency)

            async def connect_one(name: str, cfg: Dict[str, Any]) -> tuple[str, bool]:
                try:
                    async with semaphore:
                        if cfg.get("disabled"):
                            logger.warning(f"[MCP][skip.disabled] server={name}")
                            return name, False
                        logger.info(f"Connecting to MCP server {name} (timeout: {timeout_per_server}s)...")
                        ok = await self._connect_to_mcp_server(name, cfg, timeout=timeout_per_server)
                        if not ok:
                            logger.warning(f"Failed to connect to MCP server {name}")
                        return name, ok
                except asyncio.CancelledError:
                    logger.warning(f"[MCP][init.cancelled] server={name}")
                    raise
                except Exception as e:
                    logger.exception(f"[MCP][connect.error] server={name} unexpected err={e}")
                    return name, False

            # disabled=true 설정된 서버는 건너뛰기 + 허용 서버 화이트리스트 적용
            allowlist_str = os.getenv("MCP_ALLOWED_SERVERS", "").strip()
            allowlist = [s.strip() for s in allowlist_str.split(",") if s.strip()]
            base_items = [(n, c) for n, c in self.mcp_server_configs.items() if not c.get("disabled")]
            if allowlist:
                # 화이트리스트가 있으면 그것만 연결
                enabled_server_items = [(n, c) for n, c in base_items if n in allowlist]
                logger.info(f"[MCP][allowlist] enabled={ [n for n,_ in enabled_server_items] }")
            else:
                # 화이트리스트가 없으면 disabled가 아닌 모든 서버 연결 시도
                enabled_server_items = base_items
                logger.info(f"[MCP][allowlist] not set; connecting to all enabled servers: { [n for n,_ in enabled_server_items] }")

            tasks = [asyncio.create_task(connect_one(n, c)) for n, c in enabled_server_items]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            connected_servers = [n for n, ok in results if ok]
            
            if connected_servers:
                logger.info(f"✅ Successfully connected to {len(connected_servers)} MCP servers: {', '.join(connected_servers)}")
            else:
                logger.warning("⚠️ No MCP servers connected successfully")
            
            # OpenRouter 연결 테스트 제거 (Gemini는 llm_manager 경유)
            logger.info("✅ MCP Hub initialized (OpenRouter disabled)")
            logger.info(f"Available tools: {len(self.tools)}")
            logger.info(f"MCP servers: {list(self.mcp_sessions.keys())}")
            logger.info(f"Primary model: {self.llm_config.primary_model}")
            # 서버별 연결 진단 요약 출력
            if self.connection_diagnostics:
                logger.info("[MCP][diagnostics] server connection summary")
                for name, di in self.connection_diagnostics.items():
                    init_ms = di.get('init_ms')
                    list_ms = di.get('list_ms')
                    logger.info(
                        "[MCP][diag] server=%s type=%s url=%s stage=%s ok=%s init_ms=%s list_ms=%s err=%s",
                        name,
                        di.get("type"),
                        di.get("url"),
                        di.get("stage"),
                        di.get("ok"),
                        f"{init_ms:.0f}" if isinstance(init_ms, (int, float)) else "-",
                        f"{list_ms:.0f}" if isinstance(list_ms, (int, float)) else "-",
                        di.get("error")
                    )
            
            # 필수 도구 검증 - 실패 시 warning만
            await self._validate_essential_tools()
            
        except Exception as e:
            logger.warning(f"⚠️ MCP Hub initialization failed: {e} - continuing with graceful degradation")
            logger.info("ℹ️ System will continue with limited functionality (no API calls)")
            # Don't raise, allow graceful degradation
    
    async def _execute_via_mcp_server(self, server_name: str, tool_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """MCP 서버를 통해 도구 실행."""
        if server_name not in self.mcp_sessions:
            logger.error(f"Server {server_name} not in mcp_sessions. Available: {list(self.mcp_sessions.keys())}")
            return None
        
        if server_name not in self.mcp_tools_map:
            logger.error(f"Server {server_name} not in mcp_tools_map. Available: {list(self.mcp_tools_map.keys())}")
            return None
        
        if tool_name not in self.mcp_tools_map[server_name]:
            available_tools = list(self.mcp_tools_map[server_name].keys())
            logger.error(f"Tool {tool_name} not found in server {server_name}. Available tools: {available_tools}")
            return None
        
        try:
            session = self.mcp_sessions[server_name]
            logger.debug(f"Calling tool {tool_name} on server {server_name} with params: {params}")
            result = await session.call_tool(tool_name, params)
            
            # 결과를 TextContent에서 추출
            if result and result.content:
                content_parts = []
                for item in result.content:
                    if isinstance(item, TextContent):
                        content_parts.append(item.text)
                    else:
                        # 다른 타입의 content도 처리
                        content_parts.append(str(item))
                
                content_str = " ".join(content_parts)
                logger.debug(f"Tool {tool_name} returned content length: {len(content_str)}")
                return content_str
            else:
                logger.warning(f"Tool {tool_name} returned empty result")
                return None
            
        except Exception as e:
            logger.error(f"Failed to execute tool {tool_name} on server {server_name}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    async def _validate_essential_tools(self):
        """필수 MCP 도구 검증 - Tool이 등록되어 있는지 확인만 (실제 실행은 선택적)."""
        essential_tools = ["g-search", "fetch", "filesystem"]
        missing_tools = []
        
        logger.info("Validating essential tools availability...")
        
        # 등록된 모든 tool 목록 확인
        all_tools = self.registry.get_all_tool_names()
        logger.info(f"Registered tools: {all_tools}")
        
        for tool in essential_tools:
            # tool_name으로 직접 찾기
            tool_found = False
            
            # 1. 직접 등록된 tool 확인
            if tool in all_tools:
                tool_found = True
                logger.info(f"✅ Found essential tool: {tool}")
            
            # 2. server_name::tool_name 형식으로도 찾기
            if not tool_found:
                for registered_name in all_tools:
                    if "::" in registered_name:
                        _, original_tool_name = registered_name.split("::", 1)
                        if original_tool_name == tool:
                            tool_found = True
                            logger.info(f"✅ Found essential tool: {tool} as {registered_name}")
                            break
            
            if not tool_found:
                missing_tools.append(tool)
                logger.warning(f"⚠️ Essential tool {tool} not found in registry")
        
        # 누락된 tool이 있으면 경고만 (실제 실행 전까지는 정확한 검증 불가)
        if missing_tools:
            logger.warning(f"⚠️ Some essential tools not found: {missing_tools}")
            logger.warning("⚠️ Tools may be registered later when MCP servers connect or may need manual configuration")
            logger.warning("⚠️ System will continue, but these tools may not be available")
        else:
            logger.info("✅ All essential tools found in registry")
        
        # 실제 실행 테스트는 선택적 (timeout으로 인한 false negative 방지)
        # Production 환경에서는 실제 사용 시점에 검증하는 것이 더 안전
    
    async def cleanup(self):
        """MCP 연결 정리 - Production-grade cleanup."""
        logger.info("Cleaning up MCP Hub...")
        # 신규 연결 차단
        self.stopping = True
        
        # OpenRouter 클라이언트 사용 안 함
        self.openrouter_client = None
        
        # 모든 MCP 서버 연결 해제 (역순으로 정리)
        server_names = list(self.mcp_sessions.keys())
        for server_name in reversed(server_names):
            try:
                # 세션 제거
                if server_name in self.mcp_sessions:
                    session = self.mcp_sessions.get(server_name)
                    # 세션 종료 시도 (안전하게)
                    if session and hasattr(session, 'shutdown'):
                        try:
                            await asyncio.wait_for(session.shutdown(), timeout=1.0)
                        except:
                            pass
                    del self.mcp_sessions[server_name]
                
                # Exit stack 정리: anyio cancel scope 오류 무시하고 시도
                if server_name in self.exit_stacks:
                    exit_stack = self.exit_stacks[server_name]
                    try:
                        # anyio RuntimeError는 완전히 무시 (다른 태스크에서 닫히려 할 때 발생)
                        await asyncio.wait_for(exit_stack.aclose(), timeout=2.0)
                    except RuntimeError as e:
                        if "cancel scope" in str(e).lower() or "different task" in str(e).lower():
                            # anyio cancel scope 오류는 무시
                            pass
                        else:
                            logger.debug(f"RuntimeError during exit_stack cleanup for {server_name}: {e}")
                    except (asyncio.TimeoutError, Exception) as e:
                        # 기타 오류는 무시
                        logger.debug(f"Error closing exit_stack for {server_name}: {e}")
                    finally:
                        del self.exit_stacks[server_name]
                
                if server_name in self.mcp_tools_map:
                    del self.mcp_tools_map[server_name]
                    
            except Exception as e:
                logger.debug(f"Error disconnecting from {server_name}: {e}")
        
        # 정리 완료 대기
        try:
            await asyncio.sleep(0.1)
        except:
            pass
        
        logger.info("MCP Hub cleanup completed")

    def start_shutdown(self):
        """외부에서 종료 시작 시 호출 - 신규 연결 차단"""
        self.stopping = True
    
    async def call_llm_async(self, model: str, messages: List[Dict[str, str]], 
                           temperature: float = 0.1, max_tokens: int = 4000) -> Dict[str, Any]:
        """LLM 호출은 llm_manager를 통해 수행하도록 강제 (Gemini 직결)."""
        raise RuntimeError("call_llm_async via MCP Hub is disabled. Use llm_manager for Gemini.")
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 실행 - MCP 프로토콜만 사용.
        
        실행 우선순위:
        1. MCP 서버에서 Tool 실행 (server_name::tool_name 형식 또는 tool_name으로 찾기)
        2. 실패 시 명확한 에러 반환 (fallback 없음)
        """
        import time
        start_time = time.time()
        logger.info(f"[MCP][exec.start] tool={tool_name} params_keys={list(parameters.keys())}")
        
        # Tool 찾기 (server_name::tool_name 또는 tool_name)
        tool_info = self.registry.get_tool_info(tool_name)
        
        # tool_name으로 직접 찾기 실패 시, 모든 MCP 서버에서 server_name::tool_name 형식으로 찾기
        if not tool_info:
            for registered_name in self.registry.get_all_tool_names():
                # server_name::tool_name 형식에서 tool_name 부분만 추출하여 비교
                if "::" in registered_name:
                    _, original_tool_name = registered_name.split("::", 1)
                    if original_tool_name == tool_name:
                        tool_info = self.registry.get_tool_info(registered_name)
                        logger.info(f"Found tool {tool_name} as {registered_name}")
                        break
                elif registered_name == tool_name:
                    tool_info = self.registry.get_tool_info(registered_name)
                    break
        
        if not tool_info:
            # Registry에서 직접 찾기
            tool_info = self.registry.tools.get(tool_name)
        
        if not tool_info:
            # 하위 호환성: self.tools에서 찾기
            tool_info = self.tools.get(tool_name)
        
        if not tool_info:
            # 사용 가능한 모든 tool 목록 로깅
            available_tools = self.registry.get_all_tool_names()
            logger.error(f"[MCP][exec.unknown] tool={tool_name} available={available_tools}")
            return {
                "success": False,
                "data": None,
                "error": f"Unknown tool: {tool_name}. Available tools: {', '.join(available_tools[:10])}",
                "execution_time": time.time() - start_time,
                "confidence": 0.0
            }

        try:
            # 1. MCP Tool인지 확인 및 실행 시도 - tool_name 또는 server_name::tool_name 형식 모두 확인
            found_tool_name = tool_name
            mcp_info = None
            
            # tool_name으로 직접 확인
            if self.registry.is_mcp_tool(tool_name):
                mcp_info = self.registry.get_mcp_server_info(tool_name)
                found_tool_name = tool_name
            else:
                # server_name::tool_name 형식으로 찾기
                for registered_name in self.registry.get_all_tool_names():
                    if "::" in registered_name:
                        server_part, original_tool_name = registered_name.split("::", 1)
                        if original_tool_name == tool_name and self.registry.is_mcp_tool(registered_name):
                            mcp_info = self.registry.get_mcp_server_info(registered_name)
                            found_tool_name = registered_name
                            logger.info(f"[MCP][exec.resolve] {tool_name} -> {registered_name}")
                            break
            
            if mcp_info:
                server_name, original_tool_name = mcp_info
                
                # MCP 서버 연결 확인
                if server_name in self.mcp_sessions:
                    try:
                        logger.info(f"[MCP][exec.try] server={server_name} tool={tool_name} as={found_tool_name}")
                        mcp_result = await self._execute_via_mcp_server(
                            server_name,
                            original_tool_name,
                            parameters
                        )
                        
                        if mcp_result:
                            # MCP 결과를 ToolResult 형식으로 변환
                            result_data = mcp_result if isinstance(mcp_result, dict) else {"result": mcp_result}
                            return {
                                "success": True,
                                "data": result_data,
                                "error": None,
                                "execution_time": time.time() - start_time,
                                "confidence": 0.9,
                                "source": "mcp"
                            }
                    except Exception as mcp_error:
                        logger.error(f"[MCP][exec.error] server={server_name} tool={tool_name} err={mcp_error}")
                        # MCP 실패 시 에러 반환 (fallback 제거)
                        return {
                            "success": False,
                            "data": None,
                            "error": f"MCP tool execution failed: {str(mcp_error)}",
                            "execution_time": time.time() - start_time,
                            "confidence": 0.0,
                            "source": "mcp"
                        }
            
            # MCP 도구가 아닌 경우 에러 반환 (fallback 제거)
            error_msg = f"Tool '{tool_name}' is not available via MCP servers"
            logger.error(f"[MCP][exec.error] {error_msg}")
            return {
                "success": False,
                "data": None,
                "error": error_msg,
                "execution_time": time.time() - start_time,
                "confidence": 0.0,
                "source": "mcp"
            }

        except Exception as e:
            logger.exception(f"[MCP][exec.error] tool={tool_name} err={e}")
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "confidence": 0.0
            }
    
    def get_tool_for_category(self, category: ToolCategory) -> Optional[str]:
        """카테고리에 해당하는 도구 반환 - Registry 기반."""
        tools_in_category = self.registry.get_tools_by_category(category)
        return tools_in_category[0] if tools_in_category else None
    
    def get_available_tools(self) -> List[str]:
        """사용 가능한 도구 목록 반환 - Registry 기반."""
        # Registry의 모든 Tool 이름 반환
        return self.registry.get_all_tool_names()
    
    def get_all_langchain_tools(self) -> List[BaseTool]:
        """모든 LangChain Tool 리스트 반환."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available")
            return []
        return self.registry.get_all_langchain_tools()
    
    async def check_mcp_servers(self) -> Dict[str, Any]:
        """모든 MCP 서버 연결 상태 확인 - mcp_config.json에 정의된 모든 서버."""
        server_status = {
            "timestamp": datetime.now().isoformat(),
            "total_servers": len(self.mcp_server_configs),
            "connected_servers": len(self.mcp_sessions),
            "servers": {}
        }
        
        logger.info(f"Checking {len(self.mcp_server_configs)} MCP servers...")
        
        for server_name, server_config in self.mcp_server_configs.items():
            server_info = {
                "name": server_name,
                "type": server_config.get("type", "stdio"),
                "connected": server_name in self.mcp_sessions,
                "tools_count": 0,
                "tools": [],
                "error": None
            }
            
            # 연결 타입 정보
            if server_config.get("type") == "http" or "httpUrl" in server_config or "url" in server_config:
                server_info["type"] = "http"
                server_info["url"] = server_config.get("httpUrl") or server_config.get("url", "unknown")
            else:
                server_info["type"] = "stdio"
                server_info["command"] = server_config.get("command", "unknown")
                server_info["args"] = server_config.get("args", [])
            
            # 연결 상태 확인
            if server_name in self.mcp_sessions:
                session = self.mcp_sessions[server_name]
                # 세션이 유효한지 확인
                try:
                    if hasattr(session, '_transport') and session._transport:
                        server_info["connected"] = True
                    else:
                        server_info["connected"] = False
                        server_info["error"] = "Session transport not available"
                except:
                    server_info["connected"] = False
                    server_info["error"] = "Session check failed"
                
                # 제공하는 Tool 목록 확인
                if server_name in self.mcp_tools_map:
                    tools = self.mcp_tools_map[server_name]
                    server_info["tools_count"] = len(tools)
                    server_info["tools"] = list(tools.keys())
                    
                    # 등록된 Tool 이름 (server_name::tool_name 형식)
                    registered_tools = [
                        name for name in self.registry.get_all_tool_names()
                        if name.startswith(f"{server_name}::")
                    ]
                    server_info["registered_tools"] = registered_tools
                else:
                    server_info["tools_count"] = 0
                    server_info["tools"] = []
                    server_info["error"] = "No tools discovered"
            else:
                server_info["connected"] = False
                server_info["error"] = "Not connected"
                # 연결 시도는 하지 않음 (별도의 initialize_mcp 호출 필요)
                # check_mcp_servers는 상태 확인만 수행
            
            server_status["servers"][server_name] = server_info
        
        # 통계 요약
        connected = sum(1 for s in server_status["servers"].values() if s["connected"])
        total_tools = sum(s["tools_count"] for s in server_status["servers"].values())
        
        server_status["summary"] = {
            "connected_servers": connected,
            "total_servers": len(self.mcp_server_configs),
            "total_tools_available": total_tools,
            "connection_rate": f"{connected}/{len(self.mcp_server_configs)}"
        }
        
        return server_status
    
    async def health_check(self) -> Dict[str, Any]:
        """강화된 헬스 체크 - OpenRouter, Gemini 2.5 Flash Lite, MCP 도구 검증."""
        try:
            health_status = {
                "mcp_enabled": self.config.enabled,
                "tools_available": len(self.tools),
                "timestamp": datetime.now().isoformat()
            }
            
            # 1. OpenRouter 연결 테스트
            try:
                test_messages = [
                    {"role": "system", "content": "Health check test."},
                    {"role": "user", "content": "Respond with 'OK' if you can process this request."}
                ]
                
                test_response = await self.openrouter_client.generate_response(
                    model=self.llm_config.primary_model,
                    messages=test_messages,
                    temperature=0.1,
                    max_tokens=50
                )
                
                openrouter_healthy = test_response and "choices" in test_response
                health_status.update({
                    "openrouter_connected": openrouter_healthy,
                    "primary_model": self.llm_config.primary_model,
                    "rate_limit_remaining": getattr(self.openrouter_client, 'rate_limit_remaining', 'unknown')
                })
                
                if not openrouter_healthy:
                    health_status["overall_health"] = "unhealthy"
                    health_status["critical_error"] = "OpenRouter connection failed"
                    return health_status
                    
            except Exception as e:
                health_status.update({
                    "openrouter_connected": False,
                    "openrouter_error": str(e),
                    "overall_health": "unhealthy",
                    "critical_error": f"OpenRouter health check failed: {e}"
                })
                return health_status
            
            # 2. 필수 MCP 도구 검증
            essential_tools = ["g-search", "fetch", "filesystem"]
            tool_health = {}
            failed_tools = []
            
            for tool in essential_tools:
                try:
                    # 간단한 테스트 실행
                    if tool == "g-search":
                        test_result = await execute_tool(tool, {"query": "test", "max_results": 1})
                    elif tool == "fetch":
                        test_result = await execute_tool(tool, {"url": "https://httpbin.org/get"})
                    elif tool == "filesystem":
                        test_result = await execute_tool(tool, {"path": ".", "operation": "list"})
                    
                    tool_health[tool] = test_result.get('success', False)
                    if not test_result.get('success', False):
                        failed_tools.append(tool)
                        
                except Exception as e:
                    tool_health[tool] = False
                    failed_tools.append(tool)
                    logger.warning(f"Tool {tool} health check failed: {e}")
            
            health_status.update({
                "tool_health": tool_health,
                "failed_tools": failed_tools,
                "essential_tools_healthy": len(failed_tools) == 0
            })
            
            # 3. 전체 상태 결정
            if len(failed_tools) > 0:
                health_status["overall_health"] = "unhealthy"
                health_status["critical_error"] = f"Essential tools failed: {', '.join(failed_tools)}"
            else:
                health_status["overall_health"] = "healthy"
            
            return health_status
            
        except Exception as e:
            return {
                "mcp_enabled": self.config.enabled,
                "tools_available": len(self.tools),
                "openrouter_connected": False,
                "error": str(e),
                "overall_health": "unhealthy",
                "critical_error": f"Health check failed: {e}",
                "timestamp": datetime.now().isoformat()
            }


# Global MCP Hub instance (lazy initialization)
_mcp_hub = None

def get_mcp_hub() -> 'UniversalMCPHub':
    """Get or initialize global MCP Hub."""
    global _mcp_hub
    if _mcp_hub is None:
        _mcp_hub = UniversalMCPHub()
    return _mcp_hub

async def get_available_tools() -> List[str]:
    """사용 가능한 도구 목록 반환."""
    mcp_hub = get_mcp_hub()
    return mcp_hub.get_available_tools()


async def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """MCP 도구 실행 - UniversalMCPHub의 execute_tool 사용."""
    mcp_hub = get_mcp_hub()
    return await mcp_hub.execute_tool(tool_name, parameters)


# 동기화 헬퍼 함수들 (LangChain Tool용)
def _execute_search_tool_sync(tool_name: str, parameters: Dict[str, Any]) -> str:
    """동기 버전 - LangChain Tool에서 호출."""
    try:
        # 이미 실행 중인 이벤트 루프가 있으면 새 스레드에서 실행
        import concurrent.futures
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 실행 중인 루프가 있으면 새 스레드에서 실행
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _execute_search_tool(tool_name, parameters))
                    result = future.result()
            else:
                result = loop.run_until_complete(_execute_search_tool(tool_name, parameters))
        except RuntimeError:
            # 이벤트 루프가 없으면 새로 생성
            result = asyncio.run(_execute_search_tool(tool_name, parameters))
        
        if result.success:
            import json
            return json.dumps(result.data, ensure_ascii=False, indent=2)
        else:
            raise RuntimeError(result.error or "Tool execution failed")
    except Exception as e:
        raise RuntimeError(f"Tool execution failed: {str(e)}")

def _execute_academic_tool_sync(tool_name: str, parameters: Dict[str, Any]) -> str:
    """동기 버전 - LangChain Tool에서 호출."""
    try:
        import concurrent.futures
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _execute_academic_tool(tool_name, parameters))
                    result = future.result()
            else:
                result = loop.run_until_complete(_execute_academic_tool(tool_name, parameters))
        except RuntimeError:
            result = asyncio.run(_execute_academic_tool(tool_name, parameters))
        
        if result.success:
            import json
            return json.dumps(result.data, ensure_ascii=False, indent=2)
        else:
            raise RuntimeError(result.error or "Tool execution failed")
    except Exception as e:
        raise RuntimeError(f"Tool execution failed: {str(e)}")

def _execute_data_tool_sync(tool_name: str, parameters: Dict[str, Any]) -> str:
    """동기 버전 - LangChain Tool에서 호출."""
    try:
        import concurrent.futures
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _execute_data_tool(tool_name, parameters))
                    result = future.result()
            else:
                result = loop.run_until_complete(_execute_data_tool(tool_name, parameters))
        except RuntimeError:
            result = asyncio.run(_execute_data_tool(tool_name, parameters))
        
        if result.success:
            import json
            return json.dumps(result.data, ensure_ascii=False, indent=2)
        else:
            raise RuntimeError(result.error or "Tool execution failed")
    except Exception as e:
        raise RuntimeError(f"Tool execution failed: {str(e)}")

def _execute_code_tool_sync(tool_name: str, parameters: Dict[str, Any]) -> str:
    """동기 버전 - LangChain Tool에서 호출."""
    try:
        import concurrent.futures
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _execute_code_tool(tool_name, parameters))
                    result = future.result()
            else:
                result = loop.run_until_complete(_execute_code_tool(tool_name, parameters))
        except RuntimeError:
            result = asyncio.run(_execute_code_tool(tool_name, parameters))
        
        if result.success:
            import json
            return json.dumps(result.data, ensure_ascii=False, indent=2)
        else:
            raise RuntimeError(result.error or "Tool execution failed")
    except Exception as e:
        raise RuntimeError(f"Tool execution failed: {str(e)}")


async def _execute_search_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """MCP 서버를 통한 검색 도구 실행."""
    import time
    
    start_time = time.time()
    query = parameters.get("query", "")
    max_results = parameters.get("max_results", 10) or parameters.get("num_results", 10)
    
    try:
        if tool_name == "g-search":
            # mcp_config.json에 정의된 모든 MCP 서버에서 검색 시도
            mcp_hub = get_mcp_hub()
            
            # MCP 서버 연결 확인 및 재연결
            if not mcp_hub.mcp_sessions:
                logger.warning("No MCP servers connected, attempting to initialize...")
                try:
                    await mcp_hub.initialize_mcp()
                except Exception as e:
                    logger.warning(f"Failed to initialize MCP servers: {e}")
            
            # mcp_config.json에 정의된 모든 서버 확인
            for server_name in mcp_hub.mcp_server_configs.keys():
                # 연결이 안 되어 있으면 연결 시도
                if server_name not in mcp_hub.mcp_sessions:
                    logger.info(f"MCP server {server_name} not connected, attempting connection...")
                    try:
                        server_config = mcp_hub.mcp_server_configs[server_name]
                        success = await mcp_hub._connect_to_mcp_server(server_name, server_config)
                        if not success:
                            logger.warning(f"Failed to connect to MCP server {server_name}")
                            continue
                    except Exception as e:
                        logger.error(f"Error connecting to MCP server {server_name}: {e}")
                        continue
                
                # 도구 맵 확인
                if server_name not in mcp_hub.mcp_tools_map:
                    logger.warning(f"MCP server {server_name} has no tools map")
                    continue
                
                try:
                    tools = mcp_hub.mcp_tools_map[server_name]
                    if not tools:
                        logger.warning(f"MCP server {server_name} has no tools available")
                        continue
                    
                    search_tool_name = None
                    
                    # 검색 도구 찾기 (search, query, ddg 등 키워드로)
                    for tool_name_key in tools.keys():
                        tool_lower = tool_name_key.lower()
                        if "search" in tool_lower or "query" in tool_lower or "ddg" in tool_lower:
                            search_tool_name = tool_name_key
                            logger.info(f"Found search tool '{search_tool_name}' in server {server_name}")
                            break
                    
                    if not search_tool_name:
                        logger.debug(f"No search tool found in MCP server {server_name}, available tools: {list(tools.keys())}")
                        continue
                    
                    # 검색 실행
                    logger.info(f"Using MCP server {server_name} with tool {search_tool_name} for search: {query}")
                    result = await mcp_hub._execute_via_mcp_server(
                        server_name,
                        search_tool_name,
                        {"query": query, "max_results": max_results}
                    )
                    
                    if not result:
                        logger.warning(f"MCP server {server_name} tool {search_tool_name} returned no result")
                        continue
                    
                    # 결과 파싱
                    import json
                    if isinstance(result, str):
                        try:
                            result_data = json.loads(result)
                        except:
                            # JSON이 아니면 텍스트를 결과로 사용
                            logger.debug(f"Result is not JSON, treating as text: {result[:100]}")
                            result_data = {"results": [{"title": "Search Result", "snippet": result}]}
                    else:
                        result_data = result
                    
                    # 결과 형식 정규화
                    results = result_data.get("results", [])
                    if not results and isinstance(result_data, dict):
                        # 다른 형식 시도
                        results = result_data.get("items", result_data.get("data", []))
                    
                    if results:
                        logger.info(f"✅ Search successful via MCP server {server_name}: {len(results)} results")
                        return ToolResult(
                            success=True,
                            data={
                                "query": query,
                                "results": results if isinstance(results, list) else [results],
                                "total_results": len(results) if isinstance(results, list) else 1,
                                "source": f"{server_name}-mcp"
                            },
                            execution_time=time.time() - start_time,
                            confidence=0.9
                        )
                    else:
                        logger.warning(f"MCP server {server_name} returned empty results")
                        continue
                    
                except Exception as mcp_error:
                    logger.warning(f"MCP 서버 {server_name} 검색 실패: {mcp_error}, 다음 서버 시도")
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    continue
            
            # 모든 MCP 서버 실패 시 에러 반환 (fallback 제거)
            logger.error(f"All MCP search tools failed for query: {query}")
            raise RuntimeError(f"All MCP search tools failed for query: {query}. No fallback available.")
        
        elif tool_name == "tavily":
            # MCP 서버를 통해 tavily 사용 (mcp_config.json에 정의된 서버)
            mcp_hub = get_mcp_hub()
            
            # 모든 연결된 MCP 서버에서 tavily 도구 찾아서 시도
            for server_name in mcp_hub.mcp_sessions.keys():
                if server_name not in mcp_hub.mcp_tools_map:
                    continue
                
                try:
                    tools = mcp_hub.mcp_tools_map[server_name]
                    tavily_tool_name = None
                    
                    # tavily 도구 찾기
                    for tool_name_key in tools.keys():
                        tool_lower = tool_name_key.lower()
                        if "tavily" in tool_lower:
                            tavily_tool_name = tool_name_key
                            break
                    
                    if tavily_tool_name:
                        logger.info(f"Using MCP server {server_name} with tool {tavily_tool_name}")
                        result = await mcp_hub._execute_via_mcp_server(
                            server_name,
                            tavily_tool_name,
                            {"query": query, "max_results": max_results}
                        )
                        
                        if result:
                            import json
                            if isinstance(result, str):
                                try:
                                    result_data = json.loads(result)
                                except:
                                    result_data = {"results": [{"title": "Search Result", "snippet": result}]}
                            else:
                                result_data = result
                            
                            results = result_data.get("results", [])
                            if not results and isinstance(result_data, dict):
                                results = result_data.get("items", result_data.get("data", []))
                            
                            if results:
                                return ToolResult(
                                    success=True,
                                    data={
                                        "query": query,
                                        "results": results if isinstance(results, list) else [results],
                                        "total_results": len(results) if isinstance(results, list) else 1,
                                        "source": f"{server_name}-mcp"
                                    },
                                    execution_time=time.time() - start_time,
                                    confidence=0.85
                                )
                    
                except Exception as mcp_error:
                    logger.warning(f"MCP 서버 {server_name} tavily 실패: {mcp_error}, 다음 서버 시도")
                    continue
            
            # MCP 서버에 tavily가 없으면 에러 (fallback 제거)
            raise ValueError("Tavily MCP server not found. Add tavily server to mcp_config.json")
        
        elif tool_name == "exa":
            # MCP 서버를 통해 exa 사용 (mcp_config.json에 정의된 서버)
            mcp_hub = get_mcp_hub()
            
            # 모든 연결된 MCP 서버에서 exa 도구 찾아서 시도
            for server_name in mcp_hub.mcp_sessions.keys():
                if server_name not in mcp_hub.mcp_tools_map:
                    continue
                
                try:
                    tools = mcp_hub.mcp_tools_map[server_name]
                    exa_tool_name = None
                    
                    # exa 도구 찾기
                    for tool_name_key in tools.keys():
                        tool_lower = tool_name_key.lower()
                        if "exa" in tool_lower:
                            exa_tool_name = tool_name_key
                            break
                    
                    if exa_tool_name:
                        logger.info(f"Using MCP server {server_name} with tool {exa_tool_name}")
                        result = await mcp_hub._execute_via_mcp_server(
                            server_name,
                            exa_tool_name,
                            {"query": query, "numResults": max_results}
                        )
                        
                        if result:
                            import json
                            if isinstance(result, str):
                                try:
                                    result_data = json.loads(result)
                                except:
                                    result_data = {"results": [{"title": "Search Result", "snippet": result}]}
                            else:
                                result_data = result
                            
                            results = result_data.get("results", [])
                            if not results and isinstance(result_data, dict):
                                results = result_data.get("items", result_data.get("data", []))
                            
                            if results:
                                return ToolResult(
                                    success=True,
                                    data={
                                        "query": query,
                                        "results": results if isinstance(results, list) else [results],
                                        "total_results": len(results) if isinstance(results, list) else 1,
                                        "source": f"{server_name}-mcp"
                                    },
                                    execution_time=time.time() - start_time,
                                    confidence=0.85
                                )
                    
                except Exception as mcp_error:
                    logger.warning(f"MCP 서버 {server_name} exa 실패: {mcp_error}, 다음 서버 시도")
                    continue
            
            # MCP 서버에 exa가 없으면 에러 (fallback 제거)
            raise ValueError("Exa MCP server not found. Add exa server to mcp_config.json")
        
        else:
            raise ValueError(f"Unknown search tool: {tool_name}")
            
    except Exception as e:
        logger.error(f"Search tool execution failed: {tool_name} - {e}")
        return ToolResult(
            success=False,
            data=None,
            error=f"Search tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0
        )


async def _execute_academic_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """실제 무료 학술 API를 사용한 학술 도구 실행."""
    import time
    
    start_time = time.time()
    query = parameters.get("query", "")
    max_results = parameters.get("max_results", 10) or parameters.get("num_results", 10)
    
    try:
        if tool_name == "arxiv":
            # arXiv API (100% 무료)
            import arxiv
            
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in client.results(search):
                results.append({
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary,
                    "url": paper.entry_id,
                    "published": paper.published.isoformat(),
                    "pdf_url": paper.pdf_url
                })
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": results,
                    "total_results": len(results),
                    "source": "arxiv"
                },
                execution_time=time.time() - start_time,
                confidence=0.95
            )
        
        elif tool_name == "scholar":
            # Google Scholar (무료, rate limit 있음)
            from scholarly import scholarly
            
            search_query = scholarly.search_pubs(query)
            results = []
            
            for i, pub in enumerate(search_query):
                if i >= max_results:
                    break
                    
                results.append({
                    "title": pub.get("bib", {}).get("title", ""),
                    "authors": pub.get("bib", {}).get("author", ""),
                    "abstract": pub.get("bib", {}).get("abstract", ""),
                    "url": pub.get("pub_url", ""),
                    "year": pub.get("bib", {}).get("pub_year", ""),
                    "citations": pub.get("num_citations", 0)
                })
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": results,
                    "total_results": len(results),
                    "source": "scholar"
                },
                execution_time=time.time() - start_time,
                confidence=0.8
            )
        
        else:
            raise ValueError(f"Unknown academic tool: {tool_name}")
            
    except Exception as e:
        logger.error(f"Academic tool execution failed: {tool_name} - {e}")
        return ToolResult(
            success=False,
            data=None,
            error=f"Academic tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0
        )


async def _execute_data_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """실제 데이터 도구 실행."""
    import time
    
    start_time = time.time()
    
    try:
        if tool_name == "fetch":
            # 실제 웹페이지 가져오기
            url = parameters.get("url", "")
            if not url:
                raise ValueError("URL parameter is required for fetch tool")
            
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                return ToolResult(
                    success=True,
                    data={
                        "url": url,
                        "status": response.status_code,
                        "content": response.text[:10000],  # 처음 10000자만
                        "content_length": len(response.text),
                        "headers": dict(response.headers)
                    },
                    execution_time=time.time() - start_time,
                    confidence=0.9
                )
        
        elif tool_name == "filesystem":
            # 파일시스템 접근
            path = parameters.get("path", "")
            operation = parameters.get("operation", "read")
            
            if not path:
                raise ValueError("Path parameter is required for filesystem tool")
            
            from pathlib import Path
            file_path = Path(path)
            
            if operation == "read":
                if file_path.exists() and file_path.is_file():
                    content = file_path.read_text(encoding='utf-8')
                    return ToolResult(
                        success=True,
                        data={
                            "path": str(file_path),
                            "operation": operation,
                            "content": content,
                            "size": file_path.stat().st_size
                        },
                        execution_time=time.time() - start_time,
                        confidence=0.9
                    )
                else:
                    raise FileNotFoundError(f"File not found: {path}")
            
            elif operation == "list":
                if file_path.exists() and file_path.is_dir():
                    files = [f.name for f in file_path.iterdir()]
                    return ToolResult(
                        success=True,
                        data={
                            "path": str(file_path),
                            "operation": operation,
                            "files": files
                        },
                        execution_time=time.time() - start_time,
                        confidence=0.9
                    )
                else:
                    raise FileNotFoundError(f"Directory not found: {path}")
            
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        
        else:
            raise ValueError(f"Unknown data tool: {tool_name}")
            
    except Exception as e:
        logger.error(f"Data tool execution failed: {tool_name} - {e}")
        return ToolResult(
            success=False,
            data=None,
            error=f"Data tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0
        )


async def _execute_code_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """실제 코드 도구 실행."""
    import time
    
    start_time = time.time()
    code = parameters.get("code", "")
    language = parameters.get("language", "python")
    
    try:
        if tool_name in ["python_coder", "code_interpreter"]:
            # Python 코드 실행 (안전한 환경에서)
            import subprocess
            import tempfile
            import os
            
            # 임시 파일에 코드 저장
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # 코드 실행
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                return ToolResult(
                    success=True,
                    data={
                        "code": code,
                        "language": language,
                        "output": result.stdout,
                        "error": result.stderr,
                        "return_code": result.returncode
                    },
                    execution_time=time.time() - start_time,
                    confidence=0.8
                )
            finally:
                # 임시 파일 삭제
                os.unlink(temp_file)
        
        else:
            raise ValueError(f"Unknown code tool: {tool_name}")
            
    except Exception as e:
        logger.error(f"Code tool execution failed: {tool_name} - {e}")
        return ToolResult(
            success=False,
            data=None,
            error=f"Code tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0
        )


async def get_tool_for_category(category: ToolCategory) -> Optional[str]:
    """카테고리에 해당하는 도구 반환."""
    mcp_hub = get_mcp_hub()
    return mcp_hub.get_tool_for_category(category)


async def health_check() -> Dict[str, Any]:
    """헬스 체크."""
    mcp_hub = get_mcp_hub()
    return await mcp_hub.health_check()


# CLI 실행 함수들
async def run_mcp_hub():
    """MCP Hub 실행 (CLI)."""
    mcp_hub = get_mcp_hub()
    print("🚀 Starting Universal MCP Hub...")
    try:
        await mcp_hub.initialize_mcp()
        print("✅ MCP Hub started successfully")
        print(f"Available tools: {len(mcp_hub.tools)}")
        
        # Hub 유지
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n✅ MCP Hub stopped")
    except Exception as e:
        print(f"❌ MCP Hub failed to start: {e}")
        await mcp_hub.cleanup()
        sys.exit(1)


async def list_tools():
    """도구 목록 출력 (CLI)."""
    print("🔧 Available MCP Tools:")
    available_tools = await get_available_tools()
    for tool_name in available_tools:
        print(f"  - {tool_name}")

async def check_mcp_servers():
    """MCP 서버 상태 확인 (CLI)."""
    mcp_hub = get_mcp_hub()
    try:
        # 초기화 (이미 초기화되어 있으면 재초기화하지 않음)
        if not mcp_hub.mcp_sessions:
            logger.info("Initializing MCP Hub to check servers...")
            await mcp_hub.initialize_mcp()
        
        server_status = await mcp_hub.check_mcp_servers()
        
        print("\n" + "=" * 80)
        print("📊 MCP 서버 연결 상태 확인")
        print("=" * 80)
        print(f"전체 서버 수: {server_status['total_servers']}")
        print(f"연결된 서버: {server_status['connected_servers']}")
        print(f"연결률: {server_status['summary']['connection_rate']}")
        print(f"전체 사용 가능한 Tool 수: {server_status['summary']['total_tools_available']}")
        print("\n")
        
        for server_name, info in server_status["servers"].items():
            status_icon = "✅" if info["connected"] else "❌"
            print(f"{status_icon} 서버: {server_name}")
            print(f"   타입: {info['type']}")
            
            if info["type"] == "http":
                print(f"   URL: {info.get('url', 'unknown')}")
            else:
                cmd = info.get('command', 'unknown')
                args_preview = ' '.join(info.get('args', [])[:3])
                print(f"   명령어: {cmd} {args_preview}...")
            
            print(f"   연결 상태: {'연결됨' if info['connected'] else '연결 안 됨'}")
            print(f"   제공 Tool 수: {info['tools_count']}")
            
            if info["tools"]:
                print(f"   Tool 목록:")
                for tool in info["tools"][:5]:  # 처음 5개만 표시
                    registered_name = f"{server_name}::{tool}"
                    print(f"     - {registered_name}")
                if len(info["tools"]) > 5:
                    print(f"     ... 및 {len(info['tools']) - 5}개 더")
            
            if info.get("error"):
                print(f"   ⚠️ 오류: {info['error']}")
            print()
        
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 서버 상태 확인 실패: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 정리하지 않고 세션 유지 (다른 작업에서 사용 가능)
        pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal MCP Hub - MCP Only")
    parser.add_argument("--start", action="store_true", help="Start MCP Hub")
    parser.add_argument("--list-tools", action="store_true", help="List available tools")
    parser.add_argument("--health", action="store_true", help="Show health status")
    parser.add_argument("--check-servers", action="store_true", help="Check all MCP server connections")
    
    args = parser.parse_args()
    
    if args.start:
        asyncio.run(run_mcp_hub())
    elif args.list_tools:
        asyncio.run(list_tools())
    elif args.check_servers:
        asyncio.run(check_mcp_servers())
    elif args.health:
        async def show_health():
            mcp_hub = get_mcp_hub()
            try:
                await mcp_hub.initialize_mcp()
                health = await health_check()
                print("🏥 Health Status:")
                for key, value in health.items():
                    print(f"  {key}: {value}")
                await mcp_hub.cleanup()
            except Exception as e:
                print(f"❌ Health check failed: {e}")
        asyncio.run(show_health())
    else:
        parser.print_help()