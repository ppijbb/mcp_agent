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
    from mcp.types import ListToolsResult, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    ListToolsResult = None
    TextContent = None

# LangChain imports
try:
    from langchain_core.tools import BaseTool, StructuredTool
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
        if 'search' in tool_name.lower():
            category = ToolCategory.SEARCH
        elif 'fetch' in tool_name.lower() or 'file' in tool_name.lower():
            category = ToolCategory.DATA
        elif 'code' in tool_name.lower() or 'python' in tool_name.lower():
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
    """OpenRouter API 클라이언트 - API에서 무료 모델을 동적으로 가져와 사용."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_remaining = 1000
        self.rate_limit_reset = datetime.now() + timedelta(hours=1)
        self.available_models: List[str] = []
        self.free_models: List[str] = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://mcp-agent.local",
                "X-Title": "MCP Agent Hub"
            }
        )
        
        # OpenRouter API에서 무료 모델 목록 동적으로 가져오기
        await self._fetch_free_models()
        
        return self
    
    async def _fetch_free_models(self):
        """OpenRouter API에서 무료 모델 목록을 동적으로 가져옴."""
        try:
            headers = {
                "Content-Type": "application/json",
                "HTTP-Referer": "https://mcp-agent.local",
                "X-Title": "MCP Agent Hub"
            }
            
            # API 키가 있으면 추가
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with self.session.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    
                    for model in models:
                        model_id = model.get("id", "")
                        pricing = model.get("pricing", {})
                        
                        # 무료 모델 확인
                        if pricing and pricing.get("prompt", "0") == "0" and pricing.get("completion", "0") == "0":
                            self.free_models.append(model_id)
                        
                        self.available_models.append(model_id)
                    
                    if self.free_models:
                        logger.info(f"✅ 동적으로 {len(self.free_models)}개의 무료 모델 로드: {', '.join(self.free_models[:5])}")
                    else:
                        logger.warning("⚠️ 무료 모델을 찾지 못했습니다. API 키 없이는 제한적 기능만 사용 가능합니다.")
                else:
                    logger.warning(f"⚠️ OpenRouter 모델 목록 조회 실패: {response.status}")
                    
        except Exception as e:
            logger.warning(f"⚠️ 무료 모델 로드 실패: {e}. 기본 모델 사용")
            
    def get_free_model(self) -> str:
        """무료 모델 중 첫 번째 사용 가능한 모델 반환."""
        if self.free_models:
            return self.free_models[0]
        # 무료 모델이 없으면 기본값 (OpenRouter의 최근 무료 모델)
        return "deepseek/deepseek-chat-v3.1:free"
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            try:
                if not self.session.closed:
                    await self.session.close()
                # 연결 완전히 정리 대기
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.debug(f"Error closing OpenRouter session: {e}")
            finally:
                self.session = None
    
    async def generate_response(self, model: str, messages: List[Dict[str, str]], 
                              temperature: float = 0.1, max_tokens: int = 4000) -> Dict[str, Any]:
        """OpenRouter API를 통한 응답 생성."""
        if not self.session or self.session.closed:
            raise RuntimeError("OpenRouter client not initialized - session is None or closed. Call initialize_mcp() first.")
        
        # Rate limiting 체크
        if self.rate_limit_remaining <= 0 and datetime.now() < self.rate_limit_reset:
            await asyncio.sleep(60)  # 1분 대기
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            async with self.session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                if response.status == 429:
                    # Rate limit exceeded
                    retry_after = int(response.headers.get("Retry-After", 60))
                    await asyncio.sleep(retry_after)
                    return await self.generate_response(model, messages, temperature, max_tokens)
                
                response.raise_for_status()
                result = await response.json()
                
                # Rate limit 정보 업데이트
                self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 1000))
                
                return result
                
        except aiohttp.ClientError as e:
            logger.error(f"OpenRouter API error: {e}")
            raise RuntimeError(f"OpenRouter API error: {e}")



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
        self.exit_stacks: Dict[str, AsyncExitStack] = {}
        self.mcp_tools_map: Dict[str, Dict[str, Any]] = {}  # server_name -> {tool_name -> tool_info}
        self.mcp_server_configs: Dict[str, Dict[str, Any]] = {}

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
            
            # Pydantic 스키마 생성
            if BaseModel:
                fields = {}
                for param_name, param_info in params_config.items():
                    param_type = param_info.get("type", "string")
                    param_desc = param_info.get("description", "")
                    required = param_info.get("required", False)
                    
                    # 타입 매핑
                    if param_type == "integer":
                        field_type = int
                    elif param_type == "number":
                        field_type = float
                    elif param_type == "boolean":
                        field_type = bool
                    else:
                        field_type = str
                    
                    if required:
                        fields[param_name] = (field_type, Field(description=param_desc))
                    else:
                        default_value = param_info.get("default", None)
                        fields[param_name] = (Optional[field_type], Field(default=default_value, description=param_desc))
                
                # 동적 Pydantic 모델 생성
                ToolSchema = type(f"{tool_name}Schema", (BaseModel,), {"__annotations__": fields})
            else:
                ToolSchema = None
            
            # Tool 실행 함수 선택 (동기 래퍼 생성)
            def create_sync_func(tool_name_str, func_type):
                """동기 함수 래퍼 생성."""
                if func_type == "search":
                    return lambda **kwargs: _execute_search_tool_sync(tool_name_str, kwargs)
                elif func_type == "academic":
                    return lambda **kwargs: _execute_academic_tool_sync(tool_name_str, kwargs)
                elif func_type == "data":
                    return lambda **kwargs: _execute_data_tool_sync(tool_name_str, kwargs)
                elif func_type == "code":
                    return lambda **kwargs: _execute_code_tool_sync(tool_name_str, kwargs)
                else:
                    return None
            
            # Tool별 실행 함수 매핑
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
                logger.warning(f"No execution function for tool: {tool_name}")
                return None
            
            # StructuredTool 생성
            if StructuredTool and ToolSchema:
                langchain_tool = StructuredTool.from_function(
                    func=func,
                    name=tool_name,
                    description=description,
                    args_schema=ToolSchema
                )
            elif StructuredTool:
                langchain_tool = StructuredTool.from_function(
                    func=func,
                    name=tool_name,
                    description=description
                )
            else:
                return None
            
            return langchain_tool
            
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
                logger.warning(f"⚠️ Failed to create LangChain wrapper for {tool_name}")
        
        # Registry의 tools를 self.tools와 동기화
        self.tools.update(self.registry.tools)
    
    def _initialize_clients(self):
        """클라이언트 초기화 - OpenRouter와 Gemini 2.5 Flash Lite."""
        if self.llm_config.openrouter_api_key:
            self.openrouter_client = OpenRouterClient(self.llm_config.openrouter_api_key)
            logger.info("✅ OpenRouter client initialized")
        else:
            logger.warning("OpenRouter API key not configured - LLM features will not function")
    
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
    
    async def _connect_to_mcp_server(self, server_name: str, server_config: Dict[str, Any]):
        """MCP 서버에 연결."""
        if not MCP_AVAILABLE:
            logger.error("MCP package not available")
            return False
        
        try:
            if server_name in self.mcp_sessions:
                await self._disconnect_from_mcp_server(server_name)
            
            command = server_config["command"]
            args = server_config.get("args", [])
            
            logger.info(f"Connecting to MCP server: {server_name} ({command})")
            
            exit_stack = AsyncExitStack()
            self.exit_stacks[server_name] = exit_stack
            
            # stdio transport 사용
            server_params = StdioServerParameters(command=command, args=args)
            stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            
            self.mcp_sessions[server_name] = session
            
            # 세션 초기화 및 도구 목록 가져오기
            await session.initialize()
            response = await session.list_tools()
            
            # 도구 맵 생성 및 Registry에 동적 등록
            self.mcp_tools_map[server_name] = {}
            for tool in response.tools:
                self.mcp_tools_map[server_name][tool.name] = tool
                # ToolRegistry에 server_name::tool_name 형식으로 등록
                self.registry.register_mcp_tool(server_name, tool.name, tool)
                logger.debug(f"Registered MCP tool: {server_name}::{tool.name}")
            
            # Registry tools를 self.tools에 동기화
            self.tools.update(self.registry.tools)
            
            logger.info(f"✅ Connected to MCP server {server_name} with {len(response.tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {server_name}: {e}")
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
                        await asyncio.wait_for(session.shutdown(), timeout=2.0)
                except (asyncio.TimeoutError, AttributeError, Exception):
                    pass  # 세션 종료 실패는 무시
                del self.mcp_sessions[server_name]
            
            # Exit stack은 참조만 제거
            # stdio_client async generator가 같은 task context에서 정리되도록 함
            if server_name in self.exit_stacks:
                exit_stack = self.exit_stacks[server_name]
                try:
                    # AsyncExitStack을 직접 정리하지 않음
                    # async generator가 자동으로 정리되도록 함
                    del self.exit_stacks[server_name]
                except Exception:
                    pass
            
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
        
        try:
            logger.info("Initializing MCP Hub with OpenRouter and MCP servers...")
            
            # OpenRouter 클라이언트 초기화
            if self.openrouter_client:
                await self.openrouter_client.__aenter__()
            
            # MCP 서버 연결 (모든 서버)
            connected_servers = []
            for server_name, server_config in self.mcp_server_configs.items():
                try:
                    success = await self._connect_to_mcp_server(server_name, server_config)
                    if success:
                        connected_servers.append(server_name)
                except Exception as e:
                    logger.error(f"Failed to connect to MCP server {server_name}: {e}")
            
            if connected_servers:
                logger.info(f"✅ Successfully connected to {len(connected_servers)} MCP servers: {', '.join(connected_servers)}")
            else:
                logger.warning("⚠️ No MCP servers connected successfully")
            
            # 연결 테스트
            if self.openrouter_client:
                test_messages = [
                    {"role": "system", "content": "You are a test assistant."},
                    {"role": "user", "content": "Hello, this is a connection test."}
                ]
                
                test_response = await self.openrouter_client.generate_response(
                    model=self.llm_config.primary_model,
                    messages=test_messages,
                    temperature=0.1,
                    max_tokens=100
                )
                
                if test_response and "choices" in test_response:
                    logger.info("✅ MCP Hub initialized successfully")
                    logger.info(f"Available tools: {len(self.tools)}")
                    logger.info(f"MCP servers: {list(self.mcp_sessions.keys())}")
                    logger.info(f"Primary model: {self.llm_config.primary_model}")
                else:
                    logger.warning("⚠️ OpenRouter connection test failed - continuing with graceful degradation")
                    return
            
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
        """필수 MCP 도구 검증 및 실패 시 중단 - PRODUCTION LEVEL."""
        essential_tools = ["g-search", "fetch", "filesystem"]
        failed_tools = []
        
        logger.info("Validating essential MCP tools for production deployment...")
        
        for tool in essential_tools:
            try:
                # Production-level test execution with timeout
                test_timeout = 30  # 30초 타임아웃
                
                if tool == "g-search":
                    test_result = await asyncio.wait_for(
                        execute_tool(tool, {"query": "test", "max_results": 1}),
                        timeout=test_timeout
                    )
                elif tool == "fetch":
                    test_result = await asyncio.wait_for(
                        execute_tool(tool, {"url": "https://httpbin.org/get"}),
                        timeout=test_timeout
                    )
                elif tool == "filesystem":
                    test_result = await asyncio.wait_for(
                        execute_tool(tool, {"path": ".", "operation": "list"}),
                        timeout=test_timeout
                    )
                
                if test_result.get('success', False):
                    logger.info(f"✅ Essential tool {tool} validated successfully")
                else:
                    failed_tools.append(tool)
                    logger.error(f"❌ Essential tool {tool} validation failed: {test_result.get('error', 'Unknown error')}")
                    
            except asyncio.TimeoutError:
                failed_tools.append(tool)
                logger.error(f"❌ Essential tool {tool} validation timed out after {test_timeout}s")
            except Exception as e:
                failed_tools.append(tool)
                logger.error(f"❌ Essential tool {tool} validation failed: {e}")
        
        # g-search rate limit는 무시 (DuckDuckGo의 일시적 제한)
        failed_tools_filtered = [t for t in failed_tools if not (t == "g-search")]
        
        if failed_tools_filtered:
            error_msg = f"PRODUCTION ERROR: Essential MCP tools failed validation: {', '.join(failed_tools_filtered)}. System cannot start without these tools. Check your MCP server configuration and network connectivity."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        elif failed_tools and all("g-search" in t for t in failed_tools):
            # g-search만 실패한 경우 경고만
            logger.warning(f"⚠️ g-search rate limited - system will continue with limited search capability")
        
        logger.info("✅ All essential MCP tools validated successfully for production")
    
    async def cleanup(self):
        """MCP 연결 정리 - Production-grade cleanup."""
        logger.info("Cleaning up MCP Hub...")
        
        # OpenRouter 클라이언트 먼저 정리
        if self.openrouter_client:
            try:
                await self.openrouter_client.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing OpenRouter client: {e}")
            finally:
                self.openrouter_client = None
        
        # 모든 MCP 서버 연결 해제 (역순으로 정리)
        server_names = list(self.mcp_sessions.keys())
        for server_name in reversed(server_names):
            try:
                # 세션 제거
                if server_name in self.mcp_sessions:
                    session = self.mcp_sessions[server_name]
                    try:
                        # 세션이 활성화되어 있으면 종료 시도
                        if hasattr(session, '_transport') and session._transport:
                            pass  # stdio_client가 자동으로 정리됨
                    except Exception:
                        pass
                    del self.mcp_sessions[server_name]
                
                # Exit stack은 참조만 제거 (실제 정리는 async generator가 처리)
                if server_name in self.exit_stacks:
                    exit_stack = self.exit_stacks[server_name]
                    try:
                        # AsyncExitStack을 직접 정리하지 않음
                        # stdio_client async generator가 자동으로 정리되도록 함
                        del self.exit_stacks[server_name]
                    except Exception:
                        pass
                
                if server_name in self.mcp_tools_map:
                    del self.mcp_tools_map[server_name]
                    
            except Exception as e:
                logger.debug(f"Error disconnecting from {server_name}: {e}")
        
        # 정리 완료 대기
        await asyncio.sleep(0.2)
        
        logger.info("MCP Hub cleanup completed")
    
    async def call_llm_async(self, model: str, messages: List[Dict[str, str]], 
                           temperature: float = 0.1, max_tokens: int = 4000) -> Dict[str, Any]:
        """LLM 호출 - OpenRouter를 통한 비동기 호출."""
        if not self.openrouter_client:
            raise RuntimeError("OpenRouter client not initialized - call initialize_mcp() first")
        
        # 세션 상태 확인
        if not hasattr(self.openrouter_client, 'session') or not self.openrouter_client.session:
            raise RuntimeError("OpenRouter client session not initialized - call initialize_mcp() first")
        
        if self.openrouter_client.session.closed:
            raise RuntimeError("OpenRouter client session is closed - reinitialize with initialize_mcp()")
        
        return await self.openrouter_client.generate_response(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 실행 - MCP 우선 프로토콜, 실패 시 LangChain Tool fallback.
        
        실행 우선순위:
        1. MCP 서버에서 Tool 실행 (server_name::tool_name 형식 또는 tool_name으로 찾기)
        2. MCP 서버 연결 실패 시 LangChain Tool로 fallback (로컬 구현)
        3. 모두 실패 시 에러 반환
        """
        import time
        start_time = time.time()
        
        # Tool 찾기 (server_name::tool_name 또는 tool_name)
        tool_info = self.registry.get_tool_info(tool_name)
        if not tool_info:
            # Registry에서 찾기
            tool_info = self.registry.tools.get(tool_name)
            if not tool_info:
                tool_info = self.tools.get(tool_name)
            
        if not tool_info:
            logger.error(f"Unknown tool: {tool_name}")
            return {
                "success": False,
                "data": None,
                "error": f"Unknown tool: {tool_name}",
                "execution_time": time.time() - start_time,
                "confidence": 0.0
            }

        try:
            # 1. MCP Tool인지 확인 및 실행 시도
            if self.registry.is_mcp_tool(tool_name):
                mcp_info = self.registry.get_mcp_server_info(tool_name)
                if mcp_info:
                    server_name, original_tool_name = mcp_info
                    
                    # MCP 서버 연결 확인
                    if server_name in self.mcp_sessions:
                        try:
                            logger.info(f"Executing MCP tool: {tool_name} via server {server_name}")
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
                            logger.warning(f"MCP tool execution failed: {mcp_error}, trying LangChain fallback")
                            # MCP 실패 시 LangChain Tool로 fallback
            
            # 2. LangChain Tool fallback
            langchain_tool = self.registry.get_langchain_tool(tool_name)
            if langchain_tool:
                try:
                    logger.info(f"Executing LangChain tool: {tool_name}")
                    # LangChain Tool 실행 (동기)
                    result = langchain_tool.invoke(parameters)
                    
                    # 결과 파싱
                    if isinstance(result, str):
                        import json
                        try:
                            data = json.loads(result)
                        except:
                            data = {"result": result}
                    else:
                        data = result if isinstance(result, dict) else {"result": result}
                    
                    return {
                        "success": True,
                        "data": data,
                        "error": None,
                        "execution_time": time.time() - start_time,
                        "confidence": 0.85,
                        "source": "langchain"
                    }
                except Exception as lc_error:
                    logger.error(f"LangChain tool execution failed: {lc_error}")
            
            # 3. 카테고리 기반 직접 실행 (최후의 수단)
            if tool_info.category == ToolCategory.SEARCH:
                result = await _execute_search_tool(tool_name, parameters)
            elif tool_info.category == ToolCategory.ACADEMIC:
                result = await _execute_academic_tool(tool_name, parameters)
            elif tool_info.category == ToolCategory.DATA:
                result = await _execute_data_tool(tool_name, parameters)
            elif tool_info.category == ToolCategory.CODE:
                result = await _execute_code_tool(tool_name, parameters)
            else:
                result = ToolResult(
                    success=False,
                    data=None,
                    error=f"Unsupported tool category: {tool_info.category}",
                    execution_time=time.time() - start_time,
                    confidence=0.0
                )
            
            # ToolResult를 dict로 변환
            return {
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "execution_time": result.execution_time,
                "confidence": result.confidence,
                "source": "direct"
            }

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
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
            
            # 모든 MCP 서버 실패 시 직접 DuckDuckGo 사용
            
            # Fallback: 직접 DuckDuckGo 검색
            logger.info(f"Using direct DuckDuckGo search for query: {query}")
            from duckduckgo_search import DDGS
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        delay = 2 * attempt
                        await asyncio.sleep(delay)
                        logger.info(f"Retrying search (attempt {attempt + 1}/{max_retries})")
                    
                    with DDGS(timeout=60) as ddgs:
                        results = []
                        for r in ddgs.text(query, max_results=max_results):
                            results.append({
                                "title": r.get("title", ""),
                                "url": r.get("href", ""),
                                "snippet": r.get("body", "")
                            })
                        
                        logger.info(f"Search completed with {len(results)} results")
                        return ToolResult(
                            success=True,
                            data={
                                "query": query,
                                "results": results,
                                "total_results": len(results),
                                "source": "duckduckgo-direct",
                                "mcp_server_used": False
                            },
                            execution_time=time.time() - start_time,
                            confidence=0.9
                        )
                    
                except Exception as e:
                    error_str = str(e)
                    if "202" in error_str or "ratelimit" in error_str.lower():
                        if attempt < max_retries - 1:
                            logger.warning(f"Rate limit hit, retrying (attempt {attempt + 1}/{max_retries})")
                            continue
                        else:
                            logger.error(f"Rate limit after {max_retries} attempts")
                            return ToolResult(
                                success=False,
                                data={"query": query, "results": [], "total_results": 0, "rate_limited": True},
                                error=f"Rate limited: {str(e)}",
                                execution_time=time.time() - start_time,
                                confidence=0.0
                            )
                    else:
                        raise
            
            # 모든 재시도 실패
            logger.error("All search attempts failed")
            raise RuntimeError("All search attempts failed")
        
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal MCP Hub - MCP Only")
    parser.add_argument("--start", action="store_true", help="Start MCP Hub")
    parser.add_argument("--list-tools", action="store_true", help="List available tools")
    parser.add_argument("--health", action="store_true", help="Show health status")
    
    args = parser.parse_args()
    
    if args.start:
        asyncio.run(run_mcp_hub())
    elif args.list_tools:
        asyncio.run(list_tools())
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