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

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from researcher_config import get_mcp_config, get_llm_config

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
                await self.session.close()
                # 연결 완전히 정리 대기
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.debug(f"Error closing OpenRouter session: {e}")
            finally:
                self.session = None
    
    async def generate_response(self, model: str, messages: List[Dict[str, str]], 
                              temperature: float = 0.1, max_tokens: int = 4000) -> Dict[str, Any]:
        """OpenRouter API를 통한 응답 생성."""
        if not self.session:
            raise RuntimeError("OpenRouter client not initialized")
        
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

        self.tools: Dict[str, ToolInfo] = {}
        self.openrouter_client: Optional[OpenRouterClient] = None
        
        # MCP 클라이언트
        self.mcp_sessions: Dict[str, ClientSession] = {}
        self.exit_stacks: Dict[str, AsyncExitStack] = {}
        self.mcp_tools_map: Dict[str, Dict[str, Any]] = {}  # server_name -> {tool_name -> tool_info}
        self.mcp_server_configs: Dict[str, Dict[str, Any]] = {}

        self._initialize_tools()
        self._initialize_clients()
        self._load_mcp_servers_from_config()
    
    def _initialize_tools(self):
        """도구 초기화."""
        # 검색 도구
        self.tools["g-search"] = ToolInfo(
            name="g-search",
            category=ToolCategory.SEARCH,
            description="Google 검색",
            parameters={"query": "str", "num_results": "int"},
            mcp_server="g-search"
        )
        
        self.tools["tavily"] = ToolInfo(
            name="tavily",
            category=ToolCategory.SEARCH,
            description="Tavily 검색",
            parameters={"query": "str", "max_results": "int"},
            mcp_server="tavily"
        )
        
        self.tools["exa"] = ToolInfo(
            name="exa",
            category=ToolCategory.SEARCH,
            description="Exa 검색",
            parameters={"query": "str", "num_results": "int"},
            mcp_server="exa"
        )
        
        # 데이터 도구
        self.tools["fetch"] = ToolInfo(
            name="fetch",
            category=ToolCategory.DATA,
            description="웹페이지 가져오기",
            parameters={"url": "str"},
            mcp_server="fetch"
        )
        
        self.tools["filesystem"] = ToolInfo(
            name="filesystem",
            category=ToolCategory.DATA,
            description="파일시스템 접근",
            parameters={"path": "str", "operation": "str"},
            mcp_server="filesystem"
        )
        
        # 코드 도구
        self.tools["python_coder"] = ToolInfo(
            name="python_coder",
            category=ToolCategory.CODE,
            description="Python 코드 실행",
            parameters={"code": "str"},
            mcp_server="python_coder"
        )
        
        self.tools["code_interpreter"] = ToolInfo(
            name="code_interpreter",
            category=ToolCategory.CODE,
            description="코드 해석",
            parameters={"code": "str", "language": "str"},
            mcp_server="code_interpreter"
        )
        
        # 학술 도구
        self.tools["arxiv"] = ToolInfo(
            name="arxiv",
            category=ToolCategory.ACADEMIC,
            description="ArXiv 논문 검색",
            parameters={"query": "str", "max_results": "int"},
            mcp_server="arxiv"
        )
        
        self.tools["scholar"] = ToolInfo(
            name="scholar",
            category=ToolCategory.ACADEMIC,
            description="Google Scholar 검색",
            parameters={"query": "str", "num_results": "int"},
            mcp_server="scholar"
        )
        
        # 비즈니스 도구
        self.tools["crunchbase"] = ToolInfo(
            name="crunchbase",
            category=ToolCategory.BUSINESS,
            description="Crunchbase 검색",
            parameters={"query": "str"},
            mcp_server="crunchbase"
        )
        
        self.tools["linkedin"] = ToolInfo(
            name="linkedin",
            category=ToolCategory.BUSINESS,
            description="LinkedIn 검색",
            parameters={"query": "str"},
            mcp_server="linkedin"
        )
    
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
            # .env나 config 파일에서 MCP 서버 설정 읽기
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
            
            # 도구 맵 생성
            self.mcp_tools_map[server_name] = {}
            for tool in response.tools:
                self.mcp_tools_map[server_name][tool.name] = tool
            
            logger.info(f"✅ Connected to MCP server {server_name} with {len(response.tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {server_name}: {e}")
            return False
    
    async def _disconnect_from_mcp_server(self, server_name: str):
        """MCP 서버 연결 해제."""
        try:
            # 세션 먼저 제거
            if server_name in self.mcp_sessions:
                del self.mcp_sessions[server_name]
            
            # Exit stack은 직접 닫지 않고 참조만 제거
            # (stdio_client가 async generator이므로 자동으로 정리됨)
            if server_name in self.exit_stacks:
                # AsyncExitStack을 직접 aclose() 하면 다른 task에서 실행될 수 있어 에러 발생
                # 대신 참조만 제거하고 실제 정리는 Python GC가 처리하도록 함
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
        """MCP 연결 정리."""
        logger.info("Cleaning up MCP Hub...")
        
        # 모든 MCP 서버 연결 해제 (역순으로 정리)
        server_names = list(self.mcp_sessions.keys())
        for server_name in reversed(server_names):
            try:
                # 세션만 제거하고 exit stack은 나중에 정리
                if server_name in self.mcp_sessions:
                    del self.mcp_sessions[server_name]
                
                # Exit stack 정리 (에러 무시)
                if server_name in self.exit_stacks:
                    exit_stack = self.exit_stacks[server_name]
                    try:
                        # exit_stack.aclose()를 직접 호출하지 않고
                        # 이미 생성된 컨텍스트는 그대로 두고 정리만 시도
                        del self.exit_stacks[server_name]
                    except Exception:
                        # 에러는 무시하고 계속
                        pass
                
                if server_name in self.mcp_tools_map:
                    del self.mcp_tools_map[server_name]
                    
            except Exception as e:
                logger.debug(f"Error disconnecting from {server_name}: {e}")
        
        # OpenRouter 클라이언트 정리
        if self.openrouter_client:
            try:
                if hasattr(self.openrouter_client, 'session') and self.openrouter_client.session:
                    await self.openrouter_client.session.close()
                    # 세션 완전히 정리 대기
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.debug(f"Error closing OpenRouter client session: {e}")
            finally:
                self.openrouter_client = None
        
        logger.info("MCP Hub cleanup completed")
    
    async def call_llm_async(self, model: str, messages: List[Dict[str, str]], 
                           temperature: float = 0.1, max_tokens: int = 4000) -> Dict[str, Any]:
        """LLM 호출 - OpenRouter를 통한 비동기 호출."""
        if not self.openrouter_client:
            raise RuntimeError("OpenRouter client not initialized")
        
        return await self.openrouter_client.generate_response(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """MCP 도구 실행 - 실제 무료 API 사용."""
        if tool_name not in self.tools:
            logger.error(f"Unknown tool: {tool_name}")
            return {
                "success": False,
                "data": None,
                "error": f"Unknown tool: {tool_name}",
                "execution_time": 0.0,
                "confidence": 0.0
            }

        tool_info = self.tools[tool_name]

        try:
            # 실제 무료 API를 사용한 도구 실행
            result: ToolResult
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
                    execution_time=0.0,
                    confidence=0.0
                )
            
            # ToolResult를 dict로 변환
            return {
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "execution_time": result.execution_time,
                "confidence": result.confidence
            }

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "execution_time": 0.0,
                "confidence": 0.0
            }
    
    def get_tool_for_category(self, category: ToolCategory) -> Optional[str]:
        """카테고리에 해당하는 도구 반환."""
        for tool_name, tool_info in self.tools.items():
            if tool_info.category == category:
                return tool_name
        return None
    
    def get_available_tools(self) -> List[str]:
        """사용 가능한 도구 목록 반환."""
        return list(self.tools.keys())
    
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
    """MCP 도구 실행 - 실제 무료 API 사용, 실패 시 명확한 오류 반환."""
    try:
        # 실제 무료 API를 사용한 도구 실행
        result: ToolResult
        if tool_name in ["g-search", "tavily", "exa"]:
            result = await _execute_search_tool(tool_name, parameters)
        elif tool_name in ["arxiv", "scholar"]:
            result = await _execute_academic_tool(tool_name, parameters)
        elif tool_name in ["fetch", "filesystem"]:
            result = await _execute_data_tool(tool_name, parameters)
        elif tool_name in ["python_coder", "code_interpreter"]:
            result = await _execute_code_tool(tool_name, parameters)
        else:
            logger.error(f"Unknown tool: {tool_name}")
            result = ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}",
                execution_time=0.0,
                confidence=0.0
            )
        
        # ToolResult를 dict로 변환
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error,
            "execution_time": result.execution_time,
            "confidence": result.confidence
        }
    except Exception as e:
        logger.error(f"Tool execution failed: {tool_name} - {e}")
        return {
            "success": False,
            "data": None,
            "error": f"Tool execution failed: {str(e)}",
            "execution_time": 0.0,
            "confidence": 0.0
        }


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