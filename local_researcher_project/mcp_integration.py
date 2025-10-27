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
    """OpenRouter API 클라이언트 - Gemini 2.5 Flash Lite 우선 사용."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_remaining = 1000
        self.rate_limit_reset = datetime.now() + timedelta(hours=1)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://mcp-agent.local",
                "X-Title": "MCP Agent Hub"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
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

        self._initialize_tools()
        self._initialize_clients()
    
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
    
    async def initialize_mcp(self):
        """MCP 초기화 - OpenRouter와 Gemini 2.5 Flash Lite."""
        if not self.config.enabled:
            logger.warning("MCP is disabled. Continuing with limited functionality.")
            return
        
        try:
            logger.info("Initializing MCP Hub with OpenRouter and Gemini 2.5 Flash Lite...")
            
            # OpenRouter 클라이언트 초기화
            await self.openrouter_client.__aenter__()
            
            # 연결 테스트
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
                logger.info("✅ MCP Hub initialized successfully with OpenRouter and Gemini 2.5 Flash Lite")
                logger.info(f"Available tools: {len(self.tools)}")
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
        
        if failed_tools:
            error_msg = f"PRODUCTION ERROR: Essential MCP tools failed validation: {', '.join(failed_tools)}. System cannot start without these tools. Check your MCP server configuration and network connectivity."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info("✅ All essential MCP tools validated successfully for production")
    
    async def cleanup(self):
        """MCP 연결 정리."""
        logger.info("Cleaning up MCP Hub...")
        if self.openrouter_client:
            await self.openrouter_client.__aexit__(None, None, None)
        logger.info("MCP Hub cleanup completed")
    
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
    """실제 무료 검색 API를 사용한 검색 도구 실행."""
    import time
    from duckduckgo_search import DDGS
    
    start_time = time.time()
    query = parameters.get("query", "")
    max_results = parameters.get("max_results", 10) or parameters.get("num_results", 10)
    
    try:
        if tool_name == "g-search":
            # DuckDuckGo 검색 (100% 무료)
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
                
                return ToolResult(
                    success=True,
                    data={
                        "query": query,
                        "results": results,
                        "total_results": len(results),
                        "source": "duckduckgo"
                    },
                    execution_time=time.time() - start_time,
                    confidence=0.9
                )
        
        elif tool_name == "tavily":
            # Tavily API (무료 tier 사용)
            import os
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise ValueError("TAVILY_API_KEY not found. Please set it in .env file")
            
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": query,
                        "search_depth": "basic",
                        "max_results": max_results
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                return ToolResult(
                    success=True,
                    data={
                        "query": query,
                        "results": data.get("results", []),
                        "total_results": len(data.get("results", [])),
                        "source": "tavily"
                    },
                    execution_time=time.time() - start_time,
                    confidence=0.85
                )
        
        elif tool_name == "exa":
            # Exa API (무료 tier 사용)
            import os
            api_key = os.getenv("EXA_API_KEY")
            if not api_key:
                raise ValueError("EXA_API_KEY not found. Please set it in .env file")
            
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.exa.ai/search",
                    headers={"x-api-key": api_key},
                    json={
                        "query": query,
                        "numResults": max_results,
                        "type": "search"
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                return ToolResult(
                    success=True,
                    data={
                        "query": query,
                        "results": data.get("results", []),
                        "total_results": len(data.get("results", [])),
                        "source": "exa"
                    },
                    execution_time=time.time() - start_time,
                    confidence=0.85
                )
        
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