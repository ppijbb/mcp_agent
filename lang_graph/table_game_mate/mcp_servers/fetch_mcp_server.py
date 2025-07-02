#!/usr/bin/env python3
"""
Fetch MCP Server - HTTP 요청 처리

웹 API 호출, 데이터 다운로드, JSON 파싱 등
HTTP 기반 외부 데이터 접근을 위한 MCP 서버
"""

import asyncio
import aiohttp
import json
import ssl
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlparse
import logging

# MCP 관련 임포트
try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    print("⚠️ MCP 패키지가 설치되지 않음. 시뮬레이션 모드로 실행")
    MCP_AVAILABLE = False


class FetchMCPServer:
    """HTTP 요청 처리를 위한 MCP 서버"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.server = Server("fetch-server") if MCP_AVAILABLE else None
        self.request_history: List[Dict[str, Any]] = []
        
        # 보안 설정
        self.allowed_domains = [
            "boardgamegeek.com",
            "api.boardgameatlas.com",
            "www.boardgamegeek.com",
            "geekdo-static.com",
            "cf.geekdo-static.com"
        ]
        
        self.timeout = aiohttp.ClientTimeout(total=30)
        
        if MCP_AVAILABLE and self.server:
            self._register_tools()
    
    def _register_tools(self):
        """MCP 도구들 등록"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="fetch_get",
                    description="HTTP GET 요청 실행",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "요청할 URL"},
                            "headers": {"type": "object", "description": "HTTP 헤더 (선택사항)"},
                            "params": {"type": "object", "description": "쿼리 매개변수 (선택사항)"},
                            "timeout": {"type": "number", "description": "타임아웃 (초, 기본값: 30)"}
                        },
                        "required": ["url"]
                    }
                ),
                Tool(
                    name="fetch_post",
                    description="HTTP POST 요청 실행",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "요청할 URL"},
                            "data": {"type": "object", "description": "POST 데이터"},
                            "headers": {"type": "object", "description": "HTTP 헤더 (선택사항)"},
                            "json_data": {"type": "boolean", "description": "JSON으로 전송할지 여부 (기본값: true)"},
                            "timeout": {"type": "number", "description": "타임아웃 (초, 기본값: 30)"}
                        },
                        "required": ["url", "data"]
                    }
                ),
                Tool(
                    name="fetch_json",
                    description="JSON API 호출 및 파싱",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "JSON API URL"},
                            "headers": {"type": "object", "description": "HTTP 헤더 (선택사항)"},
                            "params": {"type": "object", "description": "쿼리 매개변수 (선택사항)"}
                        },
                        "required": ["url"]
                    }
                ),
                Tool(
                    name="download_file",
                    description="파일 다운로드",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "다운로드할 파일 URL"},
                            "save_path": {"type": "string", "description": "저장할 경로 (선택사항)"},
                            "max_size": {"type": "number", "description": "최대 파일 크기 (MB, 기본값: 10)"}
                        },
                        "required": ["url"]
                    }
                ),
                Tool(
                    name="get_request_history",
                    description="요청 기록 조회",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {"type": "number", "description": "조회할 기록 수 (기본값: 10)"}
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """도구 호출 처리"""
            
            try:
                if name == "fetch_get":
                    result = await self.fetch_get(**arguments)
                elif name == "fetch_post":
                    result = await self.fetch_post(**arguments)
                elif name == "fetch_json":
                    result = await self.fetch_json(**arguments)
                elif name == "download_file":
                    result = await self.download_file(**arguments)
                elif name == "get_request_history":
                    result = await self.get_request_history(**arguments)
                else:
                    result = {"error": f"알 수 없는 도구: {name}"}
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, indent=2)
                )]
                
            except Exception as e:
                error_result = {
                    "error": f"도구 실행 실패: {str(e)}",
                    "tool": name,
                    "arguments": arguments
                }
                return [TextContent(
                    type="text", 
                    text=json.dumps(error_result, ensure_ascii=False, indent=2)
                )]
    
    async def initialize_session(self):
        """HTTP 세션 초기화"""
        if not self.session or self.session.closed:
            connector = aiohttp.TCPConnector(
                ssl=ssl.create_default_context(),
                limit=100,
                limit_per_host=30
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout,
                headers={
                    "User-Agent": "TableGameMate/1.0 (Board Game AI Assistant)"
                }
            )
    
    def _is_domain_allowed(self, url: str) -> bool:
        """도메인 허용 여부 확인"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # 정확한 도메인 매칭 또는 서브도메인 허용
            for allowed in self.allowed_domains:
                if domain == allowed or domain.endswith(f".{allowed}"):
                    return True
            
            return False
        except:
            return False
    
    def _log_request(self, method: str, url: str, status: int, response_size: int):
        """요청 기록"""
        self.request_history.append({
            "method": method,
            "url": url,
            "status": status,
            "response_size": response_size,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # 기록 제한 (최근 100개만)
        if len(self.request_history) > 100:
            self.request_history = self.request_history[-100:]
    
    async def fetch_get(
        self, 
        url: str, 
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """HTTP GET 요청"""
        
        if not self._is_domain_allowed(url):
            return {"error": f"허용되지 않은 도메인: {url}"}
        
        await self.initialize_session()
        
        try:
            request_timeout = aiohttp.ClientTimeout(total=timeout or 30)
            
            async with self.session.get(
                url, 
                headers=headers, 
                params=params,
                timeout=request_timeout
            ) as response:
                
                content = await response.text()
                content_length = len(content.encode('utf-8'))
                
                self._log_request("GET", url, response.status, content_length)
                
                return {
                    "success": True,
                    "status": response.status,
                    "headers": dict(response.headers),
                    "content": content,
                    "content_length": content_length,
                    "url": str(response.url)
                }
                
        except asyncio.TimeoutError:
            return {"error": "요청 타임아웃", "url": url}
        except Exception as e:
            return {"error": f"GET 요청 실패: {str(e)}", "url": url}
    
    async def fetch_post(
        self,
        url: str,
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        json_data: bool = True,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """HTTP POST 요청"""
        
        if not self._is_domain_allowed(url):
            return {"error": f"허용되지 않은 도메인: {url}"}
        
        await self.initialize_session()
        
        try:
            request_timeout = aiohttp.ClientTimeout(total=timeout or 30)
            
            if json_data:
                async with self.session.post(
                    url,
                    json=data,
                    headers=headers,
                    timeout=request_timeout
                ) as response:
                    content = await response.text()
            else:
                async with self.session.post(
                    url,
                    data=data,
                    headers=headers,
                    timeout=request_timeout
                ) as response:
                    content = await response.text()
            
            content_length = len(content.encode('utf-8'))
            self._log_request("POST", url, response.status, content_length)
            
            return {
                "success": True,
                "status": response.status,
                "headers": dict(response.headers),
                "content": content,
                "content_length": content_length
            }
            
        except asyncio.TimeoutError:
            return {"error": "요청 타임아웃", "url": url}
        except Exception as e:
            return {"error": f"POST 요청 실패: {str(e)}", "url": url}
    
    async def fetch_json(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """JSON API 호출"""
        
        result = await self.fetch_get(url, headers, params)
        
        if not result.get("success"):
            return result
        
        try:
            json_data = json.loads(result["content"])
            result["json"] = json_data
            result["content_type"] = "application/json"
            return result
            
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON 파싱 실패: {str(e)}",
                "raw_content": result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"]
            }
    
    async def download_file(
        self,
        url: str,
        save_path: Optional[str] = None,
        max_size: float = 10.0  # MB
    ) -> Dict[str, Any]:
        """파일 다운로드"""
        
        if not self._is_domain_allowed(url):
            return {"error": f"허용되지 않은 도메인: {url}"}
        
        await self.initialize_session()
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return {"error": f"다운로드 실패: HTTP {response.status}"}
                
                # 크기 확인
                content_length = response.headers.get('content-length')
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > max_size:
                        return {"error": f"파일 크기 초과: {size_mb:.2f}MB > {max_size}MB"}
                
                content = await response.read()
                actual_size = len(content) / (1024 * 1024)
                
                if actual_size > max_size:
                    return {"error": f"파일 크기 초과: {actual_size:.2f}MB > {max_size}MB"}
                
                # 파일 저장 (경로가 지정된 경우)
                if save_path:
                    with open(save_path, 'wb') as f:
                        f.write(content)
                
                self._log_request("GET", url, response.status, len(content))
                
                return {
                    "success": True,
                    "size_mb": actual_size,
                    "content_type": response.headers.get('content-type'),
                    "saved_path": save_path,
                    "content": content if not save_path else None
                }
                
        except Exception as e:
            return {"error": f"파일 다운로드 실패: {str(e)}"}
    
    async def get_request_history(self, limit: int = 10) -> Dict[str, Any]:
        """요청 기록 조회"""
        
        recent_requests = self.request_history[-limit:] if self.request_history else []
        
        return {
            "success": True,
            "total_requests": len(self.request_history),
            "recent_requests": recent_requests,
            "summary": {
                "successful_requests": len([r for r in recent_requests if 200 <= r["status"] < 300]),
                "failed_requests": len([r for r in recent_requests if r["status"] >= 400]),
                "total_data_transferred": sum(r["response_size"] for r in recent_requests)
            }
        }
    
    async def close(self):
        """리소스 정리"""
        if self.session and not self.session.closed:
            await self.session.close()


# MCP 서버 실행
async def main():
    """Fetch MCP 서버 실행"""
    
    if not MCP_AVAILABLE:
        print("❌ MCP 패키지가 필요합니다: pip install mcp")
        return
    
    fetch_server = FetchMCPServer()
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            await fetch_server.server.run(
                read_stream, 
                write_stream, 
                InitializationOptions(
                    server_name="fetch-server",
                    server_version="1.0.0",
                    capabilities={}
                )
            )
    finally:
        await fetch_server.close()


if __name__ == "__main__":
    asyncio.run(main()) 