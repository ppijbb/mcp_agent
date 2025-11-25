"""
MCP Client for Table Game Mate

BoardGameGeek API 및 웹 검색을 위한 MCP 클라이언트
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import AsyncExitStack

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import TextContent
    from mcp.shared.exceptions import McpError
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    TextContent = None
    McpError = Exception

logger = logging.getLogger(__name__)


class MCPClientError(Exception):
    """MCP 클라이언트 오류"""
    pass


class MCPClient:
    """MCP 클라이언트 - BGG API 및 웹 검색 지원"""
    
    def __init__(self):
        """MCP 클라이언트 초기화"""
        if not MCP_AVAILABLE:
            raise MCPClientError("MCP 라이브러리가 설치되지 않았습니다. 'pip install mcp'를 실행하세요.")
        
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stacks: Dict[str, AsyncExitStack] = {}
        self._initialized = False
    
    async def _ensure_connected(self, server_name: str):
        """서버 연결 확인 및 초기화"""
        if server_name in self.sessions:
            return
        
        # MCP 서버 설정 (표준 MCP 서버 또는 커스텀 서버)
        # 실제 구현에서는 설정 파일이나 환경 변수에서 로드
        server_configs = {
            "bgg-api": {
                "command": "python",
                "args": ["-m", "lang_graph.table_game_mate.mcp_servers.bgg_mcp_server"],
            },
            "brave-search": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            },
            "fetch": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-fetch"],
            },
        }
        
        if server_name not in server_configs:
            raise MCPClientError(f"알 수 없는 서버: {server_name}")
        
        config = server_configs[server_name]
        
        try:
            exit_stack = AsyncExitStack()
            self.exit_stacks[server_name] = exit_stack
            
            # Stdio 서버 파라미터 생성
            server_params = StdioServerParameters(
                command=config["command"],
                args=config["args"],
            )
            
            # 클라이언트 연결
            stdio_streams = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            session = await exit_stack.enter_async_context(
                ClientSession(*stdio_streams)
            )
            
            # 초기화
            await session.initialize()
            
            self.sessions[server_name] = session
            logger.info(f"MCP 서버 '{server_name}' 연결 완료")
            
        except Exception as e:
            logger.error(f"MCP 서버 '{server_name}' 연결 실패: {e}")
            raise MCPClientError(f"서버 연결 실패: {e}")
    
    async def call(
        self,
        server_name: str,
        method: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        MCP 서버의 tool 호출
        
        Args:
            server_name: 서버 이름 (예: "bgg-api")
            method: 호출할 tool 이름
            params: 파라미터 딕셔너리
            
        Returns:
            결과 딕셔너리
        """
        await self._ensure_connected(server_name)
        session = self.sessions[server_name]
        
        try:
            # Tool 호출
            result = await session.call_tool(method, params or {})
            
            # 결과 파싱
            output = {}
            if result.content:
                # TextContent 추출
                text_parts = []
                for content in result.content:
                    if isinstance(content, TextContent):
                        text_parts.append(content.text)
                
                if text_parts:
                    # JSON 파싱 시도
                    try:
                        import json
                        output = json.loads("".join(text_parts))
                    except json.JSONDecodeError:
                        output = {"result": "".join(text_parts)}
                else:
                    output = {"result": str(result)}
            
            return {"success": True, **output}
            
        except McpError as e:
            logger.error(f"MCP tool 호출 오류 ({server_name}.{method}): {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"예상치 못한 오류 ({server_name}.{method}): {e}")
            return {"success": False, "error": str(e)}
    
    async def search_web(
        self,
        query: str,
        max_results: int = 3
    ) -> Dict[str, Any]:
        """
        웹 검색 (brave-search 서버 사용)
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            
        Returns:
            검색 결과 딕셔너리
        """
        try:
            result = await self.call(
                server_name="brave-search",
                method="brave_search",
                params={
                    "query": query,
                    "max_results": max_results
                }
            )
            
            if result.get("success"):
                # 결과 형식 변환
                results = result.get("results", [])
                return {
                    "success": True,
                    "results": results[:max_results]
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"웹 검색 오류: {e}")
            return {"success": False, "error": str(e), "results": []}
    
    async def fetch_content(self, url: str) -> Dict[str, Any]:
        """
        URL에서 콘텐츠 가져오기 (fetch 서버 사용)
        
        Args:
            url: 가져올 URL
            
        Returns:
            콘텐츠 딕셔너리
        """
        try:
            result = await self.call(
                server_name="fetch",
                method="fetch",
                params={"url": url}
            )
            
            if result.get("success"):
                return {
                    "success": True,
                    "content": result.get("content", result.get("result", ""))
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"콘텐츠 가져오기 오류: {e}")
            return {"success": False, "error": str(e), "content": ""}
    
    async def close(self):
        """모든 연결 종료"""
        for server_name, exit_stack in self.exit_stacks.items():
            try:
                await exit_stack.aclose()
                logger.info(f"MCP 서버 '{server_name}' 연결 종료")
            except Exception as e:
                logger.error(f"서버 '{server_name}' 종료 오류: {e}")
        
        self.sessions.clear()
        self.exit_stacks.clear()
    
    def __del__(self):
        """소멸자 - 비동기 정리"""
        if self.exit_stacks:
            # 비동기 정리는 명시적으로 close() 호출 필요
            logger.warning("MCPClient가 명시적으로 close()되지 않았습니다.")

