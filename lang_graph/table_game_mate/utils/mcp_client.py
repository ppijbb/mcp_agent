"""
MCP Client - Model Context Protocol 클라이언트

이 클래스는 Agent가 외부 데이터에 접근할 수 있게 하는 핵심 구성요소입니다.
계획서에 명시된 다양한 MCP 서버들과 연동하여 Agent의 인식 능력을 확장합니다.

지원하는 MCP 서버들:
- brave-search: 웹 검색
- fetch: 웹 콘텐츠 수집
- filesystem: 로컬 파일 관리
- sqlite: 데이터베이스 연동
- memory: 세션 메모리 관리
- bgg-api: BoardGameGeek API (커스텀)
"""

import aiohttp
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging


class MCPClient:
    """
    MCP 서버 연결 클라이언트
    
    Agent의 외부 데이터 접근 능력을 제공합니다.
    이것이 Agent를 단순한 로컬 시스템에서 벗어나 
    실제 세계의 정보에 접근할 수 있게 하는 핵심입니다.
    """
    
    def __init__(self, server_configs: Optional[Dict[str, str]] = None):
        """
        Args:
            server_configs: MCP 서버 설정 {서버명: 엔드포인트}
        """
        # 기본 MCP 서버 설정
        self.default_servers = {
            "brave-search": "http://localhost:3001",
            "fetch": "http://localhost:3002", 
            "filesystem": "http://localhost:3003",
            "sqlite": "http://localhost:3004",
            "memory": "http://localhost:3005",
            "bgg-api": "http://localhost:3006",  # 커스텀 BGG API 서버
            "game-rules": "http://localhost:3007",  # 커스텀 게임 규칙 서버
        }
        
        # 사용자 설정으로 덮어쓰기
        self.servers = {**self.default_servers}
        if server_configs:
            self.servers.update(server_configs)
        
        # 연결 상태 관리
        self.connection_status = {}
        self.usage_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "calls_by_server": {},
            "created_at": datetime.now().isoformat()
        }
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
    
    async def call(self, server_name: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP 서버 호출 - Agent의 외부 데이터 접근
        
        이 메서드를 통해 Agent는:
        - 웹에서 게임 규칙 검색
        - BoardGameGeek에서 게임 정보 수집
        - 로컬 파일에서 게임 데이터 로드
        - 데이터베이스에 경험 저장
        
        Args:
            server_name: MCP 서버 이름
            method: 호출할 메서드
            params: 메서드 파라미터
            
        Returns:
            MCP 서버 응답
        """
        if server_name not in self.servers:
            raise MCPClientError(f"Unknown MCP server: {server_name}")
        
        endpoint = f"{self.servers[server_name]}/{method}"
        
        try:
            self.usage_stats["total_calls"] += 1
            
            # 서버별 호출 통계
            if server_name not in self.usage_stats["calls_by_server"]:
                self.usage_stats["calls_by_server"][server_name] = 0
            self.usage_stats["calls_by_server"][server_name] += 1
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        self.usage_stats["successful_calls"] += 1
                        self.connection_status[server_name] = "connected"
                        return result
                    else:
                        error_text = await response.text()
                        raise MCPClientError(f"MCP server error {response.status}: {error_text}")
        
        except asyncio.TimeoutError:
            self.usage_stats["failed_calls"] += 1
            self.connection_status[server_name] = "timeout"
            raise MCPClientError(f"MCP server {server_name} timeout")
        
        except aiohttp.ClientError as e:
            self.usage_stats["failed_calls"] += 1
            self.connection_status[server_name] = "error"
            raise MCPClientError(f"MCP client error: {str(e)}")
        
        except Exception as e:
            self.usage_stats["failed_calls"] += 1
            self.connection_status[server_name] = "error"
            raise MCPClientError(f"Unexpected error: {str(e)}")
    
    async def search_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        웹 검색 - 게임 규칙 및 정보 검색
        
        Agent가 모르는 게임의 규칙을 실시간으로 검색할 수 있습니다.
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            
        Returns:
            검색 결과
        """
        return await self.call("brave-search", "search", {
            "query": query,
            "count": max_results
        })
    
    async def fetch_content(self, url: str) -> Dict[str, Any]:
        """
        웹 콘텐츠 수집 - 게임 규칙 문서 다운로드
        
        Args:
            url: 수집할 URL
            
        Returns:
            웹 콘텐츠
        """
        return await self.call("fetch", "get", {"url": url})
    
    async def save_to_memory(self, key: str, value: Any) -> Dict[str, Any]:
        """
        메모리에 데이터 저장 - Agent 학습 데이터 보관
        
        Args:
            key: 저장 키
            value: 저장할 값
            
        Returns:
            저장 결과
        """
        return await self.call("memory", "store", {
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat()
        })
    
    async def load_from_memory(self, key: str) -> Dict[str, Any]:
        """
        메모리에서 데이터 로드
        
        Args:
            key: 로드할 키
            
        Returns:
            저장된 데이터
        """
        return await self.call("memory", "retrieve", {"key": key})
    
    async def search_boardgame(self, game_name: str) -> Dict[str, Any]:
        """
        BoardGameGeek에서 게임 검색
        
        Agent가 게임 정보를 자동으로 수집할 수 있습니다.
        
        Args:
            game_name: 게임 이름
            
        Returns:
            게임 정보
        """
        return await self.call("bgg-api", "search", {"name": game_name})
    
    async def get_game_details(self, bgg_id: int) -> Dict[str, Any]:
        """
        BoardGameGeek에서 게임 상세 정보 조회
        
        Args:
            bgg_id: BoardGameGeek 게임 ID
            
        Returns:
            게임 상세 정보
        """
        return await self.call("bgg-api", "details", {"id": bgg_id})
    
    async def parse_game_rules(self, rules_text: str, game_type: str) -> Dict[str, Any]:
        """
        게임 규칙 파싱 - 자연어 규칙을 구조화
        
        Args:
            rules_text: 규칙 텍스트
            game_type: 게임 타입
            
        Returns:
            구조화된 규칙
        """
        return await self.call("game-rules", "parse", {
            "text": rules_text,
            "type": game_type
        })
    
    async def save_game_record(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        게임 기록 저장 - Agent 학습을 위한 데이터 축적
        
        Args:
            game_data: 게임 데이터
            
        Returns:
            저장 결과
        """
        return await self.call("sqlite", "insert", {
            "table": "game_records",
            "data": game_data
        })
    
    async def load_game_history(self, agent_id: str, limit: int = 10) -> Dict[str, Any]:
        """
        Agent의 게임 히스토리 로드
        
        Args:
            agent_id: Agent ID
            limit: 로드할 기록 수
            
        Returns:
            게임 히스토리
        """
        return await self.call("sqlite", "select", {
            "table": "game_records",
            "where": {"agent_id": agent_id},
            "limit": limit,
            "order_by": "created_at DESC"
        })
    
    async def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        로컬 파일 읽기
        
        Args:
            file_path: 파일 경로
            
        Returns:
            파일 내용
        """
        return await self.call("filesystem", "read", {"path": file_path})
    
    async def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        로컬 파일 쓰기
        
        Args:
            file_path: 파일 경로
            content: 파일 내용
            
        Returns:
            쓰기 결과
        """
        return await self.call("filesystem", "write", {
            "path": file_path,
            "content": content
        })
    
    async def test_all_connections(self) -> Dict[str, bool]:
        """
        모든 MCP 서버 연결 테스트
        
        Returns:
            서버별 연결 상태
        """
        results = {}
        
        for server_name in self.servers.keys():
            try:
                # 각 서버의 health check 엔드포인트 호출
                await self.call(server_name, "health", {})
                results[server_name] = True
            except Exception as e:
                self.logger.warning(f"MCP server {server_name} connection failed: {e}")
                results[server_name] = False
        
        return results
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """사용 통계 반환"""
        success_rate = 0.0
        if self.usage_stats["total_calls"] > 0:
            success_rate = self.usage_stats["successful_calls"] / self.usage_stats["total_calls"]
        
        return {
            **self.usage_stats,
            "success_rate": success_rate,
            "connection_status": self.connection_status,
            "available_servers": list(self.servers.keys()),
            "current_time": datetime.now().isoformat()
        }
    
    def add_server(self, name: str, endpoint: str):
        """새로운 MCP 서버 추가"""
        self.servers[name] = endpoint
        self.logger.info(f"Added MCP server: {name} -> {endpoint}")
    
    def remove_server(self, name: str):
        """MCP 서버 제거"""
        if name in self.servers:
            del self.servers[name]
            if name in self.connection_status:
                del self.connection_status[name]
            self.logger.info(f"Removed MCP server: {name}")


class MCPClientError(Exception):
    """MCP 클라이언트 에러"""
    pass


# 전역 MCP 클라이언트 인스턴스
_global_mcp_client: Optional[MCPClient] = None


def get_mcp_client(server_configs: Optional[Dict[str, str]] = None) -> MCPClient:
    """전역 MCP 클라이언트 반환"""
    global _global_mcp_client
    
    if _global_mcp_client is None:
        _global_mcp_client = MCPClient(server_configs)
    
    return _global_mcp_client


def reset_mcp_client():
    """MCP 클라이언트 초기화 (테스트용)"""
    global _global_mcp_client
    _global_mcp_client = None 