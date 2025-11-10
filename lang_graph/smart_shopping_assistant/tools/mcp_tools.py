"""
MCP Tools Wrapper for LangChain (Smart Shopping Assistant)

MCP 서버 도구를 LangChain Tool로 변환
"""

import logging
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.tools import tool, BaseTool, StructuredTool
from pydantic import BaseModel, Field

from srcs.core.config.loader import settings

logger = logging.getLogger(__name__)


class FileSystemReadInput(BaseModel):
    """Filesystem 읽기 입력 스키마"""
    file_path: str = Field(description="읽을 파일 경로")


class FileSystemWriteInput(BaseModel):
    """Filesystem 쓰기 입력 스키마"""
    file_path: str = Field(description="쓸 파일 경로")
    content: str = Field(description="파일 내용")


class SearchInput(BaseModel):
    """검색 입력 스키마"""
    query: str = Field(description="검색 쿼리")
    max_results: int = Field(default=10, description="최대 결과 수")


class FetchInput(BaseModel):
    """Fetch 입력 스키마"""
    url: str = Field(description="가져올 URL")


class MCPToolsWrapper:
    """
    MCP 도구를 LangChain Tool로 래핑
    
    MCP 서버와 직접 통합하지 않고, LangChain Tool 인터페이스로 제공
    실제 MCP 통합은 mcp_agent 라이브러리를 통해 수행
    """
    
    def __init__(self):
        """MCPToolsWrapper 초기화"""
        self.tools: List[BaseTool] = []
        self._initialize_tools()
    
    def _initialize_tools(self):
        """MCP 도구 초기화"""
        # Filesystem 도구
        self.tools.append(self._create_filesystem_read_tool())
        self.tools.append(self._create_filesystem_write_tool())
        
        # 검색 도구
        self.tools.append(self._create_search_tool())
        
        # Fetch 도구
        self.tools.append(self._create_fetch_tool())
        
        logger.info(f"Initialized {len(self.tools)} MCP tools")
    
    def _create_filesystem_read_tool(self) -> BaseTool:
        """Filesystem 읽기 도구 생성"""
        
        @tool("mcp_filesystem_read", args_schema=FileSystemReadInput)
        def filesystem_read(file_path: str) -> str:
            """
            파일 시스템에서 파일 읽기
            
            Args:
                file_path: 읽을 파일 경로
            
            Returns:
                파일 내용
            """
            try:
                # 실제 구현에서는 MCP 서버를 통해 파일 읽기
                # 여기서는 기본 파일 시스템 접근
                path = Path(file_path)
                if not path.exists():
                    return f"Error: File not found: {file_path}"
                
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                logger.info(f"Read file: {file_path} ({len(content)} bytes)")
                return content
            
            except Exception as e:
                error_msg = f"Error reading file {file_path}: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return filesystem_read
    
    def _create_filesystem_write_tool(self) -> BaseTool:
        """Filesystem 쓰기 도구 생성"""
        
        @tool("mcp_filesystem_write", args_schema=FileSystemWriteInput)
        def filesystem_write(file_path: str, content: str) -> str:
            """
            파일 시스템에 파일 쓰기
            
            Args:
                file_path: 쓸 파일 경로
                content: 파일 내용
            
            Returns:
                성공 메시지
            """
            try:
                # 실제 구현에서는 MCP 서버를 통해 파일 쓰기
                path = Path(file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"Wrote file: {file_path} ({len(content)} bytes)")
                return f"Successfully wrote {len(content)} bytes to {file_path}"
            
            except Exception as e:
                error_msg = f"Error writing file {file_path}: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return filesystem_write
    
    def _create_search_tool(self) -> BaseTool:
        """검색 도구 생성"""
        
        @tool("mcp_g_search", args_schema=SearchInput)
        def g_search(query: str, max_results: int = 10) -> str:
            """
            Google 검색 수행
            
            Args:
                query: 검색 쿼리
                max_results: 최대 결과 수
            
            Returns:
                검색 결과 (JSON 형식)
            """
            try:
                # 실제 구현에서는 MCP g-search 서버를 통해 검색
                # 여기서는 기본 구현 (실제 MCP 통합 필요)
                logger.info(f"Searching for: {query} (max_results: {max_results})")
                
                # MCP 서버가 없으면 에러 메시지 반환
                # 실제 구현에서는 MCP 클라이언트를 통해 검색 수행
                return f"Search results for '{query}' would be retrieved via MCP g-search server. " \
                       f"Please ensure MCP g-search server is configured."
            
            except Exception as e:
                error_msg = f"Error searching for '{query}': {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return g_search
    
    def _create_fetch_tool(self) -> BaseTool:
        """Fetch 도구 생성"""
        
        @tool("mcp_fetch", args_schema=FetchInput)
        def fetch(url: str) -> str:
            """
            URL에서 콘텐츠 가져오기
            
            Args:
                url: 가져올 URL
            
            Returns:
                URL 콘텐츠
            """
            try:
                # 실제 구현에서는 MCP fetch 서버를 통해 가져오기
                import requests
                
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                content = response.text
                logger.info(f"Fetched URL: {url} ({len(content)} bytes)")
                
                return content
            
            except Exception as e:
                error_msg = f"Error fetching URL {url}: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return fetch
    
    def get_tools(self) -> List[BaseTool]:
        """모든 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 도구 찾기"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

