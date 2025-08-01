"""
Review Generator - LLM을 사용하여 코드 리뷰 생성

이 모듈은 MCP를 통해 LLM을 호출하여 코드 리뷰를 생성합니다.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from mcp.client import MCPClient

logger = logging.getLogger(__name__)

class ReviewGenerator:
    """LLM을 사용하여 코드 리뷰를 생성하는 클래스"""
    
    def __init__(self, mcp_server_url: str = None):
        """
        리뷰 생성기 초기화
        
        Args:
            mcp_server_url (str, optional): MCP 서버 URL. 기본값은 환경 변수 MCP_SERVER_URL.
        """
        self.mcp_server_url = mcp_server_url or os.environ.get("MCP_SERVER_URL", "http://localhost:8000")
        self.mcp_client = MCPClient(self.mcp_server_url)
        logger.info(f"ReviewGenerator가 초기화되었습니다. MCP 서버: {self.mcp_server_url}")
    
    async def generate_review(self, diff_content: str, 
                             pr_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        코드 diff를 기반으로 리뷰를 생성합니다.
        
        Args:
            diff_content (str): PR의 diff 내용
            pr_metadata (Dict[str, Any], optional): PR 메타데이터 (제목, 설명 등)
            
        Returns:
            Dict[str, Any]: 생성된 리뷰 정보
        """
        try:
            # PR 메타데이터 준비
            metadata = pr_metadata or {}
            pr_title = metadata.get("title", "")
            pr_description = metadata.get("description", "")
            
            # MCP 클라이언트를 통해 리뷰 생성 요청
            response = await self.mcp_client.call_tool(
                "generate-code-review",
                {
                    "diff_content": diff_content,
                    "pr_title": pr_title,
                    "pr_description": pr_description
                }
            )
            
            # 응답 처리
            if not response or "review" not in response:
                raise ValueError("리뷰 생성에 실패했습니다.")
            
            return response
        except Exception as e:
            logger.error(f"리뷰 생성 중 오류 발생: {e}")
            raise
    
    async def generate_file_review(self, file_patch: str, 
                                  file_path: str) -> List[Dict[str, Any]]:
        """
        특정 파일의 변경사항에 대한 라인별 리뷰를 생성합니다.
        
        Args:
            file_patch (str): 파일의 patch/diff 내용
            file_path (str): 파일 경로
            
        Returns:
            List[Dict[str, Any]]: 라인별 리뷰 코멘트 목록
        """
        try:
            # 파일 확장자 추출
            file_extension = file_path.split(".")[-1] if "." in file_path else ""
            
            # MCP 클라이언트를 통해 파일 리뷰 생성 요청
            response = await self.mcp_client.call_tool(
                "generate-file-review",
                {
                    "file_patch": file_patch,
                    "file_path": file_path,
                    "file_extension": file_extension
                }
            )
            
            # 응답 처리
            if not response or "comments" not in response:
                raise ValueError("파일 리뷰 생성에 실패했습니다.")
            
            return response["comments"]
        except Exception as e:
            logger.error(f"파일 리뷰 생성 중 오류 발생: {e}")
            raise
    
    async def analyze_code_quality(self, code_content: str, 
                                  file_path: str) -> Dict[str, Any]:
        """
        코드 품질을 분석합니다.
        
        Args:
            code_content (str): 코드 내용
            file_path (str): 파일 경로
            
        Returns:
            Dict[str, Any]: 코드 품질 분석 결과
        """
        try:
            # 파일 확장자 추출
            file_extension = file_path.split(".")[-1] if "." in file_path else ""
            
            # MCP 클라이언트를 통해 코드 품질 분석 요청
            response = await self.mcp_client.call_tool(
                "analyze-code-quality",
                {
                    "code_content": code_content,
                    "file_path": file_path,
                    "file_extension": file_extension
                }
            )
            
            # 응답 처리
            if not response:
                raise ValueError("코드 품질 분석에 실패했습니다.")
            
            return response
        except Exception as e:
            logger.error(f"코드 품질 분석 중 오류 발생: {e}")
            raise
    
    async def generate_summary_review(self, pr_files: List[Dict[str, Any]], 
                                     pr_metadata: Dict[str, Any] = None) -> str:
        """
        PR 전체에 대한 요약 리뷰를 생성합니다.
        
        Args:
            pr_files (List[Dict[str, Any]]): PR의 파일 변경사항 목록
            pr_metadata (Dict[str, Any], optional): PR 메타데이터 (제목, 설명 등)
            
        Returns:
            str: 생성된 요약 리뷰
        """
        try:
            # PR 메타데이터 준비
            metadata = pr_metadata or {}
            pr_title = metadata.get("title", "")
            pr_description = metadata.get("description", "")
            
            # 파일 정보 요약
            files_summary = []
            for file in pr_files:
                files_summary.append({
                    "filename": file["filename"],
                    "status": file["status"],
                    "additions": file["additions"],
                    "deletions": file["deletions"],
                    "changes": file["changes"]
                })
            
            # MCP 클라이언트를 통해 요약 리뷰 생성 요청
            response = await self.mcp_client.call_tool(
                "generate-summary-review",
                {
                    "files": files_summary,
                    "pr_title": pr_title,
                    "pr_description": pr_description
                }
            )
            
            # 응답 처리
            if not response or "summary" not in response:
                raise ValueError("요약 리뷰 생성에 실패했습니다.")
            
            return response["summary"]
        except Exception as e:
            logger.error(f"요약 리뷰 생성 중 오류 발생: {e}")
            raise 