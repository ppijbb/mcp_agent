"""
Review Generator - LLM을 사용하여 코드 리뷰 생성 (NO FALLBACK MODE)

이 모듈은 MCP를 통해 LLM을 호출하여 코드 리뷰를 생성합니다.
모든 오류는 fallback 없이 즉시 상위로 전파됩니다.
"""

import os
import logging
import json
import sys
from typing import Dict, List, Any, Optional, Tuple
from mcp.client import MCPClient

from .config import config

logger = logging.getLogger(__name__)

class ReviewGenerator:
    """LLM을 사용하여 코드 리뷰를 생성하는 클래스 - NO FALLBACK MODE"""
    
    def __init__(self, mcp_server_url: str = None):
        """
        리뷰 생성기 초기화 - 실패 시 즉시 종료
        
        Args:
            mcp_server_url (str, optional): MCP 서버 URL. 기본값은 환경 변수 MCP_SERVER_URL.
        """
        try:
            self.mcp_server_url = mcp_server_url or os.environ.get("MCP_SERVER_URL", "http://localhost:8000")
            if not self.mcp_server_url:
                raise ValueError("MCP 서버 URL이 설정되지 않았습니다.")
            
            self.mcp_client = MCPClient(self.mcp_server_url)
            if not self.mcp_client:
                raise ValueError("MCP 클라이언트 초기화에 실패했습니다.")
            
            logger.info(f"ReviewGenerator가 초기화되었습니다. MCP 서버: {self.mcp_server_url} (NO FALLBACK MODE)")
        except Exception as e:
            logger.error(f"ReviewGenerator 초기화 중 치명적 오류 발생: {e}")
            if config.llm.fail_on_llm_error:
                sys.exit(1)
            raise
    
    async def generate_review(self, diff_content: str, 
                             pr_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        코드 diff를 기반으로 리뷰를 생성합니다 - NO FALLBACK
        
        Args:
            diff_content (str): PR의 diff 내용
            pr_metadata (Dict[str, Any], optional): PR 메타데이터 (제목, 설명 등)
            
        Returns:
            Dict[str, Any]: 생성된 리뷰 정보
            
        Raises:
            ValueError: 필수 파라미터가 없거나 리뷰 생성에 실패한 경우
        """
        # 필수 파라미터 검증
        if not diff_content:
            raise ValueError("diff_content가 비어있습니다.")
        
        # PR 메타데이터 준비
        metadata = pr_metadata or {}
        pr_title = metadata.get("title", "")
        pr_description = metadata.get("description", "")
        
        logger.info(f"리뷰 생성 시작: PR 제목={pr_title[:50]}...")
        
        # MCP 클라이언트를 통해 리뷰 생성 요청
        response = await self.mcp_client.call_tool(
            "generate-code-review",
            {
                "diff_content": diff_content,
                "pr_title": pr_title,
                "pr_description": pr_description
            }
        )
        
        # 응답 검증
        if not response:
            raise ValueError("MCP 서버로부터 응답을 받지 못했습니다.")
        
        if "review" not in response:
            raise ValueError("유효한 리뷰 데이터가 생성되지 않았습니다.")
        
        # 리뷰 내용 검증
        review_content = response.get("review")
        if not review_content or not isinstance(review_content, str) or len(review_content.strip()) == 0:
            raise ValueError("리뷰 내용이 비어있거나 유효하지 않습니다.")
        
        logger.info("리뷰 생성 완료")
        return response
    
    async def generate_file_review(self, file_patch: str, 
                                  file_path: str) -> List[Dict[str, Any]]:
        """
        특정 파일의 변경사항에 대한 라인별 리뷰를 생성합니다 - NO FALLBACK
        
        Args:
            file_patch (str): 파일의 patch/diff 내용
            file_path (str): 파일 경로
            
        Returns:
            List[Dict[str, Any]]: 라인별 리뷰 코멘트 목록
            
        Raises:
            ValueError: 필수 파라미터가 없거나 파일 리뷰 생성에 실패한 경우
        """
        # 필수 파라미터 검증
        if not file_patch:
            raise ValueError("file_patch가 비어있습니다.")
        if not file_path:
            raise ValueError("file_path가 비어있습니다.")
        
        # 파일 확장자 추출
        file_extension = file_path.split(".")[-1] if "." in file_path else ""
        
        logger.info(f"파일 리뷰 생성 시작: {file_path}")
        
        # MCP 클라이언트를 통해 파일 리뷰 생성 요청
        response = await self.mcp_client.call_tool(
            "generate-file-review",
            {
                "file_patch": file_patch,
                "file_path": file_path,
                "file_extension": file_extension
            }
        )
        
        # 응답 검증
        if not response:
            raise ValueError(f"파일 리뷰 생성 실패: {file_path} - MCP 서버 응답 없음")
        
        if "comments" not in response:
            raise ValueError(f"파일 리뷰 생성 실패: {file_path} - 유효한 코멘트 데이터 없음")
        
        comments = response["comments"]
        if not isinstance(comments, list):
            raise ValueError(f"파일 리뷰 생성 실패: {file_path} - 코멘트가 리스트 형태가 아닙니다.")
        
        # 각 코멘트 검증
        for i, comment in enumerate(comments):
            if not isinstance(comment, dict):
                raise ValueError(f"파일 리뷰 생성 실패: {file_path} - 코멘트 {i}가 딕셔너리 형태가 아닙니다.")
            
            required_fields = ["path", "position", "body"]
            for field in required_fields:
                if field not in comment:
                    raise ValueError(f"파일 리뷰 생성 실패: {file_path} - 코멘트 {i}에 {field} 필드가 없습니다.")
        
        logger.info(f"파일 리뷰 생성 완료: {file_path}, 코멘트 수: {len(comments)}")
        return comments
    
    async def analyze_code_quality(self, code_content: str, 
                                  file_path: str) -> Dict[str, Any]:
        """
        코드 품질을 분석합니다 - NO FALLBACK
        
        Args:
            code_content (str): 코드 내용
            file_path (str): 파일 경로
            
        Returns:
            Dict[str, Any]: 코드 품질 분석 결과
            
        Raises:
            ValueError: 필수 파라미터가 없거나 코드 품질 분석에 실패한 경우
        """
        # 필수 파라미터 검증
        if not code_content:
            raise ValueError("code_content가 비어있습니다.")
        if not file_path:
            raise ValueError("file_path가 비어있습니다.")
        
        # 파일 확장자 추출
        file_extension = file_path.split(".")[-1] if "." in file_path else ""
        
        logger.info(f"코드 품질 분석 시작: {file_path}")
        
        # MCP 클라이언트를 통해 코드 품질 분석 요청
        response = await self.mcp_client.call_tool(
            "analyze-code-quality",
            {
                "code_content": code_content,
                "file_path": file_path,
                "file_extension": file_extension
            }
        )
        
        # 응답 검증
        if not response:
            raise ValueError(f"코드 품질 분석 실패: {file_path} - MCP 서버 응답 없음")
        
        # 기본 구조 검증
        required_fields = ["quality_score", "issues"]
        for field in required_fields:
            if field not in response:
                logger.warning(f"코드 품질 분석 결과에 {field} 필드가 없습니다: {file_path}")
        
        # 품질 점수 검증
        quality_score = response.get("quality_score")
        if quality_score is not None and (not isinstance(quality_score, (int, float)) or quality_score < 0 or quality_score > 100):
            raise ValueError(f"코드 품질 분석 실패: {file_path} - 유효하지 않은 품질 점수: {quality_score}")
        
        # 이슈 목록 검증
        issues = response.get("issues", [])
        if not isinstance(issues, list):
            raise ValueError(f"코드 품질 분석 실패: {file_path} - 이슈 목록이 리스트 형태가 아닙니다.")
        
        logger.info(f"코드 품질 분석 완료: {file_path}, 품질 점수: {quality_score}, 이슈 수: {len(issues)}")
        return response
    
    async def generate_summary_review(self, pr_files: List[Dict[str, Any]], 
                                     pr_metadata: Dict[str, Any] = None) -> str:
        """
        PR 전체에 대한 요약 리뷰를 생성합니다 - NO FALLBACK
        
        Args:
            pr_files (List[Dict[str, Any]]): PR의 파일 변경사항 목록
            pr_metadata (Dict[str, Any], optional): PR 메타데이터 (제목, 설명 등)
            
        Returns:
            str: 생성된 요약 리뷰
            
        Raises:
            ValueError: 필수 파라미터가 없거나 요약 리뷰 생성에 실패한 경우
        """
        # 필수 파라미터 검증
        if not pr_files:
            raise ValueError("pr_files가 비어있습니다.")
        if not isinstance(pr_files, list):
            raise ValueError("pr_files가 리스트 형태가 아닙니다.")
        
        # PR 메타데이터 준비
        metadata = pr_metadata or {}
        pr_title = metadata.get("title", "")
        pr_description = metadata.get("description", "")
        
        # 파일 정보 요약
        files_summary = []
        for file in pr_files:
            if not isinstance(file, dict):
                raise ValueError("PR 파일 정보가 딕셔너리 형태가 아닙니다.")
            
            files_summary.append({
                "filename": file.get("filename", ""),
                "status": file.get("status", ""),
                "additions": file.get("additions", 0),
                "deletions": file.get("deletions", 0),
                "changes": file.get("changes", 0)
            })
        
        logger.info(f"요약 리뷰 생성 시작: 파일 수={len(files_summary)}")
        
        # MCP 클라이언트를 통해 요약 리뷰 생성 요청
        response = await self.mcp_client.call_tool(
            "generate-summary-review",
            {
                "files": files_summary,
                "pr_title": pr_title,
                "pr_description": pr_description
            }
        )
        
        # 응답 검증
        if not response:
            raise ValueError("요약 리뷰 생성 실패 - MCP 서버 응답 없음")
        
        if "summary" not in response:
            raise ValueError("요약 리뷰 생성 실패 - 유효한 요약 데이터 없음")
        
        summary = response["summary"]
        if not summary or not isinstance(summary, str) or len(summary.strip()) == 0:
            raise ValueError("요약 리뷰 생성 실패 - 요약 내용이 비어있거나 유효하지 않습니다.")
        
        logger.info("요약 리뷰 생성 완료")
        return summary 