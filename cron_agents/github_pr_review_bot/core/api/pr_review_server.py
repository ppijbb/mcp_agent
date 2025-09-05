"""
GitHub PR Review MCP Server - NO FALLBACK MODE

이 모듈은 GitHub PR 리뷰를 위한 MCP 서버를 구현합니다.
MCP 프로토콜을 통해 GitHub PR을 분석하고 코드 리뷰를 생성하는 도구를 제공합니다.
모든 오류는 fallback 없이 즉시 상위로 전파되거나 시스템을 종료시킵니다.
"""

import os
import logging
import asyncio
import sys
from typing import Dict, List, Any, Optional
from mcp.server import Server
from mcp.types import (
    Tool, 
    CallToolRequest, 
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent
)

from .github_client import GitHubClient
from .review_generator import ReviewGenerator
from .config import config

logger = logging.getLogger(__name__)

class GitHubPRReviewServer:
    """GitHub PR 리뷰를 위한 MCP 서버 - NO FALLBACK MODE"""
    
    def __init__(self, server_name: str = "github-pr-review-no-fallback"):
        """
        MCP 서버 초기화 - 실패 시 즉시 종료
        
        Args:
            server_name (str): 서버 이름
        """
        try:
            self.server = Server(server_name)
            self.github_client = GitHubClient()
            self.review_generator = ReviewGenerator()
            
            # 도구 등록
            self._register_tools()
            
            logger.info(f"GitHub PR Review MCP 서버가 초기화되었습니다: {server_name} (NO FALLBACK MODE)")
        except Exception as e:
            logger.error(f"서버 초기화 중 치명적 오류 발생: {e}")
            sys.exit(1)
    
    def _register_tools(self):
        """MCP 도구 등록 - 실패 시 즉시 종료"""
        
        @self.server.list_tools()
        async def handle_list_tools(request: ListToolsRequest) -> ListToolsResult:
            """사용 가능한 도구 목록 반환"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="review-pull-request",
                        description="GitHub PR의 코드를 리뷰합니다 (NO FALLBACK)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "repository": {
                                    "type": "string",
                                    "description": "GitHub 저장소 (owner/repo 형식)"
                                },
                                "pr_number": {
                                    "type": "integer",
                                    "description": "PR 번호"
                                },
                                "review_type": {
                                    "type": "string",
                                    "enum": ["summary", "detailed", "security", "performance"],
                                    "description": "리뷰 유형"
                                }
                            },
                            "required": ["repository", "pr_number"]
                        }
                    ),
                    Tool(
                        name="review-commit",
                        description="GitHub PR의 특정 커밋을 리뷰합니다 (NO FALLBACK)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "repository": {
                                    "type": "string",
                                    "description": "GitHub 저장소 (owner/repo 형식)"
                                },
                                "pr_number": {
                                    "type": "integer",
                                    "description": "PR 번호"
                                },
                                "commit_sha": {
                                    "type": "string",
                                    "description": "커밋 SHA"
                                }
                            },
                            "required": ["repository", "pr_number"]
                        }
                    ),
                    Tool(
                        name="analyze-code-quality",
                        description="코드 품질을 분석합니다 (NO FALLBACK)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "repository": {
                                    "type": "string",
                                    "description": "GitHub 저장소 (owner/repo 형식)"
                                },
                                "pr_number": {
                                    "type": "integer",
                                    "description": "PR 번호"
                                },
                                "file_path": {
                                    "type": "string",
                                    "description": "분석할 파일 경로"
                                }
                            },
                            "required": ["repository", "pr_number"]
                        }
                    ),
                    Tool(
                        name="submit-review",
                        description="GitHub PR에 리뷰를 등록합니다 (NO FALLBACK)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "repository": {
                                    "type": "string",
                                    "description": "GitHub 저장소 (owner/repo 형식)"
                                },
                                "pr_number": {
                                    "type": "integer",
                                    "description": "PR 번호"
                                },
                                "review_body": {
                                    "type": "string",
                                    "description": "리뷰 내용"
                                },
                                "event": {
                                    "type": "string",
                                    "enum": ["COMMENT", "APPROVE", "REQUEST_CHANGES"],
                                    "description": "리뷰 이벤트 타입"
                                },
                                "comments": {
                                    "type": "array",
                                    "description": "라인별 코멘트 목록",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "path": {"type": "string"},
                                            "position": {"type": "integer"},
                                            "body": {"type": "string"}
                                        }
                                    }
                                }
                            },
                            "required": ["repository", "pr_number", "review_body"]
                        }
                    )
                ]
            )
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """도구 호출 처리 - 오류 발생 시 즉시 종료"""
            
            try:
                if name == "review-pull-request":
                    return await self._review_pull_request(arguments)
                elif name == "review-commit":
                    return await self._review_commit(arguments)
                elif name == "analyze-code-quality":
                    return await self._analyze_code_quality(arguments)
                elif name == "submit-review":
                    return await self._submit_review(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"도구 호출 중 치명적 오류 발생: {name}, 오류: {e}")
                if config.github.fail_fast_on_error:
                    sys.exit(1)
                raise
    
    async def _review_pull_request(self, args: Dict[str, Any]) -> CallToolResult:
        """PR 리뷰 생성 - NO FALLBACK"""
        repository = args.get("repository")
        pr_number = args.get("pr_number")
        review_type = args.get("review_type", "detailed")
        
        # 필수 파라미터 검증
        if not repository:
            raise ValueError("repository 파라미터가 필요합니다.")
        if not pr_number:
            raise ValueError("pr_number 파라미터가 필요합니다.")
        
        logger.info(f"PR 리뷰 시작: {repository}#{pr_number}, 타입: {review_type}")
        
        # PR 정보 가져오기
        pr = self.github_client.get_pull_request(repository, pr_number)
        if not pr:
            raise ValueError(f"PR을 찾을 수 없습니다: {repository}#{pr_number}")
        
        # PR 메타데이터 준비
        pr_metadata = {
            "title": pr.title,
            "description": pr.body,
            "author": pr.user.login,
            "created_at": pr.created_at.isoformat(),
            "updated_at": pr.updated_at.isoformat()
        }
        
        # PR diff 가져오기
        diff_content = self.github_client.get_pr_diff(repository, pr_number)
        if not diff_content:
            raise ValueError("PR diff를 가져올 수 없습니다.")
        
        # 리뷰 생성
        review_result = await self.review_generator.generate_review(
            diff_content=diff_content,
            pr_metadata=pr_metadata
        )
        
        if not review_result:
            raise ValueError("리뷰 생성에 실패했습니다.")
        
        # 파일별 상세 리뷰 (detailed 모드인 경우)
        file_reviews = []
        if review_type == "detailed":
            pr_files = self.github_client.get_pr_files(repository, pr_number)
            
            for file in pr_files:
                if file.get("patch"):  # patch가 있는 경우만
                    file_review = await self.review_generator.generate_file_review(
                        file_patch=file["patch"],
                        file_path=file["filename"]
                    )
                    if file_review:
                        file_reviews.append({
                            "filename": file["filename"],
                            "comments": file_review
                        })
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"### PR 리뷰 결과: {repository}#{pr_number}\n\n"
                         f"**제목:** {pr.title}\n\n"
                         f"**요약:** {review_result.get('summary', '요약 없음')}\n\n"
                         f"**리뷰:** {review_result.get('review', '리뷰 없음')}"
                )
            ],
            json={
                "repository": repository,
                "pr_number": pr_number,
                "review": review_result,
                "file_reviews": file_reviews,
                "pr_metadata": pr_metadata
            }
        )
    
    async def _review_commit(self, args: Dict[str, Any]) -> CallToolResult:
        """특정 커밋 리뷰 생성 - NO FALLBACK"""
        repository = args.get("repository")
        pr_number = args.get("pr_number")
        
        # 필수 파라미터 검증
        if not repository:
            raise ValueError("repository 파라미터가 필요합니다.")
        if not pr_number:
            raise ValueError("pr_number 파라미터가 필요합니다.")
        
        # 최신 커밋 SHA 가져오기 (commit_sha가 없는 경우)
        if "commit_sha" not in args:
            latest_commit = self.github_client.get_latest_commit(repository, pr_number)
            if not latest_commit:
                raise ValueError("최신 커밋을 가져올 수 없습니다.")
            commit_sha = latest_commit["sha"]
        else:
            commit_sha = args["commit_sha"]
        
        # PR 정보 가져오기
        pr = self.github_client.get_pull_request(repository, pr_number)
        if not pr:
            raise ValueError(f"PR을 찾을 수 없습니다: {repository}#{pr_number}")
        
        # 커밋 diff 가져오기 (PR diff 활용)
        diff_content = self.github_client.get_pr_diff(repository, pr_number)
        if not diff_content:
            raise ValueError("커밋 diff를 가져올 수 없습니다.")
        
        # 리뷰 생성
        review_result = await self.review_generator.generate_review(
            diff_content=diff_content,
            pr_metadata={"title": pr.title, "description": pr.body}
        )
        
        if not review_result:
            raise ValueError("커밋 리뷰 생성에 실패했습니다.")
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"### 커밋 리뷰 결과: {repository}#{pr_number} (커밋: {commit_sha[:7]})\n\n"
                         f"**제목:** {pr.title}\n\n"
                         f"**리뷰:** {review_result.get('review', '리뷰 없음')}"
                )
            ],
            json={
                "repository": repository,
                "pr_number": pr_number,
                "commit_sha": commit_sha,
                "review": review_result
            }
        )
    
    async def _analyze_code_quality(self, args: Dict[str, Any]) -> CallToolResult:
        """코드 품질 분석 - NO FALLBACK"""
        repository = args.get("repository")
        pr_number = args.get("pr_number")
        file_path = args.get("file_path")
        
        # 필수 파라미터 검증
        if not repository:
            raise ValueError("repository 파라미터가 필요합니다.")
        if not pr_number:
            raise ValueError("pr_number 파라미터가 필요합니다.")
        
        # PR 파일 목록 가져오기
        pr_files = self.github_client.get_pr_files(repository, pr_number)
        if not pr_files:
            raise ValueError("PR 파일 목록을 가져올 수 없습니다.")
        
        # 특정 파일만 분석하거나 모든 파일 분석
        if file_path:
            files_to_analyze = [f for f in pr_files if f["filename"] == file_path]
            if not files_to_analyze:
                raise ValueError(f"파일을 찾을 수 없습니다: {file_path}")
        else:
            files_to_analyze = pr_files
        
        # 파일별 코드 품질 분석
        quality_results = []
        for file in files_to_analyze:
            if file.get("patch"):  # patch가 있는 경우만
                quality_result = await self.review_generator.analyze_code_quality(
                    code_content=file["patch"],
                    file_path=file["filename"]
                )
                if not quality_result:
                    raise ValueError(f"코드 품질 분석 실패: {file['filename']}")
                
                quality_results.append({
                    "filename": file["filename"],
                    "quality": quality_result
                })
        
        if not quality_results:
            raise ValueError("분석할 파일이 없습니다.")
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"### 코드 품질 분석 결과: {repository}#{pr_number}\n\n"
                         f"**분석된 파일 수:** {len(quality_results)}\n\n"
                         f"**주요 발견사항:** {self._summarize_quality_results(quality_results)}"
                )
            ],
            json={
                "repository": repository,
                "pr_number": pr_number,
                "quality_results": quality_results
            }
        )
    
    async def _submit_review(self, args: Dict[str, Any]) -> CallToolResult:
        """GitHub PR에 리뷰 등록 - NO FALLBACK"""
        repository = args.get("repository")
        pr_number = args.get("pr_number")
        review_body = args.get("review_body")
        event = args.get("event", "COMMENT")
        comments = args.get("comments", [])
        
        # 필수 파라미터 검증
        if not repository:
            raise ValueError("repository 파라미터가 필요합니다.")
        if not pr_number:
            raise ValueError("pr_number 파라미터가 필요합니다.")
        if not review_body:
            raise ValueError("review_body 파라미터가 필요합니다.")
        
        # GitHub API로 리뷰 등록
        review_result = self.github_client.create_review(
            repo_full_name=repository,
            pr_number=pr_number,
            body=review_body,
            event=event,
            comments=comments
        )
        
        if not review_result:
            raise ValueError("리뷰 등록에 실패했습니다.")
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"### 리뷰가 성공적으로 등록되었습니다: {repository}#{pr_number}\n\n"
                         f"**리뷰 ID:** {review_result['id']}\n"
                         f"**상태:** {review_result['state']}\n"
                         f"**URL:** {review_result['html_url']}"
                )
            ],
            json={
                "repository": repository,
                "pr_number": pr_number,
                "review_result": review_result
            }
        )
    
    def _summarize_quality_results(self, quality_results: List[Dict[str, Any]]) -> str:
        """코드 품질 결과 요약"""
        if not quality_results:
            return "분석 결과 없음"
        
        issues_count = 0
        major_issues = []
        
        for result in quality_results:
            quality = result.get("quality", {})
            issues = quality.get("issues", [])
            issues_count += len(issues)
            
            # 주요 이슈 추출 (심각도가 높은 것)
            for issue in issues:
                if issue.get("severity", "").lower() in ["high", "critical"]:
                    major_issues.append(f"{result['filename']}: {issue.get('message', '')}")
        
        summary = f"총 {issues_count}개의 이슈 발견"
        
        if major_issues:
            summary += "\n\n**주요 이슈:**\n"
            for issue in major_issues[:5]:  # 상위 5개만
                summary += f"- {issue}\n"
            
            if len(major_issues) > 5:
                summary += f"- 외 {len(major_issues) - 5}개 더..."
        
        return summary
    
    async def run(self, host: str = "0.0.0.0", port: int = 8000):
        """MCP 서버 실행 - 실패 시 즉시 종료"""
        try:
            logger.info(f"GitHub PR Review MCP 서버 시작 중... (NO FALLBACK MODE)")
            await self.server.run_http_server(host=host, port=port)
            logger.info(f"서버가 시작되었습니다: http://{host}:{port}")
        except Exception as e:
            logger.error(f"서버 실행 중 치명적 오류 발생: {e}")
            sys.exit(1)


async def main():
    """메인 함수 - NO FALLBACK MODE"""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        server = GitHubPRReviewServer()
        await server.run()
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"메인 함수에서 치명적 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 