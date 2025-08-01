"""
GitHub PR Review MCP Server

이 모듈은 GitHub PR 리뷰를 위한 MCP 서버를 구현합니다.
MCP 프로토콜을 통해 GitHub PR을 분석하고 코드 리뷰를 생성하는 도구를 제공합니다.
"""

import os
import logging
import asyncio
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

logger = logging.getLogger(__name__)

class GitHubPRReviewServer:
    """GitHub PR 리뷰를 위한 MCP 서버"""
    
    def __init__(self, server_name: str = "github-pr-review"):
        """
        MCP 서버 초기화
        
        Args:
            server_name (str): 서버 이름
        """
        self.server = Server(server_name)
        self.github_client = GitHubClient()
        self.review_generator = ReviewGenerator()
        
        # 도구 등록
        self._register_tools()
        
        logger.info(f"GitHub PR Review MCP 서버가 초기화되었습니다: {server_name}")
    
    def _register_tools(self):
        """MCP 도구 등록"""
        
        @self.server.list_tools()
        async def handle_list_tools(request: ListToolsRequest) -> ListToolsResult:
            """사용 가능한 도구 목록 반환"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="review-pull-request",
                        description="GitHub PR의 코드를 리뷰합니다",
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
                        description="GitHub PR의 특정 커밋을 리뷰합니다",
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
                        description="코드 품질을 분석합니다",
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
                        description="GitHub PR에 리뷰를 등록합니다",
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
                    ),
                    Tool(
                        name="monitor-pr-commits",
                        description="PR에 새 커밋이 추가되면 자동으로 리뷰합니다",
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
                                "interval": {
                                    "type": "integer",
                                    "description": "확인 간격 (초)"
                                }
                            },
                            "required": ["repository", "pr_number"]
                        }
                    )
                ]
            )
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """도구 호출 처리"""
            
            if name == "review-pull-request":
                return await self._review_pull_request(arguments)
            elif name == "review-commit":
                return await self._review_commit(arguments)
            elif name == "analyze-code-quality":
                return await self._analyze_code_quality(arguments)
            elif name == "submit-review":
                return await self._submit_review(arguments)
            elif name == "monitor-pr-commits":
                return await self._monitor_pr_commits(arguments)
            else:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Unknown tool: {name}"
                        )
                    ]
                )
    
    async def _review_pull_request(self, args: Dict[str, Any]) -> CallToolResult:
        """PR 리뷰 생성"""
        try:
            repository = args["repository"]
            pr_number = args["pr_number"]
            review_type = args.get("review_type", "detailed")
            
            # PR 정보 가져오기
            pr = self.github_client.get_pull_request(repository, pr_number)
            
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
            
            # 리뷰 생성
            review_result = await self.review_generator.generate_review(
                diff_content=diff_content,
                pr_metadata=pr_metadata
            )
            
            # 파일별 상세 리뷰 (detailed 모드인 경우)
            file_reviews = []
            if review_type == "detailed":
                pr_files = self.github_client.get_pr_files(repository, pr_number)
                
                for file in pr_files:
                    if file["patch"]:  # patch가 있는 경우만
                        file_review = await self.review_generator.generate_file_review(
                            file_patch=file["patch"],
                            file_path=file["filename"]
                        )
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
        except Exception as e:
            logger.error(f"PR 리뷰 생성 중 오류 발생: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"PR 리뷰 생성 중 오류 발생: {str(e)}"
                    )
                ]
            )
    
    async def _review_commit(self, args: Dict[str, Any]) -> CallToolResult:
        """특정 커밋 리뷰 생성"""
        try:
            repository = args["repository"]
            pr_number = args["pr_number"]
            
            # 최신 커밋 SHA 가져오기 (commit_sha가 없는 경우)
            if "commit_sha" not in args:
                latest_commit = self.github_client.get_latest_commit(repository, pr_number)
                commit_sha = latest_commit["sha"]
            else:
                commit_sha = args["commit_sha"]
            
            # PR 정보 가져오기
            pr = self.github_client.get_pull_request(repository, pr_number)
            
            # 커밋 diff 가져오기 (PR diff 활용)
            diff_content = self.github_client.get_pr_diff(repository, pr_number)
            
            # 리뷰 생성
            review_result = await self.review_generator.generate_review(
                diff_content=diff_content,
                pr_metadata={"title": pr.title, "description": pr.body}
            )
            
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
        except Exception as e:
            logger.error(f"커밋 리뷰 생성 중 오류 발생: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"커밋 리뷰 생성 중 오류 발생: {str(e)}"
                    )
                ]
            )
    
    async def _analyze_code_quality(self, args: Dict[str, Any]) -> CallToolResult:
        """코드 품질 분석"""
        try:
            repository = args["repository"]
            pr_number = args["pr_number"]
            file_path = args.get("file_path")
            
            # PR 파일 목록 가져오기
            pr_files = self.github_client.get_pr_files(repository, pr_number)
            
            # 특정 파일만 분석하거나 모든 파일 분석
            if file_path:
                files_to_analyze = [f for f in pr_files if f["filename"] == file_path]
            else:
                files_to_analyze = pr_files
            
            # 파일별 코드 품질 분석
            quality_results = []
            for file in files_to_analyze:
                if file["patch"]:  # patch가 있는 경우만
                    quality_result = await self.review_generator.analyze_code_quality(
                        code_content=file["patch"],
                        file_path=file["filename"]
                    )
                    quality_results.append({
                        "filename": file["filename"],
                        "quality": quality_result
                    })
            
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
        except Exception as e:
            logger.error(f"코드 품질 분석 중 오류 발생: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"코드 품질 분석 중 오류 발생: {str(e)}"
                    )
                ]
            )
    
    async def _submit_review(self, args: Dict[str, Any]) -> CallToolResult:
        """GitHub PR에 리뷰 등록"""
        try:
            repository = args["repository"]
            pr_number = args["pr_number"]
            review_body = args["review_body"]
            event = args.get("event", "COMMENT")
            comments = args.get("comments", [])
            
            # GitHub API로 리뷰 등록
            review_result = self.github_client.create_review(
                repo_full_name=repository,
                pr_number=pr_number,
                body=review_body,
                event=event,
                comments=comments
            )
            
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
        except Exception as e:
            logger.error(f"리뷰 등록 중 오류 발생: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"리뷰 등록 중 오류 발생: {str(e)}"
                    )
                ]
            )
    
    async def _monitor_pr_commits(self, args: Dict[str, Any]) -> CallToolResult:
        """PR에 새 커밋이 추가되면 자동으로 리뷰"""
        try:
            repository = args["repository"]
            pr_number = args["pr_number"]
            interval = args.get("interval", 60)  # 기본 60초
            
            # 현재 커밋 정보 가져오기
            latest_commit = self.github_client.get_latest_commit(repository, pr_number)
            last_reviewed_sha = latest_commit["sha"]
            
            # 비동기 모니터링 작업 시작
            asyncio.create_task(
                self._monitor_commits_task(repository, pr_number, last_reviewed_sha, interval)
            )
            
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"### PR 커밋 모니터링 시작: {repository}#{pr_number}\n\n"
                             f"**확인 간격:** {interval}초\n"
                             f"**마지막 검토 커밋:** {last_reviewed_sha[:7]}"
                    )
                ],
                json={
                    "repository": repository,
                    "pr_number": pr_number,
                    "interval": interval,
                    "last_reviewed_sha": last_reviewed_sha
                }
            )
        except Exception as e:
            logger.error(f"PR 모니터링 시작 중 오류 발생: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"PR 모니터링 시작 중 오류 발생: {str(e)}"
                    )
                ]
            )
    
    async def _monitor_commits_task(self, repository: str, pr_number: int, 
                                   last_reviewed_sha: str, interval: int):
        """PR 커밋 모니터링 작업"""
        logger.info(f"PR 모니터링 시작: {repository}#{pr_number}, 간격: {interval}초")
        
        while True:
            try:
                # 최신 커밋 정보 가져오기
                latest_commit = self.github_client.get_latest_commit(repository, pr_number)
                current_sha = latest_commit["sha"]
                
                # 새 커밋이 있으면 리뷰 생성
                if current_sha != last_reviewed_sha:
                    logger.info(f"새 커밋 감지: {repository}#{pr_number}, SHA: {current_sha[:7]}")
                    
                    # 리뷰 생성
                    review_args = {
                        "repository": repository,
                        "pr_number": pr_number,
                        "commit_sha": current_sha
                    }
                    review_result = await self._review_commit(review_args)
                    
                    # GitHub에 리뷰 등록
                    if "json" in review_result and "review" in review_result.json:
                        review = review_result.json["review"]
                        submit_args = {
                            "repository": repository,
                            "pr_number": pr_number,
                            "review_body": review.get("review", "자동 생성된 코드 리뷰"),
                            "event": "COMMENT"
                        }
                        await self._submit_review(submit_args)
                    
                    # 검토한 커밋 SHA 업데이트
                    last_reviewed_sha = current_sha
                
                # 일정 시간 대기
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"PR 모니터링 중 오류 발생: {e}")
                await asyncio.sleep(interval)
    
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
        """MCP 서버 실행"""
        await self.server.run_http_server(host=host, port=port)
        logger.info(f"GitHub PR Review MCP 서버가 시작되었습니다: http://{host}:{port}")


async def main():
    """메인 함수"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    server = GitHubPRReviewServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main()) 