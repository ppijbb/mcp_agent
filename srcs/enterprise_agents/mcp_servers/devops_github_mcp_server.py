import asyncio
import os
from typing import Any, Dict, Optional
import aiohttp

from mcp.server import Server
from mcp.server.models import (
    CallToolResult,
    ListToolsResult,
    Tool,
)
from mcp.types import (
    TextContent,
)


class GitHubMCPServer:
    """GitHub API를 위한 MCP Server"""

    def __init__(self):
        self.server = Server("devops-github-server")
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"
        self.session: Optional[aiohttp.ClientSession] = None
        self._setup_handlers()

    def _setup_handlers(self):
        """MCP 핸들러 설정"""

        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """사용 가능한 도구 목록 반환"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="list_repositories",
                        description="GitHub 저장소 목록 조회",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "org": {
                                    "type": "string",
                                    "description": "조직명 (선택사항)"
                                },
                                "type": {
                                    "type": "string",
                                    "description": "저장소 타입 (all, owner, member)",
                                    "default": "all"
                                }
                            }
                        }
                    ),
                    Tool(
                        name="get_pull_requests",
                        description="Pull Request 목록 조회",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "owner": {
                                    "type": "string",
                                    "description": "저장소 소유자"
                                },
                                "repo": {
                                    "type": "string",
                                    "description": "저장소 이름"
                                },
                                "state": {
                                    "type": "string",
                                    "description": "PR 상태 (open, closed, all)",
                                    "default": "open"
                                }
                            },
                            "required": ["owner", "repo"]
                        }
                    ),
                    Tool(
                        name="review_pull_request",
                        description="Pull Request 코드 리뷰 작성",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "owner": {
                                    "type": "string",
                                    "description": "저장소 소유자"
                                },
                                "repo": {
                                    "type": "string",
                                    "description": "저장소 이름"
                                },
                                "pull_number": {
                                    "type": "integer",
                                    "description": "PR 번호"
                                },
                                "event": {
                                    "type": "string",
                                    "description": "리뷰 이벤트 (APPROVE, REQUEST_CHANGES, COMMENT)",
                                    "default": "COMMENT"
                                },
                                "body": {
                                    "type": "string",
                                    "description": "리뷰 내용"
                                }
                            },
                            "required": ["owner", "repo", "pull_number", "body"]
                        }
                    ),
                    Tool(
                        name="create_issue",
                        description="GitHub 이슈 생성",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "owner": {
                                    "type": "string",
                                    "description": "저장소 소유자"
                                },
                                "repo": {
                                    "type": "string",
                                    "description": "저장소 이름"
                                },
                                "title": {
                                    "type": "string",
                                    "description": "이슈 제목"
                                },
                                "body": {
                                    "type": "string",
                                    "description": "이슈 내용"
                                },
                                "labels": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "라벨 목록"
                                },
                                "assignees": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "담당자 목록"
                                }
                            },
                            "required": ["owner", "repo", "title"]
                        }
                    ),
                    Tool(
                        name="check_workflow_runs",
                        description="GitHub Actions 워크플로우 실행 상태 확인",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "owner": {
                                    "type": "string",
                                    "description": "저장소 소유자"
                                },
                                "repo": {
                                    "type": "string",
                                    "description": "저장소 이름"
                                },
                                "status": {
                                    "type": "string",
                                    "description": "워크플로우 상태 (queued, in_progress, completed)",
                                    "default": "completed"
                                }
                            },
                            "required": ["owner", "repo"]
                        }
                    )
                ]
            )

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """도구 호출 처리"""
            try:
                if name == "list_repositories":
                    return await self._list_repositories(arguments)
                elif name == "get_pull_requests":
                    return await self._get_pull_requests(arguments)
                elif name == "review_pull_request":
                    return await self._review_pull_request(arguments)
                elif name == "create_issue":
                    return await self._create_issue(arguments)
                elif name == "check_workflow_runs":
                    return await self._check_workflow_runs(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")

            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")],
                    isError=True
                )

    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 생성 및 반환"""
        if self.session is None:
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "DevOps-Assistant-MCP-Server"
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """GitHub API 요청"""
        session = await self._get_session()
        full_url = f"{self.base_url}{url}"

        async with session.request(method, full_url, **kwargs) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise Exception(f"GitHub API error {response.status}: {error_text}")
            return await response.json()

    async def _list_repositories(self, args: Dict[str, Any]) -> CallToolResult:
        """저장소 목록 조회"""
        # Mock data for demo
        mock_repos = [
            {"name": "awesome-project", "full_name": "user/awesome-project", "description": "An awesome project", "language": "Python", "stargazers_count": 42, "forks_count": 7},
            {"name": "web-app", "full_name": "user/web-app", "description": "Modern web application", "language": "JavaScript", "stargazers_count": 23, "forks_count": 3}
        ]

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"조회된 저장소 ({len(mock_repos)}개):\n\n" +
                     "\n".join([f"• {repo['full_name']} - {repo['description']}" for repo in mock_repos])
            )]
        )

    async def _get_pull_requests(self, args: Dict[str, Any]) -> CallToolResult:
        """Pull Request 목록 조회"""
        owner = args["owner"]
        repo = args["repo"]

        # Mock data for demo
        mock_prs = [
            {"number": 42, "title": "Add new feature", "user": {"login": "dev1"}, "state": "open"},
            {"number": 41, "title": "Fix bug in authentication", "user": {"login": "dev2"}, "state": "open"}
        ]

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Pull Requests ({len(mock_prs)}개):\n\n" +
                     "\n".join([
                         f"• #{pr['number']} {pr['title']} by {pr['user']['login']} ({pr['state']})"
                         for pr in mock_prs
                     ])
            )]
        )

    async def _review_pull_request(self, args: Dict[str, Any]) -> CallToolResult:
        """Pull Request 리뷰 작성"""
        pull_number = args["pull_number"]
        event = args.get("event", "COMMENT")
        body = args["body"]

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"✅ PR #{pull_number}에 리뷰가 작성되었습니다.\n" +
                     f"이벤트: {event}\n" +
                     f"내용: {body[:100]}..."
            )]
        )

    async def _create_issue(self, args: Dict[str, Any]) -> CallToolResult:
        """GitHub 이슈 생성"""
        title = args["title"]
        body = args.get("body", "")

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"✅ 이슈가 생성되었습니다!\n" +
                     f"제목: {title}\n" +
                     f"내용: {body[:100]}..."
            )]
        )

    async def _check_workflow_runs(self, args: Dict[str, Any]) -> CallToolResult:
        """GitHub Actions 워크플로우 상태 확인"""
        owner = args["owner"]
        repo = args["repo"]

        # Mock data for demo
        mock_runs = [
            {"id": 1, "name": "CI/CD Pipeline", "status": "completed", "conclusion": "success", "head_branch": "main"},
            {"id": 2, "name": "Security Scan", "status": "completed", "conclusion": "failure", "head_branch": "feature/auth"}
        ]

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"워크플로우 실행 상태 ({len(mock_runs)}개):\n\n" +
                     "\n".join([
                         f"• {run['name']} ({run['head_branch']}) - {run['status']}/{run['conclusion']}"
                         for run in mock_runs
                     ])
            )]
        )


def main():
    """메인 실행 함수"""
    import logging
    logging.basicConfig(level=logging.INFO)

    server = GitHubMCPServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("Server stopped by user")


if __name__ == "__main__":
    main()
