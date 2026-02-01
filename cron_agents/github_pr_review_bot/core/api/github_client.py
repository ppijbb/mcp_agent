"""
GitHub Client - GitHub API와 통신하는 클라이언트 (NO FALLBACK MODE)

이 모듈은 GitHub API와 통신하여 PR 정보를 가져오고, 리뷰를 등록하는 기능을 제공합니다.
모든 오류는 fallback 없이 즉시 상위로 전파됩니다.
"""

import logging
import sys
import re
from typing import Dict, List, Any
from github import Github
from github.PullRequest import PullRequest
from github.Repository import Repository
from github.GithubException import GithubException

from .config import config

logger = logging.getLogger(__name__)


class GitHubClient:
    """GitHub API와 통신하는 클라이언트 클래스 - NO FALLBACK MODE"""

    def __init__(self, token: str = None):
        """
        GitHub 클라이언트 초기화 - 실패 시 즉시 종료

        Args:
            token (str, optional): GitHub API 토큰. 기본값은 환경 변수 GITHUB_TOKEN.

        Raises:
            ValueError: GitHub 토큰이 없거나 유효하지 않은 경우
        """
        try:
            self.token = token or config.github.token
            if not self.token:
                raise ValueError("GitHub 토큰이 필요합니다. 환경 변수 GITHUB_TOKEN을 설정하거나 직접 제공해주세요.")

            # GitHub 클라이언트 초기화 (재시도 없음)
            self.client = Github(self.token)
            if not self.client:
                raise ValueError("GitHub 클라이언트 초기화에 실패했습니다.")

            # 토큰 유효성 검증 (즉시 실패)
            user = self.client.get_user()
            if not user:
                raise ValueError("GitHub 토큰이 유효하지 않습니다.")

            # 사용자 권한 검증
            if not user.login:
                raise ValueError("GitHub 사용자 정보를 가져올 수 없습니다.")

            # API 레이트 리미트 확인
            rate_limit = self.client.get_rate_limit()
            if rate_limit.core.remaining <= 0:
                raise ValueError(f"GitHub API 레이트 리미트 초과. 리셋 시간: {rate_limit.core.reset}")

            logger.info(f"GitHub 클라이언트가 초기화되었습니다. 사용자: {user.login}, 남은 API 호출: {rate_limit.core.remaining} (NO FALLBACK MODE)")

        except GithubException as e:
            if e.status == 401:
                raise ValueError("GitHub 토큰이 유효하지 않습니다.")
            elif e.status == 403:
                raise ValueError(f"GitHub API 접근 권한이 없습니다: {e}")
            elif e.status == 429:
                raise ValueError(f"GitHub API 레이트 리미트 초과: {e}")
            else:
                raise ValueError(f"GitHub API 연결 실패: {e}")
        except Exception as e:
            logger.error(f"GitHub 클라이언트 초기화 중 치명적 오류 발생: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise

    def get_repository(self, repo_full_name: str) -> Repository:
        """
        저장소 객체를 가져옵니다 - NO FALLBACK

        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름

        Returns:
            Repository: GitHub 저장소 객체

        Raises:
            ValueError: 저장소 이름이 유효하지 않거나 저장소를 찾을 수 없는 경우
        """
        if not repo_full_name:
            raise ValueError("저장소 이름이 필요합니다.")

        if "/" not in repo_full_name:
            raise ValueError("저장소 이름은 'owner/repo' 형식이어야 합니다.")

        # 저장소 이름 형식 검증
        parts = repo_full_name.split("/")
        if len(parts) != 2:
            raise ValueError("저장소 이름은 'owner/repo' 형식이어야 합니다.")

        owner, repo_name = parts
        if not owner or not repo_name:
            raise ValueError("저장소 이름의 owner와 repo 부분이 모두 필요합니다.")

        # API 레이트 리미트 재확인
        rate_limit = self.client.get_rate_limit()
        if rate_limit.core.remaining <= 0:
            raise ValueError(f"GitHub API 레이트 리미트 초과. 리셋 시간: {rate_limit.core.reset}")

        try:
            repo = self.client.get_repo(repo_full_name)
            if not repo:
                raise ValueError(f"저장소를 찾을 수 없습니다: {repo_full_name}")

            # 저장소 접근 권한 확인
            if repo.private and not hasattr(repo, 'permissions'):
                raise ValueError(f"비공개 저장소에 접근할 권한이 없습니다: {repo_full_name}")

            logger.info(f"저장소 연결 성공: {repo_full_name}, 남은 API 호출: {rate_limit.core.remaining}")
            return repo

        except GithubException as e:
            if e.status == 404:
                raise ValueError(f"저장소를 찾을 수 없습니다: {repo_full_name}")
            elif e.status == 403:
                raise ValueError(f"저장소에 접근할 권한이 없습니다: {repo_full_name}")
            elif e.status == 429:
                raise ValueError(f"GitHub API 레이트 리미트 초과: {e}")
            else:
                raise ValueError(f"저장소를 가져오는 중 오류 발생: {e}")

    def get_pull_request(self, repo_full_name: str, pr_number: int) -> PullRequest:
        """
        PR 객체를 가져옵니다 - NO FALLBACK

        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호

        Returns:
            PullRequest: GitHub PR 객체

        Raises:
            ValueError: PR을 찾을 수 없거나 파라미터가 유효하지 않은 경우
        """
        if not repo_full_name:
            raise ValueError("저장소 이름이 필요합니다.")
        if not pr_number or pr_number <= 0:
            raise ValueError("유효한 PR 번호가 필요합니다.")

        try:
            repo = self.get_repository(repo_full_name)
            pr = repo.get_pull(pr_number)
            if not pr:
                raise ValueError(f"PR을 찾을 수 없습니다: {repo_full_name}#{pr_number}")
            return pr
        except GithubException as e:
            if e.status == 404:
                raise ValueError(f"PR을 찾을 수 없습니다: {repo_full_name}#{pr_number}")
            else:
                raise ValueError(f"PR을 가져오는 중 오류 발생: {e}")

    def get_pr_diff(self, repo_full_name: str, pr_number: int) -> str:
        """
        PR의 diff 내용을 가져옵니다 - NO FALLBACK

        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호

        Returns:
            str: PR diff 내용

        Raises:
            ValueError: diff를 가져올 수 없는 경우
        """
        pr = self.get_pull_request(repo_full_name, pr_number)

        try:
            diff_content = pr.get_patch()
            if not diff_content:
                raise ValueError(f"PR diff가 비어있습니다: {repo_full_name}#{pr_number}")
            return diff_content
        except Exception as e:
            raise ValueError(f"PR diff를 가져오는 중 오류 발생: {e}")

    def get_pr_files(self, repo_full_name: str, pr_number: int) -> List[Dict[str, Any]]:
        """
        PR에서 변경된 파일 목록을 가져옵니다 - NO FALLBACK

        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호

        Returns:
            List[Dict[str, Any]]: 변경된 파일 정보 목록

        Raises:
            ValueError: 파일 목록을 가져올 수 없는 경우
        """
        pr = self.get_pull_request(repo_full_name, pr_number)

        try:
            files = pr.get_files()
            if files is None:
                raise ValueError(f"PR 파일 목록을 가져올 수 없습니다: {repo_full_name}#{pr_number}")

            result = []
            for file in files:
                if not file:
                    continue

                file_info = {
                    "filename": file.filename or "",
                    "status": file.status or "",
                    "additions": file.additions or 0,
                    "deletions": file.deletions or 0,
                    "changes": file.changes or 0,
                    "patch": file.patch or "",
                    "previous_filename": file.previous_filename or "",
                    "blob_url": file.blob_url or "",
                    "raw_url": file.raw_url or "",
                    "contents_url": file.contents_url or "",
                    "sha": file.sha or "",
                    "previous_sha": getattr(file, 'previous_sha', None) or ""
                }

                # 필수 필드 검증
                if not file_info["filename"]:
                    logger.warning(f"파일명이 없는 파일이 발견되었습니다: {repo_full_name}#{pr_number}")
                    continue

                # 파일 변경 유형 분석
                file_info["change_type"] = self._analyze_file_change_type(file_info)
                file_info["impact_level"] = self._analyze_file_impact_level(file_info)

                result.append(file_info)

            if not result:
                raise ValueError(f"변경된 파일이 없습니다: {repo_full_name}#{pr_number}")

            logger.info(f"PR 파일 목록 가져오기 완료: {repo_full_name}#{pr_number}, 파일 수: {len(result)}")
            return result
        except Exception as e:
            raise ValueError(f"PR 파일 목록을 가져오는 중 오류 발생: {e}")

    def _analyze_file_change_type(self, file_info: Dict[str, Any]) -> str:
        """파일 변경 유형 분석"""
        status = file_info.get("status", "").lower()
        additions = file_info.get("additions", 0)
        deletions = file_info.get("deletions", 0)

        if status == "added":
            return "new_file"
        elif status == "removed":
            return "deleted_file"
        elif status == "renamed":
            return "renamed_file"
        elif status == "modified":
            if additions > deletions * 2:
                return "major_addition"
            elif deletions > additions * 2:
                return "major_deletion"
            else:
                return "modified_file"
        else:
            return "unknown_change"

    def _analyze_file_impact_level(self, file_info: Dict[str, Any]) -> str:
        """파일 변경 영향도 분석"""
        changes = file_info.get("changes", 0)
        filename = file_info.get("filename", "").lower()

        # 중요 파일 패턴
        critical_patterns = [
            "config", "settings", "env", "dockerfile", "docker-compose",
            "package.json", "requirements.txt", "pom.xml", "build.gradle",
            "main.py", "app.py", "index.js", "index.ts", "main.go"
        ]

        # 테스트 파일 패턴
        test_patterns = ["test", "spec", "__test__", ".test.", ".spec."]

        # 문서 파일 패턴
        doc_patterns = [".md", ".txt", ".rst", "readme", "changelog"]

        # 중요 파일인지 확인
        is_critical = any(pattern in filename for pattern in critical_patterns)
        is_test = any(pattern in filename for pattern in test_patterns)
        is_doc = any(pattern in filename for pattern in doc_patterns)

        if is_critical:
            return "critical"
        elif changes > 100:
            return "high"
        elif changes > 50:
            return "medium"
        elif is_test:
            return "test_impact"
        elif is_doc:
            return "documentation"
        else:
            return "low"

    def create_review_comment(self, repo_full_name: str, pr_number: int,
                              body: str, commit_id: str, path: str,
                              position: int) -> Dict[str, Any]:
        """
        PR의 특정 라인에 리뷰 코멘트를 남깁니다 - NO FALLBACK

        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호
            body (str): 코멘트 내용
            commit_id (str): 커밋 ID
            path (str): 파일 경로
            position (int): 파일에서의 위치 (라인 번호)

        Returns:
            Dict[str, Any]: 생성된 코멘트 정보

        Raises:
            ValueError: 필수 파라미터가 없거나 코멘트 생성에 실패한 경우
        """
        # 필수 파라미터 검증
        if not body:
            raise ValueError("코멘트 내용이 필요합니다.")
        if not commit_id:
            raise ValueError("커밋 ID가 필요합니다.")
        if not path:
            raise ValueError("파일 경로가 필요합니다.")
        if position <= 0:
            raise ValueError("유효한 위치 정보가 필요합니다.")

        try:
            pr = self.get_pull_request(repo_full_name, pr_number)
            comment = pr.create_review_comment(body, commit_id, path, position)

            if not comment:
                raise ValueError("리뷰 코멘트 생성에 실패했습니다.")

            return {
                "id": comment.id,
                "body": comment.body,
                "path": comment.path,
                "position": comment.position,
                "html_url": comment.html_url
            }
        except GithubException as e:
            if e.status == 422:
                raise ValueError(f"리뷰 코멘트 생성 실패: 잘못된 위치 또는 파일 경로 - {e}")
            else:
                raise ValueError(f"리뷰 코멘트를 생성하는 중 오류 발생: {e}")

    def create_review(self, repo_full_name: str, pr_number: int,
                     body: str, event: str = "COMMENT",
                     comments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        PR에 전체 리뷰를 등록합니다 - NO FALLBACK

        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호
            body (str): 리뷰 내용
            event (str): 리뷰 이벤트 타입 (COMMENT, APPROVE, REQUEST_CHANGES)
            comments (List[Dict[str, Any]], optional): 라인별 코멘트 목록

        Returns:
            Dict[str, Any]: 생성된 리뷰 정보

        Raises:
            ValueError: 필수 파라미터가 없거나 리뷰 등록에 실패한 경우
        """
        # 필수 파라미터 검증
        if not body:
            raise ValueError("리뷰 내용이 필요합니다.")

        valid_events = ["COMMENT", "APPROVE", "REQUEST_CHANGES"]
        if event not in valid_events:
            raise ValueError(f"유효하지 않은 이벤트 타입: {event}. 유효한 값: {valid_events}")

        try:
            pr = self.get_pull_request(repo_full_name, pr_number)

            # 라인별 코멘트가 있는 경우
            if comments:
                review_comments = []
                for i, comment in enumerate(comments):
                    if not isinstance(comment, dict):
                        raise ValueError(f"코멘트 {i}가 딕셔너리 형태가 아닙니다.")

                    required_fields = ["path", "position", "body"]
                    for field in required_fields:
                        if field not in comment:
                            raise ValueError(f"코멘트 {i}에 {field} 필드가 없습니다.")

                    review_comments.append({
                        "path": comment["path"],
                        "position": comment["position"],
                        "body": comment["body"]
                    })

                review = pr.create_review(
                    body=body,
                    event=event,
                    comments=review_comments
                )
            else:
                # 전체 리뷰만 있는 경우
                review = pr.create_review(
                    body=body,
                    event=event
                )

            if not review:
                raise ValueError("리뷰 등록에 실패했습니다.")

            return {
                "id": review.id,
                "body": review.body,
                "state": review.state,
                "html_url": review.html_url
            }
        except GithubException as e:
            if e.status == 422:
                raise ValueError(f"리뷰 등록 실패: 잘못된 파라미터 또는 중복 리뷰 - {e}")
            else:
                raise ValueError(f"리뷰를 생성하는 중 오류 발생: {e}")

    def get_commit_details(self, repo_full_name: str, pr_number: int) -> List[Dict[str, Any]]:
        """
        PR의 커밋 정보를 가져옵니다 - NO FALLBACK

        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호

        Returns:
            List[Dict[str, Any]]: 커밋 정보 목록

        Raises:
            ValueError: 커밋 정보를 가져올 수 없는 경우
        """
        pr = self.get_pull_request(repo_full_name, pr_number)

        try:
            commits = pr.get_commits()
            if commits is None:
                raise ValueError(f"커밋 목록을 가져올 수 없습니다: {repo_full_name}#{pr_number}")

            result = []
            for commit in commits:
                if not commit:
                    continue

                commit_info = {
                    "sha": commit.sha or "",
                    "message": commit.commit.message if commit.commit else "",
                    "author": commit.commit.author.name if commit.commit and commit.commit.author else "",
                    "date": commit.commit.author.date.isoformat() if commit.commit and commit.commit.author and commit.commit.author.date else ""
                }

                if not commit_info["sha"]:
                    logger.warning(f"SHA가 없는 커밋이 발견되었습니다: {repo_full_name}#{pr_number}")
                    continue

                result.append(commit_info)

            if not result:
                raise ValueError(f"유효한 커밋이 없습니다: {repo_full_name}#{pr_number}")

            logger.info(f"커밋 정보 가져오기 완료: {repo_full_name}#{pr_number}, 커밋 수: {len(result)}")
            return result
        except Exception as e:
            raise ValueError(f"커밋 정보를 가져오는 중 오류 발생: {e}")

    def get_latest_commit(self, repo_full_name: str, pr_number: int) -> Dict[str, Any]:
        """
        PR의 최신 커밋 정보를 가져옵니다 - NO FALLBACK

        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호

        Returns:
            Dict[str, Any]: 최신 커밋 정보

        Raises:
            ValueError: 최신 커밋을 가져올 수 없는 경우
        """
        commits = self.get_commit_details(repo_full_name, pr_number)
        if not commits:
            raise ValueError(f"PR에 커밋이 없습니다: {repo_full_name}#{pr_number}")

        # 가장 최근 커밋 반환
        latest_commit = commits[-1]
        logger.info(f"최신 커밋 정보 가져오기 완료: {repo_full_name}#{pr_number}, SHA: {latest_commit['sha'][:7]}")
        return latest_commit

    def get_detailed_changes(self, repo_full_name: str, pr_number: int) -> Dict[str, Any]:
        """
        PR의 상세한 변경사항을 분석합니다 - NO FALLBACK

        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호

        Returns:
            Dict[str, Any]: 상세한 변경사항 분석 결과

        Raises:
            ValueError: 변경사항을 분석할 수 없는 경우
        """
        try:
            pr = self.get_pull_request(repo_full_name, pr_number)
            files = self.get_pr_files(repo_full_name, pr_number)
            diff_content = self.get_pr_diff(repo_full_name, pr_number)
            commits = self.get_commit_details(repo_full_name, pr_number)

            # 변경사항 분석
            change_analysis = {
                "summary": {
                    "total_files": len(files),
                    "total_additions": sum(f.get("additions", 0) for f in files),
                    "total_deletions": sum(f.get("deletions", 0) for f in files),
                    "total_changes": sum(f.get("changes", 0) for f in files),
                    "commits_count": len(commits)
                },
                "files": files,
                "commits": commits,
                "change_categories": self._categorize_changes(files),
                "impact_analysis": self._analyze_change_impact(files, diff_content),
                "semantic_changes": self._analyze_semantic_changes(files, diff_content)
            }

            logger.info(f"상세 변경사항 분석 완료: {repo_full_name}#{pr_number}")
            return change_analysis

        except Exception as e:
            raise ValueError(f"상세 변경사항 분석 중 오류 발생: {e}")

    def _categorize_changes(self, files: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """변경사항을 카테고리별로 분류"""
        categories = {
            "new_files": [],
            "deleted_files": [],
            "modified_files": [],
            "renamed_files": [],
            "critical_files": [],
            "test_files": [],
            "documentation_files": []
        }

        for file_info in files:
            change_type = file_info.get("change_type", "")
            impact_level = file_info.get("impact_level", "")
            filename = file_info.get("filename", "").lower()

            if change_type == "new_file":
                categories["new_files"].append(file_info)
            elif change_type == "deleted_file":
                categories["deleted_files"].append(file_info)
            elif change_type == "renamed_file":
                categories["renamed_files"].append(file_info)
            elif change_type in ["modified_file", "major_addition", "major_deletion"]:
                categories["modified_files"].append(file_info)

            if impact_level == "critical":
                categories["critical_files"].append(file_info)
            elif impact_level == "test_impact":
                categories["test_files"].append(file_info)
            elif impact_level == "documentation":
                categories["documentation_files"].append(file_info)

        return categories

    def _analyze_change_impact(self, files: List[Dict[str, Any]], diff_content: str) -> Dict[str, Any]:
        """변경사항의 영향도 분석"""
        impact_analysis = {
            "high_impact_files": [],
            "breaking_changes": [],
            "api_changes": [],
            "configuration_changes": [],
            "dependency_changes": []
        }

        for file_info in files:
            filename = file_info.get("filename", "").lower()
            patch = file_info.get("patch", "")

            # API 변경 감지
            if any(pattern in filename for pattern in ["api", "endpoint", "route", "controller"]):
                if "def " in patch or "function " in patch or "export " in patch:
                    impact_analysis["api_changes"].append({
                        "file": file_info["filename"],
                        "type": "api_modification",
                        "changes": file_info["changes"]
                    })

            # 설정 파일 변경 감지
            if any(pattern in filename for pattern in ["config", "settings", "env", "yaml", "json", "xml"]):
                impact_analysis["configuration_changes"].append({
                    "file": file_info["filename"],
                    "type": "config_modification",
                    "changes": file_info["changes"]
                })

            # 의존성 변경 감지
            if any(pattern in filename for pattern in ["package.json", "requirements.txt", "pom.xml", "build.gradle", "cargo.toml"]):
                impact_analysis["dependency_changes"].append({
                    "file": file_info["filename"],
                    "type": "dependency_modification",
                    "changes": file_info["changes"]
                })

            # Breaking change 감지 (간단한 패턴)
            if any(keyword in patch.lower() for keyword in ["breaking", "deprecated", "removed", "deleted"]):
                impact_analysis["breaking_changes"].append({
                    "file": file_info["filename"],
                    "type": "potential_breaking_change",
                    "changes": file_info["changes"]
                })

            # 높은 영향도 파일
            if file_info.get("impact_level") in ["critical", "high"]:
                impact_analysis["high_impact_files"].append({
                    "file": file_info["filename"],
                    "impact_level": file_info["impact_level"],
                    "changes": file_info["changes"]
                })

        return impact_analysis

    def _analyze_semantic_changes(self, files: List[Dict[str, Any]], diff_content: str) -> Dict[str, Any]:
        """의미적 변경사항 분석"""
        semantic_analysis = {
            "feature_additions": [],
            "bug_fixes": [],
            "refactoring": [],
            "performance_improvements": [],
            "security_updates": []
        }

        # 커밋 메시지에서 의미적 변경 감지
        commit_patterns = {
            "feature": ["feat", "feature", "add", "new", "implement"],
            "bugfix": ["fix", "bug", "issue", "resolve", "correct"],
            "refactor": ["refactor", "clean", "restructure", "reorganize"],
            "performance": ["perf", "optimize", "speed", "performance", "faster"],
            "security": ["security", "secure", "vulnerability", "patch", "safe"]
        }

        # 파일 패치에서 의미적 변경 감지
        patch_patterns = {
            "feature_additions": [r"\+.*def\s+\w+", r"\+.*function\s+\w+", r"\+.*class\s+\w+"],
            "bug_fixes": [r"\+.*fix", r"\+.*bug", r"\+.*issue", r"\+.*error"],
            "refactoring": [r"\+.*refactor", r"\+.*clean", r"\+.*rename"],
            "performance_improvements": [r"\+.*optimize", r"\+.*cache", r"\+.*performance"],
            "security_updates": [r"\+.*security", r"\+.*secure", r"\+.*auth", r"\+.*encrypt"]
        }

        for file_info in files:
            filename = file_info["filename"]
            patch = file_info.get("patch", "")

            # 패치 패턴 매칭
            for category, patterns in patch_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, patch, re.IGNORECASE):
                        semantic_analysis[category].append({
                            "file": filename,
                            "pattern_matched": pattern,
                            "changes": file_info["changes"]
                        })

        return semantic_analysis

    def get_line_by_line_changes(self, repo_full_name: str, pr_number: int) -> Dict[str, Any]:
        """
        라인별 상세 변경사항 분석 - NO FALLBACK

        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호

        Returns:
            Dict[str, Any]: 라인별 변경사항 분석 결과

        Raises:
            ValueError: 라인별 변경사항을 분석할 수 없는 경우
        """
        try:
            files = self.get_pr_files(repo_full_name, pr_number)
            line_changes = {
                "file_changes": [],
                "summary": {
                    "total_files_analyzed": len(files),
                    "total_lines_added": 0,
                    "total_lines_removed": 0,
                    "total_lines_modified": 0
                }
            }

            for file_info in files:
                patch = file_info.get("patch", "")
                if not patch:
                    continue

                file_analysis = self._analyze_file_patch(file_info)
                line_changes["file_changes"].append(file_analysis)

                # 통계 업데이트
                line_changes["summary"]["total_lines_added"] += file_analysis["lines_added"]
                line_changes["summary"]["total_lines_removed"] += file_analysis["lines_removed"]
                line_changes["summary"]["total_lines_modified"] += file_analysis["lines_modified"]

            logger.info(f"라인별 변경사항 분석 완료: {repo_full_name}#{pr_number}")
            return line_changes

        except Exception as e:
            raise ValueError(f"라인별 변경사항 분석 중 오류 발생: {e}")

    def _analyze_file_patch(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """개별 파일의 패치 분석"""
        patch = file_info.get("patch", "")
        filename = file_info.get("filename", "")

        analysis = {
            "filename": filename,
            "lines_added": 0,
            "lines_removed": 0,
            "lines_modified": 0,
            "hunks": [],
            "critical_changes": [],
            "function_changes": [],
            "import_changes": []
        }

        if not patch:
            return analysis

        lines = patch.split('\n')
        current_hunk = None
        line_number = 0

        for line in lines:
            if line.startswith('@@'):
                # 새로운 hunk 시작
                if current_hunk:
                    analysis["hunks"].append(current_hunk)

                # hunk 헤더 파싱
                hunk_info = self._parse_hunk_header(line)
                current_hunk = {
                    "old_start": hunk_info["old_start"],
                    "old_lines": hunk_info["old_lines"],
                    "new_start": hunk_info["new_start"],
                    "new_lines": hunk_info["new_lines"],
                    "changes": []
                }
                line_number = 0
            elif line.startswith('+') and not line.startswith('+++'):
                # 추가된 라인
                analysis["lines_added"] += 1
                if current_hunk:
                    current_hunk["changes"].append({
                        "type": "added",
                        "line_number": line_number,
                        "content": line[1:],
                        "new_line_number": current_hunk["new_start"] + line_number
                    })

                # 중요 변경사항 감지
                self._detect_critical_changes(line[1:], analysis, "added")
                self._detect_function_changes(line[1:], analysis, "added")
                self._detect_import_changes(line[1:], analysis, "added")

            elif line.startswith('-') and not line.startswith('---'):
                # 삭제된 라인
                analysis["lines_removed"] += 1
                if current_hunk:
                    current_hunk["changes"].append({
                        "type": "removed",
                        "line_number": line_number,
                        "content": line[1:],
                        "old_line_number": current_hunk["old_start"] + line_number
                    })

                # 중요 변경사항 감지
                self._detect_critical_changes(line[1:], analysis, "removed")
                self._detect_function_changes(line[1:], analysis, "removed")
                self._detect_import_changes(line[1:], analysis, "removed")

            elif line.startswith(' '):
                # 변경되지 않은 라인
                if current_hunk:
                    current_hunk["changes"].append({
                        "type": "context",
                        "line_number": line_number,
                        "content": line[1:]
                    })

            line_number += 1

        # 마지막 hunk 추가
        if current_hunk:
            analysis["hunks"].append(current_hunk)

        return analysis

    def _parse_hunk_header(self, header: str) -> Dict[str, int]:
        """hunk 헤더 파싱"""
        import re
        # @@ -old_start,old_lines +new_start,new_lines @@
        pattern = r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@'
        match = re.match(pattern, header)

        if match:
            return {
                "old_start": int(match.group(1)),
                "old_lines": int(match.group(2)) if match.group(2) else 1,
                "new_start": int(match.group(3)),
                "new_lines": int(match.group(4)) if match.group(4) else 1
            }

        return {"old_start": 0, "old_lines": 0, "new_start": 0, "new_lines": 0}

    def _detect_critical_changes(self, line: str, analysis: Dict[str, Any], change_type: str):
        """중요 변경사항 감지"""
        critical_patterns = [
            r'password\s*=', r'api_key\s*=', r'secret\s*=',  # 하드코딩된 시크릿
            r'exec\s*\(', r'eval\s*\(',  # 코드 실행
            r'sql\s*=', r'query\s*=',  # SQL 쿼리
            r'http://', r'https://',  # URL
            r'localhost', r'127\.0\.0\.1',  # 로컬 주소
            r'admin', r'root', r'administrator'  # 관리자 관련
        ]

        for pattern in critical_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                analysis["critical_changes"].append({
                    "type": change_type,
                    "pattern": pattern,
                    "line": line.strip(),
                    "severity": "high"
                })

    def _detect_function_changes(self, line: str, analysis: Dict[str, Any], change_type: str):
        """함수 변경사항 감지"""
        function_patterns = [
            r'def\s+\w+',  # Python 함수
            r'function\s+\w+',  # JavaScript 함수
            r'public\s+\w+\s+\w+\s*\(',  # Java 메서드
            r'private\s+\w+\s+\w+\s*\('  # Java private 메서드
        ]

        for pattern in function_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                analysis["function_changes"].append({
                    "type": change_type,
                    "pattern": pattern,
                    "line": line.strip(),
                    "function_name": self._extract_function_name(line, pattern)
                })

    def _detect_import_changes(self, line: str, analysis: Dict[str, Any], change_type: str):
        """import 변경사항 감지"""
        import_patterns = [
            r'import\s+\w+',  # Python import
            r'from\s+\w+\s+import',  # Python from import
            r'require\s*\(',  # Node.js require
            r'#include\s*<',  # C/C++ include
            r'using\s+\w+;'  # C# using
        ]

        for pattern in import_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                analysis["import_changes"].append({
                    "type": change_type,
                    "pattern": pattern,
                    "line": line.strip(),
                    "module": self._extract_module_name(line, pattern)
                })

    def _extract_function_name(self, line: str, pattern: str) -> str:
        """함수명 추출"""
        import re
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            # 간단한 함수명 추출
            words = line.split()
            for i, word in enumerate(words):
                if word.lower() in ['def', 'function', 'public', 'private']:
                    if i + 1 < len(words):
                        return words[i + 1].split('(')[0]
        return "unknown"

    def _extract_module_name(self, line: str, pattern: str) -> str:
        """모듈명 추출"""
        import re
        if 'import' in line.lower():
            # Python import 문
            match = re.search(r'import\s+(\w+)', line)
            if match:
                return match.group(1)
        elif 'require' in line.lower():
            # Node.js require 문
            match = re.search(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]', line)
            if match:
                return match.group(1)
        return "unknown"
