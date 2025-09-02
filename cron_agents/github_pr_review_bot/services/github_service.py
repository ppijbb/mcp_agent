"""
GitHub Service

GitHub API와의 상호작용을 담당하는 서비스입니다.
PR 정보 조회, 리뷰 생성, 이슈 관리 등의 기능을 제공합니다.
"""

import logging
import sys
from typing import Dict, List, Any, Optional
from github import Github, GithubException

from ..core.config import config

logger = logging.getLogger(__name__)

class GitHubService:
    """GitHub API 서비스"""
    
    def __init__(self):
        """GitHub 서비스 초기화"""
        if not config.github.token:
            logger.error("GitHub 토큰이 설정되지 않았습니다.")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError("GitHub 토큰이 필요합니다.")
        
        self.github = Github(config.github.token)
        self._validate_connection()
    
    def _validate_connection(self) -> None:
        """GitHub 연결 검증"""
        try:
            # API 호출 테스트
            user = self.github.get_user()
            logger.info(f"GitHub 연결 성공: {user.login}")
            
            # Rate limit 확인
            rate_limit = self.github.get_rate_limit()
            if rate_limit.core.remaining < 100:
                logger.warning(f"GitHub API Rate Limit 부족: {rate_limit.core.remaining}개 남음")
                
        except GithubException as e:
            if e.status == 401:
                raise ValueError("GitHub 인증 실패: 토큰이 유효하지 않습니다.")
            elif e.status == 403:
                raise ValueError("GitHub 접근 권한 없음: 토큰 권한을 확인하세요.")
            elif e.status == 429:
                raise ValueError("GitHub API Rate Limit 초과: 잠시 후 다시 시도하세요.")
            else:
                raise ValueError(f"GitHub 연결 실패: {e}")
    
    def get_repository(self, repo_full_name: str) -> Any:
        """저장소 정보 조회"""
        if not repo_full_name or '/' not in repo_full_name:
            raise ValueError("저장소 이름 형식이 올바르지 않습니다. (owner/repo)")
        
        try:
            repo = self.github.get_repo(repo_full_name)
            logger.info(f"저장소 조회 성공: {repo_full_name}")
            return repo
        except GithubException as e:
            if e.status == 404:
                raise ValueError(f"저장소를 찾을 수 없습니다: {repo_full_name}")
            elif e.status == 403:
                raise ValueError(f"저장소 접근 권한 없음: {repo_full_name}")
            else:
                raise ValueError(f"저장소 조회 실패: {e}")
    
    def get_pull_request(self, repo_full_name: str, pr_number: int) -> Any:
        """Pull Request 정보 조회"""
        repo = self.get_repository(repo_full_name)
        
        try:
            pr = repo.get_pull(pr_number)
            logger.info(f"PR 조회 성공: #{pr_number}")
            return pr
        except GithubException as e:
            if e.status == 404:
                raise ValueError(f"PR을 찾을 수 없습니다: #{pr_number}")
            else:
                raise ValueError(f"PR 조회 실패: {e}")
    
    def get_pr_diff(self, repo_full_name: str, pr_number: int) -> str:
        """PR diff 조회"""
        pr = self.get_pull_request(repo_full_name, pr_number)
        
        try:
            diff = pr.diff()
            if not diff:
                raise ValueError("PR diff가 비어있습니다.")
            return diff
        except Exception as e:
            raise ValueError(f"PR diff 조회 실패: {e}")
    
    def create_review(self, repo_full_name: str, pr_number: int, 
                     body: str, event: str = "COMMENT") -> Any:
        """PR 리뷰 생성"""
        pr = self.get_pull_request(repo_full_name, pr_number)
        
        try:
            review = pr.create_review(body=body, event=event)
            logger.info(f"PR 리뷰 생성 성공: #{pr_number}")
            return review
        except Exception as e:
            raise ValueError(f"PR 리뷰 생성 실패: {e}")
    
    def get_pr_files(self, repo_full_name: str, pr_number: int) -> List[Any]:
        """PR 파일 목록 조회"""
        pr = self.get_pull_request(repo_full_name, pr_number)
        
        try:
            files = list(pr.get_files())
            if not files:
                raise ValueError("PR에 변경된 파일이 없습니다.")
            return files
        except Exception as e:
            raise ValueError(f"PR 파일 목록 조회 실패: {e}")
    
    def get_rate_limit(self) -> Dict[str, Any]:
        """API Rate Limit 정보 조회"""
        try:
            rate_limit = self.github.get_rate_limit()
            return {
                "core": {
                    "limit": rate_limit.core.limit,
                    "remaining": rate_limit.core.remaining,
                    "reset": rate_limit.core.reset
                },
                "search": {
                    "limit": rate_limit.search.limit,
                    "remaining": rate_limit.search.remaining,
                    "reset": rate_limit.search.reset
                }
            }
        except Exception as e:
            raise ValueError(f"Rate Limit 조회 실패: {e}")
