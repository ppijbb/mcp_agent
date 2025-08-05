"""
GitHub Client - GitHub API와 통신하는 클라이언트

이 모듈은 GitHub API와 통신하여 PR 정보를 가져오고, 리뷰를 등록하는 기능을 제공합니다.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from github import Github
from github.PullRequest import PullRequest
from github.Repository import Repository
from github.GithubException import GithubException

logger = logging.getLogger(__name__)

class GitHubClient:
    """GitHub API와 통신하는 클라이언트 클래스"""
    
    def __init__(self, token: str = None):
        """
        GitHub 클라이언트 초기화
        
        Args:
            token (str, optional): GitHub API 토큰. 기본값은 환경 변수 GITHUB_TOKEN.
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub 토큰이 필요합니다. 환경 변수 GITHUB_TOKEN을 설정하거나 직접 제공해주세요.")
        
        self.client = Github(self.token)
        logger.info("GitHub 클라이언트가 초기화되었습니다.")
    
    def get_repository(self, repo_full_name: str) -> Repository:
        """
        저장소 객체를 가져옵니다.
        
        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            
        Returns:
            Repository: GitHub 저장소 객체
        """
        try:
            return self.client.get_repo(repo_full_name)
        except GithubException as e:
            logger.error(f"저장소를 가져오는 중 오류 발생: {e}")
            raise
    
    def get_pull_request(self, repo_full_name: str, pr_number: int) -> PullRequest:
        """
        PR 객체를 가져옵니다.
        
        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호
            
        Returns:
            PullRequest: GitHub PR 객체
        """
        try:
            repo = self.get_repository(repo_full_name)
            return repo.get_pull(pr_number)
        except GithubException as e:
            logger.error(f"PR을 가져오는 중 오류 발생: {e}")
            raise
    
    def get_pr_diff(self, repo_full_name: str, pr_number: int) -> str:
        """
        PR의 diff 내용을 가져옵니다.
        
        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호
            
        Returns:
            str: PR diff 내용
        """
        pr = self.get_pull_request(repo_full_name, pr_number)
        return pr.get_patch()
    
    def get_pr_files(self, repo_full_name: str, pr_number: int) -> List[Dict[str, Any]]:
        """
        PR에서 변경된 파일 목록을 가져옵니다.
        
        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호
            
        Returns:
            List[Dict[str, Any]]: 변경된 파일 정보 목록
        """
        pr = self.get_pull_request(repo_full_name, pr_number)
        files = pr.get_files()
        
        result = []
        for file in files:
            result.append({
                "filename": file.filename,
                "status": file.status,
                "additions": file.additions,
                "deletions": file.deletions,
                "changes": file.changes,
                "patch": file.patch
            })
        
        return result
    
    def create_review_comment(self, repo_full_name: str, pr_number: int, 
                              body: str, commit_id: str, path: str, 
                              position: int) -> Dict[str, Any]:
        """
        PR의 특정 라인에 리뷰 코멘트를 남깁니다.
        
        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호
            body (str): 코멘트 내용
            commit_id (str): 커밋 ID
            path (str): 파일 경로
            position (int): 파일에서의 위치 (라인 번호)
            
        Returns:
            Dict[str, Any]: 생성된 코멘트 정보
        """
        try:
            pr = self.get_pull_request(repo_full_name, pr_number)
            comment = pr.create_review_comment(body, commit_id, path, position)
            
            return {
                "id": comment.id,
                "body": comment.body,
                "path": comment.path,
                "position": comment.position,
                "html_url": comment.html_url
            }
        except GithubException as e:
            logger.error(f"리뷰 코멘트를 생성하는 중 오류 발생: {e}")
            raise
    
    def create_review(self, repo_full_name: str, pr_number: int, 
                     body: str, event: str = "COMMENT",
                     comments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        PR에 전체 리뷰를 등록합니다.
        
        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호
            body (str): 리뷰 내용
            event (str): 리뷰 이벤트 타입 (COMMENT, APPROVE, REQUEST_CHANGES)
            comments (List[Dict[str, Any]], optional): 라인별 코멘트 목록
            
        Returns:
            Dict[str, Any]: 생성된 리뷰 정보
        """
        try:
            pr = self.get_pull_request(repo_full_name, pr_number)
            
            # 라인별 코멘트가 있는 경우
            if comments:
                review_comments = []
                for comment in comments:
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
            
            return {
                "id": review.id,
                "body": review.body,
                "state": review.state,
                "html_url": review.html_url
            }
        except GithubException as e:
            logger.error(f"리뷰를 생성하는 중 오류 발생: {e}")
            raise
    
    def get_commit_details(self, repo_full_name: str, pr_number: int) -> List[Dict[str, Any]]:
        """
        PR의 커밋 정보를 가져옵니다.
        
        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호
            
        Returns:
            List[Dict[str, Any]]: 커밋 정보 목록
        """
        pr = self.get_pull_request(repo_full_name, pr_number)
        commits = pr.get_commits()
        
        result = []
        for commit in commits:
            result.append({
                "sha": commit.sha,
                "message": commit.commit.message,
                "author": commit.commit.author.name,
                "date": commit.commit.author.date.isoformat()
            })
        
        return result
    
    def get_latest_commit(self, repo_full_name: str, pr_number: int) -> Dict[str, Any]:
        """
        PR의 최신 커밋 정보를 가져옵니다.
        
        Args:
            repo_full_name (str): 'owner/repo' 형식의 저장소 전체 이름
            pr_number (int): PR 번호
            
        Returns:
            Dict[str, Any]: 최신 커밋 정보
        """
        commits = self.get_commit_details(repo_full_name, pr_number)
        if not commits:
            raise ValueError("PR에 커밋이 없습니다.")
        
        # 가장 최근 커밋 반환
        return commits[-1] 