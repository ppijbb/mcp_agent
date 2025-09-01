"""
GitHub Client - GitHub API와 통신하는 클라이언트 (NO FALLBACK MODE)

이 모듈은 GitHub API와 통신하여 PR 정보를 가져오고, 리뷰를 등록하는 기능을 제공합니다.
모든 오류는 fallback 없이 즉시 상위로 전파됩니다.
"""

import os
import logging
import sys
from typing import Dict, List, Any, Optional, Tuple
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
            
            self.client = Github(self.token)
            if not self.client:
                raise ValueError("GitHub 클라이언트 초기화에 실패했습니다.")
            
            # 토큰 유효성 검증
            try:
                user = self.client.get_user()
                if not user:
                    raise ValueError("GitHub 토큰이 유효하지 않습니다.")
                logger.info(f"GitHub 클라이언트가 초기화되었습니다. 사용자: {user.login} (NO FALLBACK MODE)")
            except GithubException as e:
                if e.status == 401:
                    raise ValueError("GitHub 토큰이 유효하지 않습니다.")
                else:
                    raise ValueError(f"GitHub API 연결 테스트 실패: {e}")
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
        
        try:
            repo = self.client.get_repo(repo_full_name)
            if not repo:
                raise ValueError(f"저장소를 찾을 수 없습니다: {repo_full_name}")
            return repo
        except GithubException as e:
            if e.status == 404:
                raise ValueError(f"저장소를 찾을 수 없습니다: {repo_full_name}")
            elif e.status == 403:
                raise ValueError(f"저장소에 접근할 권한이 없습니다: {repo_full_name}")
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
                    "patch": file.patch or ""
                }
                
                # 필수 필드 검증
                if not file_info["filename"]:
                    logger.warning(f"파일명이 없는 파일이 발견되었습니다: {repo_full_name}#{pr_number}")
                    continue
                
                result.append(file_info)
            
            if not result:
                raise ValueError(f"변경된 파일이 없습니다: {repo_full_name}#{pr_number}")
            
            logger.info(f"PR 파일 목록 가져오기 완료: {repo_full_name}#{pr_number}, 파일 수: {len(result)}")
            return result
        except Exception as e:
            raise ValueError(f"PR 파일 목록을 가져오는 중 오류 발생: {e}")
    
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