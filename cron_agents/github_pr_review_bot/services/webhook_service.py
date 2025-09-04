"""
Webhook Service

GitHub 웹훅 이벤트 처리를 담당하는 서비스입니다.
웹훅 서명 검증, 이벤트 파싱, PR 이벤트 처리 등의 기능을 제공합니다.
"""

import logging
import sys
import hmac
import hashlib
import json
from typing import Dict, Any, Optional

from .review_service import ReviewService
from ..core.config import config

logger = logging.getLogger(__name__)

class WebhookService:
    """웹훅 서비스"""
    
    def __init__(self, review_service: ReviewService):
        """웹훅 서비스 초기화"""
        self.review_service = review_service
        logger.info("웹훅 서비스 초기화 완료")
    
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """웹훅 서명 검증"""
        if not config.github.webhook_secret:
            logger.error("웹훅 시크릿이 설정되지 않았습니다.")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError("웹훅 시크릿이 필요합니다.")
        
        try:
            expected_signature = hmac.new(
                config.github.webhook_secret.encode('utf-8'),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            # GitHub는 'sha256=' 접두사를 사용
            expected_signature = f"sha256={expected_signature}"
            
            # 상수 시간 비교로 타이밍 공격 방지
            is_valid = hmac.compare_digest(signature, expected_signature)
            
            if is_valid:
                logger.info("웹훅 서명 검증 성공")
            else:
                logger.warning("웹훅 서명 검증 실패")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"웹훅 서명 검증 중 오류: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError(f"웹훅 서명 검증 실패: {e}")
    
    def parse_webhook_payload(self, payload: bytes) -> Dict[str, Any]:
        """웹훅 페이로드 파싱"""
        try:
            data = json.loads(payload.decode('utf-8'))
            logger.info(f"웹훅 페이로드 파싱 성공: {data.get('action', 'unknown')} 이벤트")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"웹훅 페이로드 JSON 파싱 실패: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError(f"웹훅 페이로드 파싱 실패: {e}")
        except Exception as e:
            logger.error(f"웹훅 페이로드 파싱 중 오류: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError(f"웹훅 페이로드 파싱 실패: {e}")
    
    def should_process_pr_event(self, event_data: Dict[str, Any]) -> bool:
        """PR 이벤트 처리 여부 결정"""
        action = event_data.get('action')
        pr = event_data.get('pull_request', {})
        
        # 처리할 액션들
        processable_actions = ['opened', 'synchronize', 'reopened']
        
        if action not in processable_actions:
            logger.info(f"PR 이벤트 스킵: {action} 액션은 처리하지 않음")
            return False
        
        # PR 상태 확인
        if pr.get('state') != 'open':
            logger.info(f"PR 이벤트 스킵: PR이 열려있지 않음 (상태: {pr.get('state')})")
            return False
        
        # Draft PR 확인
        if pr.get('draft', False):
            logger.info("PR 이벤트 스킵: Draft PR")
            return False
        
        # 자동 리뷰 비활성화 확인
        if not config.github.auto_review_enabled:
            logger.info("PR 이벤트 스킵: 자동 리뷰 비활성화")
            return False
        
        # 명시적 리뷰 요청 확인
        if config.github.require_explicit_review_request:
            pr_body = pr.get('body', '')
            review_keywords = ["@review-bot", "[REVIEW]", "[리뷰요청]"]
            if not any(keyword in pr_body for keyword in review_keywords):
                logger.info("PR 이벤트 스킵: 명시적 리뷰 요청 없음")
                return False
        
        logger.info(f"PR 이벤트 처리 승인: {action} 액션")
        return True
    
    def extract_pr_info(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """GitHub 앱에서 제공하는 풍부한 PR 정보 추출"""
        pr = event_data.get('pull_request', {})
        repository = event_data.get('repository', {})
        sender = event_data.get('sender', {})
        
        if not pr or not repository:
            raise ValueError("PR 또는 저장소 정보가 없습니다.")
        
        # 기본 PR 정보
        pr_info = {
            "pr_number": pr.get('number'),
            "repo_full_name": repository.get('full_name'),
            "repo_name": repository.get('name'),
            "repo_owner": repository.get('owner', {}).get('login'),
            "pr_title": pr.get('title'),
            "pr_body": pr.get('body', ''),
            "pr_state": pr.get('state'),
            "pr_draft": pr.get('draft', False),
            "head_ref": pr.get('head', {}).get('ref'),
            "base_ref": pr.get('base', {}).get('ref'),
            "action": event_data.get('action')
        }
        
        # GitHub 앱에서 제공하는 추가 정보들
        pr_info.update({
            # PR 작성자 정보
            "author": {
                "login": pr.get('user', {}).get('login'),
                "id": pr.get('user', {}).get('id'),
                "type": pr.get('user', {}).get('type'),
                "avatar_url": pr.get('user', {}).get('avatar_url'),
                "html_url": pr.get('user', {}).get('html_url')
            },
            
            # 이벤트 트리거 사용자 정보
            "sender": {
                "login": sender.get('login'),
                "id": sender.get('id'),
                "type": sender.get('type'),
                "avatar_url": sender.get('avatar_url')
            },
            
            # 저장소 상세 정보
            "repository": {
                "id": repository.get('id'),
                "name": repository.get('name'),
                "full_name": repository.get('full_name'),
                "private": repository.get('private'),
                "html_url": repository.get('html_url'),
                "description": repository.get('description'),
                "language": repository.get('language'),
                "topics": repository.get('topics', []),
                "created_at": repository.get('created_at'),
                "updated_at": repository.get('updated_at'),
                "pushed_at": repository.get('pushed_at'),
                "size": repository.get('size'),
                "stargazers_count": repository.get('stargazers_count'),
                "watchers_count": repository.get('watchers_count'),
                "forks_count": repository.get('forks_count'),
                "open_issues_count": repository.get('open_issues_count')
            },
            
            # PR 상세 정보
            "pr_details": {
                "id": pr.get('id'),
                "node_id": pr.get('node_id'),
                "html_url": pr.get('html_url'),
                "diff_url": pr.get('diff_url'),
                "patch_url": pr.get('patch_url'),
                "issue_url": pr.get('issue_url'),
                "commits_url": pr.get('commits_url'),
                "review_comments_url": pr.get('review_comments_url'),
                "review_comment_url": pr.get('review_comment_url'),
                "comments_url": pr.get('comments_url'),
                "statuses_url": pr.get('statuses_url'),
                "number": pr.get('number'),
                "state": pr.get('state'),
                "locked": pr.get('locked'),
                "title": pr.get('title'),
                "body": pr.get('body'),
                "user": pr.get('user'),
                "labels": pr.get('labels', []),
                "milestone": pr.get('milestone'),
                "assignees": pr.get('assignees', []),
                "requested_reviewers": pr.get('requested_reviewers', []),
                "requested_teams": pr.get('requested_teams', []),
                "draft": pr.get('draft'),
                "merged": pr.get('merged'),
                "mergeable": pr.get('mergeable'),
                "rebaseable": pr.get('rebaseable'),
                "mergeable_state": pr.get('mergeable_state'),
                "merged_by": pr.get('merged_by'),
                "comments": pr.get('comments'),
                "review_comments": pr.get('review_comments'),
                "maintainer_can_modify": pr.get('maintainer_can_modify'),
                "commits": pr.get('commits'),
                "additions": pr.get('additions'),
                "deletions": pr.get('deletions'),
                "changed_files": pr.get('changed_files'),
                "created_at": pr.get('created_at'),
                "updated_at": pr.get('updated_at'),
                "closed_at": pr.get('closed_at'),
                "merged_at": pr.get('merged_at')
            },
            
            # 브랜치 정보
            "branches": {
                "head": {
                    "label": pr.get('head', {}).get('label'),
                    "ref": pr.get('head', {}).get('ref'),
                    "sha": pr.get('head', {}).get('sha'),
                    "user": pr.get('head', {}).get('user'),
                    "repo": pr.get('head', {}).get('repo')
                },
                "base": {
                    "label": pr.get('base', {}).get('label'),
                    "ref": pr.get('base', {}).get('ref'),
                    "sha": pr.get('base', {}).get('sha'),
                    "user": pr.get('base', {}).get('user'),
                    "repo": pr.get('base', {}).get('repo')
                }
            },
            
            # 라벨 및 마일스톤 정보
            "labels": [label.get('name') for label in pr.get('labels', [])],
            "milestone": pr.get('milestone', {}).get('title') if pr.get('milestone') else None,
            
            # 리뷰어 정보
            "reviewers": {
                "requested": [reviewer.get('login') for reviewer in pr.get('requested_reviewers', [])],
                "teams": [team.get('name') for team in pr.get('requested_teams', [])]
            },
            
            # 통계 정보
            "stats": {
                "additions": pr.get('additions', 0),
                "deletions": pr.get('deletions', 0),
                "changed_files": pr.get('changed_files', 0),
                "commits": pr.get('commits', 0),
                "comments": pr.get('comments', 0),
                "review_comments": pr.get('review_comments', 0)
            }
        })
        
        return pr_info
    
    def process_pr_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """PR 이벤트 처리 - GitHub 앱 정보 활용"""
        try:
            # GitHub 앱에서 제공하는 풍부한 PR 정보 추출
            pr_info = self.extract_pr_info(event_data)
            
            # PR 리뷰 처리 (GitHub 앱 정보 전달)
            result = self.review_service.process_pr_review(
                repo_full_name=pr_info["repo_full_name"],
                pr_number=pr_info["pr_number"],
                github_context=pr_info
            )
            
            # 결과에 PR 정보 추가
            result["pr_info"] = pr_info
            
            logger.info(f"PR 이벤트 처리 완료: #{pr_info['pr_number']} (GitHub 앱 정보 활용)")
            return result
            
        except Exception as e:
            logger.error(f"PR 이벤트 처리 실패: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError(f"PR 이벤트 처리 실패: {e}")
    
    def handle_webhook_event(self, event_type: str, payload: bytes, signature: str) -> Dict[str, Any]:
        """웹훅 이벤트 처리 (전체 프로세스)"""
        try:
            # 서명 검증
            if not self.verify_signature(payload, signature):
                raise ValueError("웹훅 서명 검증 실패")
            
            # 페이로드 파싱
            event_data = self.parse_webhook_payload(payload)
            
            # 이벤트 타입별 처리
            if event_type == "pull_request":
                if self.should_process_pr_event(event_data):
                    return self.process_pr_event(event_data)
                else:
                    return {"status": "skipped", "reason": "event_not_processable"}
            else:
                logger.info(f"이벤트 타입 '{event_type}'은 처리하지 않음")
                return {"status": "ignored", "reason": "unsupported_event_type"}
                
        except Exception as e:
            logger.error(f"웹훅 이벤트 처리 실패: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError(f"웹훅 이벤트 처리 실패: {e}")
