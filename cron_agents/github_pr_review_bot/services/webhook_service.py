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
        """PR 정보 추출"""
        pr = event_data.get('pull_request', {})
        repository = event_data.get('repository', {})
        
        if not pr or not repository:
            raise ValueError("PR 또는 저장소 정보가 없습니다.")
        
        return {
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
    
    def process_pr_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """PR 이벤트 처리"""
        try:
            # PR 정보 추출
            pr_info = self.extract_pr_info(event_data)
            
            # PR 리뷰 처리
            result = self.review_service.process_pr_review(
                repo_full_name=pr_info["repo_full_name"],
                pr_number=pr_info["pr_number"]
            )
            
            # 결과에 PR 정보 추가
            result["pr_info"] = pr_info
            
            logger.info(f"PR 이벤트 처리 완료: #{pr_info['pr_number']}")
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
