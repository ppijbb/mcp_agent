"""
PR Review Application

GitHub PR 리뷰 봇의 메인 애플리케이션입니다.
서비스들을 조합하여 전체적인 PR 리뷰 프로세스를 관리합니다.
"""

import logging
import sys
from typing import Dict, Any, Optional

from ..services import GitHubService, MCPService, ReviewService, WebhookService
from ..core.config import config

logger = logging.getLogger(__name__)

class PRReviewApp:
    """PR 리뷰 애플리케이션"""
    
    def __init__(self):
        """애플리케이션 초기화"""
        self._initialize_services()
        logger.info("PR 리뷰 애플리케이션 초기화 완료")
    
    def _initialize_services(self) -> None:
        """서비스들 초기화 - MCP 통합 (Gemini CLI + vLLM)"""
        try:
            # 핵심 서비스들 초기화
            self.github_service = GitHubService()
            self.mcp_service = MCPService()
            
            # 의존성이 있는 서비스들 초기화
            self.review_service = ReviewService(
                github_service=self.github_service,
                mcp_service=self.mcp_service
            )
            self.webhook_service = WebhookService(
                review_service=self.review_service
            )
            
            logger.info("모든 서비스 초기화 완료 (MCP 통합 - Gemini CLI + vLLM)")
            
        except Exception as e:
            logger.error(f"서비스 초기화 실패: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError(f"서비스 초기화 실패: {e}")
    
    def process_webhook(self, event_type: str, payload: bytes, signature: str) -> Dict[str, Any]:
        """웹훅 이벤트 처리"""
        try:
            result = self.webhook_service.handle_webhook_event(
                event_type=event_type,
                payload=payload,
                signature=signature
            )
            
            logger.info(f"웹훅 처리 완료: {result.get('status', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"웹훅 처리 실패: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError(f"웹훅 처리 실패: {e}")
    
    def review_pr(self, repo_full_name: str, pr_number: int) -> Dict[str, Any]:
        """PR 리뷰 수행"""
        try:
            result = self.review_service.process_pr_review(
                repo_full_name=repo_full_name,
                pr_number=pr_number
            )
            
            logger.info(f"PR 리뷰 완료: #{pr_number}")
            return result
            
        except Exception as e:
            logger.error(f"PR 리뷰 실패: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError(f"PR 리뷰 실패: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """헬스 체크"""
        try:
            # GitHub 서비스 상태
            github_status = {
                "connected": True,
                "rate_limit": self.github_service.get_rate_limit()
            }
            
            # MCP 서비스 상태
            mcp_status = self.review_service.get_mcp_health()
            
            # 전체 상태
            overall_status = {
                "status": "healthy",
                "services": {
                    "github": github_status,
                    "mcp": mcp_status
                },
                "config": {
                    "auto_review_enabled": config.github.auto_review_enabled,
                    "require_explicit_request": config.github.require_explicit_review_request,
                    "fail_fast_enabled": config.github.fail_fast_on_error,
                    "free_ai_review": True,
                    "mcp_integration": True
                }
            }
            
            return overall_status
            
        except Exception as e:
            logger.error(f"헬스 체크 실패: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def get_service_info(self) -> Dict[str, Any]:
        """서비스 정보 조회"""
        return {
            "github_service": {
                "class": self.github_service.__class__.__name__,
                "initialized": True
            },
            "mcp_service": {
                "class": self.mcp_service.__class__.__name__,
                "initialized": True,
                "available_tools": len(self.mcp_service.get_available_tools())
            },
            "review_service": {
                "class": self.review_service.__class__.__name__,
                "initialized": True,
                "free_ai_review": True,
                "mcp_integration": True
            },
            "webhook_service": {
                "class": self.webhook_service.__class__.__name__,
                "initialized": True
            }
        }
    
    def get_mcp_usage_stats(self) -> Dict[str, Any]:
        """MCP 사용량 통계 조회"""
        try:
            return self.review_service.get_mcp_usage_stats()
        except Exception as e:
            logger.error(f"MCP 사용량 조회 실패: {e}")
            return {"error": str(e)}
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """사용 가능한 도구 목록 조회"""
        try:
            return self.review_service.get_available_tools()
        except Exception as e:
            logger.error(f"도구 목록 조회 실패: {e}")
            return []
