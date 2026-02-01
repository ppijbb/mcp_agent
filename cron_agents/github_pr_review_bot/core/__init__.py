"""
Core 모듈 - GitHub PR Review Bot의 핵심 기능들

이 패키지는 봇의 핵심 기능들을 기능별로 분류하여 포함합니다:
- config.py: 설정 관리
- api/: 외부 API 통합 (GitHub API, 웹훅 서버)
- ai/: 인공지능 서비스 (Gemini CLI, MCP 통합, 리뷰 생성)
- storage/: 데이터 저장 및 관리 (캐시, 큐)
- monitoring/: 모니터링 및 성능 관리 (메트릭, 배치 처리)
"""

from .config import Config

# API 모듈
from .api import GitHubClient, GitHubPRReviewServer

# AI 모듈
from .ai import gemini_service, MCPIntegrationManager, ReviewGenerator, ReviewEnhancer, review_enhancer

# Storage 모듈
from .storage import CacheManager, TaskQueue

# Monitoring 모듈
from .monitoring import MetricsCollector, BatchProcessor

__all__ = [
    "Config",
    "GitHubClient",
    "GitHubPRReviewServer",
    "gemini_service",
    "MCPIntegrationManager",
    "ReviewGenerator",
    "ReviewEnhancer",
    "review_enhancer",
    "CacheManager",
    "TaskQueue",
    "MetricsCollector",
    "BatchProcessor"
]
