"""
Metrics Collector - Prometheus 기반 메트릭 수집

이 모듈은 Prometheus를 사용한 메트릭 수집 시스템을 제공합니다.
성능 모니터링, 알림, 대시보드 등을 지원합니다.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, 
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess
)
import psutil

from .config import config

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Prometheus 기반 메트릭 수집기"""
    
    def __init__(self, registry: CollectorRegistry = None):
        """
        메트릭 수집기 초기화
        
        Args:
            registry (CollectorRegistry, optional): Prometheus 레지스트리
        """
        self.registry = registry or CollectorRegistry()
        self.enable_metrics = config.monitoring.enable_metrics
        
        if not self.enable_metrics:
            logger.info("Metrics collection is disabled")
            return
        
        # 메트릭 정의
        self._define_metrics()
        
        logger.info("MetricsCollector initialized")
    
    def _define_metrics(self):
        """메트릭 정의"""
        # 카운터 메트릭
        self.review_requests_total = Counter(
            'github_pr_review_requests_total',
            'Total number of PR review requests',
            ['repository', 'status', 'provider'],
            registry=self.registry
        )
        
        self.review_comments_total = Counter(
            'github_pr_review_comments_total',
            'Total number of review comments created',
            ['repository', 'type'],
            registry=self.registry
        )
        
        self.api_requests_total = Counter(
            'github_api_requests_total',
            'Total number of GitHub API requests',
            ['endpoint', 'status'],
            registry=self.registry
        )
        
        self.cache_hits_total = Counter(
            'cache_hits_total',
            'Total number of cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses_total = Counter(
            'cache_misses_total',
            'Total number of cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        self.task_queue_total = Counter(
            'task_queue_total',
            'Total number of tasks submitted to queue',
            ['task_type', 'status'],
            registry=self.registry
        )
        
        # 게이지 메트릭
        self.active_reviews = Gauge(
            'active_reviews',
            'Number of active reviews',
            ['repository'],
            registry=self.registry
        )
        
        self.queue_size = Gauge(
            'queue_size',
            'Number of tasks in queue',
            ['queue_name'],
            registry=self.registry
        )
        
        self.cache_size = Gauge(
            'cache_size',
            'Number of items in cache',
            ['cache_type'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        # 히스토그램 메트릭
        self.review_duration = Histogram(
            'review_duration_seconds',
            'Time spent on review generation',
            ['repository', 'provider'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'Time spent on API requests',
            ['endpoint'],
            registry=self.registry
        )
        
        self.cache_access_duration = Histogram(
            'cache_access_duration_seconds',
            'Time spent on cache operations',
            ['operation'],
            registry=self.registry
        )
        
        # 요약 메트릭
        self.review_size = Summary(
            'review_size_bytes',
            'Size of generated reviews',
            ['repository'],
            registry=self.registry
        )
        
        self.diff_size = Summary(
            'diff_size_bytes',
            'Size of PR diffs',
            ['repository'],
            registry=self.registry
        )
    
    def record_review_request(self, repository: str, status: str, provider: str = "unknown"):
        """리뷰 요청 기록"""
        if not self.enable_metrics:
            return
        
        self.review_requests_total.labels(
            repository=repository,
            status=status,
            provider=provider
        ).inc()
    
    def record_review_comment(self, repository: str, comment_type: str):
        """리뷰 코멘트 기록"""
        if not self.enable_metrics:
            return
        
        self.review_comments_total.labels(
            repository=repository,
            type=comment_type
        ).inc()
    
    def record_api_request(self, endpoint: str, status: str, duration: float = None):
        """API 요청 기록"""
        if not self.enable_metrics:
            return
        
        self.api_requests_total.labels(
            endpoint=endpoint,
            status=status
        ).inc()
        
        if duration is not None:
            self.api_request_duration.labels(endpoint=endpoint).observe(duration)
    
    def record_cache_hit(self, cache_type: str = "default"):
        """캐시 히트 기록"""
        if not self.enable_metrics:
            return
        
        self.cache_hits_total.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str = "default"):
        """캐시 미스 기록"""
        if not self.enable_metrics:
            return
        
        self.cache_misses_total.labels(cache_type=cache_type).inc()
    
    def record_task_submission(self, task_type: str, status: str):
        """작업 제출 기록"""
        if not self.enable_metrics:
            return
        
        self.task_queue_total.labels(
            task_type=task_type,
            status=status
        ).inc()
    
    def set_active_reviews(self, repository: str, count: int):
        """활성 리뷰 수 설정"""
        if not self.enable_metrics:
            return
        
        self.active_reviews.labels(repository=repository).set(count)
    
    def set_queue_size(self, queue_name: str, size: int):
        """큐 크기 설정"""
        if not self.enable_metrics:
            return
        
        self.queue_size.labels(queue_name=queue_name).set(size)
    
    def set_cache_size(self, cache_type: str, size: int):
        """캐시 크기 설정"""
        if not self.enable_metrics:
            return
        
        self.cache_size.labels(cache_type=cache_type).set(size)
    
    def record_review_duration(self, repository: str, duration: float, provider: str = "unknown"):
        """리뷰 생성 시간 기록"""
        if not self.enable_metrics:
            return
        
        self.review_duration.labels(
            repository=repository,
            provider=provider
        ).observe(duration)
    
    def record_cache_access_duration(self, operation: str, duration: float):
        """캐시 접근 시간 기록"""
        if not self.enable_metrics:
            return
        
        self.cache_access_duration.labels(operation=operation).observe(duration)
    
    def record_review_size(self, repository: str, size_bytes: int):
        """리뷰 크기 기록"""
        if not self.enable_metrics:
            return
        
        self.review_size.labels(repository=repository).observe(size_bytes)
    
    def record_diff_size(self, repository: str, size_bytes: int):
        """diff 크기 기록"""
        if not self.enable_metrics:
            return
        
        self.diff_size.labels(repository=repository).observe(size_bytes)
    
    def update_system_metrics(self):
        """시스템 메트릭 업데이트"""
        if not self.enable_metrics:
            return
        
        try:
            # 메모리 사용량
            memory_info = psutil.virtual_memory()
            self.memory_usage.set(memory_info.used)
            
            # CPU 사용량
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        메트릭 요약 정보 가져오기
        
        Returns:
            Dict[str, Any]: 메트릭 요약
        """
        if not self.enable_metrics:
            return {"enabled": False}
        
        try:
            # 메트릭 데이터 수집
            metrics_data = generate_latest(self.registry).decode('utf-8')
            
            # 간단한 통계 계산
            summary = {
                "enabled": True,
                "timestamp": datetime.now().isoformat(),
                "metrics_count": len(metrics_data.split('\n')),
                "registry_size": len(self.registry._collector_to_names)
            }
            
            return summary
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {"error": str(e)}
    
    def export_metrics(self) -> str:
        """
        메트릭 내보내기 (Prometheus 형식)
        
        Returns:
            str: Prometheus 형식의 메트릭 데이터
        """
        if not self.enable_metrics:
            return "# Metrics collection is disabled\n"
        
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return f"# Error exporting metrics: {e}\n"
    
    def get_metrics_content_type(self) -> str:
        """메트릭 Content-Type 반환"""
        return CONTENT_TYPE_LATEST
    
    def create_custom_metric(self, metric_type: str, name: str, description: str, 
                           labels: List[str] = None) -> Any:
        """
        사용자 정의 메트릭 생성
        
        Args:
            metric_type (str): 메트릭 타입 (counter, gauge, histogram, summary)
            name (str): 메트릭 이름
            description (str): 메트릭 설명
            labels (List[str], optional): 라벨 목록
            
        Returns:
            Any: 생성된 메트릭 객체
        """
        if not self.enable_metrics:
            return None
        
        try:
            if metric_type == "counter":
                return Counter(name, description, labels or [], registry=self.registry)
            elif metric_type == "gauge":
                return Gauge(name, description, labels or [], registry=self.registry)
            elif metric_type == "histogram":
                return Histogram(name, description, labels or [], registry=self.registry)
            elif metric_type == "summary":
                return Summary(name, description, labels or [], registry=self.registry)
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")
        except Exception as e:
            logger.error(f"Error creating custom metric: {e}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        메트릭 수집기 상태 확인
        
        Returns:
            Dict[str, Any]: 상태 정보
        """
        try:
            if not self.enable_metrics:
                return {
                    "status": "disabled",
                    "enabled": False
                }
            
            # 메트릭 내보내기 테스트
            metrics_data = self.export_metrics()
            
            return {
                "status": "healthy",
                "enabled": True,
                "metrics_count": len(metrics_data.split('\n')),
                "registry_size": len(self.registry._collector_to_names)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "enabled": self.enable_metrics,
                "error": str(e)
            }

# 전역 메트릭 수집기 인스턴스
metrics_collector = MetricsCollector() 