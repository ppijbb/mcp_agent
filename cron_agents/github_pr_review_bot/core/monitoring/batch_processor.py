"""
Batch Processor - 요청 배치 처리 및 속도 제한

이 모듈은 API 호출을 배치로 처리하고 속도 제한을 관리하여 비용을 최적화합니다.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import threading

from .config import config

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """배치 요청 항목"""
    id: str
    data: Any
    callback: Callable
    priority: int = 0
    created_at: datetime = None
    retry_count: int = 0
    max_retries: int = 3

class RateLimiter:
    """속도 제한기"""
    
    def __init__(self, max_requests: int, time_window: int):
        """
        속도 제한기 초기화
        
        Args:
            max_requests (int): 시간 창 내 최대 요청 수
            time_window (int): 시간 창 (초)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = threading.Lock()
    
    def can_make_request(self) -> bool:
        """요청 가능 여부 확인"""
        with self.lock:
            now = time.time()
            
            # 오래된 요청 제거
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            return len(self.requests) < self.max_requests
    
    def record_request(self):
        """요청 기록"""
        with self.lock:
            self.requests.append(time.time())
    
    def wait_time(self) -> float:
        """다음 요청까지 대기 시간"""
        with self.lock:
            if len(self.requests) < self.max_requests:
                return 0.0
            
            oldest_request = self.requests[0]
            wait_time = oldest_request + self.time_window - time.time()
            return max(0.0, wait_time)

class BatchProcessor:
    """배치 처리기"""
    
    def __init__(self, batch_size: int = None, max_wait_time: float = 5.0):
        """
        배치 처리기 초기화
        
        Args:
            batch_size (int): 배치 크기
            max_wait_time (float): 최대 대기 시간 (초)
        """
        self.batch_size = batch_size or config.cost_optimization.batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.processing = False
        self.lock = threading.Lock()
        
        # 속도 제한기들
        self.rate_limiters = {
            "openai": RateLimiter(60, 60),  # 분당 60 요청
            "github": RateLimiter(5000, 3600),  # 시간당 5000 요청
            "mcp": RateLimiter(100, 60)  # 분당 100 요청
        }
        
        # 배치 처리 스레드 시작
        self._start_batch_processor()
        
        logger.info(f"Batch Processor 초기화 완료 - 배치 크기: {self.batch_size}")
    
    def _start_batch_processor(self):
        """배치 처리 스레드 시작"""
        def process_batches():
            while True:
                try:
                    self._process_next_batch()
                    time.sleep(0.1)  # 100ms 간격으로 체크
                except Exception as e:
                    logger.error(f"배치 처리 중 오류: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=process_batches, daemon=True)
        thread.start()
    
    def add_request(self, request: BatchRequest, service_type: str = "default") -> str:
        """
        요청 추가
        
        Args:
            request (BatchRequest): 배치 요청
            service_type (str): 서비스 유형
            
        Returns:
            str: 요청 ID
        """
        if request.created_at is None:
            request.created_at = datetime.now()
        
        with self.lock:
            self.pending_requests.append((request, service_type))
            self.pending_requests.sort(key=lambda x: x[0].priority, reverse=True)
        
        logger.debug(f"요청 추가: {request.id} (서비스: {service_type})")
        return request.id
    
    def _process_next_batch(self):
        """다음 배치 처리"""
        with self.lock:
            if not self.pending_requests or self.processing:
                return
            
            # 배치 크기만큼 요청 선택
            batch = self.pending_requests[:self.batch_size]
            self.pending_requests = self.pending_requests[self.batch_size:]
            self.processing = True
        
        try:
            # 서비스별로 그룹화
            service_groups = {}
            for request, service_type in batch:
                if service_type not in service_groups:
                    service_groups[service_type] = []
                service_groups[service_type].append(request)
            
            # 각 서비스별로 처리
            for service_type, requests in service_groups.items():
                self._process_service_batch(service_type, requests)
        
        finally:
            with self.lock:
                self.processing = False
    
    def _process_service_batch(self, service_type: str, requests: List[BatchRequest]):
        """서비스별 배치 처리"""
        rate_limiter = self.rate_limiters.get(service_type)
        
        if rate_limiter and not rate_limiter.can_make_request():
            wait_time = rate_limiter.wait_time()
            logger.debug(f"속도 제한으로 인한 대기: {wait_time:.2f}초")
            time.sleep(wait_time)
        
        # 배치 처리 실행
        try:
            if service_type == "openai":
                self._process_openai_batch(requests)
            elif service_type == "github":
                self._process_github_batch(requests)
            elif service_type == "mcp":
                self._process_mcp_batch(requests)
            else:
                self._process_default_batch(requests)
            
            # 성공한 요청들 기록
            if rate_limiter:
                for _ in requests:
                    rate_limiter.record_request()
        
        except Exception as e:
            logger.error(f"{service_type} 배치 처리 실패: {e}")
            # 실패한 요청들 재시도
            self._retry_failed_requests(requests)
    
    def _process_openai_batch(self, requests: List[BatchRequest]):
        """OpenAI 배치 처리"""
        # OpenAI API는 배치 요청을 지원하므로 여러 요청을 하나로 합침
        combined_data = []
        for request in requests:
            combined_data.append(request.data)
        
        # 실제 OpenAI API 호출 (여기서는 시뮬레이션)
        logger.info(f"OpenAI 배치 처리: {len(requests)}개 요청")
        
        # 결과를 각 요청의 콜백으로 전달
        for i, request in enumerate(requests):
            try:
                result = {"status": "success", "data": f"processed_{i}"}
                request.callback(result)
            except Exception as e:
                logger.error(f"OpenAI 요청 콜백 실패: {e}")
    
    def _process_github_batch(self, requests: List[BatchRequest]):
        """GitHub 배치 처리"""
        # GitHub API는 개별 요청으로 처리
        for request in requests:
            try:
                # 실제 GitHub API 호출 시뮬레이션
                result = {"status": "success", "data": f"github_result_{request.id}"}
                request.callback(result)
            except Exception as e:
                logger.error(f"GitHub 요청 실패: {e}")
                self._retry_request(request)
    
    def _process_mcp_batch(self, requests: List[BatchRequest]):
        """MCP 배치 처리"""
        # MCP 서버들은 개별 요청으로 처리
        for request in requests:
            try:
                # 실제 MCP 서버 호출 시뮬레이션
                result = {"status": "success", "data": f"mcp_result_{request.id}"}
                request.callback(result)
            except Exception as e:
                logger.error(f"MCP 요청 실패: {e}")
                self._retry_request(request)
    
    def _process_default_batch(self, requests: List[BatchRequest]):
        """기본 배치 처리"""
        for request in requests:
            try:
                result = {"status": "success", "data": f"default_result_{request.id}"}
                request.callback(result)
            except Exception as e:
                logger.error(f"기본 요청 실패: {e}")
                self._retry_request(request)
    
    def _retry_failed_requests(self, requests: List[BatchRequest]):
        """실패한 요청들 재시도"""
        for request in requests:
            self._retry_request(request)
    
    def _retry_request(self, request: BatchRequest):
        """개별 요청 재시도"""
        if request.retry_count < request.max_retries:
            request.retry_count += 1
            logger.info(f"요청 재시도: {request.id} (시도 {request.retry_count}/{request.max_retries})")
            
            # 재시도 대기 시간 (지수 백오프)
            wait_time = min(2 ** request.retry_count, 60)  # 최대 60초
            time.sleep(wait_time)
            
            # 다시 대기열에 추가
            with self.lock:
                self.pending_requests.append((request, "default"))
        else:
            logger.error(f"요청 최대 재시도 횟수 초과: {request.id}")
            try:
                error_result = {"status": "error", "message": "최대 재시도 횟수 초과"}
                request.callback(error_result)
            except Exception as e:
                logger.error(f"에러 콜백 실패: {e}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """대기열 상태 조회"""
        with self.lock:
            return {
                "pending_requests": len(self.pending_requests),
                "processing": self.processing,
                "batch_size": self.batch_size,
                "rate_limiters": {
                    service: {
                        "current_requests": len(limiter.requests),
                        "max_requests": limiter.max_requests,
                        "time_window": limiter.time_window,
                        "can_make_request": limiter.can_make_request(),
                        "wait_time": limiter.wait_time()
                    }
                    for service, limiter in self.rate_limiters.items()
                }
            }
    
    def clear_queue(self):
        """대기열 정리"""
        with self.lock:
            cleared_count = len(self.pending_requests)
            self.pending_requests.clear()
            logger.info(f"대기열 정리 완료: {cleared_count}개 요청 제거")
    
    def wait_for_completion(self, timeout: float = 30.0) -> bool:
        """모든 요청 완료 대기"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                if not self.pending_requests and not self.processing:
                    return True
            
            time.sleep(0.1)
        
        return False

# 전역 인스턴스
batch_processor = BatchProcessor()
