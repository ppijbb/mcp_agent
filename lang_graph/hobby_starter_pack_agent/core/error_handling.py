"""
에러 처리 및 복구 메커니즘
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, TypeVar, Awaitable
from enum import Enum
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorCategory(Enum):
    """에러 카테고리"""
    SYSTEM_ERROR = "system_error"
    NETWORK_ERROR = "network_error"
    DATABASE_ERROR = "database_error"
    AUTHENTICATION_ERROR = "authentication_error"
    VALIDATION_ERROR = "validation_error"
    BUSINESS_LOGIC_ERROR = "business_logic_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"


class ErrorSeverity(Enum):
    """에러 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorHandler:
    """에러 처리 핸들러"""
    
    def __init__(self):
        """ErrorHandler 초기화"""
        self.error_log: list = []
        self.error_counts: Dict[str, int] = {}
    
    def handle_error(
        self,
        error: Exception,
        context: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        에러 처리 및 로깅
        
        Args:
            error: 발생한 예외
            context: 에러 발생 컨텍스트
            severity: 에러 심각도
            category: 에러 카테고리
            metadata: 추가 메타데이터
        
        Returns:
            에러 정보 딕셔너리
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "severity": severity.value,
            "category": category.value,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        # 에러 로깅
        log_level = self._get_log_level(severity)
        logger.log(log_level, f"Error in {context}: {error}", exc_info=True)
        
        # 에러 카운트 업데이트
        error_key = f"{category.value}:{context}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # 에러 로그에 추가
        self.error_log.append(error_info)
        
        # 심각한 에러는 알림 (실제로는 알림 시스템 연동)
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_alert(error_info)
        
        return error_info
    
    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """심각도에 따른 로그 레벨 반환"""
        severity_map = {
            ErrorSeverity.LOW: logging.DEBUG,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }
        return severity_map.get(severity, logging.WARNING)
    
    def _send_alert(self, error_info: Dict[str, Any]):
        """심각한 에러 알림 전송"""
        logger.critical(f"ALERT: {error_info}")
        # 실제로는 알림 시스템 (이메일, Slack 등)에 전송
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """에러 통계 반환"""
        return {
            "total_errors": len(self.error_log),
            "error_counts": self.error_counts,
            "recent_errors": self.error_log[-10:] if self.error_log else [],
        }


class CircuitBreaker:
    """Circuit Breaker 패턴 구현"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        CircuitBreaker 초기화
        
        Args:
            failure_threshold: 실패 임계값
            recovery_timeout: 복구 타임아웃 (초)
            expected_exception: 예상되는 예외 타입
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Circuit Breaker를 통한 함수 호출
        
        Args:
            func: 호출할 함수
            *args: 함수 인자
            **kwargs: 함수 키워드 인자
        
        Returns:
            함수 반환값
        
        Raises:
            Exception: Circuit이 열려있거나 함수 실행 실패
        """
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    async def call_async(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """
        Circuit Breaker를 통한 비동기 함수 호출
        
        Args:
            func: 호출할 비동기 함수
            *args: 함수 인자
            **kwargs: 함수 키워드 인자
        
        Returns:
            함수 반환값
        
        Raises:
            Exception: Circuit이 열려있거나 함수 실행 실패
        """
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """성공 시 처리"""
        self.failure_count = 0
        if self.state == "half_open":
            self.state = "closed"
            logger.info("Circuit breaker CLOSED (recovered)")
    
    def _on_failure(self):
        """실패 시 처리"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """리셋 시도 여부 확인"""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def reset(self):
        """Circuit Breaker 리셋"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"
        logger.info("Circuit breaker manually reset")


class RetryHandler:
    """재시도 핸들러"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0
    ):
        """
        RetryHandler 초기화
        
        Args:
            max_retries: 최대 재시도 횟수
            initial_delay: 초기 지연 시간 (초)
            backoff_factor: 지연 시간 배수
            max_delay: 최대 지연 시간 (초)
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
    
    def retry(
        self,
        func: Callable[..., T],
        *args,
        retry_on: type = Exception,
        **kwargs
    ) -> T:
        """
        재시도 로직이 포함된 함수 호출
        
        Args:
            func: 호출할 함수
            *args: 함수 인자
            retry_on: 재시도할 예외 타입
            **kwargs: 함수 키워드 인자
        
        Returns:
            함수 반환값
        
        Raises:
            Exception: 최대 재시도 횟수 초과 시
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except retry_on as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = min(
                        self.initial_delay * (self.backoff_factor ** attempt),
                        self.max_delay
                    )
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.max_retries} after {delay}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Max retries ({self.max_retries}) exceeded")
        
        raise last_exception
    
    async def retry_async(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        retry_on: type = Exception,
        **kwargs
    ) -> T:
        """
        재시도 로직이 포함된 비동기 함수 호출
        
        Args:
            func: 호출할 비동기 함수
            *args: 함수 인자
            retry_on: 재시도할 예외 타입
            **kwargs: 함수 키워드 인자
        
        Returns:
            함수 반환값
        
        Raises:
            Exception: 최대 재시도 횟수 초과 시
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except retry_on as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = min(
                        self.initial_delay * (self.backoff_factor ** attempt),
                        self.max_delay
                    )
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.max_retries} after {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries ({self.max_retries}) exceeded")
        
        raise last_exception


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60
):
    """Circuit Breaker 데코레이터"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        breaker = CircuitBreaker(failure_threshold, recovery_timeout)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """재시도 데코레이터"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        retry_handler = RetryHandler(max_retries, initial_delay, backoff_factor)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return retry_handler.retry(func, *args, **kwargs)
        
        return wrapper
    return decorator

