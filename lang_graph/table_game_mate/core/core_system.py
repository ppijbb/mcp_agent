"""
Table Game Mate 핵심 시스템
통합된 에러 처리, 캐싱, 성능 모니터링
"""

import asyncio
import time
import json
import hashlib
import pickle
import os
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict, deque
import logging

from ..utils.logger import get_logger

# ============================================================================
# 에러 처리 시스템
# ============================================================================

class ErrorSeverity(Enum):
    """에러 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """에러 카테고리"""
    LLM_ERROR = "llm_error"
    MCP_ERROR = "mcp_error"
    AGENT_ERROR = "agent_error"
    SYSTEM_ERROR = "system_error"

@dataclass
class ErrorRecord:
    """에러 기록"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    agent_id: Optional[str]
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    resolved: bool = False
    retry_count: int = 0
    max_retries: int = 3

class ErrorHandler:
    """에러 핸들러"""
    
    def __init__(self):
        self.logger = get_logger("error_handler")
        self.error_records: List[ErrorRecord] = []
        self.retry_strategies = {
            ErrorCategory.LLM_ERROR: {"max_retries": 3, "retry_delay": 1.0},
            ErrorCategory.MCP_ERROR: {"max_retries": 5, "retry_delay": 2.0},
            ErrorCategory.AGENT_ERROR: {"max_retries": 2, "retry_delay": 0.5}
        }
    
    async def handle_error(self, error: Exception, severity: ErrorSeverity, 
                          category: ErrorCategory, agent_id: Optional[str] = None) -> str:
        """에러 처리"""
        error_id = f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.error_records)}"
        
        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            agent_id=agent_id,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace="",  # 간소화
            context={}
        )
        
        self.error_records.append(error_record)
        self.logger.error(f"Error: {error_id} - {str(error)}")
        
        return error_id
    
    def get_error_summary(self) -> Dict[str, Any]:
        """에러 요약"""
        return {
            "total_errors": len(self.error_records),
            "unresolved_errors": len([r for r in self.error_records if not r.resolved]),
            "errors_by_severity": defaultdict(int),
            "errors_by_category": defaultdict(int)
        }

# ============================================================================
# 캐싱 시스템
# ============================================================================

class CachePolicy(Enum):
    """캐시 정책"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"

@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0

class MemoryCache:
    """메모리 캐시"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """값 조회"""
        if key in self.cache:
            entry = self.cache[key]
            entry.accessed_at = datetime.now()
            entry.access_count += 1
            self.cache.move_to_end(key)
            return entry.value
        return None
    
    def set(self, key: str, value: Any) -> bool:
        """값 저장"""
        if len(self.cache) >= self.max_size:
            # LRU 정책으로 제거
            self.cache.popitem(last=False)
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            accessed_at=datetime.now()
        )
        
        self.cache[key] = entry
        return True
    
    def clear(self):
        """캐시 정리"""
        self.cache.clear()

class CacheManager:
    """캐시 관리자"""
    
    def __init__(self):
        self.memory_cache = MemoryCache()
        self.cache_stats = {"hits": 0, "misses": 0}
    
    async def get(self, key: str, default: Any = None) -> Any:
        """캐시에서 값 조회"""
        value = self.memory_cache.get(key)
        if value is not None:
            self.cache_stats["hits"] += 1
            return value
        
        self.cache_stats["misses"] += 1
        return default
    
    async def set(self, key: str, value: Any) -> bool:
        """캐시에 값 저장"""
        return self.memory_cache.set(key, value)
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total * 100) if total > 0 else 0
        
        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "memory_size": len(self.memory_cache.cache)
        }

# ============================================================================
# 성능 모니터링 시스템
# ============================================================================

class MetricType(Enum):
    """메트릭 타입"""
    COUNTER = "counter"
    GAUGE = "gauge"
    TIMER = "timer"

@dataclass
class Metric:
    """메트릭"""
    name: str
    metric_type: MetricType
    values: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_value(self, value: float):
        """값 추가"""
        self.values.append((datetime.now(), value))
    
    def get_latest(self) -> Optional[float]:
        """최신 값"""
        return self.values[-1][1] if self.values else None
    
    def get_average(self, minutes: int = 5) -> Optional[float]:
        """평균값"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent = [v for t, v in self.values if t > cutoff]
        return sum(recent) / len(recent) if recent else None

class PerformanceMonitor:
    """성능 모니터"""
    
    def __init__(self):
        self.logger = get_logger("performance_monitor")
        self.metrics: Dict[str, Metric] = {}
        self.monitoring_active = False
        
        # 기본 메트릭 등록
        self.register_metric("cpu_usage", MetricType.GAUGE)
        self.register_metric("memory_usage", MetricType.GAUGE)
        self.register_metric("response_time", MetricType.TIMER)
    
    def register_metric(self, name: str, metric_type: MetricType):
        """메트릭 등록"""
        self.metrics[name] = Metric(name=name, metric_type=metric_type)
    
    def record_value(self, metric_name: str, value: float):
        """메트릭 값 기록"""
        if metric_name in self.metrics:
            self.metrics[metric_name].add_value(value)
    
    def start_monitoring(self):
        """모니터링 시작"""
        self.monitoring_active = True
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        self.logger.info("Performance monitoring stopped")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트"""
        return {
            "monitoring_active": self.monitoring_active,
            "metrics": {
                name: {
                    "latest": metric.get_latest(),
                    "average_5min": metric.get_average(5)
                }
                for name, metric in self.metrics.items()
            }
        }

# ============================================================================
# 통합 시스템 관리자
# ============================================================================

class CoreSystem:
    """통합 핵심 시스템"""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.cache_manager = CacheManager()
        self.performance_monitor = PerformanceMonitor()
        self.logger = get_logger("core_system")
    
    async def initialize(self):
        """시스템 초기화"""
        self.performance_monitor.start_monitoring()
        self.logger.info("Core system initialized")
    
    async def shutdown(self):
        """시스템 종료"""
        self.performance_monitor.stop_monitoring()
        self.logger.info("Core system shutdown")
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태"""
        return {
            "error_summary": self.error_handler.get_error_summary(),
            "cache_stats": self.cache_manager.get_stats(),
            "performance_report": self.performance_monitor.get_performance_report()
        }

# ============================================================================
# 유틸리티 함수들
# ============================================================================

def handle_errors(severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                  category: ErrorCategory = ErrorCategory.SYSTEM_ERROR):
    """에러 처리 데코레이터"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                core_system = get_core_system()
                await core_system.error_handler.handle_error(e, severity, category)
                raise
        return async_wrapper
    return decorator

def cached(key_prefix: str = "", ttl_seconds: Optional[int] = None):
    """캐싱 데코레이터"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            core_system = get_core_system()
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # 캐시에서 조회
            cached_result = await core_system.cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 함수 실행 및 결과 캐싱
            result = await func(*args, **kwargs)
            await core_system.cache_manager.set(cache_key, result)
            return result
        return async_wrapper
    return decorator

def monitor_performance(metric_name: str):
    """성능 모니터링 데코레이터"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            core_system = get_core_system()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                core_system.performance_monitor.record_value(metric_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                core_system.performance_monitor.record_value(f"{metric_name}_error", duration)
                raise
        return async_wrapper
    return decorator

# ============================================================================
# 전역 인스턴스
# ============================================================================

_core_system = None

def get_core_system() -> CoreSystem:
    """전역 핵심 시스템 인스턴스"""
    global _core_system
    if _core_system is None:
        _core_system = CoreSystem()
    return _core_system
