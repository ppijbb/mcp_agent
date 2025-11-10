"""
성능 최적화 모듈
"""

import asyncio
import logging
import time
import functools
from typing import Dict, Any, Optional, Callable, TypeVar, Awaitable
from datetime import datetime, timedelta
import json
import hashlib

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheManager:
    """캐시 관리자"""
    
    def __init__(self, default_ttl: int = 3600):
        """
        CacheManager 초기화
        
        Args:
            default_ttl: 기본 TTL (초)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값 가져오기
        
        Args:
            key: 캐시 키
        
        Returns:
            캐시된 값 또는 None
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # TTL 확인
        if datetime.now() > entry["expires_at"]:
            del self.cache[key]
            return None
        
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        캐시에 값 저장
        
        Args:
            key: 캐시 키
            value: 저장할 값
            ttl: TTL (초, None이면 기본값 사용)
        """
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": datetime.now(),
        }
    
    def delete(self, key: str):
        """캐시에서 값 삭제"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """캐시 전체 삭제"""
        self.cache.clear()
    
    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        캐시 키 생성
        
        Args:
            prefix: 키 접두사
            *args: 위치 인자
            **kwargs: 키워드 인자
        
        Returns:
            생성된 캐시 키
        """
        key_data = {
            "prefix": prefix,
            "args": args,
            "kwargs": sorted(kwargs.items()),
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"


class PerformanceOptimizer:
    """성능 최적화 도구"""
    
    def __init__(self):
        """PerformanceOptimizer 초기화"""
        self.metrics: Dict[str, list] = {}
        self.cache_manager = CacheManager()
    
    def measure_time(self, func_name: str):
        """실행 시간 측정 데코레이터"""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.time() - start_time
                    if func_name not in self.metrics:
                        self.metrics[func_name] = []
                    self.metrics[func_name].append(elapsed)
                    logger.debug(f"{func_name} took {elapsed:.3f}s")
            return wrapper
        return decorator
    
    async def measure_time_async(self, func_name: str):
        """비동기 실행 시간 측정 데코레이터"""
        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.time() - start_time
                    if func_name not in self.metrics:
                        self.metrics[func_name] = []
                    self.metrics[func_name].append(elapsed)
                    logger.debug(f"{func_name} took {elapsed:.3f}s")
            return wrapper
        return decorator
    
    def get_statistics(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = {}
        for func_name, times in self.metrics.items():
            if times:
                stats[func_name] = {
                    "count": len(times),
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times),
                }
        return stats
    
    def cached(self, ttl: Optional[int] = None):
        """캐싱 데코레이터"""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = self.cache_manager.generate_key(
                    func.__name__,
                    *args,
                    **kwargs
                )
                
                # 캐시에서 확인
                cached_value = self.cache_manager.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value
                
                # 캐시 미스 - 함수 실행
                result = func(*args, **kwargs)
                self.cache_manager.set(cache_key, result, ttl)
                logger.debug(f"Cache miss for {func.__name__}, cached result")
                return result
            return wrapper
        return decorator
    
    async def cached_async(self, ttl: Optional[int] = None):
        """비동기 캐싱 데코레이터"""
        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                cache_key = self.cache_manager.generate_key(
                    func.__name__,
                    *args,
                    **kwargs
                )
                
                # 캐시에서 확인
                cached_value = self.cache_manager.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value
                
                # 캐시 미스 - 함수 실행
                result = await func(*args, **kwargs)
                self.cache_manager.set(cache_key, result, ttl)
                logger.debug(f"Cache miss for {func.__name__}, cached result")
                return result
            return wrapper
        return decorator

