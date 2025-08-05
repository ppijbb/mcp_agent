"""
Cache Manager - Redis 기반 캐시 관리

이 모듈은 Redis를 사용한 고성능 캐시 시스템을 제공합니다.
메모리 캐시, 분산 캐시, TTL 관리 등을 지원합니다.
"""

import json
import logging
import time
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import redis
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config

logger = logging.getLogger(__name__)

class CacheManager:
    """Redis 기반 캐시 관리자"""
    
    def __init__(self, redis_url: str = None, ttl: int = None):
        """
        캐시 관리자 초기화
        
        Args:
            redis_url (str, optional): Redis URL
            ttl (int, optional): 기본 TTL (초)
        """
        self.redis_url = redis_url or config.cache.redis_url
        self.default_ttl = ttl or config.cache.ttl
        self.max_size = config.cache.max_size
        self.enable_cache = config.cache.enable_cache
        
        # Redis 연결
        self.redis_client = None
        self._connect_redis()
        
        # 메모리 캐시 (Redis 연결 실패 시 대체)
        self._memory_cache = {}
        self._memory_cache_ttl = {}
        
        logger.info(f"CacheManager initialized with TTL: {self.default_ttl}s")
    
    def _connect_redis(self):
        """Redis 연결"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            # 연결 테스트
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using memory cache.")
            self.redis_client = None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def get(self, key: str, default: Any = None) -> Any:
        """
        캐시에서 값 가져오기
        
        Args:
            key (str): 캐시 키
            default (Any): 기본값
            
        Returns:
            Any: 캐시된 값 또는 기본값
        """
        if not self.enable_cache:
            return default
        
        try:
            if self.redis_client:
                # Redis에서 가져오기
                value = self.redis_client.get(key)
                if value is not None:
                    return json.loads(value)
            else:
                # 메모리 캐시에서 가져오기
                if key in self._memory_cache:
                    if time.time() < self._memory_cache_ttl[key]:
                        return self._memory_cache[key]
                    else:
                        # TTL 만료
                        del self._memory_cache[key]
                        del self._memory_cache_ttl[key]
        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
        
        return default
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        캐시에 값 저장
        
        Args:
            key (str): 캐시 키
            value (Any): 저장할 값
            ttl (int, optional): TTL (초)
            
        Returns:
            bool: 저장 성공 여부
        """
        if not self.enable_cache:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            
            if self.redis_client:
                # Redis에 저장
                serialized_value = json.dumps(value, ensure_ascii=False)
                return self.redis_client.setex(key, ttl, serialized_value)
            else:
                # 메모리 캐시에 저장
                self._memory_cache[key] = value
                self._memory_cache_ttl[key] = time.time() + ttl
                
                # 캐시 크기 제한
                if len(self._memory_cache) > self.max_size:
                    self._cleanup_memory_cache()
                
                return True
        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        캐시에서 값 삭제
        
        Args:
            key (str): 캐시 키
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    del self._memory_cache_ttl[key]
                    return True
        except Exception as e:
            logger.error(f"Cache delete error for key '{key}': {e}")
        
        return False
    
    def exists(self, key: str) -> bool:
        """
        캐시 키 존재 여부 확인
        
        Args:
            key (str): 캐시 키
            
        Returns:
            bool: 키 존재 여부
        """
        try:
            if self.redis_client:
                return bool(self.redis_client.exists(key))
            else:
                return key in self._memory_cache and time.time() < self._memory_cache_ttl[key]
        except Exception as e:
            logger.error(f"Cache exists error for key '{key}': {e}")
        
        return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """
        캐시 키 TTL 설정
        
        Args:
            key (str): 캐시 키
            ttl (int): TTL (초)
            
        Returns:
            bool: 설정 성공 여부
        """
        try:
            if self.redis_client:
                return bool(self.redis_client.expire(key, ttl))
            else:
                if key in self._memory_cache:
                    self._memory_cache_ttl[key] = time.time() + ttl
                    return True
        except Exception as e:
            logger.error(f"Cache expire error for key '{key}': {e}")
        
        return False
    
    def ttl(self, key: str) -> int:
        """
        캐시 키의 남은 TTL 확인
        
        Args:
            key (str): 캐시 키
            
        Returns:
            int: 남은 TTL (초), -1은 만료되지 않음, -2는 키가 없음
        """
        try:
            if self.redis_client:
                return self.redis_client.ttl(key)
            else:
                if key in self._memory_cache_ttl:
                    remaining = self._memory_cache_ttl[key] - time.time()
                    return max(0, int(remaining))
                return -2
        except Exception as e:
            logger.error(f"Cache TTL error for key '{key}': {e}")
        
        return -2
    
    def clear(self, pattern: str = None) -> int:
        """
        캐시 정리
        
        Args:
            pattern (str, optional): 삭제할 키 패턴
            
        Returns:
            int: 삭제된 키 수
        """
        try:
            if self.redis_client:
                if pattern:
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        return self.redis_client.delete(*keys)
                    return 0
                else:
                    return self.redis_client.flushdb()
            else:
                if pattern:
                    # 패턴 매칭 (간단한 구현)
                    deleted_count = 0
                    keys_to_delete = []
                    for key in self._memory_cache.keys():
                        if pattern.replace('*', '') in key:
                            keys_to_delete.append(key)
                    
                    for key in keys_to_delete:
                        del self._memory_cache[key]
                        del self._memory_cache_ttl[key]
                        deleted_count += 1
                    
                    return deleted_count
                else:
                    deleted_count = len(self._memory_cache)
                    self._memory_cache.clear()
                    self._memory_cache_ttl.clear()
                    return deleted_count
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
        
        return 0
    
    def get_keys(self, pattern: str = "*") -> List[str]:
        """
        패턴에 맞는 키 목록 가져오기
        
        Args:
            pattern (str): 키 패턴
            
        Returns:
            List[str]: 키 목록
        """
        try:
            if self.redis_client:
                return [key.decode('utf-8') for key in self.redis_client.keys(pattern)]
            else:
                # 간단한 패턴 매칭
                keys = []
                for key in self._memory_cache.keys():
                    if pattern.replace('*', '') in key:
                        keys.append(key)
                return keys
        except Exception as e:
            logger.error(f"Cache get_keys error: {e}")
        
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 정보 가져오기
        
        Returns:
            Dict[str, Any]: 캐시 통계
        """
        try:
            if self.redis_client:
                info = self.redis_client.info()
                return {
                    "type": "redis",
                    "connected": True,
                    "keys": info.get("db0", {}).get("keys", 0),
                    "memory_usage": info.get("used_memory_human", "N/A"),
                    "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_misses", 1), 1)
                }
            else:
                return {
                    "type": "memory",
                    "connected": False,
                    "keys": len(self._memory_cache),
                    "memory_usage": "N/A",
                    "max_size": self.max_size
                }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"error": str(e)}
    
    def _cleanup_memory_cache(self):
        """메모리 캐시 정리 (LRU 방식)"""
        if len(self._memory_cache) <= self.max_size:
            return
        
        # TTL이 가장 짧은 항목들 제거
        current_time = time.time()
        expired_keys = []
        
        for key, expiry in self._memory_cache_ttl.items():
            if current_time >= expiry:
                expired_keys.append(key)
        
        # 만료된 키들 제거
        for key in expired_keys:
            del self._memory_cache[key]
            del self._memory_cache_ttl[key]
        
        # 여전히 크기가 크면 가장 오래된 항목들 제거
        if len(self._memory_cache) > self.max_size:
            sorted_keys = sorted(
                self._memory_cache_ttl.items(),
                key=lambda x: x[1]
            )
            
            keys_to_remove = len(self._memory_cache) - self.max_size
            for key, _ in sorted_keys[:keys_to_remove]:
                del self._memory_cache[key]
                del self._memory_cache_ttl[key]
    
    def health_check(self) -> Dict[str, Any]:
        """
        캐시 상태 확인
        
        Returns:
            Dict[str, Any]: 상태 정보
        """
        try:
            if self.redis_client:
                self.redis_client.ping()
                return {
                    "status": "healthy",
                    "type": "redis",
                    "connected": True
                }
            else:
                return {
                    "status": "healthy",
                    "type": "memory",
                    "connected": False
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "type": "redis" if self.redis_client else "memory"
            }
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        if self.redis_client:
            self.redis_client.close()

# 전역 캐시 인스턴스
cache_manager = CacheManager() 