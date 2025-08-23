"""
통합 캐싱 시스템
Table Game Mate의 성능 최적화를 위한 다층 캐싱 관리
"""
import os
import asyncio
import json
import hashlib
import pickle
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import OrderedDict
import logging

from ..utils.logger import get_logger


class CacheLevel(Enum):
    """캐시 레벨"""
    MEMORY = "memory"       # 메모리 캐시 (가장 빠름)
    DISK = "disk"           # 디스크 캐시 (중간 속도)
    REMOTE = "remote"       # 원격 캐시 (느림, 공유 가능)


class CachePolicy(Enum):
    """캐시 정책"""
    LRU = "lru"             # Least Recently Used
    LFU = "lfu"             # Least Frequently Used
    FIFO = "fifo"           # First In First Out
    TTL = "ttl"             # Time To Live


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_access(self):
        """접근 정보 업데이트"""
        self.accessed_at = datetime.now()
        self.access_count += 1
    
    def is_expired(self, ttl_seconds: Optional[int] = None) -> bool:
        """만료 여부 확인"""
        if ttl_seconds is None:
            return False
        
        expiry_time = self.created_at + timedelta(seconds=ttl_seconds)
        return datetime.now() > expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata
        }


class MemoryCache:
    """메모리 캐시 구현"""
    
    def __init__(self, max_size: int = 1000, policy: CachePolicy = CachePolicy.LRU):
        self.max_size = max_size
        self.policy = policy
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.logger = get_logger("memory_cache")
        
        # 정책별 정렬 함수
        self._sort_functions = {
            CachePolicy.LRU: lambda x: x.accessed_at,
            CachePolicy.LFU: lambda x: x.access_count,
            CachePolicy.FIFO: lambda x: x.created_at
        }
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        if key in self.cache:
            entry = self.cache[key]
            entry.update_access()
            
            # LRU 정책일 때 접근된 항목을 맨 뒤로 이동
            if self.policy == CachePolicy.LRU:
                self.cache.move_to_end(key)
            
            return entry.value
        
        return None
    
    def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """캐시에 값 저장"""
        try:
            # 크기 제한 확인
            if len(self.cache) >= self.max_size:
                self._evict_entries()
            
            # 새 엔트리 생성
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                size_bytes=self._calculate_size(value),
                metadata=metadata or {}
            )
            
            self.cache[key] = entry
            
            # LRU 정책일 때 새 항목을 맨 뒤로 이동
            if self.policy == CachePolicy.LRU:
                self.cache.move_to_end(key)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set cache entry: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """캐시에서 항목 삭제"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self):
        """캐시 전체 삭제"""
        self.cache.clear()
    
    def _evict_entries(self):
        """캐시 항목 제거 (정책에 따라)"""
        if not self.cache:
            return
        
        # 정책에 따른 정렬
        sort_func = self._sort_functions.get(self.policy, lambda x: x.accessed_at)
        sorted_entries = sorted(self.cache.items(), key=lambda x: sort_func(x[1]))
        
        # 가장 낮은 우선순위 항목들 제거 (25% 제거)
        evict_count = max(1, len(sorted_entries) // 4)
        
        for i in range(evict_count):
            key = sorted_entries[i][0]
            del self.cache[key]
    
    def _calculate_size(self, value: Any) -> int:
        """값의 크기 계산 (바이트)"""
        try:
            return len(pickle.dumps(value))
        except:
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "policy": self.policy.value,
            "total_size_bytes": sum(entry.size_bytes for entry in self.cache.values()),
            "avg_access_count": sum(entry.access_count for entry in self.cache.values()) / len(self.cache) if self.cache else 0
        }


class DiskCache:
    """디스크 캐시 구현"""
    
    def __init__(self, cache_dir: str = "./cache", max_size_mb: int = 100):
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self.logger = get_logger("disk_cache")
        
        # 캐시 디렉토리 생성
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    def get(self, key: str) -> Optional[Any]:
        """디스크에서 값 조회"""
        try:
            cache_file = self._get_cache_file_path(key)
            
            if not os.path.exists(cache_file):
                return None
            
            # 파일 읽기
            with open(cache_file, 'rb') as f:
                entry_data = pickle.load(f)
            
            # 만료 확인
            if entry_data.get("expires_at") and datetime.now() > entry_data["expires_at"]:
                self.delete(key)
                return None
            
            # 접근 시간 업데이트
            entry_data["accessed_at"] = datetime.now()
            with open(cache_file, 'wb') as f:
                pickle.dump(entry_data, f)
            
            return entry_data["value"]
            
        except Exception as e:
            self.logger.error(f"Failed to read from disk cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """디스크에 값 저장"""
        try:
            cache_file = self._get_cache_file_path(key)
            
            # 캐시 크기 확인 및 정리
            self._check_cache_size()
            
            # 엔트리 데이터 생성
            entry_data = {
                "key": key,
                "value": value,
                "created_at": datetime.now(),
                "accessed_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(seconds=ttl_seconds) if ttl_seconds else None,
                "metadata": metadata or {}
            }
            
            # 파일에 저장
            with open(cache_file, 'wb') as f:
                pickle.dump(entry_data, f)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write to disk cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """디스크에서 항목 삭제"""
        try:
            cache_file = self._get_cache_file_path(key)
            
            if os.path.exists(cache_file):
                os.remove(cache_file)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete from disk cache: {e}")
            return False
    
    def clear(self):
        """디스크 캐시 전체 삭제"""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to clear disk cache: {e}")
    
    def _get_cache_file_path(self, key: str) -> str:
        """캐시 파일 경로 생성"""
        # 키를 해시하여 파일명 생성
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def _check_cache_size(self):
        """캐시 크기 확인 및 정리"""
        try:
            total_size = 0
            cache_files = []
            
            # 모든 캐시 파일 크기 계산
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    file_path = os.path.join(self.cache_dir, filename)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    total_size += file_size
                    
                    cache_files.append((file_path, file_size, os.path.getmtime(file_path)))
            
            # 크기 제한 초과 시 오래된 파일부터 삭제
            if total_size > self.max_size_mb:
                # 수정 시간 기준으로 정렬 (오래된 것부터)
                cache_files.sort(key=lambda x: x[2])
                
                for file_path, file_size, _ in cache_files:
                    if total_size <= self.max_size_mb:
                        break
                    
                    os.remove(file_path)
                    total_size -= file_size
                    
        except Exception as e:
            self.logger.error(f"Failed to check cache size: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """디스크 캐시 통계"""
        try:
            total_files = 0
            total_size = 0
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    total_files += 1
                    file_path = os.path.join(self.cache_dir, filename)
                    total_size += os.path.getsize(file_path)
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "max_size_mb": self.max_size_mb,
                "cache_dir": self.cache_dir
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get disk cache stats: {e}")
            return {}


class CacheManager:
    """통합 캐시 관리자"""
    
    def __init__(self):
        self.logger = get_logger("cache_manager")
        
        # 캐시 레벨별 인스턴스
        self.memory_cache = MemoryCache(max_size=1000, policy=CachePolicy.LRU)
        self.disk_cache = DiskCache(cache_dir="./cache", max_size_mb=100)
        
        # 캐시 설정
        self.default_ttl = 3600  # 1시간
        self.enable_memory_cache = True
        self.enable_disk_cache = True
        
        # 캐시 히트 통계
        self.cache_stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "disk_hits": 0,
            "disk_misses": 0,
            "total_requests": 0
        }
    
    async def get(
        self, 
        key: str, 
        default: Any = None,
        use_memory: bool = True,
        use_disk: bool = True
    ) -> Any:
        """캐시에서 값 조회 (메모리 → 디스크 순서)"""
        self.cache_stats["total_requests"] += 1
        
        # 메모리 캐시 확인
        if use_memory and self.enable_memory_cache:
            value = self.memory_cache.get(key)
            if value is not None:
                self.cache_stats["memory_hits"] += 1
                return value
            else:
                self.cache_stats["memory_misses"] += 1
        
        # 디스크 캐시 확인
        if use_disk and self.enable_disk_cache:
            value = self.disk_cache.get(key)
            if value is not None:
                self.cache_stats["disk_hits"] += 1
                
                # 메모리 캐시에도 저장 (다음 조회 시 빠른 접근)
                if use_memory and self.enable_memory_cache:
                    self.memory_cache.set(key, value)
                
                return value
            else:
                self.cache_stats["disk_misses"] += 1
        
        return default
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        use_memory: bool = True,
        use_disk: bool = True
    ) -> bool:
        """캐시에 값 저장"""
        success = True
        
        # TTL 설정
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl
        
        # 메모리 캐시에 저장
        if use_memory and self.enable_memory_cache:
            success &= self.memory_cache.set(key, value, metadata)
        
        # 디스크 캐시에 저장
        if use_disk and self.enable_disk_cache:
            success &= self.disk_cache.set(key, value, ttl_seconds, metadata)
        
        return success
    
    async def delete(self, key: str) -> bool:
        """캐시에서 항목 삭제"""
        success = True
        
        if self.enable_memory_cache:
            success &= self.memory_cache.delete(key)
        
        if self.enable_disk_cache:
            success &= self.disk_cache.delete(key)
        
        return success
    
    async def clear(self, level: Optional[CacheLevel] = None):
        """캐시 정리"""
        if level is None or level == CacheLevel.MEMORY:
            self.memory_cache.clear()
        
        if level is None or level == CacheLevel.DISK:
            self.disk_cache.clear()
    
    async def get_or_set(
        self, 
        key: str, 
        default_factory: Callable,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """값이 없으면 생성하여 저장하고 반환"""
        # 캐시에서 조회 시도
        value = await self.get(key)
        
        if value is not None:
            return value
        
        # 값이 없으면 생성
        if asyncio.iscoroutinefunction(default_factory):
            value = await default_factory()
        else:
            value = default_factory()
        
        # 캐시에 저장
        await self.set(key, value, ttl_seconds, metadata)
        
        return value
    
    def get_stats(self) -> Dict[str, Any]:
        """전체 캐시 통계"""
        memory_stats = self.memory_cache.get_stats()
        disk_stats = self.disk_cache.get_stats()
        
        # 히트율 계산
        total_memory = self.cache_stats["memory_hits"] + self.cache_stats["memory_misses"]
        total_disk = self.cache_stats["disk_hits"] + self.cache_stats["disk_misses"]
        
        memory_hit_rate = (self.cache_stats["memory_hits"] / total_memory * 100) if total_memory > 0 else 0
        disk_hit_rate = (self.cache_stats["disk_hits"] / total_disk * 100) if total_disk > 0 else 0
        
        return {
            "cache_stats": self.cache_stats,
            "memory_cache": memory_stats,
            "disk_cache": disk_stats,
            "hit_rates": {
                "memory": memory_hit_rate,
                "disk": disk_hit_rate
            },
            "total_requests": self.cache_stats["total_requests"]
        }
    
    def set_memory_cache_policy(self, policy: CachePolicy):
        """메모리 캐시 정책 변경"""
        self.memory_cache.policy = policy
    
    def set_cache_limits(self, memory_max_size: Optional[int] = None, disk_max_size_mb: Optional[int] = None):
        """캐시 크기 제한 설정"""
        if memory_max_size is not None:
            self.memory_cache.max_size = memory_max_size
        
        if disk_max_size_mb is not None:
            self.disk_cache.max_size_mb = disk_max_size_mb


# 전역 캐시 매니저 인스턴스
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """전역 캐시 매니저 인스턴스 반환"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


# 데코레이터: 자동 캐싱
def cached(
    key_prefix: str = "",
    ttl_seconds: Optional[int] = None,
    use_memory: bool = True,
    use_disk: bool = True
):
    """함수 결과 자동 캐싱 데코레이터"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            
            # 캐시 키 생성
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # 캐시에서 조회 시도
            cached_result = await cache_manager.get(cache_key, use_memory=use_memory, use_disk=use_disk)
            if cached_result is not None:
                return cached_result
            
            # 함수 실행
            result = await func(*args, **kwargs)
            
            # 결과 캐싱
            await cache_manager.set(cache_key, result, ttl_seconds, use_memory=use_memory, use_disk=use_disk)
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            
            # 캐시 키 생성
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # 캐시에서 조회 시도
            cached_result = cache_manager.get(cache_key, use_memory=use_memory, use_disk=use_disk)
            if cached_result is not None:
                return cached_result
            
            # 함수 실행
            result = func(*args, **kwargs)
            
            # 결과 캐싱 (비동기로 실행)
            asyncio.create_task(
                cache_manager.set(cache_key, result, ttl_seconds, use_memory=use_memory, use_disk=use_disk)
            )
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
