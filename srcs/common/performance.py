"""
Performance Optimization Utilities

Provides caching, rate limiting, and performance monitoring utilities
for agent operations to improve efficiency and reduce resource usage.
"""

import time
import asyncio
import threading
from functools import wraps, lru_cache
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class SimpleCache:
    """
    Thread-safe in-memory cache with TTL support.

    Provides basic caching functionality with automatic expiration
    of cached items based on time-to-live (TTL).
    
    Uses threading.Lock for both sync and async compatibility.
    """

    def __init__(self, max_size: int = 100, default_ttl: int = 300):
        """
        Initialize the cache with size and TTL limits.

        Args:
            max_size: Maximum number of items to store in cache
            default_ttl: Default time-to-live in seconds for cached items
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.Lock()  # Use threading.Lock for sync/async compatibility

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache if it exists and hasn't expired.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value if valid, None otherwise
        """
        with self._lock:
            if key in self._cache:
                item = self._cache[key]
                if time.time() < item['expires']:
                    return item['value']
                else:
                    # Remove expired item
                    del self._cache[key]
        return None

    async def aget(self, key: str) -> Optional[Any]:
        """
        Async retrieve item from cache if it exists and hasn't expired.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value if valid, None otherwise
        """
        return self.get(key)  # Use sync method with thread-safe lock

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store item in cache with TTL.

        Args:
            key: Cache key for the item
            value: Value to cache
            ttl: Custom TTL in seconds, uses default if not provided
        """
        with self._lock:
            # Remove oldest item if cache is full
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache.keys(),
                              key=lambda k: self._cache[k]['created'])
                del self._cache[oldest_key]

            ttl = ttl or self._default_ttl
            self._cache[key] = {
                'value': value,
                'created': time.time(),
                'expires': time.time() + ttl
            }

    async def aset(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Async store item in cache with TTL.

        Args:
            key: Cache key for the item
            value: Value to cache
            ttl: Custom TTL in seconds, uses default if not provided
        """
        self.set(key, value, ttl)  # Use sync method with thread-safe lock

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()

    async def aclear(self) -> None:
        """Async clear all cached items."""
        self.clear()  # Use sync method with thread-safe lock


def rate_limit(calls_per_second: float = 1.0):
    """
    Decorator to rate limit function calls.

    Args:
        calls_per_second: Maximum number of calls allowed per second

    Returns:
        Decorated function with rate limiting

    Example:
        @rate_limit(calls_per_second=0.5)  # Max 1 call every 2 seconds
        async def api_call():
            return await make_request()
    """
    min_interval = 1.0 / calls_per_second
    
    def decorator(func: Callable) -> Callable:
        last_call_time = [0.0]  # Use list to allow modification in closure
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            elapsed = current_time - last_call_time[0]
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)
            
            last_call_time[0] = time.time()
            return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_time = time.time()
            elapsed = current_time - last_call_time[0]
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                await asyncio.sleep(sleep_time)
            
            last_call_time[0] = time.time()
            return await func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def performance_monitor(func: Callable = None, *, log_calls: bool = True):
    """
    Decorator to monitor function performance.
    
    Args:
        log_calls: Whether to log function calls and timing
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                execution_time = time.time() - start_time
                if log_calls:
                    logger.info(f"{f.__name__} executed in {execution_time:.4f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                if log_calls:
                    logger.error(f"{f.__name__} failed after {execution_time:.4f}s: {e}")
                raise
        
        @wraps(f)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await f(*args, **kwargs)
                execution_time = time.time() - start_time
                if log_calls:
                    logger.info(f"{f.__name__} executed in {execution_time:.4f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                if log_calls:
                    logger.error(f"{f.__name__} failed after {execution_time:.4f}s: {e}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(f):
            return async_wrapper
        else:
            return sync_wrapper
    
    # Handle both @performance_monitor and @performance_monitor() usage
    if func is None:
        return decorator
    else:
        return decorator(func)


def memoize_strict(maxsize: int = 128, ttl: Optional[int] = None):
    """
    Strict memoization with optional TTL.
    
    Args:
        maxsize: Maximum cache size
        ttl: Time-to-live in seconds (None for no expiry)
    """
    cache: Dict[str, Dict[str, Any]] = {}
    keys_order = []
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check cache
            if key in cache:
                entry = cache[key]
                if ttl is None or (time.time() - entry['timestamp']) < ttl:
                    return entry['result']
                else:
                    # Remove expired entry
                    cache.pop(key)
                    if key in keys_order:
                        keys_order.remove(key)
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            if len(cache) >= maxsize:
                # Remove oldest entry
                oldest_key = keys_order.pop(0)
                cache.pop(oldest_key)
            
            cache[key] = {
                'result': result,
                'timestamp': time.time()
            }
            keys_order.append(key)
            
            return result
        
        return wrapper
    
    return decorator


class ResourceMonitor:
    """
    Monitor system resources for performance optimization.
    """
    
    def __init__(self):
        """Initialize resource monitor."""
        self.start_time = time.time()
        self.call_counts: Dict[str, int] = {}
        self.execution_times: Dict[str, list] = {}
    
    def record_call(self, func_name: str, execution_time: float):
        """
        Record a function call for monitoring.
        
        Args:
            func_name: Name of the function
            execution_time: Execution time in seconds
        """
        self.call_counts[func_name] = self.call_counts.get(func_name, 0) + 1
        if func_name not in self.execution_times:
            self.execution_times[func_name] = []
        self.execution_times[func_name].append(execution_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource monitoring statistics."""
        stats = {
            'uptime': time.time() - self.start_time,
            'call_counts': self.call_counts.copy(),
            'performance': {}
        }
        
        for func_name, times in self.execution_times.items():
            if times:
                stats['performance'][func_name] = {
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_calls': len(times)
                }
        
        return stats


# Global cache instance
default_cache = SimpleCache()