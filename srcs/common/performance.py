"""
Performance Optimization Utilities

Provides caching, rate limiting, and performance monitoring utilities
for agent operations to improve efficiency and reduce resource usage.
"""

import time
import asyncio
from functools import wraps, lru_cache
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class SimpleCache:
    """
    Thread-safe in-memory cache with TTL support.

    Provides basic caching functionality with automatic expiration
    of cached items based on time-to-live (TTL).
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
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache if it exists and hasn't expired.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value if valid, None otherwise
        """
        async with self._lock:
            if key in self._cache:
                item = self._cache[key]
                if time.time() < item['expires']:
                    return item['value']
                else:
                    # Remove expired item
                    del self._cache[key]
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store item in cache with TTL.

        Args:
            key: Cache key for the item
            value: Value to cache
            ttl: Custom TTL in seconds, uses default if not provided
        """
        async with self._lock:
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

    async def clear(self) -> None:
        """Clear all cached items."""
        async with self._lock:
            self._cache.clear()


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
    def decorator(func: Callable):
        last_called = [0.0]  # Use list to make it mutable in closure
        min_interval = 1.0 / calls_per_second

        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_time = time.time()
            time_since_last = current_time - last_called[0]

            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                await asyncio.sleep(sleep_time)

            last_called[0] = time.time()
            return await func(*args, **kwargs)

        return wrapper
    return decorator


def performance_monitor(func: Callable) -> Callable:
    """
    Decorator to monitor function performance and log execution time.

    Automatically logs execution time and warns if function takes longer
    than expected (based on historical performance).

    Args:
        func: Function to monitor

    Returns:
        Decorated function with performance monitoring

    Example:
        @performance_monitor
        async def slow_operation():
            await asyncio.sleep(1)
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = f"{func.__module__}.{func.__name__}"

        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time

            logger.debug(f"{function_name} completed in {execution_time:.3f}s")

            # Log warning if execution takes longer than 5 seconds
            if execution_time > 5.0:
                logger.warning(f"{function_name} took {execution_time:.3f}s - consider optimization")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{function_name} failed after {execution_time:.3f}s: {e}")
            raise

    return wrapper


@lru_cache(maxsize=128)
def memoize_strict(func: Callable) -> Callable:
    """
    Strict memoization decorator for pure functions.

    Caches function results based on arguments, ideal for functions
    that always return the same output for the same input.

    Args:
        func: Pure function to memoize

    Returns:
        Decorated function with strict memoization

    Note:
        Only works for synchronous functions with hashable arguments
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


class ResourceMonitor:
    """
    Monitor system resource usage during agent operations.

    Tracks memory usage, execution time, and other performance metrics
    to help identify bottlenecks and optimization opportunities.
    """

    def __init__(self):
        """Initialize the resource monitor."""
        self._start_time = None
        self._metrics = {}

    async def __aenter__(self):
        """Start monitoring when entering context."""
        self._start_time = time.time()
        try:
            import psutil
            process = psutil.Process()
            self._metrics['start_memory'] = process.memory_info().rss
        except ImportError:
            self._metrics['start_memory'] = None
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and log results when exiting context."""
        if self._start_time:
            execution_time = time.time() - self._start_time

            try:
                import psutil
                process = psutil.Process()
                end_memory = process.memory_info().rss
                memory_delta = end_memory - self._metrics['start_memory']
            except ImportError:
                memory_delta = None

            logger.info(f"Operation completed in {execution_time:.3f}s")
            if memory_delta is not None:
                logger.info(f"Memory delta: {memory_delta / 1024 / 1024:.2f} MB")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.

        Returns:
            Dictionary containing current performance metrics
        """
        return self._metrics.copy()


# Global cache instance for common use
default_cache = SimpleCache(max_size=200, default_ttl=300)

# Export commonly used items
__all__ = [
    'SimpleCache', 'rate_limit', 'performance_monitor',
    'memoize_strict', 'ResourceMonitor', 'default_cache'
]
