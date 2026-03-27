"""
Performance Optimization Module

This module provides caching and performance measurement utilities for the
hobby starter pack agent system, including decorators for timing and caching.

Classes:
    CacheManager: TTL-based cache manager for storing computed values
    PerformanceOptimizer: Performance measurement and caching utilities

Example:
    >>> optimizer = PerformanceOptimizer()
    >>> cached_func = optimizer.cached(ttl=3600)(my_function)
"""

import asyncio
import functools
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, TypeVar, Awaitable

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheManager:
    """
    TTL-based cache manager for storing computed values.
    
    Provides simple in-memory caching with time-to-live (TTL) support.
    Automatically evicts expired entries on access.
    
    Attributes:
        cache: Internal dictionary storing cached values
        default_ttl: Default time-to-live in seconds
    
    Example:
        >>> cache = CacheManager(default_ttl=3600)
        >>> cache.set("key", "value", ttl=1800)
        >>> cache.get("key")
        'value'
    """
    
    def __init__(self, default_ttl: int = 3600):
        """
        Initialize the CacheManager.
        
        Args:
            default_ttl: Default TTL in seconds for cached entries
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from cache.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        if datetime.now() > entry["expires_at"]:
            del self.cache[key]
            return None
        
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store a value in cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: TTL in seconds (uses default if None)
        """
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": datetime.now(),
        }
    
    def delete(self, key: str) -> None:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key to delete
        """
        if key in self.cache:
            del self.cache[key]
    
    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()
    
    def generate_key(self, prefix: str, *args: Any, **kwargs: Any) -> str:
        """
        Generate a cache key from prefix and arguments.
        
        Args:
            prefix: Key prefix for namespacing
            *args: Positional arguments to include in key
            **kwargs: Keyword arguments to include in key
            
        Returns:
            Generated cache key string
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
    """
    Performance measurement and caching utilities.
    
    Provides decorators for measuring execution time and caching function
    results. Tracks metrics for performance analysis.
    
    Attributes:
        metrics: Dictionary storing execution time metrics
        cache_manager: CacheManager instance for caching
    
    Example:
        >>> optimizer = PerformanceOptimizer()
        >>> 
        >>> @optimizer.cached(ttl=3600)
        ... def expensive_function(x):
        ...     return x * 2
        >>> 
        >>> @optimizer.measure_time("my_function")
        ... def my_function():
        ...     pass
    """
    
    def __init__(self):
        """Initialize the PerformanceOptimizer."""
        self.metrics: Dict[str, list] = {}
        self.cache_manager = CacheManager()
    
    def measure_time(self, func_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Create a decorator that measures and logs execution time.
        
        Args:
            func_name: Name to use for logging metrics
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
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
    
    def measure_time_async(self, func_name: str):
        """
        Create an async decorator that measures and logs execution time.
        
        Args:
            func_name: Name to use for logging metrics
            
        Returns:
            Decorator function for async functions
        """
        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
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
        """
        Get performance statistics for all measured functions.
        
        Returns:
            Dictionary with function names as keys and stats as values,
            including count, average, min, max, and total time
        """
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
    
    def cached(self, ttl: Optional[int] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Create a caching decorator with TTL support.
        
        Args:
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            Decorator function that caches function results
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                cache_key = self.cache_manager.generate_key(
                    func.__name__,
                    *args,
                    **kwargs
                )
                
                cached_value = self.cache_manager.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value
                
                result = func(*args, **kwargs)
                self.cache_manager.set(cache_key, result, ttl)
                logger.debug(f"Cache miss for {func.__name__}, cached result")
                return result
            return wrapper
        return decorator
    
    def cached_async(self, ttl: Optional[int] = None):
        """
        Create an async caching decorator with TTL support.
        
        Args:
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            Decorator function for async functions that caches results
        """
        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                cache_key = self.cache_manager.generate_key(
                    func.__name__,
                    *args,
                    **kwargs
                )
                
                cached_value = self.cache_manager.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value
                
                result = await func(*args, **kwargs)
                self.cache_manager.set(cache_key, result, ttl)
                logger.debug(f"Cache miss for {func.__name__}, cached result")
                return result
            return wrapper
        return decorator

