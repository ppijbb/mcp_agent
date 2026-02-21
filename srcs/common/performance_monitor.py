"""
Performance Monitoring Utilities

Simple performance monitoring for key functions without complex dependencies.
"""

import asyncio
import time
import functools
from typing import Dict, Any, Optional, Callable
from collections import defaultdict, deque
import threading


class PerformanceMonitor:
    """
    Simple performance monitoring utility for tracking function execution times.
    """
    
    def __init__(self, max_history: int = 1000):
        """Initialize the performance monitor."""
        self.max_history = max_history
        self._timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def record_timing(self, func_name: str, duration: float) -> None:
        """Record a function execution timing."""
        with self._lock:
            self._timings[func_name].append(duration)
            self._counts[func_name] += 1
    
    def get_stats(self, func_name: str) -> Optional[Dict[str, Any]]:
        """Get performance statistics for a specific function."""
        with self._lock:
            if func_name not in self._timings or not self._timings[func_name]:
                return None
            
            timings = list(self._timings[func_name])
            return {
                "count": self._counts[func_name],
                "avg": sum(timings) / len(timings),
                "min": min(timings),
                "max": max(timings),
                "total": sum(timings)
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all tracked functions."""
        return {name: self.get_stats(name) for name in self._timings.keys()}
    
    def clear_stats(self, func_name: Optional[str] = None) -> None:
        """Clear statistics for a function or all functions."""
        with self._lock:
            if func_name:
                self._timings[func_name].clear()
                self._counts[func_name] = 0
            else:
                self._timings.clear()
                self._counts.clear()


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def monitor_performance(func_name: Optional[str] = None, monitor: Optional[PerformanceMonitor] = None):
    """
    Decorator to monitor function performance.
    
    Args:
        func_name: Name to use for tracking (defaults to function name)
        monitor: Performance monitor instance (defaults to global monitor)
    """
    def decorator(func: Callable) -> Callable:
        name = func_name or f"{func.__module__}.{func.__qualname__}"
        perf_monitor = monitor or _global_monitor
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                perf_monitor.record_timing(name, duration)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                perf_monitor.record_timing(name, duration)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def get_performance_stats(func_name: Optional[str] = None) -> Dict[str, Any]:
    """Get performance statistics from the global monitor."""
    if func_name:
        return _global_monitor.get_stats(func_name) or {}
    return _global_monitor.get_all_stats()


def clear_performance_stats(func_name: Optional[str] = None) -> None:
    """Clear performance statistics from the global monitor."""
    _global_monitor.clear_stats(func_name)