"""
Performance Monitoring Utilities

Simple performance monitoring for key functions without complex dependencies.
"""

import time
import functools
from typing import Dict, Any, Optional, Callable
from collections import defaultdict, deque
import threading


class PerformanceMonitor:
    """
    Simple performance monitoring utility for tracking function execution times.
    
    A lightweight, thread-safe performance monitoring system that tracks execution
    metrics for functions without external dependencies. Uses deques for efficient
    memory management and provides statistical analysis of timing data.
    
    Attributes:
        max_history: Maximum number of timing records to keep per function
        _timings: Thread-safe storage of execution timing data
        _counts: Thread-safe execution count tracking
        _lock: Threading lock for thread-safe operations
        
    Example:
        monitor = PerformanceMonitor(max_history=500)
        
        @monitor_performance(monitor=monitor)
        def my_function():
            pass
            
        # Get statistics
        stats = monitor.get_stats("my_function")
        print(f"Average: {stats['avg']:.3f}s, Count: {stats['count']}")
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor with configurable history size.
        
        Args:
            max_history: Maximum number of timing records to retain per function.
                        Uses deque with maxlen for automatic memory management.
        """
        self.max_history = max_history
        self._timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def record_timing(self, func_name: str, duration: float) -> None:
        """
        Record a function execution timing in a thread-safe manner.
        
        Args:
            func_name: Name of the function being tracked
            duration: Execution time in seconds
            
        Note:
            Thread-safe - can be called from multiple threads concurrently.
            Old records are automatically discarded when max_history is exceeded.
        """
        with self._lock:
            self._timings[func_name].append(duration)
            self._counts[func_name] += 1
    
    def get_stats(self, func_name: str) -> Optional[Dict[str, Any]]:
        """
        Get performance statistics for a specific function.
        
        Args:
            func_name: Name of the function to retrieve stats for
            
        Returns:
            Dictionary containing statistics or None if no data exists:
            - count: Total number of executions
            - avg: Average execution time
            - min: Minimum execution time
            - max: Maximum execution time  
            - total: Total execution time
            
        Note:
            Returns None if the function has not been tracked.
        """
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
    
def get_all_stats(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get performance statistics for all tracked functions.
        
        Returns:
            Dictionary mapping function names to their statistics dictionaries.
            Functions with no recorded timings will return None values.
            
        Note:
            This method creates a snapshot of current statistics.
        """
        return {name: self.get_stats(name) for name in self._timings.keys()}
    
    def clear_stats(self, func_name: Optional[str] = None) -> None:
        """
        Clear statistics for a specific function or all functions.
        
        Args:
            func_name: Function name to clear, or None to clear all functions
            
        Note:
            Thread-safe operation. Use with caution as it permanently deletes data.
        """
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
        return _global_monitor.get_stats(func_name)
    return _global_monitor.get_all_stats()


def clear_performance_stats(func_name: Optional[str] = None) -> None:
    """Clear performance statistics from the global monitor."""
    _global_monitor.clear_stats(func_name)


# Import asyncio for function type checking
import asyncio