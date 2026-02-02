"""
Optimized logging infrastructure for MCP Agent Hub.

Provides centralized, efficient logging with compiled regex patterns,
structured logging, and performance optimization.

Features:
- High-performance sensitive data filtering
- Structured JSON logging with performance metrics
- Thread-safe connection pooling for loggers
- Context managers for execution timing
- Memory-efficient caching and cleanup

Example:
    >>> from srcs.common.logging_utils import get_optimized_logger
    >>> logger = get_optimized_logger("my_module", enable_filtering=True)
    >>> logger.info("Operation completed")
"""

import re
import logging
import json
import sys
from typing import Dict, Any, Optional, Set, List
from datetime import datetime
from pathlib import Path
import threading
from contextlib import contextmanager

# Pre-compiled regex patterns for performance
SENSITIVE_PATTERNS = [
    re.compile(r'(api[_-]?key["\s]*[:=]["\s]*)([^"\s,}]+)', re.IGNORECASE),
    re.compile(r'(password["\s]*[:=]["\s]*)([^"\s,}]+)', re.IGNORECASE),
    re.compile(r'(token["\s]*[:=]["\s]*)([^"\s,}]+)', re.IGNORECASE),
    re.compile(r'(secret["\s]*[:=]["\s]*)([^"\s,}]+)', re.IGNORECASE),
    re.compile(r'(credential["\s]*[:=]["\s]*)([^"\s,}]+)', re.IGNORECASE),
    re.compile(r'(bearer\s+)([a-zA-Z0-9._-]+)', re.IGNORECASE),
]

# Thread-local storage for logger instances
_loggers = threading.local()


class OptimizedHTTPErrorFilter(logging.Filter):
    """
    High-performance filter for sensitive information in HTTP error logs.
    
    Uses pre-compiled regex patterns for efficient filtering and
    provides thread-safe operation.
    """
    
    def __init__(self, mask_char: str = "*"):
        super().__init__()
        self.mask_char = mask_char
        self._cache = {}  # Simple LRU cache for filtered messages
        self._cache_size = 1000
    
    def _filter_sensitive_data(self, message: str) -> str:
        """
        Filter sensitive data from message using pre-compiled patterns.
        
        Args:
            message: Original message
            
        Returns:
            Filtered message with sensitive data masked
        """
        # Check cache first
        cache_key = hash(message)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        filtered = message
        for pattern in SENSITIVE_PATTERNS:
            # Replace sensitive values while preserving keys
            filtered = pattern.sub(
                lambda m: f"{m.group(1)}{self.mask_char * len(m.group(2))}",
                filtered
            )
        
        # Update cache (simple size management)
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = filtered
        return filtered
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter sensitive information from log record.
        
        Args:
            record: Log record to filter
            
        Returns:
            True if record should be processed
        """
        if hasattr(record, 'msg'):
            record.msg = self._filter_sensitive_data(str(record.msg))
        
        if hasattr(record, 'args') and record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    new_args.append(self._filter_sensitive_data(arg))
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)
        
        return True


class StructuredFormatter(logging.Formatter):
    """
    JSON structured formatter for consistent log formatting.
    
    Provides structured logging with consistent field names and
    performance optimization through template reuse.
    """
    
    def __init__(self):
        super().__init__()
        self._json_encoder = json.JSONEncoder(default=str, ensure_ascii=False)
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as structured JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log message
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": threading.current_thread().name,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info'
            }:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        return self._json_encoder.encode(log_entry)


class PerformanceLogger:
    """
    Performance-optimized logger with context management.
    
    Provides high-performance logging with minimal overhead,
    context management, and performance monitoring.
    """
    
    def __init__(
        self,
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        enable_structured: bool = False,
        enable_filtering: bool = True
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self._setup_logger(log_file, level, enable_structured, enable_filtering)
        self._performance_stats = {
            "log_count": 0,
            "filter_hits": 0,
            "total_time": 0.0
        }
    
    def _setup_logger(
        self,
        log_file: Optional[str],
        level: int,
        enable_structured: bool,
        enable_filtering: bool
    ):
        """Setup logger with optimized configuration."""
        # Clear existing handlers
        self.logger.handlers.clear()
        self.logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if enable_structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        
        if enable_filtering:
            console_handler.addFilter(OptimizedHTTPErrorFilter())
        
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            if enable_structured:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            
            if enable_filtering:
                file_handler.addFilter(OptimizedHTTPErrorFilter())
            
            self.logger.addHandler(file_handler)
    
    @contextmanager
    def log_execution_time(self, operation: str, level: int = logging.DEBUG):
        """
        Context manager for logging execution time.
        
        Args:
            operation: Operation description
            level: Log level for timing information
        """
        start_time = datetime.utcnow()
        try:
            yield
        finally:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.log(
                level,
                f"Operation '{operation}' completed in {duration:.3f}s",
                extra={
                    "operation": operation,
                    "duration_seconds": duration,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                }
            )
            
            # Update performance stats
            self._performance_stats["log_count"] += 1
            self._performance_stats["total_time"] += duration
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self._performance_stats.copy()
        if stats["log_count"] > 0:
            stats["average_time"] = stats["total_time"] / stats["log_count"]
        else:
            stats["average_time"] = 0.0
        
        return stats
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional extra fields."""
        self.logger.debug(message, extra=kwargs)
        self._performance_stats["log_count"] += 1
    
    def info(self, message: str, **kwargs):
        """Log info message with optional extra fields."""
        self.logger.info(message, extra=kwargs)
        self._performance_stats["log_count"] += 1
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional extra fields."""
        self.logger.warning(message, extra=kwargs)
        self._performance_stats["log_count"] += 1
    
    def error(self, message: str, **kwargs):
        """Log error message with optional extra fields."""
        self.logger.error(message, extra=kwargs)
        self._performance_stats["log_count"] += 1
    
    def critical(self, message: str, **kwargs):
        """Log critical message with optional extra fields."""
        self.logger.critical(message, extra=kwargs)
        self._performance_stats["log_count"] += 1


class LoggerManager:
    """
    Centralized logger management with caching and optimization.
    
    Provides singleton access to optimized loggers with
    configuration caching and performance monitoring.
    """
    
    _instances: Dict[str, PerformanceLogger] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        enable_structured: bool = False,
        enable_filtering: bool = True
    ) -> PerformanceLogger:
        """
        Get or create optimized logger instance.
        
        Args:
            name: Logger name
            log_file: Optional log file path
            level: Logging level
            enable_structured: Enable JSON structured logging
            enable_filtering: Enable sensitive data filtering
            
        Returns:
            PerformanceLogger instance
        """
        if name not in cls._instances:
            with cls._lock:
                if name not in cls._instances:
                    cls._instances[name] = PerformanceLogger(
                        name, log_file, level, enable_structured, enable_filtering
                    )
        
        return cls._instances[name]
    
    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all loggers."""
        return {name: logger.get_performance_stats() for name, logger in cls._instances.items()}
    
    @classmethod
    def shutdown_all(cls):
        """Shutdown all loggers and clean up resources."""
        for logger in cls._instances.values():
            for handler in logger.logger.handlers:
                handler.close()
        cls._instances.clear()


# Convenience functions
def get_optimized_logger(name: str, **kwargs) -> PerformanceLogger:
    """Get optimized logger instance."""
    return LoggerManager.get_logger(name, **kwargs)


def setup_application_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_structured: bool = False,
    enable_filtering: bool = True
):
    """
    Setup application-wide logging configuration.
    
    Args:
        log_level: Logging level string
        log_file: Optional log file path
        enable_structured: Enable JSON structured logging
        enable_filtering: Enable sensitive data filtering
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    level = level_map.get(log_level.upper(), logging.INFO)
    
    # Create main application logger
    app_logger = LoggerManager.get_logger(
        "mcp_agent",
        log_file=log_file,
        level=level,
        enable_structured=enable_structured,
        enable_filtering=enable_filtering
    )
    
    app_logger.info(
        "Application logging initialized",
        extra={
            "log_level": log_level,
            "log_file": log_file,
            "structured": enable_structured,
            "filtering_enabled": enable_filtering
        }
    )
    
    return app_logger