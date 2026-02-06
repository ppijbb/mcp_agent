"""
Standardized error handling and logging utilities for MCP Agent Hub.

Provides unified error handling patterns, logging configuration, and
response formatting across the application.

Features:
- Structured error classification with severity and category
- Centralized error handling with decorators
- HTTP error filtering for sensitive data
- Standardized response formatting
- Performance monitoring for error handling

Example:
    >>> from srcs.common.error_handling import handle_errors, ErrorSeverity
    >>> @handle_errors(severity=ErrorSeverity.HIGH)
    ... def risky_operation():
    ...     return "success"
"""

import logging
import traceback
import functools
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from enum import Enum
import json
import sys
from datetime import datetime

# Type variables for decorators
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')


class ErrorSeverity(Enum):
    """
    Error severity levels for categorization and prioritization.
    
    Defines the severity levels used throughout the application to classify
    errors and determine appropriate response strategies. Each severity level
    has different implications for logging, user notification, and recovery actions.
    
    Values:
        LOW: Minor issues that don't affect core functionality
        MEDIUM: Significant issues that may impact user experience
        HIGH: Serious issues that require immediate attention
        CRITICAL: Critical failures that may cause data loss or system instability
        
    Example:
        @handle_errors(severity=ErrorSeverity.HIGH)
        def critical_function():
            pass
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """
    Error categories for better organization and targeted handling.
    
    Groups related error types together to enable category-specific
    error handling strategies, monitoring, and alerting. Each category
    represents a different domain of errors with distinct characteristics.
    
    Categories:
        NETWORK: Network connectivity, timeouts, DNS resolution issues
        API: External API failures, rate limits, authentication issues
        VALIDATION: Input validation failures, malformed data
        PROCESSING: Data processing errors, transformation failures
        SYSTEM: Operating system, filesystem, resource exhaustion
        USER_INPUT: Invalid user input, formatting errors
        
    Example:
        raise APIError("Rate limit exceeded", 
                      category=ErrorCategory.API,
                      severity=ErrorSeverity.HIGH)
    """
    NETWORK = "network"
    API = "api"
    VALIDATION = "validation"
    PROCESSING = "processing"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    CONFIGURATION = "configuration"


class AgentError(Exception):
    """
    Base error class for all agent-related errors.
    
    Provides structured error information with severity, category,
    and additional context for debugging.
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.PROCESSING,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.details = details or {}
        self.original_error = original_error
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/response."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "details": self.details,
            "timestamp": self.timestamp,
            "original_error": str(self.original_error) if self.original_error else None
        }


class ErrorHandler:
    """
    Centralized error handling and logging manager.
    
    Provides consistent error handling patterns, logging configuration,
    and error response formatting.
    """
    
    def __init__(self, logger_name: str = "mcp_agent"):
        self.logger = logging.getLogger(logger_name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure logger with consistent formatting."""
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        include_traceback: bool = True
    ):
        """
        Log error with structured information.
        
        Args:
            error: Exception to log
            context: Additional context information
            include_traceback: Whether to include traceback in logs
        """
        error_info = {
            "error_type": error.__class__.__name__,
            "message": str(error),
            "context": context or {}
        }
        
        if isinstance(error, AgentError):
            error_info.update(error.to_dict())
        
        if include_traceback:
            error_info["traceback"] = traceback.format_exc()
        
        # Determine log level based on severity
        if isinstance(error, AgentError):
            if error.severity == ErrorSeverity.CRITICAL:
                self.logger.critical(json.dumps(error_info, default=str))
            elif error.severity == ErrorSeverity.HIGH:
                self.logger.error(json.dumps(error_info, default=str))
            elif error.severity == ErrorSeverity.MEDIUM:
                self.logger.warning(json.dumps(error_info, default=str))
            else:
                self.logger.info(json.dumps(error_info, default=str))
        else:
            self.logger.error(json.dumps(error_info, default=str))
    
    def handle_agent_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle agent error and return standardized response.
        
        Args:
            error: Exception to handle
            context: Additional context information
            
        Returns:
            Standardized error response dictionary
        """
        self.log_error(error, context)
        
        if isinstance(error, AgentError):
            return {
                "success": False,
                "error": error.to_dict(),
                "context": context or {}
            }
        else:
            # Wrap non-AgentError exceptions
            agent_error = AgentError(
                message=str(error),
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.PROCESSING,
                original_error=error
            )
            return {
                "success": False,
                "error": agent_error.to_dict(),
                "context": context or {}
            }
    
    def create_success_response(
        self,
        data: Any,
        message: str = "Operation completed successfully",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create standardized success response.
        
        Args:
            data: Response data
            message: Success message
            metadata: Additional metadata
            
        Returns:
            Standardized success response
        """
        return {
            "success": True,
            "data": data,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }


# Global error handler instance
default_error_handler = ErrorHandler()


def handle_errors(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.PROCESSING,
    reraise: bool = False,
    return_on_error: Any = None
):
    """
    Decorator for standardized error handling in functions.
    
    Args:
        severity: Error severity level
        category: Error category
        reraise: Whether to reraise exceptions after handling
        return_on_error: Value to return on error (if not reraising)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "args": str(args)[:200],  # Limit length
                    "kwargs": str(kwargs)[:200]  # Limit length
                }
                
                # Create AgentError if not already
                if not isinstance(e, AgentError):
                    agent_error = AgentError(
                        message=str(e),
                        severity=severity,
                        category=category,
                        details={"original_function": func.__name__},
                        original_error=e
                    )
                else:
                    agent_error = e
                
                # Log the error
                default_error_handler.log_error(agent_error, context)
                
                if reraise:
                    raise agent_error
                else:
                    return return_on_error
        
        return wrapper  # type: ignore
    return decorator


def safe_execute(
    func: Callable[[], T],
    default_value: Optional[T] = None,
    error_message: str = "Operation failed",
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
) -> Optional[T]:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        default_value: Default value to return on error
        error_message: Error message for logging
        severity: Error severity level
        
    Returns:
        Function result or default value
    """
    try:
        return func()
    except Exception as e:
        agent_error = AgentError(
            message=error_message,
            severity=severity,
            category=ErrorCategory.PROCESSING,
            original_error=e
        )
        default_error_handler.log_error(agent_error)
        return default_value


def validate_input(data: Any, required_fields: Optional[list] = None) -> Dict[str, Any]:
    """
    Validate input data with common checks.
    
    Args:
        data: Data to validate
        required_fields: List of required field names
        
    Returns:
        Validation result dictionary
    """
    errors = []
    
    if data is None:
        errors.append("Data cannot be None")
    
    if isinstance(data, dict) and required_fields:
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
    
    if errors:
        raise AgentError(
            message="Validation failed",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            details={"validation_errors": errors}
        )
    
    return {"valid": True, "data": data}


# HTTPErrorFilter for filtering sensitive information from logs
class HTTPErrorFilter(logging.Filter):
    """Filter to hide sensitive information in HTTP error logs."""
    
    SENSITIVE_PATTERNS = [
        "api_key", "password", "token", "secret", "credential"
    ]
    
    def filter(self, record):
        """Filter sensitive information from log records."""
        if hasattr(record, 'msg') and record.msg:
            msg = str(record.msg)
            for pattern in self.SENSITIVE_PATTERNS:
                if pattern.lower() in msg.lower():
                    record.msg = "[FILTERED: Contains sensitive information]"
                    break
        return True


def setup_filtered_logger(name: str) -> logging.Logger:
    """
    Create a logger with HTTPErrorFilter applied.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger with filter
    """
    logger = logging.getLogger(name)
    if not any(isinstance(f, HTTPErrorFilter) for f in logger.filters):
        logger.addFilter(HTTPErrorFilter())
    return logger