"""
Error definitions and utilities for MCP Agent Hub.

Provides custom exception classes for different error types encountered
in the MCP Agent system, along with utility functions for safe execution
and error handling.

Classes:
    MCPError: Base exception for all MCP-Agent errors
    ConfigError: Configuration-related errors
    APIError: External API errors
    WorkflowError: Workflow-related errors
    CircuitBreakerOpen: Circuit breaker state errors
    EncryptionError: Encryption/decryption errors
    ValidationError: Input validation errors
    SecurityError: Security-related errors
"""

from typing import Dict, Any, Optional, Callable


class MCPError(Exception):
    """Base exception class for all MCP-Agent errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.message = message
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigError(MCPError):
    """Raised for errors in configuration."""

    def __str__(self):
        return f"ConfigError: {self.message}"


class APIError(MCPError):
    """Raised for errors related to external APIs."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response = response or {}

    def __str__(self):
        if self.status_code:
            return f"APIError [{self.status_code}]: {self.message}"
        return f"APIError: {self.message}"


class WorkflowError(MCPError):
    """Workflow-related errors."""

    def __str__(self):
        return f"WorkflowError: {self.message}"


class CircuitBreakerOpen(MCPError):
    """Raised when the circuit breaker is open."""

    def __str__(self):
        return f"CircuitBreakerOpen: {self.message}"


class EncryptionError(MCPError):
    """Raised for errors during encryption or decryption."""

    def __str__(self):
        return f"EncryptionError: {self.message}"


class ValidationError(MCPError):
    """Raised for input validation errors."""

    def __str__(self):
        return f"ValidationError: {self.message}"


class SecurityError(MCPError):
    """Raised for security-related errors."""

    def __str__(self):
        return f"SecurityError: {self.message}"


def safe_execute(func: Callable, default: Any = None, error_type: type = MCPError, *args, **kwargs) -> Any:
    """
    Safely execute a function with standardized error handling.
    
    Args:
        func: Function to execute
        default: Default value to return on error
        error_type: Type of error to raise on failure (default: MCPError)
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default value if error occurs
        
    Raises:
        error_type: If error occurs and default is None
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if default is not None:
            return default
        raise error_type(f"Failed to execute {func.__name__}: {str(e)}")



def validate_input(value: Any, field_name: str, required: bool = True, 
                  value_type: Optional[type] = None, min_length: Optional[int] = None, 
                  max_length: Optional[int] = None) -> None:
    """
    Standardized input validation with consistent error messages.
    
    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        required: Whether the field is required
        value_type: Expected type of the value
        min_length: Minimum length (for strings/lists)
        max_length: Maximum length (for strings/lists)
        
    Raises:
        ValidationError: If validation fails
    """
    if required and value is None:
        raise ValidationError(f"{field_name} is required")
    
    if value is not None:
        if value_type and not isinstance(value, value_type):
            raise ValidationError(f"{field_name} must be of type {value_type.__name__}")
        
        if isinstance(value, (str, list)) and min_length is not None and len(value) < min_length:
            raise ValidationError(f"{field_name} must be at least {min_length} characters/items")
        
        if isinstance(value, (str, list)) and max_length is not None and len(value) > max_length:
            raise ValidationError(f"{field_name} must be at most {max_length} characters/items")
