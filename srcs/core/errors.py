from typing import Dict, Any, Optional


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


class APIError(MCPError):
    """Raised for errors related to external APIs."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response = response or {}


class WorkflowError(MCPError):
    """Workflow-related errors."""


class CircuitBreakerOpen(MCPError):
    """Raised when the circuit breaker is open."""


class EncryptionError(MCPError):
    """Raised for errors during encryption or decryption."""


class ValidationError(MCPError):
    """Raised for input validation errors."""


class SecurityError(MCPError):
    """Raised for security-related errors."""
