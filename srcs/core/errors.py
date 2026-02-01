class MCPError(Exception):
    """Base exception class for all MCP-Agent errors."""


class ConfigError(MCPError):
    """Raised for errors in configuration."""


class APIError(MCPError):
    """Raised for errors related to external APIs."""


class WorkflowError(MCPError):
    """Workflow-related errors."""


class CircuitBreakerOpen(MCPError):
    """Raised when the circuit breaker is open."""


class EncryptionError(MCPError):
    """Raised for errors during encryption or decryption."""
