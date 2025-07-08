class MCPError(Exception):
    """Base exception class for all MCP-Agent errors."""
    pass

class ConfigError(MCPError):
    """Raised for errors in configuration."""
    pass

class APIError(MCPError):
    """Raised for errors related to external APIs."""
    pass

class WorkflowError(MCPError):
    """Workflow-related errors."""
    pass

class CircuitBreakerOpen(MCPError):
    """Raised when the circuit breaker is open."""
    pass

class EncryptionError(MCPError):
    """Raised for errors during encryption or decryption."""
    pass 