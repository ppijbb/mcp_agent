# srcs.common package
"""
Common utilities and modules for the MCP Agent System.

This package provides shared functionality used across all agents including:
- Configuration management
- Utility functions
- Performance optimization tools
- Template classes for agent development
"""

from .config import (
    DEFAULT_SERVERS, COMPLIANCE_FRAMEWORKS, REPORT_TIMESTAMP_FORMAT,
    get_output_dir, get_timestamp, AGENT_INSTRUCTION_TEMPLATE,
    COMMON_GUIDELINES, OUTPUT_FORMAT_GUIDELINES,
    DEFAULT_REQUEST_TIMEOUT, MAX_RETRY_ATTEMPTS, CONCURRENT_REQUEST_LIMIT,
    CACHE_TTL_SHORT, CACHE_TTL_MEDIUM, CACHE_TTL_LONG
)

from .utils import (
    EnhancedJSONEncoder, setup_agent_app, ensure_output_directory,
    configure_filesystem_server
)

from .performance import (
    SimpleCache, rate_limit, performance_monitor,
    memoize_strict, ResourceMonitor, default_cache
)

# New infrastructure modules
try:
    from .compatibility import apply_all_compatibility_patches
    from .error_handling import (
        ErrorHandler, AgentError, ErrorSeverity, ErrorCategory,
        handle_errors, safe_execute, validate_input, default_error_handler
    )
    from .logging_utils import (
        get_optimized_logger, setup_application_logging,
        LoggerManager, PerformanceLogger, OptimizedHTTPErrorFilter
    )
    from .connection_pool import create_improved_connection_pool, ImprovedConnectionPool
    
    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False

__all__ = [
    # Config exports
    'DEFAULT_SERVERS', 'COMPLIANCE_FRAMEWORKS', 'REPORT_TIMESTAMP_FORMAT',
    'get_output_dir', 'get_timestamp', 'AGENT_INSTRUCTION_TEMPLATE',
    'COMMON_GUIDELINES', 'OUTPUT_FORMAT_GUIDELINES',
    'DEFAULT_REQUEST_TIMEOUT', 'MAX_RETRY_ATTEMPTS', 'CONCURRENT_REQUEST_LIMIT',
    'CACHE_TTL_SHORT', 'CACHE_TTL_MEDIUM', 'CACHE_TTL_LONG',

    # Utils exports
    'EnhancedJSONEncoder', 'setup_agent_app', 'ensure_output_directory',
    'configure_filesystem_server',

    # Performance exports
    'SimpleCache', 'rate_limit', 'performance_monitor',
    'memoize_strict', 'ResourceMonitor', 'default_cache',
    
    # Infrastructure exports (new modules)
    'INFRASTRUCTURE_AVAILABLE'
]

# Add infrastructure exports if available
if INFRASTRUCTURE_AVAILABLE:
    __all__.extend([
        # Compatibility
        'apply_all_compatibility_patches',
        
        # Error Handling
        'ErrorHandler', 'AgentError', 'ErrorSeverity', 'ErrorCategory',
        'handle_errors', 'safe_execute', 'validate_input', 'default_error_handler',
        
        # Logging
        'get_optimized_logger', 'setup_application_logging',
        'LoggerManager', 'PerformanceLogger', 'OptimizedHTTPErrorFilter',
        
        # Connection Pool
        'create_improved_connection_pool', 'ImprovedConnectionPool'
    ])
