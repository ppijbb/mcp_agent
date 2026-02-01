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
    'memoize_strict', 'ResourceMonitor', 'default_cache'
] 