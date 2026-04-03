"""
Agent base module.

Provides the BaseAgent abstract class for all MCP agents with common
functionality including MCPApp initialization, circuit breaker, async
session management, and workflow execution.

Classes:
    BaseAgent: Abstract base class for all MCP agents
    
Functions:
    async_memoize: Memoization decorator for async functions
    _cleanup_mcp_apps: Clean up all active MCPApp instances on process exit
    _register_cleanup: Register cleanup handler for MCPApp instances
"""
