from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    """
    MCP server configuration schema.

    Defines configuration for individual MCP servers including command,
    arguments, environment variables, and operational parameters.

    Attributes:
        command: Command to execute for starting the MCP server
        args: List of command line arguments for the server
        env: Environment variables dictionary for server process
        timeout: Request timeout in seconds (10-300 range)
        retry_count: Number of retry attempts for failed requests (0-5 range)
        enabled: Whether this server configuration is active
    """
    command: str
    args: List[str]
    env: Dict[str, str] = Field(default_factory=dict)
    timeout: int = Field(default=60, ge=10, le=300)
    retry_count: int = Field(default=3, ge=0, le=5)
    enabled: bool = True


class LoggingConfig(BaseModel):
    """
    Logging configuration schema.

    Defines logging behavior including log levels, file management,
    and rotation policies for the agent system.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file, defaults to logs/mcp_agent.log
        rotation: Log file rotation size threshold
        retention: Log file retention period
    """
    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_file: Optional[str] = "logs/mcp_agent.log"
    rotation: str = "10 MB"
    retention: str = "7 days"


class SecurityConfig(BaseModel):
    """
    Security configuration schema.

    Defines security parameters including encryption settings
    and access control for the agent system.

    Attributes:
        encryption_key: Optional encryption key for sensitive data
        allowed_hosts: List of allowed host patterns for access
    """
    encryption_key: Optional[str] = None
    allowed_hosts: List[str] = ["*"]


class CacheConfig(BaseModel):
    """
    Cache configuration schema.

    Defines caching behavior including cache type, TTL settings,
    and backend-specific configuration.

    Attributes:
        enabled: Whether caching is enabled
        type: Cache backend type (in-memory or redis)
        ttl: Default time-to-live for cache entries in seconds
        redis_url: Redis connection URL if redis backend is used
    """
    enabled: bool = True
    type: str = Field(default="in-memory", pattern="^(in-memory|redis)$")
    ttl: int = Field(default=3600, ge=60)
    redis_url: Optional[str] = None


class AppConfig(BaseModel):
    """
    Main application configuration schema.

    Central configuration model that aggregates all subsystem
    configurations for the MCP agent system.

    Attributes:
        app_name: Name of the application
        environment: Deployment environment (development, staging, production)
        logging: Logging configuration instance
        security: Security configuration instance
        cache: Cache configuration instance
        mcp_servers: Dictionary of MCP server configurations
    """
    app_name: str = "MCP_Agent_System"
    environment: str = "development"

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    mcp_servers: Dict[str, MCPServerConfig] = Field(default_factory=dict)

    class Config:
        validate_assignment = True
