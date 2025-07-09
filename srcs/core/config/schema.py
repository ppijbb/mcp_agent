from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, HttpUrl

class MCPServerConfig(BaseModel):
    """MCP 서버 개별 설정"""
    command: str
    args: List[str]
    env: Dict[str, str] = Field(default_factory=dict)
    timeout: int = Field(default=60, ge=10, le=300)
    retry_count: int = Field(default=3, ge=0, le=5)
    enabled: bool = True

class LoggingConfig(BaseModel):
    """로깅 설정"""
    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_file: Optional[str] = "logs/mcp_agent.log"
    rotation: str = "10 MB"
    retention: str = "7 days"

class SecurityConfig(BaseModel):
    """보안 관련 설정"""
    encryption_key: Optional[str] = None
    allowed_hosts: List[str] = ["*"]

class CacheConfig(BaseModel):
    """캐시 설정"""
    enabled: bool = True
    type: str = Field(default="in-memory", pattern="^(in-memory|redis)$")
    ttl: int = Field(default=3600, ge=60)
    redis_url: Optional[str] = None

class AppConfig(BaseModel):
    """전체 에이전트 애플리케이션 설정 스키마"""
    app_name: str = "MCP_Agent_System"
    environment: str = "development"
    
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    
    mcp_servers: Dict[str, MCPServerConfig] = Field(default_factory=dict)

    class Config:
        validate_assignment = True 