"""
Configuration Management - 고도화된 설정 관리

이 모듈은 애플리케이션의 모든 설정을 중앙에서 관리합니다.
환경 변수, 설정 파일, 동적 설정 변경 등을 지원합니다.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class LLMConfig(BaseSettings):
    """LLM 설정"""
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    default_provider: str = Field("openai", env="DEFAULT_LLM_PROVIDER")
    max_tokens: int = Field(4000, env="MAX_TOKENS")
    temperature: float = Field(0.1, env="TEMPERATURE")
    timeout: int = Field(30, env="LLM_TIMEOUT")
    
    @validator("default_provider")
    def validate_provider(cls, v):
        if v not in ["openai", "anthropic", "google"]:
            raise ValueError("default_provider must be one of: openai, anthropic, google")
        return v

class GitHubConfig(BaseSettings):
    """GitHub 설정"""
    token: str = Field(..., env="GITHUB_TOKEN")
    webhook_secret: Optional[str] = Field(None, env="GITHUB_WEBHOOK_SECRET")
    app_id: Optional[str] = Field(None, env="GITHUB_APP_ID")
    private_key_path: Optional[str] = Field(None, env="GITHUB_PRIVATE_KEY_PATH")
    rate_limit: int = Field(5000, env="GITHUB_RATE_LIMIT")
    
    @validator("token")
    def validate_token(cls, v):
        if not v:
            raise ValueError("GitHub token is required")
        return v

class DatabaseConfig(BaseSettings):
    """데이터베이스 설정"""
    url: str = Field("sqlite:///./pr_review_bot.db", env="DATABASE_URL")
    pool_size: int = Field(10, env="DB_POOL_SIZE")
    max_overflow: int = Field(20, env="DB_MAX_OVERFLOW")
    echo: bool = Field(False, env="DB_ECHO")
    
    @validator("url")
    def validate_url(cls, v):
        if not v:
            raise ValueError("Database URL is required")
        return v

class CacheConfig(BaseSettings):
    """캐시 설정"""
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    ttl: int = Field(3600, env="CACHE_TTL")
    max_size: int = Field(1000, env="CACHE_MAX_SIZE")
    enable_cache: bool = Field(True, env="ENABLE_CACHE")

class QueueConfig(BaseSettings):
    """작업 큐 설정"""
    broker_url: str = Field("redis://localhost:6379/1", env="CELERY_BROKER_URL")
    result_backend: str = Field("redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")
    task_serializer: str = Field("json", env="CELERY_TASK_SERIALIZER")
    result_serializer: str = Field("json", env="CELERY_RESULT_SERIALIZER")
    accept_content: List[str] = Field(["json"], env="CELERY_ACCEPT_CONTENT")
    timezone: str = Field("UTC", env="CELERY_TIMEZONE")
    enable_utc: bool = Field(True, env="CELERY_ENABLE_UTC")

class MonitoringConfig(BaseSettings):
    """모니터링 설정"""
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    enable_tracing: bool = Field(True, env="ENABLE_TRACING")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")

class SecurityConfig(BaseSettings):
    """보안 설정"""
    enable_rate_limiting: bool = Field(True, env="ENABLE_RATE_LIMITING")
    max_requests_per_minute: int = Field(60, env="MAX_REQUESTS_PER_MINUTE")
    enable_cors: bool = Field(True, env="ENABLE_CORS")
    allowed_origins: List[str] = Field(["*"], env="ALLOWED_ORIGINS")
    jwt_secret: Optional[str] = Field(None, env="JWT_SECRET")

class Config(BaseSettings):
    """메인 설정 클래스"""
    
    # 환경 설정
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    
    # 서버 설정
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    webhook_port: int = Field(8080, env="WEBHOOK_PORT")
    
    # 서브 설정들
    llm: LLMConfig = LLMConfig()
    github: GitHubConfig = GitHubConfig()
    database: DatabaseConfig = DatabaseConfig()
    cache: CacheConfig = CacheConfig()
    queue: QueueConfig = QueueConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    security: SecurityConfig = SecurityConfig()
    
    # 동적 설정 저장소
    _dynamic_settings: Dict[str, Any] = {}
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        # 환경 변수 파일 로드
        load_dotenv()
        super().__init__(**kwargs)
        
        # 설정 검증
        self._validate_config()
        
        logger.info(f"Configuration loaded for environment: {self.environment}")
    
    def _validate_config(self):
        """설정 검증"""
        if self.environment not in ["development", "staging", "production"]:
            raise ValueError("Environment must be one of: development, staging, production")
        
        # LLM API 키 검증
        if self.llm.default_provider == "openai" and not self.llm.openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI as default provider")
        elif self.llm.default_provider == "anthropic" and not self.llm.anthropic_api_key:
            raise ValueError("Anthropic API key is required when using Anthropic as default provider")
        elif self.llm.default_provider == "google" and not self.llm.google_api_key:
            raise ValueError("Google API key is required when using Google as default provider")
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정값 가져오기 (동적 설정 우선)"""
        if key in self._dynamic_settings:
            return self._dynamic_settings[key]
        
        # 중첩된 설정 접근 (예: "llm.openai_api_key")
        if "." in key:
            parts = key.split(".")
            current = self
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return default
            return current
        
        return getattr(self, key, default)
    
    def set(self, key: str, value: Any):
        """동적 설정값 설정"""
        self._dynamic_settings[key] = value
        logger.info(f"Dynamic setting updated: {key} = {value}")
    
    def reload(self):
        """설정 재로드"""
        load_dotenv()
        self._validate_config()
        logger.info("Configuration reloaded")
    
    def export(self) -> Dict[str, Any]:
        """설정 내보내기 (민감한 정보 제외)"""
        config_dict = {}
        
        # 기본 설정
        config_dict.update({
            "environment": self.environment,
            "debug": self.debug,
            "host": self.host,
            "port": self.port,
            "webhook_port": self.webhook_port,
        })
        
        # LLM 설정 (API 키 제외)
        config_dict["llm"] = {
            "default_provider": self.llm.default_provider,
            "max_tokens": self.llm.max_tokens,
            "temperature": self.llm.temperature,
            "timeout": self.llm.timeout,
        }
        
        # GitHub 설정 (토큰 제외)
        config_dict["github"] = {
            "rate_limit": self.github.rate_limit,
        }
        
        # 기타 설정들
        config_dict.update({
            "database": self.database.dict(),
            "cache": self.cache.dict(),
            "queue": self.queue.dict(),
            "monitoring": self.monitoring.dict(),
            "security": self.security.dict(),
        })
        
        return config_dict
    
    def save_to_file(self, filepath: str):
        """설정을 파일로 저장"""
        config_dict = self.export()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved to: {filepath}")

# 전역 설정 인스턴스
config = Config() 