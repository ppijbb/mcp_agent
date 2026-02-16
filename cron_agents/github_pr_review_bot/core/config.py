"""
Configuration Management - 고도화된 설정 관리

이 모듈은 애플리케이션의 모든 설정을 중앙에서 관리합니다.
환경 변수, 설정 파일, 동적 설정 변경 등을 지원합니다.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class GeminiConfig(BaseSettings):
    """Gemini CLI 설정 - 무료 로컬 AI"""
    gemini_cli_path: str = Field("gemini", env="GEMINI_CLI_PATH")
    gemini_model: str = Field("gemini-2.0-flash", env="GEMINI_MODEL")
    max_requests_per_day: int = Field(1000, env="GEMINI_MAX_REQUESTS_PER_DAY")
    timeout: int = Field(60, env="GEMINI_TIMEOUT")
    # Fallback 방지 설정 - 오류 발생 시 즉시 종료
    fail_on_gemini_error: bool = Field(True, env="GEMINI_FAIL_ON_ERROR")
    require_valid_response: bool = Field(True, env="GEMINI_REQUIRE_VALID_RESPONSE")
    # 프롬프트 템플릿 설정
    review_prompt_template: str = Field(
        "다음 코드를 GitHub PR 리뷰 관점에서 분석해주세요:\n\n{code}\n\n언어: {language}\n파일: {file_path}\n\n코드 품질, 보안, 성능, 스타일을 종합적으로 검토하고 구체적인 개선사항을 제안해주세요.",
        env="GEMINI_REVIEW_PROMPT_TEMPLATE"
    )


class VLLMConfig(BaseSettings):
    """vLLM 설정 - OpenAI 형식 API"""
    base_url: Optional[str] = Field(None, env="VLLM_BASE_URL")
    model_name: str = Field("Qwen/Qwen2.5-1.5B-Instruct", env="VLLM_MODEL_NAME")
    max_tokens: int = Field(2000, env="VLLM_MAX_TOKENS")
    temperature: float = Field(0.1, env="VLLM_TEMPERATURE")
    timeout: int = Field(60, env="VLLM_TIMEOUT")
    # Fallback 방지 설정
    fail_on_vllm_error: bool = Field(True, env="VLLM_FAIL_ON_ERROR")
    require_valid_response: bool = Field(True, env="VLLM_REQUIRE_VALID_RESPONSE")

    @validator("gemini_cli_path")
    def validate_gemini_path(cls, v):
        if not v:
            raise ValueError("gemini_cli_path is required")
        return v


class GitHubConfig(BaseSettings):
    """GitHub 설정 - 비용 최적화"""
    token: str = Field(..., env="GITHUB_TOKEN")
    webhook_secret: Optional[str] = Field(None, env="GITHUB_WEBHOOK_SECRET")
    app_id: Optional[str] = Field(None, env="GITHUB_APP_ID")
    private_key_path: Optional[str] = Field(None, env="GITHUB_PRIVATE_KEY_PATH")
    rate_limit: int = Field(5000, env="GITHUB_RATE_LIMIT")
    # PR 리뷰 기본 설정 - 명시적 요청이 있을 때만 활성화
    auto_review_enabled: bool = Field(False, env="GITHUB_AUTO_REVIEW_ENABLED")
    require_explicit_review_request: bool = Field(True, env="GITHUB_REQUIRE_EXPLICIT_REVIEW_REQUEST")
    # 비용 최적화: 중요한 PR만 리뷰
    min_pr_size_threshold: int = Field(50, env="MIN_PR_SIZE_THRESHOLD")  # 최소 변경 라인 수
    max_pr_size_threshold: int = Field(1000, env="MAX_PR_SIZE_THRESHOLD")  # 최대 변경 라인 수
    skip_draft_prs: bool = Field(True, env="SKIP_DRAFT_PRS")  # 드래프트 PR 스킵
    skip_auto_merge_prs: bool = Field(True, env="SKIP_AUTO_MERGE_PRS")  # 자동 머지 PR 스킵
    # 오류 처리 설정 - fallback 없이 즉시 종료
    fail_fast_on_error: bool = Field(True, env="GITHUB_FAIL_FAST_ON_ERROR")
    max_retry_attempts: int = Field(0, env="GITHUB_MAX_RETRY_ATTEMPTS")  # 0 = 재시도 없음

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


class OptimizationConfig(BaseSettings):
    """최적화 설정 - MCP 통합 (Gemini CLI + vLLM)"""
    # 캐시 설정
    enable_aggressive_caching: bool = Field(True, env="ENABLE_AGGRESSIVE_CACHING")
    cache_review_results: bool = Field(True, env="CACHE_REVIEW_RESULTS")
    cache_ttl_hours: int = Field(24, env="CACHE_TTL_HOURS")

    # 배치 처리
    enable_batch_processing: bool = Field(True, env="ENABLE_BATCH_PROCESSING")
    batch_size: int = Field(5, env="BATCH_SIZE")

    # MCP 모니터링
    enable_mcp_monitoring: bool = Field(True, env="ENABLE_MCP_MONITORING")
    gemini_usage_tracking: bool = Field(True, env="GEMINI_USAGE_TRACKING")
    vllm_usage_tracking: bool = Field(True, env="VLLM_USAGE_TRACKING")


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
    gemini: GeminiConfig = GeminiConfig()
    vllm: VLLMConfig = VLLMConfig()
    github: GitHubConfig = GitHubConfig()
    database: DatabaseConfig = DatabaseConfig()
    cache: CacheConfig = CacheConfig()
    queue: QueueConfig = QueueConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    security: SecurityConfig = SecurityConfig()
    optimization: OptimizationConfig = OptimizationConfig()

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

        # Gemini CLI 검증
        import shutil
        if not shutil.which(self.gemini.gemini_cli_path):
            raise ValueError(f"Gemini CLI not found at: {self.gemini.gemini_cli_path}. Please install gemini-cli first.")

        # vLLM 설정 검증 (선택적)
        if self.vllm.base_url and not self.vllm.base_url.startswith(('http://', 'https://')):
            raise ValueError("vLLM base_url must start with http:// or https://")

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

        # Gemini 설정
        config_dict["gemini"] = {
            "gemini_cli_path": self.gemini.gemini_cli_path,
            "gemini_model": self.gemini.gemini_model,
            "max_requests_per_day": self.gemini.max_requests_per_day,
            "timeout": self.gemini.timeout,
        }

        # vLLM 설정
        config_dict["vllm"] = {
            "base_url": self.vllm.base_url,
            "model_name": self.vllm.model_name,
            "max_tokens": self.vllm.max_tokens,
            "temperature": self.vllm.temperature,
            "timeout": self.vllm.timeout,
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
            "optimization": self.optimization.dict(),
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
