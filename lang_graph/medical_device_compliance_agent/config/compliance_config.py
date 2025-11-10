"""
의료기기 규제 컴플라이언스 Agent 설정
"""

import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM 설정"""
    preferred_provider: str = Field(default="groq", description="선호하는 LLM Provider")
    budget_limit: Optional[float] = Field(default=None, description="예산 제한")
    temperature: float = Field(default=0.1, description="Temperature")
    max_tokens: int = Field(default=4000, description="최대 토큰 수")


class MCPServerConfig(BaseModel):
    """MCP 서버 설정"""
    filesystem_enabled: bool = Field(default=True, description="Filesystem 서버 활성화")
    g_search_enabled: bool = Field(default=True, description="g-search 서버 활성화")
    fetch_enabled: bool = Field(default=True, description="fetch 서버 활성화")
    github_enabled: bool = Field(default=False, description="GitHub 서버 활성화")


class RegulatoryFrameworkConfig(BaseModel):
    """규제 프레임워크 설정"""
    fda_510k_enabled: bool = Field(default=True, description="FDA 510(k) 활성화")
    ce_marking_enabled: bool = Field(default=True, description="CE 마킹 활성화")
    iso13485_enabled: bool = Field(default=True, description="ISO 13485 활성화")


class ComplianceConfig(BaseModel):
    """의료기기 규제 컴플라이언스 Agent 전체 설정"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    mcp_servers: MCPServerConfig = Field(default_factory=MCPServerConfig)
    regulatory_frameworks: RegulatoryFrameworkConfig = Field(default_factory=RegulatoryFrameworkConfig)
    
    output_dir: str = Field(default="medical_device_compliance_reports", description="출력 디렉토리")
    enable_monitoring: bool = Field(default=True, description="지속적 모니터링 활성화")
    
    @classmethod
    def from_env(cls) -> "ComplianceConfig":
        """환경 변수에서 설정 로드"""
        return cls(
            llm=LLMConfig(
                preferred_provider=os.getenv("MDC_PREFERRED_PROVIDER", "groq"),
                budget_limit=float(os.getenv("MDC_BUDGET_LIMIT", "0")) if os.getenv("MDC_BUDGET_LIMIT") else None,
                temperature=float(os.getenv("MDC_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("MDC_MAX_TOKENS", "4000")),
            ),
            mcp_servers=MCPServerConfig(
                filesystem_enabled=os.getenv("MDC_FILESYSTEM_ENABLED", "true").lower() == "true",
                g_search_enabled=os.getenv("MDC_G_SEARCH_ENABLED", "true").lower() == "true",
                fetch_enabled=os.getenv("MDC_FETCH_ENABLED", "true").lower() == "true",
                github_enabled=os.getenv("MDC_GITHUB_ENABLED", "false").lower() == "true",
            ),
            regulatory_frameworks=RegulatoryFrameworkConfig(
                fda_510k_enabled=os.getenv("MDC_FDA_510K_ENABLED", "true").lower() == "true",
                ce_marking_enabled=os.getenv("MDC_CE_MARKING_ENABLED", "true").lower() == "true",
                iso13485_enabled=os.getenv("MDC_ISO13485_ENABLED", "true").lower() == "true",
            ),
            output_dir=os.getenv("MDC_OUTPUT_DIR", "medical_device_compliance_reports"),
            enable_monitoring=os.getenv("MDC_ENABLE_MONITORING", "true").lower() == "true",
        )

