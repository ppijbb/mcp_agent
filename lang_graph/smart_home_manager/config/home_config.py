"""
스마트 홈 매니저 Agent 설정
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


class HomeConfig(BaseModel):
    """스마트 홈 매니저 Agent 전체 설정"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    mcp_servers: MCPServerConfig = Field(default_factory=MCPServerConfig)
    
    output_dir: str = Field(default="smart_home_reports", description="출력 디렉토리")
    data_dir: str = Field(default="home_data", description="데이터 저장 디렉토리")
    
    # 홈 관련 설정
    default_devices: List[str] = Field(
        default_factory=lambda: ["lighting", "heating", "cooling", "security"],
        description="기본 기기 카테고리 목록"
    )
    energy_optimization_enabled: bool = Field(default=True, description="에너지 최적화 활성화")
    security_monitoring_enabled: bool = Field(default=True, description="보안 모니터링 활성화")
    maintenance_alerts_enabled: bool = Field(default=True, description="유지보수 알림 활성화")
    automation_enabled: bool = Field(default=True, description="자동화 활성화")
    
    @classmethod
    def from_env(cls) -> "HomeConfig":
        """환경 변수에서 설정 로드"""
        return cls(
            llm=LLMConfig(
                preferred_provider=os.getenv("SHM_PREFERRED_PROVIDER", "groq"),
                budget_limit=float(os.getenv("SHM_BUDGET_LIMIT", "0")) if os.getenv("SHM_BUDGET_LIMIT") else None,
                temperature=float(os.getenv("SHM_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("SHM_MAX_TOKENS", "4000")),
            ),
            mcp_servers=MCPServerConfig(
                filesystem_enabled=os.getenv("SHM_FILESYSTEM_ENABLED", "true").lower() == "true",
                g_search_enabled=os.getenv("SHM_G_SEARCH_ENABLED", "true").lower() == "true",
                fetch_enabled=os.getenv("SHM_FETCH_ENABLED", "true").lower() == "true",
            ),
            output_dir=os.getenv("SHM_OUTPUT_DIR", "smart_home_reports"),
            data_dir=os.getenv("SHM_DATA_DIR", "home_data"),
            default_devices=os.getenv("SHM_DEFAULT_DEVICES", "lighting,heating,cooling,security").split(","),
            energy_optimization_enabled=os.getenv("SHM_ENERGY_OPTIMIZATION_ENABLED", "true").lower() == "true",
            security_monitoring_enabled=os.getenv("SHM_SECURITY_MONITORING_ENABLED", "true").lower() == "true",
            maintenance_alerts_enabled=os.getenv("SHM_MAINTENANCE_ALERTS_ENABLED", "true").lower() == "true",
            automation_enabled=os.getenv("SHM_AUTOMATION_ENABLED", "true").lower() == "true",
        )

