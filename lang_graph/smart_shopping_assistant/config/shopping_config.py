"""
스마트 쇼핑 어시스턴트 Agent 설정
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


class ShoppingConfig(BaseModel):
    """스마트 쇼핑 어시스턴트 Agent 전체 설정"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    mcp_servers: MCPServerConfig = Field(default_factory=MCPServerConfig)
    
    output_dir: str = Field(default="smart_shopping_reports", description="출력 디렉토리")
    data_dir: str = Field(default="shopping_data", description="데이터 저장 디렉토리")
    
    # 쇼핑 관련 설정
    default_stores: List[str] = Field(
        default_factory=lambda: ["Amazon", "eBay", "Walmart", "Target", "Best Buy"],
        description="기본 쇼핑몰 목록"
    )
    price_comparison_range: int = Field(default=10, description="가격 비교 결과 수")
    max_recommendations: int = Field(default=5, description="최대 추천 제품 수")
    
    @classmethod
    def from_env(cls) -> "ShoppingConfig":
        """환경 변수에서 설정 로드"""
        return cls(
            llm=LLMConfig(
                preferred_provider=os.getenv("SSA_PREFERRED_PROVIDER", "groq"),
                budget_limit=float(os.getenv("SSA_BUDGET_LIMIT", "0")) if os.getenv("SSA_BUDGET_LIMIT") else None,
                temperature=float(os.getenv("SSA_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("SSA_MAX_TOKENS", "4000")),
            ),
            mcp_servers=MCPServerConfig(
                filesystem_enabled=os.getenv("SSA_FILESYSTEM_ENABLED", "true").lower() == "true",
                g_search_enabled=os.getenv("SSA_G_SEARCH_ENABLED", "true").lower() == "true",
                fetch_enabled=os.getenv("SSA_FETCH_ENABLED", "true").lower() == "true",
            ),
            output_dir=os.getenv("SSA_OUTPUT_DIR", "smart_shopping_reports"),
            data_dir=os.getenv("SSA_DATA_DIR", "shopping_data"),
            default_stores=os.getenv("SSA_DEFAULT_STORES", "Amazon,eBay,Walmart,Target,Best Buy").split(","),
            price_comparison_range=int(os.getenv("SSA_PRICE_COMPARISON_RANGE", "10")),
            max_recommendations=int(os.getenv("SSA_MAX_RECOMMENDATIONS", "5")),
        )

