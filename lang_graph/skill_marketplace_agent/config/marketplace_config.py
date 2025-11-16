"""
Skill Marketplace Agent 설정
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional


class MarketplaceConfig(BaseModel):
    """
    Skill Marketplace Agent 설정 스키마
    """
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    output_dir: str = Field(default="marketplace_reports", description="리포트 및 결과물 저장 디렉토리")
    data_dir: str = Field(default="marketplace_data", description="Marketplace 데이터 저장 디렉토리")
    
    # LLM 설정
    default_llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="기본 LLM 온도")
    default_llm_max_tokens: int = Field(default=4000, gt=0, description="기본 LLM 최대 토큰")
    
    # Marketplace 설정
    default_commission_rate: float = Field(default=0.15, ge=0.0, le=1.0, description="기본 거래 수수료율 (15%)")
    premium_learner_subscription: float = Field(default=9.99, description="프리미엄 학습자 구독료 (월)")
    premium_instructor_subscription: float = Field(default=29.99, description="프리미엄 강사 구독료 (월)")
    
    # 매칭 설정
    max_match_results: int = Field(default=5, gt=0, description="최대 매칭 결과 수")
    match_score_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="최소 매칭 점수")
    
    # 워크플로우 설정
    max_retries_per_step: int = Field(default=3, ge=0, description="각 워크플로우 단계별 최대 재시도 횟수")
    
    # 리포트 설정
    report_timestamp_format: str = Field(default="%Y%m%d_%H%M%S", description="리포트 파일명에 사용될 타임스탬프 형식")
    
    @classmethod
    def from_env(cls) -> "MarketplaceConfig":
        """환경 변수에서 설정 로드"""
        import os
        return cls(
            output_dir=os.getenv("MARKETPLACE_OUTPUT_DIR", "marketplace_reports"),
            data_dir=os.getenv("MARKETPLACE_DATA_DIR", "marketplace_data"),
        )

