"""
Financial Agent Configuration Management
환경 변수 기반 설정 시스템 - 모든 값은 필수, 기본값 없음
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ModelProvider(Enum):
    """LLM Provider Enum"""
    GROQ = "groq"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude"


@dataclass
class LLMConfig:
    """LLM 관련 설정 (기존 Gemini 전용)"""
    api_key: str
    model: str
    temperature: float


@dataclass
class MultiModelLLMConfig:
    """Multi-Model LLM 관련 설정"""
    preferred_provider: Optional[ModelProvider]
    groq_api_key: Optional[str]
    openrouter_api_key: Optional[str]
    google_api_key: Optional[str]
    openai_api_key: Optional[str]
    anthropic_api_key: Optional[str]


@dataclass
class MCPConfig:
    """MCP 서버 관련 설정"""
    timeout: int
    data_period: str


@dataclass
class TradingConfig:
    """거래 관련 설정"""
    default_shares: int
    max_trade_amount: float
    commission_rate: float
    affiliate_commission_rate: float


@dataclass
class ChartAnalysisConfig:
    """차트 분석 관련 설정"""
    chart_image_save_path: str
    chart_analysis_llm_model: Optional[str]
    max_ohlcv_data_points: int
    exit_prediction_horizon_days: int


@dataclass
class WorkflowConfig:
    """워크플로우 관련 설정"""
    valid_risk_profiles: List[str]
    default_tickers: List[str]


@dataclass
class FinancialAgentConfig:
    """전체 설정 통합"""
    llm: LLMConfig  # 기존 Gemini 전용 (하위 호환성)
    multi_model_llm: MultiModelLLMConfig  # Multi-Model LLM 설정
    mcp: MCPConfig
    trading: TradingConfig
    workflow: WorkflowConfig
    chart_analysis: ChartAnalysisConfig


def _get_required_env(key: str, var_type: type = str) -> Any:
    """필수 환경 변수를 가져오고 타입 변환"""
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"필수 환경 변수 {key}가 설정되지 않았습니다.")
    
    try:
        if var_type == str:
            return value
        elif var_type == int:
            return int(value)
        elif var_type == float:
            return float(value)
        elif var_type == list:
            return [item.strip() for item in value.split(',')]
        else:
            return var_type(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"환경 변수 {key}의 값 '{value}'을 {var_type.__name__}로 변환할 수 없습니다: {e}")


def _validate_temperature(temp: float) -> None:
    """Temperature 값 범위 검증"""
    if not 0.0 <= temp <= 2.0:
        raise ValueError(f"LLM_TEMPERATURE는 0.0-2.0 범위여야 합니다. 현재 값: {temp}")


def _validate_timeout(timeout: int) -> None:
    """Timeout 값 범위 검증"""
    if timeout <= 0:
        raise ValueError(f"MCP_TIMEOUT은 양수여야 합니다. 현재 값: {timeout}")


def _validate_risk_profiles(profiles: List[str]) -> None:
    """리스크 프로필 값 검증"""
    valid_values = {"conservative", "moderate", "aggressive"}
    invalid_profiles = [p for p in profiles if p not in valid_values]
    if invalid_profiles:
        raise ValueError(f"유효하지 않은 리스크 프로필: {invalid_profiles}. 유효한 값: {valid_values}")


def _get_optional_env(key: str, var_type: type = str) -> Optional[Any]:
    """선택적 환경 변수를 가져오고 타입 변환"""
    value = os.getenv(key)
    if value is None:
        return None
    
    try:
        if var_type == str:
            return value
        elif var_type == int:
            return int(value)
        elif var_type == float:
            return float(value)
        else:
            return var_type(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"환경 변수 {key}의 값 '{value}'을 {var_type.__name__}로 변환할 수 없습니다: {e}")


def _validate_shares(shares: int) -> None:
    """거래 수량 값 검증"""
    if shares <= 0:
        raise ValueError(f"DEFAULT_SHARES는 양수여야 합니다. 현재 값: {shares}")


def _validate_commission_rate(rate: float) -> None:
    """수수료율 값 검증"""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"COMMISSION_RATE는 0.0-1.0 범위여야 합니다. 현재 값: {rate}")


def load_config() -> FinancialAgentConfig:
    """
    환경 변수에서 모든 설정을 로드합니다.
    모든 환경 변수는 필수이며, 누락 시 ValueError 발생
    """
    try:
        # LLM 설정 (기존 Gemini 전용 - 하위 호환성)
        llm_config = LLMConfig(
            api_key=_get_required_env("GEMINI_API_KEY", str),
            model=_get_required_env("GEMINI_MODEL", str),
            temperature=_get_required_env("LLM_TEMPERATURE", float)
        )
        _validate_temperature(llm_config.temperature)
        
        # Multi-Model LLM 설정
        preferred_provider_str = _get_optional_env("PREFERRED_LLM_PROVIDER", str)
        preferred_provider = None
        if preferred_provider_str:
            try:
                preferred_provider = ModelProvider(preferred_provider_str.lower())
            except ValueError:
                raise ValueError(f"유효하지 않은 PREFERRED_LLM_PROVIDER: {preferred_provider_str}")
        
        multi_model_llm_config = MultiModelLLMConfig(
            preferred_provider=preferred_provider,
            groq_api_key=_get_optional_env("GROQ_API_KEY", str),
            openrouter_api_key=_get_optional_env("OPENROUTER_API_KEY", str),
            google_api_key=_get_optional_env("GOOGLE_API_KEY", str),
            openai_api_key=_get_optional_env("OPENAI_API_KEY", str),
            anthropic_api_key=_get_optional_env("ANTHROPIC_API_KEY", str),
        )
        
        # MCP 설정
        mcp_config = MCPConfig(
            timeout=_get_required_env("MCP_TIMEOUT", int),
            data_period=_get_required_env("MCP_DATA_PERIOD", str)
        )
        _validate_timeout(mcp_config.timeout)
        
        # 거래 설정
        trading_config = TradingConfig(
            default_shares=_get_required_env("DEFAULT_SHARES", int),
            max_trade_amount=_get_required_env("MAX_TRADE_AMOUNT", float),
            commission_rate=_get_required_env("COMMISSION_RATE", float),
            affiliate_commission_rate=_get_required_env("AFFILIATE_COMMISSION_RATE", float),
        )
        _validate_shares(trading_config.default_shares)
        _validate_commission_rate(trading_config.commission_rate)
        _validate_commission_rate(trading_config.affiliate_commission_rate)
        
        # 워크플로우 설정
        workflow_config = WorkflowConfig(
            valid_risk_profiles=_get_required_env("RISK_PROFILES", list),
            default_tickers=_get_required_env("DEFAULT_TICKERS", list)
        )
        _validate_risk_profiles(workflow_config.valid_risk_profiles)
        
        # 차트 분석 설정 (선택적, 기본값 사용)
        chart_image_save_path = _get_optional_env("CHART_IMAGE_SAVE_PATH", str) or "chart_images"
        chart_analysis_llm_model = _get_optional_env("CHART_ANALYSIS_LLM_MODEL", str)
        max_ohlcv_data_points = _get_optional_env("MAX_OHLCV_DATA_POINTS", int) or 1000
        exit_prediction_horizon_days = _get_optional_env("EXIT_PREDICTION_HORIZON_DAYS", int) or 30
        
        chart_analysis_config = ChartAnalysisConfig(
            chart_image_save_path=chart_image_save_path,
            chart_analysis_llm_model=chart_analysis_llm_model,
            max_ohlcv_data_points=max_ohlcv_data_points,
            exit_prediction_horizon_days=exit_prediction_horizon_days
        )
        
        return FinancialAgentConfig(
            llm=llm_config,
            multi_model_llm=multi_model_llm_config,
            mcp=mcp_config,
            trading=trading_config,
            workflow=workflow_config,
            chart_analysis=chart_analysis_config
        )
        
    except ValueError as e:
        raise ValueError(f"설정 로드 실패: {e}")


# 전역 설정 인스턴스 - 초기화 시 로드됨
_config: FinancialAgentConfig = None


def get_config() -> FinancialAgentConfig:
    """설정 인스턴스 반환"""
    if _config is None:
        raise RuntimeError("설정이 로드되지 않았습니다. load_config()를 먼저 호출하세요.")
    return _config


def initialize_config() -> None:
    """설정 초기화 - 애플리케이션 시작 시 호출"""
    global _config
    _config = load_config()
    print("✅ Financial Agent 설정이 성공적으로 로드되었습니다.")


# 편의 함수들
def get_llm_config() -> LLMConfig:
    """LLM 설정 반환"""
    return get_config().llm


def get_mcp_config() -> MCPConfig:
    """MCP 설정 반환"""
    return get_config().mcp


def get_workflow_config() -> WorkflowConfig:
    """워크플로우 설정 반환"""
    return get_config().workflow


def get_multi_model_llm_config() -> MultiModelLLMConfig:
    """Multi-Model LLM 설정 반환"""
    return get_config().multi_model_llm


def get_trading_config() -> TradingConfig:
    """거래 설정 반환"""
    return get_config().trading


def get_chart_analysis_config() -> ChartAnalysisConfig:
    """차트 분석 설정 반환"""
    return get_config().chart_analysis
