"""
Financial Agent Configuration Management
환경 변수 기반 설정 시스템 - 모든 값은 필수, 기본값 없음
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM 관련 설정"""
    api_key: str
    model: str
    temperature: float


@dataclass
class MCPConfig:
    """MCP 서버 관련 설정"""
    timeout: int
    data_period: str


@dataclass
class WorkflowConfig:
    """워크플로우 관련 설정"""
    valid_risk_profiles: List[str]
    default_tickers: List[str]


@dataclass
class FinancialAgentConfig:
    """전체 설정 통합"""
    llm: LLMConfig
    mcp: MCPConfig
    workflow: WorkflowConfig


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


def load_config() -> FinancialAgentConfig:
    """
    환경 변수에서 모든 설정을 로드합니다.
    모든 환경 변수는 필수이며, 누락 시 ValueError 발생
    """
    try:
        # LLM 설정
        llm_config = LLMConfig(
            api_key=_get_required_env("GEMINI_API_KEY", str),
            model=_get_required_env("GEMINI_MODEL", str),
            temperature=_get_required_env("LLM_TEMPERATURE", float)
        )
        _validate_temperature(llm_config.temperature)
        
        # MCP 설정
        mcp_config = MCPConfig(
            timeout=_get_required_env("MCP_TIMEOUT", int),
            data_period=_get_required_env("MCP_DATA_PERIOD", str)
        )
        _validate_timeout(mcp_config.timeout)
        
        # 워크플로우 설정
        workflow_config = WorkflowConfig(
            valid_risk_profiles=_get_required_env("RISK_PROFILES", list),
            default_tickers=_get_required_env("DEFAULT_TICKERS", list)
        )
        _validate_risk_profiles(workflow_config.valid_risk_profiles)
        
        return FinancialAgentConfig(
            llm=llm_config,
            mcp=mcp_config,
            workflow=workflow_config
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
