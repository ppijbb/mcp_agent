"""
Feature Flags - 선택적 기능 활성화

기존 코드 변경 없이 새로운 기능을 선택적으로 활성화하기 위한 기능 플래그 모듈.
환경 변수를 통해 각 기능을 독립적으로 활성화/비활성화할 수 있습니다.
"""

import os
import logging

logger = logging.getLogger(__name__)


class FeatureFlags:
    """기능 플래그 - 새로운 기능을 선택적으로 활성화"""
    
    # MCP 안정성 강화 기능
    ENABLE_MCP_STABILITY = os.getenv("ENABLE_MCP_STABILITY", "false").lower() == "true"
    
    # Guardrails 검증 기능
    ENABLE_GUARDRAILS = os.getenv("ENABLE_GUARDRAILS", "false").lower() == "true"
    
    # Agent Tool Wrapper 기능 (Cross-Agent Communication)
    ENABLE_AGENT_TOOLS = os.getenv("ENABLE_AGENT_TOOLS", "false").lower() == "true"
    
    # YAML 설정 파일 지원
    ENABLE_YAML_CONFIG = os.getenv("ENABLE_YAML_CONFIG", "false").lower() == "true"
    
    # MCP 백그라운드 헬스체크
    ENABLE_MCP_HEALTH_BACKGROUND = os.getenv("ENABLE_MCP_HEALTH_BACKGROUND", "false").lower() == "true"
    
    @classmethod
    def get_all_flags(cls) -> dict:
        """모든 기능 플래그 상태 반환"""
        return {
            "mcp_stability": cls.ENABLE_MCP_STABILITY,
            "guardrails": cls.ENABLE_GUARDRAILS,
            "agent_tools": cls.ENABLE_AGENT_TOOLS,
            "yaml_config": cls.ENABLE_YAML_CONFIG,
            "mcp_health_background": cls.ENABLE_MCP_HEALTH_BACKGROUND,
        }
    
    @classmethod
    def log_status(cls):
        """현재 활성화된 기능 플래그 로깅"""
        flags = cls.get_all_flags()
        active_flags = [name for name, enabled in flags.items() if enabled]
        if active_flags:
            logger.info(f"Active feature flags: {', '.join(active_flags)}")
        else:
            logger.debug("No feature flags enabled (using default behavior)")

