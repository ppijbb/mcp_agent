"""
Multi-Agent Automation Service

실제 mcp_agent 라이브러리를 사용한 코드 리뷰, 자동 문서화, 성능 테스트, 보안 검증을 담당하는
Multi-Agent 시스템입니다. Gemini CLI를 통한 최종 명령어 실행을 지원합니다.
"""

__version__ = "2.0.0"
__author__ = "MCP Agent Team"

from .orchestrator import MultiAgentOrchestrator
from .gemini_executor import GeminiCLIExecutor

__all__ = [
    "MultiAgentOrchestrator",
    "GeminiCLIExecutor"
] 