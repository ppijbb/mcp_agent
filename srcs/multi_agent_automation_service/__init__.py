"""
Multi-Agent Automation Service
==============================

Python mcp_agent 라이브러리 기반 Multi-Agent 시스템
Gemini CLI를 통한 최종 명령 실행
"""

__version__ = "1.0.0"
__author__ = "AI Agent Development Team"

from .agents import (
    CodeReviewAgent,
    DocumentationAgent, 
    PerformanceTestAgent,
    SecurityDeploymentAgent
)

from .orchestrator import MultiAgentOrchestrator
from .gemini_cli_executor import GeminiCLIExecutor

__all__ = [
    "CodeReviewAgent",
    "DocumentationAgent",
    "PerformanceTestAgent", 
    "SecurityDeploymentAgent",
    "MultiAgentOrchestrator",
    "GeminiCLIExecutor"
] 