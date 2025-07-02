"""
DevOps Productivity Agent Package
=================================
MCP 기반 개발자 생산성 자동화 에이전트

Components:
- DevOpsAssistantMCPAgent: 핵심 에이전트 클래스
- 6가지 주요 기능: 코드리뷰, 배포확인, 이슈분석, 스탠드업, 성능분석, 보안스캔
- 대화형 실행기 및 편의 함수들

Model: gemini-2.5-flash-lite-preview-0607
"""

from .agents.devops_assistant_agent import (
    DevOpsAssistantMCPAgent,
    DevOpsTaskType,
    DevOpsResult,
    CodeReviewRequest,
    DeploymentStatus,
    IssueAnalysis,
    TeamActivity,
    create_devops_assistant,
    run_code_review,
    run_deployment_check,
    run_issue_analysis,
    run_team_standup,
    run_performance_analysis,
    run_security_scan
)

__version__ = "1.0.0"
__author__ = "DevOps Team"
__description__ = "MCP 기반 개발자 생산성 자동화 에이전트"

__all__ = [
    "DevOpsAssistantMCPAgent",
    "DevOpsTaskType", 
    "DevOpsResult",
    "CodeReviewRequest",
    "DeploymentStatus",
    "IssueAnalysis",
    "TeamActivity",
    "create_devops_assistant",
    "run_code_review",
    "run_deployment_check",
    "run_issue_analysis",
    "run_team_standup", 
    "run_performance_analysis",
    "run_security_scan"
] 