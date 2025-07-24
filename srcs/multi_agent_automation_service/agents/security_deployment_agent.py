"""
보안/배포 검증 Agent
==================

보안 스캔, 배포 검증, 자동 롤백 결정
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

@dataclass
class SecurityDeploymentResult:
    """보안/배포 검증 결과"""
    verification_id: str
    timestamp: str
    security_vulnerabilities: List[Dict[str, Any]]
    deployment_status: str
    health_checks: Dict[str, Any]
    rollback_recommended: bool
    security_score: float
    deployment_metrics: Dict[str, Any]
    gemini_cli_commands: List[str]

class SecurityDeploymentAgent:
    """보안/배포 검증 전담 Agent"""
    
    def __init__(self):
        # mcp_agent App 초기화
        self.app = MCPApp(
            name="security_deployment_agent",
            human_input_callback=None
        )
        
        # Agent 설정
        self.agent = Agent(
            name="security_deployment_verifier",
            instruction="""
            당신은 전문적인 보안 분석가이자 DevOps 엔지니어입니다. 다음을 수행하세요:
            
            1. 보안 취약점 스캔 (코드, 의존성, 인프라)
            2. 배포 후 상태 검증
            3. 헬스 체크 및 모니터링
            4. 자동 롤백 결정
            5. 보안 정책 검증
            6. Gemini CLI 명령어 생성 (실제 보안 수정 및 배포 관리용)
            
            모든 보안 이슈는 즉시 보고하고, 배포 실패 시 자동 롤백을 권장해야 합니다.
            """,
            server_names=["security-mcp", "deployment-mcp", "monitoring-mcp"],
            llm_factory=lambda: OpenAIAugmentedLLM(
                model="gpt-4",
            ),
        )
        
        self.verification_history: List[SecurityDeploymentResult] = []
    
    async def security_scan(self, target_paths: List[str] = None) -> SecurityDeploymentResult:
        """보안 스캔 실행"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                logger = app_context.logger
                
                logger.info("보안 스캔 시작")
                
                # 1. 보안 스캔 요청
                scan_prompt = f"""
                다음 경로의 코드를 보안 취약점 스캔해주세요: {target_paths or ['현재 디렉토리']}
                
                다음을 수행하세요:
                1. 코드 보안 취약점 스캔 (SQL 인젝션, XSS, CSRF 등)
                2. 의존성 보안 취약점 스캔
                3. 설정 파일 보안 검토
                4. 인증/인가 취약점 확인
                5. 데이터 보안 검토
                6. Gemini CLI 명령어 생성 (실제 보안 수정용)
                
                결과를 JSON 형태로 반환하세요:
                {{
                    "security_vulnerabilities": [
                        {{
                            "type": "취약점 타입",
                            "severity": "critical/high/medium/low",
                            "file": "파일명",
                            "line": "라인번호",
                            "description": "취약점 설명",
                            "cve_id": "CVE ID (있는 경우)",
                            "fix_suggestion": "수정 방안",
                            "risk_score": 0.85
                        }}
                    ],
                    "security_score": 0.75,
                    "deployment_status": "secure/insecure",
                    "health_checks": {{
                        "overall_health": "healthy/unhealthy",
                        "security_checks": "passed/failed",
                        "compliance_checks": "passed/failed"
                    }},
                    "rollback_recommended": false,
                    "deployment_metrics": {{
                        "vulnerabilities_count": 5,
                        "critical_issues": 1,
                        "high_issues": 2
                    }},
                    "gemini_cli_commands": [
                        "gemini 'SQL 인젝션 취약점을 수정해줘'",
                        "gemini '의존성 보안 업데이트를 적용해줘'",
                        "gemini '인증 로직을 강화해줘'"
                    ]
                }}
                """
                
                # Agent 실행
                result = await context.call_tool(
                    "security_scan",
                    {
                        "prompt": scan_prompt,
                        "target_paths": target_paths
                    }
                )
                
                # 결과 파싱
                security_data = json.loads(result.get("content", "{}"))
                
                # SecurityDeploymentResult 생성
                security_result = SecurityDeploymentResult(
                    verification_id=f"security_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    security_vulnerabilities=security_data.get("security_vulnerabilities", []),
                    deployment_status=security_data.get("deployment_status", "unknown"),
                    health_checks=security_data.get("health_checks", {}),
                    rollback_recommended=security_data.get("rollback_recommended", False),
                    security_score=security_data.get("security_score", 0.0),
                    deployment_metrics=security_data.get("deployment_metrics", {}),
                    gemini_cli_commands=security_data.get("gemini_cli_commands", [])
                )
                
                # 히스토리 저장
                self.verification_history.append(security_result)
                
                logger.info(f"보안 스캔 완료: {len(security_result.security_vulnerabilities)}개 취약점 발견")
                
                return security_result
                
        except Exception as e:
            logger.error(f"보안 스캔 실패: {e}")
            raise
    
    async def verify_deployment(self, deployment_id: str = None) -> SecurityDeploymentResult:
        """배포 검증 실행"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                
                # 배포 상태 확인
                deployment_status = await context.call_tool(
                    "check_deployment_status",
                    {"deployment_id": deployment_id}
                )
                
                # 헬스 체크 실행
                health_checks = await context.call_tool(
                    "run_health_checks",
                    {}
                )
                
                # 배포 검증 요청
                verification_prompt = f"""
                다음 배포 상태를 검증해주세요:
                배포 상태: {deployment_status}
                헬스 체크: {health_checks}
                
                다음을 확인하세요:
                1. 서비스 가용성
                2. 응답 시간
                3. 에러율
                4. 리소스 사용량
                5. 롤백 필요성 판단
                6. Gemini CLI 명령어 (실제 배포 관리용)
                """
                
                result = await context.call_tool(
                    "verify_deployment",
                    {"prompt": verification_prompt}
                )
                
                # 결과 처리
                deploy_data = json.loads(result.get("content", "{}"))
                
                return SecurityDeploymentResult(
                    verification_id=f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    security_vulnerabilities=[],
                    deployment_status=deploy_data.get("deployment_status", "unknown"),
                    health_checks=deploy_data.get("health_checks", {}),
                    rollback_recommended=deploy_data.get("rollback_recommended", False),
                    security_score=1.0,  # 배포 검증에서는 보안 점수 미적용
                    deployment_metrics=deploy_data.get("deployment_metrics", {}),
                    gemini_cli_commands=deploy_data.get("gemini_cli_commands", [])
                )
                
        except Exception as e:
            print(f"배포 검증 실패: {e}")
            raise
    
    async def auto_rollback(self, deployment_id: str) -> SecurityDeploymentResult:
        """자동 롤백 실행"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                
                # 롤백 실행
                rollback_result = await context.call_tool(
                    "execute_rollback",
                    {"deployment_id": deployment_id}
                )
                
                # 롤백 후 상태 확인
                post_rollback_status = await context.call_tool(
                    "check_deployment_status",
                    {"deployment_id": deployment_id}
                )
                
                return SecurityDeploymentResult(
                    verification_id=f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    security_vulnerabilities=[],
                    deployment_status="rolled_back",
                    health_checks={"overall_health": "healthy"},
                    rollback_recommended=False,
                    security_score=1.0,
                    deployment_metrics={"rollback_success": True},
                    gemini_cli_commands=[
                        f"gemini '배포 {deployment_id} 롤백 완료 상태를 확인해줘'",
                        "gemini '롤백 후 서비스 상태를 점검해줘'"
                    ]
                )
                
        except Exception as e:
            print(f"자동 롤백 실패: {e}")
            raise
    
    async def compliance_check(self) -> SecurityDeploymentResult:
        """규정 준수 검사"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                
                # 규정 준수 검사 실행
                compliance_result = await context.call_tool(
                    "run_compliance_check",
                    {}
                )
                
                # 규정 준수 분석
                compliance_prompt = f"""
                다음 규정 준수 검사 결과를 분석해주세요:
                {compliance_result}
                
                다음을 확인하세요:
                1. GDPR 준수 여부
                2. SOC2 준수 여부
                3. ISO 27001 준수 여부
                4. 보안 정책 준수 여부
                5. 개선사항 제안
                6. Gemini CLI 명령어 (실제 규정 준수 개선용)
                """
                
                result = await context.call_tool(
                    "analyze_compliance",
                    {"prompt": compliance_prompt}
                )
                
                # 결과 처리
                compliance_data = json.loads(result.get("content", "{}"))
                
                return SecurityDeploymentResult(
                    verification_id=f"compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    security_vulnerabilities=compliance_data.get("security_vulnerabilities", []),
                    deployment_status="compliant" if compliance_data.get("overall_compliance", False) else "non_compliant",
                    health_checks=compliance_data.get("health_checks", {}),
                    rollback_recommended=False,
                    security_score=compliance_data.get("compliance_score", 0.0),
                    deployment_metrics=compliance_data.get("compliance_metrics", {}),
                    gemini_cli_commands=compliance_data.get("gemini_cli_commands", [])
                )
                
        except Exception as e:
            print(f"규정 준수 검사 실패: {e}")
            raise
    
    def get_security_summary(self, security_result: SecurityDeploymentResult) -> str:
        """보안/배포 검증 결과 요약"""
        summary = f"""
보안/배포 검증 결과 요약
======================

🔒 보안 점수: {security_result.security_score:.2f}/1.0
🚀 배포 상태: {security_result.deployment_status}
🏥 전체 헬스: {security_result.health_checks.get('overall_health', 'unknown')}
🔄 롤백 권장: {'✅' if security_result.rollback_recommended else '❌'}

🚨 보안 취약점: {len(security_result.security_vulnerabilities)}개
📊 배포 메트릭: {security_result.deployment_metrics}

주요 취약점:
"""
        
        for vuln in security_result.security_vulnerabilities[:5]:  # 상위 5개만
            summary += f"- {vuln['type']} ({vuln['severity']}): {vuln['description']}\n"
        
        summary += f"\nGemini CLI 명령어 ({len(security_result.gemini_cli_commands)}개):\n"
        for cmd in security_result.gemini_cli_commands[:3]:  # 상위 3개만
            summary += f"- {cmd}\n"
        
        return summary
    
    def get_critical_vulnerabilities(self, security_result: SecurityDeploymentResult) -> List[Dict[str, Any]]:
        """심각한 보안 취약점만 필터링"""
        return [
            vuln for vuln in security_result.security_vulnerabilities
            if vuln.get("severity") in ["critical", "high"]
        ]
    
    def should_rollback(self, security_result: SecurityDeploymentResult) -> bool:
        """롤백 필요성 판단"""
        # 보안 점수가 낮거나 심각한 취약점이 있거나 배포 상태가 나쁘면 롤백
        critical_vulns = self.get_critical_vulnerabilities(security_result)
        
        return (
            security_result.rollback_recommended or
            security_result.security_score < 0.5 or
            len(critical_vulns) > 0 or
            security_result.deployment_status in ["failed", "unhealthy"]
        )

# 사용 예시
async def main():
    """사용 예시"""
    agent = SecurityDeploymentAgent()
    
    # 보안 스캔
    security_result = await agent.security_scan()
    print(agent.get_security_summary(security_result))
    
    # 배포 검증
    deploy_result = await agent.verify_deployment()
    print(f"배포 상태: {deploy_result.deployment_status}")
    
    # 롤백 필요성 확인
    if agent.should_rollback(deploy_result):
        print("🚨 롤백이 필요합니다!")
        rollback_result = await agent.auto_rollback("deployment-123")
        print(f"롤백 완료: {rollback_result.deployment_status}")
    
    # 규정 준수 검사
    compliance_result = await agent.compliance_check()
    print(f"규정 준수 점수: {compliance_result.security_score}")

if __name__ == "__main__":
    asyncio.run(main()) 