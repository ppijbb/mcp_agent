"""
Security Agent

실제 mcp_agent 라이브러리를 사용한 보안 검증 및 배포 검증 전문 Agent입니다.
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from srcs.common.utils import setup_agent_app, save_report


@dataclass
class SecurityResult:
    """보안 검증 결과"""
    target_path: str
    scan_type: str  # SECURITY_SCAN, DEPLOYMENT_VERIFICATION, COMPLIANCE_CHECK
    vulnerabilities: List[Dict[str, Any]]
    recommendations: List[str]
    gemini_commands: List[str]
    should_rollback: bool
    timestamp: datetime


class SecurityAgent:
    """보안 검증 및 배포 검증 전담 Agent - 실제 mcp_agent 표준 사용"""
    
    def __init__(self):
        self.app = setup_agent_app("security_system")
        self.agent = Agent(
            name="security_analyzer",
            instruction="""
            당신은 전문적인 보안 분석가입니다. 다음을 수행하세요:
            
            1. 코드 보안 취약점 스캔 및 분석
            2. 의존성 보안 검사
            3. 배포 전 보안 검증
            4. 규정 준수 검사
            5. 보안 문제 해결을 위한 Gemini CLI 명령어 생성
            6. 필요시 자동 롤백 권고
            
            MCP 서버의 도구들을 활용하여 실제 코드를 분석하고,
            보안 위험을 식별하고 해결 방안을 제시하세요.
            """,
            server_names=["filesystem", "github"],  # 실제 MCP 서버명
        )
        self.security_history: List[SecurityResult] = []
    
    async def security_scan(self, target_path: str = "srcs") -> SecurityResult:
        """보안 스캔 수행"""
        async with self.app.run() as app_context:
            context = app_context.context
            logger = app_context.logger
            
            # 파일시스템 서버 설정
            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
                logger.info("Filesystem server configured")
            
            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
                
                # 보안 스캔 수행
                scan_prompt = f"""
                다음 경로의 코드를 보안 관점에서 스캔하세요: {target_path}
                
                다음을 검사하세요:
                1. SQL 인젝션 취약점
                2. XSS 취약점
                3. 인증/인가 취약점
                4. 민감 정보 노출
                5. 의존성 보안 취약점
                6. 암호화 관련 문제
                7. 입력 검증 부족
                8. 로깅 및 모니터링 부족
                
                각 취약점에 대한 구체적인 Gemini CLI 명령어를 생성하세요.
                """
                
                result = await llm.generate_str(
                    message=scan_prompt,
                    request_params=RequestParams(model="gpt-4o")
                )
                
                # 결과 파싱 및 구조화
                security_result = self._parse_security_result(result, target_path, "SECURITY_SCAN")
                self.security_history.append(security_result)
                
                return security_result
    
    async def verify_deployment(self, deployment_path: str = ".") -> SecurityResult:
        """배포 검증"""
        async with self.app.run() as app_context:
            context = app_context.context
            
            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            
            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
                
                prompt = f"""
                다음 배포를 검증하세요: {deployment_path}
                
                다음을 확인하세요:
                1. 환경 변수 보안 설정
                2. 네트워크 보안 설정
                3. 파일 권한 설정
                4. 로그 보안 설정
                5. 백업 및 복구 설정
                6. 모니터링 및 알림 설정
                7. 접근 제어 설정
                
                배포 검증을 위한 Gemini CLI 명령어를 생성하세요.
                """
                
                result = await llm.generate_str(
                    message=prompt,
                    request_params=RequestParams(model="gpt-4o")
                )
                
                security_result = self._parse_security_result(result, deployment_path, "DEPLOYMENT_VERIFICATION")
                self.security_history.append(security_result)
                
                return security_result
    
    async def compliance_check(self, target_path: str = "srcs") -> SecurityResult:
        """규정 준수 검사"""
        async with self.app.run() as app_context:
            context = app_context.context
            
            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            
            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
                
                prompt = f"""
                다음 경로의 코드에 대한 규정 준수를 검사하세요: {target_path}
                
                다음 규정을 확인하세요:
                1. GDPR (개인정보보호)
                2. SOX (회계 규정)
                3. HIPAA (의료정보보호)
                4. PCI-DSS (결제카드 보안)
                5. ISO 27001 (정보보안)
                6. NIST (국가표준)
                
                규정 준수 검사를 위한 Gemini CLI 명령어를 생성하세요.
                """
                
                result = await llm.generate_str(
                    message=prompt,
                    request_params=RequestParams(model="gpt-4o")
                )
                
                security_result = self._parse_security_result(result, target_path, "COMPLIANCE_CHECK")
                self.security_history.append(security_result)
                
                return security_result
    
    async def auto_rollback(self, deployment_id: str = "latest") -> SecurityResult:
        """자동 롤백 수행"""
        async with self.app.run() as app_context:
            context = app_context.context
            
            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            
            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
                
                prompt = f"""
                배포 {deployment_id}에 대한 롤백을 수행하세요.
                
                다음을 확인하세요:
                1. 롤백 대상 확인
                2. 롤백 전 백업 상태 확인
                3. 롤백 실행
                4. 롤백 후 검증
                5. 롤백 결과 보고
                
                롤백을 위한 Gemini CLI 명령어를 생성하세요.
                """
                
                result = await llm.generate_str(
                    message=prompt,
                    request_params=RequestParams(model="gpt-4o")
                )
                
                security_result = self._parse_security_result(result, f"rollback_{deployment_id}", "ROLLBACK")
                security_result.should_rollback = True
                self.security_history.append(security_result)
                
                return security_result
    
    def get_security_summary(self) -> Dict[str, Any]:
        """보안 검증 요약 정보"""
        if not self.security_history:
            return {"message": "No security scans performed yet"}
        
        scan_types = {}
        total_vulnerabilities = 0
        critical_vulnerabilities = 0
        rollback_count = 0
        
        for result in self.security_history:
            scan_type = result.scan_type
            if scan_type not in scan_types:
                scan_types[scan_type] = 0
            scan_types[scan_type] += 1
            
            total_vulnerabilities += len(result.vulnerabilities)
            critical_vulnerabilities += sum(1 for vuln in result.vulnerabilities 
                                          if vuln.get("severity") == "CRITICAL")
            
            if result.should_rollback:
                rollback_count += 1
        
        return {
            "total_scans": len(self.security_history),
            "scan_types": scan_types,
            "total_vulnerabilities": total_vulnerabilities,
            "critical_vulnerabilities": critical_vulnerabilities,
            "rollback_count": rollback_count,
            "recent_scans": [
                {
                    "target_path": result.target_path,
                    "scan_type": result.scan_type,
                    "vulnerabilities_count": len(result.vulnerabilities),
                    "should_rollback": result.should_rollback,
                    "timestamp": result.timestamp.isoformat()
                }
                for result in self.security_history[-5:]  # 최근 5개
            ]
        }
    
    def get_critical_vulnerabilities(self) -> List[Dict[str, Any]]:
        """중요한 취약점 식별"""
        critical_vulns = []
        
        for result in self.security_history:
            for vuln in result.vulnerabilities:
                if vuln.get("severity") in ["HIGH", "CRITICAL"]:
                    critical_vulns.append({
                        "target_path": result.target_path,
                        "vulnerability": vuln,
                        "scan_type": result.scan_type,
                        "timestamp": result.timestamp.isoformat()
                    })
        
        return critical_vulns
    
    def should_rollback(self, threshold: int = 3) -> bool:
        """롤백 필요 여부 판단"""
        recent_scans = self.security_history[-5:]  # 최근 5개 스캔
        critical_count = 0
        
        for result in recent_scans:
            critical_count += sum(1 for vuln in result.vulnerabilities 
                                if vuln.get("severity") == "CRITICAL")
        
        return critical_count >= threshold
    
    def _parse_security_result(self, result: str, target_path: str, scan_type: str) -> SecurityResult:
        """보안 검증 결과 파싱"""
        # 실제 구현에서는 더 정교한 파싱 로직 필요
        vulnerabilities = []
        recommendations = []
        gemini_commands = []
        
        # 간단한 파싱 예시
        lines = result.split('\n')
        current_section = None
        
        for line in lines:
            if "## 취약점" in line or "## Vulnerabilities" in line:
                current_section = "vulnerabilities"
            elif "## 권장사항" in line or "## Recommendations" in line:
                current_section = "recommendations"
            elif "## Gemini CLI 명령어" in line:
                current_section = "commands"
            elif line.strip().startswith('-') and current_section:
                content = line.strip()[1:].strip()
                if current_section == "vulnerabilities":
                    vulnerabilities.append({"description": content, "severity": "MEDIUM"})
                elif current_section == "recommendations":
                    recommendations.append(content)
                elif current_section == "commands":
                    gemini_commands.append(content)
        
        # 롤백 필요 여부 판단
        should_rollback = any("CRITICAL" in vuln.get("description", "") 
                             for vuln in vulnerabilities)
        
        return SecurityResult(
            target_path=target_path,
            scan_type=scan_type,
            vulnerabilities=vulnerabilities,
            recommendations=recommendations,
            gemini_commands=gemini_commands,
            should_rollback=should_rollback,
            timestamp=datetime.now()
        )


async def main():
    """테스트 실행"""
    agent = SecurityAgent()
    
    # 보안 스캔
    result = await agent.security_scan()
    print(f"Security scan completed for: {result.target_path}")
    print(f"Found {len(result.vulnerabilities)} vulnerabilities")
    print(f"Generated {len(result.gemini_commands)} Gemini CLI commands")
    print(f"Should rollback: {result.should_rollback}")
    
    # 요약 정보
    summary = agent.get_security_summary()
    print(f"Security summary: {summary}")
    
    # 중요 취약점
    critical_vulns = agent.get_critical_vulnerabilities()
    print(f"Critical vulnerabilities: {len(critical_vulns)}")


if __name__ == "__main__":
    asyncio.run(main()) 