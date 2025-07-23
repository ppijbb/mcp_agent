"""
보안 Agent
==========

보안 정책 및 규정 준수를 관리하는 Agent
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

@dataclass
class SecurityPolicy:
    """보안 정책"""
    name: str
    type: str  # pod_security, network_policy, rbac, etc.
    rules: List[str]
    severity: str  # low, medium, high, critical
    enabled: bool = True

@dataclass
class SecurityViolation:
    """보안 위반"""
    id: str
    policy_name: str
    resource_name: str
    violation_type: str
    severity: str
    description: str
    timestamp: datetime
    status: str  # open, resolved, false_positive

@dataclass
class SecurityReport:
    """보안 보고서"""
    timestamp: datetime
    total_violations: int
    critical_violations: int
    high_violations: int
    medium_violations: int
    low_violations: int
    compliance_score: float
    recommendations: List[str]

class SecurityAgent:
    """mcp_agent 기반 보안 Agent"""
    
    def __init__(self, output_dir: str = "security_reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # mcp_agent App 초기화
        self.app = MCPApp(
            name="security_agent",
            human_input_callback=None
        )
        
        # Agent 설정
        self.agent = Agent(
            name="security_agent",
            instruction="보안 정책 및 규정 준수를 관리하는 전문 Agent입니다.",
            server_names=["security-mcp", "k8s-mcp"],
            llm_factory=lambda: OpenAIAugmentedLLM(
                model="gpt-4",
            ),
        )
        
        # 보안 상태
        self.security_policies: List[SecurityPolicy] = []
        self.violations: List[SecurityViolation] = []
        self.compliance_history: List[SecurityReport] = []
        
        # 기본 보안 정책 설정
        self._setup_default_security_policies()
    
    def _setup_default_security_policies(self):
        """기본 보안 정책 설정"""
        self.security_policies = [
            SecurityPolicy(
                name="pod_security_standards",
                type="pod_security",
                rules=[
                    "Do not run containers as root",
                    "Do not allow privilege escalation",
                    "Use read-only root filesystem"
                ],
                severity="high"
            ),
            SecurityPolicy(
                name="network_policies",
                type="network_policy",
                rules=[
                    "Default deny all ingress traffic",
                    "Default deny all egress traffic",
                    "Allow only necessary ports"
                ],
                severity="medium"
            ),
            SecurityPolicy(
                name="rbac_policies",
                type="rbac",
                rules=[
                    "Use least privilege principle",
                    "Regular role audits",
                    "No cluster-admin for regular users"
                ],
                severity="high"
            )
        ]
    
    async def start_security_monitoring(self, namespace: str = "default"):
        """보안 모니터링 시작"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                logger = app_context.logger
                
                logger.info(f"Starting security monitoring for namespace: {namespace}")
                
                # 1. 보안 스캔 실행
                await self._run_security_scan(namespace, context)
                
                # 2. 취약점 평가
                await self._assess_vulnerabilities(namespace, context)
                
                # 3. 규정 준수 검사
                await self._check_compliance(namespace, context)
                
                # 4. 실시간 위협 감지 시작
                await self._start_threat_detection(context)
                
                logger.info("Security monitoring started successfully")
                
        except Exception as e:
            logger.error(f"Failed to start security monitoring: {e}")
            raise
    
    async def _run_security_scan(self, namespace: str, context):
        """보안 스캔 실행"""
        logger = context.logger
        
        # Pod 보안 스캔
        pod_violations = await self._scan_pod_security(namespace, context)
        self.violations.extend(pod_violations)
        
        # 네트워크 정책 스캔
        network_violations = await self._scan_network_policies(namespace, context)
        self.violations.extend(network_violations)
        
        # RBAC 스캔
        rbac_violations = await self._scan_rbac_policies(namespace, context)
        self.violations.extend(rbac_violations)
        
        # Secret 스캔
        secret_violations = await self._scan_secrets(namespace, context)
        self.violations.extend(secret_violations)
        
        logger.info(f"Security scan completed. Found {len(self.violations)} violations")
    
    async def _scan_pod_security(self, namespace: str, context) -> List[SecurityViolation]:
        """Pod 보안 스캔"""
        violations = []
        
        # Pod 목록 조회
        pods = await self._get_pods(namespace, context)
        
        for pod in pods:
            # root 사용자 체크
            if pod.get("run_as_root", False):
                violations.append(SecurityViolation(
                    id=f"pod_root_{pod['name']}",
                    policy_name="pod_security_standards",
                    resource_name=pod["name"],
                    violation_type="root_user",
                    severity="high",
                    description="Pod is running as root user",
                    timestamp=datetime.now(),
                    status="open"
                ))
            
            # 권한 상승 체크
            if pod.get("allow_privilege_escalation", False):
                violations.append(SecurityViolation(
                    id=f"pod_privilege_{pod['name']}",
                    policy_name="pod_security_standards",
                    resource_name=pod["name"],
                    violation_type="privilege_escalation",
                    severity="high",
                    description="Pod allows privilege escalation",
                    timestamp=datetime.now(),
                    status="open"
                ))
        
        return violations
    
    async def _scan_network_policies(self, namespace: str, context) -> List[SecurityViolation]:
        """네트워크 정책 스캔"""
        violations = []
        
        # 네트워크 정책 조회
        network_policies = await self._get_network_policies(namespace, context)
        
        # 기본 거부 정책 체크
        if not network_policies:
            violations.append(SecurityViolation(
                id=f"network_policy_missing_{namespace}",
                policy_name="network_policies",
                resource_name=namespace,
                violation_type="missing_policy",
                severity="medium",
                description="No network policies defined",
                timestamp=datetime.now(),
                status="open"
            ))
        
        return violations
    
    async def _scan_rbac_policies(self, namespace: str, context) -> List[SecurityViolation]:
        """RBAC 정책 스캔"""
        violations = []
        
        # RBAC 규칙 조회
        rbac_rules = await self._get_rbac_rules(namespace, context)
        
        # 과도한 권한 체크
        for rule in rbac_rules:
            if rule.get("role") == "cluster-admin":
                violations.append(SecurityViolation(
                    id=f"rbac_cluster_admin_{rule['user']}",
                    policy_name="rbac_policies",
                    resource_name=rule["user"],
                    violation_type="excessive_privileges",
                    severity="critical",
                    description="User has cluster-admin privileges",
                    timestamp=datetime.now(),
                    status="open"
                ))
        
        return violations
    
    async def _scan_secrets(self, namespace: str, context) -> List[SecurityViolation]:
        """Secret 스캔"""
        violations = []
        
        # Secret 목록 조회
        secrets = await self._get_secrets(namespace, context)
        
        for secret in secrets:
            # 만료된 Secret 체크
            if secret.get("expired", False):
                violations.append(SecurityViolation(
                    id=f"secret_expired_{secret['name']}",
                    policy_name="secret_management",
                    resource_name=secret["name"],
                    violation_type="expired_secret",
                    severity="medium",
                    description="Secret has expired",
                    timestamp=datetime.now(),
                    status="open"
                ))
        
        return violations
    
    async def _assess_vulnerabilities(self, namespace: str, context):
        """취약점 평가"""
        logger = context.logger
        
        # 컨테이너 이미지 취약점 스캔
        vulnerabilities = await self._scan_container_vulnerabilities(namespace, context)
        
        # 취약점을 위반으로 변환
        for vuln in vulnerabilities:
            self.violations.append(SecurityViolation(
                id=f"vuln_{vuln['cve']}",
                policy_name="vulnerability_management",
                resource_name=vuln["image"],
                violation_type="vulnerability",
                severity=vuln["severity"],
                description=f"Vulnerability found: {vuln['cve']} - {vuln['description']}",
                timestamp=datetime.now(),
                status="open"
            ))
        
        logger.info(f"Vulnerability assessment completed. Found {len(vulnerabilities)} vulnerabilities")
    
    async def _check_compliance(self, namespace: str, context):
        """규정 준수 검사"""
        logger = context.logger
        
        # CIS Kubernetes 벤치마크 체크
        cis_compliance = await self._check_cis_compliance(context)
        
        # GDPR 준수 체크 (데이터 관련)
        gdpr_compliance = await self._check_gdpr_compliance(namespace, context)
        
        # SOX 준수 체크 (재무 관련)
        sox_compliance = await self._check_sox_compliance(context)
        
        # 준수 점수 계산
        compliance_score = self._calculate_compliance_score()
        
        # 보안 보고서 생성
        report = SecurityReport(
            timestamp=datetime.now(),
            total_violations=len(self.violations),
            critical_violations=len([v for v in self.violations if v.severity == "critical"]),
            high_violations=len([v for v in self.violations if v.severity == "high"]),
            medium_violations=len([v for v in self.violations if v.severity == "medium"]),
            low_violations=len([v for v in self.violations if v.severity == "low"]),
            compliance_score=compliance_score,
            recommendations=self._generate_security_recommendations()
        )
        
        self.compliance_history.append(report)
        await self._save_security_report(report)
        
        logger.info(f"Compliance check completed. Score: {compliance_score}%")
    
    async def _start_threat_detection(self, context):
        """실시간 위협 감지 시작"""
        logger = context.logger
        
        # 주기적 위협 감지 태스크 시작
        asyncio.create_task(self._periodic_threat_detection(context))
        
        logger.info("Real-time threat detection started")
    
    async def _periodic_threat_detection(self, context):
        """주기적 위협 감지"""
        logger = context.logger
        
        while True:
            try:
                # 비정상적인 활동 감지
                anomalies = await self._detect_anomalies(context)
                
                # 위협 지표 분석
                threats = await self._analyze_threat_indicators(context)
                
                # 새로운 위반 사항 처리
                for anomaly in anomalies:
                    await self._handle_security_anomaly(anomaly, context)
                
                for threat in threats:
                    await self._handle_security_threat(threat, context)
                
                logger.debug("Threat detection cycle completed")
                
                # 30초 대기
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Threat detection error: {e}")
                await asyncio.sleep(60)
    
    def _calculate_compliance_score(self) -> float:
        """준수 점수 계산"""
        if not self.violations:
            return 100.0
        
        total_violations = len(self.violations)
        critical_violations = len([v for v in self.violations if v.severity == "critical"])
        high_violations = len([v for v in self.violations if v.severity == "high"])
        
        # 가중 점수 계산
        score = 100.0
        score -= critical_violations * 10  # critical 위반당 10점 차감
        score -= high_violations * 5       # high 위반당 5점 차감
        score -= (total_violations - critical_violations - high_violations) * 2  # 나머지 위반당 2점 차감
        
        return max(0.0, score)
    
    def _generate_security_recommendations(self) -> List[str]:
        """보안 권장사항 생성"""
        recommendations = []
        
        critical_violations = [v for v in self.violations if v.severity == "critical"]
        high_violations = [v for v in self.violations if v.severity == "high"]
        
        if critical_violations:
            recommendations.append("Immediately address critical security violations")
        
        if high_violations:
            recommendations.append("Prioritize fixing high severity violations")
        
        if len([v for v in self.violations if v.violation_type == "root_user"]) > 0:
            recommendations.append("Remove root user privileges from pods")
        
        if len([v for v in self.violations if v.violation_type == "missing_policy"]) > 0:
            recommendations.append("Implement missing network policies")
        
        if len([v for v in self.violations if v.violation_type == "excessive_privileges"]) > 0:
            recommendations.append("Review and reduce excessive RBAC privileges")
        
        return recommendations
    
    async def _save_security_report(self, report: SecurityReport):
        """보안 보고서 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"security_report_{timestamp}.json")
        
        report_data = {
            "security_report": asdict(report),
            "violations": [asdict(v) for v in self.violations],
            "policies": [asdict(p) for p in self.security_policies]
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"Security report saved to: {report_file}")
    
    # 시뮬레이션 메서드들
    async def _get_pods(self, namespace: str, context) -> List[Dict[str, Any]]:
        """Pod 목록 조회"""
        return [
            {"name": "app-pod-1", "run_as_root": False, "allow_privilege_escalation": False},
            {"name": "app-pod-2", "run_as_root": True, "allow_privilege_escalation": False},
            {"name": "app-pod-3", "run_as_root": False, "allow_privilege_escalation": True}
        ]
    
    async def _get_network_policies(self, namespace: str, context) -> List[Dict[str, Any]]:
        """네트워크 정책 조회"""
        return []  # 빈 목록으로 시뮬레이션
    
    async def _get_rbac_rules(self, namespace: str, context) -> List[Dict[str, Any]]:
        """RBAC 규칙 조회"""
        return [
            {"user": "admin", "role": "cluster-admin"},
            {"user": "developer", "role": "developer"}
        ]
    
    async def _get_secrets(self, namespace: str, context) -> List[Dict[str, Any]]:
        """Secret 목록 조회"""
        return [
            {"name": "db-secret", "expired": False},
            {"name": "api-key", "expired": True}
        ]
    
    async def _scan_container_vulnerabilities(self, namespace: str, context) -> List[Dict[str, Any]]:
        """컨테이너 취약점 스캔"""
        return [
            {
                "cve": "CVE-2023-1234",
                "severity": "high",
                "description": "Remote code execution vulnerability",
                "image": "nginx:1.19"
            }
        ]
    
    async def _check_cis_compliance(self, context) -> Dict[str, Any]:
        """CIS 준수 체크"""
        return {"compliant": True, "score": 85}
    
    async def _check_gdpr_compliance(self, namespace: str, context) -> Dict[str, Any]:
        """GDPR 준수 체크"""
        return {"compliant": True, "data_encryption": True}
    
    async def _check_sox_compliance(self, context) -> Dict[str, Any]:
        """SOX 준수 체크"""
        return {"compliant": True, "audit_logs": True}
    
    async def _detect_anomalies(self, context) -> List[Dict[str, Any]]:
        """비정상 활동 감지"""
        return []  # 시뮬레이션
    
    async def _analyze_threat_indicators(self, context) -> List[Dict[str, Any]]:
        """위협 지표 분석"""
        return []  # 시뮬레이션
    
    async def _handle_security_anomaly(self, anomaly: Dict[str, Any], context):
        """보안 비정상 처리"""
        print(f"🚨 Security anomaly detected: {anomaly}")
    
    async def _handle_security_threat(self, threat: Dict[str, Any], context):
        """보안 위협 처리"""
        print(f"⚠️ Security threat detected: {threat}")
    
    async def get_security_violations(self) -> List[SecurityViolation]:
        """보안 위반 조회"""
        return self.violations
    
    async def get_compliance_history(self) -> List[SecurityReport]:
        """준수 히스토리 조회"""
        return self.compliance_history
    
    async def add_security_policy(self, policy: SecurityPolicy):
        """보안 정책 추가"""
        self.security_policies.append(policy)
    
    async def resolve_violation(self, violation_id: str):
        """위반 해결"""
        for violation in self.violations:
            if violation.id == violation_id:
                violation.status = "resolved"
                break

# 사용 예시
async def main():
    """사용 예시"""
    agent = SecurityAgent()
    
    # 보안 모니터링 시작
    await agent.start_security_monitoring("default")
    
    # 10초 대기
    await asyncio.sleep(10)
    
    # 위반 사항 조회
    violations = await agent.get_security_violations()
    print(f"Security violations found: {len(violations)}")
    
    # 준수 히스토리 조회
    history = await agent.get_compliance_history()
    if history:
        latest = history[-1]
        print(f"Latest compliance score: {latest.compliance_score}%")

if __name__ == "__main__":
    asyncio.run(main()) 