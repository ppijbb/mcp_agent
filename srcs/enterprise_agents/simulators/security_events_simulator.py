"""
Security Events Simulator

보안 이벤트 및 취약점 스캔 시뮬레이션.
실제 보안 이벤트 패턴과 공격 시그니처를 모방합니다.
"""

import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid

from srcs.common.simulation_utils import (
    ProbabilityDistributions, PatternGenerator, TimeSeriesGenerator
)


class SecurityEventsSimulator:
    """보안 이벤트 시뮬레이터"""
    
    def __init__(self, seed: Optional[int] = None):
        """초기화"""
        self.rng = random.Random(seed) if seed else random.Random()
        self.prob_dist = ProbabilityDistributions(seed)
        self.pattern_gen = PatternGenerator(seed)
        self.time_series_gen = TimeSeriesGenerator(seed)
    
    def generate_attack_patterns(
        self,
        start_time: datetime,
        duration_hours: int,
        attack_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        공격 패턴 생성 (실제 공격 시그니처 모방)
        
        Args:
            start_time: 시작 시간
            duration_hours: 지속 시간 (시간)
            attack_types: 공격 타입 리스트
        """
        if attack_types is None:
            attack_types = [
                "brute_force", "sql_injection", "xss", "ddos",
                "malware", "phishing", "unauthorized_access", "data_exfiltration"
            ]
        
        events = []
        duration_seconds = duration_hours * 3600
        
        # 공격 타입별 발생 빈도
        attack_rates = {
            "brute_force": 0.1,  # 시간당 0.1회
            "sql_injection": 0.05,
            "xss": 0.03,
            "ddos": 0.02,
            "malware": 0.01,
            "phishing": 0.08,
            "unauthorized_access": 0.04,
            "data_exfiltration": 0.005
        }
        
        current_time = start_time
        elapsed = 0
        
        while elapsed < duration_seconds:
            for attack_type in attack_types:
                rate = attack_rates.get(attack_type, 0.01)
                if self.rng.random() < (rate / 3600.0):  # 초당 확률로 변환
                    event = self._generate_attack_event(attack_type, current_time)
                    events.append(event)
            
            elapsed += 1
            current_time = start_time + timedelta(seconds=elapsed)
        
        return events
    
    def _generate_attack_event(self, attack_type: str, timestamp: datetime) -> Dict[str, Any]:
        """개별 공격 이벤트 생성"""
        event_id = f"sec-{uuid.uuid4().hex[:8]}"
        
        # 공격 타입별 특성
        attack_profiles = {
            "brute_force": {
                "severity": "medium",
                "source_ip": f"{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}",
                "target": f"user-{self.rng.randint(1, 100)}",
                "attempts": self.rng.randint(10, 1000),
                "description": "Multiple failed login attempts detected"
            },
            "sql_injection": {
                "severity": "high",
                "source_ip": f"{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}",
                "target": f"/api/endpoint-{self.rng.randint(1, 50)}",
                "payload": "SELECT * FROM users WHERE id=1 OR 1=1",
                "description": "SQL injection attempt detected in query parameters"
            },
            "xss": {
                "severity": "medium",
                "source_ip": f"{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}",
                "target": f"/page-{self.rng.randint(1, 20)}",
                "payload": "<script>alert('XSS')</script>",
                "description": "Cross-site scripting attempt detected"
            },
            "ddos": {
                "severity": "critical",
                "source_ips": [f"{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}" for _ in range(self.rng.randint(10, 100))],
                "target": f"server-{self.rng.randint(1, 10)}",
                "requests_per_second": self.rng.randint(1000, 10000),
                "description": "Distributed denial of service attack detected"
            },
            "malware": {
                "severity": "high",
                "source": f"file-{self.rng.randint(1, 100)}.exe",
                "target": f"host-{self.rng.randint(1, 50)}",
                "malware_type": self.rng.choice(["trojan", "ransomware", "spyware", "worm"]),
                "description": "Malware detected on system"
            },
            "phishing": {
                "severity": "medium",
                "source_email": f"attacker{self.rng.randint(1, 100)}@example.com",
                "target_email": f"user{self.rng.randint(1, 100)}@company.com",
                "subject": "Urgent: Verify your account",
                "description": "Phishing email detected"
            },
            "unauthorized_access": {
                "severity": "high",
                "source_ip": f"{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}",
                "target_resource": f"/admin/endpoint-{self.rng.randint(1, 20)}",
                "user": f"user-{self.rng.randint(1, 100)}",
                "description": "Unauthorized access attempt to restricted resource"
            },
            "data_exfiltration": {
                "severity": "critical",
                "source": f"host-{self.rng.randint(1, 50)}",
                "destination": f"{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}",
                "data_size_mb": self.rng.randint(10, 1000),
                "data_type": self.rng.choice(["customer_data", "financial_data", "intellectual_property"]),
                "description": "Suspicious data exfiltration detected"
            }
        }
        
        profile = attack_profiles.get(attack_type, {})
        
        event = {
            "event_id": event_id,
            "timestamp": timestamp.isoformat(),
            "attack_type": attack_type,
            "severity": profile.get("severity", "medium"),
            "description": profile.get("description", f"{attack_type} attack detected"),
            "status": self.rng.choice(["detected", "blocked", "investigating", "resolved"]),
            **{k: v for k, v in profile.items() if k not in ["severity", "description"]}
        }
        
        return event
    
    def simulate_vulnerability_scan(
        self,
        target_hosts: List[str],
        scan_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        취약점 스캔 시뮬레이션 (실제 스캐너 출력 형식)
        
        Args:
            target_hosts: 대상 호스트 리스트
            scan_type: 스캔 타입 ("quick", "comprehensive", "deep")
        """
        scan_id = f"scan-{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        
        # 스캔 타입별 소요 시간
        scan_durations = {
            "quick": 300,  # 5분
            "comprehensive": 1800,  # 30분
            "deep": 3600  # 1시간
        }
        duration_seconds = scan_durations.get(scan_type, 1800)
        
        # 취약점 데이터베이스
        vulnerability_db = [
            {"cve": "CVE-2023-1234", "severity": "critical", "name": "Remote Code Execution", "cvss": 9.8},
            {"cve": "CVE-2023-5678", "severity": "high", "name": "SQL Injection", "cvss": 8.5},
            {"cve": "CVE-2023-9012", "severity": "high", "name": "Cross-Site Scripting", "cvss": 7.8},
            {"cve": "CVE-2023-3456", "severity": "medium", "name": "Information Disclosure", "cvss": 5.5},
            {"cve": "CVE-2023-7890", "severity": "medium", "name": "Privilege Escalation", "cvss": 6.2},
            {"cve": "CVE-2023-2345", "severity": "low", "name": "Weak Encryption", "cvss": 3.5},
            {"cve": "CVE-2023-6789", "severity": "low", "name": "Information Leakage", "cvss": 2.8}
        ]
        
        scan_results = []
        
        for host in target_hosts:
            # 호스트별 취약점 수 (확률 분포)
            num_vulns = self.prob_dist.sample(
                ProbabilityDistributions.DistributionType.POISSON,
                lam=3.0
            )
            num_vulns = max(0, min(len(vulnerability_db), num_vulns))
            
            # 취약점 선택
            host_vulns = self.rng.sample(vulnerability_db, num_vulns)
            
            # 포트 스캔 결과
            open_ports = []
            common_ports = [22, 80, 443, 3306, 5432, 8080, 9200]
            for port in common_ports:
                if self.rng.random() < 0.5:  # 50% 확률로 포트 오픈
                    open_ports.append({
                        "port": port,
                        "protocol": "tcp",
                        "service": self._get_service_name(port),
                        "version": f"{self.rng.randint(1, 10)}.{self.rng.randint(0, 9)}"
                    })
            
            host_result = {
                "host": host,
                "status": "up",
                "open_ports": open_ports,
                "vulnerabilities": [
                    {
                        **vuln,
                        "port": self.rng.choice(open_ports)["port"] if open_ports else None,
                        "description": f"Vulnerability {vuln['name']} detected",
                        "recommendation": f"Apply patch for {vuln['cve']}"
                    }
                    for vuln in host_vulns
                ],
                "scan_duration_seconds": duration_seconds / len(target_hosts)
            }
            scan_results.append(host_result)
        
        # 전체 스캔 결과
        total_vulns = sum(len(r["vulnerabilities"]) for r in scan_results)
        critical_vulns = sum(1 for r in scan_results for v in r["vulnerabilities"] if v["severity"] == "critical")
        high_vulns = sum(1 for r in scan_results for v in r["vulnerabilities"] if v["severity"] == "high")
        
        scan_report = {
            "scan_id": scan_id,
            "scan_type": scan_type,
            "start_time": start_time.isoformat(),
            "end_time": (start_time + timedelta(seconds=duration_seconds)).isoformat(),
            "duration_seconds": duration_seconds,
            "targets": target_hosts,
            "results": scan_results,
            "summary": {
                "total_hosts": len(target_hosts),
                "hosts_scanned": len(target_hosts),
                "total_vulnerabilities": total_vulns,
                "critical": critical_vulns,
                "high": high_vulns,
                "medium": sum(1 for r in scan_results for v in r["vulnerabilities"] if v["severity"] == "medium"),
                "low": sum(1 for r in scan_results for v in r["vulnerabilities"] if v["severity"] == "low")
            }
        }
        
        return scan_report
    
    def _get_service_name(self, port: int) -> str:
        """포트 번호로 서비스 이름 반환"""
        port_services = {
            22: "ssh",
            80: "http",
            443: "https",
            3306: "mysql",
            5432: "postgresql",
            8080: "http-proxy",
            9200: "elasticsearch"
        }
        return port_services.get(port, "unknown")
    
    def generate_threat_intelligence(
        self,
        threat_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        위협 인텔리전스 데이터 생성
        
        Args:
            threat_types: 위협 타입 리스트
        """
        if threat_types is None:
            threat_types = [
                "malware", "phishing", "apt", "ransomware",
                "botnet", "zero_day", "insider_threat"
            ]
        
        threats = []
        
        for threat_type in threat_types:
            # 위협별 발생 빈도
            num_threats = self.prob_dist.sample(
                ProbabilityDistributions.DistributionType.POISSON,
                lam=2.0
            )
            
            for _ in range(num_threats):
                threat = self._generate_threat_intel(threat_type)
                threats.append(threat)
        
        return threats
    
    def _generate_threat_intel(self, threat_type: str) -> Dict[str, Any]:
        """개별 위협 인텔리전스 생성"""
        threat_id = f"threat-{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now() - timedelta(hours=self.rng.randint(0, 168))
        
        threat_profiles = {
            "malware": {
                "name": f"Malware-{self.rng.randint(1000, 9999)}",
                "description": "New malware variant detected in the wild",
                "indicators": [
                    f"hash-{uuid.uuid4().hex[:16]}",
                    f"{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}"
                ],
                "severity": "high"
            },
            "phishing": {
                "name": f"Phishing-Campaign-{self.rng.randint(1, 100)}",
                "description": "Active phishing campaign targeting organizations",
                "indicators": [
                    f"phishing-{self.rng.randint(1, 100)}.example.com",
                    f"attacker{self.rng.randint(1, 100)}@malicious.com"
                ],
                "severity": "medium"
            },
            "apt": {
                "name": f"APT-{self.rng.choice(['Lazarus', 'Fancy Bear', 'Cozy Bear'])}",
                "description": "Advanced Persistent Threat activity detected",
                "indicators": [
                    f"{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}",
                    f"apt-domain-{self.rng.randint(1, 100)}.com"
                ],
                "severity": "critical"
            },
            "ransomware": {
                "name": f"Ransomware-{self.rng.choice(['LockBit', 'BlackCat', 'REvil'])}",
                "description": "Ransomware attack pattern detected",
                "indicators": [
                    f"ransomware-{uuid.uuid4().hex[:16]}",
                    f"{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}"
                ],
                "severity": "critical"
            }
        }
        
        profile = threat_profiles.get(threat_type, {
            "name": f"Threat-{threat_type}",
            "description": f"{threat_type} threat detected",
            "indicators": [],
            "severity": "medium"
        })
        
        threat = {
            "threat_id": threat_id,
            "type": threat_type,
            "name": profile["name"],
            "description": profile["description"],
            "indicators": profile["indicators"],
            "severity": profile["severity"],
            "first_seen": timestamp.isoformat(),
            "last_seen": datetime.now().isoformat(),
            "affected_systems": self.rng.randint(1, 100),
            "confidence": self.rng.uniform(0.6, 1.0)
        }
        
        return threat
    
    def generate_security_logs(
        self,
        start_time: datetime,
        duration_hours: int,
        log_sources: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        보안 로그 생성 (실제 보안 로그 포맷 모방)
        
        Args:
            start_time: 시작 시간
            duration_hours: 지속 시간 (시간)
            log_sources: 로그 소스 리스트
        """
        if log_sources is None:
            log_sources = ["firewall", "ids", "waf", "auth", "audit"]
        
        logs = []
        duration_seconds = duration_hours * 3600
        
        # 로그 소스별 발생 빈도
        log_rates = {
            "firewall": 100,  # 시간당 100개
            "ids": 50,
            "waf": 30,
            "auth": 200,
            "audit": 20
        }
        
        current_time = start_time
        elapsed = 0
        
        while elapsed < duration_seconds:
            for source in log_sources:
                rate = log_rates.get(source, 10)
                if self.rng.random() < (rate / 3600.0):
                    log_entry = self._generate_security_log(source, current_time)
                    logs.append(log_entry)
            
            elapsed += 1
            current_time = start_time + timedelta(seconds=elapsed)
        
        return logs
    
    def _generate_security_log(self, source: str, timestamp: datetime) -> Dict[str, Any]:
        """개별 보안 로그 생성"""
        log_templates = {
            "firewall": {
                "format": "firewall",
                "message": f"Blocked connection from {self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)} to port {self.rng.choice([22, 80, 443, 3306])}",
                "action": "blocked",
                "protocol": "tcp"
            },
            "ids": {
                "format": "ids",
                "message": f"Intrusion detected: {self.rng.choice(['SQL injection', 'XSS', 'Brute force', 'Port scan'])}",
                "signature_id": self.rng.randint(1000, 9999),
                "severity": self.rng.choice(["low", "medium", "high"])
            },
            "waf": {
                "format": "waf",
                "message": f"WAF blocked request: {self.rng.choice(['Malicious payload', 'Suspicious pattern', 'Rate limit exceeded'])}",
                "rule_id": f"WAF-{self.rng.randint(1, 100)}",
                "action": "blocked"
            },
            "auth": {
                "format": "auth",
                "message": f"Authentication {'success' if self.rng.random() < 0.8 else 'failed'} for user user-{self.rng.randint(1, 100)}",
                "user": f"user-{self.rng.randint(1, 100)}",
                "result": self.rng.choice(["success", "failed"])
            },
            "audit": {
                "format": "audit",
                "message": f"Audit event: {self.rng.choice(['File access', 'Permission change', 'Configuration change', 'User creation'])}",
                "event_type": self.rng.choice(["access", "modify", "create", "delete"]),
                "resource": f"/path/to/resource-{self.rng.randint(1, 100)}"
            }
        }
        
        template = log_templates.get(source, {
            "format": "generic",
            "message": f"Security event from {source}"
        })
        
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "source": source,
            "level": self.rng.choice(["INFO", "WARNING", "ERROR"]),
            **template
        }
        
        return log_entry











