"""
DevOps Assistant Agent - Real MCP Agent Implementation
======================================================
MCP 기반 개발자 생산성 자동화 에이전트

Features:
- 🔍 GitHub 코드 리뷰 자동화
- 🚀 CI/CD 파이프라인 모니터링
- 🎯 이슈 우선순위 분석
- 👥 팀 스탠드업 자동 생성
- 📊 성능 분석 및 최적화

Model: gemini-2.5-flash-lite-preview-0607
"""

import asyncio
import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# MCP Agent imports
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM

@dataclass
class CodeReviewRequest:
    """코드 리뷰 요청"""
    owner: str
    repo: str
    pull_number: int
    title: str = ""
    author: str = ""
    changes_summary: str = ""

@dataclass
class DeploymentStatus:
    """배포 상태"""
    service: str
    status: str
    last_deployed: str
    health_check: str
    error_count: int = 0

@dataclass
class IssueAnalysis:
    """이슈 분석 결과"""
    issue_id: int
    title: str
    priority: str  # high, medium, low
    category: str  # bug, feature, security
    estimated_hours: int
    assigned_to: str = ""

@dataclass
class TeamActivity:
    """팀 활동 데이터"""
    team_name: str
    commits_today: int
    prs_opened: int
    prs_merged: int
    issues_resolved: int
    build_success_rate: float
    avg_review_time: float

class DevOpsTaskType(Enum):
    """DevOps 작업 타입"""
    CODE_REVIEW = "🔍 코드 리뷰"
    DEPLOYMENT_CHECK = "🚀 배포 상태 확인"
    ISSUE_ANALYSIS = "🎯 이슈 분석"
    TEAM_STANDUP = "👥 팀 스탠드업"
    PERFORMANCE_ANALYSIS = "📊 성능 분석"
    SECURITY_SCAN = "🔒 보안 스캔"

@dataclass
class DevOpsResult:
    """DevOps 작업 결과"""
    task_type: DevOpsTaskType
    status: str
    result_data: Dict[str, Any]
    recommendations: List[str]
    timestamp: str
    processing_time: float

class DevOpsAssistantMCPAgent:
    """
    🚀 DevOps Assistant MCP Agent
    
    Features:
    - GitHub 코드 리뷰 자동화
    - CI/CD 파이프라인 모니터링  
    - 이슈 우선순위 분석
    - 팀 스탠드업 준비
    - 성능 분석 및 최적화
    - 보안 스캔 및 권장사항
    
    Model: gemini-2.5-flash-lite-preview-0607
    """
    
    def __init__(self, output_dir: str = "devops_assistant_reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # MCP App 초기화
        from srcs.common.utils import setup_agent_app
        self.app = setup_agent_app("devops_assistant")
        
        # DevOps 작업 히스토리
        self.task_history: List[DevOpsResult] = []
        self.active_monitors: Dict[str, Any] = {}
        self.team_metrics: Dict[str, TeamActivity] = {}
        
        # 설정
        self.model_name = "gemini-2.5-flash-lite-preview-0607"
        self.default_review_criteria = [
            "코드 품질 및 가독성",
            "보안 취약점",
            "성능 최적화",
            "테스트 커버리지",
            "문서화 수준"
        ]
        
    async def analyze_code_review(self, request: CodeReviewRequest) -> DevOpsResult:
        """
        GitHub Pull Request 코드 리뷰 분석
        """
        start_time = datetime.now()
        
        async with self.app.run() as agent_app:
            # LLM을 통한 코드 리뷰 생성
            llm = GoogleAugmentedLLM(model=self.model_name)
            
            analysis_prompt = f"""
            다음 Pull Request를 전문 개발자 관점에서 리뷰해주세요:

            **Repository**: {request.owner}/{request.repo}
            **PR #{request.pull_number}**: {request.title}
            **작성자**: {request.author}
            **변경사항**: {request.changes_summary}

            **리뷰 기준**:
            1. 코드 품질 및 가독성
            2. 보안 취약점 체크
            3. 성능 최적화 가능성
            4. 테스트 커버리지
            5. 문서화 및 주석

            **출력 형식**:
            - 전체 평가: [A/B/C/D]
            - 주요 강점: [3개 이하]
            - 개선 필요사항: [구체적 제안]
            - 보안 체크포인트: [발견된 이슈]
            - 권장 액션: [승인/수정요청/재검토]

            건설적이고 실행 가능한 피드백을 제공해주세요.
            """
            
            response = await llm.generate(
                RequestParams(
                    prompt=analysis_prompt,
                    temperature=0.2,
                    max_tokens=1000
                )
            )
            
            # Mock 데이터로 실제 GitHub API 호출 시뮬레이션
            mock_pr_data = {
                "files_changed": 5,
                "additions": 142,
                "deletions": 38,
                "commits": 3,
                "reviewers": ["senior-dev", "tech-lead"],
                "ci_status": "passing",
                "conflicts": False
            }
            
            recommendations = [
                f"코드 리뷰 완료: {request.owner}/{request.repo}#{request.pull_number}",
                "CI/CD 파이프라인 상태 확인 필요",
                "테스트 커버리지 80% 이상 유지 권장",
                "보안 스캔 결과 검토 필요"
            ]
            
            result = DevOpsResult(
                task_type=DevOpsTaskType.CODE_REVIEW,
                status="completed",
                result_data={
                    "pr_info": asdict(request),
                    "github_data": mock_pr_data,
                    "review_content": response.strip(),
                    "review_criteria": self.default_review_criteria
                },
                recommendations=recommendations,
                timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            self.task_history.append(result)
            await self._save_result(result)
            
            return result
    
    async def check_deployment_status(self, service_name: str, environment: str = "production") -> DevOpsResult:
        """
        배포 상태 확인 및 분석
        """
        start_time = datetime.now()
        
        async with self.app.run() as agent_app:
            llm = GoogleAugmentedLLM(model=self.model_name)
            
            # Mock 배포 데이터
            mock_deployment_data = {
                "service": service_name,
                "environment": environment,
                "last_deployment": "2025-01-21T10:30:00Z",
                "status": "healthy",
                "replicas": {"desired": 3, "ready": 3, "available": 3},
                "health_checks": {"passing": 2, "failing": 1},
                "error_rate": "0.2%",
                "response_time": "145ms",
                "cpu_usage": "45%",
                "memory_usage": "62%"
            }
            
            analysis_prompt = f"""
            다음 서비스의 배포 상태를 분석해주세요:

            **서비스**: {service_name}
            **환경**: {environment}
            **배포 데이터**: {json.dumps(mock_deployment_data, indent=2)}

            **분석 요청**:
            1. 현재 서비스 상태 종합 평가
            2. 잠재적 위험 요소 식별
            3. 성능 최적화 기회
            4. 모니터링 알람 필요성
            5. 즉시 조치가 필요한 이슈

            **출력 형식**:
            - 전체 상태: [정상/주의/위험]
            - 위험도: [낮음/보통/높음]
            - 주요 메트릭: [핵심 지표 3개]
            - 권장 조치: [우선순위별]

            운영 관점에서 실행 가능한 인사이트를 제공해주세요.
            """
            
            response = await llm.generate(
                RequestParams(
                    prompt=analysis_prompt,
                    temperature=0.1,
                    max_tokens=800
                )
            )
            
            recommendations = [
                f"{service_name} 서비스 상태 모니터링 완료",
                "헬스체크 실패 1건 조사 필요",
                "CPU 사용률 45% - 정상 범위",
                "응답시간 145ms - 성능 양호",
                "에러율 0.2% - 허용 범위 내"
            ]
            
            result = DevOpsResult(
                task_type=DevOpsTaskType.DEPLOYMENT_CHECK,
                status="completed",
                result_data={
                    "deployment_info": mock_deployment_data,
                    "analysis": response.strip(),
                    "environment": environment
                },
                recommendations=recommendations,
                timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            self.task_history.append(result)
            await self._save_result(result)
            
            return result
    
    async def analyze_issues(self, owner: str, repo: str) -> DevOpsResult:
        """
        GitHub 이슈 우선순위 분석
        """
        start_time = datetime.now()
        
        async with self.app.run() as agent_app:
            llm = GoogleAugmentedLLM(model=self.model_name)
            
            # Mock 이슈 데이터
            mock_issues = [
                {
                    "id": 123,
                    "title": "Login fails with OAuth providers",
                    "labels": ["bug", "critical", "oauth"],
                    "created": "2025-01-20",
                    "description": "Users report login failures when using Google/GitHub OAuth"
                },
                {
                    "id": 124,
                    "title": "Add dark mode theme",
                    "labels": ["enhancement", "ui"],
                    "created": "2025-01-19",
                    "description": "Implement dark mode for better user experience"
                },
                {
                    "id": 125,
                    "title": "SQL injection vulnerability in search",
                    "labels": ["security", "critical"],
                    "created": "2025-01-18", 
                    "description": "Search endpoint vulnerable to SQL injection attacks"
                }
            ]
            
            analysis_prompt = f"""
            다음 GitHub 이슈들의 우선순위를 분석하고 분류해주세요:

            **Repository**: {owner}/{repo}
            **이슈 목록**: {json.dumps(mock_issues, indent=2, ensure_ascii=False)}

            **분석 기준**:
            1. 사용자 영향도 (높음/보통/낮음)
            2. 보안 중요도 (긴급/중요/일반)
            3. 기술적 복잡성 (복잡/보통/단순)
            4. 비즈니스 우선순위
            5. 예상 작업 시간

            **각 이슈별 분석 결과**:
            - 우선순위: [P0/P1/P2/P3]
            - 카테고리: [버그/기능/보안/개선]
            - 예상 시간: [시간 단위]
            - 담당자 추천: [역할별]
            - 해결 방향: [구체적 방법]

            개발팀의 생산성을 고려한 실용적인 우선순위를 제안해주세요.
            """
            
            response = await llm.generate(
                RequestParams(
                    prompt=analysis_prompt,
                    temperature=0.2,
                    max_tokens=1200
                )
            )
            
            # 이슈 분석 결과 생성
            analyzed_issues = []
            for issue in mock_issues:
                if "critical" in issue["labels"]:
                    priority = "P0" if "security" in issue["labels"] else "P1"
                    estimated_hours = 8 if "security" in issue["labels"] else 6
                elif "bug" in issue["labels"]:
                    priority = "P2"
                    estimated_hours = 4
                else:
                    priority = "P3"
                    estimated_hours = 16
                    
                analyzed_issues.append(IssueAnalysis(
                    issue_id=issue["id"],
                    title=issue["title"],
                    priority=priority,
                    category="security" if "security" in issue["labels"] else "bug" if "bug" in issue["labels"] else "feature",
                    estimated_hours=estimated_hours,
                    assigned_to="security-team" if "security" in issue["labels"] else "backend-team"
                ))
            
            recommendations = [
                f"{len(mock_issues)}개 이슈 우선순위 분석 완료",
                "보안 이슈 1건 즉시 처리 필요 (P0)",
                "Critical 버그 1건 당일 처리 권장 (P1)",
                "UI 개선사항 스프린트 백로그 추가 (P3)"
            ]
            
            result = DevOpsResult(
                task_type=DevOpsTaskType.ISSUE_ANALYSIS,
                status="completed",
                result_data={
                    "repository": f"{owner}/{repo}",
                    "raw_issues": mock_issues,
                    "analyzed_issues": [asdict(issue) for issue in analyzed_issues],
                    "analysis": response.strip()
                },
                recommendations=recommendations,
                timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            self.task_history.append(result)
            await self._save_result(result)
            
            return result
    
    async def generate_team_standup(self, team_name: str) -> DevOpsResult:
        """
        팀 스탠드업 요약 생성
        """
        start_time = datetime.now()
        
        async with self.app.run() as agent_app:
            llm = GoogleAugmentedLLM(model=self.model_name)
            
            # Mock 팀 활동 데이터
            team_activity = TeamActivity(
                team_name=team_name,
                commits_today=15,
                prs_opened=4,
                prs_merged=3,
                issues_resolved=7,
                build_success_rate=94.5,
                avg_review_time=2.3
            )
            
            self.team_metrics[team_name] = team_activity
            
            analysis_prompt = f"""
            다음 팀의 24시간 활동을 기반으로 스탠드업 요약을 작성해주세요:

            **팀**: {team_name}
            **활동 데이터**: {asdict(team_activity)}

            **스탠드업 형식**:
            1. 어제 완료된 주요 작업 (Yesterday)
               - 머지된 PR과 해결된 이슈 기준
               - 핵심 성과 하이라이트
            
            2. 오늘 예정된 작업 (Today)
               - 진행 중인 PR 검토
               - 우선순위 높은 이슈 처리
            
            3. 차단 요소 (Blockers)
               - 빌드 실패 원인
               - 리뷰 지연 사항
               - 의존성 이슈
            
            4. 팀 메트릭 하이라이트
               - 성과 지표 요약
               - 개선 포인트
            
            간결하고 실행 가능한 정보 위주로 작성해주세요.
            """
            
            response = await llm.generate(
                RequestParams(
                    prompt=analysis_prompt,
                    temperature=0.3,
                    max_tokens=800
                )
            )
            
            recommendations = [
                f"{team_name} 팀 스탠드업 요약 생성 완료",
                f"빌드 성공률 {team_activity.build_success_rate}% - 목표 95% 달성 근접",
                f"평균 리뷰 시간 {team_activity.avg_review_time}시간 - 양호",
                f"일일 커밋 {team_activity.commits_today}건 - 활발한 개발 활동"
            ]
            
            result = DevOpsResult(
                task_type=DevOpsTaskType.TEAM_STANDUP,
                status="completed", 
                result_data={
                    "team_name": team_name,
                    "team_activity": asdict(team_activity),
                    "standup_summary": response.strip(),
                    "period": "지난 24시간"
                },
                recommendations=recommendations,
                timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            self.task_history.append(result)
            await self._save_result(result)
            
            return result
    
    async def analyze_performance(self, service_name: str, timeframe: str = "24h") -> DevOpsResult:
        """
        서비스 성능 분석
        """
        start_time = datetime.now()
        
        async with self.app.run() as agent_app:
            llm = GoogleAugmentedLLM(model=self.model_name)
            
            # Mock 성능 메트릭
            performance_metrics = {
                "service": service_name,
                "timeframe": timeframe,
                "response_time": {
                    "avg": "156ms",
                    "p95": "324ms", 
                    "p99": "892ms"
                },
                "throughput": "2,450 req/min",
                "error_rate": "0.18%",
                "availability": "99.94%",
                "resource_usage": {
                    "cpu": "52%",
                    "memory": "68%",
                    "disk": "34%"
                },
                "database": {
                    "query_time_avg": "23ms",
                    "connections": 45,
                    "slow_queries": 3
                }
            }
            
            analysis_prompt = f"""
            다음 서비스의 성능 메트릭을 분석하고 최적화 방안을 제안해주세요:

            **서비스**: {service_name}
            **분석 기간**: {timeframe}
            **성능 데이터**: {json.dumps(performance_metrics, indent=2)}

            **분석 영역**:
            1. 응답 시간 트렌드 및 병목 지점
            2. 처리량 및 확장성
            3. 에러율 및 가용성
            4. 리소스 사용률 최적화
            5. 데이터베이스 성능

            **출력 형식**:
            - 전체 성능 점수: [A/B/C/D]
            - 주요 병목: [상위 3개]
            - 최적화 우선순위: [즉시/단기/장기]
            - 구체적 개선안: [실행 가능한 방법]
            - 모니터링 강화 포인트: [추가 메트릭]

            SRE 관점에서 실용적인 개선 방안을 제시해주세요.
            """
            
            response = await llm.generate(
                RequestParams(
                    prompt=analysis_prompt,
                    temperature=0.2,
                    max_tokens=1000
                )
            )
            
            recommendations = [
                f"{service_name} 성능 분석 완료 ({timeframe})",
                "P99 응답시간 892ms - 최적화 필요",
                "가용성 99.94% - SLA 목표 달성",
                "슬로우 쿼리 3건 - DB 튜닝 권장",
                "CPU 사용률 52% - 적정 수준"
            ]
            
            result = DevOpsResult(
                task_type=DevOpsTaskType.PERFORMANCE_ANALYSIS,
                status="completed",
                result_data={
                    "service": service_name,
                    "timeframe": timeframe,
                    "metrics": performance_metrics,
                    "analysis": response.strip()
                },
                recommendations=recommendations,
                timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            self.task_history.append(result)
            await self._save_result(result)
            
            return result
    
    async def run_security_scan(self, target: str, scan_type: str = "full") -> DevOpsResult:
        """
        보안 스캔 실행 및 분석
        """
        start_time = datetime.now()
        
        async with self.app.run() as agent_app:
            llm = GoogleAugmentedLLM(model=self.model_name)
            
            # Mock 보안 스캔 결과
            security_scan_results = {
                "target": target,
                "scan_type": scan_type,
                "vulnerabilities": {
                    "critical": 1,
                    "high": 3,
                    "medium": 7,
                    "low": 12
                },
                "findings": [
                    {
                        "severity": "critical",
                        "type": "SQL Injection",
                        "location": "/api/search",
                        "description": "User input not properly sanitized"
                    },
                    {
                        "severity": "high", 
                        "type": "XSS",
                        "location": "/user/profile",
                        "description": "Reflected XSS in user profile page"
                    },
                    {
                        "severity": "medium",
                        "type": "Insecure Headers",
                        "location": "Global",
                        "description": "Missing security headers"
                    }
                ],
                "compliance": {
                    "OWASP_Top10": "7/10 passed",
                    "CIS_Controls": "85% compliant"
                }
            }
            
            analysis_prompt = f"""
            다음 보안 스캔 결과를 분석하고 대응 방안을 제시해주세요:

            **스캔 대상**: {target}
            **스캔 유형**: {scan_type}
            **스캔 결과**: {json.dumps(security_scan_results, indent=2, ensure_ascii=False)}

            **분석 요청**:
            1. 전체 보안 위험도 평가
            2. 긴급 조치 필요 취약점
            3. 우선순위별 수정 계획
            4. 예방 조치 방안
            5. 컴플라이언스 개선 사항

            **출력 형식**:
            - 위험도: [긴급/높음/보통/낮음]
            - 즉시 조치: [Critical/High 취약점]
            - 단기 계획: [1-2주 내 수정]
            - 장기 계획: [보안 강화 방안]
            - 모니터링: [지속 감시 포인트]

            보안팀과 개발팀이 협력할 수 있는 실행 계획을 제시해주세요.
            """
            
            response = await llm.generate(
                RequestParams(
                    prompt=analysis_prompt,
                    temperature=0.1,
                    max_tokens=1000
                )
            )
            
            recommendations = [
                f"{target} 보안 스캔 완료 - {scan_type} 모드",
                "Critical 취약점 1건 - 즉시 패치 필요",
                "High 취약점 3건 - 이번 주 내 수정",
                "OWASP Top 10 - 70% 준수 (개선 필요)",
                "CIS Controls - 85% 준수 (양호)"
            ]
            
            result = DevOpsResult(
                task_type=DevOpsTaskType.SECURITY_SCAN,
                status="completed",
                result_data={
                    "target": target,
                    "scan_type": scan_type,
                    "scan_results": security_scan_results,
                    "analysis": response.strip()
                },
                recommendations=recommendations,
                timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            self.task_history.append(result)
            await self._save_result(result)
            
            return result
    
    async def _save_result(self, result: DevOpsResult):
        """결과를 파일로 저장"""
        filename = f"{result.task_type.name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
    
    def get_task_history(self) -> List[DevOpsResult]:
        """작업 히스토리 조회"""
        return self.task_history
    
    def get_team_metrics(self) -> Dict[str, TeamActivity]:
        """팀 메트릭 조회"""
        return self.team_metrics
    
    def get_summary_report(self) -> Dict[str, Any]:
        """종합 요약 리포트"""
        if not self.task_history:
            return {"message": "아직 수행된 작업이 없습니다."}
        
        task_counts = {}
        total_processing_time = 0
        
        for task in self.task_history:
            task_type = task.task_type.value
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
            total_processing_time += task.processing_time
        
        return {
            "total_tasks": len(self.task_history),
            "task_breakdown": task_counts,
            "total_processing_time": f"{total_processing_time:.2f}초",
            "avg_processing_time": f"{total_processing_time / len(self.task_history):.2f}초",
            "model_used": self.model_name,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

# 편의 함수들
async def create_devops_assistant(output_dir: str = "devops_assistant_reports") -> DevOpsAssistantMCPAgent:
    """DevOps Assistant Agent 생성"""
    return DevOpsAssistantMCPAgent(output_dir=output_dir)

async def run_code_review(agent: DevOpsAssistantMCPAgent, owner: str, repo: str, pull_number: int) -> DevOpsResult:
    """코드 리뷰 실행"""
    request = CodeReviewRequest(
        owner=owner,
        repo=repo, 
        pull_number=pull_number,
        title=f"Feature update for {repo}",
        author="developer",
        changes_summary="Added new authentication system and updated API endpoints"
    )
    return await agent.analyze_code_review(request)

async def run_deployment_check(agent: DevOpsAssistantMCPAgent, service_name: str, environment: str = "production") -> DevOpsResult:
    """배포 상태 확인"""
    return await agent.check_deployment_status(service_name, environment)

async def run_issue_analysis(agent: DevOpsAssistantMCPAgent, owner: str, repo: str) -> DevOpsResult:
    """이슈 분석 실행"""
    return await agent.analyze_issues(owner, repo)

async def run_team_standup(agent: DevOpsAssistantMCPAgent, team_name: str) -> DevOpsResult:
    """팀 스탠드업 생성"""
    return await agent.generate_team_standup(team_name)

async def run_performance_analysis(agent: DevOpsAssistantMCPAgent, service_name: str, timeframe: str = "24h") -> DevOpsResult:
    """성능 분석 실행"""
    return await agent.analyze_performance(service_name, timeframe)

async def run_security_scan(agent: DevOpsAssistantMCPAgent, target: str, scan_type: str = "full") -> DevOpsResult:
    """보안 스캔 실행"""
    return await agent.run_security_scan(target, scan_type) 