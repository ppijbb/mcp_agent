# DevOps Productivity Agent 아키텍처

## 개요

DevOps Productivity Agent는 Model Context Protocol (MCP) 기반의 Agentic 아키텍처를 사용하여 멀티클라우드 DevOps 작업을 자동화합니다. 이 에이전트는 LLM이 직접 판단하고 적절한 MCP 도구를 선택하여 복잡한 DevOps 작업을 수행합니다.

## 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                    DevOps Productivity Agent                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   BaseAgent     │    │   Orchestrator  │    │ Gemini 2.5   │ │
│  │   (MCP Core)    │◄──►│   (Tool Select) │◄──►│ Flash Latest │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Servers                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ AWS Knowledge│  │   GitHub    │  │ Prometheus  │  │Kubernetes│ │
│  │    Base     │  │ Operations  │  │   Metrics   │  │ Cluster │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
│  ┌─────────────┐  ┌─────────────┐                              │
│  │ GCP Admin   │  │ Azure Admin │                              │
│  └─────────────┘  └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Cloud Resources                              │
├─────────────────────────────────────────────────────────────────┤
│  AWS: EC2, S3, Lambda, CloudFormation, ECS/EKS                 │
│  GitHub: Repositories, PRs, Issues, CI/CD Pipelines            │
│  Prometheus: Metrics, Alerts, Monitoring                       │
│  Kubernetes: Clusters, Pods, Services, ConfigMaps              │
│  GCP: Compute Engine, Cloud Storage, Cloud Functions           │
│  Azure: Virtual Machines, Blob Storage, Functions              │
└─────────────────────────────────────────────────────────────────┘
```

## 핵심 컴포넌트

### 1. DevOpsProductivityAgent (BaseAgent 상속)

**역할**: 메인 에이전트 클래스
**특징**:
- `BaseAgent`를 상속받아 MCP 서버 자동 연결
- 6개의 MCP 서버와 연동: AWS, GitHub, Prometheus, Kubernetes, GCP, Azure
- Gemini 2.5 Flash Latest 모델 사용
- Circuit Breaker 패턴으로 오류 처리

```python
class DevOpsProductivityAgent(BaseAgent):
    def __init__(self, output_dir: str = "devops_reports"):
        super().__init__(
            name="devops_productivity_agent",
            instruction="전문 DevOps 엔지니어...",
            server_names=["aws-kb", "github", "prometheus", "kubernetes", "gcp-admin", "azure-admin"]
        )
```

### 2. Orchestrator (도구 선택 자동화)

**역할**: LLM이 요청을 분석하여 적절한 MCP 도구를 자동 선택
**특징**:
- 수동 라우팅 로직 제거
- LLM이 컨텍스트를 기반으로 도구 선택
- 동적 도구 조합 가능

```python
async def run_workflow(self, request: str, context: Dict[str, Any] = None):
    orchestrator = self.get_orchestrator([])
    result = await orchestrator.execute(request, workflow_context)
```

### 3. MCP 서버 통합

#### AWS Knowledge Base MCP
- **서버**: `@modelcontextprotocol/server-aws-kb`
- **기능**: EC2, S3, Lambda, CloudFormation 관리
- **도구**: 인스턴스 상태 확인, 리소스 생성/삭제, 스택 배포

#### GitHub Operations MCP
- **서버**: `@modelcontextprotocol/server-github`
- **기능**: 리포지토리, PR, 이슈, CI/CD 관리
- **도구**: 리포지토리 분석, PR 리뷰, 워크플로우 모니터링

#### Prometheus Metrics MCP
- **서버**: `@modelcontextprotocol/server-prometheus`
- **기능**: 메트릭 수집 및 모니터링
- **도구**: 쿼리 실행, 알림 설정, 대시보드 생성

#### Kubernetes Cluster MCP
- **서버**: `@modelcontextprotocol/server-kubernetes`
- **기능**: 클러스터 및 워크로드 관리
- **도구**: Pod 상태 확인, 리소스 스케일링, 배포 관리

#### 멀티클라우드 MCP
- **GCP**: `@modelcontextprotocol/server-gcp`
- **Azure**: `@modelcontextprotocol/server-azure`
- **기능**: 클라우드 간 리소스 조정

## Agentic Flow 프로세스

### 1. 요청 수신
```
사용자 요청 → DevOpsProductivityAgent.run_workflow()
```

### 2. 컨텍스트 준비
```python
workflow_context = {
    "request": request,
    "capabilities": self.capabilities,
    "timestamp": datetime.now().isoformat(),
    **(context or {})
}
```

### 3. Orchestrator 실행
```python
orchestrator = self.get_orchestrator([])
result = await orchestrator.execute(request, workflow_context)
```

### 4. LLM 도구 선택
- LLM이 요청을 분석
- 사용 가능한 MCP 도구 중 적절한 도구 선택
- 도구 실행 및 결과 수집

### 5. 결과 반환
```python
return {
    "status": "success",
    "request": request,
    "result": result,
    "output_file": output_file,
    "timestamp": datetime.now().isoformat()
}
```

## 주요 개선사항

### 1. 하드코딩 제거
- **기존**: `GitHubClient`, `PrometheusClient` 클래스로 직접 API 호출
- **개선**: MCP 서버를 통한 간접 호출로 표준화

### 2. 수동 라우팅 제거
- **기존**: `process_request()` 내 if-else 분기 로직
- **개선**: Orchestrator가 자동으로 도구 선택

### 3. LLM 모델 업데이트
- **기존**: `gemini-2.5-flash-lite-preview-0607`
- **개선**: `gemini-2.5-flash-latest`

### 4. 오류 처리 개선
- **기존**: Try-catch 블록으로 개별 처리
- **개선**: BaseAgent의 Circuit Breaker 패턴 활용

### 5. 설정 중앙화
- **기존**: 하드코딩된 URL/Token
- **개선**: `mcp_agent.config.yaml` 중앙 관리

## 설정 파일 구조

### mcp_agent.config.yaml
```yaml
mcp:
  servers:
    aws-kb:
      url: "npx -y @modelcontextprotocol/server-aws-kb"
      description: "AWS Knowledge Base MCP Server"
      enabled: true
    github:
      url: "npx -y @modelcontextprotocol/server-github"
      description: "GitHub Operations MCP Server"
      enabled: true
    # ... 기타 MCP 서버들
```

### env.example
```bash
# 필수 환경변수
GOOGLE_API_KEY=your-gemini-api-key
GITHUB_TOKEN=your-github-token
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret

# 선택적 환경변수
PROMETHEUS_URL=http://localhost:9090
GCP_PROJECT_ID=your-gcp-project
AZURE_SUBSCRIPTION_ID=your-azure-subscription
```

## 사용 예제

### 1. AWS 리소스 관리
```python
request = "AWS EC2 인스턴스 상태를 확인해주세요"
result = await agent.run_workflow(request)
```

### 2. GitHub 작업
```python
request = "microsoft 조직의 리포지토리를 분석해주세요"
result = await agent.run_workflow(request)
```

### 3. 멀티클라우드 조정
```python
request = "AWS와 GCP의 리소스 사용량을 비교해주세요"
result = await agent.run_workflow(request)
```

## 장점

1. **확장성**: 새로운 MCP 서버 추가가 용이
2. **유지보수성**: 하드코딩 제거로 코드 간소화
3. **안정성**: Circuit Breaker 패턴으로 오류 처리
4. **표준화**: MCP 프로토콜로 일관된 인터페이스
5. **자동화**: LLM이 도구를 자동 선택하여 수동 개입 최소화

## 향후 개선 방향

1. **추가 MCP 서버**: Terraform, Ansible, Docker 등
2. **워크플로우 최적화**: 복잡한 작업의 단계별 분해
3. **모니터링 강화**: 에이전트 성능 메트릭 수집
4. **보안 강화**: MCP 서버 간 인증 및 권한 관리
