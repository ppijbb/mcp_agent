# 🚀 K8s Cloud Operations Agent

Python MCP 기반 Kubernetes 및 클라우드 운영 관리 시스템

## 📋 개요

이 프로젝트는 **OpenAI Agents SDK**와 **Pydantic AI**의 MCP(Model Context Protocol) 지원을 활용하여 Kubernetes 및 클라우드 환경의 운영을 자동화하는 지능형 Agent 시스템입니다.

### 🌟 주요 특징

- **동적 설정 생성**: YAML 파일 없이 실시간으로 Kubernetes 설정 생성
- **실시간 모니터링**: 시스템 상태 및 성능 실시간 추적
- **자동화된 배포**: Blue-Green 배포 및 자동 롤백
- **보안 관리**: 실시간 보안 스캔 및 규정 준수 검사
- **비용 최적화**: 클라우드 비용 분석 및 최적화 권장
- **장애 대응**: 예측적 장애 방지 및 자동 복구

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    K8s Cloud Operations Agent               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Config    │ │ Deployment  │ │ Monitoring  │           │
│  │   Agent     │ │   Agent     │ │   Agent     │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │  Security   │ │    Cost     │ │  Incident   │           │
│  │   Agent     │ │ Optimizer   │ │ Response    │           │
│  │             │ │   Agent     │ │   Agent     │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│                    MCP Framework                           │
│              (OpenAI Agents SDK + Pydantic AI)             │
├─────────────────────────────────────────────────────────────┤
│              Kubernetes & Cloud APIs                        │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Agent 구성

### 1. 동적 설정 생성 Agent
- **역할**: 워크로드 요구사항 분석 및 최적화된 Kubernetes 설정 생성
- **기능**: 
  - 실시간 환경 분석
  - 동적 YAML 매니페스트 생성
  - 보안 정책 자동 생성
  - 모니터링 설정 생성

### 2. 배포 관리 Agent
- **역할**: 애플리케이션 배포 및 업데이트 관리
- **기능**:
  - Blue-Green 배포 자동화
  - Rolling Update 관리
  - 자동 롤백
  - 배포 상태 모니터링

### 3. 모니터링 Agent
- **역할**: 시스템 상태 및 성능 모니터링
- **기능**:
  - 실시간 메트릭 수집
  - 예측적 분석
  - 자동 스케일링
  - 알림 생성

### 4. 보안 Agent
- **역할**: 보안 정책 및 규정 준수 관리
- **기능**:
  - 실시간 보안 스캔
  - 취약점 평가
  - 규정 준수 검사
  - 위협 감지

### 5. 비용 최적화 Agent
- **역할**: 클라우드 비용 최적화 및 관리
- **기능**:
  - 실시간 비용 모니터링
  - 최적화 권장사항
  - 예산 관리
  - 비용 예측

### 6. 장애 대응 Agent
- **역할**: 장애 감지 및 자동 복구
- **기능**:
  - 예측적 장애 방지
  - 자동 복구 프로세스
  - 실시간 복구 전략 생성
  - 장애 보고서 생성

## 🚀 빠른 시작

### 1. 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# 또는 개발 환경에서
pip install -e .
```

### 2. 기본 사용법

```bash
# 전체 운영 프로세스 실행
python main.py --mode full --workload my-app --type web --image nginx:latest

# 동적 설정 생성만
python main.py --mode config --workload my-app --type web

# 배포만
python main.py --mode deploy --workload my-app --image nginx:latest

# 모니터링만
python main.py --mode monitor --namespace default
```

### 3. 개별 Agent 사용

```python
import asyncio
from agents.dynamic_config_agent import DynamicConfigGenerator, WorkloadRequirements

async def generate_config():
    generator = DynamicConfigGenerator()
    
    requirements = WorkloadRequirements(
        name="my-web-app",
        type="web",
        cpu_request="100m",
        memory_request="128Mi",
        replicas=3,
        environment="prod",
        security_level="high",
        scaling_requirements={"min": 2, "max": 10}
    )
    
    config = await generator.generate_k8s_config(requirements)
    print(f"Configuration generated: {config}")

asyncio.run(generate_config())
```

## 📁 프로젝트 구조

```
k8s_cloud_operations_agent/
├── agents/                          # Agent 구현
│   ├── dynamic_config_agent.py     # 동적 설정 생성
│   ├── deployment_agent.py         # 배포 관리
│   ├── monitoring_agent.py         # 모니터링
│   ├── security_agent.py           # 보안 관리
│   ├── cost_optimizer_agent.py     # 비용 최적화
│   └── incident_response_agent.py  # 장애 대응
├── mcp_servers/                     # MCP 서버 구현
├── utils/                          # 유틸리티 함수
├── configs/                        # 설정 파일
├── tests/                          # 테스트 코드
├── main.py                         # 메인 실행 파일
├── requirements.txt                # 의존성 목록
└── README.md                       # 프로젝트 문서
```

## 🔧 설정

### 환경 변수

```bash
# OpenAI API 키 (LLM 사용 시)
export OPENAI_API_KEY="your-api-key"

# Kubernetes 설정
export KUBECONFIG="~/.kube/config"

# 클라우드 자격 증명
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

### MCP 서버 설정

각 Agent는 필요한 MCP 서버에 자동으로 연결됩니다:

- `k8s-mcp`: Kubernetes API 서버
- `monitoring-mcp`: Prometheus/Grafana 서버
- `security-mcp`: 보안 스캔 서버
- `cost-mcp`: 클라우드 비용 API 서버

## 📊 모니터링 및 대시보드

### 생성되는 보고서

- **설정 보고서**: `generated_configs/`
- **배포 보고서**: `deployment_reports/`
- **모니터링 보고서**: `monitoring_reports/`
- **보안 보고서**: `security_reports/`
- **비용 보고서**: `cost_reports/`
- **장애 보고서**: `incident_reports/`

### 실시간 메트릭

- CPU/메모리 사용률
- 네트워크 트래픽
- 애플리케이션 성능
- 보안 위반
- 비용 트렌드
- 장애 발생률

## 🔒 보안

### 보안 기능

- **Pod Security Standards**: 컨테이너 보안 강화
- **Network Policies**: 네트워크 트래픽 제어
- **RBAC**: 역할 기반 접근 제어
- **Secret Management**: 민감 정보 관리
- **Vulnerability Scanning**: 취약점 스캔

### 규정 준수

- **CIS Kubernetes Benchmark**
- **GDPR 준수** (데이터 관련)
- **SOX 준수** (재무 관련)

## 💰 비용 최적화

### 최적화 전략

- **미사용 리소스 제거**: 자동 감지 및 정리
- **오버프로비저닝 최적화**: 리소스 크기 조정
- **예약 인스턴스**: 장기 사용 시 할인
- **스팟 인스턴스**: 비용 절약을 위한 활용
- **자동 스케일링**: 수요에 따른 동적 조정

## 🚨 장애 대응

### 자동 복구

- **Pod 재시작**: 실패한 Pod 자동 재시작
- **서비스 복구**: 서비스 장애 시 자동 복구
- **리소스 스케일링**: 부족한 리소스 자동 확장
- **네트워크 정책**: 네트워크 문제 자동 해결

### 예측적 방지

- **성능 저하 감지**: 문제 발생 전 조기 경고
- **리소스 예측**: 사용량 트렌드 분석
- **장애 패턴 학습**: 반복 장애 방지

## 🧪 테스트

```bash
# 단위 테스트 실행
pytest tests/

# 통합 테스트 실행
pytest tests/integration/

# 코드 커버리지
pytest --cov=agents tests/
```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🆘 지원

- **문서**: [Wiki](https://github.com/your-repo/wiki)
- **이슈**: [GitHub Issues](https://github.com/your-repo/issues)
- **토론**: [GitHub Discussions](https://github.com/your-repo/discussions)

## 🔄 로드맵

### v1.1 (예정)
- [ ] 멀티 클러스터 지원
- [ ] 서비스 메시 통합
- [ ] 고급 ML 기반 예측

### v1.2 (예정)
- [ ] 웹 대시보드
- [ ] API 엔드포인트
- [ ] 플러그인 시스템

### v2.0 (예정)
- [ ] 엣지 컴퓨팅 지원
- [ ] 하이브리드 클라우드
- [ ] AI 기반 의사결정

---

**Made with ❤️ using OpenAI Agents SDK & Pydantic AI MCP** 