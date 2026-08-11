# 시연환경 구성 가이드 (Demo Environment Requirements)

아래 Agent들은 실제 인프라, 하드웨어, 또는 클라우드 서비스가 필요합니다.
메인 화면의 "🔧 시연환경 구성 필요" 섹션에 표시되며, 데모를 위해서는 별도의 시연 환경 구성이 필요합니다.

> **참고**: 일부 Agent는 모의 데이터로 기능만 시연할 수 있습니다.

---

## 구성 등급 안내

| 등급 | 구성 | 대상 |
| --- | --- | --- |
| 🟢 최소 구성 | 로컬 개발 환경 (Docker, minikube, 테스트 계정) | 빠른 기능 검증용 |
| 🔴 완전 구성 | 클라우드 계정, Kubernetes 클러스터, 모니터링 스택 | 실제 동작 데모용 |

---

## 1. 🛸 Drone Scout Agent

**페이지**: `pages/drone_scout.py`

- **필요 환경**: 드론 하드웨어 또는 드론 시뮬레이터
- **권장 구성**:
  - 물리 드론: MAVLink 지원 비행 컨트롤러, 지상국 제어 소프트웨어
  - 시뮬레이터: Gazebo, AirSim 등 PX4/ArduPilot 호환 시뮬레이터 (권장)
- **가상 임무 입력**: 자연어로 정찰 임무를 입력하고 경로/영역 계획을 확인할 수 있음

---

## 2. 🤖 AIOps Orchestrator Agent

**페이지**: `pages/aiops_orchestrator.py`

- **필요 환경**: 실제 서버/인프라, Kubernetes, 모니터링 시스템
- **권장 구성**:
  - Kubernetes 클러스터 (minikube, kind 또는 관리형 클러스터)
  - 모니터링 스택: Prometheus, Grafana, Alertmanager
  - 로그 수집: Loki, Elastic Stack 등
  - `kubectl` 및 클러스터 접근 권한 (kubeconfig)
- **최소 구성**: minikube + 모의 메트릭 데이터로 워크플로우 검증

---

## 3. 🚀 DevOps Assistant Agent

**페이지**: `pages/devops_assistant.py`

- **필요 환경**: GitHub 계정, AWS/GCP/Azure, Kubernetes 클러스터
- **권장 구성**:
  - GitHub: Personal Access Token (repo, workflow 권한)
  - 클라우드: AWS / GCP / Azure 중 하나 이상의 계정과 CLI 인증
  - Kubernetes: 배포 대상 클러스터 및 kubeconfig
  - CI/CD: GitHub Actions 등 파이프라인 실행 환경
- **최소 구성**: GitHub 계정만으로 저장소/PR 관련 작업 시연 가능

---

## 4. 🏗️ AI Architect Agent

**페이지**: `pages/ai_architect.py`

- **필요 환경**: AI/ML 인프라, GPU 클러스터, 성능 벤치마크 환경
- **권장 구성**:
  - GPU 클러스터 또는 GPU 서버 (CUDA 환경)
  - ML 프레임워크: PyTorch / TensorFlow
  - 모델 레지스트리 및 성능 벤치마크 도구
- **최소 구성**: CPU 환경 + 모의 데이터로 아키텍처 설계 흐름만 확인 가능

---

## 5. 🔒 Cybersecurity Agent

**페이지**: `pages/cybersecurity_agent.py`

- **필요 환경**: 보안 인프라, 방화벽, 보안 스캐닝 도구
- **권장 구성**:
  - 보안 스캐닝 도구: Nmap, OpenVAS, Trivy 등
  - 방화벽 / IDS / IPS 로그 소스
  - 테스트용 격리 네트워크 (실제 운영 인프라 대상 스캔 금지)
- **최소 구성**: 테스트용 로컬 네트워크 + 모의 로그 데이터

---

## 공통 사항

- 모든 Agent는 Streamlit UI(`main.py`)에서 카드 버튼을 통해 접근합니다.
- 외부 서비스 연동이 필요한 Agent는 환경 변수로 API 키/자격 증명을 주입하세요
  (자세한 형식은 `README.md`의 "External MCP servers" 섹션 참고).
- 운영 인프라를 대상으로 하는 시나리오는 반드시 **격리된 테스트 환경**에서만 실행하세요.
- 시크릿은 코드에 하드코딩하지 말고 `mcp_agent.secrets.yaml` 또는 환경 변수로 관리하세요.
