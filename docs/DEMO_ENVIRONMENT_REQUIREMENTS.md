# 시연환경 구성 가이드 (Demo Environment Requirements)

일부 에이전트는 실제 인프라, 하드웨어, 또는 클라우드 서비스가 필요합니다.
데모를 위해서는 별도의 시연 환경 구성이 필요합니다.

## 구성 수준

### 최소 구성
- 로컬 개발 환경 (Docker, minikube, 테스트 계정)
- 아래 에이전트의 기능만 가볍게 확인하는 용도

### 완전 구성
- 클라우드 계정 (AWS, GCP, Azure)
- Kubernetes 클러스터
- 모니터링 스택 (Prometheus, Grafana 등)

## 에이전트별 필요 환경

### 🛸 Drone Scout Agent
- 필요: 드론 하드웨어 또는 시뮬레이터
- 자연어 임무를 입력하여 자율 드론 정찰 실행

### 🤖 AIOps Orchestrator Agent
- 필요: 실제 서버/인프라, Kubernetes, 모니터링 시스템
- AI 기반 IT 운영 자동화 및 모니터링

### 🚀 DevOps Assistant Agent
- 필요: GitHub 계정, AWS/GCP/Azure, Kubernetes 클러스터
- GitHub, AWS, Kubernetes 등 개발자 생산성 자동화

### 🏗️ AI Architect Agent
- 필요: AI/ML 인프라, GPU 클러스터, 성능 벤치마크 환경
- 진화형 AI 아키텍처 설계 및 자동 최적화

### 🔒 Cybersecurity Agent
- 필요: 보안 인프라, 방화벽, 보안 스캐닝 도구
- 사이버 보안 인프라 관리 및 위협 분석

## 참고
- 일부 Agent는 모의 데이터로 기능만 시연할 수 있습니다.
- 각 에이전트의 페이지에서 상세 실행 방법과 설정을 확인하세요.
- 환경 변수와 시크릿 설정은 `configs/` 및 `mcp_agent.config.yaml`에서 관리합니다.