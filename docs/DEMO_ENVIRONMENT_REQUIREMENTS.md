# 시연환경 구성 가이드

MCP Agent Hub의 일부 에이전트는 실제 인프라, 하드웨어, 또는 클라우드 서비스가 필요합니다.
데모를 실행하기 전에 아래 항목을 확인하세요.

## 필요 에이전트별 구성 요구사항

| Agent | 필요 환경 | 최소 구성 | 권장 구성 |
| --- | --- | --- | --- |
| Drone Scout | 드론 하드웨어 또는 시뮬레이터 | 로컬 드론 시뮬레이터 | 실드론 + MAVSDK |
| AIOps Orchestrator | 서버/인프라, 모니터링 | Docker + 로컬 메트릭 수집 | Kubernetes + Prometheus/Grafana |
| DevOps Assistant | GitHub, 클라우드 계정 | GitHub 계정 + 로컬 테스트 저장소 | AWS/GCP/Azure + GitHub Actions |
| AI Architect | AI/ML 인프라, GPU | CPU 기반 모의 벤치마크 | GPU 클러스터 + MLflow |
| Cybersecurity | 보안 인프라, 스캐닝 도구 | 로컬 샘플 네트워크 | 방화벽 + 취약점 스캐너 |

## 공통 준비사항

1. **환경 변수 설정**: `.env` 파일에 필요한 API 키 설정
   (`GOOGLE_API_KEY`, `GITHUB_TOKEN`, `AWS_SECRET_ACCESS_KEY` 등)
2. **의존성 설치**: `pip install -r requirements.txt`
3. **선택 모듈**: 인프라별 CLI 도구 (kubectl, docker, aws-cli 등) 설치

일부 에이전트는 모의 데이터로 기능만 시연할 수 있으며, 이 경우 위 환경 구성 없이
Streamlit 앱(`streamlit run main.py`)에서 해당 페이지에 접근하면 됩니다.