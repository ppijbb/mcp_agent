# Local Researcher - Gemini CLI + Open Deep Research Integration

## 프로젝트 개요

Local Researcher는 Gemini CLI와 Open Deep Research를 통합하여 로컬 환경에서 고성능 리서치 시스템을 구축하는 프로젝트입니다. 이 시스템은 다음과 같은 특징을 가집니다:

- **로컬 실행**: 모든 데이터와 처리가 로컬에서 이루어집니다
- **Gemini CLI 통합**: 명령줄 인터페이스를 통한 직관적인 리서치 워크플로우
- **Open Deep Research 활용**: 다중 에이전트 아키텍처를 통한 심층 리서치
- **모듈화된 설계**: 확장 가능하고 유지보수가 용이한 구조
- **프로덕션 레벨**: 실제 비즈니스 환경에서 사용할 수 있는 수준의 품질

## 시스템 아키텍처

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gemini CLI    │────│  Local Researcher │────│ Open Deep Res.  │
│   Interface     │    │   Orchestrator   │    │   Multi-Agent   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Command       │    │   Research       │    │   Search Tools  │
│   Processing    │    │   Workflow       │    │   & APIs        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Report        │    │   Data Storage   │    │   Export        │
│   Generation    │    │   & Cache        │    │   & Sharing     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 주요 기능

### 1. 통합 명령줄 인터페이스
- Gemini CLI를 통한 자연어 명령 처리
- 리서치 토픽 정의 및 워크플로우 관리
- 실시간 진행 상황 모니터링

### 2. 고급 리서치 엔진
- Open Deep Research의 다중 에이전트 시스템 활용
- 자동화된 웹 검색 및 학술 자료 수집
- 실시간 데이터 분석 및 인사이트 추출

### 3. 지능형 워크플로우
- 토픽별 맞춤형 리서치 전략 수립
- 자동화된 보고서 생성 및 구조화
- 품질 검증 및 피드백 시스템

### 4. 로컬 데이터 관리
- 모든 데이터의 로컬 저장 및 관리
- 캐싱 시스템을 통한 성능 최적화
- 보안 및 개인정보 보호

## 설치 및 설정

### 필수 요구사항

- Node.js 20+
- Python 3.11+
- Git
- Docker (선택사항)

### 1. 프로젝트 클론

```bash
git clone <repository-url>
cd local_researcher_project
```

### 2. Gemini CLI 설치

```bash
npm install -g @google/gemini-cli
```

### 3. Python 환경 설정

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 4. 의존성 설치

```bash
pip install -r requirements.txt
npm install
```

### 5. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일을 편집하여 API 키 및 설정을 구성
```

### 6. Open Deep Research 설정

```bash
cd open_researcher/open_deep_research
pip install -e .
```

## 사용 방법

### 기본 사용법

```bash
# 리서치 시작
gemini research "인공지능의 최신 동향과 미래 전망"

# 특정 도메인 리서치
gemini research --domain "technology" "블록체인 기술의 발전"

# 상세 분석 요청
gemini research --depth "comprehensive" "기후변화 대응 기술"
```

### 고급 기능

```bash
# 커스텀 워크플로우 정의
gemini workflow create my_research_workflow

# 배치 리서치 실행
gemini batch research topics.txt

# 실시간 모니터링
gemini monitor research_status
```

## 프로젝트 구조

```
local_researcher_project/
├── src/
│   ├── core/                 # 핵심 기능
│   ├── cli/                  # CLI 인터페이스
│   ├── research/             # 리서치 엔진
│   ├── agents/               # 에이전트 시스템
│   ├── storage/              # 데이터 저장소
│   └── utils/                # 유틸리티
├── configs/                  # 설정 파일
├── tests/                    # 테스트
├── docs/                     # 문서
├── examples/                 # 예제
└── scripts/                  # 스크립트
```

## 개발 가이드

### 새로운 리서치 도구 추가

1. `src/research/tools/` 디렉토리에 새 도구 클래스 생성
2. `BaseResearchTool` 클래스 상속
3. `configs/tools.yaml`에 설정 추가
4. 테스트 작성 및 실행

### 커스텀 에이전트 개발

1. `src/agents/` 디렉토리에 새 에이전트 클래스 생성
2. `BaseAgent` 클래스 상속
3. 워크플로우에 통합
4. 성능 테스트 및 최적화

## 성능 최적화

- 멀티프로세싱을 통한 병렬 처리
- Redis 캐싱 시스템 활용
- 비동기 I/O 처리
- 메모리 사용량 최적화

## 보안 고려사항

- 모든 API 키의 안전한 관리
- 로컬 데이터 암호화
- 네트워크 통신 보안
- 접근 권한 제어

## 라이선스

MIT License

## 기여 방법

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 지원

- Issues: GitHub Issues
- Documentation: `/docs` 디렉토리
- Examples: `/examples` 디렉토리 