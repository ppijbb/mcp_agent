# 🤖 Multi-Agent Automation Service

**Python mcp_agent 라이브러리 기반 Multi-Agent 시스템**  
**Gemini CLI를 통한 최종 명령 실행**

## 📋 개요

이 프로젝트는 **4개 전문 Agent**들이 협업하여 코드 리뷰, 자동 문서화, 성능 테스트, 보안/배포 검증을 수행하고, **Gemini CLI**를 통해 실제 수정 작업을 실행하는 자동화 시스템입니다.

### 🌟 주요 특징

- **🤖 Multi-Agent 협업**: 4개 전문 Agent의 역할 분담
- **🔧 Gemini CLI 통합**: 실제 코드 수정 및 문서 업데이트
- **⚡ 병렬 처리**: Agent들의 동시 실행으로 효율성 극대화
- **📅 스케줄링**: Cron 기반 자동 실행
- **📊 실시간 모니터링**: 실행 결과 및 성능 추적

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Agent Orchestrator                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Code      │  │Documentation│  │Performance  │        │
│  │  Review     │  │   Agent     │  │   Test      │        │
│  │   Agent     │  │             │  │   Agent     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐                                          │
│  │ Security &  │                                          │
│  │ Deployment  │                                          │
│  │   Agent     │                                          │
│  └─────────────┘                                          │
├─────────────────────────────────────────────────────────────┤
│                    Gemini CLI Executor                      │
├─────────────────────────────────────────────────────────────┤
│              실제 코드 수정 및 문서 업데이트                 │
└─────────────────────────────────────────────────────────────┘
```

## 🤖 Agent 역할 분담

### 1. **CodeReviewAgent** 🔍
- **역할**: 코드 품질 검토 및 보안 취약점 발견
- **기능**:
  - 코드 품질 분석 (가독성, 성능, 유지보수성)
  - 보안 취약점 스캔 (SQL 인젝션, XSS, CSRF 등)
  - 코딩 표준 준수 여부 확인
  - 개선사항 제안
  - Gemini CLI 명령어 생성

### 2. **DocumentationAgent** 📝
- **역할**: 자동 문서화 및 API 문서 업데이트
- **기능**:
  - 코드 변경사항 분석
  - README.md 자동 업데이트
  - CHANGELOG.md 자동 업데이트
  - API 문서 생성
  - 새로운 기능에 대한 문서 작성

### 3. **PerformanceTestAgent** ⚡
- **역할**: 성능 분석 및 테스트 생성
- **기능**:
  - CPU/메모리/네트워크 사용량 분석
  - 병목 지점 발견
  - 자동 테스트 케이스 생성
  - 성능 벤치마크 실행
  - 최적화 방안 제안

### 4. **SecurityDeploymentAgent** 🔒
- **역할**: 보안 스캔 및 배포 검증
- **기능**:
  - 보안 취약점 스캔
  - 배포 후 상태 검증
  - 자동 롤백 결정
  - 헬스 체크 및 모니터링
  - 규정 준수 검사

## 🚀 설치 및 설정

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. Gemini CLI 설치
```bash
npx https://github.com/google-gemini/gemini-cli
```

### 3. 환경 변수 설정
```bash
export GEMINI_API_KEY="your_gemini_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

## 📖 사용법

### 전체 자동화 실행
```bash
python -m multi_agent_automation_service full
```

### 코드 리뷰만 실행
```bash
python -m multi_agent_automation_service review
```

### 배포 워크플로우 실행
```bash
python -m multi_agent_automation_service deploy --deployment-id "deployment-123"
```

### 개별 Agent 실행
```bash
python -m multi_agent_automation_service individual
```

### 스케줄러 시작
```bash
python -m multi_agent_automation_service scheduler
```

### 상태 확인
```bash
python -m multi_agent_automation_service status
```

### 특정 경로 지정
```bash
python -m multi_agent_automation_service full --paths src/ tests/ docs/
```

## ⏰ 스케줄링

자동화된 스케줄링 기능:

- **매일 새벽 2시**: 전체 자동화 실행
- **매주 월요일 오전 9시**: 코드 리뷰 워크플로우
- **매시간**: 배포 상태 확인

## 📊 실행 결과 예시

### 전체 자동화 결과
```
Multi-Agent 자동화 실행 요약
===========================

📊 Agent 실행 결과:
- code_review: ✅ 성공
- documentation: ✅ 성공
- performance_test: ✅ 성공
- security_deployment: ✅ 성공

🔧 Gemini CLI 명령어 실행: 12개
- 성공: 11개
- 실패: 1개
```

### 코드 리뷰 결과
```
코드 리뷰 결과 요약
==================

📁 검토된 파일: 15개
🔍 발견된 이슈: 8개
🚨 보안 취약점: 2개
💡 개선 제안: 12개
⭐ 코드 품질 점수: 0.85/1.0
```

## 🔧 기술 스택

### 핵심 라이브러리
- **mcp-agent**: Multi-Agent 협업 프레임워크
- **Gemini CLI**: AI 기반 코드 수정 도구
- **asyncio**: 비동기 처리
- **schedule**: 스케줄링

### AI/ML
- **OpenAI GPT-4**: Agent 의사결정
- **Google Gemini**: 코드 분석 및 수정
- **Anthropic Claude**: 보안 분석

### 모니터링
- **Prometheus**: 메트릭 수집
- **Structlog**: 구조화된 로깅

## 📁 프로젝트 구조

```
multi_agent_automation_service/
├── __init__.py                 # 패키지 초기화
├── main.py                     # 메인 실행 파일
├── orchestrator.py             # Multi-Agent Orchestrator
├── gemini_cli_executor.py      # Gemini CLI 실행기
├── agents/                     # Agent 모듈
│   ├── __init__.py
│   ├── code_review_agent.py    # 코드 리뷰 Agent
│   ├── documentation_agent.py  # 문서화 Agent
│   ├── performance_test_agent.py # 성능 테스트 Agent
│   └── security_deployment_agent.py # 보안/배포 Agent
├── requirements.txt            # 의존성 목록
└── README.md                   # 프로젝트 문서
```

## 🎯 워크플로우

### 1. 전체 자동화 워크플로우
```
1. 4개 Agent 병렬 실행
   ├── CodeReviewAgent: 코드 품질 검토
   ├── DocumentationAgent: 문서 업데이트
   ├── PerformanceTestAgent: 성능 분석
   └── SecurityDeploymentAgent: 보안 스캔

2. 결과 수집 및 분석

3. Gemini CLI 명령어 통합

4. 실제 수정 작업 실행

5. 최종 보고서 생성
```

### 2. 코드 리뷰 워크플로우
```
1. 코드 리뷰 실행
2. 심각한 이슈 발견 시 보안 스캔 추가
3. Gemini CLI 명령어 실행
4. 수정 결과 확인
```

### 3. 배포 워크플로우
```
1. 배포 상태 검증
2. 롤백 필요성 판단
   ├── 필요 시: 자동 롤백 실행
   └── 불필요 시: 성능 테스트 + 문서 업데이트
3. Gemini CLI 명령어 실행
```

## 🔍 모니터링 및 로깅

### 실행 히스토리
- 모든 Agent 실행 결과 저장
- Gemini CLI 명령어 실행 로그
- 성공/실패 통계

### 성능 메트릭
- 실행 시간 추적
- 리소스 사용량 모니터링
- 병목 지점 분석

## 🛠️ 개발 및 확장

### 새로운 Agent 추가
1. `agents/` 디렉토리에 새 Agent 클래스 생성
2. `__init__.py`에 Agent 등록
3. `orchestrator.py`에 워크플로우 추가

### 커스텀 워크플로우
```python
async def custom_workflow(self):
    # 1. Agent 실행
    result = await self.agent.run_task()
    
    # 2. Gemini CLI 명령어 생성
    commands = self.generate_gemini_commands(result)
    
    # 3. 명령어 실행
    await self.gemini_executor.execute_batch_commands(commands)
```

## 🚨 문제 해결

### 일반적인 문제

1. **Gemini CLI 설치 실패**
   ```bash
   # Node.js 버전 확인
   node --version  # v18+ 필요
   
   # 재설치
   npm uninstall -g @google/gemini-cli
   npx https://github.com/google-gemini/gemini-cli
   ```

2. **API 키 오류**
   ```bash
   # 환경 변수 확인
   echo $GEMINI_API_KEY
   echo $OPENAI_API_KEY
   ```

3. **권한 오류**
   ```bash
   # 실행 권한 부여
   chmod +x main.py
   ```

## 📈 성능 최적화

### 병렬 처리
- Agent들의 동시 실행으로 전체 실행 시간 단축
- 비동기 I/O 활용

### 캐싱
- 중복 분석 결과 캐싱
- Gemini CLI 응답 캐싱

### 리소스 관리
- 메모리 사용량 최적화
- CPU 사용률 모니터링

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 라이선스

MIT License

## 🙏 감사의 말

- [mcp-agent](https://github.com/mcp-sh/mcp-agent) - Multi-Agent 협업 프레임워크
- [Gemini CLI](https://github.com/google-gemini/gemini-cli) - AI 기반 코드 수정 도구
- [OpenAI](https://openai.com/) - GPT-4 모델 제공

---

**Made with ❤️ using Python mcp_agent + Gemini CLI** 