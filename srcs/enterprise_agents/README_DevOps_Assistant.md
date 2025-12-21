# 🚀 DevOps Assistant Agent

**MCP 기반 개발자 생산성 자동화 에이전트**

DevOps Assistant는 Model Context Protocol(MCP)을 활용하여 개발 워크플로우를 자동화하고 팀 생산성을 향상시키는 AI 에이전트입니다. Gemini 2.0 Flash 모델을 사용하여 코드 리뷰, 배포 관리, 시스템 모니터링을 지능적으로 처리합니다.

## ✨ 주요 기능

### 🔍 코드 리뷰 자동화
- Pull Request 자동 분석 및 리뷰
- 코드 품질, 보안, 성능 관점에서 종합 평가
- 건설적이고 실행 가능한 피드백 제공
- 테스트 커버리지 및 문서화 검토

### 🚀 배포 상태 관리
- GitHub Actions 워크플로우 모니터링
- CI/CD 파이프라인 상태 실시간 추적
- 배포 실패 원인 분석 및 해결책 제안
- 자동 알림 및 에스컬레이션

### 🎯 이슈 우선순위 분석
- GitHub 이슈 자동 분류 및 우선순위 지정
- 사용자 영향도, 기술적 복잡성 기반 분석
- 담당자 추천 및 작업 시간 예측
- 버그/기능 요청 구분 및 라벨링

### 👥 팀 협업 지원
- 일일 스탠드업 요약 자동 생성
- 팀 활동 메트릭 수집 및 분석
- 차단 요소 식별 및 해결 방안 제시
- 커뮤니케이션 채널 통합

### 📊 성능 분석 및 모니터링
- 시스템 성능 메트릭 실시간 분석
- 병목 지점 식별 및 최적화 제안
- 용량 계획 및 알람 설정 가이드
- 기술 부채 분석 및 개선 로드맵

## 🛠️ 기술 스택

- **AI 모델**: Google Gemini 2.0 Flash
- **프로토콜**: Model Context Protocol (MCP)
- **언어**: Python 3.8+
- **비동기**: asyncio, aiohttp
- **API 통합**: GitHub, Slack, Jira
- **설정**: YAML/JSON 구성 파일

## 📦 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements_devops_assistant.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```env
# 필수: Google AI API Key
GOOGLE_API_KEY=your_google_ai_api_key_here

# 선택사항: GitHub 통합
GITHUB_TOKEN=your_github_personal_access_token

# 선택사항: Slack 통합
SLACK_WEBHOOK_URL=your_slack_webhook_url
SLACK_BOT_TOKEN=your_slack_bot_token

# 선택사항: Jira 통합
JIRA_API_KEY=your_jira_api_key
JIRA_BASE_URL=your_jira_instance_url
```

### 3. Agent Card 구성

`devops_assistant_agent_card.yaml` 파일에서 에이전트 설정을 조정할 수 있습니다:

```yaml
model_configuration:
  primary_model: "gemini-2.0-flash-exp"
  temperature: 0.2
  max_tokens: 4000

monitoring:
  check_interval: 300  # 5분
  enabled: true

notifications:
  slack_webhook: "${SLACK_WEBHOOK_URL}"
  email_enabled: false
```

## 🚀 사용법

### 대화형 데모 실행

```bash
python run_devops_assistant.py
```

메뉴에서 원하는 기능을 선택하여 테스트할 수 있습니다:

```
🚀 DevOps Assistant Agent - Demo Menu
============================================================
1. 🔍 코드 리뷰 자동화
2. 🚀 배포 상태 확인
3. 🎯 이슈 우선순위 분석
4. 👥 팀 스탠드업 준비
5. 📊 성능 분석
6. 🔄 연속 모니터링 모드 (시작)
7. 🛑 연속 모니터링 모드 (중지)
8. 📋 에이전트 상태 확인
9. ❌ 종료
============================================================
```

### 빠른 데모 실행

```bash
python run_devops_assistant.py --mode quick
```

모든 기능을 자동으로 순차 실행하여 빠르게 확인할 수 있습니다.

### 프로그래밍 방식 사용

```python
from devops_assistant_agent import DevOpsAssistantAgent, run_code_review

# 에이전트 초기화
agent = DevOpsAssistantAgent()

# 코드 리뷰 실행
await run_code_review(agent, "microsoft", "vscode", 42)

# 배포 상태 확인
await run_deployment_check(agent, "kubernetes", "kubernetes")

# 이슈 분석
await run_issue_analysis(agent, "facebook", "react")
```

## 🔧 커스터마이징

### 새로운 MCP 도구 추가

MCP 서버에 새로운 도구를 추가하려면:

1. `mcp_servers/devops_github_mcp_server.py`에 새 도구 정의
2. 해당 도구의 핸들러 메서드 구현
3. Agent Card에 도구 정보 추가

```python
Tool(
    name="new_tool",
    description="새로운 도구 설명",
    inputSchema={
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "매개변수"}
        },
        "required": ["param"]
    }
)
```

### 워크플로우 확장

새로운 자동화 워크플로우를 추가하려면:

1. `DevOpsAssistantAgent` 클래스에 새 핸들러 메서드 추가
2. `_process_task` 메서드에 작업 타입 등록
3. 실행 함수 생성

```python
async def _handle_new_workflow(self, data: Dict[str, Any]):
    """새로운 워크플로우 처리"""
    # 구현 로직
    pass
```

## 📊 성능 메트릭

DevOps Assistant는 다음 KPI를 추적합니다:

- **코드 리뷰 시간 단축**: 목표 50%
- **배포 실패율 감소**: 목표 30%
- **이슈 해결 시간 단축**: 목표 40%
- **팀 커뮤니케이션 효율성 향상**

## 🔒 보안 고려사항

- **인증**: GitHub Personal Access Token, Slack Bot Token 필요
- **권한**: 저장소 읽기, PR 코멘트 작성, 배포 상태 업데이트만 필요
- **데이터 처리**: 코드는 로컬에서만 분석, 민감한 정보 자동 마스킹
- **암호화**: 로그 데이터 암호화 저장

## 🚫 제한사항

### 기술적 제한사항
- 복잡한 아키텍처 결정은 인간 개발자의 검토 필요
- 보안 취약점 분석은 전문 도구와 병행 사용 권장
- 대규모 리팩토링은 수동 검토 필수

### 운영 제한사항
- API 요청 제한 (GitHub: 5000/시간, Google AI: 모델별 상이)
- 네트워크 연결 의존성
- 외부 서비스 가용성에 따른 기능 제약

## 🤝 통합 가이드

### GitHub 통합

```python
# GitHub API 설정
# Use environment variable: GITHUB_TOKEN
import os
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# 지원 기능
- 저장소 목록 조회
- Pull Request 관리
- 이슈 생성 및 관리
- 워크플로우 상태 확인
```

### Slack 통합

```python
# Slack Webhook 설정
SLACK_WEBHOOK_URL = "https://hooks.slack.com/..."

# 지원 기능
- 채널 메시지 발송
- 코드 리뷰 요약 알림
- 배포 상태 알림
- 팀 스탠드업 요약
```

### Jira 통합

```python
# Jira API 설정
JIRA_API_KEY = "your_api_key"
JIRA_BASE_URL = "https://yourcompany.atlassian.net"

# 지원 기능
- 티켓 생성 및 업데이트
- 백로그 우선순위 관리
- 개발자 할당
- 스프린트 계획
```

## 🧪 테스트

### 단위 테스트 실행

```bash
pytest tests/test_devops_assistant.py -v
```

### 통합 테스트 실행

```bash
pytest tests/test_integration.py -v --asyncio-mode=auto
```

### 성능 테스트

```bash
python -m pytest tests/test_performance.py --benchmark-only
```

## 📈 모니터링 및 로그

### 로그 레벨 설정

```python
import logging
logging.getLogger("DevOpsAssistant").setLevel(logging.DEBUG)
```

### 메트릭 수집

에이전트는 다음 메트릭을 자동으로 수집합니다:

- 처리된 작업 수
- 평균 응답 시간
- 성공/실패율
- API 호출 횟수
- 시스템 리소스 사용량

## 🔄 업데이트 및 유지보수

### 에이전트 업데이트

```bash
git pull origin main
pip install -r requirements_devops_assistant.txt --upgrade
```

### 설정 백업

```bash
# Agent Card 백업
cp devops_assistant_agent_card.yaml devops_assistant_agent_card.yaml.bak

# 로그 백업
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

## 🆘 문제 해결

### 일반적인 문제

1. **API 키 오류**
   ```
   ValueError: GOOGLE_API_KEY 환경변수를 설정해주세요
   ```
   → `.env` 파일에 유효한 Google AI API 키를 설정

2. **MCP 연결 실패**
   ```
   ❌ MCP 서버 연결 실패
   ```
   → Python 경로와 MCP 서버 파일 위치를 확인

3. **메모리 부족**
   ```
   OutOfMemoryError
   ```
   → `max_output_tokens` 값을 줄이거나 시스템 메모리를 늘림

### 로그 확인

```bash
tail -f logs/devops_assistant.log
```

## 📞 지원 및 기여

- **이슈 리포트**: GitHub Issues에서 버그 리포트 또는 기능 요청
- **기여 가이드**: CONTRIBUTING.md 참조
- **커뮤니티**: Discussions에서 질문 및 아이디어 공유

## 📄 라이선스

MIT License - 자세한 내용은 LICENSE 파일 참조

---

**🚀 DevOps Assistant Agent로 개발 생산성을 한 단계 업그레이드하세요!** 