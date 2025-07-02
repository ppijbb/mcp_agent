# 🚀 DevOps Assistant Agent

**MCP 기반 개발자 생산성 자동화 에이전트**

[![Model](https://img.shields.io/badge/Model-gemini--2.5--flash--lite--preview--0607-blue)](https://github.com/google/generative-ai)
[![Framework](https://img.shields.io/badge/Framework-MCP_Agent-green)](https://github.com/modelcontextprotocol/mcp_agent)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📋 개요

DevOps Assistant Agent는 개발팀의 생산성 향상을 위한 종합적인 자동화 도구입니다. GitHub 코드 리뷰부터 CI/CD 모니터링, 보안 스캔까지 DevOps 전 영역을 지원합니다.

## ✨ 주요 기능

### 🔍 **코드 리뷰 자동화**
- GitHub Pull Request 자동 분석
- 코드 품질, 보안, 성능 평가
- 건설적 피드백 및 개선사항 제안
- 승인/수정요청 권장사항 제공

### 🚀 **CI/CD 파이프라인 모니터링**
- 배포 상태 실시간 확인
- 서비스 헬스체크 분석
- 리소스 사용률 모니터링
- 잠재적 위험 요소 식별

### 🎯 **이슈 우선순위 분석**
- GitHub 이슈 자동 분류
- 우선순위 매트릭스 (P0-P3)
- 예상 작업시간 추정
- 팀별 업무 배분 제안

### 👥 **팀 스탠드업 자동 생성**
- 24시간 팀 활동 요약
- 완료/진행/차단 사항 정리
- 핵심 메트릭 하이라이트
- 실행 가능한 액션 아이템

### 📊 **성능 분석 및 최적화**
- 응답시간, 처리량, 에러율 분석
- 병목 지점 식별
- 리소스 최적화 방안 제시
- SRE 관점의 개선 권장사항

### 🔒 **보안 스캔 및 컴플라이언스**
- 취약점 자동 탐지
- OWASP Top 10 준수 확인
- 우선순위별 보안 패치 계획
- 컴플라이언스 개선사항 제안

## 🛠️ 설치 및 설정

### 1. 환경 요구사항
```bash
Python 3.8+
MCP Agent Framework
Google Generative AI API Key
```

### 2. 의존성 설치
```bash
pip install -r config/requirements.txt
```

### 3. 설정 파일
MCP Agent 설정 파일을 준비하세요:
```yaml
# configs/mcp_agent.config.yaml
model:
  name: "gemini-2.5-flash-lite-preview-0607"
  temperature: 0.2
  max_tokens: 2000

api:
  google_api_key: "${GOOGLE_API_KEY}"
```

### 4. 환경 변수 설정
```bash
export GOOGLE_API_KEY="your-google-api-key"
```

## 🚀 사용법

### 대화형 실행
```bash
cd scripts
python run_devops_assistant.py
```

### 프로그래매틱 사용
```python
import asyncio
from agents.devops_assistant_agent import create_devops_assistant, run_code_review

async def main():
    # 에이전트 생성
    agent = await create_devops_assistant()
    
    # 코드 리뷰 실행
    result = await run_code_review(
        agent, 
        owner="myorg",
        repo="myrepo", 
        pull_number=123
    )
    
    print(f"리뷰 결과: {result.status}")
    for rec in result.recommendations:
        print(f"- {rec}")

asyncio.run(main())
```

## 📱 인터페이스

### 메인 메뉴
```
🛠️  DevOps Assistant Agent - Main Menu
============================================================
1. 🔍 코드 리뷰 분석 (Code Review)
2. 🚀 배포 상태 확인 (Deployment Check)
3. 🎯 이슈 우선순위 분석 (Issue Analysis)
4. 👥 팀 스탠드업 생성 (Team Standup)
5. 📊 성능 분석 (Performance Analysis)
6. 🔒 보안 스캔 (Security Scan)
7. 📋 작업 히스토리 (Task History)
8. 📈 종합 리포트 (Summary Report)
9. 🏢 팀 메트릭 (Team Metrics)
0. 🚪 종료 (Exit)
============================================================
```

## 📊 출력 예시

### 코드 리뷰 결과
```json
{
  "task_type": "🔍 코드 리뷰",
  "status": "completed",
  "processing_time": 2.35,
  "recommendations": [
    "코드 리뷰 완료: myorg/myrepo#123",
    "CI/CD 파이프라인 상태 확인 필요",
    "테스트 커버리지 80% 이상 유지 권장",
    "보안 스캔 결과 검토 필요"
  ]
}
```

### 팀 스탠드업 요약
```
👥 Backend Team 스탠드업 요약

📅 어제 완료된 작업:
- 3개 PR 머지 (인증 시스템 개선)
- 7개 이슈 해결 (버그 수정 위주)
- 15회 커밋 (활발한 개발 활동)

🎯 오늘 예정된 작업:
- 4개 PR 리뷰 대기
- P0 보안 이슈 1건 처리
- 성능 최적화 테스트

🚫 차단 요소:
- 빌드 성공률 94.5% (목표 95% 미달성)
- 평균 리뷰 시간 2.3시간 (양호)
```

## 🏗️ 아키텍처

```
srcs/devops_productivity_agent/
├── 📁 agents/                      # 핵심 에이전트 모듈
│   ├── devops_assistant_agent.py   # 메인 에이전트 클래스
│   └── __init__.py                 # 에이전트 모듈 초기화
├── 📁 scripts/                     # 실행 스크립트들
│   ├── run_devops_assistant.py     # 대화형 실행기
│   └── __init__.py                 # 스크립트 모듈 초기화
├── 📁 config/                      # 설정 파일들
│   ├── requirements.txt            # 패키지 의존성
│   └── __init__.py                 # 설정 모듈 초기화
├── 📁 docs/                        # 문서 파일들
│   └── README.md                   # 프로젝트 문서
└── __init__.py                     # 패키지 루트 초기화

데이터 클래스:
├── CodeReviewRequest              # 코드 리뷰 요청
├── DeploymentStatus              # 배포 상태
├── IssueAnalysis                 # 이슈 분석 결과
├── TeamActivity                  # 팀 활동 데이터
└── DevOpsResult                  # 작업 결과

작업 유형:
├── CODE_REVIEW                   # 🔍 코드 리뷰
├── DEPLOYMENT_CHECK              # 🚀 배포 확인
├── ISSUE_ANALYSIS                # 🎯 이슈 분석
├── TEAM_STANDUP                  # 👥 스탠드업
├── PERFORMANCE_ANALYSIS          # 📊 성능 분석
└── SECURITY_SCAN                 # 🔒 보안 스캔
```

## 🔧 고급 사용법

### 1. 배치 처리
```python
async def batch_analysis():
    from agents.devops_assistant_agent import create_devops_assistant, run_code_review, run_deployment_check, run_security_scan
    
    agent = await create_devops_assistant()
    
    # 여러 작업 동시 실행
    tasks = [
        run_code_review(agent, "org", "repo1", 123),
        run_deployment_check(agent, "web-api"),
        run_security_scan(agent, "https://api.example.com")
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

### 2. 커스텀 설정
```python
from agents.devops_assistant_agent import DevOpsAssistantMCPAgent

agent = DevOpsAssistantMCPAgent(
    output_dir="custom_reports"
)
```

### 3. 결과 저장 및 분석
```python
# 작업 히스토리 조회
history = agent.get_task_history()

# 팀 메트릭 조회  
metrics = agent.get_team_metrics()

# 종합 리포트 생성
report = agent.get_summary_report()
```

## 📈 성능 최적화

### 모델 설정
- **Temperature**: 0.1-0.3 (분석 작업에 최적화)
- **Max Tokens**: 800-1200 (작업 유형별 조정)
- **Response Time**: 평균 2-5초

### 메모리 사용량
- **기본 사용량**: ~50MB
- **작업당 추가**: ~5-10MB
- **최대 권장**: 100개 작업 히스토리

## 🔍 문제 해결

### 일반적인 오류

**1. Google API 키 오류**
```bash
export GOOGLE_API_KEY="your-actual-api-key"
```

**2. MCP Agent 설정 오류**
```bash
# 설정 파일 경로 확인
ls configs/mcp_agent.config.yaml
```

**3. 의존성 충돌**
```bash
pip install --upgrade mcp_agent google-generativeai
```

### 로그 확인
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 지원

- 🐛 **버그 리포트**: [Issues](https://github.com/your-org/devops-assistant/issues)
- 💬 **토론**: [Discussions](https://github.com/your-org/devops-assistant/discussions)
- 📧 **이메일**: devops-assistant@example.com

## 🙏 감사의 글

- [MCP Agent Framework](https://github.com/modelcontextprotocol/mcp_agent)
- [Google Generative AI](https://github.com/google/generative-ai)
- [GitHub API](https://docs.github.com/en/rest)

---

**🚀 DevOps Assistant Agent로 팀의 생산성을 한 단계 높여보세요!** 