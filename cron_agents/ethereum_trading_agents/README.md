# Ethereum Trading Multi-Agent System

이더리움 거래를 위한 멀티 에이전트 시스템으로, Gemini 2.5 Flash AI와 MCP(Multi-Chain Protocol)를 사용하여 5분 단위로 자동 거래를 수행합니다.

**🚀 LangChain 기반 모듈화된 아키텍처로 완전히 재구성되었습니다!**

## 🏗️ 새로운 모듈화된 아키텍처

이 시스템은 이제 LangChain의 모범 사례를 따르는 완전히 모듈화된 구조를 가지고 있습니다:

```
ethereum_trading_agents/
├── agents/                    # 에이전트 구현체
│   ├── trading_agent.py      # 핵심 거래 에이전트
│   ├── gemini_agent.py       # Gemini 기반 분석 에이전트
│   ├── langchain_agent.py    # LangChain 기반 에이전트
│   └── multi_agent_orchestrator.py # 다중 에이전트 조율
├── chains/                    # LangChain 워크플로우 체인
│   ├── trading_chain.py      # 거래 워크플로우 체인
│   └── analysis_chain.py     # 분석 체인
├── memory/                    # 메모리 관리 시스템
│   └── trading_memory.py     # 거래 메모리 관리
├── prompts/                   # 프롬프트 템플릿
│   └── trading_prompts.py    # 거래 관련 프롬프트
├── utils/                     # 유틸리티 및 도구
│   ├── database.py           # 데이터베이스 관리
│   ├── mcp_client.py         # MCP 클라이언트
│   ├── config.py             # 설정 관리
│   └── cron_scheduler.py     # 크론 스케줄러
├── tests/                     # 테스트 스위트
├── main.py                   # 통합된 메인 진입점
└── requirements.txt          # 의존성 파일
```

## 🚀 주요 특징

### **기존 기능**
- **Gemini 2.0 Flash AI**: Google의 최신 AI 모델을 사용한 시장 분석 및 거래 결정
- **LangChain 0.3.0 + LangGraph 통합**: 최신 LangChain 프레임워크와 LangGraph를 활용한 향상된 에이전트 오케스트레이션
- **강화된 MCP 통합**: 우선순위 기반 병렬 처리, 타임아웃 관리, 재시도 로직을 갖춘 고급 MCP 서버 활용
- **멀티 에이전트 구조**: 전통적 에이전트 3개 + LangChain 향상 에이전트 3개 (총 6개)
- **5분 단위 실행**: cron 스케줄러를 통한 정기적인 거래 사이클 실행
- **완전한 기록 관리**: 모든 실행 내용을 데이터베이스에 저장하고 다음 실행 시 참조
- **리스크 관리**: 일일 거래 한도, 손실 한도 등 체계적인 리스크 관리
- **Fallback 제거**: 오류 발생 시 오류 문구만 출력, 더미 거래나 대체 로직 없음

### **새로운 LangChain 기반 기능**
- **TradingChain**: 완전한 거래 워크플로우 오케스트레이션
- **AnalysisChain**: 포괄적인 시장 분석 (기술적, 기본적, 감정, 패턴)
- **TradingMemory**: Redis 기반 분산 메모리 시스템
- **Custom Prompts**: 전문적인 거래 프롬프트 템플릿
- **자동화된 워크플로우**: 15분마다 시장 분석, 일일 포트폴리오 리뷰, 주간 성과 분석

## 🏗️ 시스템 아키텍처

### **전체 시스템 구조**
```
┌─────────────────┐    ┌──────────────────────┐     ┌─────────────────┐
│   CronScheduler │    │ EthereumTrading      │     │ TradingChain    │
│                 │    │ System               │     │                 │
│ - 자동화된 작업   │──▶│ - 시스템 오케스트레이션  │──▶ │ - 거래 워크플로우 │
│ - 스케줄 관리     │    │ - 컴포넌트 초기화      │     │ - 체인 실행      │
└─────────────────┘    └──────────────────────┘     └─────────────────┘
                                │                          │
                                ▼                          ▼
                       ┌─────────────────┐         ┌─────────────────┐
                       │ MultiAgent      │         │ AnalysisChain   │
                       │ Orchestrator    │         │                 │
                       │ - 에이전트 조율   │         │ - 시장 분석      │
                       │ - 상태 모니터링   │         │ - 데이터 처리     │
                       └─────────────────┘         └─────────────────┘
                                │                             │
                                ▼                             ▼
                       ┌─────────────────┐         ┌─────────────────┐
                       │ TradingMemory   │         │ TradingDatabase │
                       │                 │         │                 │
                       │ - 컨텍스트 저장   │         │ - 실행 기록      │
                       │ - Redis 백업     │         │ - 거래 데이터    │
                       └─────────────────┘         └─────────────────┘
```

### **에이전트 구조**
```
┌──────────────────┐    ┌─────────────────┐    ┌───────────────────┐
│ TradingAgent     │    │ GeminiAgent     │    │ TradingAgentChain │
│                  │    │                 │    │                   │
│ - 전통적 거래 로직 │    │ - AI 기반 분석    │    │ - LangChain 통합  │
│ - 리스크 관리      │    │ - 감정 분석      │    │ - 체인 실행        │
└──────────────────┘    └─────────────────┘    └───────────────────┘
```

## 📦 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

**참고**: 
- `google-generativeai` 라이브러리는 더 이상 사용되지 않으며, `google-genai`로 업데이트되었습니다. 이는 Google의 최신 생성형 AI 라이브러리입니다.
- **LangChain 0.3.0**을 사용하므로 Python 3.9 이상이 필요합니다.
- Pydantic 2.0을 사용하여 향상된 데이터 검증을 제공합니다.

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 설정하세요:

```bash
# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API (LangChain용)
OPENAI_API_KEY=your_openai_api_key_here

# Ethereum Configuration
ETHEREUM_RPC_URL=https://mainnet.infura.io/v3/YOUR_PROJECT_ID
ETHEREUM_PRIVATE_KEY=your_private_key_here
ETHEREUM_ADDRESS=your_ethereum_address_here

# Trading Configuration
MIN_TRADE_AMOUNT_ETH=0.01
MAX_TRADE_AMOUNT_ETH=1.0
STOP_LOSS_PERCENT=5.0
TAKE_PROFIT_PERCENT=10.0

# Risk Management
MAX_DAILY_TRADES=10
MAX_DAILY_LOSS_ETH=0.1

# MCP Server URLs
MCP_ETHEREUM_TRADING_URL=http://localhost:3005
MCP_MARKET_DATA_URL=http://localhost:3006

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/trading_db
```

### 3. MCP 서버 실행

#### 이더리움 거래 MCP 서버
```bash
cd mcp_servers/ethereum_trading_mcp
python server.py
```

#### 시장 데이터 MCP 서버
```bash
cd mcp_servers/market_data_mcp
python server.py
```

## 🎯 사용법

### 1. 기본 시스템 실행
```bash
# 시스템 실행
python main.py
```

### 2. LangChain 체인 사용
```python
from chains import TradingChain, AnalysisChain

# 시장 분석
analysis_results = await analysis_chain.execute_comprehensive_analysis(
    market_data=market_data,
    analysis_type="technical"
)

# 거래 워크플로우
workflow_results = await trading_chain.execute_trading_workflow(
    market_data=market_data,
    trading_strategy="momentum",
    portfolio_status=portfolio_status
)
```

### 3. 메모리 관리
```python
from memory import TradingMemory, MemoryType

# 컨텍스트 저장
await memory.store(
    key="current_strategy",
    value=trading_strategy,
    memory_type=MemoryType.TRADING_CONTEXT
)

# 컨텍스트 검색
strategy = await memory.retrieve(
    key="current_strategy",
    memory_type=MemoryType.TRADING_CONTEXT
)
```

### 4. 커스텀 프롬프트 사용
```python
from prompts import get_prompt, create_custom_prompt

# 미리 정의된 프롬프트 사용
market_prompt = get_prompt("market_analysis")
risk_prompt = get_prompt("risk_assessment")

# 커스텀 프롬프트 생성
custom_prompt = create_custom_prompt(
    template="Analyze {data} for {purpose}",
    input_variables=["data", "purpose"],
    system_message="You are a trading expert"
)
```

## 🔄 자동화된 워크플로우

### 크론 작업
시스템은 다음 자동화된 작업을 포함합니다:

- **시장 분석**: 15분마다 실행
- **포트폴리오 리뷰**: 매일 오전 9시
- **성과 분석**: 매주 일요일 오전 10시

### 커스텀 자동화
```python
# 커스텀 크론 작업 추가
await cron_scheduler.add_job(
    func=your_function,
    trigger="interval",
    minutes=30,
    id="custom_job"
)
```

## 📊 시스템 모니터링

### 시스템 상태 확인
```python
status = await system.get_system_status()
print(f"시스템: {status['status']}")
print(f"컴포넌트: {status['components']}")
print(f"메모리 사용률: {status['memory_stats']['memory_usage_percent']}%")
```

### 성능 지표
- 응답 시간 추적
- 메모리 사용량 모니터링
- 오류율 분석
- 에이전트 성능 지표

## 🧪 테스트

### 테스트 실행
```bash
# 모든 테스트 실행
pytest

# 커버리지와 함께 실행
pytest --cov=.

# 특정 테스트 파일 실행
pytest tests/test_trading_agent.py

# 비동기 테스트 실행
pytest --asyncio-mode=auto
```

## 🔒 보안 기능

- 암호화된 데이터 저장
- 보안 API 통신
- 역할 기반 접근 제어
- 감사 로깅
- 입력 검증

## 🚀 배포

### Docker
```bash
# 이미지 빌드
docker build -t ethereum-trading-agents .

# 컨테이너 실행
docker run -d \
  --name trading-agents \
  --env-file .env \
  -p 8000:8000 \
  ethereum-trading-agents
```

### Kubernetes
```bash
# 배포 적용
kubectl apply -f k8s/deployment.yaml

# 서비스 적용
kubectl apply -f k8s/service.yaml
```

## 📈 성능 최적화

- Async/await 패턴
- Redis 캐싱
- 연결 풀링
- 백그라운드 작업 처리
- 메모리 최적화

## 🤝 기여

### 개발 설정
```bash
# 개발 의존성 설치
pip install -r requirements-dev.txt

# pre-commit 훅 설정
pre-commit install

# 코드 포맷팅
black .
isort .

# 린팅
flake8
mypy .
```

### 코드 표준
- PEP 8 준수
- 타입 힌트 사용
- 포괄적인 테스트 작성
- 모든 함수 문서화
- LangChain 패턴 준수

## 📚 문서

- [LangChain Documentation](https://python.langchain.com/)
- [에이전트 아키텍처 가이드](docs/agents.md)
- [체인 개발 가이드](docs/chains.md)
- [메모리 시스템 가이드](docs/memory.md)
- [API 참조](docs/api.md)

## 🆘 지원

### 문제 해결
- 로그에서 오류 메시지 확인
- 환경 변수 검증
- Redis 및 데이터베이스 실행 확인
- API 키 유효성 확인

### 일반적인 문제
1. **연결 오류**: 데이터베이스 및 Redis URL 확인
2. **API 오류**: API 키 설정 확인
3. **메모리 문제**: Redis 메모리 사용량 모니터링
4. **성능 문제**: 비동기 패턴 및 캐싱 확인

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 제공됩니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- LangChain 팀의 훌륭한 프레임워크
- OpenAI, Google, Anthropic의 LLM API
- 오픈소스 거래 커뮤니티
- 기여자 및 유지보수자

---

**참고**: 이는 프로덕션 준비가 완료된 거래 시스템입니다. 실제 자금으로 사용하기 전에 안전한 환경에서 충분히 테스트하세요.
