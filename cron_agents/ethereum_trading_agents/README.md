# Ethereum Trading Multi-Agent System

이더리움 거래를 위한 멀티 에이전트 시스템으로, Gemini 2.5 Flash AI와 MCP(Multi-Chain Protocol)를 사용하여 5분 단위로 자동 거래를 수행합니다.

**🚀 LangChain 기반 모듈화된 아키텍처로 완전히 재구성되었습니다!**

**📧 새로운 기능: 자동 거래 리포트 및 이메일 알림 시스템!**

## 🏗️ 새로운 모듈화된 아키텍처

이 시스템은 이제 LangChain의 모범 사례를 따르는 완전히 모듈화된 구조를 가지고 있습니다:

```
ethereum_trading_agents/
├── agents/                    # 에이전트 구현체
│   ├── trading_agent.py      # 핵심 거래 에이전트
│   ├── gemini_agent.py       # Gemini 기반 분석 에이전트
│   ├── langchain_agent.py    # LangChain 기반 에이전트
│   ├── trading_report_agent.py # 거래 리포트 생성 에이전트
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
│   ├── cron_scheduler.py     # 크론 스케줄러
│   ├── email_service.py      # 이메일 서비스
│   ├── trading_monitor.py    # 거래 모니터링
│   └── data_collector.py     # 데이터 수집기
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

### **🆕 새로운 이메일 리포트 및 모니터링 기능**
- **실시간 거래 모니터링**: 블록체인에서 거래 발생 시 자동 감지
- **상세한 거래 리포트**: "언제", "어떤 거래", "얼마나", "왜" 정보를 포함한 포괄적인 분석
- **자동 이메일 알림**: 거래 발생 시 즉시 알림 및 상세 리포트 전송
- **MCP 이메일 연동**: MCP 서버를 통한 이메일 전송 (SMTP 폴백 지원)
- **일일/주간/월간 요약**: 정기적인 포트폴리오 성과 리포트
- **리스크 분석**: 거래별 리스크 평가 및 권장사항 제공
- **시장 컨텍스트 분석**: 거래 시점의 시장 상황 및 전략적 요인 분석

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
                                │                             │
                                ▼                             ▼
                       ┌─────────────────┐         ┌─────────────────┐
                       │ TradingMonitor  │         │ EmailService    │
                       │                 │         │                 │
                       │ - 거래 모니터링   │         │ - 이메일 전송    │
                       │ - 리포트 생성      │         │ - MCP 연동      │
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

┌──────────────────┐    ┌─────────────────┐
│ TradingReport    │    │ TradingMonitor  │
│ Agent            │    │                 │
│                  │    │                 │
│ - 리포트 생성      │    │ - 거래 모니터링   │
│ - 분석 로직       │    │ - 자동 알림      │
└──────────────────┘    └─────────────────┘
```

## 📧 이메일 리포트 시스템

### **거래 발생 시 자동 알림**
- **즉시 알림**: 거래 확인 후 30초 내 이메일 전송
- **상세 리포트**: 거래 후 포괄적인 분석 리포트 전송
- **HTML + 텍스트**: 모든 이메일 클라이언트 호환

### **포함 정보**
1. **언제 (When)**: 거래 실행 시간, 블록 번호, 타임스탬프
2. **어떤 거래 (What)**: 거래 유형, 스마트 컨트랙트 상호작용 여부
3. **얼마나 (How Much)**: 거래 금액, 가스 사용량, 가스 가격
4. **왜 (Why)**: 시장 분석, 기술적 지표, 뉴스 영향, 전략적 요인

### **이메일 전송 방식**
- **1순위**: MCP 서버를 통한 이메일 전송
- **2순위**: SMTP를 통한 직접 이메일 전송
- **자동 폴백**: MCP 실패 시 자동으로 SMTP 사용

## 🔍 거래 모니터링 시스템

### **실시간 모니터링**
- **블록 단위 스캔**: 새로운 블록마다 거래 확인
- **주소 필터링**: 특정 주소만 모니터링 가능
- **중복 방지**: 이미 처리된 거래 재처리 방지

### **자동 리포트 생성**
- **거래별 리포트**: 각 거래마다 개별 상세 리포트
- **일일 요약**: 매일 자정 자동 생성 및 전송
- **주간/월간 요약**: 정기적인 성과 분석 리포트

### **리스크 관리**
- **실시간 리스크 평가**: 거래별 리스크 점수 계산
- **권장사항 제공**: 리스크 완화 전략 제시
- **포트폴리오 모니터링**: 전체 포트폴리오 리스크 추적

## 📦 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

**참고**: 
- Python 3.9+ 필요
- LangChain 0.3.0+ 호환성 확인
- Redis 서버 실행 필요

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp env_example.txt .env

# 필수 설정값 입력
GEMINI_API_KEY=your_gemini_api_key
ETHEREUM_RPC_URL=your_ethereum_rpc_url
ETHEREUM_PRIVATE_KEY=your_private_key
ETHEREUM_ADDRESS=your_ethereum_address

# 이메일 설정
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECIPIENTS=trader1@example.com,trader2@example.com

# MCP 서버 설정
MCP_ETHEREUM_TRADING_URL=http://localhost:3005
MCP_MARKET_DATA_URL=http://localhost:3006
MCP_EMAIL_URL=http://localhost:3007
MCP_EMAIL_API_KEY=your_mcp_email_api_key

# 모니터링 설정
MONITORING_ADDRESSES=0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6
MONITORING_INTERVAL_SECONDS=60
REPORT_GENERATION_DELAY_SECONDS=30
```

### 3. 시스템 실행

```bash
# 메인 시스템 실행
python -m ethereum_trading_agents.main

# 또는 직접 실행
python main.py
```

## 🚀 사용법

### **자동 모니터링 시작**
시스템 실행 시 자동으로 거래 모니터링이 시작됩니다:

```python
# 시스템 상태 확인
status = await system.get_system_status()
print(f"Trading Monitor: {status['components']['trading_monitor']}")
print(f"Email Service: {status['components']['email_service']}")
```

### **수동 리포트 생성**
특정 거래에 대한 리포트를 수동으로 생성:

```python
# 특정 거래 해시로 리포트 생성
report = await system.trading_monitor.force_report_generation(
    "0x1234567890abcdef..."
)

# 이메일로 전송
success = await system.trading_report_agent.send_report_email(
    "0x1234567890abcdef..."
)
```

### **모니터링 상태 확인**
```python
# 모니터링 상태 조회
monitor_status = system.trading_monitor.get_monitoring_status()
print(f"Active: {monitor_status['monitoring_active']}")
print(f"Processed Transactions: {monitor_status['processed_transactions_count']}")
print(f"Daily Trades: {monitor_status['daily_trades_count']}")
```

### **거래 히스토리 조회**
```python
# 전체 거래 히스토리
history = await system.trading_monitor.get_transaction_history(limit=50)

# 특정 주소 거래 히스토리
address_history = await system.trading_monitor.get_transaction_history(
    address="0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
    limit=20
)
```

## 📊 리포트 예시

### **거래 알림 이메일**
```
제목: ⚡ Ethereum Transaction Executed - 0x1234567890...

내용:
- Transaction Hash: 0x1234567890abcdef...
- Status: Confirmed
- Amount: 1.5 ETH
- Gas Used: 21,000
- Timestamp: 2024-01-15 14:30:25
```

### **상세 거래 리포트**
```
제목: 🚀 Ethereum Trading Report - 2024-01-15 14:30

내용:
📊 Transaction Summary
- Block Number: 18,456,789
- From/To Addresses
- Value: 1.5 ETH
- Gas Details

🔍 Market Analysis
- Current ETH Price: $2,850
- 24h Change: +3.2%
- Market Sentiment: Bullish
- Technical Indicators: RSI 45.2, MACD Bullish

💡 Trading Insights
- Trade executed based on: Positive market sentiment indicating upward momentum; RSI indicates oversold conditions, potential buying opportunity
- Risk Level: Medium
- Recommendations: Standard risk management practices apply
```

### **일일 요약 리포트**
```
제목: 📊 Daily Trading Summary - 2024-01-15

내용:
📈 Portfolio Summary
- Total Trades: 8
- Successful Trades: 7
- Total Volume: 12.5 ETH
- Success Rate: 87.5%

🔄 Today's Trades
- 상세 거래 목록 테이블
```

## 🔧 고급 설정

### **모니터링 주소 설정**
```bash
# 특정 주소만 모니터링
MONITORING_ADDRESSES=0x1234...,0x5678...,0x9abc...

# 모든 주소 모니터링 (빈 값)
MONITORING_ADDRESSES=
```

### **리포트 생성 지연 시간 조정**
```bash
# 거래 확인 후 리포트 생성까지 대기 시간 (초)
REPORT_GENERATION_DELAY_SECONDS=60
```

### **모니터링 간격 조정**
```bash
# 블록 스캔 간격 (초)
MONITORING_INTERVAL_SECONDS=30
```

### **이메일 수신자 관리**
```bash
# 여러 수신자 설정 (쉼표로 구분)
EMAIL_RECIPIENTS=trader1@company.com,trader2@company.com,manager@company.com

# 개별 수신자별 맞춤 설정 가능
```

## 🛠️ 문제 해결

### **이메일 전송 실패**
1. **MCP 이메일 서버 확인**: `MCP_EMAIL_URL` 및 `MCP_EMAIL_API_KEY` 설정 확인
2. **SMTP 설정 확인**: Gmail 앱 비밀번호 사용 권장
3. **방화벽 설정**: 포트 587 (SMTP) 및 3007 (MCP) 열기

### **모니터링이 작동하지 않음**
1. **환경 변수 확인**: `MONITORING_ADDRESSES` 설정 확인
2. **로그 확인**: `ethereum_trading.log` 파일에서 오류 메시지 확인
3. **네트워크 연결**: 이더리움 RPC 및 MCP 서버 연결 상태 확인

### **리포트 생성 실패**
1. **API 키 확인**: 모든 필요한 API 키가 설정되어 있는지 확인
2. **데이터 수집기 상태**: `DataCollector` 연결 상태 확인
3. **메모리 사용량**: 시스템 리소스 부족 여부 확인

## 🔮 향후 개발 계획

### **단기 계획 (1-2개월)**
- [ ] 웹 대시보드 추가
- [ ] 모바일 알림 앱 개발
- [ ] 고급 차트 및 분석 도구

### **중기 계획 (3-6개월)**
- [ ] 다중 체인 지원 (Polygon, BSC 등)
- [ ] AI 기반 거래 전략 최적화
- [ ] 실시간 포트폴리오 분석

### **장기 계획 (6개월+)**
- [ ] 기관급 거래 시스템
- [ ] 규제 준수 및 감사 기능
- [ ] 글로벌 거래소 연동

## 📞 지원 및 문의

### **기술 지원**
- **GitHub Issues**: 버그 리포트 및 기능 요청
- **Documentation**: 상세한 API 문서 및 사용법
- **Community**: 개발자 커뮤니티 및 포럼

### **상업적 지원**
- **Enterprise Solutions**: 기업용 맞춤 솔루션
- **Consulting**: 거래 시스템 설계 및 최적화
- **Training**: 팀 교육 및 워크샵

---

**⚠️ 주의사항**: 이 시스템은 교육 및 연구 목적으로 제작되었습니다. 실제 거래에 사용하기 전에 충분한 테스트와 검증이 필요합니다. 암호화폐 거래는 높은 위험을 수반하므로 신중하게 접근하시기 바랍니다.
