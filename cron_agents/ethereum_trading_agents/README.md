# Ethereum Trading Multi-Agent System

이더리움 거래를 위한 멀티 에이전트 시스템으로, Gemini 2.5 Flash AI와 MCP(Multi-Chain Protocol)를 사용하여 5분 단위로 자동 거래를 수행합니다.

## 주요 특징

- **Gemini 2.5 Flash AI**: Google의 최신 AI 모델을 사용한 시장 분석 및 거래 결정
- **MCP 통합**: 이더리움 거래 및 시장 데이터 수집을 위한 MCP 서버 활용
- **멀티 에이전트 구조**: 보수적, 공격적, 균형잡힌 전략을 가진 3개 에이전트
- **5분 단위 실행**: cron 스케줄러를 통한 정기적인 거래 사이클 실행
- **완전한 기록 관리**: 모든 실행 내용을 데이터베이스에 저장하고 다음 실행 시 참조
- **리스크 관리**: 일일 거래 한도, 손실 한도 등 체계적인 리스크 관리
- **Fallback 제거**: 오류 발생 시 오류 문구만 출력, 더미 거래나 대체 로직 없음

## 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CronScheduler │    │ MultiAgent      │    │ TradingAgent    │
│                 │    │ Orchestrator    │    │                 │
│ - 5분 주기 실행 │───▶│ - 에이전트 조율 │───▶│ - 시장 분석     │
│ - 스케줄 관리   │    │ - 상태 모니터링 │    │ - 거래 결정     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   GeminiAgent   │    │   MCPClient     │
                       │                 │    │                 │
                       │ - AI 분석       │    │ - MCP 서버 통신 │
                       │ - 거래 결정     │    │ - 데이터 수집   │
                       └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ TradingDatabase │    │ MCP Servers     │
                       │                 │    │                 │
                       │ - 실행 기록     │    │ - 이더리움 거래 │
                       │ - 거래 데이터   │    │ - 시장 데이터   │
                       └─────────────────┘    └─────────────────┘
```

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 설정하세요:

```bash
# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

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

## 사용법

### 1. 크론 스케줄러 실행 (권장)

5분 단위로 자동 실행되는 크론 스케줄러:

```bash
cd cron_agents/ethereum_trading_agents
python cron_scheduler.py
```

### 2. 수동 실행

단일 실행을 위한 메인 스크립트:

```bash
cd cron_agents/ethereum_trading_agents
python main.py
```

### 3. 개별 에이전트 테스트

```python
from cron_agents.ethereum_trading_agents.trading_agent import TradingAgent

# 에이전트 생성
agent = TradingAgent("test_agent")

# 거래 사이클 실행
result = await agent.execute_trading_cycle()
print(result)
```

## 에이전트 전략

### 1. Conservative Trader (보수적 거래자)
- 낮은 리스크 선호
- 작은 거래 금액
- 엄격한 손절매/익절매

### 2. Aggressive Trader (공격적 거래자)
- 높은 리스크 허용
- 큰 거래 금액
- 빠른 진입/청산

### 3. Balanced Trader (균형잡힌 거래자)
- 중간 리스크 수준
- 적응적 거래 전략
- 시장 상황에 따른 유연한 대응

## 데이터베이스 스키마

### agent_executions
- 에이전트 실행 기록
- 상태, 입력/출력 데이터, 오류 메시지

### trading_decisions
- 거래 결정 내역
- 결정 유형, 데이터, 시장 상황, 추론

### market_snapshots
- 시장 데이터 스냅샷
- 가격, 변동률, 거래량, 기술적 지표

### risk_records
- 리스크 관리 기록
- 일일 거래 수, 손실, 리스크 레벨

## 모니터링 및 로깅

### 로그 파일
- `ethereum_trading.log`: 일반 거래 로그
- `ethereum_trading_cron.log`: 크론 스케줄러 로그

### 상태 확인
```python
from cron_agents.ethereum_trading_agents.cron_scheduler import CronScheduler

scheduler = CronScheduler()
status = await scheduler.get_scheduler_status()
print(status)
```

## 리스크 관리

### 1. 거래 한도
- 최소 거래 금액: 0.01 ETH
- 최대 거래 금액: 1.0 ETH
- 일일 최대 거래 수: 10회

### 2. 손실 관리
- 일일 최대 손실: 0.1 ETH
- 손절매: 5% 손실 시
- 익절매: 10% 이익 시

### 3. 가스비 관리
- 실시간 가스 가격 모니터링
- 거래 비용 최적화

## 오류 처리

시스템은 **Fallback을 제공하지 않습니다**. 오류 발생 시:

1. 오류 메시지를 로그에 기록
2. 거래를 중단하고 "hold" 상태로 전환
3. 다음 실행 시 이전 기록을 기반으로 동작

더미 거래나 대체 로직은 구현되지 않았습니다.

## 성능 최적화

### 1. 비동기 처리
- 모든 I/O 작업을 비동기로 처리
- 에이전트 동시 실행 지원

### 2. 배치 데이터 수집
- MCP 서버에서 데이터를 배치로 수집
- 네트워크 요청 최소화

### 3. 메모리 관리
- 데이터베이스 연결 풀링
- 정기적인 리소스 정리

## 보안 고려사항

### 1. API 키 보안
- 환경 변수를 통한 민감 정보 관리
- API 키 노출 방지

### 2. 개인키 보안
- 하드코딩 금지
- 환경 변수로 관리

### 3. 네트워크 보안
- HTTPS 통신
- MCP 서버 인증

## 확장성

### 1. 새로운 에이전트 추가
```python
class CustomTradingAgent(TradingAgent):
    def __init__(self, agent_name: str):
        super().__init__(agent_name)
        # 커스텀 로직 추가
    
    async def custom_strategy(self):
        # 커스텀 전략 구현
        pass
```

### 2. 새로운 MCP 서버 통합
```python
class CustomMCPClient(MCPClient):
    async def custom_function(self):
        # 커스텀 MCP 기능
        pass
```

## 문제 해결

### 1. 일반적인 오류

#### Gemini API 오류
- API 키 확인
- 할당량 확인
- 네트워크 연결 확인

#### MCP 서버 연결 오류
- 서버 상태 확인
- 포트 설정 확인
- 방화벽 설정 확인

#### 데이터베이스 오류
- 파일 권한 확인
- 디스크 공간 확인
- SQLite 버전 확인

### 2. 로그 분석
```bash
# 최근 오류 확인
grep "ERROR" ethereum_trading.log | tail -20

# 특정 에이전트 로그 확인
grep "conservative_trader" ethereum_trading.log
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다.

## 면책 조항

이 시스템은 교육 및 연구 목적으로 제작되었습니다. 실제 거래에 사용할 경우 발생하는 손실에 대해 책임지지 않습니다. 암호화폐 거래는 높은 리스크를 수반하므로 신중하게 접근하시기 바랍니다.
