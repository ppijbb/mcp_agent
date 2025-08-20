# Ethereum Trading Multi-Agent System

이더리움 거래를 위한 멀티 에이전트 시스템으로, Gemini 2.5 Flash AI와 MCP(Multi-Chain Protocol)를 사용하여 5분 단위로 자동 거래를 수행합니다.

## 주요 특징

- **Gemini 2.0 Flash AI**: Google의 최신 AI 모델을 사용한 시장 분석 및 거래 결정
- **LangChain 0.3.0 + LangGraph 통합**: 최신 LangChain 프레임워크와 LangGraph를 활용한 향상된 에이전트 오케스트레이션
- **강화된 MCP 통합**: 우선순위 기반 병렬 처리, 타임아웃 관리, 재시도 로직을 갖춘 고급 MCP 서버 활용
- **멀티 에이전트 구조**: 전통적 에이전트 3개 + LangChain 향상 에이전트 3개 (총 6개)
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

**참고**: 
- `google-generativeai` 라이브러리는 더 이상 사용되지 않으며, `google-genai`로 업데이트되었습니다. 이는 Google의 최신 생성형 AI 라이브러리입니다.
- **LangChain 0.3.0**을 사용하므로 Python 3.9 이상이 필요합니다.
- Pydantic 2.0을 사용하여 향상된 데이터 검증을 제공합니다.

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

#### 전통적 에이전트
```python
from cron_agents.ethereum_trading_agents.trading_agent import TradingAgent

# 에이전트 생성
agent = TradingAgent("test_agent")

# 거래 사이클 실행
result = await agent.execute_trading_cycle()
print(result)
```

#### LangChain + LangGraph 향상 에이전트
```python
from cron_agents.ethereum_trading_agents.langchain_agent import LangChainTradingAgent

# 향상된 에이전트 생성
agent = LangChainTradingAgent("test_enhanced_agent")

# LangGraph 워크플로우 실행
result = await agent.execute_trading_cycle()
print(result)

# MCP 강화 기능 테스트
async with agent.mcp_client:
    enhanced_ops = await agent.mcp_client.enhanced_mcp_operations()
    print(f"MCP Health Score: {enhanced_ops.get('health_score', 0)}%")
```

## 에이전트 전략

### 전통적 에이전트 (Traditional Agents)

#### 1. Conservative Trader (보수적 거래자)
- 낮은 리스크 선호
- 작은 거래 금액
- 엄격한 손절매/익절매

#### 2. Aggressive Trader (공격적 거래자)
- 높은 리스크 허용
- 큰 거래 금액
- 빠른 진입/청산

#### 3. Balanced Trader (균형잡힌 거래자)
- 중간 리스크 수준
- 적응적 거래 전략
- 시장 상황에 따른 유연한 대응

### LangChain 향상 에이전트 (LangChain Enhanced Agents)

#### 1. LangChain Conservative (LangChain 보수적 거래자)
- LangChain 0.3.0의 도구 기반 에이전트 시스템 활용
- 향상된 시장 분석 및 리스크 평가
- 구조화된 의사결정 프로세스

#### 2. LangChain Aggressive (LangChain 공격적 거래자)
- LangChain 에이전트 실행기를 통한 고급 거래 전략
- 실시간 시장 연구 및 인사이트 분석
- 동적 리스크 관리

#### 3. LangChain Balanced (LangChain 균형잡힌 거래자)
- LangChain 프레임워크의 최신 기능 활용
- 멀티 도구 기반 의사결정
- 적응형 전략 실행

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

### 4. LangChain 0.3.0 + LangGraph 최적화
- 도구 기반 에이전트 시스템으로 효율적인 작업 분배
- 구조화된 프롬프트 템플릿으로 일관된 응답 품질
- 에이전트 실행기(AgentExecutor)를 통한 최적화된 실행
- LangGraph 워크플로우로 복잡한 거래 프로세스 자동화
- 상태 기반 워크플로우 관리로 에러 처리 및 복구 자동화

### 5. 강화된 MCP 기능
- 우선순위 기반 병렬 처리 (고/중/저 우선순위)
- 지능형 타임아웃 관리 (15초/25초/45초)
- 자동 재시도 로직 및 에러 복구
- MCP 서버 상태 모니터링 및 헬스 스코어링
- 실시간 MCP 작업 추적 및 로깅

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

#### 전통적 에이전트
```python
class CustomTradingAgent(TradingAgent):
    def __init__(self, agent_name: str):
        super().__init__(agent_name)
        # 커스텀 로직 추가
    
    async def custom_strategy(self):
        # 커스텀 전략 구현
        pass
```

#### LangChain 향상 에이전트
```python
class CustomLangChainAgent(LangChainTradingAgent):
    def __init__(self, agent_name: str):
        super().__init__(agent_name)
        # 추가 도구 및 커스텀 로직
    
    def _setup_langchain_components(self):
        super()._setup_langchain_components()
        # 추가 도구 등록
        self.tools.append(self._create_custom_tool())
```

### 2. 새로운 MCP 서버 통합
```python
class CustomMCPClient(MCPClient):
    async def custom_function(self):
        # 커스텀 MCP 기능
        pass
```

### 3. LangChain 도구 확장
```python
@tool
def custom_market_tool(self) -> BaseTool:
    return BaseTool(
        name="custom_market_analysis",
        description="Custom market analysis tool",
        func=self._custom_analysis,
        args_schema=CustomSchema
    )
```

## 문제 해결

### 1. 일반적인 오류

#### Gemini API 오류
- API 키 확인
- 할당량 확인
- 네트워크 연결 확인
- `google-genai` 라이브러리 버전 확인 (최신 버전 권장)

#### LangChain 0.3.0 + LangGraph 오류
- Python 3.9 이상 버전 확인
- Pydantic 2.0 호환성 확인
- LangChain 패키지 버전 확인 (`langchain>=0.3.0,<0.4.0`)
- LangGraph 패키지 버전 확인 (`langgraph>=0.2.20,<0.3`)
- 도구(Tools) 설정 및 스키마 검증 확인
- 워크플로우 컴파일 및 상태 그래프 설정 확인

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
