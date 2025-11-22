# Financial Agent - 종합 재무 관리 및 투자 에이전트 시스템

## 개요

Financial Agent는 LangGraph와 MCP(Model Context Protocol)를 활용한 종합 재무 관리 및 투자 에이전트 시스템입니다. Multi-Model LLM Manager(Groq/OpenRouter 우선, Gemini/OpenAI/Claude 지원)와 실시간 시장 데이터를 통합하여 재무 분석, 세금 최적화, 부채 관리, 재무 목표 추적, 그리고 투자 의사결정을 자동화합니다.

## 주요 특징

- **종합 재무 관리**: 재무 분석, 세금 최적화, 부채 관리, 재무 목표 추적을 통합 제공
- **Multi-Model LLM 지원**: Groq/OpenRouter 우선 사용, Gemini/OpenAI/Claude 지원 (NO FALLBACK 정책)
- **자율 에이전트 워크플로우**: 13개의 전문 에이전트가 협력하여 재무 관리 및 투자 프로세스 수행
- **실시간 데이터 수집**: MCP를 통한 yfinance 기반 실시간 시장 데이터 및 뉴스 수집
- **구조적 상업성**: 거래 수수료 및 제휴 수수료 계산을 통한 수익 모델
- **Production Ready**: 모든 환경 변수 필수, 하드코딩 제거, 명확한 에러 처리

## 시스템 아키텍처

```
┌─────────────────────┐
│ Financial Analyzer  │ (재무 분석)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Tax Optimizer       │ (세금 최적화)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Debt Manager        │ (부채 관리)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Goal Tracker        │ (재무 목표 추적)
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼──────┐  ┌───▼──────────┐
│ Market   │  │ News         │
│ Data     │  │ Collector    │
│ Collector│  │              │
└────┬─────┘  └──────┬───────┘
     │                │
     └────────┬───────┘
              │
     ┌────────▼────────┐
     │ Sync Node       │
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │ News Analyzer   │
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │ Chief           │
     │ Strategist      │
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │ Portfolio       │
     │ Manager         │
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │ Trader          │
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │ Commission      │
     │ Calculator      │
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │ Auditor         │
     └─────────────────┘
```

## 에이전트 구성

### 재무 관리 에이전트 (신규)
1. **Financial Analyzer**: 소비 패턴 분석, 예산 관리, 저축 목표 추적, 재무 건강 점수 계산
2. **Tax Optimizer**: 공제 항목 발견, 세금 계산, 세금 최적화 전략 제안
3. **Debt Manager**: 대출 상환 전략, 이자 최소화 계획, 부채 구조 분석
4. **Goal Tracker**: 장기 재무 목표 관리, 목표 달성 진행률 추적, 목표별 투자 전략 제안
5. **Commission Calculator**: 거래 수수료 계산, 제휴 수수료 계산, 수익 추적

### 투자 워크플로우 에이전트 (기존)
6. **Market Data Collector**: 기술적 지표 수집 (RSI, MACD, 이동평균)
7. **News Collector**: 실시간 뉴스 수집
8. **Sync Node**: 병렬 데이터 수집 동기화
9. **News Analyzer**: 뉴스 감성 분석
10. **Chief Strategist**: 종합 시장 전망 생성
11. **Portfolio Manager**: 투자 계획 수립
12. **Trader**: 실제 거래 실행 (하드 코딩 제거, 설정 기반)
13. **Auditor**: 거래 결과 검증

## 설치 및 설정

### 1. 환경 설정

```bash
# uv 환경 사용 (권장)
uv venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate     # Windows

# 또는 conda 환경 사용
conda create -n financial-agent python=3.10
conda activate financial-agent
```

### 2. 의존성 설치

```bash
pip install langgraph langchain-google-genai langchain-groq langchain-openai langchain-anthropic mcp yfinance pandas numpy
```

### 3. 환경 변수 설정

`env.example` 파일을 `.env`로 복사하고 실제 값으로 수정:

```bash
cp env.example .env
```

필수 환경 변수:
```env
# LLM 설정 (기존 Gemini 전용 - 하위 호환성)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
LLM_TEMPERATURE=0.7

# Multi-Model LLM 설정 (최소 1개 필요, Groq 권장)
PREFERRED_LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# MCP 서버 설정
MCP_TIMEOUT=30
MCP_DATA_PERIOD=3mo

# 거래 설정
DEFAULT_SHARES=10
MAX_TRADE_AMOUNT=10000.0
COMMISSION_RATE=0.005
AFFILIATE_COMMISSION_RATE=0.03

# 워크플로우 설정
RISK_PROFILES=conservative,moderate,aggressive
DEFAULT_TICKERS=NVDA,AMD,QCOM,AAPL,MSFT
```

### 4. API 키 발급

- **Groq API Key**: [Groq Console](https://console.groq.com/)에서 발급 (권장)
- **OpenRouter API Key**: [OpenRouter](https://openrouter.ai/)에서 발급
- **Gemini API Key**: [Google AI Studio](https://aistudio.google.com/)에서 발급
- **OpenAI API Key**: [OpenAI Platform](https://platform.openai.com/)에서 발급
- **Anthropic API Key**: [Anthropic Console](https://console.anthropic.com/)에서 발급
- 모든 환경 변수는 필수이며, 누락 시 시스템이 시작되지 않습니다.
- Multi-Model LLM: 최소 1개의 API 키가 필요합니다 (GROQ_API_KEY 권장)

## 사용법

### 기본 실행

```bash
python graph.py "NVDA,AMD,QCOM" aggressive
```

### 매개변수

- **티커 목록**: 분석할 주식 티커 (쉼표로 구분)
- **리스크 프로필**: `conservative`, `moderate`, `aggressive` 중 선택

### 실행 예시

```bash
# 보수적 투자 전략
python graph.py "AAPL,MSFT" conservative

# 공격적 투자 전략
python graph.py "NVDA,AMD,QCOM,TSLA" aggressive

# 균형 투자 전략
python graph.py "GOOGL,AMZN,META" moderate
```

## 출력 결과

시스템은 다음과 같은 결과를 제공합니다:

### 재무 관리 결과
- **재무 분석**: 소비 패턴, 예산 상태, 저축 목표 진행률, 재무 건강 점수
- **세금 최적화**: 공제 항목, 세금 계산, 최적화 전략 및 예상 절감액
- **부채 관리**: 부채 구조 분석, 상환 전략, 이자 최소화 계획
- **재무 목표**: 목표별 진행률, 투자 전략 제안, 목표 달성 가능성 분석
- **수수료 계산**: 거래 수수료, 제휴 수수료, 총 수익

### 투자 워크플로우 결과
- **기술적 분석**: RSI, MACD, 이동평균 등
- **뉴스 감성 분석**: 시장 뉴스의 감성 점수
- **시장 전망**: LLM 기반 종합 분석 결과
- **투자 계획**: 매수/매도/보유 권장사항
- **거래 실행**: 실제 거래 시뮬레이션 결과 (설정 기반 수량)
- **일일 손익**: 거래 결과 기반 P&L

## Production 배포

### 환경 요구사항

- Python 3.10+
- 모든 환경 변수 필수 설정
- 안정적인 인터넷 연결 (API 호출용)
- 충분한 메모리 (병렬 처리용)

### 모니터링

시스템은 다음과 같은 에러 상황을 명확히 보고합니다:

- 환경 변수 누락
- API 키 인증 실패
- MCP 서버 연결 실패
- LLM 호출 실패
- 데이터 수집 실패

### 로그

모든 에이전트의 활동은 상태 로그에 기록되며, 에러 발생 시 명확한 메시지를 제공합니다.

## 문제 해결

### 일반적인 문제

1. **환경 변수 누락**
   ```
   ❌ 설정 로드 실패: 필수 환경 변수 GEMINI_API_KEY가 설정되지 않았습니다.
   ```
   → `.env` 파일 확인 및 API 키 설정

2. **MCP 서버 연결 실패**
   ```
   ❌ MCP 서버 연결 실패 - 도구: get_technical_indicators
   ```
   → 네트워크 연결 및 yfinance 의존성 확인

3. **LLM 호출 실패**
   ```
   ❌ LLM 호출 중 에러 발생: API quota exceeded
   ```
   → API 키 할당량 확인

## 새로운 기능

### 재무 분석
- 소비 패턴 분석 (최근 30일 기준)
- 예산 관리 및 추적
- 저축 목표 진행률 추적
- 재무 건강 점수 계산 (0-100점)

### 세금 최적화
- 공제 항목 자동 발견
- 세금 계산 (한국 세법 기준)
- 세금 최적화 전략 제안
- 예상 절감액 계산

### 부채 관리
- 부채 구조 분석
- 상환 전략 생성 (Snowball/Avalanche)
- 이자 최소화 계획
- 상환 기간 단축 시뮬레이션

### 재무 목표 추적
- 장기 재무 목표 설정 및 관리
- 목표별 진행률 추적
- 목표 기간에 따른 투자 전략 제안
- 목표 달성 가능성 분석

### 구조적 상업성
- 거래 수수료 계산 (거래당 0.5-1%)
- 제휴 수수료 계산 (금융 상품 추천 시 3-5%)
- 수익 추적 및 분석

## 개발자 가이드

### 새로운 에이전트 추가

1. `agents/` 디렉토리에 새 에이전트 파일 생성
2. 노드 함수 구현 (예: `new_agent_node(state: AgentState) -> Dict`)
3. `agents/__init__.py`에 노드 함수 임포트 추가
4. `graph.py`의 `_build_graph()` 메서드에 노드 및 엣지 추가
5. `state.py`에 필요한 상태 필드 추가

### 설정 확장

`config.py`의 설정 클래스에 새 필드 추가:

```python
@dataclass
class NewConfig:
    new_setting: str

def _get_required_env(key: str, var_type: type = str) -> Any:
    # 새 환경 변수 처리 로직 추가
```

### Multi-Model LLM 사용

시스템은 자동으로 사용 가능한 LLM Provider를 선택합니다:
1. Groq (우선순위 1)
2. OpenRouter (우선순위 2)
3. Gemini (우선순위 3)
4. OpenAI (우선순위 4)
5. Claude (우선순위 5)

`PREFERRED_LLM_PROVIDER` 환경 변수로 선호하는 Provider를 지정할 수 있습니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다. 기여하기 전에 이슈를 먼저 생성해 주세요.

## 연락처

프로젝트 관련 문의사항이 있으시면 이슈를 통해 연락해 주세요.
