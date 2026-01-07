# Money Maker Agent

24/7 자동 수익화 시스템 - 모든 작업이 자동으로 실행되며 수익을 계좌로 자동 송금합니다.

## 개요

Money Maker Agent는 리스크가 낮은 여러 에이전트를 통합하여 24시간 자동으로 수익을 창출하는 시스템입니다. 서버가 실행되는 동안 모든 작업이 자동으로 수행되며, 설정 파일만 수정하면 전략을 변경할 수 있습니다.

## 주요 기능

- **완전 자동화**: 서버 실행 중 모든 작업 자동 수행
- **실시간 설정 변경**: YAML/JSON 파일 수정 시 자동 반영
- **가계부 관리**: 모든 거래 자동 기록 및 추적
- **자동 송금**: 일일/주간/월간 자동 송금 (임계값 도달 시)
- **다중 에이전트**: 여러 에이전트 동시 실행

## 지원 에이전트

### Phase 1 (현재 구현됨)

1. **부채 자동 관리 에이전트**
   - 모든 부채 자동 추적
   - 최적 상환 전략 계산
   - 이자 절감 (연간 $2,000-$20,000)
   - 리스크: ⭐⭐⭐⭐⭐ (거의 없음)

2. **쿠폰/할인 에이전트**
   - 온라인 쿠폰 자동 수집
   - 제휴 링크 자동 생성
   - 콘텐츠 자동 생성 및 게시
   - 예상 월 수익: $300-$5,000
   - 리스크: ⭐⭐⭐⭐⭐ (거의 없음)

### Phase 2 (향후 구현)

3. **AI 콘텐츠 생성 에이전트**
4. **데이터 수집 및 판매 에이전트**

### Phase 3 (향후 구현)

5. **다중 자산 자동 투자**
6. **경쟁사 모니터링 에이전트**
7. **자동 드롭쉬핑 에이전트**

## 설치

### 1. 의존성 설치

```bash
cd money_maker_agent
pip install -r requirements.txt
```

### 2. 설정 파일 구성

#### `config/config.yaml` 수정

```yaml
system:
  name: "Money Maker Agent"
  mode: "production"

payout:
  enabled: true
  threshold: 100.0  # 최소 $100부터 송금
  schedule: "daily"  # 일일 송금
  time: "23:00"  # UTC 시간

agents:
  debt_management:
    enabled: true
    priority: 1
  
  coupon_discount:
    enabled: true
    priority: 2
```

#### 계좌 정보 설정

```python
from money_maker_agent.core.account_manager import AccountManager
from pathlib import Path

account_manager = AccountManager(Path("config/accounts.json"))
account_manager.set_payout_account(
    bank_name="Your Bank",
    account_number="1234567890",
    routing_number="123456789",
    account_holder_name="Your Name"
)
```

또는 환경 변수로 암호화 키 설정:

```bash
export MONEY_MAKER_ENCRYPTION_KEY="your_base64_encoded_key"
```

## 사용법

### 기본 실행

```bash
python main.py
```

### 백그라운드 실행 (Linux/Mac)

```bash
nohup python main.py > money_maker.log 2>&1 &
```

### Docker 실행 (선택사항)

```bash
docker build -t money-maker-agent .
docker run -d --name money-maker money-maker-agent
```

## 설정 파일 모니터링

시스템은 `config/config.yaml`과 `config/agents_config.yaml`을 실시간으로 모니터링합니다. 파일을 수정하면 자동으로 리로드되어 변경사항이 즉시 반영됩니다.

### 예시: 에이전트 비활성화

```yaml
agents:
  coupon_discount:
    enabled: false  # 이 줄만 수정하면 즉시 비활성화
```

### 예시: 송금 임계값 변경

```yaml
payout:
  threshold: 500.0  # $500로 변경하면 즉시 적용
```

## 가계부 및 히스토리

모든 거래는 `data/ledger.db` SQLite 데이터베이스에 자동으로 기록됩니다.

### 거래 조회

```python
from money_maker_agent.core.ledger import Ledger
from pathlib import Path

ledger = Ledger(Path("data/ledger.db"))

# 일일 요약
summary = ledger.get_daily_summary()
print(f"오늘 수익: ${summary['total_income']:.2f}")
print(f"오늘 지출: ${summary['total_expenses']:.2f}")
print(f"순이익: ${summary['net_profit']:.2f}")

# 에이전트별 성과
performance = ledger.get_agent_performance("coupon_discount")
print(f"총 수익: ${performance['total_revenue']:.2f}")

# 총 자산
assets = ledger.get_total_assets()
print(f"총 자산: ${assets.get('USD', 0.0):.2f}")
```

## 자동 송금

시스템은 설정된 스케줄에 따라 자동으로 수익을 계좌로 송금합니다.

- **일일 송금**: 매일 지정된 시간에 $100 이상이면 송금
- **주간 송금**: 매주 $500 이상이면 송금
- **월간 송금**: 매월 $1,000 이상이면 송금

송금 히스토리는 `ledger.db`의 `payouts` 테이블에 기록됩니다.

## 모니터링

### 로그 파일

- `money_maker_agent.log`: 모든 로그 기록

### 시스템 상태 확인

```python
status = orchestrator.get_status()
print(f"실행 중인 에이전트: {status['active_agents']}")
print(f"총 자산: ${status['assets'].get('USD', 0.0):.2f}")
```

## 보안

- 계좌 정보는 Fernet 암호화로 저장됩니다
- 암호화 키는 환경 변수로 관리하세요
- API 키는 환경 변수로 설정하세요

```bash
export MONEY_MAKER_ENCRYPTION_KEY="your_key"
export OPENAI_API_KEY="your_key"
export GEMINI_API_KEY="your_key"
```

## 문제 해결

### 설정 파일 오류

```bash
# 설정 파일 검증
python -c "from money_maker_agent.core.config_manager import ConfigManager; cm = ConfigManager('config/config.yaml'); print(cm.validate())"
```

### 데이터베이스 초기화

```bash
# 데이터베이스 삭제 후 재시작하면 자동으로 재생성됩니다
rm data/ledger.db
python main.py
```

### 에이전트가 실행되지 않음

1. 설정 파일에서 `enabled: true` 확인
2. 로그 파일에서 오류 메시지 확인
3. 에이전트별 설정 확인 (`agents_config.yaml`)

## 라이선스

MIT License

## 기여

이슈 및 풀 리퀘스트 환영합니다.

