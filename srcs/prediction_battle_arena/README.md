# 🎮 Prediction Battle Arena

**실시간 예측 배틀, 가상 화폐 베팅, 글로벌 리더보드를 통한 강렬한 도파민을 제공하는 에이전트 시스템**

## 📋 개요

Prediction Battle Arena는 mcp_agent 라이브러리를 활용하여 구현된 실시간 예측 경쟁 플랫폼입니다. 사용자들이 예측을 생성하고, 베팅을 하며, 실시간으로 경쟁하는 과정에서 강렬한 도파민을 경험할 수 있습니다.

## ✨ 주요 기능

### 1. ⚡ 실시간 배틀
- **5분/15분/30분 배틀**: 빠른 배틀부터 장기 배틀까지
- **실시간 순위 변동**: WebSocket을 통한 실시간 업데이트
- **다중 참가자**: 최대 100명 동시 참가

### 2. 💰 가상 화폐 베팅 시스템
- **베팅 배율**: 1.5x ~ 10x
- **연승 보너스**: 3연승(20%), 5연승(50%), 10연승(100%), 20연승(200%)
- **랜덤 보너스**: 10% 확률로 100x 잭팟
- **일일 상한선**: 중독 방지

### 3. 🏆 글로벌 리더보드
- **실시간 순위**: 전 세계 사용자와 경쟁
- **다중 카테고리**: 전체/주간/월간 리더보드
- **순위 변동 알림**: "You're #1!" 실시간 알림

### 4. 🎰 슬롯 머신 스타일 보상
- **랜덤 보상**: 분석 완료 시 랜덤 보상
- **희귀도 시스템**: 일반(70%) / 희귀(25%) / 전설(4%) / 신화(1%)
- **보상 종류**: 가상 화폐, 배지, 특별 스킨, 분석 부스트

### 5. 📊 실시간 스트리밍
- **라이브 관람**: 다른 사용자의 분석 실시간 관람
- **좋아요/응원**: 소셜 인터랙션
- **트렌딩 인사이트**: 인기 예측 피드

### 6. 🎮 미션 & 퀘스트
- **일일 미션**: "5번 예측하기" (보상: 100 코인)
- **주간 챌린지**: "10연승 달성" (보상: 1,000 코인)
- **특별 이벤트**: "월드컵 우승 예측" (보상: 10,000 코인)

### 7. 🎨 수집 요소
- **인사이트 카드**: 예측 성공 시 획득
- **희귀 카드 조합**: 특별 보상
- **컬렉션 완성도**: 진행률 추적

### 8. 💬 소셜 피드
- **인기 인사이트**: 트렌딩 예측
- **좋아요/공유**: 소셜 인터랙션
- **실시간 피드**: 최신 활동 업데이트

## 🏗️ 아키텍처

### 시스템 구조

```
PredictionBattleAgent (메인 에이전트)
├── PredictionAgent (예측 생성)
├── BattleManagerAgent (배틀 관리)
├── RewardCalculatorAgent (보상 계산)
├── LeaderboardAgent (리더보드 관리)
└── SocialFeedAgent (소셜 피드)

서비스 레이어
├── RealtimeService (WebSocket 실시간 통신)
├── RedisService (리더보드 및 상태 관리)
└── RewardService (보상 계산 로직)

도구 레이어
├── PredictionTools (예측 생성/평가)
├── BettingTools (베팅 처리)
├── RewardTools (보상 계산)
└── LeaderboardTools (리더보드 관리)
```

### 데이터 모델

- **Battle**: 배틀 정보 (상태, 참가자, 예측, 베팅)
- **Prediction**: 예측 정보 (내용, 정확도, 결과)
- **User**: 사용자 정보 (코인, 통계, 레벨)

## 🚀 설치 및 실행

### 필수 요구사항

```bash
# Python 3.10+
pip install mcp-agent
pip install redis  # 선택사항 (없으면 메모리 기반 저장소 사용)
```

### 환경 변수 설정

```bash
# Redis 설정 (선택사항)
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0

# WebSocket 설정
export WEBSOCKET_PORT=8765

# 배틀 설정
export BATTLE_DURATION_QUICK=300
export BATTLE_DURATION_STANDARD=900
export BATTLE_DURATION_EXTENDED=1800
export MIN_PARTICIPANTS=2
export MAX_PARTICIPANTS=100
```

### 실행 예제

```python
from srcs.prediction_battle_arena import PredictionBattleAgent

# 에이전트 생성
agent = PredictionBattleAgent()

# 배틀 참가
result = await agent.run(
    user_id="user_123",
    battle_type="quick",  # quick/standard/extended
    prediction_topic="비트코인 가격 예측",
    action="join"  # create/join
)

print(result)
```

## 📊 워크플로우

1. **배틀 생성/참가**: 사용자가 배틀에 참가
2. **예측 생성**: MCP 도구(g-search, fetch)를 활용하여 예측 생성
3. **베팅 처리**: 가상 화폐로 베팅
4. **실시간 업데이트**: WebSocket을 통한 실시간 순위 변동
5. **결과 계산**: 정확도 기반 보상 계산
6. **리더보드 업데이트**: Redis를 통한 글로벌 리더보드 갱신
7. **소셜 피드 생성**: 인기 예측 및 성과 공유

## 🎯 구조적 상업성

### 수익 모델

1. **프리미엄 구독**: 월 $9.99
   - 더 많은 베팅
   - 특별 이벤트 참가
   - 광고 제거

2. **가상 화폐 판매**: $4.99 = 1,000 코인
   - 추가 베팅 자금
   - 특별 아이템 구매

3. **광고**: 무료 사용자에게 광고 표시
   - 배너 광고
   - 동영상 광고 (보상 제공)

4. **B2B 확장**: 기업용 예측 대시보드
   - 월 $99
   - 팀 리더보드
   - 맞춤형 분석

## 🔧 기술 스택

- **mcp_agent**: 멀티 에이전트 오케스트레이션
- **LangChain**: LLM 통합
- **WebSocket**: 실시간 통신
- **Redis**: 리더보드 및 상태 관리 (선택사항)
- **Gemini 2.5 Flash**: LLM 모델

## 📝 주요 파일

```
srcs/prediction_battle_arena/
├── prediction_battle_agent.py    # 메인 에이전트
├── models/                        # 데이터 모델
│   ├── battle.py
│   ├── prediction.py
│   └── user.py
├── tools/                         # MCP 도구
│   ├── prediction_tools.py
│   ├── betting_tools.py
│   ├── reward_tools.py
│   └── leaderboard_tools.py
├── services/                      # 서비스 레이어
│   ├── realtime_service.py
│   ├── redis_service.py
│   └── reward_service.py
└── agents/                        # 특화 에이전트
    ├── prediction_agent.py
    ├── battle_manager_agent.py
    ├── reward_calculator_agent.py
    ├── leaderboard_agent.py
    └── social_feed_agent.py
```

## 🎮 사용 예제

### 배틀 생성 및 참가

```python
# 배틀 생성
result = await agent.run(
    user_id="user_123",
    battle_type="quick",
    prediction_topic="AI 기술 트렌드 예측",
    action="create"
)

# 배틀 참가
result = await agent.run(
    user_id="user_456",
    battle_type="quick",
    action="join"
)
```

### 예측 생성

```python
# 예측 도구 직접 사용
from srcs.prediction_battle_arena.tools import PredictionTools

tools = PredictionTools()
prediction = tools.get_tool_by_name("prediction_create")

result = await prediction.ainvoke({
    "user_id": "user_123",
    "battle_id": "battle_456",
    "topic": "비트코인 가격 예측",
    "context": "최근 시장 동향"
})
```

## 🐛 문제 해결

### Redis 연결 실패
- Redis가 없는 경우 자동으로 메모리 기반 저장소로 전환
- 프로덕션 환경에서는 Redis 사용 권장

### WebSocket 연결 실패
- 포트 충돌 확인
- 방화벽 설정 확인

## 📈 향후 계획

- [ ] WebSocket 서버 구현
- [ ] Streamlit UI 대시보드
- [ ] 모바일 앱 연동
- [ ] NFT 인사이트 카드
- [ ] VR/AR 통합

## 📄 라이선스

이 프로젝트는 mcp_agent 라이브러리를 기반으로 구현되었습니다.

---

**강렬하고 짜릿한 도파민을 경험하세요! 🎉**

