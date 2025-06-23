# 취미활동 스타터팩 에이전트 (Hobby Starter Pack Agent) 개발 기획서

## 📋 프로젝트 개요

### 프로젝트명
**취미활동 스타터팩 에이전트 (HSP Agent)**

### 프로젝트 목표
개인의 일상에 맞춰 새로운 취미를 제안하고, 시작부터 성장까지 체계적으로 가이드하며, 주간 랭크와 관찰 일지를 통해 동기부여를 제공하는 AI 기반 멀티 에이전트 시스템 개발

### 핵심 가치 제안
- **개인화**: 사용자의 일상, 관심사, 위치를 고려한 맞춤형 취미 추천
- **체계적 성장**: 초보자부터 고급자까지 단계별 로드맵 제공
- **게임화**: 주간 랭크와 수준 평가를 통한 지속적 동기부여
- **커뮤니티 연결**: 지역 기반 동호회 및 모임 매칭
- **성장 추적**: 주간 관찰 일지를 통한 발전 과정 기록

---

## 🎯 비즈니스 케이스

### 시장 분석
- **타겟 시장**: 25-45세 직장인, 새로운 취미를 찾는 사람들, 웰빙과 자기계발에 관심 있는 개인
- **시장 규모**: 글로벌 취미 시장 약 4,400억 달러 (2024년 기준)
- **성장률**: 연평균 7.2% 성장 예상

### 경쟁 우위
1. **기존 서비스와의 차별점**
   - 단순 정보 제공 → 일상 통합형 맞춤 솔루션
   - 정적 콘텐츠 → 동적 성장 추적 및 피드백
   - 개별 활동 → 커뮤니티 연결 및 소셜 기능

2. **독창적 요소**
   - LangGraph 기반 멀티 에이전트 아키텍처
   - 실시간 일상 분석 및 취미 통합
   - 게임화된 성장 추적 시스템

### 수익 모델
1. **구독 기반 모델**: 월 $9.99 (기본), $19.99 (프리미엄)
2. **커뮤니티 매칭 수수료**: 모임 참여 시 10% 수수료
3. **제휴 마케팅**: 취미 관련 제품 및 서비스 추천 커미션
4. **기업 B2B**: 직원 웰빙 프로그램 제공

---

## 🏗️ 시스템 아키텍처

### 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                     HSP Agent System                            │
├─────────────────────────────────────────────────────────────────┤
│  Frontend Layer (Streamlit/React)                              │
├─────────────────────────────────────────────────────────────────┤
│  API Gateway & Authentication                                   │
├─────────────────────────────────────────────────────────────────┤
│  LangGraph Multi-Agent Orchestrator                            │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────┐ │
│  │ Hobby       │ Schedule    │ Community   │ Progress        │ │
│  │ Discovery   │ Integration │ Connector   │ Tracker         │ │
│  │ Agent       │ Agent       │ Agent       │ Agent           │ │
│  └─────────────┴─────────────┴─────────────┴─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Data Processing Layer                                          │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────┐ │
│  │ User        │ Hobby       │ Location    │ Activity        │ │
│  │ Analytics   │ Database    │ Services    │ Logger          │ │
│  └─────────────┴─────────────┴─────────────┴─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  External Integrations                                          │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────┐ │
│  │ Calendar    │ Location    │ Social      │ Payment         │ │
│  │ APIs        │ APIs        │ Networks    │ Gateway         │ │
│  └─────────────┴─────────────┴─────────────┴─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### LangGraph 에이전트 아키텍처

```python
# 에이전트 상태 정의
class HSPAgentState(TypedDict):
    user_id: str
    user_profile: Dict[str, Any]
    hobby_preferences: List[str]
    current_hobbies: List[Dict]
    schedule_data: Dict[str, Any]
    location_data: Dict[str, Any]
    community_matches: List[Dict]
    weekly_progress: Dict[str, Any]
    recommendations: List[Dict]
    current_step: str
    session_data: Dict[str, Any]
```

---

## 🔧 세부 기능 명세서

### 1. 취미 탐색 및 제안 에이전트 (Hobby Discovery Agent)

#### 기능 개요
사용자의 관심사, 성격, 라이프스타일을 분석하여 맞춤형 취미를 제안

#### 상세 기능
1. **사용자 프로파일링**
   - 초기 설문 조사 (관심 분야, 성격 유형, 선호 활동 스타일)
   - 소셜 미디어 활동 분석 (선택적)
   - 기존 활동 이력 분석

2. **취미 추천 알고리즘**
   ```python
   def recommend_hobbies(user_profile: Dict, constraints: Dict) -> List[HobbyRecommendation]:
       """
       취미 추천 알고리즘
       
       Args:
           user_profile: 사용자 프로필 (관심사, 성격, 선호도)
           constraints: 제약 조건 (시간, 예산, 위치)
       
       Returns:
           추천 취미 리스트 (우선순위, 적합도 점수 포함)
       """
       # 1. 성격 유형별 취미 매칭
       personality_match = match_personality_to_hobbies(user_profile.personality)
       
       # 2. 관심사 기반 필터링
       interest_filter = filter_by_interests(user_profile.interests)
       
       # 3. 제약 조건 적용
       feasible_hobbies = apply_constraints(constraints)
       
       # 4. 유사 사용자 기반 추천
       collaborative_filter = get_similar_user_hobbies(user_profile)
       
       # 5. 최종 점수 계산 및 랭킹
       return calculate_final_scores_and_rank(
           personality_match, interest_filter, 
           feasible_hobbies, collaborative_filter
       )
   ```

3. **취미 정보 제공**
   - 취미별 상세 설명
   - 시작 비용 및 필요 장비
   - 예상 학습 기간
   - 난이도 및 신체적 요구사항

#### 데이터 입력/출력
- **입력**: 사용자 설문, 활동 이력, 선호도 데이터
- **출력**: 추천 취미 리스트 (적합도 점수, 시작 가이드 포함)

### 2. 일정 통합 및 최적화 에이전트 (Schedule Integration Agent)

#### 기능 개요
사용자의 일상 스케줄을 분석하여 취미 활동 시간을 최적화하고 통합

#### 상세 기능
1. **스케줄 분석**
   ```python
   def analyze_user_schedule(calendar_data: Dict, lifestyle_data: Dict) -> ScheduleAnalysis:
       """
       사용자 스케줄 분석
       
       Args:
           calendar_data: 캘린더 데이터
           lifestyle_data: 생활 패턴 데이터
       
       Returns:
           스케줄 분석 결과 (가용 시간, 패턴, 추천 시간대)
       """
       # 1. 고정 일정 추출
       fixed_schedule = extract_fixed_commitments(calendar_data)
       
       # 2. 자유 시간 계산
       free_time_slots = calculate_free_time(fixed_schedule)
       
       # 3. 활동 패턴 분석
       activity_patterns = analyze_activity_patterns(lifestyle_data)
       
       # 4. 최적 취미 시간 추천
       optimal_hobby_times = recommend_hobby_times(
           free_time_slots, activity_patterns
       )
       
       return ScheduleAnalysis(
           free_time=free_time_slots,
           patterns=activity_patterns,
           recommendations=optimal_hobby_times
       )
   ```

2. **스마트 스케줄링**
   - 취미별 최적 시간대 제안
   - 단기/장기 목표 설정
   - 유연한 일정 조정 알고리즘

3. **효율성 최적화**
   - 이동 시간 최소화
   - 연관 활동 그룹화
   - 에너지 레벨 고려 스케줄링

#### 데이터 입력/출력
- **입력**: 캘린더 데이터, 위치 정보, 생활 패턴
- **출력**: 최적화된 취미 스케줄, 시간 효율성 팁

### 3. 커뮤니티 연결 에이전트 (Community Connector Agent)

#### 기능 개요
지역 기반 취미 커뮤니티와 모임을 매칭하고 연결

#### 상세 기능
1. **지역 커뮤니티 발견**
   ```python
   def find_local_communities(user_location: Location, hobby: str, preferences: Dict) -> List[Community]:
       """
       지역 커뮤니티 검색
       
       Args:
           user_location: 사용자 위치
           hobby: 관심 취미
           preferences: 모임 선호도 (시간대, 수준, 연령대 등)
       
       Returns:
           매칭된 커뮤니티 리스트
       """
       # 1. 반경 내 커뮤니티 검색
       nearby_communities = search_communities_by_radius(user_location, hobby)
       
       # 2. 선호도 기반 필터링
       filtered_communities = filter_by_preferences(nearby_communities, preferences)
       
       # 3. 활성도 및 평점 고려
       ranked_communities = rank_by_activity_and_rating(filtered_communities)
       
       # 4. 매칭 점수 계산
       return calculate_community_match_scores(ranked_communities, preferences)
   ```

2. **소셜 매칭**
   - 유사한 수준의 사용자 매칭
   - 성격 및 관심사 기반 친구 추천
   - 멘토-멘티 매칭 시스템

3. **모임 관리**
   - 모임 일정 통합
   - 참여 확인 및 리마인더
   - 모임 후 피드백 수집

#### 데이터 입력/출력
- **입력**: 위치 데이터, 취미 정보, 사용자 선호도
- **출력**: 매칭된 커뮤니티/모임 리스트, 참여 가이드

### 4. 진행 상황 추적 에이전트 (Progress Tracker Agent)

#### 기능 개요
주간 활동을 추적하고 랭크를 부여하며 관찰 일지를 생성

#### 상세 기능
1. **활동 추적 시스템**
   ```python
   def track_weekly_progress(user_id: str, hobby: str, activities: List[Activity]) -> WeeklyProgress:
       """
       주간 진행 상황 추적
       
       Args:
           user_id: 사용자 ID
           hobby: 취미 종류
           activities: 주간 활동 리스트
       
       Returns:
           주간 진행 상황 보고서
       """
       # 1. 목표 대비 달성률 계산
       goal_achievement = calculate_goal_achievement(activities)
       
       # 2. 랭크 점수 계산
       rank_score = calculate_rank_score(goal_achievement, activities)
       
       # 3. 수준 평가
       skill_level = evaluate_skill_level(activities, hobby)
       
       # 4. 성장 지표 분석
       growth_metrics = analyze_growth_metrics(activities)
       
       return WeeklyProgress(
           achievement_rate=goal_achievement,
           rank_score=rank_score,
           skill_level=skill_level,
           growth_metrics=growth_metrics
       )
   ```

2. **랭킹 시스템**
   - 목표 달성률 기반 점수 계산
   - 전주 대비 향상도 평가
   - 커뮤니티 내 상대적 순위

3. **수준 평가 시스템**
   ```python
   SKILL_LEVELS = {
       "초심자": {"range": (0, 100), "description": "기본기 습득 단계"},
       "초급자": {"range": (101, 300), "description": "기초 실력 형성 단계"},
       "중급자": {"range": (301, 700), "description": "안정적 실력 단계"},
       "중상급자": {"range": (701, 1200), "description": "숙련된 실력 단계"},
       "고급자": {"range": (1201, 2000), "description": "전문가 수준 단계"},
       "마스터": {"range": (2001, 3000), "description": "전문가 수준"},
       "그랜드마스터": {"range": (3001, float('inf')), "description": "최고 수준"}
   }
   ```

4. **관찰 일지 생성**
   ```python
   def generate_weekly_journal(progress_data: WeeklyProgress, activities: List[Activity]) -> str:
       """
       주간 관찰 일지 자동 생성
       
       Args:
           progress_data: 주간 진행 데이터
           activities: 활동 기록
       
       Returns:
           자연어로 작성된 관찰 일지
       """
       journal_template = """
       ## 📅 {date_range} 주간 취미 활동 관찰 일지
       
       ### 🎯 이번 주 성과
       - 목표 달성률: {achievement_rate}%
       - 랭크 변화: {rank_change}
       - 현재 수준: {current_level}
       
       ### 📊 활동 요약
       {activity_summary}
       
       ### 🔍 주요 관찰 사항
       {key_observations}
       
       ### 📈 성장 지표
       {growth_indicators}
       
       ### 💡 다음 주 개선 포인트
       {improvement_suggestions}
       """
       
       return journal_template.format(**generate_journal_content(progress_data, activities))
   ```

#### 데이터 입력/출력
- **입력**: 활동 로그, 목표 설정, 시간 기록
- **출력**: 주간 랭크, 수준 평가, 관찰 일지

---

## 🗄️ 데이터베이스 스키마

### 사용자 테이블
```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    profile_data JSONB,
    preferences JSONB,
    location_data JSONB
);
```

### 취미 테이블
```sql
CREATE TABLE hobbies (
    hobby_id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL,
    difficulty_level INTEGER CHECK (difficulty_level BETWEEN 1 AND 10),
    equipment_needed JSONB,
    description TEXT,
    skill_progression JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 사용자 취미 테이블
```sql
CREATE TABLE user_hobbies (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    hobby_id UUID REFERENCES hobbies(hobby_id),
    start_date DATE NOT NULL,
    current_level INTEGER DEFAULT 1,
    total_hours DECIMAL(10,2) DEFAULT 0,
    goals JSONB,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 활동 로그 테이블
```sql
CREATE TABLE activity_logs (
    log_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    hobby_id UUID REFERENCES hobbies(hobby_id),
    activity_date DATE NOT NULL,
    duration_minutes INTEGER NOT NULL,
    activity_type VARCHAR(100),
    notes TEXT,
    satisfaction_score INTEGER CHECK (satisfaction_score BETWEEN 1 AND 10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 주간 진행 상황 테이블
```sql
CREATE TABLE weekly_progress (
    progress_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    hobby_id UUID REFERENCES hobbies(hobby_id),
    week_start_date DATE NOT NULL,
    week_end_date DATE NOT NULL,
    goal_hours DECIMAL(5,2),
    actual_hours DECIMAL(5,2),
    achievement_rate DECIMAL(5,2),
    rank_score INTEGER,
    level_progression JSONB,
    journal_content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 커뮤니티 테이블
```sql
CREATE TABLE communities (
    community_id UUID PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    hobby_id UUID REFERENCES hobbies(hobby_id),
    location JSONB,
    member_count INTEGER DEFAULT 0,
    activity_level VARCHAR(20),
    contact_info JSONB,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🔌 API 명세서

### 1. 사용자 관리 API

#### POST /api/users/register
```json
{
  "email": "user@example.com",
  "name": "홍길동",
  "profile": {
    "age": 30,
    "occupation": "developer",
    "interests": ["technology", "fitness", "music"]
  }
}
```

#### GET /api/users/{user_id}/profile
```json
{
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "email": "user@example.com",
  "name": "홍길동",
  "profile": {...},
  "current_hobbies": [...]
}
```

### 2. 취미 추천 API

#### POST /api/recommendations/hobbies
```json
{
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "constraints": {
    "max_budget": 100000,
    "available_time_per_week": 5,
    "location_radius": 10
  }
}
```

#### Response
```json
{
  "recommendations": [
    {
      "hobby_id": "hobby-123",
      "name": "기타 연주",
      "match_score": 0.92,
      "reasons": ["음악에 대한 관심", "창의적 성향"],
      "estimated_cost": 80000,
      "time_investment": "주 3시간",
      "difficulty": 3
    }
  ]
}
```

### 3. 스케줄 통합 API

#### POST /api/schedule/integrate
```json
{
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "hobby_id": "hobby-123",
  "calendar_data": {...},
  "preferences": {
    "preferred_time": "evening",
    "session_duration": 60
  }
}
```

### 4. 진행 상황 추적 API

#### POST /api/progress/log-activity
```json
{
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "hobby_id": "hobby-123",
  "duration_minutes": 90,
  "activity_type": "practice",
  "notes": "코드 연습 및 새로운 곡 연주",
  "satisfaction_score": 8
}
```

#### GET /api/progress/weekly/{user_id}/{hobby_id}
```json
{
  "week_start": "2024-01-15",
  "week_end": "2024-01-21",
  "goal_achievement": 85.5,
  "rank_score": 1250,
  "current_level": "중급자",
  "level_progress": 65,
  "journal": "이번 주는 새로운 코드를 배우며...",
  "growth_metrics": {
    "practice_time": "+15%",
    "skill_improvement": "+8%"
  }
}
```

---

## 🛠️ 개발 환경 및 기술 스택

### 백엔드
- **언어**: Python 3.11+
- **프레임워크**: FastAPI
- **LangGraph**: 멀티 에이전트 오케스트레이션
- **데이터베이스**: PostgreSQL 15+
- **캐시**: Redis
- **메시지 큐**: Celery + Redis

### 프론트엔드
- **프레임워크**: Streamlit (MVP), React (Production)
- **상태 관리**: Redux Toolkit
- **UI 라이브러리**: Material-UI
- **차트**: Chart.js, D3.js

### AI/ML
- **LLM**: OpenAI GPT-4, Claude 3.5
- **임베딩**: OpenAI Embeddings
- **벡터 데이터베이스**: Chroma
- **추천 시스템**: Collaborative Filtering + Content-based

### 인프라
- **클라우드**: AWS / Azure
- **컨테이너**: Docker + Kubernetes
- **CI/CD**: GitHub Actions
- **모니터링**: Prometheus + Grafana

### 외부 통합
- **캘린더**: Google Calendar API, Outlook API
- **위치**: Google Maps API, Foursquare API
- **소셜**: Facebook Graph API, Meetup API
- **결제**: Stripe API

---

## 📅 개발 로드맵

### Phase 1: MVP 개발 (8주)
**주차 1-2**: 프로젝트 셋업 및 기본 아키텍처
- LangGraph 기반 에이전트 구조 설계
- 데이터베이스 스키마 구현
- 기본 API 구조 개발

**주차 3-4**: 핵심 에이전트 개발
- 취미 추천 에이전트 구현
- 스케줄 통합 에이전트 구현
- 기본 사용자 인터페이스 개발

**주차 5-6**: 진행 상황 추적 시스템
- 활동 로깅 시스템 구현
- 랭킹 및 수준 평가 알고리즘
- 관찰 일지 생성 기능

**주차 7-8**: 통합 테스트 및 MVP 완성
- 에이전트 간 워크플로우 통합
- 기본 UI/UX 완성
- 초기 사용자 테스트

### Phase 2: 커뮤니티 기능 추가 (6주)
**주차 9-10**: 커뮤니티 연결 에이전트
- 지역 커뮤니티 검색 기능
- 매칭 알고리즘 구현

**주차 11-12**: 소셜 기능 확장
- 사용자 간 매칭 시스템
- 모임 관리 기능

**주차 13-14**: 고급 기능 개발
- 멘토-멘티 시스템
- 그룹 활동 스케줄링

### Phase 3: 고도화 및 확장 (8주)
**주차 15-16**: AI 고도화
- 추천 알고리즘 개선
- 개인화 모델 정교화

**주차 17-18**: 모바일 앱 개발
- React Native 앱 개발
- 푸시 알림 시스템

**주차 19-20**: 결제 및 비즈니스 로직
- 구독 모델 구현
- 제휴 마케팅 시스템

**주차 21-22**: 최종 테스트 및 배포
- 부하 테스트
- 보안 감사
- 프로덕션 배포

---

## 💰 예산 및 리소스

### 개발 인력 (22주 기준)
- **시니어 백엔드 개발자** (1명): $120,000
- **AI/ML 엔지니어** (1명): $130,000
- **프론트엔드 개발자** (1명): $100,000
- **DevOps 엔지니어** (0.5명): $60,000
- **UI/UX 디자이너** (0.5명): $40,000
- **프로젝트 매니저** (0.3명): $30,000

**총 인건비**: $480,000

### 인프라 및 서비스 비용 (1년)
- **클라우드 서비스**: $24,000
- **외부 API 비용**: $18,000
- **AI 모델 사용 비용**: $15,000
- **기타 SaaS 도구**: $12,000

**총 운영비**: $69,000

### 총 예산
- **개발비**: $480,000
- **운영비**: $69,000
- **마케팅 비용**: $100,000
- **예비비**: $50,000

**총 예산**: $699,000

---

## 🔍 성공 지표 (KPI)

### 사용자 지표
- **MAU (Monthly Active Users)**: 10,000명 (1년 목표)
- **사용자 유지율**: 70% (3개월 기준)
- **일일 활성 사용자**: 2,000명

### 비즈니스 지표
- **구독 전환율**: 15%
- **월 매출**: $50,000 (1년 목표)
- **고객 생애 가치 (LTV)**: $150

### 제품 지표
- **취미 시작 성공률**: 80%
- **주간 활동 로그 작성률**: 60%
- **커뮤니티 참여율**: 40%
- **사용자 만족도**: 4.5/5.0

---

## 🚀 마케팅 전략

### 런칭 전략
1. **베타 테스터 모집**: 취미 커뮤니티 대상 100명
2. **인플루언서 협업**: 취미 관련 유튜버, 블로거
3. **콘텐츠 마케팅**: 취미 시작 가이드, 성공 사례

### 성장 전략
1. **바이럴 기능**: 친구 초대 리워드
2. **파트너십**: 취미 용품 업체, 교육 기관
3. **커뮤니티 마케팅**: 오프라인 모임 스폰서십

---

## ⚠️ 위험 요소 및 대응 방안

### 기술적 위험
- **AI 모델 성능 이슈**: 다중 모델 백업, 지속적 튜닝
- **확장성 문제**: 마이크로서비스 아키텍처, 수평 확장

### 비즈니스 위험
- **사용자 확보 어려움**: 프리미엄 기능 무료 체험, 바이럴 마케팅
- **경쟁사 출현**: 핵심 기능 차별화, 특허 출원

### 운영 위험
- **데이터 보안**: 암호화, 접근 제어, 정기 감사
- **법적 컴플라이언스**: GDPR, 개인정보보호법 준수

---

## 📞 다음 단계

1. **스테이크홀더 승인**: 기획서 검토 및 승인
2. **팀 구성**: 핵심 개발진 채용
3. **개발 환경 구축**: AWS/Azure 계정, CI/CD 파이프라인
4. **프로토타입 개발**: 2주 내 기본 기능 프로토타입
5. **사용자 인터뷰**: 타겟 사용자 10명 인터뷰

---

**문서 작성일**: 2024년 12월 19일  
**작성자**: HSP Agent 개발팀  
**버전**: 1.0  
**다음 리뷰**: 2024년 12월 26일 