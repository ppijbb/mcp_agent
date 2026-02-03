# 취미활동 스타터팩 에이전트 (Hobby Starter Pack Agent) 개발 기획서

## 📋 프로젝트 개요

### 프로젝트명
**취미활동 스타터팩 에이전트 (HSP Agent)**

### 프로젝트 목표
개인의 일상에 맞춰 새로운 취미를 제안하고, 시작부터 성장까지 체계적으로 가이드하며, 주간 랭크와 관찰 일지를 통해 동기부여를 제공하는 차세대 AI 기반 멀티 에이전트 시스템 개발

### 핵심 가치 제안
- **개인화**: 사용자의 일상, 관심사, 위치를 고려한 에이전트 기반 맞춤형 취미 추천
- **체계적 성장**: AutoGen 에이전트 합의를 통한 초보자부터 고급자까지 단계별 로드맵 제공
- **게임화**: 주간 랭크와 수준 평가를 통한 지속적 동기부여
- **커뮤니티 연결**: MCP 서버 기반 지역 동호회 및 모임 매칭
- **성장 추적**: AI 에이전트가 생성하는 주간 관찰 일지를 통한 발전 과정 기록

### 기술적 혁신 요소
- **AutoGen + LangGraph 하이브리드 아키텍처**: 에이전트 합의와 워크플로우 관리의 분리
- **A2A 프로토콜 브릿지**: 프레임워크 간 원활한 에이전트 통신
- **MCP 서버 통합**: 표준화된 외부 도구 연결
- **에이전트 기반 의사결정**: 하드코딩 없는 순수 LLM 판단 시스템

---

## 🎯 비즈니스 케이스

### 시장 분석
- **타겟 시장**: 25-45세 직장인, 새로운 취미를 찾는 사람들, 웰빙과 자기계발에 관심 있는 개인
- **시장 규모**: 글로벌 취미 시장 약 4,400억 달러 (2024년 기준)
- **성장률**: 연평균 7.2% 성장 예상

### 경쟁 우위
1. **기존 서비스와의 차별점**
   - 단순 정보 제공 → AI 에이전트 기반 일상 통합형 맞춤 솔루션
   - 정적 콘텐츠 → 동적 에이전트 합의를 통한 성장 추적 및 피드백
   - 개별 활동 → MCP 서버 기반 커뮤니티 연결 및 소셜 기능

2. **독창적 요소**
   - AutoGen + LangGraph 하이브리드 멀티 에이전트 아키텍처
   - A2A 프로토콜을 통한 에이전트 간 통신
   - MCP 서버 기반 실시간 일상 분석 및 취미 통합
   - 게임화된 성장 추적 시스템

### 수익 모델
1. **구독 기반 모델**: 월 $9.99 (기본), $19.99 (프리미엄)
2. **커뮤니티 매칭 수수료**: 모임 참여 시 10% 수수료
3. **제휴 마케팅**: 취미 관련 제품 및 서비스 추천 커미션
4. **기업 B2B**: 직원 웰빙 프로그램 제공

---

## 🏗️ 차세대 시스템 아키텍처

### AutoGen + LangGraph 하이브리드 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                   HSP Agent System v2.0                        │
├─────────────────────────────────────────────────────────────────┤
│  Frontend Layer (Streamlit/React)                              │
├─────────────────────────────────────────────────────────────────┤
│  API Gateway & Authentication                                   │
├─────────────────────────────────────────────────────────────────┤
│  LangGraph Workflow Orchestrator                                │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              AutoGen Agent Consensus Engine                │ │
│  │  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │ │
│  │  │ Profile  │ Hobby    │ Schedule │Community │ Progress │  │ │
│  │  │ Analyst  │Discovery │Integrator│ Matcher  │ Tracker  │  │ │
│  │  └──────────┴──────────┴──────────┴──────────┴──────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  A2A Protocol Bridge Layer                                      │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────┐ │
│  │ Agent       │ Message     │ Session     │ Consensus       │ │
│  │ Registry    │ Router      │ Manager     │ Coordinator     │ │
│  └─────────────┴─────────────┴─────────────┴─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  MCP Server Pool                                                │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐ │
│  │Calendar │ Maps    │Weather  │Social   │E-comm   │Education│ │
│  │   MCP   │  MCP    │  MCP    │  MCP    │  MCP    │   MCP   │ │
│  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘ │
│  ┌─────────┬─────────┬─────────┬─────────────────────────────┐ │
│  │Fitness  │ Music   │Reading  │      Cooking Recipes        │ │
│  │  MCP    │  MCP    │  MCP    │           MCP               │ │
│  └─────────┴─────────┴─────────┴─────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Agent Decision Engine (No Hardcoding)                          │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────┐ │
│  │ LLM-based   │ Dynamic     │ Context     │ Empty Value     │ │
│  │ Decisions   │ Evaluation  │ Analysis    │ Fallback        │ │
│  └─────────────┴─────────────┴─────────────┴─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 에이전트 기반 아키텍처 핵심 원칙

1. **AutoGen 에이전트 합의**
   - 모든 중요한 의사결정은 전문 에이전트들의 합의를 통해 결정
   - ProfileAnalyst, HobbyDiscoverer, ScheduleIntegrator, CommunityMatcher, ProgressTracker
   - DecisionModerator가 에이전트 간 의견 조율

2. **LangGraph 워크플로우 관리**
   - 전체 시스템 흐름을 구조화된 그래프로 관리
   - 조건부 라우팅을 통한 동적 워크플로우 실행
   - 에이전트 합의 결과에 따른 다음 단계 결정

3. **A2A 프로토콜 브릿지**
   - AutoGen과 LangGraph 간 원활한 통신
   - 에이전트 등록, 메시지 라우팅, 세션 관리
   - 프레임워크 간 호환성 보장

4. **MCP 서버 통합**
   - 10개의 전문화된 MCP 서버 연결
   - Google Calendar, Maps, Weather, Social Media 등
   - 표준화된 인터페이스를 통한 외부 도구 접근

---

## 🔧 에이전트별 상세 기능 명세서

### 1. 취미 탐색 및 제안 에이전트 (HobbyDiscoverer Agent)

#### AutoGen 기반 협업 기능
```python
class HobbyDiscovererAgent(AssistantAgent):
    """AutoGen 기반 취미 발견 전문가"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        super().__init__(
            name="HobbyDiscoverer",
            system_message="",  # 에이전트가 동적으로 생성
            llm_config=llm_config,
            description="개인에게 최적화된 새로운 취미 활동을 발견하고 추천"
        )
    
    async def analyze_user_context(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 컨텍스트 분석 - 순수 LLM 기반 판단"""
        # 하드코딩된 가중치 없이 LLM이 모든 것을 결정
        prompt = self._generate_analysis_prompt(user_data)
        result = await self._call_llm(prompt)
        return json.loads(result) if result else {}
```

#### MCP 서버 연동 기능
- **교육 플랫폼 MCP**: 취미 관련 강의 및 튜토리얼 검색
- **전자상거래 MCP**: 취미 용품 및 도구 정보 수집
- **소셜 미디어 MCP**: 취미 관련 커뮤니티 및 그룹 탐색

#### 데이터 입력/출력
- **입력**: 사용자 설문, 활동 이력, 선호도 데이터 (MCP 서버 통해 수집)
- **출력**: 에이전트 합의를 통한 추천 취미 리스트

### 2. 일정 통합 및 최적화 에이전트 (ScheduleIntegrator Agent)

#### AutoGen 합의 메커니즘
```python
async def optimize_schedule_with_consensus(self, schedule_data: Dict, hobby_requirements: Dict) -> Dict:
    """에이전트 합의를 통한 스케줄 최적화"""
    # ProfileAnalyst와 HobbyDiscoverer와 협의
    relevant_agents = ["profile_analyst", "hobby_discoverer", "decision_moderator"]
    consensus_chat = self.create_consensus_chat(relevant_agents)
    
    # 합의 과정 실행
    result = await self._run_consensus(consensus_chat, schedule_data, hobby_requirements)
    return result if result else {}  # 빈 값 폴백
```

#### MCP 서버 활용
- **Google Calendar MCP**: 실시간 일정 동기화 및 관리
- **날씨 MCP**: 야외 취미 활동을 위한 날씨 정보 연동
- **위치 MCP**: 취미 장소까지의 이동 시간 계산

### 3. 커뮤니티 연결 에이전트 (CommunityMatcher Agent)

#### A2A 프로토콜 기반 통신
```python
async def find_community_matches(self, user_profile: Dict, hobby: str) -> List[Dict]:
    """A2A 메시지를 통한 커뮤니티 매칭"""
    # 다른 에이전트들과 A2A 프로토콜로 통신
    message = A2AMessage(
        sender_agent="community_matcher",
        receiver_agent="profile_analyst",
        message_type="community_analysis_request",
        payload={"user_profile": user_profile, "target_hobby": hobby},
        timestamp=datetime.now().isoformat(),
        session_id=self.current_session_id
    )
    
    response = await self.a2a_bridge.send_message(message)
    return response.get("matches", [])  # 빈 값 폴백
```

#### MCP 서버 연동
- **소셜 미디어 MCP**: 지역 커뮤니티 및 모임 검색
- **Google Maps MCP**: 지역 기반 장소 및 커뮤니티 검색
- **이벤트 MCP**: 취미 관련 이벤트 및 모임 정보

### 4. 진행 상황 추적 에이전트 (ProgressTracker Agent)

#### 에이전트 기반 동적 평가
```python
async def generate_weekly_insights(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
    """에이전트 기반 주간 인사이트 생성"""
    # 모든 평가 기준을 LLM이 동적으로 결정
    prompt = f"""
    주간 활동 데이터를 분석하여 개인화된 인사이트를 생성해주세요.
    하드코딩된 가중치나 기준 없이 사용자의 상황에 맞게 판단해주세요.
    
    활동 데이터: {json.dumps(activity_data, ensure_ascii=False)}
    
    다음을 포함하여 분석해주세요:
    1. 성취도 및 진전 상황
    2. 개인별 성장 패턴
    3. 개선이 필요한 영역
    4. 다음 주 맞춤형 목표
    5. 동기부여 메시지
    
    JSON 형태로 응답해주세요. 정보가 부족하면 빈 값을 반환해주세요.
    """
    
    try:
        response = await self._call_llm(prompt)
        return json.loads(response) if response else self._empty_insight_response()
    except Exception:
        return self._empty_insight_response()  # 빈 값 폴백
```

#### MCP 서버 활용
- **피트니스 트래킹 MCP**: 운동 관련 취미 활동 데이터 수집
- **음악 플랫폼 MCP**: 음악 관련 활동 추적
- **독서 플랫폼 MCP**: 독서 진행 상황 모니터링

---

## 🗄️ 데이터베이스 스키마 (업데이트)

### 에이전트 세션 테이블
```sql
CREATE TABLE agent_sessions (
    session_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    agent_type VARCHAR(50) NOT NULL,
    consensus_data JSONB,
    a2a_messages JSONB,
    mcp_responses JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 에이전트 합의 로그 테이블
```sql
CREATE TABLE agent_consensus_logs (
    log_id UUID PRIMARY KEY,
    session_id UUID REFERENCES agent_sessions(session_id),
    decision_point VARCHAR(100) NOT NULL,
    participating_agents JSONB,
    consensus_result JSONB,
    confidence_score DECIMAL(3,2),
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### MCP 서버 상호작용 로그 테이블
```sql
CREATE TABLE mcp_interaction_logs (
    interaction_id UUID PRIMARY KEY,
    session_id UUID REFERENCES agent_sessions(session_id),
    server_name VARCHAR(50) NOT NULL,
    capability_used VARCHAR(100) NOT NULL,
    request_payload JSONB,
    response_data JSONB,
    response_time_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🔌 업데이트된 API 명세서

### 1. 에이전트 합의 API

#### POST /api/agents/consensus
```json
{
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "decision_point": "hobby_discovery",
  "context_data": {
    "user_profile": {...},
    "preferences": {...},
    "constraints": {...}
  },
  "participating_agents": ["hobby_discoverer", "profile_analyst", "decision_moderator"]
}
```

#### Response
```json
{
  "consensus_id": "consensus-456",
  "decision_result": {
    "recommendations": [...],
    "reasoning": "에이전트들의 합의를 통한 추천 근거",
    "confidence_score": 0.92
  },
  "participating_agents": ["hobby_discoverer", "profile_analyst", "decision_moderator"],
  "processing_time_ms": 1500
}
```

### 2. MCP 서버 통합 API

#### POST /api/mcp/call
```json
{
  "server_name": "google_calendar",
  "capability": "find_free_time",
  "parameters": {
    "start_date": "2024-01-15",
    "end_date": "2024-01-21",
    "duration": 60,
    "preferences": ["evening", "weekend"]
  }
}
```

#### Response
```json
{
  "success": true,
  "server_response": {
    "available_slots": [
      {
        "date": "2024-01-16",
        "time_range": "19:00-20:00",
        "confidence": 0.95
      }
    ]
  },
  "response_time_ms": 250
}
```

### 3. A2A 프로토콜 API

#### POST /api/a2a/send-message
```json
{
  "sender_agent": "community_matcher",
  "receiver_agent": "profile_analyst",
  "message_type": "profile_analysis_request",
  "payload": {
    "user_id": "123e4567-e89b-12d3-a456-426614174000",
    "analysis_scope": "community_compatibility"
  }
}
```

---

## 🛠️ 업데이트된 기술 스택

### 핵심 에이전트 프레임워크
- **AutoGen**: 에이전트 간 협업 및 합의 메커니즘
- **LangGraph**: 워크플로우 구조화 및 상태 관리
- **A2A Protocol**: 프레임워크 간 통신 표준

### MCP 서버 생태계
- **Google Calendar MCP**: 일정 관리
- **Google Maps MCP**: 위치 기반 서비스
- **Weather API MCP**: 날씨 정보
- **Social Media MCP**: 커뮤니티 검색
- **E-commerce MCP**: 취미 용품 정보
- **Education MCP**: 강의 및 튜토리얼
- **Fitness Tracker MCP**: 운동 활동 추적
- **Music Platform MCP**: 음악 활동 지원
- **Reading Platform MCP**: 독서 활동 추적
- **Cooking Recipes MCP**: 요리 관련 정보

### 백엔드 & 인프라
- **언어**: Python 3.11+
- **API 프레임워크**: FastAPI
- **데이터베이스**: PostgreSQL 15+ (에이전트 세션 저장)
- **캐시**: Redis (A2A 메시지 큐잉)
- **메시지 큐**: Celery + Redis

---

## 📅 업데이트된 개발 로드맵

### Phase 1: 하이브리드 아키텍처 구축 (3주)
**주차 1**: AutoGen + LangGraph 기본 구조
- AutoGen 에이전트 기본 클래스 구현
- LangGraph 워크플로우 스켈레톤 구축
- A2A 프로토콜 브릿지 프로토타입

**주차 2**: 핵심 MCP 서버 연결
- Google Calendar, Maps MCP 서버 연결
- MCP 서버 매니저 구현
- 기본 에이전트 기반 의사결정 엔진

**주차 3**: 통합 테스트 및 검증
- 에이전트 간 합의 메커니즘 테스트
- MCP 서버 통신 안정성 검증
- A2A 프로토콜 메시지 라우팅 테스트

### Phase 2: 전문 에이전트 완성 (4주)
**주차 4-5**: 핵심 에이전트 구현
- ProfileAnalyst, HobbyDiscoverer 에이전트
- ScheduleIntegrator, CommunityMatcher 에이전트
- 에이전트별 MCP 서버 연동

**주차 6-7**: 고급 기능 개발
- ProgressTracker 에이전트
- 에이전트 간 복잡한 합의 시나리오
- 실시간 학습 및 적응 메커니즘

### Phase 3: 고도화 및 배포 (3주)
**주차 8-9**: 시스템 최적화
- 에이전트 성능 튜닝
- MCP 서버 풀 확장 (날씨, 소셜미디어, 교육 등)
- A2A 프로토콜 고도화

**주차 10**: 배포 및 모니터링
- 프로덕션 환경 배포
- 에이전트 활동 모니터링 시스템
- 사용자 피드백 수집 메커니즘

---

## 💰 업데이트된 예산 및 리소스

### 개발 인력 (10주 기준)
- **시니어 AI 에이전트 엔지니어** (1명): $80,000
- **AutoGen/LangGraph 전문가** (1명): $85,000
- **MCP 서버 개발자** (1명): $70,000
- **A2A 프로토콜 엔지니어** (0.5명): $35,000
- **백엔드 개발자** (1명): $60,000
- **DevOps 엔지니어** (0.5명): $30,000

**총 인건비**: $360,000

### 기술 및 인프라 비용
- **AutoGen/LangGraph 라이센스**: $5,000
- **MCP 서버 개발 및 호스팅**: $15,000
- **클라우드 서비스 (AWS/Azure)**: $12,000
- **LLM API 사용료**: $20,000
- **기타 도구 및 서비스**: $8,000

**총 기술비**: $60,000

### 총 예산: $420,000

---

## 🔍 업데이트된 성공 지표 (KPI)

### 에이전트 성능 지표
- **에이전트 합의 성공률**: 95% 이상
- **의사결정 평균 시간**: 3초 이내
- **MCP 서버 응답률**: 99% 이상
- **A2A 메시지 전송 성공률**: 99.9% 이상

### 사용자 경험 지표
- **추천 정확도**: 85% 이상 (사용자 만족도 기준)
- **취미 시작 성공률**: 80% 이상
- **커뮤니티 매칭 성공률**: 70% 이상
- **주간 활동 지속률**: 75% 이상

### 비즈니스 지표
- **MAU (Monthly Active Users)**: 10,000명 (1년 목표)
- **에이전트 기반 개인화 만족도**: 4.5/5.0
- **구독 전환율**: 20% (기존 15%에서 향상)
- **사용자 생애 가치 (LTV)**: $200 (기존 $150에서 향상)

---

## 🚀 차세대 마케팅 전략

### AI 에이전트 중심 마케팅
1. **에이전트 개인화 체험**: "당신만을 위한 AI 취미 전문가"
2. **멀티 에이전트 협업 시연**: 실시간 에이전트 합의 과정 공개
3. **MCP 생태계 홍보**: 연결된 10개 서비스의 통합 경험

### 기술 커뮤니티 참여
1. **오픈소스 기여**: A2A 프로토콜 브릿지 공개
2. **기술 컨퍼런스**: AutoGen + LangGraph 사례 발표
3. **개발자 커뮤니티**: MCP 서버 개발 가이드 제공

---

## ⚠️ 리스크 관리

### 기술적 위험
- **에이전트 합의 실패**: 다중 백업 에이전트 및 타임아웃 처리
- **MCP 서버 장애**: 서버별 폴백 메커니즘 및 헬스체크
- **A2A 프로토콜 호환성**: 표준 준수 및 버전 관리

### 운영 위험
- **LLM API 제한**: 다중 LLM 제공업체 연동
- **데이터 보안**: 에이전트 세션 암호화 및 접근 제어
- **확장성**: 마이크로서비스 아키텍처 및 컨테이너화

---

## 📞 다음 단계

1. **기술 검증**: AutoGen + LangGraph 통합 프로토타입 개발 (1주)
2. **MCP 서버 설계**: 우선순위 높은 5개 서버 설계 (1주)
3. **A2A 프로토콜 명세**: 상세 통신 프로토콜 문서화 (3일)
4. **에이전트 팀 구성**: AI 에이전트 전문 개발진 채용 (2주)
5. **개발 환경 구축**: 통합 개발 환경 및 CI/CD 파이프라인 (1주)

---

**문서 작성일**: 2024년 12월 19일  
**작성자**: HSP Agent 개발팀  
**버전**: 2.0.0 (AutoGen + LangGraph + A2A + MCP)  
**다음 리뷰**: 2024년 12월 26일 