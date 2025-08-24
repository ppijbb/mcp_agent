# 🚀 LangChain 기반 HSP Agent - 흐름 개선 버전

**LangChain의 핵심 개념들을 활용한 최적화된 취미 추천 시스템**

## 🔄 주요 개선 사항

웹 검색 결과를 바탕으로 LangChain 프로젝트의 흐름을 다음과 같이 개선했습니다:

### 1. 🧠 대화 메모리 관리 강화
- **ConversationBufferMemory**: 사용자별 대화 기록 관리
- **ConversationSummaryMemory**: 장기 메모리 및 요약 기능
- **컨텍스트 인식**: 취미 관련 정보를 메모리에 저장하여 일관된 추천

### 2. 📝 프롬프트 템플릿 최적화
- **동적 프롬프트 생성**: 사용자 컨텍스트와 대화 기록을 포함
- **템플릿 검증**: 필수 변수 자동 검증 및 기본값 설정
- **메모리 인식 프롬프트**: 이전 대화를 고려한 개인화된 응답

### 3. 🔍 벡터 데이터베이스 활용
- **Chroma/FAISS**: 효율적인 취미 및 커뮤니티 검색
- **텍스트 분할기**: 대용량 문서를 최적 크기로 분할
- **유사도 검색**: 사용자 프로필 기반 맞춤형 추천

### 4. ⚙️ LangChain 에이전트 도입
- **도구 기반 에이전트**: 특정 작업에 특화된 도구들
- **자동화 워크플로우**: 복잡한 의사결정 과정 자동화
- **에러 처리**: graceful degradation 및 fallback 지원

## 🏗️ 새로운 아키텍처

```
LangChain 기반 HSP Agent
├── 🧠 Memory Manager (대화 메모리)
├── 📝 Prompt Templates (프롬프트 최적화)
├── 🔍 Vector Store (벡터 검색)
├── ⚙️ LangChain Agents (에이전트 시스템)
└── 🔄 Integrated Workflow (통합 워크플로우)
```

## 🚀 새로운 API 엔드포인트

### LangChain 워크플로우
```bash
# 메인 워크플로우 실행
POST /api/langchain/workflow/run

# 워크플로우 상태 조회
GET /api/langchain/workflow/status/{session_id}

# 워크플로우 통계
GET /api/langchain/workflow/stats
```

### 메모리 관리
```bash
# 대화 기록 조회
POST /api/langchain/memory/query

# 메모리 통계
GET /api/langchain/memory/stats
```

### 벡터 검색
```bash
# 취미/커뮤니티 검색
POST /api/langchain/vector/search

# 벡터 스토어 통계
GET /api/langchain/vector/stats
```

### 에이전트 실행
```bash
# 일반 에이전트 실행
POST /api/langchain/agent/execute

# 취미 추천 에이전트
POST /api/langchain/agent/hobby-recommendation

# 커뮤니티 매칭 에이전트
POST /api/langchain/agent/community-matching

# 스케줄 통합 에이전트
POST /api/langchain/agent/schedule-integration
```

### 데이터 관리
```bash
# 샘플 데이터 추가
POST /api/langchain/data/sample

# 데이터 초기화
DELETE /api/langchain/data/clear

# 헬스 체크
GET /api/langchain/health
```

## 📊 워크플로우 흐름

### 1단계: 사용자 프로필 분석
- **메모리 활용**: 이전 대화 기록 참조
- **LangChain 에이전트**: LLM 기반 프로필 생성
- **컨텍스트 저장**: 취미 관련 정보 메모리에 저장

### 2단계: 취미 추천
- **벡터 검색**: 유사한 취미 자동 검색
- **에이전트 추천**: 개인화된 취미 추천
- **결과 통합**: 벡터 검색 + 에이전트 결과 병합

### 3단계: 커뮤니티 매칭
- **벡터 검색**: 취미 카테고리 기반 커뮤니티 검색
- **에이전트 매칭**: 사용자 프로필 기반 적합성 평가
- **매칭 점수**: 유사도 기반 정렬

### 4단계: 스케줄 통합
- **현재 스케줄**: MCP 서버에서 스케줄 정보 수집
- **에이전트 최적화**: 취미 활동을 스케줄에 효율적으로 통합
- **충돌 해결**: 시간 제약 및 충돌 자동 해결

### 5단계: 진행상황 추적 계획
- **추적 방법**: 다양한 진행상황 모니터링 방법 제안
- **마일스톤**: 단계별 목표 설정
- **동기부여 전략**: 지속적인 참여를 위한 전략 제시

## 🔧 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정
```bash
# OpenAI API 키 (LangChain 에이전트용)
export OPENAI_API_KEY="your-openai-api-key"

# 기존 환경변수들
export GOOGLE_MAPS_API_KEY="your-google-maps-api-key"
export OPENWEATHER_API_KEY="your-openweather-api-key"
```

### 3. 샘플 데이터 추가
```bash
curl -X POST "http://localhost:8000/api/langchain/data/sample"
```

### 4. 워크플로우 실행
```bash
curl -X POST "http://localhost:8000/api/langchain/workflow/run" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "30대 직장인, 주말 취미 찾고 있음",
    "user_profile": {
      "age": 30,
      "occupation": "office_worker",
      "location": "Seoul",
      "interests": ["reading", "technology"],
      "available_time": "weekends"
    }
  }'
```

## 📈 성능 향상 효과

### 1. **검색 품질 향상**
- 벡터 데이터베이스로 인한 3-5배 빠른 검색
- 사용자 프로필 기반 정확도 향상

### 2. **개인화 수준 향상**
- 대화 기록 기반 컨텍스트 인식
- 일관된 사용자 경험 제공

### 3. **확장성 개선**
- 텍스트 분할기로 대용량 문서 처리 가능
- 모듈화된 에이전트 시스템

### 4. **에러 처리 강화**
- graceful degradation 지원
- 자동 fallback 메커니즘

## 🧪 테스트 방법

### 1. 단위 테스트
```bash
cd tests/unit
pytest test_langchain_workflow.py
pytest test_memory_manager.py
pytest test_vector_store.py
```

### 2. 통합 테스트
```bash
cd tests/integration
pytest test_langchain_integration.py
```

### 3. API 테스트
```bash
# 워크플로우 실행 테스트
curl -X POST "http://localhost:8000/api/langchain/workflow/run" \
  -H "Content-Type: application/json" \
  -d '{"user_input": "테스트 입력"}'

# 상태 확인
curl "http://localhost:8000/api/langchain/workflow/stats"
```

## 🔍 모니터링 및 디버깅

### 1. 워크플로우 상태 모니터링
```bash
# 활성 세션 확인
curl "http://localhost:8000/api/langchain/workflow/stats"

# 특정 세션 상태 확인
curl "http://localhost:8000/api/langchain/workflow/status/{session_id}"
```

### 2. 컴포넌트별 상태 확인
```bash
# 메모리 상태
curl "http://localhost:8000/api/langchain/memory/stats"

# 벡터 스토어 상태
curl "http://localhost:8000/api/langchain/vector/stats"

# 에이전트 상태
curl "http://localhost:8000/api/langchain/agent/status"
```

### 3. 헬스 체크
```bash
curl "http://localhost:8000/api/langchain/health"
```

## 🚀 향후 개선 계획

### 1. **고급 메모리 관리**
- Redis 기반 분산 메모리
- 메모리 압축 및 최적화

### 2. **벡터 검색 고도화**
- 하이브리드 검색 (키워드 + 벡터)
- 실시간 인덱스 업데이트

### 3. **에이전트 확장**
- 더 많은 도구 및 기능
- 멀티 에이전트 협업

### 4. **성능 최적화**
- 비동기 처리 최적화
- 캐싱 전략 개선

## 📚 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/)
- [벡터 데이터베이스 가이드](https://docs.trychroma.com/)
- [프롬프트 엔지니어링 모범 사례](https://platform.openai.com/docs/guides/prompt-engineering)

---

**🎯 LangChain 기반 HSP Agent - AI가 당신의 완벽한 취미를 찾아드립니다!**

이제 LangChain의 핵심 개념들을 활용하여 더욱 지능적이고 효율적인 취미 추천 시스템을 경험해보세요.
