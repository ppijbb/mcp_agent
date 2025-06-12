# `pages` 디렉토리 개선 작업 목록 (To-Do List)

이 문서는 `pages` 디렉토리의 파일들에서 발견된 **모든 폴백 전략과 하드코딩 문제를 완전히 제거**하기 위한 작업 목록을 정의합니다.

## 🎯 핵심 원칙

**폴백 전략 완전 제거**: 모든 시스템은 실제 구현체와만 동작해야 하며, 폴백이나 모의 데이터는 허용하지 않습니다.
- 의존성이 없으면 시스템이 실행되지 않아야 합니다
- 모든 데이터는 실제 소스에서 가져와야 합니다
- 하드코딩된 응답이나 샘플 데이터는 완전히 제거해야 합니다

## 🚨 즉시 제거해야 할 항목들

### 1. **모든 폴백 함수 제거**
- `pages/finance_health.py`의 `get_backup_market_data()`, `get_backup_crypto_data()` 완전 삭제
- `pages/seo_doctor.py`의 `render_fallback_interface()` 완전 삭제
- `pages/ai_architect.py`의 폴백 응답 로직 완전 삭제

### 2. **모든 모의/시뮬레이션 로직 제거**
- `pages/business_strategy.py`의 하드코딩된 템플릿 응답 완전 삭제
- `pages/data_generator.py`의 `generate_ai_smart_data()` 샘플 데이터 완전 삭제
- `pages/rag_agent.py`의 키워드 매칭 기반 응답 사전 완전 삭제
- `pages/decision_agent.py`의 `MockDecisionAgent` 및 모든 샘플 데이터 생성 함수 완전 삭제

### 3. **모든 하드코딩된 데이터 제거**
- UI 기본값, 샘플 데이터, 정적 옵션 리스트 모두 제거
- 파일 경로, 디렉토리명 하드코딩 완전 제거

---

## 📂 파일별 완전 제거 작업 목록

#### 📄 `pages/ai_architect.py`
**🗑️ 제거 대상:**
- `generate_architect_text_output()` 함수의 모든 폴백 응답 로직
- `execute_architect_agent()` 함수의 시뮬레이션 로직
- `ai_architect_reports/` 하드코딩된 경로

**✅ 대체 방안:**
- 실제 `EvolutionaryAIArchitectAgent` 구현체만 사용
- 에이전트 실패 시 명확한 에러 메시지와 함께 실행 중단
- 동적 경로 설정 시스템 구축

#### 📄 `pages/business_strategy.py`
**🗑️ 제거 대상:**
- `execute_business_strategy_agent()` 함수의 하드코딩된 템플릿 응답 전체
- `business_strategy_reports/` 하드코딩된 경로

**✅ 대체 방안:**
- 실제 비즈니스 전략 분석 AI 에이전트 호출만 허용
- 에이전트 미구현 시 페이지 접근 차단
- 동적 경로 설정 시스템 구축

#### 📄 `pages/cybersecurity.py`
**🗑️ 제거 대상:**
- `cybersecurity_infrastructure_reports/` 하드코딩된 경로

**✅ 대체 방안:**
- 동적 경로 설정 시스템 구축

#### 📄 `pages/data_generator.py`
**🗑️ 제거 대상:**
- `generate_ai_smart_data()` 함수의 "김철수", "이영희" 등 모든 샘플 데이터
- UI의 모든 하드코딩된 선택 옵션들
- `data_generator_reports/` 하드코딩된 경로

**✅ 대체 방안:**
- 실제 AI 데이터 생성 에이전트만 사용
- UI 옵션을 외부 API나 설정에서 동적 로드
- 에이전트 미구현 시 페이지 접근 차단
- 동적 경로 설정 시스템 구축

#### 📄 `pages/decision_agent.py`
**🗑️ 제거 대상:**
- `MockDecisionAgent` 클래스 전체 삭제
- `create_sample_interactions()` 함수 전체 삭제
- `generate_sample_decision_history()` 함수 전체 삭제
- 모든 하드코딩된 테스트 시나리오, 성능 지표, 시스템 설정
- `decision_agent_reports/` 하드코딩된 경로

**✅ 대체 방안:**
- 실제 `DecisionAgent` 구현체만 사용
- 실시간 의사결정 데이터만 표시
- 에이전트 미구현 시 페이지 접근 차단
- 동적 경로 설정 시스템 구축

#### 📄 `pages/finance_health.py`
**🗑️ 제거 대상:**
- `get_backup_market_data()` 함수 전체 삭제
- `get_backup_crypto_data()` 함수 전체 삭제
- `get_real_economic_indicators()`, `get_real_market_data()` 함수의 하드코딩된 데이터
- 재무 건전성 점수 계산의 하드코딩된 로직
- 모든 샘플 포트폴리오 데이터
- `finance_health_reports/` 하드코딩된 경로

**✅ 대체 방안:**
- 실제 금융 API (yfinance, FRED API 등)만 사용
- API 실패 시 명확한 에러 메시지와 함께 기능 중단
- 점수 계산 로직을 외부 룰 엔진으로 완전 분리
- 동적 경로 설정 시스템 구축

#### 📄 `pages/hr_recruitment.py`
**🗑️ 제거 대상:**
- `recruitment_reports/` 하드코딩된 경로
- UI의 하드코딩된 예시 텍스트들

**✅ 대체 방안:**
- 동적 경로 설정 시스템 구축
- UI 텍스트를 다국어 지원 시스템으로 분리

#### 📄 `pages/rag_agent.py`
**🗑️ 제거 대상:**
- `generate_rag_response()` 함수의 키워드 매칭 기반 응답 사전 전체
- `sample_questions` 하드코딩된 질문 리스트

**✅ 대체 방안:**
- 실제 RAG 시스템 (VectorDB + LLM) 구현체만 사용
- RAG 시스템 미구현 시 페이지 접근 차단
- 질문 예시를 동적으로 생성하거나 외부에서 로드

#### 📄 `pages/research.py`
**🗑️ 제거 대상:**
- `research_reports/` 하드코딩된 경로
- UI의 하드코딩된 예시 텍스트들

**✅ 대체 방안:**
- 동적 경로 설정 시스템 구축
- UI 텍스트를 다국어 지원 시스템으로 분리

#### 📄 `pages/seo_doctor.py`
**🗑️ 제거 대상:**
- `render_fallback_interface()` 함수 전체 삭제
- `LIGHTHOUSE_AVAILABLE` 체크 로직 삭제
- `progress_steps` 하드코딩된 진행 단계
- 점수별 색상 결정 하드코딩된 로직
- `seo_doctor_reports/` 하드코딩된 경로

**✅ 대체 방안:**
- `lighthouse` 라이브러리 필수 의존성으로 설정
- 의존성 없으면 애플리케이션 시작 차단
- UI 로직을 테마 시스템으로 분리
- 동적 경로 설정 시스템 구축

#### 📄 `pages/travel_scout.py`
**🗑️ 제거 대상:**
- UI 기본값 "Seoul", "Tokyo" 하드코딩

**✅ 대체 방안:**
- 사용자 위치 기반 동적 기본값 설정
- 또는 기본값 없이 필수 입력으로 변경

#### 📄 `pages/urban_hive.py`
**🗑️ 제거 대상:**
- `analysis_options` 하드코딩된 리스트
- 정적 연결 상태 표시 로직

**✅ 대체 방안:**
- 분석 옵션을 외부 API에서 동적 로드
- 실시간 서버 상태 체크 시스템 구현
- 연결 실패 시 명확한 에러 표시

#### 📄 `pages/workflow.py`
**🗑️ 제거 대상:**
- 모든 하드코딩된 에이전트 지시사항(instruction)
- 모든 하드코딩된 작업(task) 정의
- `workflow_reports/` 하드코딩된 경로

**✅ 대체 방안:**
- 워크플로우 정의를 외부 YAML/JSON 파일로 완전 분리
- 에이전트 프롬프트 템플릿 시스템 구축
- 동적 경로 설정 시스템 구축

---

## 🎯 실행 우선순위

### Phase 1: 폴백 시스템 완전 제거 (즉시 실행)
1. 모든 `get_backup_*`, `render_fallback_*` 함수 삭제
2. 모든 `Mock*` 클래스 및 시뮬레이션 로직 삭제
3. 하드코딩된 샘플 데이터 생성 함수 삭제

### Phase 2: 실제 구현체 연동 (1주차)
1. 실제 AI 에이전트 호출 로직 구현
2. 실제 외부 API 연동 구현
3. 의존성 체크 및 에러 핸들링 강화

### Phase 3: 동적 설정 시스템 구축 (2주차)
1. 중앙 설정 관리 시스템
2. 외부 데이터 소스 연동
3. 실시간 상태 모니터링 시스템

**결과**: 모든 기능이 실제 구현체와만 동작하며, 폴백이나 모의 데이터 없이 완전한 시스템으로 전환 