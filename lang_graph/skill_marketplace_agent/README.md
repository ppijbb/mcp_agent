# 🎓 Skill Marketplace & Learning Coach Agent

**LangChain + LangGraph 기반 양면 시장(Two-Sided Marketplace) 스킬 학습 플랫폼 Agent**

학습자와 강사/컨텐츠를 매칭하고, 네트워크 효과를 통한 구조적 상업성을 실현하는 Multi-Agent 시스템

## 🏗️ 아키텍처 개요

Skill Marketplace Agent는 다음 6개의 전문 Agent로 구성된 Multi-Agent 시스템입니다:

### 1. 🤖 전문 Agents

- **LearnerProfileAnalyzerAgent**: 학습자 프로필 분석 전문가
- **SkillPathRecommenderAgent**: 맞춤형 스킬 학습 경로 추천 전문가
- **InstructorMatcherAgent**: 최적 강사 매칭 전문가 (양면 시장 핵심)
- **ContentRecommenderAgent**: 학습 컨텐츠 추천 전문가
- **LearningProgressTrackerAgent**: 학습 진행 추적 및 피드백 전문가
- **MarketplaceOrchestratorAgent**: 종합 오케스트레이터 (매칭, 결제, 리뷰)

### 2. 🔄 LangGraph 워크플로우

```mermaid
graph TD
    A[사용자 요청] --> B[입력 검증]
    B --> C[학습자 프로필 분석]
    C --> D[스킬 경로 추천]
    D --> E[강사 매칭] ← 양면 시장 핵심
    E --> F[컨텐츠 추천]
    F --> G[학습 계획 생성]
    G --> H[거래 처리] ← 수익화
    H --> I[최종 리포트 생성]
    I --> J[완료]
```

### 3. 💰 구조적 상업성 (운영 과정에서 수익화)

#### 양면 시장 수수료 모델
- **거래 수수료**: 학습 세션 거래당 15-20% 수수료
- **프리미엄 구독**: 
  - 학습자: 월 $9.99 (무제한 매칭, 우선 매칭, 학습 리포트)
  - 강사: 월 $29.99 (프로필 노출, 우선 검색, 통계 대시보드)

#### B2B 확장
- 기업 교육 프로그램 패키지 판매 ($5,000-50,000/년)
- 기업 복지 프로그램 연계

#### 데이터 수익화
- 익명화된 학습 패턴 데이터 → 기업 인사이트 판매
- 스킬 트렌드 분석 리포트 ($500-2,000/리포트)

#### 네트워크 효과
- 강사 증가 → 학습자 선택권 증가 → 학습자 증가
- 학습자 증가 → 강사 수요 증가 → 강사 증가
- 양방향 성장 루프로 플랫폼 가치 지수적 증가

### 4. 🛠️ MCP 서버 연동

- **Filesystem**: 데이터 저장 및 관리
- **g-search**: 스킬 정보 및 학습 컨텐츠 검색
- **fetch**: 외부 API 데이터 수집

## 🚀 빠른 시작

### 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일을 편집하여 API 키들을 설정
```

### 실행 방법

```bash
# 메인 실행
python -m lang_graph.skill_marketplace_agent.main
```

### 사용 예시

```python
from lang_graph.skill_marketplace_agent.main import SkillMarketplaceAgent
from lang_graph.skill_marketplace_agent.llm.model_manager import ModelProvider

# Agent 초기화
agent = SkillMarketplaceAgent(preferred_provider=ModelProvider.GROQ)

# 워크플로우 실행
result = await agent.run_marketplace_workflow(
    user_input="Python 프로그래밍을 배우고 싶어요. 초보자에게 맞는 강사와 학습 경로를 추천해주세요.",
    learner_id="learner_001",
    learner_profile={
        "learner_id": "learner_001",
        "goals": ["Python 프로그래밍 마스터"],
        "current_skills": {"Python": "beginner"},
        "learning_style": "visual",
        "preferred_format": "one-on-one",
        "budget_range": "medium",
        "time_availability": "medium"
    }
)
```

## 📁 프로젝트 구조

```
lang_graph/skill_marketplace_agent/
├── __init__.py
├── main.py                 # 메인 진입점
├── agents/                 # 전문 Agents
│   ├── learner_profile_analyzer.py
│   ├── skill_path_recommender.py
│   ├── instructor_matcher.py
│   ├── content_recommender.py
│   ├── learning_progress_tracker.py
│   └── marketplace_orchestrator.py
├── chains/                 # LangGraph 워크플로우
│   ├── state_management.py
│   └── marketplace_chain.py
├── tools/                  # 도구 모음
│   ├── mcp_tools.py
│   ├── marketplace_tools.py
│   └── learning_tools.py
├── llm/                    # LLM 관리
│   ├── model_manager.py
│   └── fallback_handler.py
├── config/                 # 설정
│   └── marketplace_config.py
└── utils/                  # 유틸리티
    └── validators.py
```

## 🎯 주요 기능

### 1. 학습자 프로필 분석
- 학습 목표, 현재 스킬 수준, 학습 스타일 분석
- 예산 및 시간 가용성 파악
- 맞춤형 학습 경로 추천

### 2. 양면 시장 매칭 (핵심 기능)
- 학습자-강사 최적 매칭
- 매칭 점수 계산 (스킬, 수준, 스타일, 예산, 시간대)
- 실시간 강사 가용성 확인
- 상위 5명 강사 추천

### 3. 스킬 학습 경로 추천
- 단계별 학습 경로 생성
- 선수 스킬 및 의존성 분석
- 예상 소요 시간 계산
- 학습 리소스 추천

### 4. 학습 컨텐츠 추천
- 온라인 강의, 튜토리얼, 책, 비디오 추천
- 학습 스타일 기반 컨텐츠 매칭
- 예산 및 시간 제약 고려

### 5. 학습 진행 추적
- 세션별 진행률 기록
- 스킬 향상 추적 및 분석
- 학습 리포트 생성
- 다음 단계 추천

### 6. Marketplace 거래 처리
- 학습 세션 생성 및 예약
- 결제 처리 (시뮬레이션)
- 수수료 계산 (15-20%)
- 거래 기록 관리

## 🔧 환경 설정

### 필수 환경변수

```bash
# LLM API Keys (최소 1개 필요)
GROQ_API_KEY=your_groq_api_key          # 우선 사용
OPENROUTER_API_KEY=your_openrouter_key  # 우선 사용
GOOGLE_API_KEY=your_google_api_key      # Fallback
OPENAI_API_KEY=your_openai_api_key      # Fallback
ANTHROPIC_API_KEY=your_anthropic_key    # Fallback
```

### 선택적 환경변수

```bash
# 디렉토리 설정
MARKETPLACE_OUTPUT_DIR=marketplace_reports
MARKETPLACE_DATA_DIR=marketplace_data
```

## 💼 비즈니스 모델

### 수익 구조

1. **거래 수수료**: 15-20% (주 수익원)
2. **프리미엄 구독**: 
   - 학습자: 월 $9.99
   - 강사: 월 $29.99
3. **B2B 프로그램**: 기업당 $5,000-50,000/년
4. **데이터 인사이트**: 리포트당 $500-2,000

### 예상 ROI

- **1년차**: 50,000 학습자, 5,000 강사 → $2M ARR
- **3년차**: 500,000 학습자, 50,000 강사 → $50M+ ARR

### 네트워크 효과

- **양면 시장 구조**: 학습자와 강사 모두 증가할수록 플랫폼 가치 증가
- **데이터 축적**: 학습 패턴 데이터 축적 → 더 나은 매칭 및 추천
- **생태계 구축**: 강사, 컨텐츠 제공자, 기업 파트너 확장

## 🛡️ 기술적 특징

- **LangChain 기반**: Multi-Agent 시스템
- **LangGraph**: 워크플로우 관리
- **Multi-Model LLM**: Groq/OpenRouter 우선, 자동 Fallback
- **양면 시장 매칭**: 지능형 매칭 알고리즘
- **Production-Ready**: 에러 처리, 로깅, 모니터링

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**🎓 Skill Marketplace Agent - 구조적 상업성을 실현하는 양면 시장 학습 플랫폼!**

