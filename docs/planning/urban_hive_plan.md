# 🤖 AI 에이전트 기획서: Urban Hive (어반 하이브)

> **컨셉:** 도시를 하나의 유기체로 보고, 자원 낭비를 막고 주민 유대를 극대화하여 공동의 문제를 해결하는 초지역적 소셜 AI

---

## 🏗️ 아키텍처 설계

`어반 하이브`는 실제 데이터를 기반으로 동작하는 실용적인 시스템입니다.

- **User Interface:** 주민을 위한 `모바일 앱`과 지자체/관리자를 위한 `웹 대시보드`로 구성됩니다.
- **AI Core (MCP Server):** 세 가지 핵심 도구가 독립적인 MCP 서버로 작동합니다.
    - **`ResourceMatcher_Tool`:** 물품 공유, 남는 음식 매칭 등 자원 매칭을 담당합니다.
    - **`SocialConnector_Tool`:** 관심사 기반의 소모임 제안, 고립 가구 케어 등 관계 연결을 담당합니다.
    - **`UrbanAnalyst_Tool`:** 쓰레기 무단 투기 지역 분석, 위험 지역 예측 등 도시 문제를 분석합니다.
- **Backend Infrastructure:** `FastAPI` 기반의 API 서버가 전체 요청을 처리하며, `PostgreSQL/PostGIS` 데이터베이스와 `Redis` 캐시를 사용합니다. `공공 데이터 API`와 연동하여 외부 정보를 활용합니다.

```mermaid
graph TD;
    subgraph "User Interface"
        A[Mobile App <br/>(주민용)]
        B[Web Dashboard <br/>(지자체/관리자용)]
    end

    subgraph "AI Core (MCP Server)"
        C[ResourceMatcher_Tool]
        D[SocialConnector_Tool]
        E[UrbanAnalyst_Tool]
    end

    subgraph "Backend Infrastructure"
        F[API Server <br/>(FastAPI)]
        G[Database <br/>(PostgreSQL/PostGIS)]
        H[Cache <br/>(Redis)]
        I[Public Data API]
    end

    A & B --> F;
    F --> C & D & E;
    C & D & E --> G & H & I;
```

---

## 📋 단계별 개발 로드맵

### **Phase 1: MVP (4주) - '우리 동네 자원 공유'**

- **목표:** 주민 간 물품/음식/재능 공유 기능 구현.
- **주요 작업:**
    1.  `ResourceMatcher_Tool` MCP 서버 개발 (기본 매칭 로직).
    2.  사용자 가입 및 프로필 기능 개발.
    3.  물품/재능 "드려요", "필요해요" 게시 기능 구현.
    4.  지도 기반 자원 확인 기능 (모바일 앱).
    5.  핵심 API 및 데이터베이스 스키마 설계.

### **Phase 2: Core (6주) - '고립 없는 커뮤니티'**

- **목표:** 사회적 연결 기능 추가를 통해 커뮤니티 강화.
- **주요 작업:**
    1.  `SocialConnector_Tool` MCP 서버 개발 (관심사/프로필 분석).
    2.  소모임 생성 및 참여 기능.
    3.  "산책 친구 찾기", "반려동물 모임" 등 자동 매칭 기능.
    4.  고립 위험군 식별을 위한 활동 데이터 분석 모델 개발.

### **Phase 3: Expansion (8주) - '데이터 기반 도시 해결'**

- **목표:** 도시 문제 분석 및 해결 기능 도입으로 공공의 이익 창출.
- **주요 작업:**
    1.  `UrbanAnalyst_Tool` MCP 서버 개발.
    2.  공공 데이터(범죄율, 쓰레기 민원 등) 연동.
    3.  관리자용 웹 대시보드 개발 (데이터 시각화).
    4.  AI 분석 기반 도시 문제 리포트 자동 생성 기능. 