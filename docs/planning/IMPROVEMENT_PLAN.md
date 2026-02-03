# Product Planner Agent v2.0 개선 계획

**기준 문서**: `planning/NEW_PRODUCT_PLANNER_AGENT_EVALUATION_REPORT.md`

이 문서는 Product Planner Agent v2.0의 평가 결과에 따라 식별된 개선 영역을 해결하기 위한 작업 계획을 정의합니다.

## 🎯 주요 개선 목표

평가 보고서에서 지적된 두 가지 최우선 순위(P1) 과제를 해결하여 에이전트의 완성도를 높이는 것을 목표로 합니다.

1.  **P1-1: ReAct 패턴 구현 (예상 작업량: 20-25시간)**
2.  **P1-2: 실제 MCP 서버 연동 강화 (예상 작업량: 10-15시간)**

완료 시, 에이전트는 **95점 이상의 '최우수+' 등급** 달성이 예상됩니다.

---

## 📋 세부 작업 계획

### 1. ReAct 패턴 구현 (P1-1)

**목표**: 각 에이전트가 자율적인 `THOUGHT -> ACTION -> OBSERVATION` 사이클을 통해 작업을 수행하도록 아키텍처를 개선하여 유연성과 문제 해결 능력을 극대화합니다.

- [ ] **1-1. `CoordinatorAgent` 분석**:
    - `agents/coordinator_agent.py`의 현재 워크플로우 실행 로직을 분석합니다.
- [ ] **1-2. ReAct 패턴 설계**:
    - `CoordinatorAgent` 내에 `THOUGHT -> ACTION -> OBSERVATION` 루프를 설계합니다.
    - `THOUGHT`: LLM을 사용하여 주어진 태스크를 기반으로 10개 전문 에이전트를 활용한 단계별 실행 계획을 수립합니다.
    - `ACTION`: 수립된 계획에 따라 다른 에이전트들을 순차적 또는 병렬적으로 호출합니다.
    - `OBSERVATION`: 에이전트 실행 결과를 종합하고, 최종 결과물을 생성하거나 다음 단계를 위한 관찰 결과를 도출합니다.
- [ ] **1-3. ReAct 패턴 구현**:
    - 설계에 따라 `CoordinatorAgent`의 `run` 또는 관련 메소드를 수정합니다.
    - 보고서의 예시 코드를 참고하여 구현합니다.
- [ ] **1-4. 타 에이전트 연동 방식 수정**:
    - `CoordinatorAgent`의 변경에 따라, 다른 9개 에이전트가 호출되고 결과를 반환하는 방식을 검토하고 필요 시 수정합니다.
- [ ] **1-5. 통합 테스트**:
    - ReAct 패턴이 적용된 전체 워크플로우가 정상적으로 동작하는지 테스트합니다.

### 2. 실제 MCP 서버 연동 강화 (P1-2)

**목표**: Figma 및 Notion 연동을 가이드 제공 수준에서 실제 API 호출 코드로 구현하여 MCP 서버 통합도를 100%로 끌어올립니다.

- [ ] **2-1. Figma 연동 구현**:
    - [ ] `agents/figma_analyzer_agent.py` 분석 및 수정: Figma API를 직접 호출하여 디자인을 분석하는 코드를 구현합니다.
    - [ ] `agents/figma_creator_agent.py` 분석 및 수정: Figma API를 직접 호출하여 새로운 디자인 에셋을 생성하는 코드를 구현합니다.
- [ ] **2-2. Notion 연동 구현**:
    - [ ] `agents/notion_document_agent.py` 분석 및 수정: Notion API를 직접 호출하여 최종 산출물을 Notion 페이지로 자동 생성하고 문서화하는 코드를 구현합니다.
- [ ] **2-3. API Key 및 환경 변수 설정 가이드 업데이트**:
    - 실제 API 연동에 필요한 API 키 및 환경 변수 설정 방법을 문서화하고 사용자에게 안내합니다.
- [ ] **2-4. 연동 테스트**:
    - Figma 및 Notion API 연동 기능이 정상적으로 동작하는지 테스트합니다.

---

## 🚀 다음 단계

1.  **ReAct 패턴 구현** 작업부터 시작합니다.
2.  `agents/coordinator_agent.py` 파일을 읽고 분석하여 현재 구조를 파악합니다. 