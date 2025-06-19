# 📋 Streamlit UI 현대화 및 표준화 계획

**📅 작성일**: 2024년 12월 20일
**✒️ 작성자**: Gemini AI (UI/UX Architect)
**🎯 목적**: `pages/` 디렉토리 내의 Streamlit 애플리케이션들의 코드 품질을 표준화하고, `srcs/basic_agents/streamlit_agent.py`의 모범 사례를 적용하여 안정성과 유지보수성을 극대화한다.

---

## 1. 문제 진단: 왜 UI 리팩터링이 시급한가?

`pages/urban_hive.py`, `pages/product_planner.py`, `pages/seo_doctor.py` 등의 코드를 분석한 결과, 다음과 같은 심각한 문제점들이 공통적으로 발견되었습니다.

- **비효율적인 상태 관리**: 대부분의 페이지가 상호작용할 때마다 에이전트 객체를 새로 생성하고 있습니다. 이는 리소스를 낭비하고, 에이전트의 상태(예: 대화 기록)를 유지할 수 없게 만드는 치명적인 결함입니다.
- **불안정한 비동기 처리**: `asyncio.run()`을 UI 코드 내에서 직접 호출하는 패턴은 Streamlit의 실행 모델과 충돌하여 얘기치 못한 오류나 성능 저하를 유발할 수 있습니다. 이는 매우 불안정한 구조입니다.
- **로직과 뷰의 강한 결합**: 에이전트를 실행하는 비즈니스 로직이 UI를 그리는 `st.button`과 같은 코드 블록 안에 깊숙이 섞여 있습니다. 이는 코드의 가독성을 해치고 재사용과 테스트를 거의 불가능하게 만듭니다.
- **코드 중복 및 비표준화**: 각 페이지가 자신만의 방식으로 에이전트를 초기화하고 실행 로직을 처리하고 있어, 일관성이 없고 유지보수 비용을 증가시킵니다.

이 문제들을 방치하면, 새로운 기능을 추가하거나 기존 버그를 수정하는 작업이 점점 더 어려워지는 **기술 부채의 악순환**에 빠지게 될 것입니다.

## 2. 새로운 표준: `streamlit_agent.py`의 핵심 패턴

`srcs/basic_agents/streamlit_agent.py`는 이러한 문제들을 해결할 수 있는 훌륭한 청사진을 제공합니다. 이 패턴의 핵심 요소는 다음과 같습니다.

1.  **`AgentState` 데이터 클래스**: `agent`와 `llm` 객체를 함께 캡슐화하여 상태를 명확하게 관리합니다.
2.  **`get_agent_state` 함수**: `st.session_state`를 활용하여 에이전트의 생명주기를 관리합니다. 한 번 생성된 에이전트 객체는 세션 내에서 재사용되어 효율성을 극대화하고, 페이지가 다시 로드되어도 연결 상태 등을 안정적으로 유지합니다.
3.  **최상위 비동기 함수 (`async def main`)**: Streamlit 스크립트 전체를 비동기 컨텍스트에서 실행하여, `await` 키워드를 통해 비동기 함수를 자연스럽고 안정적으로 호출합니다.
4.  **채팅 UI 기반 인터페이스**: 대화형 에이전트에 적합한 표준 UI를 제공하여 사용자 경험의 일관성을 확보합니다.

## 3. 실행 계획: 3단계 리팩터링 프로세스

### **1단계: `streamlit_utils.py` 공통 모듈 생성 (즉시 실행)**

가장 먼저, `streamlit_agent.py`의 핵심 로직을 모든 페이지에서 재사용할 수 있도록 공통 모듈로 추출해야 합니다.

**실행 내용:**
-   `srcs/common/streamlit_utils.py` 파일을 생성합니다.
-   `srcs/basic_agents/streamlit_agent.py`에서 `AgentState` 데이터 클래스와 `get_agent_state` 함수를 `srcs/common/streamlit_utils.py`로 옮깁니다.

이 작업을 통해 모든 Streamlit 페이지에서 표준화된 방식으로 에이전트 상태를 관리할 수 있는 기반이 마련됩니다.

### **2단계: `pages` 애플리케이션 리팩터링**

1단계에서 만든 공통 모듈을 사용하여 `pages/` 내의 모든 파일을 점진적으로 리팩터링합니다.

**리팩터링 가이드:**

1.  **파일 구조 변경**: 기존 코드를 `async def main():` 함수로 감싸고, 파일 끝에 `if __name__ == "__main__": asyncio.run(main())`을 추가합니다.
2.  **상태 관리 교체**: 페이지 상단에서 `st.session_state`에 직접 에이전트를 할당하거나, 버튼 클릭 시마다 에이전트를 생성하는 코드를 모두 제거합니다. 대신, `main` 함수 시작 부분에서 `get_agent_state`를 호출하여 에이전트 상태를 가져옵니다.
3.  **비동기 호출 변경**: `asyncio.run(agent_function(...))`와 같은 코드를 모두 `await agent_function(...)`으로 변경합니다.
4.  **UI 분리 (선택 사항)**: 가능하다면 UI 렌더링 함수와 에이전트 실행 로직 함수를 분리하여 코드 구조를 개선합니다.

**예시: `pages/urban_hive.py` 리팩터링 (Before & After)**

**Before (현재 코드의 문제점):**
```python
# pages/urban_hive.py (일부)

# 세션에 직접 할당 (불안정)
if 'urban_mcp_agent' not in st.session_state:
    st.session_state.urban_mcp_agent = UrbanHiveMCPAgent()

# 버튼 클릭 시 asyncio.run() 직접 호출 (위험)
if st.button("🔍 매칭 찾기", key="resource_match"):
    if resource_query:
        with st.spinner("..."):
            try:
                result = asyncio.run(run_urban_analysis(...)) # 문제 지점
                st.success("✅ Real MCP Agent 매칭 완료!")
                st.markdown(result)
            except Exception as e:
                st.error(f"MCP Agent 오류: {str(e)}")
```

**After (개선된 코드 예시):**
```python
# pages/urban_hive.py (리팩터링 후)
import streamlit as st
import asyncio
from srcs.common.streamlit_utils import get_agent_state, AgentState # 1. 공통 모듈 임포트
from srcs.urban_hive.urban_hive_agent import UrbanHiveMCPAgent
# ... other imports

async def main(): # 2. async main 함수로 전체 구조 변경
    setup_page("🏙️ Urban Hive Agent", "🏙️")
    
    # 3. get_agent_state로 안정적인 상태 관리
    # UrbanHiveMCPAgent는 LLM을 직접 attach하지 않으므로 llm_class는 제외
    state: AgentState = await get_agent_state(
        key="urban_hive_agent",
        agent_class=UrbanHiveMCPAgent
    )
    urban_agent = state.agent

    st.markdown("...") # UI 렌더링
    
    # ... 탭 및 UI 구성 ...
    
    if st.button("🔍 매칭 찾기", key="resource_match"):
        if resource_query:
            with st.spinner("실제 MCP Agent가 매칭을 분석하고 있습니다..."):
                try:
                    # 4. await으로 안전하게 비동기 함수 호출
                    # 실제 run_urban_analysis가 agent 객체를 필요로 하도록 수정 필요
                    result = await urban_agent.run_analysis(
                        location="",
                        category="SOCIAL_SERVICES",
                        query=f"자원 매칭 요청: {resource_query}"
                    )
                    st.success("✅ Real MCP Agent 매칭 완료!")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"MCP Agent 오류: {str(e)}")
        else:
            st.warning("검색할 내용을 입력해주세요.")


if __name__ == "__main__":
    asyncio.run(main()) # 5. 스크립트 실행
```

### **3단계: 리팩터링 우선순위**

모든 페이지를 한 번에 바꿀 수 없으므로, 중요도에 따라 우선순위를 정합니다.

1.  **최우선 (P0)**: `pages/product_planner.py`, `pages/urban_hive.py`, `pages/seo_doctor.py`, `pages/travel_scout.py`
    -   `CRITICAL_PROJECT_STATUS_REPORT.md`에서 언급된 4대 핵심 에이전트의 UI입니다. 이 페이지들의 안정화가 가장 시급합니다.
2.  **차순위 (P1)**: `pages/rag_agent.py`, `pages/research.py`, `pages/workflow.py`
    -   사용 빈도가 높거나 프로젝트의 핵심 기능을 담당하는 에이전트의 UI입니다.
3.  **후순위 (P2)**: 나머지 모든 `pages/` 파일
    -   전체적인 코드 품질 통일성을 위해 점진적으로 리팩터링을 완료합니다.

---

이 계획을 따르면 모든 Streamlit UI가 일관된 고품질 표준을 갖게 되어, **더 빠르고 안정적으로 기능을 개발하고 유지보수**할 수 있는 기반이 될 것입니다. 