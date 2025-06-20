import streamlit as st
import asyncio
from mcp_agent.app import MCPApp
from mcp_agent.config import get_settings
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from srcs.urban_hive.urban_hive_agent import UrbanHiveMCPAgent, UrbanDataCategory
from srcs.common.page_utils import setup_page, render_home_button
from srcs.common.styles import get_common_styles, get_page_header
from srcs.common.streamlit_log_handler import setup_streamlit_logging

app = MCPApp(
    name="urban_hive_app",
    settings=get_settings("configs/mcp_agent.config.yaml"),
)

async def main():
    await app.initialize()
    setup_page("🏙️ Urban Hive Agent", "🏙️")
    st.markdown(get_common_styles(), unsafe_allow_html=True)
    header_html = get_page_header(
        "urban",
        "🏙️ Urban Hive Agent",
        "AI 기반 도시 데이터 분석 플랫폼. 교통, 안전, 환경, 부동산 등 다양한 도시 문제를 심층 분석합니다."
    )
    st.markdown(header_html, unsafe_allow_html=True)
    render_home_button()

    # --- Real-time Log Display ---
    log_expander = st.expander("실시간 실행 로그", expanded=False)
    log_container = log_expander.empty()
    setup_streamlit_logging(["mcp_agent", "urban_hive_agent"], log_container) # 에이전트 로거 추가
    # --- End Log Display ---

    st.markdown("---")

    # 메인 콘텐츠 영역에 채팅 인터페이스와 정보 표시
    main_content_col, agent_info_col = st.columns([2, 1])

    with main_content_col:
        # 채팅 인터페이스
        st.subheader("💬 도시 데이터 분석 요청")
        st.markdown(
            "아래 채팅창에 분석하고 싶은 도시 문제에 대해 질문해주세요. "
            "예를 들어, 특정 지역의 부동산 동향, 교통 상황, 환경 문제 등을 물어볼 수 있습니다."
        )

    # Initialize UrbanHiveMCPAgent directly in session state
    if 'urban_hive_agent' not in st.session_state:
        llm_instance = OpenAIAugmentedLLM()
        st.session_state.urban_hive_agent = UrbanHiveMCPAgent(app=app, llm=llm_instance)

    with main_content_col: # 채팅 인터페이스 아래에 표시
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        "안녕하세요! 저는 Urban Hive 에이전트입니다. 🏙️\n"
                        "분석하고 싶은 도시, 주제, 기간 등을 알려주시면 관련 데이터를 분석해 드릴게요.\n"
                        "예: '서울 성동구의 최근 3개월간 부동산 시장 동향과 전망을 알려줘.'"
                    ),
                }
            ]

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("예: '서울 강남구 아파트의 최근 3개월 시세와 시장 동향을 알려줘'"):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            with st.chat_message("assistant"):
                response = ""
                with st.spinner("데이터 분석 중..."):
                    agent: UrbanHiveMCPAgent = st.session_state.urban_hive_agent
                    response = await agent.run(prompt)
                st.markdown(response)

            st.session_state["messages"].append({"role": "assistant", "content": response})

    with agent_info_col:
        st.markdown("### ✨ Urban Hive 특징")
        st.markdown("- **실시간 데이터 기반 분석**: 최신 도시 현황 반영")
        st.markdown("- **다각적 인사이트 제공**: 교통, 안전, 환경, 부동산 등 종합 분석")
        st.markdown("- **예측 모델링**: 미래 도시 변화 예측 및 선제적 대응 방안 제시")
        st.markdown("- **실행 가능한 솔루션**: 데이터 기반 정책 제언 및 실행 계획 수립 지원")

    # 에이전트 정보 및 기능 안내 (메인 콘텐츠 영역으로 이동)
    st.markdown("---")
    with st.expander("💡 Urban Hive Agent 정보 더보기", expanded=False):
        st.markdown("## 💡 Urban Hive Agent란?")
        st.markdown(
            "Urban Hive는 복잡한 도시 데이터를 분석하여 시민 생활 개선과 지속 가능한 도시 발전에 필요한 "
            "통찰력을 제공하는 AI 에이전트입니다. 자연어 질문을 통해 특정 지역의 다양한 도시 문제에 대한 "
            "심층 분석 보고서를 받아보세요."
        )
        st.markdown("---")
        st.markdown("###  주요 분석 카테고리")
        # 카테고리를 2열로 표시
        cat_cols = st.columns(2)
        for i, category in enumerate(UrbanDataCategory):
            with cat_cols[i % 2]:
                st.markdown(f"- {category.value}")
        
        st.markdown("---")
        st.markdown("### 🚀 활용 예시")
        st.markdown("- '서울 강남구의 최근 1개월간 교통 혼잡도와 해결 방안은?'")
        st.markdown("- '부산 해운대구의 여름철 관광객 안전 문제와 대응 전략은?'")
        st.markdown("- '인천 송도 국제도시의 미세먼지 현황과 환경 개선 방안을 알려줘.'")
        st.markdown("- '대전 유성구의 신규 아파트 단지 주변 상권 활성화 가능성은?'")

if __name__ == "__main__":
    asyncio.run(main())
