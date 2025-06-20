import streamlit as st
import asyncio
from mcp_agent.app import MCPApp
from mcp_agent.config import get_settings
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from srcs.common.streamlit_utils import get_agent_state
from srcs.urban_hive.urban_hive_agent import UrbanHiveAgent

app = MCPApp(
    name="urban_hive_app",
    settings=get_settings("configs/mcp_agent.config.yaml"),
)

async def main():
    await app.initialize()

    st.title("🏙️ Urban Hive Agent")
    st.caption("🚀 Your AI-powered urban data analysis assistant")

    # Use the state management pattern
    state = await get_agent_state(
        key="urban_hive_agent",
        agent_class=UrbanHiveAgent,
        llm_class=OpenAIAugmentedLLM,
        name="UrbanHiveAgent",
        instruction="""You are an AI-powered urban data analysis assistant.
        Your goal is to provide insights into real estate, market trends, and local sentiments
        by leveraging dynamic data sources and advanced analysis.
        Start by asking the user what urban data they are interested in.""",
        server_names=["urban_hive_mcp_server", "g-search-mcp"],
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "안녕하세요! 어떤 도시 데이터 분석을 도와드릴까요?"}
        ]

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("예: '서울 강남구 아파트의 최근 3개월 시세와 시장 동향을 알려줘'"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            response = ""
            with st.spinner("데이터 분석 중..."):
                # The UrbanHiveAgent is designed to run a full analysis based on a single prompt
                # It doesn't require conversational history for its core task.
                response = await state.agent.run(prompt)
            st.markdown(response)

        st.session_state["messages"].append({"role": "assistant", "content": response})


if __name__ == "__main__":
    asyncio.run(main())

