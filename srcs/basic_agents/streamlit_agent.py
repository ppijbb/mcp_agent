import streamlit as st
import asyncio
from mcp_agent.app import MCPApp
from srcs.common.utils import setup_agent_app


def main():
    st.title("MCP Agent Interface")

    app = setup_agent_app("streamlit_app")

    # The rest of the streamlit logic can be implemented here
    st.write("Streamlit Agent is running with the new config system.")

    # Example of running an async function from a synchronous context (Streamlit)
    # This might need a more robust solution in a real app
    if st.button("Run a simple async task"):
        st.write("Running...")
        result = asyncio.run(simple_async_task(app))
        st.write(f"Task result: {result}")


async def simple_async_task(app: MCPApp):
    async with app.run() as app_context:
        app_context.logger.info("Simple async task executed from Streamlit.")
        return "Success!"


if __name__ == "__main__":
    main()
