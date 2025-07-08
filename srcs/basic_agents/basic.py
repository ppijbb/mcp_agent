import asyncio
import os
import time

from mcp_agent.app import MCPApp
from mcp_agent.config import (
    AgentConfig,
    MCPAppConfig,
)
from srcs.common.utils import setup_agent_app
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM


async def main():
    app = setup_agent_app("basic_app")
    
    # Run the app
    async with app.run() as app_context:
        # Get the logger from the app context
        logger = app_context.logger
        context = app_context.context

        logger.info("Current config:", data=context.config.model_dump())

        # Add the current directory to the filesystem server's args
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        finder_agent = Agent(
            name="finder",
            instruction="""You are an agent with access to the filesystem, 
            as well as the ability to fetch URLs. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls, 
            and return the URI and CONTENTS of the closest match.""",
            server_names=["fetch", "filesystem"],
        )

        async with finder_agent:
            logger.info("finder: Connected to server, calling list_tools...")
            result = await finder_agent.list_tools()
            logger.info("Tools available:", data=result.model_dump())

            llm = await finder_agent.attach_llm(OpenAIAugmentedLLM)
            result = await llm.generate_str(
                message="Print the contents of mcp_agent.config.yaml verbatim",
            )
            logger.info(f"mcp_agent.config.yaml contents: {result}")

            # Let's switch the same agent to a different LLM
            llm = await finder_agent.attach_llm(GoogleAugmentedLLM)

            result = await llm.generate_str(
                message="Print the first 2 paragraphs of https://modelcontextprotocol.io/introduction",
            )
            logger.info(f"First 2 paragraphs of Model Context Protocol docs: {result}")

            # Multi-turn conversations
            result = await llm.generate_str(
                message="Summarize those paragraphs in a 128 character tweet in korean",
                # You can configure advanced options by setting the request_params object
                request_params=RequestParams(
                    # See https://modelcontextprotocol.io/docs/concepts/sampling#model-preferences for more details
                    modelPreferences=ModelPreferences(
                        costPriority=0.1, speedPriority=0.2, intelligencePriority=0.7
                    ),
                    # You can also set the model directly using the 'model' field
                    # Generally request_params type aligns with the Sampling API type in MCP
                ),
            )
            logger.info(f"Paragraph as a tweet: {result}")


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")