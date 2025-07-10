"""
Figma Analyzer Agent
Figma 디자인을 분석하여 구조화된 요구사항을 추출하는 전문 Agent
"""

# ---------------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------------
import base64
import json
from typing import Any, Dict

from srcs.core.agent.base import BaseAgent
from srcs.product_planner_agent.utils.logger import get_product_planner_logger

logger = get_product_planner_logger("agent.figma_analyzer")

class FigmaAnalyzerAgent(BaseAgent):
    """
    A specialized agent that interacts with the Figma Context MCP server.
    This agent is now a thin client that interacts with a dedicated Figma Context MCP server.
    """
    def __init__(self, **kwargs):
        super().__init__("figma_analyzer_agent", **kwargs)
        # The MCP client is no longer created here.
        # The framework will provide tools from the 'figma' server
        # when the agent is run within an app context that has the server registered.
        logger.info("FigmaAnalyzerAgent initialized.")

    async def run_workflow(self, context: Any) -> Dict[str, Any]:
        """
        Runs the design analysis task by calling the Figma Context MCP server.
        """
        figma_file_id = context.get("figma_file_id")
        if not figma_file_id:
            msg = "Figma file ID is required."
            logger.error(msg)
            raise ValueError(msg)

        logger.info(f"Requesting Figma analysis for file_id: {figma_file_id} from MCP.")
        try:
            # The framework will automatically route this call to the 'figma' MCP server.
            # We assume a tool named 'analyze_figma_file' is available on that server.
            analysis_result = await self.app.tools.figma.analyze_figma_file(file_id=figma_file_id)
            logger.info("Successfully received analysis from Figma MCP.")
            return analysis_result
        except Exception as e:
            logger.error(f"Figma MCP call failed: {e}", exc_info=True)
            # Propagate the error to the main coordinator
            raise

    async def add_comment(self, file_id: str, node_id: str, comment: str) -> Dict[str, Any]:
        """
        Adds a comment to a specific node in a Figma file via the Figma MCP server.
        """
        logger.info(f"Adding comment to Figma file {file_id}, node {node_id}")
        try:
            # The framework will automatically route this call to the 'figma' MCP server.
            result = await self.app.tools.figma.add_comment_to_node(
                file_id=file_id,
                node_id=node_id,
                comment=comment
            )
            logger.info("Successfully added comment via Figma MCP.")
            return result
        except Exception as e:
            logger.error(f"Figma 'add comment' failed: {e}", exc_info=True)
            raise 