"""
Conversation Agent
사용자와 대화를 통해 상세한 제품 요구사항을 수집하는 Agent
"""

from typing import Dict, Any
import json
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from srcs.core.agent.base import BaseAgent
from srcs.product_planner_agent.prompts import PROMPT
from srcs.product_planner_agent.utils.llm_utils import get_llm_factory
from srcs.product_planner_agent.utils.logger import get_product_planner_logger

logger = get_product_planner_logger("agent.conversation")


class ConversationAgent(BaseAgent):
    """사용자 대화 및 요구사항 수집 전문 Agent"""
    
    def __init__(self, **kwargs):
        super().__init__("conversation_agent", **kwargs)
        logger.info("ConversationAgent initialized.")

    async def collect_requirements_via_chat(self, initial_prompt: str) -> Dict[str, Any]:
        """
        사용자의 초기 질문을 바탕으로 대화를 통해 제품 요구사항을 수집합니다.
        (실제 채팅 기능은 추후 구현)
        """
        logger.info(f"Collecting requirements from initial prompt: '{initial_prompt[:50]}...'")
        prompt = f"""
        You are a product planning conversation specialist. A user has provided the following initial request. 
        Your task is to interpret this request and formulate a structured summary of product requirements.

        **User's Initial Request:**
        "{initial_prompt}"

        **Instructions:**
        1.  **Identify Core Goal:** What is the main objective the user wants to achieve?
        2.  **Extract Key Features:** List the key features or functionalities mentioned or implied.
        3.  **Clarifying Questions:** Formulate 3-5 important follow-up questions to gather more details.
        
        Provide the output in a structured JSON format.
        """
        
        try:
            # Use the LLM instance from the app context
            requirements_str = await self.app.llm.generate_str(
                prompt,
                request_params=RequestParams(
                    model="gemini-1.5-flash-latest", # This should be consistent with llm_utils
                    temperature=0.7,
                )
            )
            requirements = json.loads(requirements_str)
            requirements["status"] = "collected_successfully"
            logger.info("Successfully collected and structured requirements.")
            return {"collected_requirements": requirements}
        except Exception as e:
            logger.error(f"Error collecting requirements: {e}", exc_info=True)
            raise 