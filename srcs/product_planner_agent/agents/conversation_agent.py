"""
Conversation Agent
사용자와 대화를 통해 상세한 제품 요구사항을 수집하는 Agent
"""

from srcs.core.agent.base import BaseAgent, AgentContext
from typing import Dict, Any
import json
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.logging.logger import get_logger


logger = get_logger("conversation_agent")


class ConversationAgent(BaseAgent):
    """사용자 대화 및 요구사항 수집 전문 Agent"""
    
    def __init__(self):
        super().__init__("conversation_agent")

    async def run_workflow(self, context: AgentContext) -> Dict[str, Any]:
        """
        사용자의 초기 질문을 바탕으로 대화를 통해 제품 요구사항을 수집합니다.
        (실제 채팅 기능은 추후 구현)
        """
        initial_query = context.get("initial_query")
        if not initial_query:
            self.logger.error("Initial query not provided in the context.")
            context.set("error", "Initial query is required.")
            return

        prompt = f"""
        You are a product planning conversation specialist. A user has provided the following initial request. 
        Your task is to interpret this request and formulate a structured summary of product requirements.

        **User's Initial Request:**
        "{initial_query}"

        **Instructions:**
        1.  **Identify Core Goal:** What is the main objective the user wants to achieve?
        2.  **Extract Key Features:** List the key features or functionalities mentioned or implied.
        3.  **Clarifying Questions:** Formulate 3-5 important follow-up questions to gather more details.
        
        Provide the output in a structured JSON format.
        """
        
        try:
            result_str = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4, response_format={"type": "json_object"}))
            requirements = json.loads(result_str)
            requirements["status"] = "collected_successfully"
            context.set("requirements", requirements)
            return requirements
        except Exception as e:
            self.logger.error("Error collecting requirements: %s", e, exc_info=True)
            error_result = {
                "error": str(e),
                "status": "collection_failed"
            }
            context.set("error", error_result)
            return error_result 