"""
Marketing Strategist Agent
마케팅 전략 수립, Go-to-Market 계획 및 사용자 획득 전략을 관리하는 Agent
"""

from srcs.core.agent.base import BaseAgent, AgentContext
from srcs.core.errors import APIError, WorkflowError
from typing import Dict, Any
import json
from mcp_agent.workflows.llm.augmented_llm import RequestParams


class MarketingStrategistAgent(BaseAgent):
    """마케팅 전략 및 사용자 획득 전문 Agent"""
    
    def __init__(self):
        super().__init__("marketing_strategist_agent")

    async def run_workflow(self, context: AgentContext) -> Dict[str, Any]:
        """
        PRD와 비즈니스 계획을 바탕으로 Go-to-Market(GTM) 전략을 수립합니다.
        """
        prd_content = context.get("prd_content")
        business_plan = context.get("business_plan")

        if not prd_content or not business_plan:
            raise WorkflowError("PRD content and business plan are required.")

        prompt = f"""
        You are a senior marketing strategist. Based on the provided PRD and business plan, develop a Go-to-Market (GTM) strategy.

        **PRD Content:**
        {json.dumps(prd_content, indent=2, ensure_ascii=False)}

        **Business Plan:**
        {json.dumps(business_plan, indent=2, ensure_ascii=False)}

        **Instructions:**
        1.  **Target Audience:** Define the primary target audience and key personas.
        2.  **Positioning:** Create a compelling product positioning statement.
        3.  **Channel Strategy:** Recommend the most effective marketing channels (e.g., social media, content marketing, SEO).
        4.  **Launch Campaign:** Outline a creative concept for the initial launch campaign.

        Provide the output in a structured JSON format.
        """
        
        try:
            result_str = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.6, response_format={"type": "json_object"}))
            marketing_strategy = json.loads(result_str)
            marketing_strategy["status"] = "created_successfully"
            context.set("marketing_strategy", marketing_strategy)
            return marketing_strategy
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Failed to decode LLM response: {e}") from e
        except Exception as e:
            raise APIError(f"Failed to develop marketing strategy: {e}") from e 