"""
Market Research Agent
시장 조사 및 트렌드 분석을 수행하는 전문 Agent
"""

from typing import Dict, Any
import json
from mcp_agent.workflows.llm.augmented_llm import RequestParams

from srcs.core.agent.base import BaseAgent
from srcs.core.errors import APIError, WorkflowError


class MarketResearchAgent(BaseAgent):
    """시장 조사 전문 Agent"""

    def __init__(self):
        super().__init__("market_research_agent")

    async def run_workflow(self, context: Any) -> Dict[str, Any]:
        """제품 컨텍스트를 기반으로 시장 규모, 성장률, 트렌드를 분석합니다."""
        product_context = context.get("product_context")
        if not product_context:
            raise WorkflowError("Product context is required.")

        prompt = f"""
        You are a seasoned market analyst. Based on the following product context, perform a comprehensive market analysis.

        **Product Context:**
        {json.dumps(product_context, indent=2, ensure_ascii=False)}

        **Instructions:**
        1.  Estimate TAM, SAM, and SOM.
        2.  Identify annual growth rate.
        3.  List 3-5 key market trends relevant to the product.
        4.  Provide data sources or assumptions.

        Provide the output in structured JSON format.
        """
        try:
            result_str = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4, response_format={"type": "json_object"}))
            data = json.loads(result_str)
            data["status"] = "analysis_successful"
            context.set("market_analysis", data)
            return data
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Failed to decode LLM response: {e}") from e
        except Exception as e:
            raise APIError(f"Market analysis failed: {e}") from e
