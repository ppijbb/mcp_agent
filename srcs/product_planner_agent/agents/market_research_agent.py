"""
Market Research Agent
시장 조사 및 트렌드 분석을 수행하는 전문 Agent
"""

from mcp_agent.agents.agent import Agent
from typing import Dict, Any
import json
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.logging.logger import get_logger

logger = get_logger("market_research_agent")


class MarketResearchAgent:
    """시장 조사 전문 Agent"""

    def __init__(self, llm=None):
        self.llm = llm
        self.agent_instance = self.create_agent()

    async def analyze_market(self, product_context: Dict[str, Any]) -> Dict[str, Any]:
        """제품 컨텍스트를 기반으로 시장 규모, 성장률, 트렌드를 분석합니다."""
        if not self.llm:
            # LLM이 없는 경우 목업 데이터
            return {
                "tam": "$5B",
                "sam": "$1B",
                "som": "$200M",
                "growth_rate": "20% YoY",
                "key_trends": ["AI 개인화", "모바일 우선"],
                "status": "analysis_mockup"
            }

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
            result_str = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4, response_format="json"))
            data = json.loads(result_str)
            data["status"] = "analysis_successful"
            return data
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            return {"error": str(e), "status": "analysis_failed"}

    @staticmethod
    def create_agent() -> Agent:
        """Agent 인스턴스 생성"""
        instruction = """
        You are a market research expert. Gather and synthesize market data, trends, and sizing for product planning.
        """
        return Agent(name="market_research", instruction=instruction, server_names=["fetch", "filesystem"]) 