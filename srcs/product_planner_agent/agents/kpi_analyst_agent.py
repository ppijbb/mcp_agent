"""
KPI Analyst Agent
핵심 성과 지표 설정, 분석 프레임워크 및 성능 추적 시스템을 관리하는 Agent
"""

from typing import Dict, Any
import json
from mcp_agent.workflows.llm.augmented_llm import RequestParams

from srcs.core.agent.base import BaseAgent
from srcs.core.errors import APIError, WorkflowError


class KPIAnalystAgent(BaseAgent):
    """KPI 설정 및 성과 분석 전문 Agent"""

    def __init__(self):
        super().__init__("kpi_analyst_agent")

    async def run_workflow(self, context: Any) -> Dict[str, Any]:
        """
        PRD와 비즈니스 계획을 바탕으로 제품의 성공을 측정할 핵심 성과 지표(KPI)를 정의합니다.
        """
        prd_content = context.get("prd_content")
        business_plan = context.get("business_plan")

        if not prd_content or not business_plan:
            raise WorkflowError("PRD content and business plan are required.")

        prompt = f"""
        You are a senior data analyst. Based on the provided PRD and business plan, define key performance indicators (KPIs).

        **PRD Content:**
        {json.dumps(prd_content, indent=2, ensure_ascii=False)}

        **Business Plan:**
        {json.dumps(business_plan, indent=2, ensure_ascii=False)}

        **Instructions:**
        1.  **North Star Metric:** Define one primary "North Star" metric that captures the core value of the product.
        2.  **User-centric KPIs:** Define 3-5 KPIs related to user engagement, retention, and satisfaction.
        3.  **Business-centric KPIs:** Define 3-5 KPIs related to revenue, growth, and market position.
        4.  **Measurement Plan:** Briefly describe how each KPI could be measured.

        Provide the output in a structured JSON format.
        """

        try:
            result_str = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.5, response_format={"type": "json_object"}))
            kpi_definition = json.loads(result_str)
            kpi_definition["status"] = "created_successfully"
            context.set("kpi_definition", kpi_definition)
            return kpi_definition
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Failed to decode LLM response: {e}") from e
        except Exception as e:
            raise APIError(f"Failed to define KPIs: {e}") from e
