"""
Operations Agent
서비스 운영, 인프라 계획, 고객 지원 및 비즈니스 운영 전략을 관리하는 Agent
"""

from srcs.core.agent.base import BaseAgent, AgentContext
from srcs.core.errors import APIError, WorkflowError
from typing import Dict, Any
import json
from mcp_agent.workflows.llm.augmented_llm import RequestParams


class OperationsAgent(BaseAgent):
    """서비스 운영 및 비즈니스 운영 전문 Agent"""
    
    def __init__(self):
        super().__init__("operations_agent")

    async def run_workflow(self, context: AgentContext) -> Dict[str, Any]:
        """
        PRD의 기술 사양을 바탕으로 서비스 운영 계획을 수립합니다.
        """
        prd_content = context.get("prd_content")
        if not prd_content:
            raise WorkflowError("PRD content is required.")

        technical_spec = prd_content.get("technical_specifications", {})

        prompt = f"""
        You are a senior operations manager. Based on the provided technical specifications from a PRD, create a service operations plan.

        **Technical Specifications:**
        {json.dumps(technical_spec, indent=2, ensure_ascii=False)}

        **Instructions:**
        1.  **Infrastructure Plan:** Recommend a cloud infrastructure setup (e.g., cloud provider, key services, architecture).
        2.  **Monitoring Plan:** Suggest tools and strategies for monitoring system health and performance.
        3.  **Customer Support Plan:** Outline a basic customer support process and required tools.
        4.  **Deployment Plan:** Describe a CI/CD pipeline strategy.

        Provide the output in a structured JSON format.
        """
        
        try:
            result_str = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.5, response_format={"type": "json_object"}))
            operations_plan = json.loads(result_str)
            operations_plan["status"] = "created_successfully"
            context.set("operations_plan", operations_plan)
            return operations_plan
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Failed to decode LLM response: {e}") from e
        except Exception as e:
            raise APIError(f"Failed to plan operations: {e}") from e 