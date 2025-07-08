"""
Project Manager Agent
개발 일정, 마일스톤, 리소스 할당을 관리하는 Agent
"""

from srcs.core.agent.base import BaseAgent, AgentContext
from srcs.core.errors import APIError, WorkflowError
from typing import Dict, Any
import json
from mcp_agent.workflows.llm.augmented_llm import RequestParams


class ProjectManagerAgent(BaseAgent):
    """프로젝트 관리 및 일정 계획 전문 Agent"""
    
    def __init__(self):
        super().__init__("project_manager_agent")

    async def run_workflow(self, context: AgentContext) -> Dict[str, Any]:
        """
        PRD와 비즈니스 계획을 바탕으로 개발 로드맵, 스프린트 계획, 리소스 할당안을 포함하는 프로젝트 계획을 수립합니다.
        """
        prd_content = context.get("prd_content")
        business_plan = context.get("business_plan")

        if not prd_content or not business_plan:
            raise WorkflowError("PRD content and business plan are required.")

        prompt = f"""
        You are a senior project manager. Based on the provided PRD and business plan, create a comprehensive project plan.

        **PRD Content:**
        {json.dumps(prd_content, indent=2, ensure_ascii=False)}

        **Business Plan:**
        {json.dumps(business_plan, indent=2, ensure_ascii=False)}

        **Instructions:**
        1.  **Development Roadmap:** Create a high-level roadmap for the next 6-12 months.
        2.  **Sprint Plan:** Suggest a sprint structure (e.g., duration, key ceremonies).
        3.  **Resource Allocation:** Estimate the required team size and roles.
        4.  **Risk Assessment:** Identify potential risks and suggest mitigation strategies.

        Provide the output in a structured JSON format.
        """
        
        try:
            result_str = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.5, response_format={"type": "json_object"}))
            project_plan = json.loads(result_str)
            project_plan["status"] = "created_successfully"
            context.set("project_plan", project_plan)
            return project_plan
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Failed to decode LLM response: {e}") from e
        except Exception as e:
            raise APIError(f"Failed to create project plan: {e}") from e 