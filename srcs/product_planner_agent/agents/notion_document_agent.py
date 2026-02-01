"""
Notion Document Agent
노션 문서 작성, 지식 베이스 구축 및 협업 문서 워크플로우를 관리하는 Agent
"""

from typing import Dict, Any
import json
from mcp_agent.workflows.llm.augmented_llm import RequestParams

from srcs.core.agent.base import BaseAgent
from srcs.core.errors import APIError


class NotionDocumentAgent(BaseAgent):
    """노션 문서 관리 및 지식 베이스 구축 전문 Agent"""

    def __init__(self):
        super().__init__("notion_document_agent")

    async def run_workflow(self, context: Any) -> Dict[str, Any]:
        """
        모든 에이전트의 결과물을 종합하여 Notion에 저장할 최종 보고서를 생성합니다.
        (실제 Notion API 연동은 추후 구현)
        """
        all_results = context.get_all()

        final_report_content = "## 📝 Product Plan Final Report\n\n"
        for key, value in all_results.items():
            final_report_content += f"### {key.replace('_', ' ').title()}\n\n"
            if isinstance(value, (dict, list)):
                final_report_content += f"```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```\n\n"
            else:
                final_report_content += f"{str(value)}\n\n"

        prompt = f"""
        You are a documentation specialist. Based on the provided comprehensive results from multiple agents, create a final, well-structured summary report suitable for a Notion page.

        **Aggregated Data:**
        {final_report_content}

        **Instructions:**
        1.  **Executive Summary:** Write a brief, insightful executive summary.
        2.  **Structure the Content:** Organize the data logically with clear headings (e.g., Business Plan, Marketing Strategy, Project Plan).
        3.  **Key Highlights:** Extract and list the most important decisions and plans in a "Key Highlights" section.
        4.  **Next Steps:** Define clear, actionable next steps for the project team.

        Provide the output as a single, well-formatted markdown string.
        """

        try:
            final_report = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4))
            # 실제 Notion 페이지 생성 대신, 결과물과 목업 URL 반환
            result = {
                "notion_page_content": final_report,
                "notion_page_url": "https://www.notion.so/mock-page-url-generated",
                "status": "created_successfully"
            }
            context.set("final_report", result)
            return result
        except Exception as e:
            raise APIError(f"Failed to create project workspace: {e}") from e
