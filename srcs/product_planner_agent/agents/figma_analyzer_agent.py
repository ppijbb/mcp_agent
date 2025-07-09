"""
Figma Analyzer Agent
Figma 디자인을 분석하여 구조화된 요구사항을 추출하는 전문 Agent
"""

# ---------------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------------
import base64
import json

from mcp_agent.context import AgentContext
from srcs.core.agent.base import BaseAgent
from srcs.core.errors import ConfigError, APIError

from srcs.product_planner_agent.utils import env_settings as env


# ---------------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------------

# logger = get_logger("figma_analyzer_agent") # This line is removed as per the new_code.


class FigmaAnalyzerAgent(BaseAgent):
    """Figma 디자인 분석 전문 Agent"""

    def __init__(self):
        super().__init__("figma_analyzer_agent")

    def _get_instruction(self, figma_url: str) -> str:
        return f"""
        You are a Figma design analyst. Analyze the Figma design at: {figma_url}

        **CRITICAL: First validate the Figma URL accessibility before proceeding.**
        
        If the URL is accessible, provide comprehensive analysis on:
        
        1. **Design Structure Analysis**:
           - Page hierarchy and organization
           - Component library usage
           - Layout patterns (grid, flexbox, etc.)
           - Responsive design considerations
        
        2. **UI/UX Analysis**:
           - User interface patterns and interactions
           - Navigation flow and user journey
           - Accessibility considerations
           - Design system consistency
        
        3. **Technical Requirements**:
           - Required frontend technologies
           - Component architecture recommendations  
           - Integration points and APIs needed
           - Performance considerations
        
        4. **Content Analysis**:
           - Text content and copy requirements
           - Image and media requirements
           - Data structure needs
        
        **Output Format**: Provide structured analysis in markdown format.
        **Validation**: If URL is not accessible, clearly state the limitation and provide analysis based on URL structure and common Figma patterns.
        
        Example for inaccessible URL:
        "The Figma URL [URL] is not accessible. Based on the URL structure, this appears to be a [file/prototype]. A typical Figma file includes pages, frames, and components. Analysis is limited without access."
        """

    async def run_workflow(self, context: AgentContext) -> dict:
        """Figma 디자인을 분석해 PRD 작성을 위한 구조화 데이터를 반환합니다."""
        figma_api_key = context.get("figma_api_key") or env.get("FIGMA_API_KEY")
        figma_file_id = context.get("figma_file_id") or env.get("FIGMA_FILE_ID")
        figma_node_id = context.get("figma_node_id") or env.get("FIGMA_NODE_ID")

        if not all([figma_api_key, figma_file_id, figma_node_id]):
            raise ConfigError(
                "FIGMA_API_KEY, FIGMA_FILE_ID, FIGMA_NODE_ID must be provided via args or environment variables."
            )

        self.logger.info("🎨 Figma 분석 요청 수신 – file_id=%s node_id=%s", figma_file_id, figma_node_id)

        try:
            # The original implementation had a mock result.
            # In a real scenario, you would use an LLM with the instruction.
            # For now, we keep the mock result to avoid breaking the flow.
            
            self.logger.info("🎨 Starting Figma analysis (mock) – file_id=%s node_id=%s", figma_file_id, figma_node_id)

            analysis_result = {
                "file_id": figma_file_id,
                "node_id": figma_node_id,
                "design_structure": {
                    "pages": 3,
                    "components": 42,
                },
                "ui_ux_findings": [
                    "Consistent design tokens detected",
                    "Primary navigation uses hamburger menu on mobile",
                ],
                "technical_requirements": {
                    "frontend": "React",
                    "design_system": "Material",
                },
                "status": "analysis_limited",
                "note": "Returned mock data – Figma API integration pending.",
            }
            
            context.set("analysis_result", analysis_result)
            return analysis_result

        except Exception as exc:
            raise APIError(f"Figma analysis failed: {exc}") from exc 