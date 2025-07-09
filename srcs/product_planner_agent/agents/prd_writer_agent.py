"""
PRD Writer Agent
디자인 분석 결과를 바탕으로 전문적인 제품 요구사항 문서를 작성하는 Agent
"""

from srcs.core.agent.base import BaseAgent
from srcs.core.errors import APIError, WorkflowError
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import aiohttp

from mcp_agent.context import AgentContext
from srcs.product_planner_agent.prompt import PROMPT
from srcs.product_planner_agent.utils.llm_utils import get_llm_factory

class PRDWriterAgent(BaseAgent):
    """
    Agent responsible for drafting the PRD document based on various inputs.
    """
    def __init__(self):
        super().__init__("prd_writer_agent")

    async def _get_figma_summary(self, figma_file_id: str) -> Optional[Dict[str, Any]]:
        # This should be handled by the FigmaAnalyzerAgent.
        # This method is kept for now to show the dependency, but it should be removed
        # and the result should be passed in the context.
        return None

    async def _get_notion_content(self, notion_page_id: str) -> Optional[Dict[str, Any]]:
        # This should be handled by a Notion-specific agent.
        return None

    async def run_workflow(self, context: AgentContext) -> Dict[str, Any]:
        """
        Drafts the PRD using the product brief and feedback from the context.
        """
        product_brief = context.get("product_brief", {})
        feedback = context.get("feedback")
        figma_file_id = context.get("figma_file_id")
        notion_page_id = context.get("notion_page_id")
        figma_context = context.get("figma_analysis") # Assuming this is passed from FigmaAnalyzerAgent

        notion_context = await self._get_notion_content(notion_page_id)
        
        prompt = f"""
        As a Senior Product Manager, your task is to write a detailed Product Requirements Document (PRD).

        Use the following inputs to create the PRD:

        1.  **Core Product Brief**:
            ```json
            {json.dumps(product_brief, indent=2)}
            ```
        
        2.  **Figma Design Prototype Summary**:
            Analyze the following summary of the Figma design. Pay attention to user flows, component names, and screen layouts to define the user experience and functional requirements.
            ```json
            {json.dumps(figma_context, indent=2) if figma_context else "No Figma design context provided."}
            ```

        3.  **Notion Planning Document** (if available):
            Incorporate the background, goals, and technical notes from the following Notion document into the PRD.
            ```text
            {json.dumps(notion_context, indent=2) if notion_context else "No Notion planning context provided."}
            ```

        4.  **Additional Feedback** (if available):
            ```
            {feedback if feedback else "No additional feedback provided."}
            ```

        **PRD Structure**:
        Your PRD must follow this structure precisely. Fill in every section with relevant details derived from the provided context.

        - **1. Introduction**:
          - **1.1. Problem Statement**: What user problem are we solving?
          - **1.2. Goal**: What is the primary business objective? What are the key success metrics (KPIs)?
          - **1.3. Target Audience**: Who are the primary users?
        - **2. Product Requirements**:
          - **2.1. User Stories**: Write detailed user stories (As a [user type], I want to [action] so that [benefit]). Infer these from the product brief, Figma flows, and Notion doc.
          - **2.2. Functional Requirements**: List specific features (e.g., 'User authentication', 'Dashboard view'). Use details from the Figma components and Notion specs.
          - **2.3. Non-Functional Requirements**: (e.g., Performance, Security, Usability).
        - **3. Design & UX**:
          - **3.1. Design Mockups**: Reference the Figma file ID ({figma_file_id or 'N/A'}).
          - **3.2. User Flow**: Describe the high-level user journey, referencing specific frames from the Figma summary.
        - **4. Assumptions and Constraints**: List any assumptions made or technical constraints identified.
        
        Generate the PRD in a structured JSON format.
        """
        try:
            prd_json_str = await self.app.llm.generate_str(
                message=prompt,
                request_params=RequestParams(
                    model="gemini-1.5-flash-latest",
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
            )
            prd_data = json.loads(prd_json_str)
            context.set("prd_data", prd_data)
            return prd_data
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Failed to decode LLM response: {e}") from e

    async def _save_prd(self, context: AgentContext):
        """PRD 파일을 Google Drive에 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"PRD_{timestamp}.md"
        
        upload_url = f"{self.google_drive_mcp_url}/upload"
        payload = {
            "fileName": file_name,
            "content": context.get("prd_data")
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(upload_url, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    if result.get("success"):
                        file_id = result.get("fileId")
                        self.logger.info(f"Successfully uploaded PRD to Google Drive. File ID: {file_id}")
                        return {
                            "drive_file_id": file_id,
                            "file_url": f"https://docs.google.com/document/d/{file_id}",
                            "status": "uploaded",
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        raise APIError(f"MCP upload failed: {result.get('message')}")

        except Exception as e:
            raise APIError(f"Failed to save PRD file to Google Drive: {e}") from e



    async def _refine_prd_section(self, context: AgentContext):
        """특정 PRD 섹션 개선"""
        logger.info(f"🔄 Refining PRD section: {section_name}")
        
        if not self.llm:
            return {"error": "No LLM available for section refinement"}
            
        prompt = f"""
        Refine the following PRD section based on additional requirements.
        
        Section: {section_name}
        Current Content: {current_content}
        Additional Requirements: {additional_requirements}
        
        Improve the section by:
        1. Incorporating the additional requirements
        2. Enhancing clarity and specificity
        3. Ensuring consistency with PRD standards
        4. Adding missing details or considerations
        
        Return the refined section content.
        """
        
        try:
            refined_content = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4))
            return {
                "section_name": section_name,
                "refined_content": refined_content,
                "status": "refined",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Section refinement failed: {e}")
            return {
                "section_name": section_name,
                "status": "refinement_failed",
                "error": str(e),
                "original_content": current_content
            }

    async def _validate_prd_completeness(self, context: AgentContext):
        """PRD 완성도 검증"""
        logger.info("🔍 Validating PRD completeness")
        
        if not self.llm:
            return {"error": "No LLM available for validation"}
            
        prompt = f"""
        Validate the completeness and quality of this PRD.
        
        PRD Content: {prd_content}
        
        Check for:
        1. **Required Sections**: All essential PRD sections present
        2. **Content Quality**: Sufficient detail and clarity
        3. **Consistency**: Coherent throughout the document
        4. **Actionability**: Clear enough for development teams
        5. **Missing Elements**: What's missing or needs improvement
        
        Provide a validation report with scores (1-10) and specific recommendations.
        """
        
        try:
            validation_result = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.3))
            return {
                "validation_report": validation_result,
                "status": "validated",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"PRD validation failed: {e}")
            return {
                "status": "validation_failed",
                "error": str(e),
                "basic_check": {
                    "word_count": len(prd_content.split()),
                    "section_count": prd_content.count('#'),
                    "note": "Basic metrics only - detailed validation unavailable"
                }
            } 