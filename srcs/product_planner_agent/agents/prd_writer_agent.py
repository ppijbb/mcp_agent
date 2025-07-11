"""
PRD Writer Agent
디자인 분석 결과를 바탕으로 전문적인 제품 요구사항 문서를 작성하는 Agent
"""

import json
from datetime import datetime
from typing import Any, Dict

from mcp_agent.logging.logger import get_logger
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from srcs.core.agent.base import BaseAgent
from srcs.core.errors import APIError, WorkflowError
from srcs.product_planner_agent.utils.logger import get_product_planner_logger

logger = get_product_planner_logger(__name__)

class PRDWriterAgent(BaseAgent):
    """
    Agent responsible for drafting the PRD document based on various inputs.
    It uses the application context for LLM and MCP client access.
    """

    def __init__(self, **kwargs):
        super().__init__("prd_writer_agent", **kwargs)
        logger.info("PRDWriterAgent initialized.")
        # All dependencies like logger, llm, and mcp_client are accessed
        # via self or self.app, provided by the BaseAgent and app context.

    async def run_workflow(self, context: Any) -> Dict[str, Any]:
        """
        Drafts the PRD using the product brief and feedback from the context.
        """
        logger.info("🖊️ Starting PRD generation workflow...")
        product_brief = context.get("product_brief", {})
        feedback = context.get("feedback")
        figma_context = context.get("figma_analysis")
        
        logger.info(f"Starting PRD draft for product: '{product_brief.get('product_name', 'UntitledProduct')[:50]}...'")
        
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

        3.  **Additional Feedback** (if available):
            ```
            {feedback if feedback else "No additional feedback provided."}
            ```

        **PRD Structure**:
        Your PRD must follow this structure precisely. Fill in every section with relevant details derived from the provided context.
        - **1. Introduction**: (Problem Statement, Goal, Target Audience)
        - **2. Product Requirements**: (User Stories, Functional Requirements, Non-Functional Requirements)
        - **3. Design & UX**: (Design Mockups, User Flow)
        - **4. Assumptions and Constraints**:
        
        Generate the PRD in a structured JSON format.
        """
        try:
            prd_json_str = await self.app.llm.generate_str(
                message=prompt,
                request_params=RequestParams(
                    model="gemini-1.5-flash-latest",
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
            )
            prd_data = json.loads(prd_json_str)
            context["prd_data"] = prd_data
            
            # After creating the PRD, save it using the dedicated method.
            save_result = await self._save_prd(context)
            prd_data["gdrive_metadata"] = save_result

            return prd_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode LLM response: {e}")
            raise WorkflowError(f"Failed to decode LLM response: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during PRD generation: {e}")
            raise APIError(f"Failed to generate PRD: {e}") from e


    async def _save_prd(self, context: Any) -> Dict[str, Any]:
        """Saves the PRD file to Google Drive using the 'gdrive' MCP server."""
        logger.info("💾 Saving PRD to Google Drive via MCP...")
        prd_content = context.get("prd_content")
        file_name = f"PRD_{prd_content.get('product_name', 'UntitledProduct')}.json"
        
        if not prd_content:
            raise WorkflowError("Cannot save PRD: prd_content not found in context.")

        try:
            # The framework will route this to the 'gdrive' MCP's 'upload_file' tool.
            response = await self.app.tools.gdrive.upload_file(
                file_name=file_name,
                content=json.dumps(prd_content, indent=2),
                mime_type="application/json"
            )
            logger.info(f"Successfully saved PRD to Google Drive. File ID: {response.get('fileId')}")
            return {"status": "success", "drive_file_id": response.get("fileId")}
        except Exception as e:
            logger.error(f"Failed to save PRD to Google Drive: {e}", exc_info=True)
            raise APIError(f"Failed to save PRD file to Google Drive: {e}") from e

    async def _refine_prd_section(self, context: Any) -> Dict[str, Any]:
        """Refines a specific PRD section based on additional requirements."""
        section_name = context.get("section_name")
        current_content = context.get("current_content")
        additional_requirements = context.get("additional_requirements")
        
        logger.info(f"🔄 Refining PRD section: {section_name}")
        
        if not all([section_name, current_content, additional_requirements]):
            raise WorkflowError("Missing required data (section_name, current_content, additional_requirements) for refinement.")
            
        prompt = f"""
        Refine the following PRD section based on additional requirements.
        Section: {section_name}
        Current Content: {current_content}
        Additional Requirements: {additional_requirements}
        Return the refined section content.
        """
        
        try:
            # Using self.app.llm, inherited from BaseAgent
            refined_content = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4))
            return {
                "section_name": section_name,
                "refined_content": refined_content,
                "status": "refined"
            }
        except Exception as e:
            logger.error(f"Section refinement failed: {e}", exc_info=True)
            raise APIError(f"Section refinement failed: {e}") from e

    async def _validate_prd_completeness(self, context: Any) -> Dict[str, Any]:
        """Validates the completeness and quality of the PRD."""
        prd_content = context.get("prd_content")
        logger.info("Validating PRD completeness...")
        
        if not prd_content:
            raise WorkflowError("Cannot validate PRD: prd_content not found in context.")
            
        prompt = f"""
        As a senior product manager, please validate the following PRD for completeness and quality.
        PRD Content: {json.dumps(prd_content, indent=2)}
        Check for required sections, content quality, and consistency.
        Provide a validation report with scores (1-10) and specific recommendations.
        """
        
        try:
            # Using self.app.llm, inherited from BaseAgent
            validation_result = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.1))
            return {
                "validation_report": validation_result,
                "status": "validated"
            }
        except Exception as e:
            logger.error(f"PRD validation failed: {e}", exc_info=True)
            raise APIError(f"PRD validation failed: {e}") from e 