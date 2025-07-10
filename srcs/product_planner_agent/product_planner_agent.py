#!/usr/bin/env python3
"""
Product Planner Agent
"""
import asyncio
import re
from urllib.parse import unquote
from typing import Any, Dict
from datetime import datetime
import json

from srcs.core.agent.base import BaseAgent
from srcs.product_planner_agent.agents.figma_analyzer_agent import FigmaAnalyzerAgent
from srcs.product_planner_agent.agents.prd_writer_agent import PRDWriterAgent
from srcs.product_planner_agent.coordinators.reporting_coordinator import ReportingCoordinator
from srcs.product_planner_agent.utils.logger import get_product_planner_logger

logger = get_product_planner_logger("main_agent")


class ProductPlannerAgent(BaseAgent):
    """
    Coordinates the entire product planning process by orchestrating various sub-agents.
    This version is refactored to be simpler and delegate tasks to specialized agents
    and coordinators, following the new architecture.
    """

    def __init__(self):
        super().__init__("product_planner_agent")
        # Sub-agents are initialized here, but their LLM dependencies are handled by the app context.
        self.figma_analyzer = FigmaAnalyzerAgent()
        self.prd_writer = PRDWriterAgent()
        self.reporting_coordinator = ReportingCoordinator()
        logger.info("ProductPlannerAgent and its sub-components initialized.")

    async def _save_final_report(self, report_data: Dict[str, Any], product_concept: str) -> Dict[str, Any]:
        """Saves the final report to Google Drive using the 'gdrive' MCP server."""
        logger.info("💾 Saving final report to Google Drive via MCP...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sanitize product_concept for use in a filename
            safe_concept_name = re.sub(r'[\\/*?:"<>|]', "", product_concept)[:50]
            file_name = f"Final_Report_{safe_concept_name}_{timestamp}.json"
            
            report_content = json.dumps(report_data, indent=2, ensure_ascii=False)

            # Use the tool provided by BaseAgent's MCPApp instance
            response = await self.app.tools.gdrive.upload_file(
                file_name=file_name,
                content=report_content,
                mime_type="application/json"
            )
            
            if not response or not response.get("success"):
                raise Exception(f"MCP upload failed. Response: {response}")

            file_id = response.get("fileId")
            logger.info(f"✅ Final report saved successfully. File ID: {file_id}")
            return {"status": "success", "drive_file_id": file_id}
        except Exception as e:
            logger.error(f"❌ Failed to save final report to Google Drive: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    async def run_workflow(self, context: Any) -> Dict[str, Any]:
        """
        Executes the product planning workflow from concept to final report.
        """
        product_concept = context.get("product_concept")
        user_persona = context.get("user_persona")
        figma_file_id = context.get("figma_file_id")

        logger.info(f"Starting workflow for product concept: '{product_concept[:50]}...'")

        try:
            # 1. Analyze Figma file if provided
            if figma_file_id:
                logger.info(f"Analyzing Figma file with ID: {figma_file_id}")
                figma_context = context.copy()
                figma_context["figma_file_id"] = figma_file_id
                analysis_result = await self.figma_analyzer.run_workflow(figma_context)
                context["figma_analysis"] = analysis_result
                logger.info("Figma analysis completed.")
            else:
                logger.info("No Figma file ID provided, skipping analysis.")
                context["figma_analysis"] = {"status": "skipped", "reason": "No Figma file ID provided"}

            # 2. Draft Product Requirements Document (PRD)
            logger.info("Drafting PRD...")
            prd_context = context.copy()
            # Pass all necessary information to the sub-context
            prd_context["product_concept"] = product_concept
            prd_context["user_persona"] = user_persona
            prd_context["figma_analysis"] = context.get("figma_analysis")
            prd_result = await self.prd_writer.run_workflow(prd_context)
            context["prd_draft"] = prd_result
            logger.info("PRD drafting completed.")

            # 3. Generate the Final Report using the Reporting Coordinator
            logger.info("Generating final report...")
            report_context = context.copy()
            report_context["product_concept"] = product_concept
            report_context["user_persona"] = user_persona
            report_context["prd_draft"] = prd_result
            final_report = await self.reporting_coordinator.generate_final_report(report_context)
            context["final_report"] = final_report
            logger.info("Final report generation completed.")

            # 4. Save the final report to Google Drive
            save_status = await self._save_final_report(final_report, product_concept)
            final_report["save_status"] = save_status

            logger.info("🎉 Product Planner Workflow Completed Successfully!")
            return final_report

        except Exception as e:
            logger.critical(f"💥 Workflow execution failed: {e}", exc_info=True)
            context.set("error", str(e))
            context.set("status", "failed")
            # Re-raise the exception to be handled by the calling script
            raise
