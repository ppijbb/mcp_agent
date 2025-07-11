#!/usr/bin/env python3
"""
Product Planner Agent
"""
import asyncio
import re
from urllib.parse import unquote
from typing import Any, Dict, Optional
from datetime import datetime
import json

from srcs.core.agent.base import BaseAgent
from srcs.product_planner_agent.agents.figma_analyzer_agent import FigmaAnalyzerAgent
from srcs.product_planner_agent.agents.prd_writer_agent import PRDWriterAgent
from srcs.product_planner_agent.coordinators.reporting_coordinator import ReportingCoordinator
from srcs.product_planner_agent.utils.logger import get_product_planner_logger
from srcs.common.utils import get_gen_client

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
        
        # Add state management for conversational mode
        self.state = {
            "step": "init",
            "data": {
                "product_concept": None,
                "user_persona": None,
                "figma_file_id": None,
                "figma_analysis": None,
                "prd_draft": None,
                "final_report": None
            },
            "history": []
        }

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

    def _extract_figma_ids(self, figma_url: str) -> tuple[str, str]:
        """Extracts Figma file ID and node ID from a Figma URL."""
        try:
            # Remove query parameters and fragment
            url_path = unquote(figma_url).split('?', 1)[0].split('#', 1)[0]
            
            # Extract file ID and node ID
            file_id_match = re.search(r'/file/([a-zA-Z0-9_-]+)', url_path)
            node_id_match = re.search(r'/node/([a-zA-Z0-9_-]+)', url_path)

            file_id = file_id_match.group(1) if file_id_match else None
            node_id = node_id_match.group(1) if node_id_match else None

            if not file_id:
                raise ValueError("Could not extract Figma file ID from URL.")

            return file_id, node_id
        except Exception as e:
            logger.error(f"Error extracting Figma IDs from URL {figma_url}: {e}", exc_info=True)
            raise

    async def process_message(self, user_message: str) -> Dict[str, Any]:
        """Process a user message and advance the planning state."""
        self.state["history"].append({"role": "user", "content": user_message})
        response = {"message": "", "state": self.state["step"]}
        
        try:
            if self.state["step"] == "init":
                # Parse initial inputs from message or ask for them
                # For simplicity, assume message contains JSON with product_concept and user_persona
                try:
                    inputs = json.loads(user_message)
                    self.state["data"]["product_concept"] = inputs.get("product_concept")
                    self.state["data"]["user_persona"] = inputs.get("user_persona")
                    self.state["data"]["figma_url"] = inputs.get("figma_url")
                    if self.state["data"]["figma_url"]:
                        figma_file_id, node_id = self._extract_figma_ids(self.state["data"]["figma_url"])
                        self.state["data"]["figma_file_id"] = figma_file_id
                        self.state["data"]["figma_node_id"] = node_id
                except json.JSONDecodeError:
                    response["message"] = "Please provide product concept, user persona, and optional Figma URL in JSON format."
                    return response
                
                if not self.state["data"]["product_concept"] or not self.state["data"]["user_persona"]:
                    response["message"] = "Product concept and user persona are required."
                    return response
                
                self.state["step"] = "figma_analysis"
                response["message"] = "Starting product planning. Analyzing Figma if provided..."

            if self.state["step"] == "figma_analysis" and self.state["data"]["figma_file_id"]:
                logger.info(f"Analyzing Figma file with ID: {self.state['data']['figma_file_id']}")
                figma_context = {}  # Use self.state["data"] directly in sub-agent if needed
                analysis_result = await self.figma_analyzer.run_workflow(figma_context)
                self.state["data"]["figma_analysis"] = analysis_result
                logger.info("Figma analysis completed.")
                response["message"] += "\nFigma analysis complete."
                self.state["step"] = "prd_drafting"
            
            if self.state["step"] == "figma_analysis" and not self.state["data"]["figma_file_id"]:
                self.state["data"]["figma_analysis"] = {"status": "skipped"}
                self.state["step"] = "prd_drafting"
            
            if self.state["step"] == "prd_drafting":
                logger.info("Drafting PRD...")
                prd_context = self.state["data"]
                prd_result = await self.prd_writer.run_workflow(prd_context)
                self.state["data"]["prd_draft"] = prd_result
                logger.info("PRD drafting completed.")
                response["message"] += "\nPRD draft complete. Generating final report..."
                self.state["step"] = "report_generation"
            
            if self.state["step"] == "report_generation":
                logger.info("Generating final report...")
                report_context = self.state["data"]
                final_report = await self.reporting_coordinator.generate_final_report(report_context)
                self.state["data"]["final_report"] = final_report
                logger.info("Final report generation completed.")
                response["message"] += "\nFinal report generated."
                self.state["step"] = "save_report"
            
            if self.state["step"] == "save_report":
                save_status = await self._save_final_report(self.state["data"]["final_report"], self.state["data"]["product_concept"])
                self.state["data"]["final_report"]["save_status"] = save_status
                response["message"] += "\nReport saved to Google Drive."
                self.state["step"] = "complete"
            
            if self.state["step"] == "complete":
                response["message"] += "\nPlanning complete!"
                response["final_report"] = self.state["data"]["final_report"]
            
            self.state["history"].append({"role": "assistant", "content": response["message"]})
            return response
        
        except Exception as e:
            logger.error(f"Error in process_message: {str(e)}")
            response["message"] = f"Error: {str(e)}"
            return response

    def get_state(self) -> Dict[str, Any]:
        """Get current state for serialization."""
        return self.state

    def set_state(self, state: Dict[str, Any]):
        """Set state from serialized data."""
        self.state = state
