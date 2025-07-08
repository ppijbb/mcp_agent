#!/usr/bin/env python3
"""
Product Planner Agent
"""

import asyncio
import re
from urllib.parse import unquote
from srcs.core.agent.base import BaseAgent, AgentContext
from srcs.product_planner_agent.utils.status_logger import StatusLogger
from mcp_agent.logging.logger import get_logger

# Import sub-agents
from .agents.figma_analyzer_agent import FigmaAnalyzerAgent
from .agents.prd_writer_agent import PRDWriterAgent
from .agents.business_planner_agent import BusinessPlannerAgent
from .agents.market_research_agent import MarketResearchAgent
from .agents.kpi_analyst_agent import KPIAnalystAgent
from .agents.operations_agent import OperationsAgent
from .agents.marketing_strategist_agent import MarketingStrategistAgent
from .agents.project_manager_agent import ProjectManagerAgent
from .agents.notion_document_agent import NotionDocumentAgent

logger = get_logger("product_planner_agent")

class ProductPlannerAgent(BaseAgent):
    """
    The main agent for the product planning workflow.
    """

    def __init__(self):
        super().__init__("product_planner_agent")

    def _parse_figma_url(self, url: str) -> tuple[str | None, str | None]:
        """
        Extracts file_id and node_id from a Figma URL.
        """
        file_id_match = re.search(r'figma\.com/file/([^/]+)', url)
        file_id = file_id_match.group(1) if file_id_match else None
        
        node_id_match = re.search(r'node-id=([^&]+)', url)
        node_id = unquote(node_id_match.group(1)) if node_id_match else None
        
        return file_id, node_id

    async def run_workflow(self, context: AgentContext):
        """
        Runs the product planning workflow.
        """
        figma_url = context.get("figma_url")
        if not figma_url:
            self.logger.error("Figma URL not provided in the context.")
            context.set("error", "Figma URL is required.")
            return

        file_id, node_id = self._parse_figma_url(figma_url)
        if not file_id or not node_id:
            self.logger.error(f"Invalid Figma URL: {figma_url}")
            context.set("error", "Invalid Figma URL.")
            return

        figma_api_key = self.settings.get("figma.api_key")
        if not figma_api_key:
            self.logger.error("Figma API key not configured.")
            context.set("error", "Figma API key is not configured.")
            return
            
        self.logger.info("🚀 Starting Product Planner Workflow...")

        workflow_steps = [
            "Figma Design Analysis",
            "Market Research",
            "Product Requirements Document (PRD)",
            "Business Planning",
            "KPI Analysis",
            "Operations Planning",
            "Marketing Strategy",
            "Project Management",
            "Final Report Generation"
        ]
        status_logger = StatusLogger(steps=workflow_steps)
        context.set("status_logger", status_logger)

        try:
            # Instantiate sub-agents
            figma_analyzer = FigmaAnalyzerAgent()
            market_research = MarketResearchAgent()
            prd_writer = PRDWriterAgent()
            business_planner = BusinessPlannerAgent()
            kpi_analyst = KPIAnalystAgent()
            operations_planner = OperationsAgent()
            marketing_strategist = MarketingStrategistAgent()
            project_manager = ProjectManagerAgent()
            notion_document_agent = NotionDocumentAgent()

            # 1. Figma Design Analysis
            status_logger.update_status("Figma Design Analysis", "in_progress")
            context.set("figma_api_key", figma_api_key)
            context.set("figma_file_id", file_id)
            context.set("figma_node_id", node_id)
            await figma_analyzer.run_workflow(context)
            status_logger.update_status("Figma Design Analysis", "completed")
            self.logger.info("✅ Figma Design Analysis Completed")

            # 2. Market Research
            status_logger.update_status("Market Research", "in_progress")
            context.set("product_context", context.get("figma_analysis"))
            await market_research.run_workflow(context)
            status_logger.update_status("Market Research", "completed")
            self.logger.info("✅ Market Research Completed")

            # 3. PRD 작성
            status_logger.update_status("Product Requirements Document (PRD)", "in_progress")
            await prd_writer.run_workflow(context)
            status_logger.update_status("Product Requirements Document (PRD)", "completed")
            self.logger.info("✅ Product Requirements Document (PRD) Completed")

            # 4. 비즈니스 기획
            status_logger.update_status("Business Planning", "in_progress")
            context.set("prd_content", context.get("prd_data"))
            await business_planner.run_workflow(context)
            status_logger.update_status("Business Planning", "completed")
            self.logger.info("✅ Business Planning Completed")
            
            # 5, 6, 7. Run KPI, Operations, and Marketing in parallel
            status_logger.update_status("KPI Analysis", "in_progress")
            status_logger.update_status("Operations Planning", "in_progress")
            status_logger.update_status("Marketing Strategy", "in_progress")
            
            kpi_task = kpi_analyst.run_workflow(context)
            operations_task = operations_planner.run_workflow(context)
            marketing_task = marketing_strategist.run_workflow(context)
            
            await asyncio.gather(kpi_task, operations_task, marketing_task)
            
            status_logger.update_status("KPI Analysis", "completed")
            self.logger.info("✅ KPI Analysis Completed")
            status_logger.update_status("Operations Planning", "completed")
            self.logger.info("✅ Operations Planning Completed")
            status_logger.update_status("Marketing Strategy", "completed")
            self.logger.info("✅ Marketing Strategy Completed")

            # 8. Project Management
            status_logger.update_status("Project Management", "in_progress")
            await project_manager.run_workflow(context)
            status_logger.update_status("Project Management", "completed")
            self.logger.info("✅ Project Management Completed")

            # 9. Final Report Generation
            status_logger.update_status("Final Report Generation", "in_progress")
            await notion_document_agent.run_workflow(context)
            status_logger.update_status("Final Report Generation", "completed")
            self.logger.info("✅ Final Report Generation Completed")

            self.logger.info("🎉 Product Planner Workflow Completed Successfully!")
            context.set("status", "completed")

        except Exception as e:
            self.logger.error(f"💥 Workflow execution failed: {e}", exc_info=True)
            current_step = next((step for step, status in status_logger.get_status().items() if status == "in_progress"), None)
            if current_step:
                status_logger.update_status(current_step, "failed")
            context.set("error", str(e))
            context.set("status", "failed") 