#!/usr/bin/env python3
"""
MCP Server for Local Researcher
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.autonomous_orchestrator import AutonomousOrchestrator
from src.agents.task_analyzer import TaskAnalyzerAgent
from src.agents.task_decomposer import TaskDecomposerAgent
from src.agents.research_agent import ResearchAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.core.mcp_integration import MCPIntegrationManager
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_default_logging

# Setup logging
setup_default_logging()
logger = logging.getLogger(__name__)

class LocalResearcherMCPServer:
    """MCP Server for Autonomous Research System functionality."""
    
    def __init__(self):
        """Initialize the MCP server."""
        self.config_manager = ConfigManager()
        self.mcp_manager = MCPIntegrationManager()
        
        # Initialize specialized agents
        self.task_analyzer = TaskAnalyzerAgent()
        self.task_decomposer = TaskDecomposerAgent()
        self.research_agent = ResearchAgent()
        self.evaluation_agent = EvaluationAgent()
        self.validation_agent = ValidationAgent()
        self.synthesis_agent = SynthesisAgent()
        
        # Initialize orchestrator
        self.orchestrator = AutonomousOrchestrator(
            config_path=None,
            agents={
                'analyzer': self.task_analyzer,
                'decomposer': self.task_decomposer,
                'researcher': self.research_agent,
                'evaluator': self.evaluation_agent,
                'validator': self.validation_agent,
                'synthesizer': self.synthesis_agent
            },
            mcp_manager=self.mcp_manager
        )
        
        self.active_research = {}
        logger.info("Autonomous Research System MCP Server initialized")
    
    async def start_autonomous_research(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start autonomous research with multi-agent orchestration."""
        try:
            # Start autonomous research
            objective_id = await self.orchestrator.start_autonomous_research(user_request, context)
            
            return {
                "success": True,
                "objective_id": objective_id,
                "message": f"Autonomous research started for: {user_request}",
                "user_request": user_request,
                "context": context or {}
            }
        except Exception as e:
            logger.error(f"Failed to start autonomous research: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_research_status(self, objective_id: str) -> Dict[str, Any]:
        """Get the status of an autonomous research objective."""
        try:
            status = await self.orchestrator.get_research_status(objective_id)
            if not status:
                return {
                    "success": False,
                    "error": "Research objective not found"
                }
            
            return {
                "success": True,
                "objective_id": objective_id,
                "status": status.get('status', 'unknown'),
                "user_request": status.get('user_request', ''),
                "created_at": status.get('created_at', ''),
                "analyzed_objectives": status.get('analyzed_objectives', []),
                "decomposed_tasks": status.get('decomposed_tasks', []),
                "assigned_agents": status.get('assigned_agents', []),
                "execution_results": status.get('execution_results', []),
                "evaluation_results": status.get('evaluation_results', {}),
                "validation_results": status.get('validation_results', {}),
                "final_synthesis": status.get('final_synthesis', {})
            }
        except Exception as e:
            logger.error(f"Failed to get research status: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_research(self) -> Dict[str, Any]:
        """List all autonomous research objectives."""
        try:
            objectives = await self.orchestrator.list_research()
            return {
                "success": True,
                "objectives": objectives,
                "total_count": len(objectives)
            }
        except Exception as e:
            logger.error(f"Failed to list research: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def cancel_research(self, objective_id: str) -> Dict[str, Any]:
        """Cancel an autonomous research objective."""
        try:
            success = await self.orchestrator.cancel_research(objective_id)
            return {
                "success": success,
                "message": f"Research objective {objective_id} {'cancelled' if success else 'not found'}"
            }
        except Exception as e:
            logger.error(f"Failed to cancel research: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_report_content(self, objective_id: str) -> Dict[str, Any]:
        """Get the content of an autonomous research deliverable."""
        try:
            status = await self.orchestrator.get_research_status(objective_id)
            if not status:
                return {
                    "success": False,
                    "error": "Research objective not found"
                }
            
            final_synthesis = status.get('final_synthesis', {})
            deliverable_path = final_synthesis.get('deliverable_path')
            
            if not deliverable_path:
                return {
                    "success": False,
                    "error": "Deliverable not found or not completed"
                }
            
            report_path = Path(deliverable_path)
            if not report_path.exists():
                return {
                    "success": False,
                    "error": "Deliverable file not found"
                }
            
            content = report_path.read_text(encoding='utf-8')
            return {
                "success": True,
                "objective_id": objective_id,
                "content": content,
                "file_path": str(report_path),
                "deliverable_format": final_synthesis.get('deliverable_format', 'unknown')
            }
        except Exception as e:
            logger.error(f"Failed to get report content: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Perform web search."""
        try:
            from src.research.tools.web_search import WebSearchTool
            
            search_tool = WebSearchTool({
                'max_results': max_results,
                'timeout': 30
            })
            
            results = await search_tool.arun(query)
            return {
                "success": True,
                "query": query,
                "results": results.get('results', []),
                "total_results": results.get('total_results', 0)
            }
        except Exception as e:
            logger.error(f"Failed to search web: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_report(self, topic: str, content: str, format: str = "markdown") -> Dict[str, Any]:
        """Generate a report from content."""
        try:
            from src.generation.markdown_generator import MarkdownGenerator
            
            # Prepare report data
            report_data = {
                'title': f"Research Report: {topic}",
                'topic': topic,
                'summary': content[:200] + "..." if len(content) > 200 else content,
                'key_findings': [f"Content about {topic}"],
                'analysis': content,
                'insights': [f"Generated report on {topic}"],
                'conclusion': f"Report completed for {topic}"
            }
            
            generator = MarkdownGenerator({
                'output_directory': './outputs',
                'include_toc': True,
                'include_metadata': True
            })
            
            document = await generator.agenerate(report_data)
            
            return {
                "success": True,
                "file_path": document.file_path,
                "content": document.content
            }
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Global server instance
server = LocalResearcherMCPServer()

# MCP Tool functions
async def mcp_start_autonomous_research(user_request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Start autonomous research project."""
    return await server.start_autonomous_research(user_request, context)

async def mcp_get_research_status(objective_id: str) -> Dict[str, Any]:
    """Get research status."""
    return await server.get_research_status(objective_id)

async def mcp_list_research() -> Dict[str, Any]:
    """List all research projects."""
    return await server.list_research()

async def mcp_cancel_research(objective_id: str) -> Dict[str, Any]:
    """Cancel research project."""
    return await server.cancel_research(objective_id)

async def mcp_get_report_content(objective_id: str) -> Dict[str, Any]:
    """Get report content."""
    return await server.get_report_content(objective_id)

async def mcp_search_web(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search the web."""
    return await server.search_web(query, max_results)

async def mcp_generate_report(topic: str, content: str, format: str = "markdown") -> Dict[str, Any]:
    """Generate a report."""
    return await server.generate_report(topic, content, format)

if __name__ == "__main__":
    # Test the server
    async def test_server():
        print("🧪 Testing Local Researcher MCP Server...")
        
        # Test web search
        result = await mcp_search_web("artificial intelligence trends 2024")
        print(f"Web search result: {result['success']}")
        
        # Test research start
        result = await mcp_start_autonomous_research("AI trends 2024", {"domain": "technology"})
        print(f"Research start result: {result['success']}")
        
        if result['success']:
            objective_id = result['objective_id']
            
            # Wait a bit
            await asyncio.sleep(2)
            
            # Check status
            status = await mcp_get_research_status(objective_id)
            print(f"Research status: {status['status']}")
            
            # Wait for completion
            max_wait = 30
            wait_time = 0
            while wait_time < max_wait and status['status'] not in ['completed', 'failed']:
                await asyncio.sleep(1)
                status = await mcp_get_research_status(objective_id)
                wait_time += 1
            
            if status['status'] == 'completed':
                print(f"✅ Research completed! Deliverable: {status.get('final_synthesis', {}).get('deliverable_path', 'N/A')}")
            else:
                print(f"❌ Research failed or timed out")
    
    asyncio.run(test_server())
