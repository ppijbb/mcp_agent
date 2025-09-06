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

from src.core.research_orchestrator import ResearchOrchestrator, ResearchRequest
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_default_logging

# Setup logging
setup_default_logging()
logger = logging.getLogger(__name__)

class LocalResearcherMCPServer:
    """MCP Server for Local Researcher functionality."""
    
    def __init__(self):
        """Initialize the MCP server."""
        self.config_manager = ConfigManager()
        self.orchestrator = ResearchOrchestrator(self.config_manager)
        self.active_research = {}
        logger.info("Local Researcher MCP Server initialized")
    
    async def start_research(self, topic: str, domain: str = "general", depth: str = "basic") -> Dict[str, Any]:
        """Start a new research project."""
        try:
            request = ResearchRequest(
                topic=topic,
                domain=domain,
                depth=depth,
                sources=["web"],
                output_format="markdown"
            )
            
            research_id = self.orchestrator.start_research(request)
            
            return {
                "success": True,
                "research_id": research_id,
                "message": f"Research started for topic: {topic}"
            }
        except Exception as e:
            logger.error(f"Failed to start research: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_research_status(self, research_id: str) -> Dict[str, Any]:
        """Get the status of a research project."""
        try:
            status = self.orchestrator.get_research_status(research_id)
            if not status:
                return {
                    "success": False,
                    "error": "Research project not found"
                }
            
            return {
                "success": True,
                "research_id": research_id,
                "status": status.status,
                "progress": status.progress,
                "current_step": status.current_step,
                "report_path": status.report_path,
                "error_message": status.error_message
            }
        except Exception as e:
            logger.error(f"Failed to get research status: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_research(self) -> Dict[str, Any]:
        """List all research projects."""
        try:
            projects = self.orchestrator.get_research_list()
            return {
                "success": True,
                "projects": projects
            }
        except Exception as e:
            logger.error(f"Failed to list research: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def cancel_research(self, research_id: str) -> Dict[str, Any]:
        """Cancel a research project."""
        try:
            success = self.orchestrator.cancel_research(research_id)
            return {
                "success": success,
                "message": f"Research {research_id} {'cancelled' if success else 'not found'}"
            }
        except Exception as e:
            logger.error(f"Failed to cancel research: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_report_content(self, research_id: str) -> Dict[str, Any]:
        """Get the content of a research report."""
        try:
            status = self.orchestrator.get_research_status(research_id)
            if not status or not status.report_path:
                return {
                    "success": False,
                    "error": "Report not found or not completed"
                }
            
            report_path = Path(status.report_path)
            if not report_path.exists():
                return {
                    "success": False,
                    "error": "Report file not found"
                }
            
            content = report_path.read_text(encoding='utf-8')
            return {
                "success": True,
                "research_id": research_id,
                "content": content,
                "file_path": str(report_path)
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
async def mcp_start_research(topic: str, domain: str = "general", depth: str = "basic") -> Dict[str, Any]:
    """Start a research project."""
    return await server.start_research(topic, domain, depth)

async def mcp_get_research_status(research_id: str) -> Dict[str, Any]:
    """Get research status."""
    return await server.get_research_status(research_id)

async def mcp_list_research() -> Dict[str, Any]:
    """List all research projects."""
    return await server.list_research()

async def mcp_cancel_research(research_id: str) -> Dict[str, Any]:
    """Cancel research project."""
    return await server.cancel_research(research_id)

async def mcp_get_report_content(research_id: str) -> Dict[str, Any]:
    """Get report content."""
    return await server.get_report_content(research_id)

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
        result = await mcp_start_research("AI trends 2024", "technology", "basic")
        print(f"Research start result: {result['success']}")
        
        if result['success']:
            research_id = result['research_id']
            
            # Wait a bit
            await asyncio.sleep(2)
            
            # Check status
            status = await mcp_get_research_status(research_id)
            print(f"Research status: {status['status']}")
            
            # Wait for completion
            max_wait = 30
            wait_time = 0
            while wait_time < max_wait and status['status'] not in ['completed', 'failed']:
                await asyncio.sleep(1)
                status = await mcp_get_research_status(research_id)
                wait_time += 1
            
            if status['status'] == 'completed':
                print(f"✅ Research completed! Report: {status['report_path']}")
            else:
                print(f"❌ Research failed or timed out")
    
    asyncio.run(test_server())
