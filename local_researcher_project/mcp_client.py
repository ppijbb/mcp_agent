#!/usr/bin/env python3
"""
MCP Client for Local Researcher
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

class LocalResearcherMCPClient:
    """MCP Client for Local Researcher functionality."""
    
    def __init__(self):
        """Initialize the MCP client."""
        self.config_manager = ConfigManager()
        self.orchestrator = ResearchOrchestrator(self.config_manager)
        logger.info("Local Researcher MCP Client initialized")
    
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
                "message": f"Research started for topic: {topic}",
                "topic": topic,
                "domain": domain,
                "depth": depth
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
                "error_message": status.error_message,
                "created_at": status.created_at.isoformat() if status.created_at else None,
                "started_at": status.started_at.isoformat() if status.started_at else None,
                "completed_at": status.completed_at.isoformat() if status.completed_at else None
            }
        except Exception as e:
            logger.error(f"Failed to get research status: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_research(self, status_filter: Optional[str] = None) -> Dict[str, Any]:
        """List all research projects."""
        try:
            projects = self.orchestrator.get_research_list(status_filter)
            return {
                "success": True,
                "projects": projects,
                "total_count": len(projects)
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
                "message": f"Research {research_id} {'cancelled' if success else 'not found or already completed'}",
                "research_id": research_id
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
                    "error": "Report not found or research not completed"
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
                "file_path": str(report_path),
                "file_size": len(content),
                "word_count": len(content.split())
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
                "total_results": results.get('total_results', 0),
                "provider": results.get('provider', 'unknown')
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
                'conclusion': f"Report completed for {topic}",
                'content_stats': {
                    'word_count': len(content.split()),
                    'reading_time_minutes': max(1, len(content.split()) // 200)
                }
            }
            
            generator = MarkdownGenerator({
                'output_directory': './outputs',
                'include_toc': True,
                'include_metadata': True
            })
            
            document = await generator.agenerate(report_data)
            
            return {
                "success": True,
                "topic": topic,
                "file_path": document.file_path,
                "content": document.content,
                "format": format,
                "word_count": len(document.content.split())
            }
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def wait_for_completion(self, research_id: str, max_wait: int = 60) -> Dict[str, Any]:
        """Wait for research to complete."""
        try:
            wait_time = 0
            while wait_time < max_wait:
                status = await self.get_research_status(research_id)
                if not status['success']:
                    return status
                
                if status['status'] in ['completed', 'failed', 'cancelled']:
                    return status
                
                await asyncio.sleep(2)
                wait_time += 2
            
            return {
                "success": False,
                "error": "Research timed out",
                "research_id": research_id
            }
        except Exception as e:
            logger.error(f"Failed to wait for completion: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Global client instance
client = LocalResearcherMCPClient()

# Convenience functions
async def start_research(topic: str, domain: str = "general", depth: str = "basic") -> Dict[str, Any]:
    """Start a research project."""
    return await client.start_research(topic, domain, depth)

async def get_research_status(research_id: str) -> Dict[str, Any]:
    """Get research status."""
    return await client.get_research_status(research_id)

async def list_research(status_filter: Optional[str] = None) -> Dict[str, Any]:
    """List research projects."""
    return await client.list_research(status_filter)

async def cancel_research(research_id: str) -> Dict[str, Any]:
    """Cancel research project."""
    return await client.cancel_research(research_id)

async def get_report_content(research_id: str) -> Dict[str, Any]:
    """Get report content."""
    return await client.get_report_content(research_id)

async def search_web(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search the web."""
    return await client.search_web(query, max_results)

async def generate_report(topic: str, content: str, format: str = "markdown") -> Dict[str, Any]:
    """Generate a report."""
    return await client.generate_report(topic, content, format)

async def wait_for_completion(research_id: str, max_wait: int = 60) -> Dict[str, Any]:
    """Wait for research to complete."""
    return await client.wait_for_completion(research_id, max_wait)

if __name__ == "__main__":
    # Test the client
    async def test_client():
        print("🧪 Testing Local Researcher MCP Client...")
        
        # Test web search
        print("\n1. Testing web search...")
        result = await search_web("artificial intelligence trends 2024", 3)
        print(f"Web search: {result['success']}")
        if result['success']:
            print(f"Found {result['total_results']} results")
        
        # Test research start
        print("\n2. Testing research start...")
        result = await start_research("AI trends 2024", "technology", "basic")
        print(f"Research start: {result['success']}")
        
        if result['success']:
            research_id = result['research_id']
            print(f"Research ID: {research_id}")
            
            # Wait for completion
            print("\n3. Waiting for completion...")
            status = await wait_for_completion(research_id, 30)
            print(f"Final status: {status['status']}")
            
            if status['status'] == 'completed':
                print(f"✅ Research completed!")
                
                # Get report content
                print("\n4. Getting report content...")
                report = await get_report_content(research_id)
                if report['success']:
                    print(f"Report saved: {report['file_path']}")
                    print(f"Word count: {report['word_count']}")
                else:
                    print(f"Failed to get report: {report['error']}")
            else:
                print(f"❌ Research failed: {status.get('error_message', 'Unknown error')}")
        
        # List all research
        print("\n5. Listing all research...")
        projects = await list_research()
        if projects['success']:
            print(f"Total projects: {projects['total_count']}")
            for project in projects['projects']:
                print(f"  - {project['research_id']}: {project['status']}")
    
    asyncio.run(test_client())
