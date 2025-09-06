#!/usr/bin/env python3
"""
Local Researcher - Main Entry Point

This is the main entry point for the Local Researcher system that integrates
Gemini CLI with Open Deep Research for local research automation.
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.research_orchestrator import ResearchOrchestrator, ResearchRequest
from src.cli.gemini_integration import GeminiCLIIntegration
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

# Set up logging
logger = setup_logger("main", log_level="INFO")


class LocalResearcher:
    """Main Local Researcher application class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize Local Researcher.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.orchestrator = ResearchOrchestrator(config_path)
        self.gemini_integration = GeminiCLIIntegration(config_path)
        
        logger.info("Local Researcher initialized")
    
    async def start_research(self, topic: str, **kwargs) -> str:
        """Start a new research project.
        
        Args:
            topic: Research topic
            **kwargs: Additional research parameters
            
        Returns:
            Research ID
        """
        try:
            # Create research request
            request = ResearchRequest(
                topic=topic,
                domain=kwargs.get('domain'),
                depth=kwargs.get('depth', 'standard'),
                sources=kwargs.get('sources', []),
                output_format=kwargs.get('format', 'markdown')
            )
            
            # Start research
            research_id = await self.orchestrator.start_research(request)
            
            logger.info(f"Research started: {research_id}")
            return research_id
            
        except Exception as e:
            logger.error(f"Failed to start research: {e}")
            raise
    
    async def get_research_status(self, research_id: str) -> Optional[Dict[str, Any]]:
        """Get research status.
        
        Args:
            research_id: Research ID
            
        Returns:
            Research status or None if not found
        """
        try:
            status = self.orchestrator.get_research_status(research_id)
            if status:
                return {
                    'research_id': research_id,
                    'topic': status.topic,
                    'status': status.status,
                    'progress': status.progress,
                    'report_path': status.report_path,
                    'created_at': status.created_at.isoformat() if status.created_at else None,
                    'completed_at': status.completed_at.isoformat() if status.completed_at else None,
                    'error_message': status.error_message
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get research status: {e}")
            return None
    
    async def list_research(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List research projects.
        
        Args:
            status_filter: Filter by status (optional)
            
        Returns:
            List of research projects
        """
        try:
            research_list = self.orchestrator.list_active_research()
            
            if status_filter:
                research_list = [r for r in research_list if r.status == status_filter]
            
            return [
                {
                    'research_id': r.request_id,
                    'topic': r.topic,
                    'status': r.status,
                    'progress': r.progress,
                    'report_path': r.report_path,
                    'created_at': r.created_at.isoformat() if r.created_at else None,
                    'completed_at': r.completed_at.isoformat() if r.completed_at else None
                }
                for r in research_list
            ]
            
        except Exception as e:
            logger.error(f"Failed to list research: {e}")
            return []
    
    async def cancel_research(self, research_id: str) -> bool:
        """Cancel a research project.
        
        Args:
            research_id: Research ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            success = self.orchestrator.cancel_research(research_id)
            if success:
                logger.info(f"Research cancelled: {research_id}")
            else:
                logger.warning(f"Failed to cancel research: {research_id}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to cancel research: {e}")
            return False
    
    async def run_interactive_mode(self):
        """Run in interactive mode."""
        try:
            await self.gemini_integration.run_interactive_mode()
        except Exception as e:
            logger.error(f"Interactive mode failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.orchestrator.cleanup()
            await self.gemini_integration.cleanup()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


async def main():
    """Main function."""
    try:
        # Initialize Local Researcher
        researcher = LocalResearcher()
        
        # Check command line arguments
        if len(sys.argv) < 2:
            print("Usage: python main.py <command> [args...]")
            print("Commands:")
            print("  research <topic> [options]  - Start a new research project")
            print("  status [research_id]        - Check research status")
            print("  list [--status=STATUS]      - List research projects")
            print("  cancel <research_id>        - Cancel a research project")
            print("  interactive                 - Run in interactive mode")
            print("  help                        - Show this help")
            return
        
        command = sys.argv[1]
        
        if command == "research":
            if len(sys.argv) < 3:
                print("Error: Research topic is required")
                return
            
            topic = sys.argv[2]
            options = {}
            
            # Parse options
            for i in range(3, len(sys.argv)):
                arg = sys.argv[i]
                if arg.startswith("--"):
                    if "=" in arg:
                        key, value = arg[2:].split("=", 1)
                        options[key] = value
                    else:
                        options[arg[2:]] = True
            
            # Start research
            research_id = await researcher.start_research(topic, **options)
            print(f"Research started: {research_id}")
            print(f"Topic: {topic}")
            print(f"Use 'python main.py status {research_id}' to check progress")
            
        elif command == "status":
            if len(sys.argv) >= 3:
                research_id = sys.argv[2]
                status = await researcher.get_research_status(research_id)
                if status:
                    print(f"Research Status: {research_id}")
                    print(f"Topic: {status['topic']}")
                    print(f"Status: {status['status']}")
                    print(f"Progress: {status['progress']:.1f}%")
                    if status['report_path']:
                        print(f"Report: {status['report_path']}")
                    if status['error_message']:
                        print(f"Error: {status['error_message']}")
                else:
                    print(f"Research not found: {research_id}")
            else:
                # List all research
                research_list = await researcher.list_research()
                if research_list:
                    print("Active Research Projects:")
                    for research in research_list:
                        print(f"  {research['research_id']}: {research['topic']} ({research['status']}, {research['progress']:.1f}%)")
                else:
                    print("No active research projects")
                    
        elif command == "list":
            status_filter = None
            for arg in sys.argv[2:]:
                if arg.startswith("--status="):
                    status_filter = arg.split("=", 1)[1]
            
            research_list = await researcher.list_research(status_filter)
            if research_list:
                print("Research Projects:")
                for research in research_list:
                    print(f"  {research['research_id']}: {research['topic']} ({research['status']}, {research['progress']:.1f}%)")
            else:
                print("No research projects found")
                
        elif command == "cancel":
            if len(sys.argv) < 3:
                print("Error: Research ID is required")
                return
            
            research_id = sys.argv[2]
            success = await researcher.cancel_research(research_id)
            if success:
                print(f"Research cancelled: {research_id}")
            else:
                print(f"Failed to cancel research: {research_id}")
                
        elif command == "interactive":
            await researcher.run_interactive_mode()
            
        elif command == "help":
            print("Local Researcher - Available Commands:")
            print("")
            print("  research <topic> [options]  - Start a new research project")
            print("    Options:")
            print("      --domain=DOMAIN         - Research domain (technology, science, business, general)")
            print("      --depth=DEPTH           - Research depth (basic, standard, comprehensive)")
            print("      --sources=SOURCES       - Comma-separated list of sources")
            print("      --format=FORMAT         - Output format (markdown, pdf, html)")
            print("")
            print("  status [research_id]        - Check research status")
            print("  list [--status=STATUS]      - List research projects")
            print("  cancel <research_id>        - Cancel a research project")
            print("  interactive                 - Run in interactive mode")
            print("  help                        - Show this help")
            print("")
            print("Examples:")
            print("  python main.py research 'Artificial Intelligence trends'")
            print("  python main.py research 'Climate change' --domain=science --depth=comprehensive")
            print("  python main.py status research_20240101_1234_5678")
            print("  python main.py list --status=completed")
            print("  python main.py interactive")
            
        else:
            print(f"Unknown command: {command}")
            print("Use 'python main.py help' for available commands")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
    finally:
        # Cleanup
        try:
            if 'researcher' in locals():
                await researcher.cleanup()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
