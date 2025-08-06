"""
Gemini CLI Integration Module

This module provides integration between the Local Researcher system
and Gemini CLI for seamless command-line research operations.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import subprocess
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.research_orchestrator import ResearchOrchestrator, ResearchRequest
from ..utils.config_manager import ConfigManager
from ..utils.logger import setup_logger


class GeminiCLIIntegration:
    """
    Integration layer between Local Researcher and Gemini CLI.
    
    This class handles the communication and command processing
    between Gemini CLI and the local research system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Gemini CLI integration."""
        self.console = Console()
        self.logger = setup_logger("gemini_cli_integration")
        self.config_manager = ConfigManager(config_path)
        self.orchestrator = ResearchOrchestrator(config_path)
        
        # Gemini CLI configuration
        self.gemini_cli_path = self._find_gemini_cli()
        self.research_commands = self._load_research_commands()
        
        self.logger.info("Gemini CLI Integration initialized")
    
    def _find_gemini_cli(self) -> str:
        """Find the Gemini CLI installation."""
        try:
            # Try to find gemini CLI in PATH
            result = subprocess.run(
                ["which", "gemini"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            try:
                # Try npm global installation
                result = subprocess.run(
                    ["npm", "list", "-g", "@google/gemini-cli"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    return "gemini"
                else:
                    self.logger.warning("Gemini CLI not found in PATH")
                    return "gemini"  # Assume it's available
            except Exception as e:
                self.logger.error(f"Error finding Gemini CLI: {e}")
                return "gemini"
    
    def _load_research_commands(self) -> Dict[str, Any]:
        """Load research command definitions."""
        commands_file = Path(__file__).parent / "research_commands.json"
        if commands_file.exists():
            with open(commands_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return self._get_default_commands()
    
    def _get_default_commands(self) -> Dict[str, Any]:
        """Get default research command definitions."""
        return {
            "research": {
                "description": "Start a new research project",
                "usage": "gemini research <topic> [options]",
                "options": {
                    "--domain": "Specify research domain",
                    "--depth": "Research depth (basic|standard|comprehensive)",
                    "--sources": "Comma-separated list of sources",
                    "--format": "Output format (markdown|pdf|html)"
                }
            },
            "status": {
                "description": "Check research status",
                "usage": "gemini status [research_id]"
            },
            "list": {
                "description": "List all research projects",
                "usage": "gemini list [--active|--completed|--failed]"
            },
            "cancel": {
                "description": "Cancel a research project",
                "usage": "gemini cancel <research_id>"
            }
        }
    
    async def process_gemini_command(self, command: str, args: List[str]) -> Dict[str, Any]:
        """
        Process a Gemini CLI command and execute corresponding research operation.
        
        Args:
            command: The command to execute
            args: Command arguments
            
        Returns:
            Result dictionary with status and data
        """
        self.logger.info(f"Processing Gemini command: {command} {args}")
        
        try:
            if command == "research":
                return await self._handle_research_command(args)
            elif command == "status":
                return await self._handle_status_command(args)
            elif command == "list":
                return await self._handle_list_command(args)
            elif command == "cancel":
                return await self._handle_cancel_command(args)
            elif command == "help":
                return await self._handle_help_command(args)
            else:
                return {
                    "success": False,
                    "error": f"Unknown command: {command}",
                    "available_commands": list(self.research_commands.keys())
                }
                
        except Exception as e:
            self.logger.error(f"Error processing command {command}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_research_command(self, args: List[str]) -> Dict[str, Any]:
        """Handle the research command."""
        if not args:
            return {
                "success": False,
                "error": "Research topic is required",
                "usage": self.research_commands["research"]["usage"]
            }
        
        # Parse arguments
        topic = args[0]
        options = self._parse_options(args[1:])
        
        # Create research request
        request = ResearchRequest(
            topic=topic,
            domain=options.get("--domain"),
            depth=options.get("--depth", "standard"),
            sources=options.get("--sources", "").split(",") if options.get("--sources") else [],
            output_format=options.get("--format", "markdown")
        )
        
        # Start research
        research_id = await self.orchestrator.start_research(request)
        
        return {
            "success": True,
            "research_id": research_id,
            "topic": topic,
            "message": f"Research started: {research_id}"
        }
    
    async def _handle_status_command(self, args: List[str]) -> Dict[str, Any]:
        """Handle the status command."""
        if not args:
            # Return status of all active research
            active_research = self.orchestrator.list_active_research()
            return {
                "success": True,
                "active_research": [research.dict() for research in active_research]
            }
        else:
            # Return status of specific research
            research_id = args[0]
            status = self.orchestrator.get_research_status(research_id)
            
            if status:
                return {
                    "success": True,
                    "research_status": status.dict()
                }
            else:
                return {
                    "success": False,
                    "error": f"Research not found: {research_id}"
                }
    
    async def _handle_list_command(self, args: List[str]) -> Dict[str, Any]:
        """Handle the list command."""
        options = self._parse_options(args)
        filter_type = options.get("--filter", "all")
        
        active_research = self.orchestrator.list_active_research()
        
        if filter_type == "active":
            filtered_research = [r for r in active_research if r.status == "running"]
        elif filter_type == "completed":
            filtered_research = [r for r in active_research if r.status == "completed"]
        elif filter_type == "failed":
            filtered_research = [r for r in active_research if r.status == "failed"]
        else:
            filtered_research = active_research
        
        return {
            "success": True,
            "research_list": [research.dict() for research in filtered_research]
        }
    
    async def _handle_cancel_command(self, args: List[str]) -> Dict[str, Any]:
        """Handle the cancel command."""
        if not args:
            return {
                "success": False,
                "error": "Research ID is required",
                "usage": self.research_commands["cancel"]["usage"]
            }
        
        research_id = args[0]
        success = self.orchestrator.cancel_research(research_id)
        
        if success:
            return {
                "success": True,
                "message": f"Research cancelled: {research_id}"
            }
        else:
            return {
                "success": False,
                "error": f"Research not found or already completed: {research_id}"
            }
    
    async def _handle_help_command(self, args: List[str]) -> Dict[str, Any]:
        """Handle the help command."""
        if not args:
            # Show general help
            return {
                "success": True,
                "help": self._format_general_help()
            }
        else:
            # Show help for specific command
            command = args[0]
            if command in self.research_commands:
                return {
                    "success": True,
                    "help": self._format_command_help(command)
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown command: {command}",
                    "available_commands": list(self.research_commands.keys())
                }
    
    def _parse_options(self, args: List[str]) -> Dict[str, str]:
        """Parse command line options."""
        options = {}
        i = 0
        while i < len(args):
            if args[i].startswith("--"):
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    options[args[i]] = args[i + 1]
                    i += 2
                else:
                    options[args[i]] = "true"
                    i += 1
            else:
                i += 1
        return options
    
    def _format_general_help(self) -> str:
        """Format general help message."""
        help_text = "Local Researcher - Available Commands:\n\n"
        
        for command, info in self.research_commands.items():
            help_text += f"  {command}: {info['description']}\n"
            help_text += f"    Usage: {info['usage']}\n\n"
        
        return help_text
    
    def _format_command_help(self, command: str) -> str:
        """Format help for specific command."""
        if command not in self.research_commands:
            return f"Unknown command: {command}"
        
        info = self.research_commands[command]
        help_text = f"Command: {command}\n"
        help_text += f"Description: {info['description']}\n"
        help_text += f"Usage: {info['usage']}\n"
        
        if "options" in info:
            help_text += "\nOptions:\n"
            for option, description in info["options"].items():
                help_text += f"  {option}: {description}\n"
        
        return help_text
    
    def display_result(self, result: Dict[str, Any]):
        """Display the result in a user-friendly format."""
        if not result.get("success", False):
            self.console.print(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
            return
        
        # Display based on command type
        if "research_id" in result:
            self.console.print(f"[green]✓ Research started: {result['research_id']}[/green]")
            self.console.print(f"Topic: {result['topic']}")
        
        elif "research_status" in result:
            status = result["research_status"]
            self.console.print(f"[blue]Research Status: {status['request_id']}[/blue]")
            self.console.print(f"Topic: {status['topic']}")
            self.console.print(f"Status: {status['status']}")
            self.console.print(f"Progress: {status['progress']:.1f}%")
            
            if status.get('report_path'):
                self.console.print(f"Report: {status['report_path']}")
        
        elif "research_list" in result:
            research_list = result["research_list"]
            if not research_list:
                self.console.print("[yellow]No research projects found[/yellow]")
                return
            
            table = Table(title="Research Projects")
            table.add_column("ID", style="cyan")
            table.add_column("Topic", style="green")
            table.add_column("Status", style="blue")
            table.add_column("Progress", style="yellow")
            
            for research in research_list:
                table.add_row(
                    research["request_id"],
                    research["topic"],
                    research["status"],
                    f"{research['progress']:.1f}%"
                )
            
            self.console.print(table)
        
        elif "message" in result:
            self.console.print(f"[green]✓ {result['message']}[/green]")
        
        elif "help" in result:
            self.console.print(Panel(result["help"], title="Help"))
    
    async def run_interactive_mode(self):
        """Run the integration in interactive mode."""
        self.console.print("[bold blue]Local Researcher - Gemini CLI Integration[/bold blue]")
        self.console.print("Type 'help' for available commands or 'exit' to quit.\n")
        
        while True:
            try:
                user_input = self.console.input("[bold green]local-researcher> [/bold green]")
                
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                if not user_input.strip():
                    continue
                
                # Parse command and arguments
                parts = user_input.split()
                command = parts[0]
                args = parts[1:] if len(parts) > 1 else []
                
                # Process command
                result = await self.process_gemini_command(command, args)
                self.display_result(result)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except Exception as e:
                self.logger.error(f"Error in interactive mode: {e}")
                self.console.print(f"[red]Error: {e}[/red]")
        
        self.console.print("[blue]Goodbye![/blue]")
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.orchestrator.cleanup() 