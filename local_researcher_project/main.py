#!/usr/bin/env python3
"""
Autonomous Multi-Agent Research System

This is a fully autonomous multi-agent research system that:
1. Self-analyzes user requests and objectives
2. Dynamically decomposes tasks and assigns them to specialized agents
3. Orchestrates multi-agent collaboration with MCP integration
4. Performs critical evaluation and recursive execution
5. Validates results against original objectives
6. Generates comprehensive final deliverables

No fallback or dummy code - production-level autonomous operation only.
"""

import asyncio
import sys
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.autonomous_orchestrator import LangGraphOrchestrator
from src.agents.task_analyzer import TaskAnalyzerAgent
from src.agents.task_decomposer import TaskDecomposerAgent
from src.agents.research_agent import ResearchAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.core.mcp_integration import MCPIntegrationManager
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

# Set up logging
logger = setup_logger("autonomous_research", log_level="INFO")


class ResearchObjective:
    """Represents a research objective with autonomous analysis capabilities."""
    
    def __init__(self, user_request: str, context: Optional[Dict[str, Any]] = None):
        self.objective_id = str(uuid.uuid4())
        self.user_request = user_request
        self.context = context or {}
        self.analyzed_objectives = []
        self.decomposed_tasks = []
        self.assigned_agents = []
        self.execution_results = []
        self.evaluation_results = []
        self.validation_results = []
        self.final_synthesis = None
        self.created_at = datetime.now()
        self.status = "initialized"
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "user_request": self.user_request,
            "context": self.context,
            "analyzed_objectives": self.analyzed_objectives,
            "decomposed_tasks": self.decomposed_tasks,
            "assigned_agents": self.assigned_agents,
            "execution_results": self.execution_results,
            "evaluation_results": self.evaluation_results,
            "validation_results": self.validation_results,
            "final_synthesis": self.final_synthesis,
            "created_at": self.created_at.isoformat(),
            "status": self.status
        }


class AutonomousResearchSystem:
    """Fully autonomous multi-agent research system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the autonomous research system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.mcp_manager = MCPIntegrationManager(config_path)
        
        # Initialize specialized agents
        self.task_analyzer = TaskAnalyzerAgent(config_path)
        self.task_decomposer = TaskDecomposerAgent(config_path)
        self.research_agent = ResearchAgent(config_path)
        self.evaluation_agent = EvaluationAgent(config_path)
        self.validation_agent = ValidationAgent(config_path)
        self.synthesis_agent = SynthesisAgent(config_path)
        
        # Initialize LangGraph orchestrator
        self.orchestrator = LangGraphOrchestrator(
            config_path=config_path,
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
        
        # Active research objectives
        self.active_objectives: Dict[str, ResearchObjective] = {}
        
        logger.info("Autonomous Research System initialized with full agent orchestration")
    
    async def start_autonomous_research(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Start fully autonomous research with multi-agent orchestration.
        
        This method implements the complete autonomous workflow:
        1. Self-analysis of user request and objectives
        2. Dynamic task decomposition and agent assignment
        3. Multi-agent execution with MCP integration
        4. Critical evaluation and recursive execution
        5. Result validation against original objectives
        6. Final synthesis and deliverable generation
        
        Args:
            user_request: The user's research request
            context: Additional context for the research
            
        Returns:
            Research objective ID
        """
        try:
            # Create research objective
            objective = ResearchObjective(user_request, context)
            self.active_objectives[objective.objective_id] = objective
            
            logger.info(f"Starting autonomous research for objective: {objective.objective_id}")
            logger.info(f"User request: {user_request}")
            
            # Use the new LLM-based orchestrator
            objective_id = await self.orchestrator.start_autonomous_research(user_request, context)
            
            # Get the updated objective from orchestrator
            objective = self.active_objectives.get(objective_id)
            if not objective:
                # Create a simple objective for tracking
                objective = ResearchObjective(user_request, context)
                objective.objective_id = objective_id
                self.active_objectives[objective_id] = objective
            
            logger.info(f"Autonomous research completed successfully: {objective_id}")
            
            return objective_id
            
        except Exception as e:
            logger.error(f"Autonomous research failed: {e}")
            if 'objective_id' in locals() and objective_id in self.active_objectives:
                self.active_objectives[objective_id].status = "failed"
            raise
    
    async def get_research_status(self, objective_id: str) -> Optional[Dict[str, Any]]:
        """Get autonomous research status with full orchestration details.
        
        Args:
            objective_id: Research objective ID
            
        Returns:
            Complete research status or None if not found
        """
        try:
            # Get status from orchestrator
            status = await self.orchestrator.get_research_status(objective_id)
            if status:
                return status
            
            # Fallback to local objective if exists
            if objective_id in self.active_objectives:
                objective = self.active_objectives[objective_id]
                return objective.to_dict()
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get research status: {e}")
            return None
    
    async def list_research(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all autonomous research objectives.
        
        Args:
            status_filter: Filter by status (optional)
            
        Returns:
            List of research objectives with full orchestration details
        """
        try:
            # Get list from orchestrator
            objectives = await self.orchestrator.list_research()
            
            if status_filter:
                objectives = [obj for obj in objectives if obj.get('status') == status_filter]
            
            return objectives
            
        except Exception as e:
            logger.error(f"Failed to list research: {e}")
            return []
    
    async def cancel_research(self, objective_id: str) -> bool:
        """Cancel an autonomous research objective.
        
        Args:
            objective_id: Research objective ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            # Cancel through orchestrator
            success = await self.orchestrator.cancel_research(objective_id)
            
            if success:
                logger.info(f"Research objective cancelled: {objective_id}")
            else:
                logger.warning(f"Failed to cancel research objective: {objective_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cancel research: {e}")
            return False
    
    async def run_interactive_mode(self):
        """Run in interactive autonomous mode."""
        try:
            print("🤖 Autonomous Multi-Agent Research System")
            print("=" * 50)
            print("This system will autonomously analyze your request,")
            print("decompose tasks, assign agents, execute research,")
            print("evaluate results, and generate comprehensive deliverables.")
            print("=" * 50)
            
            while True:
                try:
                    user_input = input("\n🔍 Enter your research request (or 'quit' to exit): ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if not user_input:
                        print("❌ Please enter a research request.")
                        continue
                    
                    print(f"\n🚀 Starting autonomous research for: {user_input}")
                    print("⏳ This may take several minutes...")
                    
                    # Start LLM-based autonomous research
                    objective_id = await self.orchestrator.start_autonomous_research(user_input)
                    
                    print(f"✅ Research objective created: {objective_id}")
                    print("📊 Monitoring progress...")
                    
                    # Monitor progress
                    while True:
                        status = await self.get_research_status(objective_id)
                        if not status:
                            print("❌ Research objective not found")
                            break
                        
                        print(f"📈 Status: {status['status']}")
                        
                        if status['status'] in ['completed', 'failed', 'cancelled']:
                            if status['status'] == 'completed':
                                print("🎉 Research completed successfully!")
                                if status.get('final_synthesis', {}).get('deliverable_path'):
                                    print(f"📄 Deliverable: {status['final_synthesis']['deliverable_path']}")
                            else:
                                print(f"❌ Research {status['status']}")
                            break
                        
                        await asyncio.sleep(2)
                    
                except KeyboardInterrupt:
                    print("\n⏹️  Operation cancelled by user")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    logger.error(f"Interactive mode error: {e}")
                    
        except Exception as e:
            logger.error(f"Interactive mode failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.orchestrator.cleanup()
            await self.mcp_manager.cleanup()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


async def main():
    """Main function for Autonomous Multi-Agent Research System."""
    try:
        # Initialize Autonomous Research System
        system = AutonomousResearchSystem()
        
        # Check command line arguments
        if len(sys.argv) < 2:
            print("🤖 Autonomous Multi-Agent Research System")
            print("=" * 50)
            print("Usage: python main.py <command> [args...]")
            print("")
            print("Commands:")
            print("  research <request> [options]  - Start autonomous research")
            print("  status [objective_id]         - Check research status")
            print("  list [--status=STATUS]        - List research objectives")
            print("  cancel <objective_id>         - Cancel research objective")
            print("  interactive                   - Run in interactive mode")
            print("  help                          - Show this help")
            print("")
            print("This system autonomously:")
            print("  • Analyzes your request and objectives")
            print("  • Decomposes tasks and assigns specialized agents")
            print("  • Executes multi-agent research with MCP integration")
            print("  • Evaluates results and performs recursive refinement")
            print("  • Validates results against original objectives")
            print("  • Generates comprehensive final deliverables")
            return
        
        command = sys.argv[1]
        
        if command == "research":
            if len(sys.argv) < 3:
                print("❌ Error: Research request is required")
                print("Example: python main.py research 'Analyze AI trends in healthcare'")
                return
            
            user_request = " ".join(sys.argv[2:])
            context = {}
            
            # Parse any options (for future extensibility)
            for i, arg in enumerate(sys.argv[2:], 2):
                if arg.startswith("--"):
                    if "=" in arg:
                        key, value = arg[2:].split("=", 1)
                        context[key] = value
                    else:
                        context[arg[2:]] = True
                    # Remove from user_request
                    user_request = user_request.replace(arg, "").strip()
            
            print(f"🚀 Starting autonomous research for: {user_request}")
            print("⏳ This may take several minutes...")
            
            # Start autonomous research
            objective_id = await system.orchestrator.start_autonomous_research(user_request, context)
            print(f"✅ Research objective created: {objective_id}")
            print(f"📊 Use 'python main.py status {objective_id}' to check progress")
            
        elif command == "status":
            if len(sys.argv) >= 3:
                objective_id = sys.argv[2]
                status = await system.get_research_status(objective_id)
                if status:
                    print(f"📊 Research Objective Status: {objective_id}")
                    print(f"🔍 Request: {status['user_request']}")
                    print(f"📈 Status: {status['status']}")
                    print(f"🕒 Created: {status['created_at']}")
                    
                    if status['analyzed_objectives']:
                        print(f"🎯 Analyzed Objectives: {len(status['analyzed_objectives'])}")
                    
                    if status['decomposed_tasks']:
                        print(f"📋 Decomposed Tasks: {len(status['decomposed_tasks'])}")
                    
                    if status['assigned_agents']:
                        print(f"🤖 Assigned Agents: {len(status['assigned_agents'])}")
                    
                    if status['execution_results']:
                        print(f"⚡ Execution Results: {len(status['execution_results'])}")
                    
                    if status['evaluation_results']:
                        eval_score = status['evaluation_results'].get('overall_score', 0)
                        print(f"📊 Evaluation Score: {eval_score:.2f}")
                    
                    if status['validation_results']:
                        val_score = status['validation_results'].get('validation_score', 0)
                        print(f"✅ Validation Score: {val_score:.2f}%")
                    
                    if status['final_synthesis']:
                        print(f"📄 Final Deliverable: {status['final_synthesis'].get('deliverable_path', 'N/A')}")
                else:
                    print(f"❌ Research objective not found: {objective_id}")
            else:
                # List all research objectives
                objectives = await system.list_research()
                if objectives:
                    print("📋 Active Research Objectives:")
                    for obj in objectives:
                        print(f"  {obj['objective_id']}: {obj['user_request'][:50]}... ({obj['status']})")
                else:
                    print("📭 No active research objectives")
                    
        elif command == "list":
            status_filter = None
            for arg in sys.argv[2:]:
                if arg.startswith("--status="):
                    status_filter = arg.split("=", 1)[1]
            
            objectives = await system.list_research(status_filter)
            if objectives:
                print("📋 Research Objectives:")
                for obj in objectives:
                    print(f"  {obj['objective_id']}: {obj['user_request'][:50]}... ({obj['status']})")
            else:
                print("📭 No research objectives found")
                
        elif command == "cancel":
            if len(sys.argv) < 3:
                print("❌ Error: Objective ID is required")
                return
            
            objective_id = sys.argv[2]
            success = await system.cancel_research(objective_id)
            if success:
                print(f"✅ Research objective cancelled: {objective_id}")
            else:
                print(f"❌ Failed to cancel research objective: {objective_id}")
                
        elif command == "interactive":
            await system.run_interactive_mode()
            
        elif command == "help":
            print("🤖 Autonomous Multi-Agent Research System")
            print("=" * 50)
            print("This system provides fully autonomous research capabilities with:")
            print("")
            print("🧠 Self-Analysis: Automatically analyzes user requests and objectives")
            print("🔧 Task Decomposition: Dynamically breaks down complex tasks")
            print("🤖 Multi-Agent Orchestration: Assigns specialized agents to tasks")
            print("🔗 MCP Integration: Leverages Model Context Protocol for enhanced capabilities")
            print("📊 Critical Evaluation: Performs recursive evaluation and refinement")
            print("✅ Result Validation: Ensures results match original objectives")
            print("📄 Final Synthesis: Generates comprehensive deliverables")
            print("")
            print("Commands:")
            print("  research <request> [options]  - Start autonomous research")
            print("  status [objective_id]         - Check research status")
            print("  list [--status=STATUS]        - List research objectives")
            print("  cancel <objective_id>         - Cancel research objective")
            print("  interactive                   - Run in interactive mode")
            print("  help                          - Show this help")
            print("")
            print("Examples:")
            print("  python main.py research 'Analyze AI trends in healthcare'")
            print("  python main.py research 'Compare renewable energy technologies'")
            print("  python main.py status obj_12345")
            print("  python main.py list --status=completed")
            print("  python main.py interactive")
            
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python main.py help' for available commands")
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        try:
            if 'system' in locals():
                await system.cleanup()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
