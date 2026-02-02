#!/usr/bin/env python3
"""
MCP Agent System Runner

Unified execution script for all agents in the MCP Agent system.
Allows users to easily run basic agents, enterprise agents, or utility scripts.
"""

import argparse
import importlib
import sys
import os
from typing import Optional, Dict, Any


def list_agents():
    """List all available agents"""
    print("\n=== Available Agents ===\n")

    print("📝 Basic Agents:")
    basic_agents = {
        "basic": "Basic functionality and testing agent",
        "researcher": "Research and information gathering agent",
        "researcher_v2": "Enhanced research agent using common modules",
        "parallel": "Parallel processing demonstration agent",
        "streamlit": "Streamlit web interface agent",
        "data_generator": "Data generation and synthesis agent",
        "enhanced_data_generator": "Advanced data generation with ML",
        "rag": "Retrieval-Augmented Generation agent",
        "travel_scout": "Incognito mode travel search agent for best deals"
    }

    for agent, description in basic_agents.items():
        print(f"  • {agent:20} - {description}")

    print("\n🏢 Enterprise Agents:")
    enterprise_agents = {
        "hr_recruitment": "HR Recruitment and talent acquisition automation",
        "legal_compliance": "Legal compliance and contract analysis",
        "cybersecurity": "Cybersecurity infrastructure and threat detection",
        "supply_chain": "Supply chain orchestration and optimization",
        "customer_clv": "Customer lifetime value and experience optimization",
        "esg_carbon": "ESG and carbon neutrality management",
        "workplace": "Hybrid workplace optimization and management",
        "innovation": "Product innovation acceleration and development"
    }

    for agent, description in enterprise_agents.items():
        print(f"  • {agent:20} - {description}")

    print("\n🛠️  Utility Scripts:")
    utils = {
        "mental": "Mental model analysis and visualization",
        "mental_viz": "Mental model interactive visualization",
        "swarm": "Multi-agent swarm coordination",
        "workflow": "Workflow orchestration and management"
    }

    for util, description in utils.items():
        print(f"  • {util:20} - {description}")

    print("\n🔧 Development Examples:")
    examples = {
        "common_demo": "Demonstration of common modules usage",
        "template_basic": "Basic agent template example",
        "template_enterprise": "Enterprise agent template example"
    }

    for example, description in examples.items():
        print(f"  • {example:20} - {description}")

    print("\n🧬 Specialized Agents:")
    specialized_agents = {
        "genome": "Genome analysis and bioinformatics agent",
        "goal_setter": "Goal setting and planning agent",
        "decision": "Decision making and analysis agent",
        "ai_architect": "AI architecture design and optimization agent"
    }

    for agent, description in specialized_agents.items():
        print(f"  • {agent:20} - {description}")


def run_basic_agent(agent_name):
    """Run a basic agent"""
    agent_map = {
        "basic": "basic_agents.basic",
        "researcher": "basic_agents.researcher",
        "researcher_v2": "basic_agents.researcher_v2",
        "parallel": "basic_agents.parallel",
        "streamlit": "basic_agents.streamlit_agent",
        "data_generator": "basic_agents.data_generator",
        "enhanced_data_generator": "basic_agents.enhanced_data_generator",
        "rag": "basic_agents.rag_agent",
        "travel_scout": "travel_scout.travel_scout_agent"
    }

    if agent_name not in agent_map:
        print(f"❌ Unknown basic agent: {agent_name}")
        return False

    try:
        print(f"🚀 Starting basic agent: {agent_name}")
        module_name = agent_map[agent_name]

        if agent_name == "streamlit":
            # Special handling for Streamlit agent
            import subprocess
            result = subprocess.run(
                ["python", "streamlit_agent.py"],
                cwd="basic_agents",
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                raise RuntimeError(f"Streamlit agent failed: {result.stderr}")
            print(result.stdout)
        else:
            # Safe import and execution
            try:
                module = importlib.import_module(module_name)
                if not hasattr(module, 'main'):
                    raise AttributeError(f"Module {module_name} missing main function")
                main_func = getattr(module, 'main')
                if not callable(main_func):
                    raise TypeError(f"main in {module_name} is not callable")
                main_func()
            except ImportError as e:
                raise ImportError(f"Failed to import module {module_name}: {e}")
            except Exception as e:
                raise RuntimeError(f"Error executing {module_name}.main(): {e}")
        return True
    except Exception as e:
        print(f"❌ Error running basic agent {agent_name}: {str(e)}")
        return False


def run_enterprise_agent(agent_name):
    """Run an enterprise agent"""
    agent_map = {
        "hr_recruitment": "enterprise_agents.hr_recruitment_agent",
        "legal_compliance": "enterprise_agents.legal_compliance_agent",
        "cybersecurity": "enterprise_agents.cybersecurity_infrastructure_agent",
        "supply_chain": "enterprise_agents.supply_chain_orchestrator_agent",
        "customer_clv": "enterprise_agents.customer_lifetime_value_agent",
        "esg_carbon": "enterprise_agents.esg_carbon_neutral_agent",
        "workplace": "enterprise_agents.hybrid_workplace_optimizer_agent",
        "innovation": "enterprise_agents.product_innovation_accelerator_agent"
    }

    if agent_name not in agent_map:
        print(f"❌ Unknown enterprise agent: {agent_name}")
        return False

    try:
        print(f"🏢 Starting enterprise agent: {agent_name}")
        module_name = agent_map[agent_name]
        
        # Safe import and execution
        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, 'main'):
                raise AttributeError(f"Module {module_name} missing main function")
            main_func = getattr(module, 'main')
            if not callable(main_func):
                raise TypeError(f"main in {module_name} is not callable")
            main_func()
        except ImportError as e:
            raise ImportError(f"Failed to import module {module_name}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error executing {module_name}.main(): {e}")
        return True
    except Exception as e:
        print(f"❌ Error running enterprise agent {agent_name}: {str(e)}")
        return False


def run_specialized_agent(agent_name):
    """Run a specialized agent"""
    agent_map = {
        "genome": "genome_agent.run_genome_agent",
        "goal_setter": "goal_setter_agent.goal_setter",
        "decision": "advanced_agents.decision_agent",
        "ai_architect": "evolutionary_ai_architect.architect"
    }

    if agent_name not in agent_map:
        print(f"❌ Unknown specialized agent: {agent_name}")
        return False

    try:
        print(f"🧬 Starting specialized agent: {agent_name}")
        module_name = agent_map[agent_name]

        # Safe import and execution
        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, 'main'):
                raise AttributeError(f"Module {module_name} missing main function")
            main_func = getattr(module, 'main')
            if not callable(main_func):
                raise TypeError(f"main in {module_name} is not callable")
            main_func()
        except ImportError as e:
            raise ImportError(f"Failed to import module {module_name}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error executing {module_name}.main(): {e}")
        return True
    except Exception as e:
        print(f"❌ Error running specialized agent {agent_name}: {str(e)}")
        return False


def run_utility(util_name):
    """Run a utility script"""
    util_map = {
        "mental": "enterprise_agents.mental",
        "mental_viz": "utils.mental_visualization",
        "swarm": "basic_agents.swarm",
        "workflow": "basic_agents.workflow_orchestration"
    }

    if util_name == "genome":
        return run_specialized_agent(util_name)

    if util_name not in util_map:
        print(f"❌ Unknown utility: {util_name}")
        return False

    try:
        print(f"🛠️  Starting utility: {util_name}")
        module_name = util_map[util_name]
        
        # Safe import and execution
        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, 'main'):
                raise AttributeError(f"Module {module_name} missing main function")
            main_func = getattr(module, 'main')
            if not callable(main_func):
                raise TypeError(f"main in {module_name} is not callable")
            main_func()
        except ImportError as e:
            raise ImportError(f"Failed to import module {module_name}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error executing {module_name}.main(): {e}")
        return True
    except Exception as e:
        print(f"❌ Error running utility {util_name}: {str(e)}")
        return False


def run_development_example(example_name):
    """Run development examples and demonstrations"""
    if example_name == "common_demo":
        print("🔧 Common Modules Demonstration")
        print("\nThe common modules provide shared functionality across all agents:")
        print("📦 common/imports.py - Standardized imports")
        print("⚙️  common/config.py - Shared configurations and constants")
        print("🛠️  common/utils.py - Common utility functions")
        print("📋 common/templates.py - Agent base templates")
        print("\nExample usage:")
        print("from common import *")
        print("class MyAgent(BasicAgentTemplate):")
        print("    # Your agent implementation...")
        print("\nSee basic_agents/researcher_v2.py for a complete example!")
        return True

    elif example_name == "template_basic":
        print("🔧 Basic Agent Template Example")
        print("\nTo create a new basic agent using templates:")
        print("""
from common import BasicAgentTemplate

class MyAgent(BasicAgentTemplate):
    def __init__(self):
        super().__init__(
            agent_name="my_agent",
            task_description="Your agent's main task description"
        )

    # Override methods as needed
    def create_agents(self):
        # Return list of specialized agents
        pass

    def create_evaluator(self):
        # Return quality evaluator
        pass
""")
        return True

    elif example_name == "template_enterprise":
        print("🔧 Enterprise Agent Template Example")
        print("\nTo create a new enterprise agent using templates:")
        print("""
from common import EnterpriseAgentTemplate

class MyEnterpriseAgent(EnterpriseAgentTemplate):
    def __init__(self):
        super().__init__(
            agent_name="my_enterprise_agent",
            business_scope="Global Operations"
        )

    def create_agents(self):
        # Return list of specialized enterprise agents
        pass

    def create_evaluator(self):
        evaluation_criteria = [
            ("Business Impact", 40, "ROI and value creation"),
            ("Implementation", 30, "Feasibility and execution"),
            ("Innovation", 30, "Technology and process innovation")
        ]
        return self.create_standard_evaluator(evaluation_criteria)
""")
        return True

    else:
        print(f"❌ Unknown development example: {example_name}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="MCP Agent System Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py --list                    # List all available agents
  python run_agent.py --basic researcher        # Run research agent
  python run_agent.py --basic researcher_v2     # Run enhanced research agent
  python run_agent.py --enterprise supply_chain # Run supply chain agent
  python run_agent.py --utility mental          # Run mental model utility
  python run_agent.py --specialized genome     # Run genome analysis agent
  python run_agent.py --dev common_demo         # Show common modules demo
        """
    )

    parser.add_argument("--list", action="store_true", help="List all available agents")
    parser.add_argument("--basic", metavar="AGENT", help="Run a basic agent")
    parser.add_argument("--enterprise", metavar="AGENT", help="Run an enterprise agent")
    parser.add_argument("--utility", metavar="UTIL", help="Run a utility script")
    parser.add_argument("--specialized", metavar="AGENT", help="Run a specialized agent")
    parser.add_argument("--dev", metavar="EXAMPLE", help="Run development examples")

    args = parser.parse_args()

    if args.list:
        list_agents()
        return

    if args.basic:
        success = run_basic_agent(args.basic)
        sys.exit(0 if success else 1)

    if args.enterprise:
        success = run_enterprise_agent(args.enterprise)
        sys.exit(0 if success else 1)

    if args.utility:
        success = run_utility(args.utility)
        sys.exit(0 if success else 1)

    if args.specialized:
        success = run_specialized_agent(args.specialized)
        sys.exit(0 if success else 1)

    if args.dev:
        success = run_development_example(args.dev)
        sys.exit(0 if success else 1)

    # If no arguments provided, show help
    parser.print_help()
    print("\n💡 Use --list to see all available agents")


if __name__ == "__main__":
    main()
