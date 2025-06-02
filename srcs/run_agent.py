#!/usr/bin/env python3
"""
MCP Agent System Runner

Unified execution script for all agents in the MCP Agent system.
Allows users to easily run basic agents, enterprise agents, or utility scripts.
"""

import argparse
import sys
import os
from pathlib import Path

def list_agents():
    """List all available agents"""
    print("\n=== Available Agents ===\n")
    
    print("📝 Basic Agents:")
    basic_agents = {
        "basic": "Basic functionality and testing agent",
        "researcher": "Research and information gathering agent", 
        "parallel": "Parallel processing demonstration agent",
        "streamlit": "Streamlit web interface agent",
        "data_generator": "Data generation and synthesis agent",
        "enhanced_data_generator": "Advanced data generation with ML",
        "rag": "Retrieval-Augmented Generation agent"
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

def run_basic_agent(agent_name):
    """Run a basic agent"""
    agent_map = {
        "basic": "basic_agents.basic",
        "researcher": "basic_agents.researcher", 
        "parallel": "basic_agents.parallel",
        "streamlit": "basic_agents.streamlit_agent",
        "data_generator": "basic_agents.data_generator",
        "enhanced_data_generator": "basic_agents.enhanced_data_generator",
        "rag": "basic_agents.rag_agent"
    }
    
    if agent_name not in agent_map:
        print(f"❌ Unknown basic agent: {agent_name}")
        return False
    
    try:
        print(f"🚀 Starting basic agent: {agent_name}")
        module_name = agent_map[agent_name]
        
        if agent_name == "streamlit":
            # Special handling for Streamlit agent
            os.system(f"cd basic_agents && python streamlit_agent.py")
        else:
            # Import and run the main function
            exec(f"from {module_name} import main; main()")
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
        exec(f"from {module_name} import main; main()")
        return True
    except Exception as e:
        print(f"❌ Error running enterprise agent {agent_name}: {str(e)}")
        return False

def run_utility(util_name):
    """Run a utility script"""
    util_map = {
        "mental": "utils.mental",
        "mental_viz": "utils.mental_visualization", 
        "swarm": "utils.swarm",
        "workflow": "utils.workflow_orchestration"
    }
    
    if util_name not in util_map:
        print(f"❌ Unknown utility: {util_name}")
        return False
    
    try:
        print(f"🛠️  Starting utility: {util_name}")
        module_name = util_map[util_name]
        exec(f"from {module_name} import main; main()")
        return True
    except Exception as e:
        print(f"❌ Error running utility {util_name}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="MCP Agent System Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py --list                    # List all available agents
  python run_agent.py --basic researcher        # Run research agent
  python run_agent.py --enterprise supply_chain # Run supply chain agent
  python run_agent.py --utility mental          # Run mental model utility
        """
    )
    
    parser.add_argument("--list", action="store_true", help="List all available agents")
    parser.add_argument("--basic", metavar="AGENT", help="Run a basic agent")
    parser.add_argument("--enterprise", metavar="AGENT", help="Run an enterprise agent")
    parser.add_argument("--utility", metavar="UTIL", help="Run a utility script")
    
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
    
    # If no arguments provided, show help
    parser.print_help()
    print("\n💡 Use --list to see all available agents")

if __name__ == "__main__":
    main() 