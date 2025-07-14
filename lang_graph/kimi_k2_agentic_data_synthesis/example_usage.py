"""
Example usage of the Kimi-K2 Agentic Data Synthesis System.

This script demonstrates how to set up and run the complete pipeline
for generating high-quality training data through multi-agent simulations.
"""

import asyncio
import json
from pathlib import Path
from typing import List

from .system.agentic_data_synthesis_system import AgenticDataSynthesisSystem
from .models.domain import DomainConfig, DomainCategory, ComplexityLevel
from .models.tool import ToolConfig, ToolType
from .models.agent import AgentConfig, AgentType # BehaviorPattern will be string in config
from .models.simulation import SimulationConfig, EnvironmentConfig
from .models.evaluation import EvaluationConfig, EvaluationRubric
from .models.data import DataExportConfig


async def create_example_configs():
    """Create example configurations for the Kimi-K2 system."""
    
    # Domain configurations
    domains = [
        DomainConfig(
            domain_id="web_development",
            name="Web Development",
            description="Web development and programming tasks",
            domain_type=DomainCategory.TECHNOLOGY, # Use DomainCategory
            scenarios=[
                "Create a React component",
                "Debug JavaScript code",
                "Set up a Node.js server",
                "Optimize website performance"
            ],
            complexity_levels=[ComplexityLevel.BEGINNER, ComplexityLevel.INTERMEDIATE, ComplexityLevel.ADVANCED], # Use ComplexityLevel
            required_tools=["code_editor", "terminal", "browser"]
        ),
        DomainConfig(
            domain_id="data_analysis",
            name="Data Analysis",
            description="Data analysis and visualization tasks",
            domain_type=DomainCategory.SCIENTIFIC, # Use DomainCategory
            scenarios=[
                "Analyze CSV data",
                "Create data visualizations",
                "Perform statistical analysis",
                "Generate reports"
            ],
            complexity_levels=[ComplexityLevel.BEGINNER, ComplexityLevel.INTERMEDIATE, ComplexityLevel.ADVANCED],
            required_tools=["python", "pandas", "matplotlib", "jupyter"]
        )
    ]
    
    # Tool configurations
    tools = [
        ToolConfig(
            tool_id="code_editor",
            name="Code Editor",
            description="Multi-language code editor with syntax highlighting",
            tool_type=ToolType.MCP.value, # Use .value for string representation
            mcp_server="code_editor_server",
            parameters={
                "language": "string",
                "theme": "string",
                "auto_save": "boolean"
            },
            usage_examples=[
                "Open file: main.js",
                "Edit line 42: console.log('Hello World')",
                "Save file"
            ]
        ),
        ToolConfig(
            tool_id="terminal",
            name="Terminal",
            description="Command line interface for system operations",
            tool_type=ToolType.MCP.value,
            mcp_server="terminal_server",
            parameters={
                "command": "string",
                "working_dir": "string"
            },
            usage_examples=[
                "Run: npm install",
                "Navigate: cd /project",
                "Execute: python script.py"
            ]
        ),
        ToolConfig(
            tool_id="python",
            name="Python Interpreter",
            description="Python programming language interpreter",
            tool_type=ToolType.SYNTHETIC.value,
            synthetic_behavior={
                "execution_time": "0.1-2.0s",
                "error_rate": "0.05",
                "output_format": "text"
            },
            parameters={
                "code": "string",
                "timeout": "number"
            },
            usage_examples=[
                "Execute: print('Hello World')",
                "Import: import pandas as pd",
                "Calculate: 2 + 2"
            ]
        )
    ]
    
    # Agent configurations
    agents = [
        AgentConfig(
            agent_id="senior_developer",
            name="Senior Developer",
            description="Experienced software developer with expertise in multiple languages",
            agent_type=AgentType.EXPERT.value, # Use .value
            behavior_pattern="COLLABORATIVE", # Use string for behavior_pattern
            expertise_domains=["web_development", "software_engineering"],
            tool_preferences=["code_editor", "terminal", "git"],
            communication_style="professional",
            problem_solving_approach="systematic",
            collaboration_style="mentoring"
        ),
        AgentConfig(
            agent_id="data_scientist",
            name="Data Scientist",
            description="Expert in data analysis, statistics, and machine learning",
            agent_type=AgentType.EXPERT.value,
            behavior_pattern="ANALYTICAL",
            expertise_domains=["data_analysis", "machine_learning"],
            tool_preferences=["python", "pandas", "matplotlib", "jupyter"],
            communication_style="analytical",
            problem_solving_approach="data_driven",
            collaboration_style="consultative"
        ),
        AgentConfig(
            agent_id="junior_developer",
            name="Junior Developer",
            description="Entry-level developer learning best practices",
            agent_type=AgentType.LEARNER.value,
            behavior_pattern="LEARNING",
            expertise_domains=["web_development"],
            tool_preferences=["code_editor"],
            communication_style="curious",
            problem_solving_approach="trial_and_error",
            collaboration_style="asking_questions"
        )
    ]
    
    return domains, tools, agents


async def create_simulation_configs():
    """Create example simulation configurations."""
    
    # Environment configuration
    env_config = EnvironmentConfig(
        environment_type="development_workspace",
        resources={
            "memory": "8GB",
            "cpu": "4 cores",
            "storage": "100GB"
        },
        tools_available=["code_editor", "terminal", "python", "git"],
        constraints={
            "max_execution_time": 300,
            "max_file_size": "10MB"
        }
    )
    
    # Simulation configurations
    simulations = [
        SimulationConfig(
            simulation_id="web_dev_collaboration",
            name="Web Development Collaboration",
            description="Senior and junior developers collaborating on a React project",
            agent_configs=[
                AgentConfig(agent_id="senior_developer", name="Senior Developer", description="desc", agent_type="EXPERT", behavior_pattern="COLLABORATIVE", expertise_domains=["web_development"], tool_preferences=["code_editor"], communication_style="professional", problem_solving_approach="systematic", collaboration_style="mentoring"), # Full AgentConfig required
                AgentConfig(agent_id="junior_developer", name="Junior Developer", description="desc", agent_type="LEARNER", behavior_pattern="LEARNING", expertise_domains=["web_development"], tool_preferences=["code_editor"], communication_style="curious", problem_solving_approach="trial_and_error", collaboration_style="asking_questions")
            ],
            environment_config=env_config,
            max_turns=20,
            timeout=600,
            scenario="Create a responsive React component with proper error handling"
        ),
        SimulationConfig(
            simulation_id="data_analysis_task",
            name="Data Analysis Task",
            description="Data scientist performing analysis with Python tools",
            agent_configs=[
                AgentConfig(agent_id="data_scientist", name="Data Scientist", description="desc", agent_type="EXPERT", behavior_pattern="ANALYTICAL", expertise_domains=["data_analysis"], tool_preferences=["python"], communication_style="analytical", problem_solving_approach="data_driven", collaboration_style="consultative")
            ],
            environment_config=env_config,
            max_turns=15,
            timeout=450,
            scenario="Analyze a dataset and create visualizations"
        )
    ]
    
    return simulations


async def create_evaluation_config():
    """Create example evaluation configuration."""
    
    rubric = EvaluationRubric(
        dimensions=[
            {
                "name": "tool_usage_effectiveness",
                "description": "How effectively tools are used to solve the problem",
                "weight": 0.3,
                "criteria": [
                    "Appropriate tool selection",
                    "Correct tool usage",
                    "Efficient workflow"
                ]
            },
            {
                "name": "problem_solving_quality",
                "description": "Quality of the problem-solving approach and solution",
                "weight": 0.3,
                "criteria": [
                    "Clear problem understanding",
                    "Logical solution approach",
                    "Correct final result"
                ]
            },
            {
                "name": "collaboration_effectiveness",
                "description": "Effectiveness of multi-agent collaboration",
                "weight": 0.2,
                "criteria": [
                    "Clear communication",
                    "Task coordination",
                    "Knowledge sharing"
                ]
            },
            {
                "name": "code_quality",
                "description": "Quality of generated code and documentation",
                "weight": 0.2,
                "criteria": [
                    "Code readability",
                    "Best practices",
                    "Documentation"
                ]
            }
        ]
    )
    
    return EvaluationConfig(
        evaluation_id="comprehensive_evaluation",
        name="Comprehensive Tool Usage Evaluation",
        description="Multi-dimensional evaluation of tool usage and collaboration",
        rubric=rubric,
        llm_model="gemini-2.5-flash-lite-preview-06-07", # Updated model name
        temperature=0.1,
        max_tokens=1000
    )


async def create_export_config():
    """Create example export configuration."""
    
    return DataExportConfig(
        formats=["json", "jsonl", "csv"],
        include_metadata=True,
        include_evaluations=True,
        split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
        compression=True,
        metadata_fields=[
            "simulation_id",
            "domain",
            "agents",
            "tools_used",
            "evaluation_scores"
        ]
    )


async def run_example_pipeline():
    """Run the complete Kimi-K2 pipeline with example configurations."""
    
    print("🚀 Starting Kimi-K2 Agentic Data Synthesis System Example (LangGraph Integrated)") # Updated message
    print("=" * 60)
    
    # LLM Configuration (using a dummy config for demonstration)
    llm_config = {"model": "gemini-2.5-flash-lite-preview-06-07", "api_key": "YOUR_API_KEY"} # Placeholder

    # Initialize the system with llm_config
    system = AgenticDataSynthesisSystem(
        output_dir="example_output",
        log_level="INFO",
        llm_config=llm_config
    )
    
    try:
        # Create configurations
        print("📋 Creating configurations...")
        domains, tools, agents = await create_example_configs()
        simulations = await create_simulation_configs()
        evaluation_config = await create_evaluation_config()
        export_config = await create_export_config()
        
        # Setup system components
        print("⚙️  Setting up system components...")
        system.setup_domains(domains)
        system.setup_tools(tools)
        system.setup_agents(agents) # Now creates KimiK2ConversableAgent instances
        
        # Run the full pipeline
        print("🔄 Running full pipeline...")
        # run_full_pipeline now returns results based on LangGraph state dicts
        results = await system.run_full_pipeline(
            simulation_configs=simulations,
            evaluation_config=evaluation_config,
            export_config=export_config,
            quality_threshold=0.7,
            max_concurrent_simulations=1 # Reduced for easier debugging
        )
        
        # Display results
        print("\n📊 Pipeline Results:")
        print("-" * 30)
        for key, value in results.items():
            if key != "export_paths":
                print(f"{key}: {value}")
        
        print("\n📁 Export Paths:")
        print("-" * 30)
        for format_name, file_path in results["export_paths"].items():
            print(f"{format_name}: {file_path}")
        
        # Display system statistics
        print("\n📈 System Statistics:")
        print("-" * 30)
        stats = system.get_system_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print("\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running example: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        # raise # Re-raise if you want the program to exit on error
    
    finally:
        # Cleanup
        system.cleanup()


async def main():
    """Main function to run the example."""
    await run_example_pipeline()


if __name__ == "__main__":
    asyncio.run(main()) 