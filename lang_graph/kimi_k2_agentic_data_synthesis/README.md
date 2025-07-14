# Kimi-K2 Agentic Data Synthesis System

A large-scale agentic data synthesis system for generating high-quality training data for tool usage learning, inspired by ACEBench pipeline.

## Overview

The Kimi-K2 system is designed to generate comprehensive training data for AI agents that need to learn how to use tools effectively. It creates realistic multi-agent simulations where different types of agents collaborate to solve problems using various tools, then evaluates the quality of these interactions and exports the results as training data.

## Key Features

- **Multi-Domain Support**: Supports various domains like web development, data analysis, research, and more
- **Tool Registry**: Manages both MCP (Model Context Protocol) tools and synthetic tools
- **Agent Factory**: Creates diverse agents with different expertise levels and behavior patterns
- **Simulation Engine**: Runs large-scale multi-agent simulations with realistic interactions
- **Quality Evaluation**: Uses LLM judges to evaluate simulation quality across multiple dimensions
- **Data Generation**: Exports high-quality training data in multiple formats (JSON, JSONL, CSV)
- **Scalable Architecture**: Supports concurrent simulations and batch processing

## Architecture

```
Kimi-K2 System
├── Core Components
│   ├── DomainManager - Manages domains and scenarios
│   ├── ToolRegistry - Manages MCP and synthetic tools
│   ├── AgentFactory - Creates diverse agents
│   ├── SimulationEngine - Runs multi-agent simulations
│   ├── EnvironmentManager - Manages virtual environments
│   └── UserAgentManager - Manages user agent interactions
├── Evaluation
│   ├── LLMJudgeSystem - Evaluates simulation quality
│   └── QualityFilter - Filters high-quality data
├── Data
│   └── DataGenerator - Generates and exports training data
└── Models
    ├── Domain Models - Domain and scenario definitions
    ├── Tool Models - Tool configurations and usage
    ├── Agent Models - Agent profiles and behaviors
    ├── Simulation Models - Simulation configurations and results
    ├── Evaluation Models - Evaluation rubrics and results
    └── Data Models - Training data structures
```

## Installation

The Kimi-K2 system is part of the lang_graph package. Make sure you have the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import asyncio
from lang_graph.kimi_k2_agentic_data_synthesis import AgenticDataSynthesisSystem

async def main():
    # Initialize the system
    system = AgenticDataSynthesisSystem(
        output_dir="generated_data",
        log_level="INFO"
    )
    
    # Setup domains, tools, and agents
    # (See example_usage.py for detailed configuration)
    
    # Run the complete pipeline
    results = await system.run_full_pipeline(
        simulation_configs=simulations,
        evaluation_config=evaluation_config,
        export_config=export_config,
        quality_threshold=0.7
    )
    
    print(f"Generated {results['high_quality_samples']} training samples")

asyncio.run(main())
```

### Running the Example

```bash
cd lang_graph/kimi_k2_agentic_data_synthesis
python -m example_usage
```

## Configuration

### Domain Configuration

```python
from lang_graph.kimi_k2_agentic_data_synthesis.models.domain import DomainConfig, DomainType

domain_config = DomainConfig(
    domain_id="web_development",
    name="Web Development",
    description="Web development and programming tasks",
    domain_type=DomainType.TECHNICAL,
    scenarios=[
        "Create a React component",
        "Debug JavaScript code",
        "Set up a Node.js server"
    ],
    complexity_levels=["beginner", "intermediate", "advanced"],
    required_tools=["code_editor", "terminal", "browser"]
)
```

### Tool Configuration

```python
from lang_graph.kimi_k2_agentic_data_synthesis.models.tool import ToolConfig, ToolType

tool_config = ToolConfig(
    tool_id="code_editor",
    name="Code Editor",
    description="Multi-language code editor with syntax highlighting",
    tool_type=ToolType.MCP,
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
)
```

### Agent Configuration

```python
from lang_graph.kimi_k2_agentic_data_synthesis.models.agent import AgentConfig, AgentType, BehaviorPattern

agent_config = AgentConfig(
    agent_id="senior_developer",
    name="Senior Developer",
    description="Experienced software developer with expertise in multiple languages",
    agent_type=AgentType.EXPERT,
    behavior_pattern=BehaviorPattern.COLLABORATIVE,
    expertise_domains=["web_development", "software_engineering"],
    tool_preferences=["code_editor", "terminal", "git"],
    communication_style="professional",
    problem_solving_approach="systematic",
    collaboration_style="mentoring"
)
```

### Simulation Configuration

```python
from lang_graph.kimi_k2_agentic_data_synthesis.models.simulation import SimulationConfig, EnvironmentConfig

simulation_config = SimulationConfig(
    simulation_id="web_dev_collaboration",
    name="Web Development Collaboration",
    description="Senior and junior developers collaborating on a React project",
    agent_configs=[
        AgentConfig(agent_id="senior_developer"),
        AgentConfig(agent_id="junior_developer")
    ],
    environment_config=EnvironmentConfig(
        environment_type="development_workspace",
        tools_available=["code_editor", "terminal", "python", "git"]
    ),
    max_turns=20,
    timeout=600,
    scenario="Create a responsive React component with proper error handling"
)
```

## Evaluation

The system uses LLM judges to evaluate simulation quality across multiple dimensions:

- **Tool Usage Effectiveness**: How effectively tools are used to solve problems
- **Problem Solving Quality**: Quality of the problem-solving approach and solution
- **Collaboration Effectiveness**: Effectiveness of multi-agent collaboration
- **Code Quality**: Quality of generated code and documentation

## Data Export

The system exports training data in multiple formats:

- **JSON**: Complete simulation data with metadata
- **JSONL**: Line-delimited JSON for streaming processing
- **CSV**: Tabular format for analysis

Data can be split into train/validation/test sets and includes:
- Agent interactions and tool usage
- Problem scenarios and solutions
- Evaluation scores and feedback
- Metadata and timestamps

## Advanced Usage

### Custom Domains

Create custom domains by extending the DomainConfig:

```python
custom_domain = DomainConfig(
    domain_id="custom_domain",
    name="Custom Domain",
    description="Your custom domain description",
    domain_type=DomainType.CUSTOM,
    scenarios=["Custom scenario 1", "Custom scenario 2"],
    complexity_levels=["easy", "medium", "hard"],
    required_tools=["custom_tool_1", "custom_tool_2"]
)
```

### Custom Tools

Register custom tools:

```python
custom_tool = ToolConfig(
    tool_id="custom_tool",
    name="Custom Tool",
    description="Your custom tool description",
    tool_type=ToolType.SYNTHETIC,
    synthetic_behavior={
        "execution_time": "0.5-1.5s",
        "error_rate": "0.02",
        "output_format": "json"
    },
    parameters={"input": "string", "options": "object"},
    usage_examples=["Example usage 1", "Example usage 2"]
)

system.tool_registry.register_tool(custom_tool)
```

### Batch Processing

Run multiple simulations concurrently:

```python
# Create multiple simulation configurations
simulation_configs = [
    SimulationConfig(...),
    SimulationConfig(...),
    SimulationConfig(...)
]

# Run batch with concurrency control
results = await system.run_simulation_batch(
    simulation_configs,
    max_concurrent=5
)
```

## Monitoring and Logging

The system provides comprehensive logging and monitoring:

```python
# Get system statistics
stats = system.get_system_stats()
print(f"Active simulations: {stats['active_simulations']}")
print(f"Generated data samples: {stats['generated_data']}")

# Monitor specific simulation
simulation_result = system.active_simulations["sim_123"]
print(f"Simulation status: {simulation_result.status}")
print(f"Turns completed: {len(simulation_result.turns)}")
```

## Error Handling

The system includes robust error handling:

```python
try:
    results = await system.run_full_pipeline(...)
except Exception as e:
    print(f"Pipeline failed: {e}")
    # Handle error appropriately
finally:
    system.cleanup()
```

## Performance Optimization

- Use appropriate `max_concurrent_simulations` based on your system resources
- Set reasonable `timeout` values for simulations
- Use `quality_threshold` to filter only high-quality data
- Consider using compression for large datasets

## Contributing

To contribute to the Kimi-K2 system:

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure all components are properly integrated

## License

This project is part of the lang_graph package and follows the same licensing terms.

## Support

For issues and questions:

1. Check the example usage in `example_usage.py`
2. Review the model definitions in the `models/` directory
3. Examine the core component implementations
4. Check the system logs for detailed error information 