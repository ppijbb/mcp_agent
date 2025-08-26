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

## NEW: MCP-Based Tool Selection Learning System

The system now includes a specialized MCP (Model Context Protocol) training data generation system focused on **tool selection and function calling patterns**:

### MCP Communication Simulator
- **Realistic MCP Server Simulation**: Simulates various MCP servers (file system, database, terminal, API client, etc.)
- **Function Call Patterns**: Generates realistic function calls with proper parameters and error handling
- **Communication Statistics**: Tracks success rates, response times, and error patterns

### Intelligent Tool Selection
- **Intent Analysis**: Analyzes user requests to understand intent and domain context
- **Tool Requirement Mapping**: Maps intents to required and optional tools
- **Fitness Evaluation**: Evaluates tool fitness based on capabilities and context
- **Decision Reasoning**: Generates detailed reasoning for tool selection decisions

### High-Quality Training Data
- **No Fallback Policy**: Strict quality filtering - failed data generation is skipped, not stored
- **Multi-Dimensional Quality Metrics**: Evaluates tool selection accuracy, function call success, parameter accuracy, and more
- **Context-Aware Generation**: Creates realistic scenarios with proper workspace context and tool availability

## Architecture

```
Kimi-K2 System
├── Core Components
│   ├── DomainManager - Manages domains and scenarios
│   ├── ToolRegistry - Manages MCP and synthetic tools
│   ├── AgentFactory - Creates diverse agents
│   ├── SimulationEngine - Runs multi-agent simulations
│   ├── EnvironmentManager - Manages virtual environments
│   ├── UserAgentManager - Manages user agent interactions
│   ├── MCPCommunicationSimulator - Simulates MCP server communication
│   └── ToolSelectionDecisionSimulator - Simulates intelligent tool selection
├── Evaluation
│   ├── LLMJudgeSystem - Evaluates simulation quality
│   └── QualityFilter - Filters high-quality data
├── Data
│   ├── DataGenerator - Generates and exports training data
│   └── MCPDataGenerator - Generates MCP-focused training data
└── Models
    ├── Domain Models - Domain and scenario definitions
    ├── Tool Models - Tool configurations and usage
    ├── Agent Models - Agent profiles and behaviors
    ├── Simulation Models - Simulation configurations and results
    ├── Evaluation Models - Evaluation rubrics and results
    ├── Data Models - Training data structures
    └── MCP Training Data Models - Tool selection and function call learning data
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

### NEW: MCP-Based Data Generation

```python
import asyncio
from lang_graph.kimi_k2_agentic_data_synthesis.data import MCPDataGenerator

async def main():
    # Initialize MCP data generator
    generator = MCPDataGenerator(output_directory="mcp_training_data")
    
    # Generate high-quality training data for tool selection learning
    batch = await generator.generate_training_batch(
        name="tool_selection_batch",
        description="Training data for MCP tool selection learning",
        num_tool_selection_samples=100,
        quality_threshold=0.8  # Strict quality filtering
    )
    
    if batch:
        print(f"Generated {batch.get_batch_size()} high-quality samples")
        print(f"Average quality: {batch.average_quality_score:.3f}")
        
        # Export the batch
        export_path = generator.export_batch(batch.id, format="json")
        print(f"Exported to: {export_path}")

asyncio.run(main())
```

### Running the Examples

```bash
cd lang_graph/kimi_k2_agentic_data_synthesis

# Run the original example
python -m example_usage

# Run the new MCP example
python -m example_mcp_usage
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

## NEW: MCP Tool Selection Learning

### Tool Selection Context

```python
from lang_graph.kimi_k2_agentic_data_synthesis.models.mcp_training_data import ToolSelectionContext

context = ToolSelectionContext(
    user_request="Create a new Python script for data analysis",
    current_workspace={
        "workspace_type": "development",
        "has_database": True,
        "system_access": True
    },
    available_tools=["code_editor", "terminal", "file_server", "database"],
    recent_tool_usage=["code_editor", "terminal"],
    domain_context="data_science",
    task_complexity="moderate",
    user_expertise_level="intermediate"
)
```

### Tool Selection Decision

```python
from lang_graph.kimi_k2_agentic_data_synthesis.models.mcp_training_data import ToolSelectionDecision

decision = ToolSelectionDecision(
    selected_tool="code_editor",
    selection_reasoning="Selected code_editor based on primary intent: code_development",
    alternative_tools_considered=["terminal", "file_server"],
    confidence_score=0.85,
    decision_time=0.5,
    context_factors={
        "intent_analysis": {...},
        "tool_requirements": {...},
        "tool_fitness_scores": {...}
    }
)
```

## Evaluation

The system uses LLM judges to evaluate simulation quality across multiple dimensions:

- **Tool Usage Effectiveness**: How effectively tools are used to solve problems
- **Problem Solving Quality**: Quality of the problem-solving approach and solution
- **Collaboration Effectiveness**: Effectiveness of multi-agent collaboration
- **Code Quality**: Quality of generated code and documentation

### NEW: MCP-Specific Quality Metrics

- **Tool Selection Accuracy**: How accurately the right tool is selected for the task
- **Function Call Success Rate**: Success rate of MCP function calls
- **Parameter Accuracy**: Accuracy of function parameter generation
- **Communication Reliability**: Reliability of MCP server communication
- **User Satisfaction**: User satisfaction with tool selection and execution

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

### NEW: MCP Training Data Structure

```json
{
  "id": "uuid",
  "user_request": "Create a new Python script for data analysis",
  "selection_context": {
    "user_request": "...",
    "current_workspace": {...},
    "available_tools": [...],
    "domain_context": "data_science",
    "task_complexity": "moderate"
  },
  "decision": {
    "selected_tool": "code_editor",
    "selection_reasoning": "...",
    "confidence_score": 0.85,
    "alternative_tools_considered": [...]
  },
  "function_call": {
    "function_name": "create_file",
    "parameters": {...},
    "validation_status": "valid"
  },
  "mcp_result": {
    "success": true,
    "response_data": {...},
    "execution_time": 1.2,
    "error_message": null
  },
  "success_metrics": {
    "overall": 0.88,
    "tool_selection_accuracy": 1.0,
    "function_call_success_rate": 1.0
  },
  "user_satisfaction": 0.9,
  "learning_objective": "Learn tool selection for code_development tasks",
  "common_mistakes": [...],
  "best_practices": [...]
}
```

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

### NEW: Custom MCP Servers

```python
from lang_graph.kimi_k2_agentic_data_synthesis.core import MCPCommunicationSimulator
from lang_graph.kimi_k2_agentic_data_synthesis.models.mcp_training_data import MCPToolType

mcp_simulator = MCPCommunicationSimulator()

# Register custom MCP server
mcp_simulator.register_server(
    name="custom_api_server",
    server_type=MCPToolType.API_CLIENT,
    capabilities=["custom_endpoint", "data_processing", "validation"]
)
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

### NEW: MCP System Monitoring

```python
# Get MCP communication statistics
mcp_stats = mcp_simulator.get_communication_statistics()
print(f"Total MCP calls: {mcp_stats['total_communications']}")
print(f"Success rate: {mcp_stats['overall_success_rate']:.2%}")

# Get tool selection statistics
tool_stats = tool_selection_simulator.get_decision_statistics()
print(f"Total decisions: {tool_stats['total_decisions']}")
print(f"Average confidence: {tool_stats['average_confidence']:.2f}")
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

### NEW: Strict Quality Filtering

The MCP data generation system implements strict quality filtering with **no fallback**:

```python
# Failed data generation is skipped, not stored
data_generator = MCPDataGenerator()
data_generator.min_quality_threshold = 0.8  # Strict threshold
data_generator.min_confidence_threshold = 0.7  # Minimum confidence

# Only high-quality data passes through
batch = await data_generator.generate_training_batch(
    name="high_quality_batch",
    num_tool_selection_samples=100,
    quality_threshold=0.8
)

# Failed or low-quality samples are automatically skipped
print(f"Generated {batch.get_batch_size()} samples (failed samples were skipped)")
```

## Performance Optimization

- Use appropriate `max_concurrent_simulations` based on your system resources
- Set reasonable `timeout` values for simulations
- Use `quality_threshold` to filter only high-quality data
- Consider using compression for large datasets

### NEW: MCP Performance Optimization

- **Quality Thresholds**: Set strict quality thresholds to avoid storing low-quality data
- **Batch Processing**: Generate data in batches for efficient processing
- **Parallel Generation**: Use async/await for concurrent data generation
- **Memory Management**: Automatically skip invalid data to conserve memory

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
2. Check the new MCP example in `example_mcp_usage.py`
3. Review the model definitions in the `models/` directory
4. Examine the core component implementations
5. Check the system logs for detailed error information 