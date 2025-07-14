import sys
import os

# Add the parent directory to the path to enable absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kimi_k2_agentic_data_synthesis.models.domain import DomainConfig, DomainCategory
from kimi_k2_agentic_data_synthesis.models.tool import ToolConfig, ToolType
from kimi_k2_agentic_data_synthesis.models.agent import AgentConfig, AgentRole
from kimi_k2_agentic_data_synthesis.core.domain_manager import DomainManager
from kimi_k2_agentic_data_synthesis.core.tool_registry import ToolRegistry
from kimi_k2_agentic_data_synthesis.core.agent_factory import AgentFactory
from kimi_k2_agentic_data_synthesis.core.simulation_engine import SimulationEngine

# Mock LLM Config (as it's required by AgentFactory and SimulationEngine)
llm_config = {
    "model": "gemini-2.5-flash-lite-preview-06-07",
    "temperature": 0.7,
    "max_tokens": 1000
}

# Instantiate managers (simplified, just enough to get SimulationEngine)
domain_manager = DomainManager()
tool_registry = ToolRegistry()
agent_factory = AgentFactory(llm_config=llm_config, tool_registry=tool_registry)

# Create a dummy domain and tool for agent factory init
domain_config = DomainConfig(
    domain_id="test_domain", # Corrected: use domain_id instead of id
    name="Test Domain",
    description="A test domain for graph visualization.",
    scenarios=["test_scenario"]
)
domain_manager.create_domain(
    name=domain_config.name,
    description=domain_config.description,
    category=DomainCategory.TECHNOLOGY, # Correctly pass the enum here
    # scenarios=domain_config.scenarios # Scenarios are added separately, or in create_scenario
)
tool_config = ToolConfig(tool_id="dummy_tool_id", name="dummy_tool", description="A dummy tool", parameters={})
tool_registry.register_tool(
    name=tool_config.name,
    type=ToolType.SYNTHETIC, # Using a placeholder type
    description=tool_config.description,
    parameters=tool_config.parameters
)

agent_config_1 = AgentConfig(
    agent_id="agent_alice", # Corrected: use agent_id instead of id
    name="Alice",
    role=AgentRole.EXPERT,
    description="An agent for testing.",
    tool_preferences=["dummy_tool"],
    llm_config=llm_config
)
agent_factory.create_agent(agent_config_1)


# Initialize SimulationEngine
simulation_engine = SimulationEngine(
    domain_manager=domain_manager,
    tool_registry=tool_registry,
    agent_factory=agent_factory,
    llm_config=llm_config
)

# Get the Mermaid graph string
mermaid_graph = simulation_engine.app.get_graph().draw_mermaid()

print("LangGraph Mermaid Diagram:")
print(mermaid_graph) 