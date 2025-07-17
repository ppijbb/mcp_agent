"""
Test file for the Kimi-K2 Agentic Data Synthesis System.

This file contains basic tests to verify that the system components
are working correctly.
"""

import asyncio
import pytest
from typing import List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kimi_k2_agentic_data_synthesis.system.agentic_data_synthesis_system import AgenticDataSynthesisSystem
from kimi_k2_agentic_data_synthesis.models.domain import DomainConfig, DomainCategory
from kimi_k2_agentic_data_synthesis.models.tool import ToolConfig, ToolType
from kimi_k2_agentic_data_synthesis.models.agent import AgentConfig, AgentType, BehaviorPattern
from kimi_k2_agentic_data_synthesis.models.simulation import SimulationConfig, EnvironmentConfig
from kimi_k2_agentic_data_synthesis.models.evaluation import EvaluationConfig, EvaluationRubric
from kimi_k2_agentic_data_synthesis.models.data import DataExportConfig


def test_system_initialization():
    """Test that the system initializes correctly."""
    system = AgenticDataSynthesisSystem(
        output_dir="test_output",
        log_level="INFO"
    )
    
    assert system.domain_manager is not None
    assert system.tool_registry is not None
    assert system.agent_factory is not None
    assert system.simulation_engine is not None
    assert system.llm_judge is not None
    assert system.quality_filter is not None
    assert system.data_generator is not None
    
    # Test system stats
    stats = system.get_system_stats()
    assert isinstance(stats, dict)
    assert "domains" in stats
    assert "tools" in stats
    assert "agents" in stats
    
    # Cleanup
    system.cleanup()


def test_domain_setup():
    """Test domain setup functionality."""
    system = AgenticDataSynthesisSystem()
    
    # Create test domain
    domain_config = DomainConfig(
        domain_id="test_domain",
        name="Test Domain",
        description="A test domain for testing",
        domain_type=DomainCategory.TECHNOLOGY,
        scenarios=["Test scenario 1", "Test scenario 2"],
        complexity_levels=["easy", "medium"],
        required_tools=["test_tool"]
    )
    
    # Setup domain
    system.setup_domains([domain_config])
    
    # Verify domain was added
    domains = system.domain_manager.get_all_domains()
    assert len(domains) == 1
    assert domains[0].domain_id == "test_domain"
    
    system.cleanup()


def test_tool_setup():
    """Test tool setup functionality."""
    system = AgenticDataSynthesisSystem()
    
    # Create test tool
    tool_config = ToolConfig(
        tool_id="test_tool",
        name="Test Tool",
        description="A test tool for testing",
        tool_type=ToolType.SYNTHETIC,
        synthetic_behavior={
            "execution_time": "0.1-0.5s",
            "error_rate": "0.01",
            "output_format": "text"
        },
        parameters={"input": "string"},
        usage_examples=["Test usage"]
    )
    
    # Setup tool
    system.setup_tools([tool_config])
    
    # Verify tool was registered
    tools = system.tool_registry.get_all_tools()
    assert len(tools) == 1
    assert tools[0].tool_id == "test_tool"
    
    system.cleanup()


def test_agent_setup():
    """Test agent setup functionality."""
    system = AgenticDataSynthesisSystem()
    
    # Create test agent
    agent_config = AgentConfig(
        agent_id="test_agent",
        name="Test Agent",
        description="A test agent for testing",
        agent_type=AgentType.EXPERT,
        behavior_pattern=BehaviorPattern.COLLABORATIVE,
        expertise_domains=["test_domain"],
        tool_preferences=["test_tool"],
        communication_style="professional",
        problem_solving_approach="systematic",
        collaboration_style="mentoring"
    )
    
    # Setup agent
    system.setup_agents([agent_config])
    
    # Verify agent was created
    agents = system.agent_factory.get_all_agents()
    assert len(agents) == 1
    assert agents[0].agent_id == "test_agent"
    
    system.cleanup()


def test_configuration_loading():
    """Test configuration loading from file."""
    import json
    import tempfile
    import os
    
    # Create temporary config file
    config_data = {
        "domains": [
            {
                "domain_id": "test_domain",
                "name": "Test Domain",
                "description": "A test domain",
                "domain_type": "TECHNICAL",
                "scenarios": ["Test scenario"],
                "complexity_levels": ["easy"],
                "required_tools": ["test_tool"]
            }
        ],
        "tools": [
            {
                "tool_id": "test_tool",
                "name": "Test Tool",
                "description": "A test tool",
                "tool_type": "SYNTHETIC",
                "synthetic_behavior": {
                    "execution_time": "0.1-0.5s",
                    "error_rate": "0.01",
                    "output_format": "text"
                },
                "parameters": {"input": "string"},
                "usage_examples": ["Test usage"]
            }
        ],
        "agents": [
            {
                "agent_id": "test_agent",
                "name": "Test Agent",
                "description": "A test agent",
                "agent_type": "EXPERT",
                "behavior_pattern": "COLLABORATIVE",
                "expertise_domains": ["test_domain"],
                "tool_preferences": ["test_tool"],
                "communication_style": "professional",
                "problem_solving_approach": "systematic",
                "collaboration_style": "mentoring"
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name
    
    try:
        # Initialize system with config
        system = AgenticDataSynthesisSystem(config_path=config_path)
        
        # Verify components were loaded
        domains = system.domain_manager.get_all_domains()
        tools = system.tool_registry.get_all_tools()
        agents = system.agent_factory.get_all_agents()
        
        assert len(domains) == 1
        assert len(tools) == 1
        assert len(agents) == 1
        
        system.cleanup()
        
    finally:
        # Cleanup temp file
        os.unlink(config_path)


@pytest.mark.asyncio
async def test_simulation_creation():
    """Test simulation configuration creation."""
    system = AgenticDataSynthesisSystem()
    
    # Setup basic components
    domain_config = DomainConfig(
        domain_id="test_domain",
        name="Test Domain",
        description="A test domain",
        domain_type=DomainType.TECHNICAL,
        scenarios=["Test scenario"],
        complexity_levels=["easy"],
        required_tools=["test_tool"]
    )
    
    tool_config = ToolConfig(
        tool_id="test_tool",
        name="Test Tool",
        description="A test tool",
        tool_type=ToolType.SYNTHETIC,
        synthetic_behavior={
            "execution_time": "0.1-0.5s",
            "error_rate": "0.01",
            "output_format": "text"
        },
        parameters={"input": "string"},
        usage_examples=["Test usage"]
    )
    
    agent_config = AgentConfig(
        agent_id="test_agent",
        name="Test Agent",
        description="A test agent",
        agent_type=AgentType.EXPERT,
        behavior_pattern=BehaviorPattern.COLLABORATIVE,
        expertise_domains=["test_domain"],
        tool_preferences=["test_tool"],
        communication_style="professional",
        problem_solving_approach="systematic",
        collaboration_style="mentoring"
    )
    
    system.setup_domains([domain_config])
    system.setup_tools([tool_config])
    system.setup_agents([agent_config])
    
    # Create simulation config
    env_config = EnvironmentConfig(
        environment_type="test_workspace",
        tools_available=["test_tool"],
        resources={"memory": "4GB", "cpu": "2 cores"},
        constraints={"max_execution_time": 60}
    )
    
    simulation_config = SimulationConfig(
        simulation_id="test_simulation",
        name="Test Simulation",
        description="A test simulation",
        agent_configs=[agent_config],
        environment_config=env_config,
        max_turns=5,
        timeout=120,
        scenario="Test scenario"
    )
    
    # Verify simulation config is valid
    assert simulation_config.simulation_id == "test_simulation"
    assert len(simulation_config.agent_configs) == 1
    assert simulation_config.environment_config.environment_type == "test_workspace"
    
    system.cleanup()


@pytest.mark.asyncio
async def test_evaluation_config_creation():
    """Test evaluation configuration creation."""
    rubric = EvaluationRubric(
        dimensions=[
            {
                "name": "test_dimension",
                "description": "A test evaluation dimension",
                "weight": 1.0,
                "criteria": ["Test criterion 1", "Test criterion 2"]
            }
        ]
    )
    
    evaluation_config = EvaluationConfig(
        evaluation_id="test_evaluation",
        name="Test Evaluation",
        description="A test evaluation",
        rubric=rubric,
        llm_model="gemini-2.5-flash-lite-preview-06-07",
        temperature=0.1,
        max_tokens=500
    )
    
    # Verify evaluation config is valid
    assert evaluation_config.evaluation_id == "test_evaluation"
    assert evaluation_config.llm_model == "gemini-2.5-flash-lite-preview-06-07"
    assert len(evaluation_config.rubric.dimensions) == 1


def test_export_config_creation():
    """Test export configuration creation."""
    export_config = DataExportConfig(
        formats=["json", "jsonl"],
        include_metadata=True,
        include_evaluations=True,
        split_ratios={"train": 0.8, "validation": 0.2},
        compression=False,
        metadata_fields=["simulation_id", "domain"]
    )
    
    # Verify export config is valid
    assert "json" in export_config.formats
    assert "jsonl" in export_config.formats
    assert export_config.include_metadata is True
    assert export_config.split_ratios["train"] == 0.8


def test_system_cleanup():
    """Test system cleanup functionality."""
    system = AgenticDataSynthesisSystem()
    
    # Add some test data
    system.active_simulations["test_sim"] = None
    system.generated_data.append(None)
    system.evaluation_results.append(None)
    
    # Verify data exists
    assert len(system.active_simulations) == 1
    assert len(system.generated_data) == 1
    assert len(system.evaluation_results) == 1
    
    # Cleanup
    system.cleanup()
    
    # Verify data was cleared
    assert len(system.active_simulations) == 0
    # Note: generated_data and evaluation_results are not cleared in cleanup()
    # as they represent persistent data


if __name__ == "__main__":
    # Run basic tests
    print("Running Kimi-K2 System Tests...")
    
    test_system_initialization()
    print("✅ System initialization test passed")
    
    test_domain_setup()
    print("✅ Domain setup test passed")
    
    test_tool_setup()
    print("✅ Tool setup test passed")
    
    test_agent_setup()
    print("✅ Agent setup test passed")
    
    test_configuration_loading()
    print("✅ Configuration loading test passed")
    
    asyncio.run(test_simulation_creation())
    print("✅ Simulation creation test passed")
    
    asyncio.run(test_evaluation_config_creation())
    print("✅ Evaluation config creation test passed")
    
    test_export_config_creation()
    print("✅ Export config creation test passed")
    
    test_system_cleanup()
    print("✅ System cleanup test passed")
    
    print("\n🎉 All tests passed!") 