"""
Kimi-K2 Agentic Data Synthesis System

A large-scale agentic data synthesis system for generating high-quality training data
for tool usage learning, inspired by ACEBench pipeline.

This system provides:
- Domain management for various domains
- Tool registry for MCP and synthetic tools
- Agent factory for creating diverse agents
- Simulation engine for large-scale scenarios
- Quality evaluation and filtering
- Data generation and export
"""

__version__ = "1.0.0"
__author__ = "Kimi-K2 Team"

from .core.domain_manager import DomainManager
from .core.tool_registry import ToolRegistry
from .core.agent_factory import AgentFactory
from .core.simulation_engine import SimulationEngine
from .core.environment_manager import EnvironmentManager
from .core.user_agent_manager import UserAgentManager
from .evaluation.llm_judge import LLMJudgeSystem
from .evaluation.quality_filter import QualityFilter
from .data.data_generator import DataGenerator
from .system.agentic_data_synthesis_system import AgenticDataSynthesisSystem

__all__ = [
    "DomainManager",
    "ToolRegistry", 
    "AgentFactory",
    "SimulationEngine",
    "EnvironmentManager",
    "UserAgentManager",
    "LLMJudgeSystem",
    "QualityFilter",
    "DataGenerator",
    "AgenticDataSynthesisSystem"
] 