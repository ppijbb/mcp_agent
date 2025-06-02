"""
MCP Agent System

A comprehensive multi-agent system for enterprise automation and intelligence.
Includes basic agents, enterprise-level agents, and utility modules.
"""

__version__ = "1.0.0"
__author__ = "MCP Agent Team"

# Import main modules for easy access
from .basic_agents.agent import Agent
from .basic_agents.swarm import *
from .basic_agents.workflow_orchestration import *

__all__ = [
    "Agent",
] 