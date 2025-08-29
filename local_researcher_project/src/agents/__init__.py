"""
Agents Package

This package contains modules for managing research agents
and their execution workflows.
"""

from .agent_manager import AgentManager, AgentStatus, AgentType
from .open_deep_research_adapter import OpenDeepResearchAdapter

__all__ = [
    'AgentManager',
    'AgentStatus',
    'AgentType',
    'OpenDeepResearchAdapter'
]
