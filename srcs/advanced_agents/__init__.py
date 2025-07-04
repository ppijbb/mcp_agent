"""
Advanced AI Agents Module

This module contains highly sophisticated AI agents that go beyond traditional 
basic and enterprise agents. These agents feature self-improvement capabilities,
meta-learning, and evolutionary mechanisms.

Available Agents:
- EvolutionaryAIArchitectAgent: Self-improving AI that designs and evolves AI architectures
- DecisionAgent: Mobile interaction-based automatic decision system
- FinancialAnalystAgent: Financial analysis and reporting agent
- CodeInterpreterAgent: Code execution and interpretation agent
- DevOpsAgent: CI/CD and infrastructure management agent
"""

# EvolutionaryAIArchitectAgent moved to srcs/evolutionary_ai_architect/
# from srcs.evolutionary_ai_architect import EvolutionaryAIArchitectAgent

try:
    from .decision_agent import DecisionAgentMCP as DecisionAgent
except ImportError as e:
    print(f"Warning: Could not import DecisionAgent: {e}")
    DecisionAgent = None

try:
    from .graph_react_agent import GraphReActAgent
except ImportError as e:
    print(f"Warning: Could not import GraphReActAgent: {e}")
    GraphReActAgent = None

__all__ = [
    'initialize_enterprise_agents',
    'DecisionAgent',
    'FinancialAnalystAgent',
    'CodeInterpreterAgent',
    'GraphReActAgent'
]

__version__ = '1.0.0' 