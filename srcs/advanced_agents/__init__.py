"""
Advanced AI Agents Module

This module contains highly sophisticated AI agents that go beyond traditional 
basic and enterprise agents. These agents feature self-improvement capabilities,
meta-learning, and evolutionary mechanisms.

Available Agents:
- EvolutionaryAIArchitectAgent: Self-improving AI that designs and evolves AI architectures
- DecisionAgent: Mobile interaction-based automatic decision system
"""

try:
    from .evolutionary_ai_architect_agent import EvolutionaryAIArchitectAgent
except ImportError as e:
    print(f"Warning: Could not import EvolutionaryAIArchitectAgent: {e}")
    EvolutionaryAIArchitectAgent = None

try:
    from .decision_agent import DecisionAgent
except ImportError as e:
    print(f"Warning: Could not import DecisionAgent: {e}")
    DecisionAgent = None

__all__ = [
    'EvolutionaryAIArchitectAgent',
    'DecisionAgent'
]

__version__ = '1.0.0' 