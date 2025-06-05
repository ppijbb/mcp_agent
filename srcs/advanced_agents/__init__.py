"""
Advanced AI Agents Module

This module contains highly sophisticated AI agents that go beyond traditional 
basic and enterprise agents. These agents feature self-improvement capabilities,
meta-learning, and evolutionary mechanisms.

Available Agents:
- EvolutionaryAIArchitectAgent: Self-improving AI that designs and evolves AI architectures
"""

try:
    from .evolutionary_ai_architect_agent import EvolutionaryAIArchitectAgent
except ImportError as e:
    print(f"Warning: Could not import EvolutionaryAIArchitectAgent: {e}")
    EvolutionaryAIArchitectAgent = None

__all__ = [
    'EvolutionaryAIArchitectAgent'
]

__version__ = '1.0.0' 