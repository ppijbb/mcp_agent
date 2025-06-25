"""
Multi-Persona Dialogue Agent System
===================================

This module implements a reflexive agent system where multiple, distinct AI personas
engage in a structured dialogue to explore a topic from various viewpoints,
leading to a more robust and comprehensive conclusion.

Main Components:
- MultiPersonaDialogueAgent: The main entry point to run the dialogue simulation.
- DialogueManager: Orchestrates the turn-by-turn conversation flow.
- Personas: Defines the roles and instructions for each participating agent 
  (Advocate, Critic, Skeptic, Synthesizer, MetaObserver).
"""

from .multi_persona_agent import MultiPersonaDialogueAgent
from .dialogue_manager import DialogueManager
from .personas import (
    get_advocate_persona,
    get_critic_persona,
    get_skeptic_persona,
    get_synthesizer_persona,
    get_meta_observer_persona,
)

__all__ = [
    "MultiPersonaDialogueAgent",
    "DialogueManager",
    "get_advocate_persona",
    "get_critic_persona",
    "get_skeptic_persona",
    "get_synthesizer_persona",
    "get_meta_observer_persona",
] 