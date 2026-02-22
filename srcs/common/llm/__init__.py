"""
Common LLM Utilities Module

Provides common LLM factory and fallback mechanisms for consistent
error handling across all agents.

Functions:
    get_best_fallback_models: Get list of best fallback models
    create_fallback_llm_factory: Create fallback LLM factory
    create_fallback_llm_for_agents: Create fallback LLM for agents
    create_fallback_orchestrator_llm_factory: Create orchestrator fallback factory
    try_fallback_orchestrator_execution: Try fallback execution for orchestrator
"""

from .fallback_llm import (
    get_best_fallback_models,
    create_fallback_llm_factory,
    create_fallback_llm_for_agents,
    create_fallback_orchestrator_llm_factory,
    try_fallback_orchestrator_execution
)

__all__ = [
    "get_best_fallback_models",
    "create_fallback_llm_factory",
    "create_fallback_llm_for_agents",
    "create_fallback_orchestrator_llm_factory",
    "try_fallback_orchestrator_execution",
]
