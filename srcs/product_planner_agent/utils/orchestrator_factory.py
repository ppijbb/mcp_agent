"""Utility to create a pre-configured Orchestrator shared across Coordinators.
It applies:
1. Durable execution (`durable=True`) if `AGENT_DURABLE=1`.
2. Turn budget limits forwarded via `max_loops` / Planner `max_iterations`.
3. LLM caching (wrap with CachedLLM).
4. Optional MCPAggregator exposure of servers (`fetch`, `filesystem`).
"""
from __future__ import annotations

import os
from typing import Any, Callable

from srcs.basic_agents.workflow_orchestration import Orchestrator, OpenAIAugmentedLLM

from .cached_llm import CachedLLM

# Singleton pattern to reuse orchestrator instance
_ORCHESTRATOR_CACHE: Orchestrator | None = None

def get_orchestrator() -> Orchestrator:
    global _ORCHESTRATOR_CACHE
    if _ORCHESTRATOR_CACHE is not None:
        return _ORCHESTRATOR_CACHE

    # ---------- LLM Factory with caching ----------
    def llm_factory():
        base_llm = OpenAIAugmentedLLM(model="gpt-4o-mini")
        return CachedLLM(base_llm)

    # ---------- Planner with iteration limit ----------
    max_turns = int(os.getenv("AGENT_MAX_TURNS", 20))
    
    # ---------- Optional Aggregator for fetch+filesystem ----------
    aggregator = None

    _ORCHESTRATOR_CACHE = Orchestrator(
        llm_factory=llm_factory,
        available_agents=[],  # Agents will register later
        plan_type='react',
        durable=bool(int(os.getenv("AGENT_DURABLE", "0"))),
        max_loops=max_turns,
        server_registry=aggregator,
    )
    return _ORCHESTRATOR_CACHE


def orchestrator_factory() -> Callable[[], Orchestrator]:
    """Return a callable for deferred orchestrator creation (lazy)."""
    orch = get_orchestrator()
    return lambda: orch 