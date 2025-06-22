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

from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.planner.react_planner import ReActPlanner
from mcp_agent.mcp.mcp_aggregator import MCPAggregator

from .cached_llm import CachedLLM

# Singleton pattern to reuse orchestrator instance
_ORCHESTRATOR_CACHE: Orchestrator | None = None

def get_orchestrator() -> Orchestrator:
    global _ORCHESTRATOR_CACHE
    if _ORCHESTRATOR_CACHE is not None:
        return _ORCHESTRATOR_CACHE

    # ---------- LLM Factory with caching ----------
    def llm_factory():
        base_llm = GoogleAugmentedLLM(model="gemini-2.0-flash-lite-001")
        return CachedLLM(base_llm)

    # ---------- Planner with iteration limit ----------
    max_turns = int(os.getenv("AGENT_MAX_TURNS", 20))
    planner = ReActPlanner(max_iterations=max_turns)

    # ---------- Optional Aggregator for fetch+filesystem ----------
    aggregator = None
    if os.getenv("AGENT_USE_AGGREGATOR", "0") == "1":
        try:
            aggregator = MCPAggregator(server_names=["fetch", "filesystem"])
        except Exception:
            aggregator = None  # Fallback if servers unavailable

    _ORCHESTRATOR_CACHE = Orchestrator(
        llm_factory=llm_factory,
        available_agents=[],  # Agents will register later
        planner=planner,
        durable=bool(int(os.getenv("AGENT_DURABLE", "0"))),
        max_loops=max_turns,
        server_registry=aggregator,
    )
    return _ORCHESTRATOR_CACHE


def orchestrator_factory() -> Callable[[], Orchestrator]:
    """Return a callable for deferred orchestrator creation (lazy)."""
    orch = get_orchestrator()
    return lambda: orch 