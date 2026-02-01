"""Utility to create a pre-configured Orchestrator shared across Coordinators.
It applies:
1. Durable execution (`durable=True`) if `AGENT_DURABLE=1`.
2. Turn budget limits forwarded via `max_loops` / Planner `max_iterations`.
3. LLM caching (wrap with CachedLLM).
4. Optional MCPAggregator exposure of servers (`fetch`, `filesystem`).
5. Fallback support for LLM API errors (503, overloaded, etc.)
"""
from __future__ import annotations

import os
import logging
from typing import Callable

from srcs.basic_agents.workflow_orchestration import Orchestrator
from srcs.common.llm import create_fallback_llm_factory

from .cached_llm import CachedLLM

logger = logging.getLogger(__name__)

# Singleton pattern to reuse orchestrator instance
_ORCHESTRATOR_CACHE: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _ORCHESTRATOR_CACHE
    if _ORCHESTRATOR_CACHE is not None:
        return _ORCHESTRATOR_CACHE

    # ---------- LLM Factory with caching and fallback ----------
    def llm_factory():
        # Fallback이 가능한 LLM factory 사용 (common 모듈)
        fallback_llm_factory = create_fallback_llm_factory(
            primary_model="gemini-2.5-flash-lite",
            logger_instance=logger
        )
        base_llm = fallback_llm_factory()
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
