"""CachedLLM
A lightweight wrapper around an AugmentedLLM implementing a simple in-process prompt→response cache
so that identical LLM calls within the same run do not consume additional tokens.

NOTE: For production use Redis/LRU cache + TTL. This is a minimal implementation.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Tuple
from collections import OrderedDict
import time, logging, os

class _LRUCache(OrderedDict):
    """Very small manual LRU cache for <128 items."""
    def __init__(self, maxsize: int = 128):
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

class CachedLLM:
    """Proxy object that wraps an existing AugmentedLLM instance and applies LRU caching."""

    def __init__(self, base_llm, cache_size: int = 128):
        self._llm = base_llm
        self._cache: _LRUCache = _LRUCache(maxsize=cache_size)

    def __getattr__(self, item):
        # Delegate attribute access transparently (e.g. model name etc.)
        return getattr(self._llm, item)

    async def generate_str(self, prompt: str, *, request_params: Any = None, stream: bool | None = None):
        """Cache-aware generate_str. If stream=True, just forward to base LLM (no cache)."""
        # If streaming requested, bypass cache (streaming responses are unique)
        if stream:
            # Delegate streaming call directly
            return await self._llm.generate_str(prompt, request_params=request_params, stream=True)

        key: Tuple[str, str] = (prompt, str(request_params))
        if key in self._cache:
            return self._cache[key]

        t0 = time.perf_counter()
        result = await self._llm.generate_str(prompt, request_params=request_params)
        elapsed = time.perf_counter() - t0

        # simple token count heuristic (words≈tokens)
        token_est = len(result.split())
        if os.getenv("AGENT_LOG_TOKEN", "0") == "1":
            logging.getLogger("llm").info(f"LLM call tokens≈{token_est}, elapsed={elapsed:.2f}s")

        self._cache[key] = result
        return result 