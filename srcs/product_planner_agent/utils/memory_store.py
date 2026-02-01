"""Redis-backed vector memory store (minimal).
Requires `redis` and `redisvl` packages if VECTOR mode needed.
Environment:
    REDIS_URL=redis://localhost:6379
"""
from __future__ import annotations
import json
import logging
from typing import Any, Dict

logger = logging.getLogger("memory")


class MemoryStore:
    def __init__(self):
        # Simplified: always use in-process dict; Redis removed for lightweight operation
        self._mem: Dict[str, str] = {}
        logger.info("Using in-process MemoryStore (Redis disabled)")

    def put(self, key: str, value: Any):
        data = json.dumps(value, ensure_ascii=False)
        self._mem[key] = data

    def get(self, key: str) -> Any | None:
        data = self._mem.get(key)
        if data:
            return json.loads(data)
        return None
