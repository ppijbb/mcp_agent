"""env_settings
Centralized helper for environment-variable access across Product Planner Agent.

* automatically loads a .env file at project root (if present)
* provides typed accessor functions and optional masking utilities
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load .env once at import time (idempotent)
# -----------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # up to repository root
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def get(key: str, default: Optional[str] = None, *, required: bool = False) -> str | None:
    """Retrieve an environment variable.

    Args:
        key: Name of the environment variable.
        default: Value to return if variable is not set.
        required: If True and the variable is missing, raises MissingEnvError.
    """
    from srcs.product_planner_agent.utils.errors import MissingEnvError

    value = os.getenv(key, default)
    if required and value is None:
        raise MissingEnvError(f"Required environment variable '{key}' is not set.")
    return value


def mask(value: str, *, visible: int = 4) -> str:
    """Return a masked version of a sensitive value (e.g., API key)."""
    if value is None:
        return "<NONE>"
    if len(value) <= visible:
        return "*" * len(value)
    return value[:visible] + "*" * (len(value) - visible)


__all__ = ["get", "mask"] 