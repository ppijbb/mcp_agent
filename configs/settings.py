"""Light-weight runtime configuration helpers used by Streamlit UI pages.

This module intentionally keeps a *small* surface-area.  All heavy-weight YAML
loading and validation logic lives in ``srcs.core.config.loader``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

# ---------------------------------------------------------------------------
# Public API – *ONLY* expose what the UI layer really needs
# ---------------------------------------------------------------------------
__all__ = ["get_reports_path"]

# Workspace root – resolves correctly when executed via Streamlit / CLI
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_REPORTS_DIR = _PROJECT_ROOT / "reports"

# Lazily import the validated AppConfig instance so we can piggy-back on it
try:
    from srcs.core.config.loader import settings as _settings  # noqa: WPS433 – runtime import is intentional
except Exception as exc:  # pragma: no cover
    # If core settings cannot be imported we still provide best-effort paths so
    # that the UI can run in degraded mode (e.g. during development).
    _settings = None  # type: ignore[misc]


def get_reports_path(agent_name: str) -> str:
    """Return (and auto-create) the reports directory for *agent_name*.

    1. ``$MCP_REPORTS_DIR`` takes precedence when defined – allowing users to
       customise output locations without code changes.
    2. Falls back to the project-local ``reports/`` folder.
    """
    base_dir: Path = Path(os.getenv("MCP_REPORTS_DIR", str(_DEFAULT_REPORTS_DIR)))
    path: Path = base_dir / agent_name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


# ---------------------------------------------------------------------------
# Make the helper available as an attribute on the *settings* singleton so that
# pages importing ``from srcs.core.config.loader import settings`` continue to
# work unchanged – without violating Pydantic's attribute-setting rules.
# ---------------------------------------------------------------------------
if _settings is not None and "get_reports_path" not in _settings.__dict__:
    # Directly mutating __dict__ circumvents Pydantic's validation hooks while
    # still exposing the attribute via normal attribute access (`settings.foo`).
    _settings.__dict__["get_reports_path"] = get_reports_path 