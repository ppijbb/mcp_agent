"""
Shared pytest configuration for the MCP Agent test suite.
"""
import os
import sys
from pathlib import Path

import pytest

# Make the repository root importable regardless of how pytest is invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Browser automation tests need a real browser, a display and network access.
# They are skipped by default so that `pytest tests/` stays deterministic;
# opt in with MCP_RUN_BROWSER_TESTS=1.
BROWSER_TESTS_ENABLED = os.getenv("MCP_RUN_BROWSER_TESTS", "").lower() in {"1", "true", "yes"}


def pytest_collection_modifyitems(items):
    """Skip browser automation tests unless they are explicitly enabled."""
    if BROWSER_TESTS_ENABLED:
        return

    skip_browser = pytest.mark.skip(
        reason="browser automation tests require MCP_RUN_BROWSER_TESTS=1"
    )
    for item in items:
        if "browser" in item.name.lower():
            item.add_marker(skip_browser)
