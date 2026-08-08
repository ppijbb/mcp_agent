"""
Regression tests for run_agent.py's async main() execution.

The CLI dispatcher previously called main() without awaiting, so every
agent that defined `async def main()` silently did nothing.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from srcs.run_agent import _run_main


def test_run_main_sync_function():
    def sync_main():
        return "sync done"

    assert _run_main(sync_main) == "sync done"


def test_run_main_async_function():
    async def async_main():
        return "async done"

    assert _run_main(async_main) == "async done"


def test_run_main_runs_side_effect():
    calls = []

    async def async_main():
        calls.append("executed")

    _run_main(async_main)
    assert calls == ["executed"]
