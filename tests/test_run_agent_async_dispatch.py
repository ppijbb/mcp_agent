"""
Tests for the run_agent CLI runner.

Covers:
  - _run_module_main: async/await dispatch, import errors, attribute errors
  - Correct module paths for researcher_v2 / enhanced_data_generator
"""
import sys
import types
import importlib
import asyncio
import inspect
from pathlib import Path

import pytest

# Ensure srcs package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from srcs.run_agent import _run_module_main


# ---------------------------------------------------------------------------
# Helpers: synthetic in-process modules for testing
# ---------------------------------------------------------------------------

def _install_fake_module(name: str, code: str) -> None:
    """Install a fake module into sys.modules so importlib can find it."""
    mod = types.ModuleType(name)
    exec(compile(code, f"<{name}>", "exec"), mod.__dict__)
    sys.modules[name] = mod


class TestRunModuleMain:
    """Tests for _run_module_main."""

    def setup_method(self):
        """Remove any leftover fake modules between tests."""
        self._fake = []

    def teardown_method(self):
        for name in self._fake:
            sys.modules.pop(name, None)

    def _install(self, name: str, code: str):
        _install_fake_module(name, code)
        self._fake.append(name)

    def test_sync_main_is_called(self):
        """A sync main() should be called directly."""
        self._install("test_sync_mod", "called = False\ndef main():\n    global called; called = True")
        _run_module_main("test_sync_mod")
        assert sys.modules["test_sync_mod"].called is True

    def test_async_main_is_awaited(self):
        """An async main() should actually run (not return a discarded coroutine)."""
        self._install(
            "test_async_mod",
            "called = False\nasync def main():\n    global called; called = True",
        )
        _run_module_main("test_async_mod")
        assert sys.modules["test_async_mod"].called is True

    def test_missing_main_raises(self):
        self._install("test_nomod", "x = 1")
        with pytest.raises(AttributeError, match="missing main function"):
            _run_module_main("test_nomod")

    def test_non_callable_main_raises(self):
        self._install("test_noncall", "main = 42")
        with pytest.raises(TypeError, match="not callable"):
            _run_module_main("test_noncall")

    def test_import_error_propagates(self):
        with pytest.raises(ImportError, match="Failed to import"):
            _run_module_main("nonexistent_package_this_should_not_exist_xyz")


class TestModulePathFixes:
    """Verify that researcher_v2 and enhanced_data_generator map to the correct modules."""

    @staticmethod
    def _run_agent_source() -> str:
        path = Path(__file__).resolve().parent.parent / "srcs" / "run_agent.py"
        return path.read_text()

    def test_researcher_v2_points_to_advanced_agents(self):
        """researcher_v2 module path must reference advanced_agents (not basic_agents)."""
        src = self._run_agent_source()
        assert '"researcher_v2": "advanced_agents.researcher_v2"' in src
        # The stale path must not be present anywhere
        assert '"researcher_v2": "basic_agents.researcher_v2"' not in src
        # And the file exists on disk
        py_path = Path(__file__).resolve().parent.parent / "srcs" / "advanced_agents" / "researcher_v2.py"
        assert py_path.exists(), f"Module file not found: {py_path}"

    def test_enhanced_data_generator_points_to_run_chat_data_agent(self):
        """enhanced_data_generator must point to advanced_agents.run_chat_data_agent."""
        src = self._run_agent_source()
        assert '"enhanced_data_generator": "advanced_agents.run_chat_data_agent"' in src
        # The stale path must not be present anywhere
        assert '"enhanced_data_generator": "basic_agents.enhanced_data_generator"' not in src
