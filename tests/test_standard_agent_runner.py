"""
Regression tests for StandardAgentRunner._run_module_agent.

Function-based agents (class_name is None / empty / "none") used to fall
through into the class-based execution path and crash with
getattr(module, None) -> TypeError, discarding the function result.
"""
import asyncio
import sys
import types
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Install minimal stubs for heavy third-party deps so this module can be
# imported in environments without mcp/mcp_agent/google-genai/schedule.
# In CI with the real deps installed, the real modules take precedence.
try:
    import mcp  # noqa: F401
except ModuleNotFoundError:
    mcp = types.ModuleType("mcp")
    mcp_types = types.ModuleType("mcp.types")
    mcp_types.ElicitRequestParams = None
    mcp_types.ElicitRequestURLParams = None
    mcp.types = mcp_types
    sys.modules["mcp"] = mcp
    sys.modules["mcp.types"] = mcp_types

try:
    import mcp_agent  # noqa: F401
except ModuleNotFoundError:
    mcp_agent = types.ModuleType("mcp_agent")
    mcp_agent.config = types.ModuleType("mcp_agent.config")
    mcp_agent.config._settings = None
    sys.modules["mcp_agent"] = mcp_agent
    sys.modules["mcp_agent.config"] = mcp_agent.config

try:
    import google.genai  # noqa: F401
except ModuleNotFoundError:
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai_types = types.ModuleType("google.genai.types")
    genai_types.GenerateContentConfig = type(
        "GenerateContentConfig", (), {"__init__": lambda self, *a, **k: None}
    )
    genai.types = genai_types
    google.genai = genai
    sys.modules["google"] = google
    sys.modules["google.genai"] = genai
    sys.modules["google.genai.types"] = genai_types

try:
    import schedule  # noqa: F401
except ModuleNotFoundError:
    schedule = types.ModuleType("schedule")
    schedule.Job = type("Job", (), {})
    sys.modules["schedule"] = schedule

from srcs.common.standard_agent_runner import StandardAgentRunner


# Fake agent module registered into sys.modules so importlib can load it.
_FAKE_MODULE = types.ModuleType("_test_agent_module")


def _test_function(name="world"):
    return {"greeting": f"hello {name}"}


class _TestAgentClass:
    def __init__(self, prefix="hi"):
        self.prefix = prefix

    def run(self, name="world"):
        return {"greeting": f"{self.prefix} {name}"}


_FAKE_MODULE._test_function = _test_function
_FAKE_MODULE._TestAgentClass = _TestAgentClass
sys.modules["_test_agent_module"] = _FAKE_MODULE


def _runner():
    return StandardAgentRunner()


def test_function_based_agent_returns_success():
    input_data = {
        "module_path": "_test_agent_module",
        "class_name": None,
        "method_name": "_test_function",
        "name": "alice",
    }
    result = asyncio.run(_runner()._run_module_agent("", input_data))
    assert result.success is True
    assert result.data == {"greeting": "hello alice"}


def test_function_based_agent_none_string():
    input_data = {
        "module_path": "_test_agent_module",
        "class_name": "none",
        "method_name": "_test_function",
    }
    result = asyncio.run(_runner()._run_module_agent("", input_data))
    assert result.success is True
    assert result.data == {"greeting": "hello world"}


def test_class_based_agent_still_works():
    input_data = {
        "module_path": "_test_agent_module",
        "class_name": "_TestAgentClass",
        "method_name": "run",
        "name": "bob",
        "init_kwargs": {"prefix": "hey"},
    }
    result = asyncio.run(_runner()._run_module_agent("", input_data))
    assert result.success is True
    assert result.data == {"greeting": "hey bob"}
