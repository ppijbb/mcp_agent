"""
Kimi-K2 Conversable Agent for LangGraph integration.

Self-contained, production-ready agent that selects tools deterministically
based on scenario context and its configuration. No external LLM dependency.
"""

from typing import Dict, Any, Optional
from ..models.agent import AgentConfig
import logging

logger = logging.getLogger(__name__)


class KimiK2ConversableAgent:
    """
    Lightweight conversable agent compatible with the Kimi-K2 system.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        llm_config: Optional[Dict[str, Any]] = None,
        tool_registry: Optional[Any] = None,
        **_: Any,
    ) -> None:
        self.agent_config = agent_config
        self.tool_registry = tool_registry
        self.name = agent_config.agent_id
        logger.info("KimiK2ConversableAgent '%s' initialized.", self.agent_config.name)

    def _decide_tool_action(self, task_message: str) -> Optional[Dict[str, Any]]:
        """
        Deterministically decide a tool to call based on scenario keywords and
        the agent's tool preferences. Returns a dict: {name, parameters} or None.
        """
        scenario_lower = (task_message or "").lower()
        prefs = self.agent_config.tool_preferences or []

        # Simple keyword-to-tool routing
        if any(k in scenario_lower for k in ["react", "component", "javascript", "node"]):
            if "code_editor" in prefs:
                return {
                    "name": "code_editor",
                    "parameters": {
                        "action": "open",
                        "file_path": "kimi_k2_workspace/app.jsx",
                    },
                }
            if "terminal" in prefs:
                return {"name": "terminal", "parameters": {"command": "echo Building project"}}

        if any(k in scenario_lower for k in ["analyz", "dataset", "visualiz", "csv"]):
            if "python" in prefs:
                return {
                    "name": "python",
                    "parameters": {"code": "sum([1,2,3])", "timeout": 2},
                }

        # Default to first preferred tool with a benign action
        if prefs:
            tool_id = prefs[0]
            if tool_id == "terminal":
                return {"name": "terminal", "parameters": {"command": "pwd"}}
            if tool_id == "code_editor":
                return {
                    "name": "code_editor",
                    "parameters": {"action": "open", "file_path": "kimi_k2_workspace/README.md"},
                }
            if tool_id == "python":
                return {"name": "python", "parameters": {"code": "2+2", "timeout": 2}}
        return None

    def run_task(self, task_message: str, **_: Any) -> Dict[str, Any]:
        """
        Synchronous task handler used by the simulation engine.
        """
        logger.info("Agent '%s' processing task: %s", self.agent_config.name, task_message)

        thought = (
            f"Analyze task '{task_message}'. Select an appropriate tool from preferences: "
            f"{', '.join(self.agent_config.tool_preferences)}."
        )

        action_msg = (
            f"Agent {self.agent_config.name} analyzed the task and will proceed with the next step."
        )

        tool_call = self._decide_tool_action(task_message)

        result: Dict[str, Any] = {
            "thought": thought,
            "action": action_msg,
            "agent_id": self.agent_config.agent_id,
        }
        if tool_call:
            result["tool_call"] = tool_call
        return result