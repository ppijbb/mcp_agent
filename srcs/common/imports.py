"""
Standardized Imports Module

Centralized imports for all agents to ensure consistency and reduce boilerplate.
Provides commonly used modules, types, and MCP agent components.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

# MCP Agent components (with graceful fallback)
try:
    from mcp_agent.app import MCPApp
    from mcp_agent.agents.agent import Agent
    from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
    from mcp_agent.workflows.llm.augmented_llm import RequestParams
    from mcp_agent.workflows.llm.augmented_llm import OpenAIAugmentedLLM
except ImportError:
    MCPApp = None
    Agent = None
    Orchestrator = None
    RequestParams = None
    OpenAIAugmentedLLM = None

__all__ = [
    "asyncio", "os", "json", "datetime",
    "Any", "Dict", "List", "Optional",
    "MCPApp", "Agent", "Orchestrator", "RequestParams", "OpenAIAugmentedLLM",
]
