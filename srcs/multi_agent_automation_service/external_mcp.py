"""
External MCP Server Configuration

Loads external MCP server configurations from environment variables and
configures them for use with the MCP agent system.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def _parse_args(args_str: Optional[str]) -> List[str]:
    """Parse space-separated argument string into list of tokens."""
    if not args_str:
        return []
    return [token for token in args_str.split(" ") if token]


def _maybe_json_env(env_str: Optional[str]) -> Dict[str, str]:
    """Parse JSON string environment variable into dictionary."""
    if not env_str:
        return {}
    try:
        data = json.loads(env_str)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON environment variable")
        return {}


def load_external_server_config(server_name: str) -> Optional[Dict[str, Any]]:
    """
    Load external MCP server configuration from environment variables.

    Reads MCP server execution information from environment variables and
    returns it in mcp_agent configuration format.

    Required environment variable:
        <NAME>_MCP_CMD

    Optional environment variables:
        <NAME>_MCP_ARGS           (space-separated argument string)
        <NAME>_MCP_TIMEOUT_MS     (integer, default 30000)
        <NAME>_MCP_TRUST          ("true"/"false", default true)
        <NAME>_MCP_ENV_JSON       (JSON string {"KEY":"VAL",...})

    Args:
        server_name: Name of the server to load configuration for

    Returns:
        Configuration dictionary or None if required variables are missing
    """
    key = server_name.upper().replace("-", "_")
    cmd = os.getenv(f"{key}_MCP_CMD")
    if not cmd:
        return None

    args = _parse_args(os.getenv(f"{key}_MCP_ARGS", ""))
    timeout_raw = os.getenv(f"{key}_MCP_TIMEOUT_MS", "30000")
    trust_raw = os.getenv(f"{key}_MCP_TRUST", "true").lower()
    env_json = _maybe_json_env(os.getenv(f"{key}_MCP_ENV_JSON"))

    try:
        timeout = int(timeout_raw)
    except ValueError:
        timeout = 30000

    trust = trust_raw != "false"

    return {
        "command": cmd,
        "args": args,
        "timeout": timeout,
        "trust": trust,
        **({"env": env_json} if env_json else {}),
    }


def configure_external_servers(context, candidates: List[str]) -> List[str]:
    """
    Configure external MCP servers that have environment variables set.

    For each candidate name, checks if corresponding environment variables
    exist and registers the server configuration. Skips servers that already
    exist in the configuration.

    Args:
        context: Application context with config.mcp.servers
        candidates: List of server names to check

    Returns:
        List of server names that were successfully added
    """
    added: List[str] = []
    server_map = getattr(context.config.mcp, "servers", {})

    for name in candidates:
        if name in server_map:
            continue
        cfg = load_external_server_config(name)
        if cfg:
            server_map[name] = cfg
            added.append(name)

    # Reassign to ensure changes are reflected on dict-like objects
    context.config.mcp.servers = server_map
    return added
