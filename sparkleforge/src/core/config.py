"""
Typed server specifications and conversion helpers for FastMCP configuration.

This module provides Pydantic-based models for configuring MCP (Model Context Protocol) servers
with support for both stdio and HTTP/SSE transports. It ensures type safety and validation
for server specifications used throughout the MCP Agent system.

Key Features:
- Type-safe server configuration models
- Support for stdio and HTTP/SSE transports
- Environment variable handling
- Validation and error handling
- Configuration serialization/deserialization

Usage:
    spec = HTTPServerSpec(
        url="http://127.0.0.1:8000/mcp",
        headers={"Authorization": "Bearer token"}
    )
    
    stdio_spec = StdioServerSpec(
        command="python",
        args=["server.py"],
        env={"DEBUG": "true"}
    )
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _BaseServer(BaseModel):
    """Base model for server specs."""

    # Pydantic v2 style configuration (replaces class Config)
    model_config = ConfigDict(extra="forbid")


class StdioServerSpec(_BaseServer):
    """Specification for a local MCP server launched via stdio.

    NOTE:
        The FastMCP Python client typically expects HTTP/SSE transports. Using
        `StdioServerSpec` requires a different adapter or an HTTP shim in front
        of the stdio server. Keep this for future expansion or custom runners.

    Attributes:
        command: Executable to launch (e.g., "python").
        args: Positional arguments for the process.
        env: Environment variables to set for the process.
        cwd: Optional working directory.
        keep_alive: Whether the client should try to keep a persistent session.
    """

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    keep_alive: bool = True


class HTTPServerSpec(_BaseServer):
    """Specification for a remote MCP server reachable via HTTP/SSE.

    Attributes:
        url: Full endpoint URL for the MCP server (e.g., http://127.0.0.1:8000/mcp).
        transport: The transport mechanism ("http", "streamable-http", or "sse").
        headers: Optional request headers (e.g., Authorization tokens).
        auth: Optional auth hint if your FastMCP deployment consumes it.
    """

    url: str
    transport: Literal["http", "streamable-http", "sse"] = "http"
    headers: dict[str, str] = Field(default_factory=dict)
    auth: str | None = None


ServerSpec = StdioServerSpec | HTTPServerSpec
"""Union of supported server specifications."""


def servers_to_mcp_config(servers: Mapping[str, ServerSpec]) -> dict[str, dict[str, object]]:
    """Convert programmatic server specs to the FastMCP configuration dict.

    Args:
        servers: Mapping of server name to specification.

    Returns:
        Dict suitable for initializing `fastmcp.Client({"mcpServers": ...})`.
    """
    cfg: dict[str, dict[str, object]] = {}
    for name, s in servers.items():
        if isinstance(s, StdioServerSpec):
            # FastMCP는 stdio 서버에 대해 transport 필드 없이 command, args, env만 사용
            entry: dict[str, object] = {
                "command": s.command,
                "args": s.args,
            }
            # env가 비어있지 않으면 추가 (None이 아닌 dict만)
            if s.env:
                entry["env"] = s.env
            # cwd가 있으면 추가
            if s.cwd:
                entry["cwd"] = s.cwd
            cfg[name] = entry
        else:
            # HTTP 서버의 경우 httpUrl 또는 url 사용
            entry: dict[str, object] = {}
            if s.transport == "http":
                # httpUrl 사용 (FastMCP 표준)
                entry["httpUrl"] = s.url
            else:
                # SSE나 다른 transport의 경우 url 사용
                entry["url"] = s.url
            if s.headers:
                entry["headers"] = s.headers
            if s.auth is not None:
                entry["auth"] = s.auth
            cfg[name] = entry
    return cfg
