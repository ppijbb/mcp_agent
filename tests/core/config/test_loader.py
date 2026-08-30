"""
Tests for the configuration loader's environment variable substitution.
"""
import os
import pytest

from srcs.core.config.schema import AppConfig, MCPServerConfig
from srcs.core.config.loader import (
    _load_secrets_from_env,
    _resolve_env_placeholder,
)


def _make_config(encryption_key="${ENCRYPTION_KEY}", redis_url="${REDIS_URL}") -> AppConfig:
    config = AppConfig()
    config.security.encryption_key = encryption_key
    config.cache.redis_url = redis_url
    config.mcp_servers["g-search"] = MCPServerConfig(
        command="npx",
        args=[],
        env={"GOOGLE_SEARCH_API_KEY": "${GOOGLE_SEARCH_API_KEY}"},
    )
    return config


def test_unset_env_placeholders_resolve_to_none(monkeypatch):
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    config = _make_config()
    _load_secrets_from_env(config)

    # Optional top-level fields must not retain literal "${...}" placeholders.
    assert config.security.encryption_key is None
    assert config.cache.redis_url is None


def test_set_env_placeholders_resolve_to_value(monkeypatch):
    monkeypatch.setenv("ENCRYPTION_KEY", "my-secret")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

    config = _make_config()
    _load_secrets_from_env(config)

    assert config.security.encryption_key == "my-secret"
    assert config.cache.redis_url == "redis://localhost:6379"


def test_server_env_substitution(monkeypatch):
    monkeypatch.setenv("GOOGLE_SEARCH_API_KEY", "abc123")
    monkeypatch.setenv("ENCRYPTION_KEY", "k")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

    config = _make_config()
    _load_secrets_from_env(config)

    assert config.mcp_servers["g-search"].env["GOOGLE_SEARCH_API_KEY"] == "abc123"


def test_resolve_env_placeholder_helper():
    assert _resolve_env_placeholder("plain") == "plain"
    assert _resolve_env_placeholder("${THE_DEFINITELY_UNSET_VAR}") is None
    assert _resolve_env_placeholder("${NOT_A_PLACEHOLDER") == "${NOT_A_PLACEHOLDER"
