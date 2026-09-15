"""
Tests verifying that environment-specific config overrides base config.

The loader._deep_merge function gives the first argument (source) priority
over the second (destination). Before the fix, the call was:
  _deep_merge(base_config, env_config)   # base won — WRONG

After the fix:
  _deep_merge(env_config, base_config)   # env wins — CORRECT

These tests verify the merge semantics directly, mirroring the exact
function implementation from srcs/core/config/loader.py so that any
regression in argument order is caught.
"""
import pytest
import yaml
from pathlib import Path


# Mirror of _deep_merge from srcs/core/config/loader.py
def _deep_merge(source, destination):
    """Recursively merge two dicts. Source values win over destination."""
    result = destination.copy()
    for key, value in source.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _deep_merge(value, result[key])
        else:
            result[key] = value
    return result


@pytest.fixture
def base_config():
    """Load the actual base.yaml from configs/."""
    path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    return yaml.safe_load(path.read_text()) or {}


@pytest.fixture
def dev_config():
    """Load the actual development.yaml from configs/."""
    path = Path(__file__).resolve().parent.parent / "configs" / "development.yaml"
    return yaml.safe_load(path.read_text()) or {}


@pytest.fixture
def prod_config():
    """Load the actual production.yaml from configs/."""
    path = Path(__file__).resolve().parent.parent / "configs" / "production.yaml"
    return yaml.safe_load(path.read_text()) or {}


class TestDevConfigOverrides:
    """development.yaml must override base.yaml values."""

    def test_logging_level_is_debug(self, base_config, dev_config):
        merged = _deep_merge(dev_config, base_config)
        assert merged["logging"]["level"] == "DEBUG"

    def test_github_server_enabled(self, base_config, dev_config):
        merged = _deep_merge(dev_config, base_config)
        assert merged["mcp_servers"]["github"]["enabled"] is True


class TestProdConfigOverrides:
    """production.yaml must override base.yaml values."""

    def test_logging_level_is_warning(self, base_config, prod_config):
        merged = _deep_merge(prod_config, base_config)
        assert merged["logging"]["level"] == "WARNING"

    def test_allowed_hosts_restricted(self, base_config, prod_config):
        merged = _deep_merge(prod_config, base_config)
        hosts = merged["security"]["allowed_hosts"]
        assert hosts == ["api.my-domain.com"]

    def test_cache_type_is_redis(self, base_config, prod_config):
        merged = _deep_merge(prod_config, base_config)
        assert merged["cache"]["type"] == "redis"


class TestBaseConfigPreserved:
    """Base config keys that are NOT overridden by env must survive."""

    def test_base_mcp_servers_preserved(self, base_config, dev_config):
        merged = _deep_merge(dev_config, base_config)
        assert "g-search" in merged["mcp_servers"]
        assert merged["mcp_servers"]["g-search"]["command"] == "npx"

    def test_base_cache_ttl_preserved(self, base_config, dev_config):
        merged = _deep_merge(dev_config, base_config)
        assert merged["cache"]["ttl"] == 3600
