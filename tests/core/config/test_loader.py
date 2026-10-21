"""
Test suite for the configuration loader.

Covers the merge precedence between the shared ``base.yaml`` and the
environment specific config file. The environment file must win, otherwise
production overrides (e.g. a restricted ``security.allowed_hosts``) are
silently ignored.
"""
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from srcs.core.config import loader
from srcs.core.config.loader import _deep_merge


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Keep the module level singleton from leaking between tests."""
    original_config = loader._config
    loader._config = None
    yield
    loader._config = original_config


class TestDeepMerge:
    """Unit tests for the recursive merge helper."""

    def test_overrides_win_on_scalar_conflict(self):
        merged = _deep_merge({"level": "DEBUG"}, {"level": "INFO"})
        assert merged["level"] == "DEBUG"

    def test_base_values_are_inherited(self):
        merged = _deep_merge({"level": "DEBUG"}, {"level": "INFO", "ttl": 3600})
        assert merged["ttl"] == 3600

    def test_nested_dicts_are_merged_recursively(self):
        merged = _deep_merge(
            {"security": {"allowed_hosts": ["api.example.com"]}},
            {"security": {"encryption_key": "k", "allowed_hosts": ["*"]}},
        )
        assert merged["security"]["allowed_hosts"] == ["api.example.com"]
        assert merged["security"]["encryption_key"] == "k"

    def test_source_is_not_mutated(self):
        base = {"a": {"b": 1}}
        overrides = {"a": {"b": 2}}
        merged = _deep_merge(overrides, base)
        assert merged["a"]["b"] == 2
        assert base["a"]["b"] == 1
        assert overrides["a"]["b"] == 2

    def test_missing_override_key_keeps_base_entry(self):
        merged = _deep_merge({}, {"a": 1})
        assert merged == {"a": 1}


class TestConfigPrecedence:
    """The environment config must override the base config."""

    def _write_config(self, tmp_path, base_yaml, env_yaml):
        (tmp_path / "base.yaml").write_text(base_yaml, encoding="utf-8")
        (tmp_path / "development.yaml").write_text(env_yaml, encoding="utf-8")
        loader._config_path = tmp_path
        loader._config = None
        return loader.load_config()

    def test_environment_overrides_base(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MCP_ENV", "development")
        config = self._write_config(
            tmp_path,
            "logging:\n  level: INFO\ncache:\n  type: in-memory\n  ttl: 3600\n",
            "logging:\n  level: DEBUG\ncache:\n  type: redis\n",
        )
        assert config.logging.level == "DEBUG"
        assert config.cache.type == "redis"
        # Un-overridden base values survive the merge
        assert config.cache.ttl == 3600

    def test_shipped_production_config_restricts_allowed_hosts(self, tmp_path, monkeypatch):
        """The shipped production.yaml must win over the base wildcard."""
        monkeypatch.setenv("MCP_ENV", "production")
        loader._config_path = project_root / "configs"
        loader._config = None
        config = loader.load_config()

        assert config.security.allowed_hosts == ["api.my-domain.com"]
        assert config.security.allowed_hosts != ["*"]
        assert config.logging.level == "WARNING"
        assert config.cache.type == "redis"

    def test_shipped_development_config_enables_github_server(self, tmp_path, monkeypatch):
        """The shipped development.yaml must win over the base definition."""
        monkeypatch.setenv("MCP_ENV", "development")
        loader._config_path = project_root / "configs"
        loader._config = None
        config = loader.load_config()

        assert config.logging.level == "DEBUG"
        assert config.mcp_servers["github"].enabled is True
        # Servers only defined in base.yaml are still present
        assert "g-search" in config.mcp_servers
        assert config.mcp_servers["g-search"].timeout == 60
