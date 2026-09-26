"""
Regression tests for environment configuration precedence.

``load_config`` merges ``base.yaml`` with ``{MCP_ENV}.yaml``. The environment
file holds the environment-specific *overrides*, so it must win over the base
layer. These tests drive the real ``load_config()`` entry point so that an
inverted merge order inside ``load_config`` is actually caught.
"""


from srcs.core.config import loader

BASE = {"logging": {"level": "INFO", "log_file": "logs/a.log"},
        "security": {"allowed_hosts": ["*"]},
        "cache": {"type": "in-memory", "enabled": True}}


def _load_env(monkeypatch, env_name):
    """Run the real load_config() for *env_name* against the shipped configs."""
    monkeypatch.setattr(loader, "_config", None)
    monkeypatch.setattr(loader, "_config_path", loader.Path("configs").resolve())
    monkeypatch.setenv("MCP_ENV", env_name)
    return loader.load_config()


# --------------------------------------------------------------------------
# Unit-level behaviour of the merge helper
# --------------------------------------------------------------------------

def test_env_overrides_base():
    """Environment values take precedence over base values."""
    env = {"logging": {"level": "DEBUG"}}
    merged = loader._deep_merge(env, BASE)
    assert merged["logging"]["level"] == "DEBUG"


def test_base_supplies_keys_absent_from_env():
    """Keys not overridden by the environment still come from base."""
    env = {"logging": {"level": "DEBUG"}}
    merged = loader._deep_merge(env, BASE)
    assert merged["logging"]["log_file"] == "logs/a.log"
    assert merged["cache"]["type"] == "in-memory"


def test_env_can_replace_list_wholesale():
    """A list override replaces the base list (e.g. allowed_hosts)."""
    env = {"security": {"allowed_hosts": ["api.my-domain.com"]}}
    merged = loader._deep_merge(env, BASE)
    assert merged["security"]["allowed_hosts"] == ["api.my-domain.com"]


def test_merge_does_not_mutate_inputs():
    """Merging must not modify the source or destination dictionaries."""
    base = {"a": {"b": 1}}
    env = {"a": {"c": 2}}
    loader._deep_merge(env, base)
    assert base == {"a": {"b": 1}}
    assert env == {"a": {"c": 2}}


# --------------------------------------------------------------------------
# End-to-end: the real load_config() call site
# --------------------------------------------------------------------------

def test_development_overrides_are_applied(monkeypatch):
    """development.yaml must win over base.yaml (DEBUG logging, github on)."""
    cfg = _load_env(monkeypatch, "development")
    assert cfg.environment == "development"
    assert cfg.logging.level == "DEBUG"
    assert cfg.mcp_servers["github"].enabled is True


def test_production_overrides_are_applied(monkeypatch):
    """production.yaml must win over base.yaml (WARNING logging, redis cache)."""
    cfg = _load_env(monkeypatch, "production")
    assert cfg.environment == "production"
    assert cfg.logging.level == "WARNING"
    assert cfg.cache.type == "redis"


def test_production_does_not_inherit_wildcard_allowed_hosts(monkeypatch):
    """Production must not silently fall back to base's permissive '*' hosts."""
    cfg = _load_env(monkeypatch, "production")
    assert "*" not in cfg.security.allowed_hosts
    assert cfg.security.allowed_hosts == ["api.my-domain.com"]


def test_base_values_survive_when_not_overridden(monkeypatch):
    """Values only present in base.yaml must still be present after merging."""
    cfg = _load_env(monkeypatch, "production")
    assert cfg.cache.enabled is True
    assert cfg.logging.log_file == "logs/mcp_agent.log"


def test_environment_is_recorded_in_config(monkeypatch):
    """The active MCP_ENV must be reflected on the resulting config."""
    assert _load_env(monkeypatch, "production").environment == "production"


# --------------------------------------------------------------------------
# Degradation behaviour
# --------------------------------------------------------------------------

def test_missing_config_file_returns_empty_dict(tmp_path):
    """A nonexistent config file degrades to an empty dict rather than raising."""
    assert loader._load_config_file(tmp_path / "nope.yaml") == {}


def test_empty_yaml_file_returns_empty_dict(tmp_path):
    """An empty YAML document yields {} instead of None."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert loader._load_config_file(empty) == {}


def test_schema_defaults_apply_when_no_files_found(monkeypatch, tmp_path):
    """With no config files the AppConfig schema defaults are used."""
    monkeypatch.setattr(loader, "_config", None)
    monkeypatch.setattr(loader, "_config_path", tmp_path)
    monkeypatch.setenv("MCP_ENV", "development")
    cfg = loader.load_config()
    assert cfg.logging.level == "INFO"
    assert cfg.cache.type == "in-memory"
