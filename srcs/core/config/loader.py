"""
Configuration loader for MCP Agent system.

Loads and merges configuration from YAML files, supports encrypted config files,
and provides environment variable substitution for sensitive values.
Uses singleton pattern to cache configuration after first load.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from srcs.core.config.schema import AppConfig
from srcs.core.security.crypto import decrypt_file_content

_config: AppConfig | None = None
_config_path = Path(os.getenv("MCP_CONFIG_PATH", "configs/"))


def load_config() -> AppConfig:
    """
    Load, merge, and validate configuration files and return AppConfig object.
    Uses singleton pattern to cache configuration after first load.
    
    Returns:
        AppConfig: Loaded application configuration object
        
    Note:
        - Uses singleton pattern for performance optimization
        - Returns cached config after first load
        - Includes environment variable substitution
    """
    global _config
    if _config:
        return _config

    env = os.getenv("MCP_ENV", "development")

    base_config = _load_config_file(_config_path / "base.yaml")
    env_config = _load_config_file(_config_path / f"{env}.yaml")

    if not base_config and not env_config:
        print(f"Warning: Configuration files not found at path: {_config_path}")
        base_config = {}

    merged_config = _deep_merge(base_config, env_config)
    merged_config["environment"] = env

    try:
        _config = AppConfig(**merged_config)
    except Exception as e:
        print(f"Error: Configuration validation failed: {e}")
        raise

    _load_secrets_from_env(_config)

    return _config


def _load_config_file(path: Path) -> Dict[str, Any]:
    """
    Load a configuration file. If path ends with '.enc', decrypt the file.
    
    Args:
        path: Path to the configuration file to load
        
    Returns:
        Dict[str, Any]: Loaded configuration data. Returns empty dict if file not found
        
    Note:
        - First checks for {path}.enc encrypted file
        - If encrypted file exists, attempts decryption; falls back to regular file on failure
        - Returns empty dict if neither exists
        - Optimized to check file existence only once
    """
    encrypted_path = Path(f"{path}.enc")
    regular_path = path
    
    encrypted_exists = encrypted_path.exists()
    regular_exists = regular_path.exists()
    
    if not encrypted_exists and not regular_exists:
        return {}

    if encrypted_exists:
        try:
            decrypted_content = decrypt_file_content(str(encrypted_path))
            result = yaml.safe_load(decrypted_content)
            return result if result is not None else {}
        except Exception as e:
            print(f"Warning: Failed to decrypt encrypted config file ({encrypted_path}). Trying regular config file. Error: {e}")
            if not regular_exists:
                return {}

    if regular_exists:
        try:
            with open(regular_path, "r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                return result if result is not None else {}
        except Exception as e:
            print(f"Warning: Failed to load config file ({regular_path}): {e}")

    return {}


def _deep_merge(source: Dict, destination: Dict) -> Dict:
    """
    Recursively merge two dictionaries.
    
    Args:
        source: Source dictionary (higher priority, overwrites destination)
        destination: Destination dictionary (lower priority, merged into)
        
    Returns:
        Dict: Merged destination dictionary
        
    Note:
        - Source values are merged into destination
        - If both dicts have a key that is a dict type, recursively merge
        - Otherwise source value overwrites destination value
    """
    result = destination.copy()
    for key, value in source.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _deep_merge(value, result[key])
        else:
            result[key] = value
    return result


def _load_secrets_from_env(config: AppConfig):
    """
    Load environment variable and update AppConfig object.
    
    Args:
        config: AppConfig object to update
        
    Note:
        - ENCRYPTION_KEY environment variable sets config.security.encryption_key
        - ${VAR_NAME} format in config values are replaced with env values
        - Example: GITHUB_TOKEN -> mcp_servers.github.env.GITHUB_TOKEN
        - Example: GOOGLE_API_KEY -> mcp_servers.g-search.env.GOOGLE_API_KEY
    """
    encryption_key = os.getenv("ENCRYPTION_KEY")
    if encryption_key:
        config.security.encryption_key = encryption_key

    for server_name, server_config in config.mcp_servers.items():
        for key, value in server_config.env.items():
            # Replace ${VAR_NAME} format with environment variable values
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                env_value = os.getenv(env_var)
                if env_value is not None:
                    server_config.env[key] = env_value


def get_settings() -> AppConfig:
    """Lazy-load settings to avoid crashes at import time when config files are missing."""
    return load_config()


def __getattr__(name: str) -> Any:
    """Lazy-load 'settings' on first access to avoid crashes at import time."""
    if name == "settings":
        return load_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
