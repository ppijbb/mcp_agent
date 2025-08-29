"""
Configuration Manager for Local Researcher

This module provides configuration management functionality including
loading, validation, and access to configuration settings.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Try to import dotenv, fallback if not available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    def load_dotenv(*args, **kwargs):
        pass

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration for the Local Researcher system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = {}
        self.env_loaded = False
        
        # Load environment variables if dotenv is available
        if DOTENV_AVAILABLE:
            self._load_environment()
        
        # Load configuration
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / "configs" / "config.yaml")
    
    def _load_environment(self):
        """Load environment variables from .env file."""
        try:
            project_root = Path(__file__).parent.parent.parent
            env_path = project_root / ".env"
            
            if env_path.exists():
                load_dotenv(env_path)
                self.env_loaded = True
                logger.info("Environment variables loaded from .env file")
            else:
                logger.warning(".env file not found, using system environment variables")
        except Exception as e:
            logger.error(f"Failed to load environment variables: {e}")
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                self.config = self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            "app": {
                "name": "Local Researcher",
                "version": "1.0.0",
                "environment": "development",
                "debug": True,
                "log_level": "INFO"
            },
            "output": {
                "directory": "./outputs",
                "formats": ["markdown"],
                "default_format": "markdown"
            },
            "gemini_cli": {
                "enabled": True,
                "command_prefix": "gemini"
            },
            "open_deep_research": {
                "enabled": True,
                "workflow_mode": "multi_agent"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation like 'app.name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
        except Exception as e:
            logger.error(f"Error getting config key '{key}': {e}")
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        try:
            keys = key.split('.')
            config = self.config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            logger.info(f"Configuration updated: {key} = {value}")
        except Exception as e:
            logger.error(f"Error setting config key '{key}': {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()
    
    def reload(self):
        """Reload configuration from file."""
        self._load_config()
        logger.info("Configuration reloaded")
    
    def save(self, config_path: Optional[str] = None):
        """Save current configuration to file.
        
        Args:
            config_path: Path to save configuration. If None, uses current path.
        """
        try:
            save_path = config_path or self.config_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def get_env(self, key: str, default: Any = None) -> Any:
        """Get environment variable value.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value or default
        """
        return os.getenv(key, default)
    
    def validate(self) -> bool:
        """Validate current configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            required_keys = [
                "app.name",
                "app.version",
                "output.directory"
            ]
            
            for key in required_keys:
                if self.get(key) is None:
                    logger.error(f"Missing required configuration key: {key}")
                    return False
            
            logger.info("Configuration validation passed")
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
