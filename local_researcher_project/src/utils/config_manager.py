"""
Advanced Configuration Manager for Local Researcher

This module provides advanced configuration management functionality including
loading, validation, and access to configuration settings with LangGraph optimization.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field, validator
from langchain_core.runnables import RunnableConfig

# Try to import dotenv, fallback if not available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    def load_dotenv(*args, **kwargs):
        pass

logger = logging.getLogger(__name__)


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    GOOGLE = "google"
    BING = "bing"
    NONE = "none"


class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    url: Optional[str] = Field(default=None, description="The URL of the MCP server")
    tools: Optional[List[str]] = Field(default=None, description="The tools to make available to the LLM")
    auth_required: Optional[bool] = Field(default=False, description="Whether the MCP server requires authentication")
    server_type: str = Field(default="sse", description="Type of MCP server connection (sse|stdio)")
    command: Optional[str] = Field(default=None, description="Command for stdio server type")
    args: Optional[List[str]] = Field(default=None, description="Arguments for stdio server type")


class AdvancedConfiguration(BaseModel):
    """Advanced configuration class for the Local Researcher system."""
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        description="Maximum number of retries for structured output calls from models"
    )
    allow_clarification: bool = Field(
        default=True,
        description="Whether to allow the researcher to ask clarifying questions before starting research"
    )
    max_concurrent_research_units: int = Field(
        default=5,
        description="Maximum number of research units to run concurrently"
    )
    
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        description="Search API to use for research"
    )
    max_researcher_iterations: int = Field(
        default=6,
        description="Maximum number of research iterations for the Research Supervisor"
    )
    max_react_tool_calls: int = Field(
        default=10,
        description="Maximum number of tool calling iterations in a single researcher step"
    )
    
    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4.1-mini",
        description="Model for summarizing research results"
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        description="Maximum output tokens for summarization model"
    )
    research_model: str = Field(
        default="openai:gpt-4.1",
        description="Model for conducting research"
    )
    research_model_max_tokens: int = Field(
        default=10000,
        description="Maximum output tokens for research model"
    )
    compression_model: str = Field(
        default="openai:gpt-4.1",
        description="Model for compressing research findings"
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        description="Maximum output tokens for compression model"
    )
    final_report_model: str = Field(
        default="openai:gpt-4.1",
        description="Model for writing the final report"
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        description="Maximum output tokens for final report model"
    )
    
    # Content Configuration
    max_content_length: int = Field(
        default=50000,
        description="Maximum character length for webpage content before summarization"
    )
    
    # MCP Configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        description="MCP server configuration"
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        description="Additional instructions for MCP tools"
    )
    
    # LangGraph Configuration
    langgraph_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "recursion_limit": 25,
            "debug": False,
            "checkpointer": None
        },
        description="LangGraph specific configuration"
    )
    
    # Browser Configuration
    browser_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "headless": False,
            "disable_security": True,
            "max_content_length": 2000
        },
        description="Browser automation configuration"
    )
    
    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "AdvancedConfiguration":
        """Create an AdvancedConfiguration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class ConfigManager:
    """Advanced configuration manager for the Local Researcher system with LangGraph optimization."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = {}
        self.advanced_config: Optional[AdvancedConfiguration] = None
        self.env_loaded = False
        
        # Load environment variables if dotenv is available
        if DOTENV_AVAILABLE:
            self._load_environment()
        
        # Load configuration
        self._load_config()
        
        # Initialize advanced configuration
        self._initialize_advanced_config()
    
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
                "version": "2.0.0",
                "environment": "development",
                "debug": True,
                "log_level": "INFO"
            },
            "output": {
                "directory": "./outputs",
                "formats": ["markdown", "pdf", "html"],
                "default_format": "markdown"
            },
            "gemini_cli": {
                "enabled": True,
                "command_prefix": "gemini"
            },
            "open_deep_research": {
                "enabled": True,
                "workflow_mode": "multi_agent"
            },
            "langgraph": {
                "recursion_limit": 25,
                "debug": False,
                "checkpointer": None
            },
            "browser": {
                "headless": False,
                "disable_security": True,
                "max_content_length": 2000
            },
            "search": {
                "api": "tavily",
                "max_results": 10,
                "timeout": 30
            },
            "models": {
                "summarization": "openai:gpt-4.1-mini",
                "research": "openai:gpt-4.1",
                "compression": "openai:gpt-4.1",
                "final_report": "openai:gpt-4.1"
            }
        }
    
    def _initialize_advanced_config(self):
        """Initialize advanced configuration from loaded config."""
        try:
            # Extract advanced configuration from loaded config
            advanced_config_data = {
                "max_structured_output_retries": self.get("langgraph.max_structured_output_retries", 3),
                "allow_clarification": self.get("research.allow_clarification", True),
                "max_concurrent_research_units": self.get("research.max_concurrent_units", 5),
                "search_api": SearchAPI(self.get("search.api", "tavily")),
                "max_researcher_iterations": self.get("research.max_iterations", 6),
                "max_react_tool_calls": self.get("research.max_tool_calls", 10),
                "summarization_model": self.get("models.summarization", "openai:gpt-4.1-mini"),
                "summarization_model_max_tokens": self.get("models.summarization_max_tokens", 8192),
                "research_model": self.get("models.research", "openai:gpt-4.1"),
                "research_model_max_tokens": self.get("models.research_max_tokens", 10000),
                "compression_model": self.get("models.compression", "openai:gpt-4.1"),
                "compression_model_max_tokens": self.get("models.compression_max_tokens", 8192),
                "final_report_model": self.get("models.final_report", "openai:gpt-4.1"),
                "final_report_model_max_tokens": self.get("models.final_report_max_tokens", 10000),
                "max_content_length": self.get("browser.max_content_length", 50000),
                "langgraph_config": self.get("langgraph", {}),
                "browser_config": self.get("browser", {})
            }
            
            # Add MCP configuration if available
            mcp_config_data = self.get("mcp", {})
            if mcp_config_data:
                advanced_config_data["mcp_config"] = MCPConfig(**mcp_config_data)
            
            self.advanced_config = AdvancedConfiguration(**advanced_config_data)
            logger.info("Advanced configuration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced configuration: {e}")
            # Fallback to default advanced configuration
            self.advanced_config = AdvancedConfiguration()
    
    def get_advanced_config(self) -> AdvancedConfiguration:
        """Get the advanced configuration object.
        
        Returns:
            AdvancedConfiguration object
        """
        return self.advanced_config or AdvancedConfiguration()
    
    def get_model_config(self, task_type: str) -> Dict[str, Any]:
        """Get model configuration for specific task type.
        
        Args:
            task_type: Type of task (summarization, research, compression, final_report)
            
        Returns:
            Model configuration dictionary
        """
        if not self.advanced_config:
            return {}
        
        model_configs = {
            "summarization": {
                "model": self.advanced_config.summarization_model,
                "max_tokens": self.advanced_config.summarization_model_max_tokens
            },
            "research": {
                "model": self.advanced_config.research_model,
                "max_tokens": self.advanced_config.research_model_max_tokens
            },
            "compression": {
                "model": self.advanced_config.compression_model,
                "max_tokens": self.advanced_config.compression_model_max_tokens
            },
            "final_report": {
                "model": self.advanced_config.final_report_model,
                "max_tokens": self.advanced_config.final_report_model_max_tokens
            }
        }
        
        return model_configs.get(task_type, {})
    
    def get_langgraph_config(self) -> Dict[str, Any]:
        """Get LangGraph specific configuration.
        
        Returns:
            LangGraph configuration dictionary
        """
        if not self.advanced_config:
            return {"recursion_limit": 25, "debug": False, "checkpointer": None}
        
        return self.advanced_config.langgraph_config
    
    def get_browser_config(self) -> Dict[str, Any]:
        """Get browser automation configuration.
        
        Returns:
            Browser configuration dictionary
        """
        if not self.advanced_config:
            return {"headless": False, "disable_security": True, "max_content_length": 2000}
        
        return self.advanced_config.browser_config
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration.
        
        Returns:
            Search configuration dictionary
        """
        if not self.advanced_config:
            return {"api": "tavily", "max_results": 10, "timeout": 30}
        
        return {
            "api": self.advanced_config.search_api.value,
            "max_results": self.get("search.max_results", 10),
            "timeout": self.get("search.timeout", 30)
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
