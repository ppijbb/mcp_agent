"""
Configuration Manager for GraphRAG Agent

환경변수와 YAML 설정 파일을 통한 설정 관리 시스템
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import logging


class DomainType(str, Enum):
    """Domain types for specialization"""
    GENERAL = "general"
    BIOMEDICAL = "biomedical"
    FINANCE = "finance"
    TECHNICAL = "technical"


class DataClassification(str, Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class AgentConfig(BaseModel):
    """Base configuration for agents"""
    openai_api_key: str = Field(default="", description="OpenAI API key")
    gemini_api_key: str = Field(default="", description="Gemini API key")
    model_name: str = Field(default="gemini-2.5-flash-lite-preview-06-07", description="Model name")
    max_search_results: int = Field(default=5, description="Max search results for RAG")
    context_window_size: int = Field(default=4000, description="Context window size")
    temperature: float = Field(default=0.0, description="Temperature")
    max_tokens: int = Field(default=1000, description="Max tokens")
    enable_visualization: bool = Field(default=True, description="Enable visualization")
    enable_optimization: bool = Field(default=True, description="Enable optimization")
    
    # Optimization settings
    quality_threshold: float = Field(default=0.8, description="Quality threshold for optimization")
    max_iterations: int = Field(default=10, description="Maximum optimization iterations")
    
    # Visualization settings
    output_directory: str = Field(default="./graph_visualizations", description="Output directory for visualizations")
    formats: list = Field(default=["png", "svg", "html"], description="Output formats for visualizations")
    max_nodes: int = Field(default=1000, description="Maximum nodes to visualize")
    
    @field_validator('max_search_results')
    def validate_max_search_results(cls, v):
        if v < 1 or v > 20:
            raise ValueError('max_search_results must be between 1 and 20')
        return v
    
    @field_validator('context_window_size')
    def validate_context_window_size(cls, v):
        if v < 1000 or v > 32000:
            raise ValueError('context_window_size must be between 1000 and 32000')
        return v
    
    @field_validator('quality_threshold')
    def validate_quality_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('quality_threshold must be between 0.0 and 1.0')
        return v
    
    @field_validator('max_iterations')
    def validate_max_iterations(cls, v):
        if v < 1 or v > 100:
            raise ValueError('max_iterations must be between 1 and 100')
        return v
    
    @field_validator('max_nodes')
    def validate_max_nodes(cls, v):
        if v < 10 or v > 10000:
            raise ValueError('max_nodes must be between 10 and 10000')
        return v


class GraphConfig(BaseModel):
    """Configuration for graph generation"""
    enable_domain_specialization: bool = Field(default=True, description="Enable domain specialization")
    domain_type: DomainType = Field(default=DomainType.GENERAL, description="Domain type")
    enable_security_privacy: bool = Field(default=True, description="Enable security and privacy")
    enable_query_optimization: bool = Field(default=True, description="Enable query optimization")
    default_data_classification: DataClassification = Field(default=DataClassification.INTERNAL, description="Default data classification")


class VisualizationConfig(BaseModel):
    """Configuration for graph visualization"""
    enabled: bool = Field(default=True, description="Enable graph visualization")
    output_directory: str = Field(default="./graph_visualizations", description="Output directory")
    formats: list = Field(default=["png", "svg", "html"], description="Output formats")
    max_nodes: int = Field(default=1000, description="Maximum nodes to visualize")
    
    @field_validator('max_nodes')
    def validate_max_nodes(cls, v):
        if v < 10 or v > 10000:
            raise ValueError('max_nodes must be between 10 and 10000')
        return v


class OptimizationConfig(BaseModel):
    """Configuration for graph optimization"""
    enabled: bool = Field(default=True, description="Enable graph optimization")
    quality_threshold: float = Field(default=0.8, description="Quality threshold")
    max_iterations: int = Field(default=10, description="Maximum optimization iterations")
    
    @field_validator('quality_threshold')
    def validate_quality_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('quality_threshold must be between 0.0 and 1.0')
        return v
    
    @field_validator('max_iterations')
    def validate_max_iterations(cls, v):
        if v < 1 or v > 100:
            raise ValueError('max_iterations must be between 1 and 100')
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging"""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10485760, description="Max log file size in bytes")  # 10MB
    backup_count: int = Field(default=5, description="Number of backup files")
    
    @field_validator('level')
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'level must be one of {valid_levels}')
        return v.upper()


class GraphRAGConfig(BaseModel):
    """Main configuration for GraphRAG Agent"""
    agent: AgentConfig
    graph: GraphConfig
    visualization: VisualizationConfig
    optimization: OptimizationConfig
    logging: LoggingConfig
    
    # Runtime settings
    data_file: Optional[str] = Field(default=None, description="Data file path")
    output_path: Optional[str] = Field(default=None, description="Output path")
    query: Optional[str] = Field(default=None, description="Query string")
    graph_path: Optional[str] = Field(default=None, description="Graph file path")
    
    # Operation mode
    mode: str = Field(default="create", description="Operation mode: create, query, visualize, optimize, export")
    verbose: bool = Field(default=False, description="Verbose output")


class ConfigManager:
    """Configuration manager for loading and managing settings"""
    
    def __init__(self, config_path: Optional[str] = None, system_config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to YAML configuration file (optional)
            system_config_path: Path to system configuration file (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or self._find_config_file()
        self.system_config_path = system_config_path or self._find_system_config_file()
        
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in common locations"""
        # Get the directory containing this file (config/)
        current_dir = os.path.dirname(__file__)
        # Get the parent directory (graphrag_agent/)
        parent_dir = os.path.dirname(current_dir)
        
        possible_paths = [
            "config.yaml",
            "config.yml", 
            "graphrag_config.yaml",
            "graphrag_config.yml",
            os.path.join(os.getcwd(), "config.yaml"),
            os.path.join(os.getcwd(), "config.yml"),
            os.path.join(current_dir, "config.yaml"),
            os.path.join(current_dir, "config.yml"),
            os.path.join(parent_dir, "config.yaml"),  # graphrag_agent/config.yaml
            os.path.join(parent_dir, "config.yml"),   # graphrag_agent/config.yml
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.logger.info(f"Found config file: {path}")
                return path
        
        self.logger.info("No config file found in any of the expected locations, will use defaults and environment variables")
        self.logger.debug("Searched paths:")
        for path in possible_paths:
            self.logger.debug(f"  - {path}")
        return None
    
    def _find_system_config_file(self) -> Optional[str]:
        """Find system configuration file in common locations"""
        # Get the directory containing this file (config/)
        current_dir = os.path.dirname(__file__)
        # Get the parent directory (graphrag_agent/)
        parent_dir = os.path.dirname(current_dir)
        
        possible_paths = [
            "system_config.yaml",
            "system_config.yml",
            "system.yaml",
            "system.yml",
            os.path.join(os.getcwd(), "system_config.yaml"),
            os.path.join(os.getcwd(), "system_config.yml"),
            os.path.join(current_dir, "system_config.yaml"),
            os.path.join(current_dir, "system_config.yml"),
            os.path.join(parent_dir, "system_config.yaml"),
            os.path.join(parent_dir, "system_config.yml"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.logger.info(f"Found system config file: {path}")
                return path
        
        self.logger.info("No system config file found, using defaults")
        self.logger.debug("Searched paths:")
        for path in possible_paths:
            self.logger.debug(f"  - {path}")
        return None
    
    def load_config(self) -> GraphRAGConfig:
        """
        Load configuration from YAML file and environment variables
        Priority: env > yaml > default
        
        Returns:
            GraphRAGConfig object with loaded settings
        """
        self.logger.info("Loading configuration with priority: env > yaml > default")
        
        # Start with default configuration
        config_data = self._get_default_config()
        self.logger.debug("Default configuration loaded")
        
        # Load from YAML file if exists
        if self.config_path and os.path.exists(self.config_path):
            self.logger.info(f"Loading YAML config from: {self.config_path}")
            config_data = self._load_yaml_config(config_data)
        else:
            self.logger.info("No YAML config file found, using defaults and environment variables only")
        
        # Override with environment variables
        self.logger.info("Loading environment variables...")
        config_data = self._load_env_config(config_data)
        
        # Validate and create config object
        try:
            return GraphRAGConfig(**config_data)
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "agent": {
                "openai_api_key": "",
                "model_name": "gemini-2.5-flash-lite-preview-06-07",
                "max_search_results": 5,
                "context_window_size": 4000
            },
            "graph": {
                "enable_domain_specialization": True,
                "domain_type": "general",
                "enable_security_privacy": True,
                "enable_query_optimization": True,
                "default_data_classification": "internal"
            },
            "visualization": {
                "enabled": True,
                "output_directory": "./graph_visualizations",
                "formats": ["png", "svg", "html"],
                "max_nodes": 1000
            },
            "optimization": {
                "enabled": True,
                "quality_threshold": 0.8,
                "max_iterations": 10
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_path": None,
                "max_file_size": 10485760,
                "backup_count": 5
            },
            "data_file": None,
            "output_path": None,
            "query": None,
            "graph_path": None,
            "mode": "create",
            "verbose": False
        }
    
    def _load_yaml_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path:
            self.logger.warning("No config file path specified, skipping YAML loading")
            return config_data
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            
            if yaml_config:
                # Deep merge with existing config
                config_data = self._deep_merge(config_data, yaml_config)
                self.logger.info(f"Configuration loaded from {self.config_path}")
                self.logger.debug(f"YAML config keys: {list(yaml_config.keys())}")
            else:
                self.logger.warning(f"YAML file {self.config_path} is empty or invalid")
            
        except Exception as e:
            self.logger.warning(f"Failed to load YAML config from {self.config_path}: {e}")
        
        return config_data
    
    def _load_env_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_mappings = {
            # Agent settings
            'OPENAI_API_KEY': ['agent', 'openai_api_key'],
            'GEMINI_API_KEY': ['agent', 'gemini_api_key'],
            'GRAPH_MODEL_NAME': ['agent', 'model_name'],
            'RAG_MODEL_NAME': ['agent', 'model_name'],
            'MAX_SEARCH_RESULTS': ['agent', 'max_search_results'],
            'CONTEXT_WINDOW_SIZE': ['agent', 'context_window_size'],
            'TEMPERATURE': ['agent', 'temperature'],
            'MAX_TOKENS': ['agent', 'max_tokens'],
            
            # Graph settings
            'ENABLE_DOMAIN_SPECIALIZATION': ['graph', 'enable_domain_specialization'],
            'DOMAIN_TYPE': ['graph', 'domain_type'],
            'ENABLE_SECURITY_PRIVACY': ['graph', 'enable_security_privacy'],
            'ENABLE_QUERY_OPTIMIZATION': ['graph', 'enable_query_optimization'],
            'DEFAULT_DATA_CLASSIFICATION': ['graph', 'default_data_classification'],
            
            # Visualization settings
            'ENABLE_VISUALIZATION': ['visualization', 'enabled'],
            'VISUALIZATION_OUTPUT_DIR': ['visualization', 'output_directory'],
            'VISUALIZATION_FORMATS': ['visualization', 'formats'],
            'MAX_VISUALIZATION_NODES': ['visualization', 'max_nodes'],
            
            # Optimization settings
            'ENABLE_OPTIMIZATION': ['optimization', 'enabled'],
            'OPTIMIZATION_QUALITY_THRESHOLD': ['optimization', 'quality_threshold'],
            'MAX_OPTIMIZATION_ITERATIONS': ['optimization', 'max_iterations'],
            
            # Logging settings
            'LOG_LEVEL': ['logging', 'level'],
            'LOG_FORMAT': ['logging', 'format'],
            'LOG_FILE_PATH': ['logging', 'file_path'],
            'LOG_MAX_FILE_SIZE': ['logging', 'max_file_size'],
            'LOG_BACKUP_COUNT': ['logging', 'backup_count'],
            
            # Runtime settings
            'DATA_FILE': ['data_file'],
            'OUTPUT_PATH': ['output_path'],
            'QUERY': ['query'],
            'GRAPH_PATH': ['graph_path'],
            'MODE': ['mode'],
            'VERBOSE': ['verbose']
        }
        
        env_vars_found = []
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(value, config_path)
                self._set_nested_value(config_data, config_path, converted_value)
                env_vars_found.append(f"{env_var}={converted_value}")
                self.logger.debug(f"Set {'.'.join(config_path)} = {converted_value} from {env_var}")
        
        if env_vars_found:
            self.logger.info(f"Environment variables loaded: {', '.join(env_vars_found)}")
        else:
            self.logger.info("No environment variables found")
        
        return config_data
    
    def _convert_env_value(self, value: str, config_path: list) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        if config_path[-1] in ['max_search_results', 'context_window_size', 'max_nodes', 
                              'max_iterations', 'max_file_size', 'backup_count', 'max_tokens']:
            try:
                return int(value)
            except ValueError:
                return value
        
        # Float conversion
        if config_path[-1] in ['quality_threshold', 'temperature']:
            try:
                return float(value)
            except ValueError:
                return value
        
        # List conversion (for formats)
        if config_path[-1] == 'formats':
            return [item.strip() for item in value.split(',')]
        
        return value
    
    def _set_nested_value(self, data: Dict[str, Any], path: list, value: Any):
        """Set a nested value in dictionary"""
        current = data
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, config: GraphRAGConfig, output_path: str):
        """Save configuration to YAML file"""
        try:
            config_dict = config.dict()
            
            # Remove None values and empty strings
            cleaned_config = self._clean_config_dict(config_dict)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(cleaned_config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def _clean_config_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values and empty strings from config dict"""
        cleaned = {}
        for key, value in config_dict.items():
            if value is not None and value != "" and value != []:
                if isinstance(value, dict):
                    cleaned_value = self._clean_config_dict(value)
                    if cleaned_value:
                        cleaned[key] = cleaned_value
                else:
                    cleaned[key] = value
        return cleaned
    
    def validate_config(self, config: GraphRAGConfig) -> bool:
        """Validate configuration"""
        try:
            # Check required fields
            if not config.agent.openai_api_key or config.agent.openai_api_key.strip() == "":
                self.logger.error("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
                return False
            
            # Validate API key format (basic check)
            if len(config.agent.openai_api_key) < 20:
                self.logger.error("OpenAI API key appears to be invalid (too short)")
                return False
            
            # Check file paths if provided (only for modes that require them)
            if config.data_file and config.mode in ["create"] and not os.path.exists(config.data_file):
                self.logger.error(f"Data file not found: {config.data_file}")
                return False
            
            # Only check graph_path if mode requires it
            if config.graph_path and config.mode in ["query", "visualize", "optimize"] and not os.path.exists(config.graph_path):
                self.logger.error(f"Graph file not found: {config.graph_path}")
                return False
            
            # Validate mode
            valid_modes = ["create", "query", "visualize", "optimize", "export", "status"]
            if config.mode not in valid_modes:
                self.logger.error(f"Invalid mode: {config.mode}. Valid modes: {valid_modes}")
                return False
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
