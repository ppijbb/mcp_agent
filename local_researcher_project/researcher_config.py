"""
Local Researcher Project Configuration (v2.0 - 8대 혁신)

Centralized configuration management for advanced multi-agent research system.
Supports 8 core innovations: Adaptive Supervisor, Hierarchical Compression, 
Multi-Model Orchestration, Continuous Verification, Streaming Pipeline,
Universal MCP Hub, Adaptive Context Window, Production-Grade Reliability.
"""

import os
from typing import List, Dict, Any, Optional, Literal
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict, field_validator


class TaskType(Enum):
    """Task types for Multi-Model Orchestration (혁신 3)."""
    PLANNING = "planning"
    DEEP_REASONING = "deep_reasoning"
    VERIFICATION = "verification"
    GENERATION = "generation"
    COMPRESSION = "compression"
    RESEARCH = "research"

class LLMConfig(BaseModel):
    """LLM configuration settings - Multi-Model Orchestration (혁신 3)."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    # Primary provider (OpenRouter + Gemini 2.5 Flash Lite) - NO DEFAULTS
    provider: str = Field(description="LLM provider")
    primary_model: str = Field(description="Primary model")
    temperature: float = Field(ge=0.0, le=2.0, description="Temperature")
    max_tokens: int = Field(gt=0, description="Max tokens")
    api_key: str = Field(description="Google API key (deprecated, use OPENROUTER_API_KEY)")
    
    # Multi-Model Orchestration (혁신 3) - OpenRouter Gemini 2.5 Flash-Lite 우선 - NO DEFAULTS
    planning_model: str = Field(description="Planning model")
    reasoning_model: str = Field(description="Reasoning model")
    verification_model: str = Field(description="Verification model")
    generation_model: str = Field(description="Generation model")
    compression_model: str = Field(description="Compression model")
    
    # OpenRouter API Key (Required) - NO DEFAULTS
    openrouter_api_key: str = Field(description="OpenRouter API key")
    
    # Cost optimization - NO DEFAULTS
    budget_limit: float = Field(gt=0, description="Budget limit")
    enable_cost_optimization: bool = Field(description="Enable cost optimization")
    
    @field_validator('openrouter_api_key')
    @classmethod
    def validate_openrouter_api_key(cls, v):
        if not v:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter LLM provider")
        if not v.startswith('sk-or-'):
            raise ValueError("OPENROUTER_API_KEY must start with 'sk-or-'")
        return v
    
    @field_validator('primary_model')
    @classmethod
    def validate_primary_model(cls, v):
        if not v.startswith('google/gemini-'):
            raise ValueError("Primary model must be a Gemini model (google/gemini-*)")
        return v
    
    @classmethod
    def validate_environment(cls):
        """환경 변수 검증 및 필수 설정 확인."""
        import os
        
        # 필수 환경 변수 검증
        required_vars = {
            'OPENROUTER_API_KEY': 'OpenRouter API key is required',
            'LLM_MODEL': 'LLM model must be specified'
        }
        
        missing_vars = []
        for var, message in required_vars.items():
            if not os.getenv(var):
                missing_vars.append(f"{var}: {message}")
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables:\n" + "\n".join(missing_vars))
        
        # OpenRouter API 키 형식 검증
        api_key = os.getenv('OPENROUTER_API_KEY')
        if api_key and not api_key.startswith('sk-or-'):
            raise ValueError("OPENROUTER_API_KEY must start with 'sk-or-'")
        
        return True


class AgentConfig(BaseModel):
    """Agent-specific settings with Adaptive Supervisor (혁신 1)."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    # Basic settings - NO DEFAULTS
    max_retries: int = Field(ge=0, description="Max retries")
    timeout_seconds: int = Field(gt=0, description="Timeout seconds")
    enable_self_planning: bool = Field(description="Enable self planning")
    enable_agent_communication: bool = Field(description="Enable agent communication")
    
    # Adaptive Supervisor (혁신 1) - NO DEFAULTS
    max_concurrent_research_units: int = Field(gt=0, description="Max concurrent research units")
    min_researchers: int = Field(gt=0, description="Min researchers")
    max_researchers: int = Field(gt=0, description="Max researchers")
    enable_fast_track: bool = Field(description="Enable fast track")
    enable_auto_retry: bool = Field(description="Enable auto retry")
    priority_queue_enabled: bool = Field(description="Priority queue enabled")
    
    # Quality monitoring - NO DEFAULTS
    enable_quality_monitoring: bool = Field(description="Enable quality monitoring")
    quality_threshold: float = Field(ge=0.0, le=1.0, description="Quality threshold")


@dataclass
class ResearchConfig:
    """Research-specific settings with Streaming Pipeline (혁신 5)."""
    # Basic settings - NO DEFAULTS
    max_sources: int
    search_timeout: int
    enable_academic_search: bool
    enable_web_search: bool
    enable_browser_automation: bool
    
    # Streaming Pipeline (혁신 5) - NO DEFAULTS
    enable_streaming: bool
    stream_chunk_size: int
    enable_progressive_reporting: bool
    enable_incremental_save: bool
    
    # Parallel processing - NO DEFAULTS
    enable_parallel_compression: bool
    enable_parallel_verification: bool


@dataclass
class MCPConfig:
    """MCP integration settings - Universal MCP Hub (혁신 6)."""
    enabled: bool
    server_names: List[str]
    connection_timeout: int
    
    # Universal MCP Hub (혁신 6) - MCP만 사용 - NO DEFAULTS
    enable_plugin_architecture: bool
    enable_smart_tool_selection: bool
    enable_auto_fallback: bool
    
    # Tool categories - NO DEFAULTS
    search_tools: List[str]
    data_tools: List[str]
    code_tools: List[str]
    academic_tools: List[str]
    business_tools: List[str]


@dataclass
class CompressionConfig:
    """Hierarchical Compression settings (혁신 2)."""
    enabled: bool
    enable_hierarchical_compression: bool
    compression_levels: int
    preserve_important_info: bool
    enable_compression_validation: bool
    compression_history_enabled: bool
    min_compression_ratio: float
    target_compression_ratio: float

@dataclass
class VerificationConfig:
    """Continuous Verification settings (혁신 4)."""
    enabled: bool
    enable_continuous_verification: bool
    verification_stages: int
    confidence_threshold: float
    enable_early_warning: bool
    enable_fact_check: bool
    enable_uncertainty_marking: bool

@dataclass
class ContextWindowConfig:
    """Adaptive Context Window settings (혁신 7)."""
    enabled: bool
    enable_adaptive_context: bool
    min_tokens: int
    max_tokens: int
    importance_based_preservation: bool
    enable_auto_compression: bool
    enable_long_term_memory: bool
    memory_refresh_interval: int

@dataclass
class ReliabilityConfig:
    """Production-Grade Reliability settings (혁신 8) - MCP만 사용."""
    enabled: bool
    enable_circuit_breaker: bool
    enable_exponential_backoff: bool
    enable_state_persistence: bool
    enable_health_check: bool
    enable_graceful_degradation: bool
    enable_detailed_logging: bool
    
    # Circuit breaker settings - NO DEFAULTS
    failure_threshold: int
    recovery_timeout: int
    
    # State persistence - NO DEFAULTS
    state_backend: str
    state_ttl: int

@dataclass
class OutputConfig:
    """Output and reporting settings."""
    output_dir: str
    enable_pdf_generation: bool
    enable_markdown_generation: bool
    enable_json_export: bool
    
    # Multi-format support - NO DEFAULTS
    enable_docx_export: bool
    enable_html_export: bool
    enable_latex_export: bool


class ResearcherSystemConfig(BaseModel):
    """Overall system configuration with 8 core innovations."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    # Core configurations - NO DEFAULTS
    llm: LLMConfig = Field(description="LLM configuration")
    agent: AgentConfig = Field(description="Agent configuration")
    research: ResearchConfig = Field(description="Research configuration")
    mcp: MCPConfig = Field(description="MCP configuration")
    output: OutputConfig = Field(description="Output configuration")
    
    # Innovation configurations - NO DEFAULTS
    compression: CompressionConfig = Field(description="Compression configuration")
    verification: VerificationConfig = Field(description="Verification configuration")
    context_window: ContextWindowConfig = Field(description="Context window configuration")
    reliability: ReliabilityConfig = Field(description="Reliability configuration")
    
    def model_post_init(self, __context):
        # Ensure output directory exists
        os.makedirs(self.output.output_dir, exist_ok=True)
        
        # Validate configurations
        self._validate_configurations()
    
    def _validate_configurations(self):
        """Validate configuration consistency."""
        # Validate token limits
        if self.context_window.min_tokens >= self.context_window.max_tokens:
            raise ValueError("min_tokens must be less than max_tokens")
        
        # Validate researcher limits
        if self.agent.min_researchers > self.agent.max_researchers:
            raise ValueError("min_researchers must be less than or equal to max_researchers")
        
        # Validate confidence thresholds
        if not 0.0 <= self.verification.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.compression.min_compression_ratio <= 1.0:
            raise ValueError("min_compression_ratio must be between 0.0 and 1.0")


# Global configuration instance - will be loaded from environment
config = None


def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.llm


def get_agent_config() -> AgentConfig:
    """Get agent configuration."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.agent


def get_research_config() -> ResearchConfig:
    """Get research configuration."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.research


def get_mcp_config() -> MCPConfig:
    """Get MCP configuration."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.mcp


def get_output_config() -> OutputConfig:
    """Get output configuration."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.output


def get_compression_config() -> CompressionConfig:
    """Get compression configuration (혁신 2)."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.compression


def get_verification_config() -> VerificationConfig:
    """Get verification configuration (혁신 4)."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.verification


def get_context_window_config() -> ContextWindowConfig:
    """Get context window configuration (혁신 7)."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.context_window


def get_reliability_config() -> ReliabilityConfig:
    """Get reliability configuration (혁신 8)."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.reliability


def load_config_from_env() -> ResearcherSystemConfig:
    """Load configuration from environment variables - ALL REQUIRED, NO DEFAULTS."""
    
    # Load .env file if it exists
    from pathlib import Path
    from dotenv import load_dotenv
    
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    def get_required_env(key: str, var_type: type = str):
        """Get required environment variable, raise error if missing."""
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable {key} is not set")
        
        if var_type == bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif var_type == int:
            try:
                return int(value)
            except ValueError:
                raise ValueError(f"Environment variable {key} must be an integer, got: {value}")
        elif var_type == float:
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"Environment variable {key} must be a float, got: {value}")
        return value
    
    def get_required_list_env(key: str, separator: str = ","):
        """Get required environment variable as list."""
        value = get_required_env(key)
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    # Load LLM configuration
    llm_config = LLMConfig(
        provider=get_required_env("LLM_PROVIDER"),
        primary_model=get_required_env("LLM_MODEL"),
        temperature=get_required_env("LLM_TEMPERATURE", float),
        max_tokens=get_required_env("LLM_MAX_TOKENS", int),
        api_key=get_required_env("GOOGLE_API_KEY"),
        planning_model=get_required_env("PLANNING_MODEL"),
        reasoning_model=get_required_env("REASONING_MODEL"),
        verification_model=get_required_env("VERIFICATION_MODEL"),
        generation_model=get_required_env("GENERATION_MODEL"),
        compression_model=get_required_env("COMPRESSION_MODEL"),
        openrouter_api_key=get_required_env("OPENROUTER_API_KEY"),
        budget_limit=get_required_env("BUDGET_LIMIT", float),
        enable_cost_optimization=get_required_env("ENABLE_COST_OPTIMIZATION", bool)
    )
    
    # Load Agent configuration
    agent_config = AgentConfig(
        max_retries=get_required_env("AGENT_MAX_RETRIES", int),
        timeout_seconds=get_required_env("AGENT_TIMEOUT", int),
        enable_self_planning=get_required_env("ENABLE_SELF_PLANNING", bool),
        enable_agent_communication=get_required_env("ENABLE_AGENT_COMMUNICATION", bool),
        max_concurrent_research_units=get_required_env("MAX_CONCURRENT_RESEARCH_UNITS", int),
        min_researchers=get_required_env("MIN_RESEARCHERS", int),
        max_researchers=get_required_env("MAX_RESEARCHERS", int),
        enable_fast_track=get_required_env("ENABLE_FAST_TRACK", bool),
        enable_auto_retry=get_required_env("ENABLE_AUTO_RETRY", bool),
        priority_queue_enabled=get_required_env("PRIORITY_QUEUE_ENABLED", bool),
        enable_quality_monitoring=get_required_env("ENABLE_QUALITY_MONITORING", bool),
        quality_threshold=get_required_env("QUALITY_THRESHOLD", float)
    )
    
    # Load Research configuration
    research_config = ResearchConfig(
        max_sources=get_required_env("MAX_SOURCES", int),
        search_timeout=get_required_env("SEARCH_TIMEOUT", int),
        enable_academic_search=get_required_env("ENABLE_ACADEMIC_SEARCH", bool),
        enable_web_search=get_required_env("ENABLE_WEB_SEARCH", bool),
        enable_browser_automation=get_required_env("ENABLE_BROWSER_AUTOMATION", bool),
        enable_streaming=get_required_env("ENABLE_STREAMING", bool),
        stream_chunk_size=get_required_env("STREAM_CHUNK_SIZE", int),
        enable_progressive_reporting=get_required_env("ENABLE_PROGRESSIVE_REPORTING", bool),
        enable_incremental_save=get_required_env("ENABLE_INCREMENTAL_SAVE", bool),
        enable_parallel_compression=get_required_env("ENABLE_PARALLEL_COMPRESSION", bool),
        enable_parallel_verification=get_required_env("ENABLE_PARALLEL_VERIFICATION", bool)
    )
    
    # Load MCP configuration
    mcp_config = MCPConfig(
        enabled=get_required_env("MCP_ENABLED", bool),
        server_names=get_required_list_env("MCP_SERVER_NAMES"),
        connection_timeout=get_required_env("MCP_TIMEOUT", int),
        enable_plugin_architecture=get_required_env("ENABLE_PLUGIN_ARCHITECTURE", bool),
        enable_smart_tool_selection=get_required_env("ENABLE_SMART_TOOL_SELECTION", bool),
        enable_auto_fallback=get_required_env("ENABLE_AUTO_FALLBACK", bool),
        search_tools=get_required_list_env("MCP_SEARCH_TOOLS"),
        data_tools=get_required_list_env("MCP_DATA_TOOLS"),
        code_tools=get_required_list_env("MCP_CODE_TOOLS"),
        academic_tools=get_required_list_env("MCP_ACADEMIC_TOOLS"),
        business_tools=get_required_list_env("MCP_BUSINESS_TOOLS")
    )
    
    # Load Compression configuration
    compression_config = CompressionConfig(
        enabled=get_required_env("ENABLE_HIERARCHICAL_COMPRESSION", bool),
        enable_hierarchical_compression=get_required_env("ENABLE_HIERARCHICAL_COMPRESSION", bool),
        compression_levels=get_required_env("COMPRESSION_LEVELS", int),
        preserve_important_info=get_required_env("PRESERVE_IMPORTANT_INFO", bool),
        enable_compression_validation=get_required_env("ENABLE_COMPRESSION_VALIDATION", bool),
        compression_history_enabled=get_required_env("COMPRESSION_HISTORY_ENABLED", bool),
        min_compression_ratio=get_required_env("MIN_COMPRESSION_RATIO", float),
        target_compression_ratio=get_required_env("TARGET_COMPRESSION_RATIO", float)
    )
    
    # Load Verification configuration
    verification_config = VerificationConfig(
        enabled=get_required_env("ENABLE_CONTINUOUS_VERIFICATION", bool),
        enable_continuous_verification=get_required_env("ENABLE_CONTINUOUS_VERIFICATION", bool),
        verification_stages=get_required_env("VERIFICATION_STAGES", int),
        confidence_threshold=get_required_env("CONFIDENCE_THRESHOLD", float),
        enable_early_warning=get_required_env("ENABLE_EARLY_WARNING", bool),
        enable_fact_check=get_required_env("ENABLE_FACT_CHECK", bool),
        enable_uncertainty_marking=get_required_env("ENABLE_UNCERTAINTY_MARKING", bool)
    )
    
    # Load Context Window configuration
    context_window_config = ContextWindowConfig(
        enabled=get_required_env("ENABLE_ADAPTIVE_CONTEXT", bool),
        enable_adaptive_context=get_required_env("ENABLE_ADAPTIVE_CONTEXT", bool),
        min_tokens=get_required_env("MIN_TOKENS", int),
        max_tokens=get_required_env("MAX_TOKENS", int),
        importance_based_preservation=get_required_env("IMPORTANCE_BASED_PRESERVATION", bool),
        enable_auto_compression=get_required_env("ENABLE_AUTO_COMPRESSION", bool),
        enable_long_term_memory=get_required_env("ENABLE_LONG_TERM_MEMORY", bool),
        memory_refresh_interval=get_required_env("MEMORY_REFRESH_INTERVAL", int)
    )
    
    # Load Reliability configuration
    reliability_config = ReliabilityConfig(
        enabled=get_required_env("ENABLE_PRODUCTION_RELIABILITY", bool),
        enable_circuit_breaker=get_required_env("ENABLE_CIRCUIT_BREAKER", bool),
        enable_exponential_backoff=get_required_env("ENABLE_EXPONENTIAL_BACKOFF", bool),
        enable_state_persistence=get_required_env("ENABLE_STATE_PERSISTENCE", bool),
        enable_health_check=get_required_env("ENABLE_HEALTH_CHECK", bool),
        enable_graceful_degradation=get_required_env("ENABLE_GRACEFUL_DEGRADATION", bool),
        enable_detailed_logging=get_required_env("ENABLE_DETAILED_LOGGING", bool),
        failure_threshold=get_required_env("FAILURE_THRESHOLD", int),
        recovery_timeout=get_required_env("RECOVERY_TIMEOUT", int),
        state_backend=get_required_env("STATE_BACKEND"),
        state_ttl=get_required_env("STATE_TTL", int)
    )
    
    # Load Output configuration
    output_config = OutputConfig(
        output_dir=get_required_env("OUTPUT_DIR"),
        enable_pdf_generation=get_required_env("ENABLE_PDF", bool),
        enable_markdown_generation=get_required_env("ENABLE_MARKDOWN", bool),
        enable_json_export=get_required_env("ENABLE_JSON", bool),
        enable_docx_export=get_required_env("ENABLE_DOCX", bool),
        enable_html_export=get_required_env("ENABLE_HTML", bool),
        enable_latex_export=get_required_env("ENABLE_LATEX", bool)
    )
    
    # Create and store global config instance
    global config
    config = ResearcherSystemConfig(
        llm=llm_config,
        agent=agent_config,
        research=research_config,
        mcp=mcp_config,
        output=output_config,
        compression=compression_config,
        verification=verification_config,
        context_window=context_window_config,
        reliability=reliability_config
    )
    
    return config


def update_config_from_env():
    """Update configuration from environment variables - DEPRECATED, use load_config_from_env() instead."""
    global config
    config = load_config_from_env()


# Initialize configuration from environment - will be called by main.py
# update_config_from_env()
