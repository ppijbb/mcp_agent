"""
Local Researcher Project Configuration (v2.0 - 8대 혁신)

Centralized configuration management for advanced multi-agent research system.
Supports 8 core innovations: Adaptive Supervisor, Hierarchical Compression, 
Multi-Model Orchestration, Continuous Verification, Streaming Pipeline,
Universal MCP Hub, Adaptive Context Window, Production-Grade Reliability.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum


class TaskType(Enum):
    """Task types for Multi-Model Orchestration (혁신 3)."""
    PLANNING = "planning"
    DEEP_REASONING = "deep_reasoning"
    VERIFICATION = "verification"
    GENERATION = "generation"
    COMPRESSION = "compression"
    RESEARCH = "research"

@dataclass
class LLMConfig:
    """LLM configuration settings - Multi-Model Orchestration (혁신 3)."""
    # Primary provider (Google Gemini 기본)
    provider: str = os.getenv("LLM_PROVIDER", "google")
    primary_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4000"))
    api_key: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Multi-Model Orchestration (혁신 3) - Gemini 2.5 Flash-Lite 우선
    planning_model: str = os.getenv("PLANNING_MODEL", "gemini-2.5-flash-lite")
    reasoning_model: str = os.getenv("REASONING_MODEL", "gemini-2.5-flash-lite")
    verification_model: str = os.getenv("VERIFICATION_MODEL", "gemini-2.5-flash-lite")
    generation_model: str = os.getenv("GENERATION_MODEL", "gemini-2.5-flash-lite")
    compression_model: str = os.getenv("COMPRESSION_MODEL", "gemini-2.5-flash-lite")
    
    # OpenRouter API Key
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    
    # Cost optimization
    budget_limit: float = float(os.getenv("BUDGET_LIMIT", "100.0"))
    enable_cost_optimization: bool = os.getenv("ENABLE_COST_OPTIMIZATION", "true").lower() == "true"
    
    def __post_init__(self):
        if self.provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter LLM provider")
        elif self.provider == "google" and not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Google LLM provider")


@dataclass
class AgentConfig:
    """Agent-specific settings with Adaptive Supervisor (혁신 1)."""
    # Basic settings
    max_retries: int = int(os.getenv("AGENT_MAX_RETRIES", "3"))
    timeout_seconds: int = int(os.getenv("AGENT_TIMEOUT", "300"))
    enable_self_planning: bool = os.getenv("ENABLE_SELF_PLANNING", "true").lower() == "true"
    enable_agent_communication: bool = os.getenv("ENABLE_AGENT_COMMUNICATION", "true").lower() == "true"
    
    # Adaptive Supervisor (혁신 1)
    max_concurrent_research_units: int = int(os.getenv("MAX_CONCURRENT_RESEARCH_UNITS", "5"))
    min_researchers: int = int(os.getenv("MIN_RESEARCHERS", "1"))
    max_researchers: int = int(os.getenv("MAX_RESEARCHERS", "10"))
    enable_fast_track: bool = os.getenv("ENABLE_FAST_TRACK", "true").lower() == "true"
    enable_auto_retry: bool = os.getenv("ENABLE_AUTO_RETRY", "true").lower() == "true"
    priority_queue_enabled: bool = os.getenv("PRIORITY_QUEUE_ENABLED", "true").lower() == "true"
    
    # Quality monitoring
    enable_quality_monitoring: bool = os.getenv("ENABLE_QUALITY_MONITORING", "true").lower() == "true"
    quality_threshold: float = float(os.getenv("QUALITY_THRESHOLD", "0.7"))


@dataclass
class ResearchConfig:
    """Research-specific settings with Streaming Pipeline (혁신 5)."""
    # Basic settings
    max_sources: int = int(os.getenv("MAX_SOURCES", "20"))
    search_timeout: int = int(os.getenv("SEARCH_TIMEOUT", "30"))
    enable_academic_search: bool = os.getenv("ENABLE_ACADEMIC_SEARCH", "true").lower() == "true"
    enable_web_search: bool = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
    enable_browser_automation: bool = os.getenv("ENABLE_BROWSER_AUTOMATION", "true").lower() == "true"
    
    # Streaming Pipeline (혁신 5)
    enable_streaming: bool = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
    stream_chunk_size: int = int(os.getenv("STREAM_CHUNK_SIZE", "1024"))
    enable_progressive_reporting: bool = os.getenv("ENABLE_PROGRESSIVE_REPORTING", "true").lower() == "true"
    enable_incremental_save: bool = os.getenv("ENABLE_INCREMENTAL_SAVE", "true").lower() == "true"
    
    # Parallel processing
    enable_parallel_compression: bool = os.getenv("ENABLE_PARALLEL_COMPRESSION", "true").lower() == "true"
    enable_parallel_verification: bool = os.getenv("ENABLE_PARALLEL_VERIFICATION", "true").lower() == "true"


@dataclass
class MCPConfig:
    """MCP integration settings - Universal MCP Hub (혁신 6)."""
    enabled: bool = os.getenv("MCP_ENABLED", "true").lower() == "true"
    server_names: List[str] = field(default_factory=lambda: [
        "g-search", "tavily", "exa", "fetch", "filesystem", 
        "python_coder", "code_interpreter", "arxiv", "scholar", 
        "crunchbase", "linkedin"
    ])
    connection_timeout: int = int(os.getenv("MCP_TIMEOUT", "30"))
    
    # Universal MCP Hub (혁신 6) - MCP만 사용
    enable_plugin_architecture: bool = os.getenv("ENABLE_PLUGIN_ARCHITECTURE", "true").lower() == "true"
    enable_smart_tool_selection: bool = os.getenv("ENABLE_SMART_TOOL_SELECTION", "true").lower() == "true"
    
    # Tool categories
    search_tools: List[str] = field(default_factory=lambda: ["g-search", "tavily", "exa"])
    data_tools: List[str] = field(default_factory=lambda: ["fetch", "filesystem"])
    code_tools: List[str] = field(default_factory=lambda: ["python_coder", "code_interpreter"])
    academic_tools: List[str] = field(default_factory=lambda: ["arxiv", "scholar"])
    business_tools: List[str] = field(default_factory=lambda: ["crunchbase", "linkedin"])


@dataclass
class CompressionConfig:
    """Hierarchical Compression settings (혁신 2)."""
    enabled: bool = os.getenv("ENABLE_HIERARCHICAL_COMPRESSION", "true").lower() == "true"
    enable_hierarchical_compression: bool = os.getenv("ENABLE_HIERARCHICAL_COMPRESSION", "true").lower() == "true"
    compression_levels: int = int(os.getenv("COMPRESSION_LEVELS", "3"))
    preserve_important_info: bool = os.getenv("PRESERVE_IMPORTANT_INFO", "true").lower() == "true"
    enable_compression_validation: bool = os.getenv("ENABLE_COMPRESSION_VALIDATION", "true").lower() == "true"
    compression_history_enabled: bool = os.getenv("COMPRESSION_HISTORY_ENABLED", "true").lower() == "true"
    min_compression_ratio: float = float(os.getenv("MIN_COMPRESSION_RATIO", "0.05"))
    target_compression_ratio: float = float(os.getenv("TARGET_COMPRESSION_RATIO", "0.7"))

@dataclass
class VerificationConfig:
    """Continuous Verification settings (혁신 4)."""
    enabled: bool = os.getenv("ENABLE_CONTINUOUS_VERIFICATION", "true").lower() == "true"
    enable_continuous_verification: bool = os.getenv("ENABLE_CONTINUOUS_VERIFICATION", "true").lower() == "true"
    verification_stages: int = int(os.getenv("VERIFICATION_STAGES", "3"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
    enable_early_warning: bool = os.getenv("ENABLE_EARLY_WARNING", "true").lower() == "true"
    enable_fact_check: bool = os.getenv("ENABLE_FACT_CHECK", "true").lower() == "true"
    enable_uncertainty_marking: bool = os.getenv("ENABLE_UNCERTAINTY_MARKING", "true").lower() == "true"

@dataclass
class ContextWindowConfig:
    """Adaptive Context Window settings (혁신 7)."""
    enabled: bool = os.getenv("ENABLE_ADAPTIVE_CONTEXT", "true").lower() == "true"
    enable_adaptive_context: bool = os.getenv("ENABLE_ADAPTIVE_CONTEXT", "true").lower() == "true"
    min_tokens: int = int(os.getenv("MIN_TOKENS", "2000"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1000000"))
    importance_based_preservation: bool = os.getenv("IMPORTANCE_BASED_PRESERVATION", "true").lower() == "true"
    enable_auto_compression: bool = os.getenv("ENABLE_AUTO_COMPRESSION", "true").lower() == "true"
    enable_long_term_memory: bool = os.getenv("ENABLE_LONG_TERM_MEMORY", "true").lower() == "true"
    memory_refresh_interval: int = int(os.getenv("MEMORY_REFRESH_INTERVAL", "3600"))  # seconds

@dataclass
class ReliabilityConfig:
    """Production-Grade Reliability settings (혁신 8) - MCP만 사용."""
    enabled: bool = os.getenv("ENABLE_PRODUCTION_RELIABILITY", "true").lower() == "true"
    enable_circuit_breaker: bool = os.getenv("ENABLE_CIRCUIT_BREAKER", "true").lower() == "true"
    enable_exponential_backoff: bool = os.getenv("ENABLE_EXPONENTIAL_BACKOFF", "true").lower() == "true"
    enable_state_persistence: bool = os.getenv("ENABLE_STATE_PERSISTENCE", "true").lower() == "true"
    enable_health_check: bool = os.getenv("ENABLE_HEALTH_CHECK", "true").lower() == "true"
    enable_graceful_degradation: bool = os.getenv("ENABLE_GRACEFUL_DEGRADATION", "true").lower() == "true"
    enable_detailed_logging: bool = os.getenv("ENABLE_DETAILED_LOGGING", "true").lower() == "true"
    
    # Circuit breaker settings
    failure_threshold: int = int(os.getenv("FAILURE_THRESHOLD", "5"))
    recovery_timeout: int = int(os.getenv("RECOVERY_TIMEOUT", "60"))
    
    # State persistence
    state_backend: str = os.getenv("STATE_BACKEND", "redis")
    state_ttl: int = int(os.getenv("STATE_TTL", "3600"))

@dataclass
class OutputConfig:
    """Output and reporting settings."""
    output_dir: str = os.getenv("OUTPUT_DIR", "output")
    enable_pdf_generation: bool = os.getenv("ENABLE_PDF", "true").lower() == "true"
    enable_markdown_generation: bool = os.getenv("ENABLE_MARKDOWN", "true").lower() == "true"
    enable_json_export: bool = os.getenv("ENABLE_JSON", "true").lower() == "true"
    
    # Multi-format support
    enable_docx_export: bool = os.getenv("ENABLE_DOCX", "true").lower() == "true"
    enable_html_export: bool = os.getenv("ENABLE_HTML", "true").lower() == "true"
    enable_latex_export: bool = os.getenv("ENABLE_LATEX", "false").lower() == "true"


@dataclass
class ResearcherSystemConfig:
    """Overall system configuration with 8 core innovations."""
    # Core configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Innovation configurations
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    context_window: ContextWindowConfig = field(default_factory=ContextWindowConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    
    def __post_init__(self):
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


# Global configuration instance
config = ResearcherSystemConfig()


def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    return config.llm


def get_agent_config() -> AgentConfig:
    """Get agent configuration."""
    return config.agent


def get_research_config() -> ResearchConfig:
    """Get research configuration."""
    return config.research


def get_mcp_config() -> MCPConfig:
    """Get MCP configuration."""
    return config.mcp


def get_output_config() -> OutputConfig:
    """Get output configuration."""
    return config.output


def get_compression_config() -> CompressionConfig:
    """Get compression configuration (혁신 2)."""
    return config.compression


def get_verification_config() -> VerificationConfig:
    """Get verification configuration (혁신 4)."""
    return config.verification


def get_context_window_config() -> ContextWindowConfig:
    """Get context window configuration (혁신 7)."""
    return config.context_window


def get_reliability_config() -> ReliabilityConfig:
    """Get reliability configuration (혁신 8)."""
    return config.reliability


def update_config_from_env():
    """Update configuration from environment variables."""
    # LLM settings
    if os.getenv("LLM_PROVIDER"):
        config.llm.provider = os.getenv("LLM_PROVIDER")
    if os.getenv("LLM_MODEL"):
        config.llm.primary_model = os.getenv("LLM_MODEL")
    if os.getenv("LLM_TEMPERATURE"):
        config.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))
    if os.getenv("LLM_MAX_TOKENS"):
        config.llm.max_tokens = int(os.getenv("LLM_MAX_TOKENS"))
    if os.getenv("GOOGLE_API_KEY"):
        config.llm.api_key = os.getenv("GOOGLE_API_KEY")
    if os.getenv("OPENROUTER_API_KEY"):
        config.llm.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Agent settings
    if os.getenv("AGENT_MAX_RETRIES"):
        config.agent.max_retries = int(os.getenv("AGENT_MAX_RETRIES"))
    if os.getenv("AGENT_TIMEOUT"):
        config.agent.timeout_seconds = int(os.getenv("AGENT_TIMEOUT"))
    if os.getenv("ENABLE_SELF_PLANNING"):
        config.agent.enable_self_planning = os.getenv("ENABLE_SELF_PLANNING").lower() == "true"
    if os.getenv("ENABLE_AGENT_COMMUNICATION"):
        config.agent.enable_agent_communication = os.getenv("ENABLE_AGENT_COMMUNICATION").lower() == "true"
    
    # Research settings
    if os.getenv("MAX_SOURCES"):
        config.research.max_sources = int(os.getenv("MAX_SOURCES"))
    if os.getenv("SEARCH_TIMEOUT"):
        config.research.search_timeout = int(os.getenv("SEARCH_TIMEOUT"))
    if os.getenv("ENABLE_ACADEMIC_SEARCH"):
        config.research.enable_academic_search = os.getenv("ENABLE_ACADEMIC_SEARCH").lower() == "true"
    if os.getenv("ENABLE_WEB_SEARCH"):
        config.research.enable_web_search = os.getenv("ENABLE_WEB_SEARCH").lower() == "true"
    if os.getenv("ENABLE_BROWSER_AUTOMATION"):
        config.research.enable_browser_automation = os.getenv("ENABLE_BROWSER_AUTOMATION").lower() == "true"
    
    # MCP settings
    if os.getenv("MCP_ENABLED"):
        config.mcp.enabled = os.getenv("MCP_ENABLED").lower() == "true"
    if os.getenv("MCP_TIMEOUT"):
        config.mcp.connection_timeout = int(os.getenv("MCP_TIMEOUT"))
    
    # Output settings
    if os.getenv("OUTPUT_DIR"):
        config.output.output_dir = os.getenv("OUTPUT_DIR")
    if os.getenv("ENABLE_PDF"):
        config.output.enable_pdf_generation = os.getenv("ENABLE_PDF").lower() == "true"
    if os.getenv("ENABLE_MARKDOWN"):
        config.output.enable_markdown_generation = os.getenv("ENABLE_MARKDOWN").lower() == "true"
    if os.getenv("ENABLE_JSON"):
        config.output.enable_json_export = os.getenv("ENABLE_JSON").lower() == "true"


# Initialize configuration from environment
update_config_from_env()
