"""
Local Researcher Project Configuration

Centralized configuration management for autonomous multi-agent research system.
All hardcoded values are moved to this configuration file.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class LLMConfig:
    """LLM configuration settings - Gemini only."""
    provider: str = os.getenv("LLM_PROVIDER", "google")
    model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4000"))
    api_key: str = os.getenv("GOOGLE_API_KEY", "")
    
    def __post_init__(self):
        if not self.api_key and self.provider == "google":
            raise ValueError("GOOGLE_API_KEY environment variable is required for Google LLM provider")


@dataclass
class AgentConfig:
    """Agent-specific settings."""
    max_retries: int = int(os.getenv("AGENT_MAX_RETRIES", "3"))
    timeout_seconds: int = int(os.getenv("AGENT_TIMEOUT", "300"))
    enable_self_planning: bool = os.getenv("ENABLE_SELF_PLANNING", "true").lower() == "true"
    enable_agent_communication: bool = os.getenv("ENABLE_AGENT_COMMUNICATION", "true").lower() == "true"


@dataclass
class ResearchConfig:
    """Research-specific settings."""
    max_sources: int = int(os.getenv("MAX_SOURCES", "20"))
    search_timeout: int = int(os.getenv("SEARCH_TIMEOUT", "30"))
    enable_academic_search: bool = os.getenv("ENABLE_ACADEMIC_SEARCH", "true").lower() == "true"
    enable_web_search: bool = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
    enable_browser_automation: bool = os.getenv("ENABLE_BROWSER_AUTOMATION", "true").lower() == "true"


@dataclass
class MCPConfig:
    """MCP integration settings."""
    enabled: bool = os.getenv("MCP_ENABLED", "true").lower() == "true"
    server_names: List[str] = field(default_factory=lambda: ["g-search", "fetch", "filesystem"])
    connection_timeout: int = int(os.getenv("MCP_TIMEOUT", "30"))


@dataclass
class OutputConfig:
    """Output and reporting settings."""
    output_dir: str = os.getenv("OUTPUT_DIR", "output")
    enable_pdf_generation: bool = os.getenv("ENABLE_PDF", "true").lower() == "true"
    enable_markdown_generation: bool = os.getenv("ENABLE_MARKDOWN", "true").lower() == "true"
    enable_json_export: bool = os.getenv("ENABLE_JSON", "true").lower() == "true"


@dataclass
class ResearcherSystemConfig:
    """Overall system configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def __post_init__(self):
        # Ensure output directory exists
        os.makedirs(self.output.output_dir, exist_ok=True)


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


def update_config_from_env():
    """Update configuration from environment variables."""
    # LLM settings
    if os.getenv("LLM_PROVIDER"):
        config.llm.provider = os.getenv("LLM_PROVIDER")
    if os.getenv("LLM_MODEL"):
        config.llm.model = os.getenv("LLM_MODEL")
    if os.getenv("LLM_TEMPERATURE"):
        config.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))
    if os.getenv("LLM_MAX_TOKENS"):
        config.llm.max_tokens = int(os.getenv("LLM_MAX_TOKENS"))
    if os.getenv("GOOGLE_API_KEY"):
        config.llm.api_key = os.getenv("GOOGLE_API_KEY")
    
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
