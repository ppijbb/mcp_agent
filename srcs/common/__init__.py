"""
Common Module

Contains shared resources, configurations, and utilities used across all agents:
- Common imports and dependencies
- Shared configurations and constants
- Common utilities and helper functions
- Base agent templates and patterns
"""

from .imports import *
from .config import *
from .utils import *
from .templates import *

__all__ = [
    # Core classes and functions
    "Agent", "MCPApp", "Orchestrator", "RequestParams", "OpenAIAugmentedLLM",
    "EvaluatorOptimizerLLM", "QualityRating", "get_settings",
    
    # Configuration and constants
    "DEFAULT_COMPANY_NAME", "get_output_dir", "get_timestamp",
    
    # Utilities
    "setup_agent_app", "create_executive_summary", "create_kpi_template",
    "save_deliverables",
    
    # Templates
    "AgentTemplate", "EnterpriseAgentTemplate"
] 