"""
Common Configuration Module

Shared configurations, constants, and settings used across all agents.
"""

from __future__ import annotations
from datetime import datetime, timezone
import os
from typing import Dict, Any, List
from dataclasses import dataclass

# Default company configuration
# DEFAULT_COMPANY_NAME = "TechCorp Inc." # This line is removed as per the edit hint.

# Default server list used by agents. 
# We are replacing 'filesystem' with 'gdrive' as part of the strategic shift
# towards cloud-based, collaborative tools over local file operations.
DEFAULT_SERVERS = ["gdrive", "g-search", "fetch"]

# Common compliance frameworks
COMPLIANCE_FRAMEWORKS = ["GDPR", "SOX", "HIPAA", "PCI-DSS", "ISO 27001", "NIST"]

# Report generation settings
REPORT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
# SUMMARY_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S" # This line is removed as per the edit hint.

# These helper functions can remain as they are general utility.

def get_output_dir(prefix: str, name: str) -> str:
    """Generates a standardized output directory path."""
    return f"{prefix}_{name}_reports"

def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """Returns a formatted timestamp string."""
    return datetime.now(timezone.utc).strftime(format_str)

# The following functions related to the old config system are now deprecated
# and will be removed.
# - get_settings
# - get_app_config
# - get_reports_path
# - SUMMARY_TIMESTAMP_FORMAT
# - DEFAULT_COMPANY_NAME

# Common agent instruction templates
AGENT_INSTRUCTION_TEMPLATE = """You are a {role} for {company_name}.

{specific_instructions}

{common_guidelines}

{output_format}"""

COMMON_GUIDELINES = """
Follow these guidelines:
- Provide actionable, specific recommendations
- Include quantified benefits and ROI projections where possible
- Consider implementation feasibility and resource requirements
- Maintain professional tone and executive-level insights
- Support recommendations with data and analysis
"""

OUTPUT_FORMAT_GUIDELINES = """
Output Format:
- Use clear headers and structured content
- Include executive summaries for complex analyses
- Provide specific metrics and KPIs
- Include implementation timelines and milestones
- Highlight critical success factors and risks
"""

__all__ = [
    "DEFAULT_SERVERS", "COMPLIANCE_FRAMEWORKS",
    "REPORT_TIMESTAMP_FORMAT", "get_output_dir", "get_timestamp",
    "AGENT_INSTRUCTION_TEMPLATE", "COMMON_GUIDELINES", "OUTPUT_FORMAT_GUIDELINES"
] 