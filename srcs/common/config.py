"""
Common Configuration Module

Shared configurations, constants, and settings used across all agents.
"""

from datetime import datetime

# Default company configuration
DEFAULT_COMPANY_NAME = "TechCorp Inc."

# Common server configurations
DEFAULT_SERVERS = ["filesystem", "g-search", "fetch"]

# Common compliance frameworks
COMPLIANCE_FRAMEWORKS = ["GDPR", "SOX", "HIPAA", "PCI-DSS", "ISO 27001", "NIST"]

# Report generation settings
REPORT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
SUMMARY_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_timestamp():
    """Get standardized timestamp for file naming"""
    return datetime.now().strftime(REPORT_TIMESTAMP_FORMAT)

def get_output_dir(agent_type, agent_name):
    """Generate standardized output directory name"""
    return f"{agent_name.lower().replace(' ', '_').replace('-', '_')}_reports"

def get_app_config(app_name, config_path="configs/mcp_agent.config.yaml"):
    """Get standardized app configuration"""
    return {
        "name": app_name,
        "settings": f"get_settings('{config_path}')",
        "human_input_callback": None
    }

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
    "DEFAULT_COMPANY_NAME", "DEFAULT_SERVERS", "COMPLIANCE_FRAMEWORKS",
    "REPORT_TIMESTAMP_FORMAT", "SUMMARY_TIMESTAMP_FORMAT",
    "get_timestamp", "get_output_dir", "get_app_config",
    "AGENT_INSTRUCTION_TEMPLATE", "COMMON_GUIDELINES", "OUTPUT_FORMAT_GUIDELINES"
] 