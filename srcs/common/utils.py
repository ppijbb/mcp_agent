"""
Common Utilities Module

Shared utility functions used across all agents for common operations.
"""

import os
import json
from datetime import datetime
from mcp_agent.app import MCPApp
from srcs.core.config.loader import settings

class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Enhanced JSON encoder that handles datetime objects.
    
    Extends the standard JSONEncoder to convert datetime objects to ISO format strings,
    enabling proper serialization of datetime values in JSON responses.
    
    Attributes:
        Inherits all attributes from json.JSONEncoder
        
    Methods:
        default: Override to handle datetime serialization
    """
    
    def default(self, o):
        """
        Convert objects to JSON-serializable format.
        
        Args:
            o: Object to serialize
            
        Returns:
            JSON-serializable representation of the object
            
        Notes:
            - datetime objects are converted to ISO format strings
            - All other objects are handled by the parent class
        """
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def setup_agent_app(app_name: str) -> MCPApp:
    """
    Set up and configure MCPApp with standard settings.
    
    Creates an MCPApp instance with proper configuration loading from the project's
    config files. Searches for config files in both the configs directory and project root.
    
    Args:
        app_name: Name identifier for the MCP application
        
    Returns:
        Configured MCPApp instance ready for use
        
    Raises:
        FileNotFoundError: If no valid configuration file is found
        ConfigurationError: If configuration loading fails
        
    Example:
        app = setup_agent_app("research_agent")
        async with app.run() as context:
            # Use the agent context
            pass
    """
    from mcp_agent.config import get_settings
    from pathlib import Path
    
    # Find config file path from project root
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # First try configs directory, then project root
    config_path = project_root / "configs" / "mcp_agent.config.yaml"
    if not config_path.exists():
        config_path = project_root / "mcp_agent.config.yaml"
    
    # Use mcp_agent library's standard settings
    app_settings = get_settings(str(config_path))
    
    return MCPApp(
        name=app_name,
        settings=app_settings,
        human_input_callback=None
    )


def ensure_output_directory(output_dir: str) -> str:
    """
    Create output directory if it doesn't exist.
    
    Ensures the specified directory exists for file output operations.
    If the directory already exists, no action is taken.
    
    Args:
        output_dir: Path to the directory to create
        
    Returns:
        The same directory path that was provided
        
    Raises:
        OSError: If directory creation fails due to permissions
        
    Example:
        output_path = ensure_output_directory("./reports")
        # Directory is guaranteed to exist
    """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def configure_filesystem_server(context, logger):
    """
    Configure filesystem server with current working directory.
    
    Sets up the MCP filesystem server to work with the current working directory,
    enabling file operations within the agent's execution context.
    
    Args:
        context: MCP application context containing server configuration
        logger: Logger instance for recording configuration changes
        
    Returns:
        None
        
    Side Effects:
        Modifies context.config.mcp.servers["filesystem"].args to include cwd
    """
    if "filesystem" in context.config.mcp.servers:
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
        logger.info("Filesystem server configured")

def create_executive_summary(output_dir, agent_name, company_name=None, 
                           timestamp=None, title=None, overview=None, 
                           impact_metrics=None, initiatives=None, 
                           action_items=None, investment_analysis=None,
                           kpis=None, timeline=None, next_steps=None):
    """Create standardized executive summary report"""
    
    if not timestamp:
        timestamp = get_now_formatted()
    if not company_name:
        company_name = settings.reporting.default_company_name
    if not title:
        title = f"{agent_name.title()} Executive Summary"
    
    dashboard_path = os.path.join(output_dir, f"{agent_name}_executive_summary_{timestamp}.md")
    
    with open(dashboard_path, 'w') as f:
        f.write(f"""# {title} - {company_name}
## Generated: {datetime.now().strftime(settings.reporting.timestamp_format)}

### 🎯 {overview.get('title', 'Transformation Overview')}
{overview.get('content', 'Comprehensive analysis completed with actionable strategies.')}

### 📈 Expected Business Impact
{_format_metrics(impact_metrics)}

### 🚀 Strategic Initiatives
{_format_initiatives(initiatives)}

### 🚨 Critical Action Items
{_format_action_items(action_items)}

### 💰 Investment and ROI Analysis
{_format_investment_analysis(investment_analysis)}

### 📊 Key Performance Indicators
{_format_kpis(kpis)}

### 🗓️ Implementation Timeline
{_format_timeline(timeline)}

### 📞 Next Steps
{_format_next_steps(next_steps)}

For detailed technical information and implementation guides, refer to individual reports in {output_dir}/

---
*This executive summary provides a high-level view of the {agent_name} opportunity.
For comprehensive strategies and detailed implementation plans, please review the complete analysis reports.*
""")
    
    return dashboard_path

def create_kpi_template(output_dir, agent_name, kpi_structure, timestamp=None):
    """Create standardized KPI tracking template"""
    
    if not timestamp:
        timestamp = get_now_formatted()
    
    kpi_path = os.path.join(output_dir, f"{agent_name}_kpi_template_{timestamp}.json")
    
    kpi_template = {
        **kpi_structure,
        "reporting_period": f"{datetime.now().strftime('%Y-%m')}",
        "last_updated": datetime.now().isoformat()
    }
    
    with open(kpi_path, 'w') as f:
        json.dump(kpi_template, f, indent=2)
    
    return kpi_path

def save_deliverables(orchestrator_result, output_dir, deliverable_files):
    """Save orchestrator results to specified deliverable files"""
    # Implementation would depend on orchestrator result format
    # This is a placeholder for the actual implementation
    pass

def save_report(report_data, file_path: str | None = None, output_dir: str | None = None) -> str:
    """Persist *report_data* to disk and return the absolute file path.

    • If *file_path* is provided it is treated as relative to *output_dir* (when
      given) or the current working directory.
    • If omitted, an auto‐generated name of the form ``report_YYYYmmdd_HHMMSS.json``
      is created in *output_dir* (defaults to ``reports/``).

    The function handles both ``dict``/``list`` (saved as JSON) and plain strings
    (saved verbatim).
    """
    if output_dir is None:
        output_dir = os.getenv("MCP_REPORTS_DIR", "reports")
    os.makedirs(output_dir, exist_ok=True)

    if file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"report_{timestamp}.json"

    if not os.path.isabs(file_path):
        file_path = os.path.join(output_dir, file_path)

    # Choose encoding/format based on data type
    if isinstance(report_data, (dict, list)):
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(report_data, fp, indent=2, cls=EnhancedJSONEncoder, ensure_ascii=False)
    else:
        # Fallback: store as plain UTF-8 text
        with open(file_path, "w", encoding="utf-8") as fp:
            fp.write(str(report_data))

    return os.path.abspath(file_path)

def _format_metrics(metrics):
    """Format impact metrics for executive summary"""
    if not metrics:
        return "- Metrics to be defined based on specific analysis"
    
    formatted = []
    for metric, value in metrics.items():
        formatted.append(f"- **{metric}**: {value}")
    return "\n".join(formatted)

def _format_initiatives(initiatives):
    """Format strategic initiatives for executive summary"""
    if not initiatives:
        return "1. **Initiative Planning** - Define specific strategic initiatives"
    
    formatted = []
    for i, (title, description) in enumerate(initiatives.items(), 1):
        formatted.append(f"{i}. **{title}** - {description}")
    return "\n".join(formatted)

def _format_action_items(action_items):
    """Format action items for executive summary"""
    if not action_items:
        return "- [ ] Define specific action items based on analysis"
    
    formatted = []
    for item in action_items:
        formatted.append(f"- [ ] {item}")
    return "\n".join(formatted)

def _format_investment_analysis(investment_analysis):
    """Format investment analysis for executive summary"""
    if not investment_analysis:
        return "**Investment Analysis**: To be determined based on specific requirements and scope."
    
    formatted = []
    for phase, details in investment_analysis.items():
        formatted.append(f"**{phase}**: {details}")
    return "\n".join(formatted)

def _format_kpis(kpis):
    """Format KPIs for executive summary"""
    if not kpis:
        return "- Performance metrics to be defined based on specific objectives"
    
    formatted = []
    for kpi, target in kpis.items():
        formatted.append(f"- {kpi}: Target {target}")
    return "\n".join(formatted)

def _format_timeline(timeline):
    """Format implementation timeline for executive summary"""
    if not timeline:
        return "**Timeline**: Phased implementation approach to be defined."
    
    formatted = []
    for period, activities in timeline.items():
        formatted.append(f"**{period}**: {activities}")
    return "\n".join(formatted)

def _format_next_steps(next_steps):
    """Format next steps for executive summary"""
    if not next_steps:
        return "1. Define specific next steps based on analysis results"
    
    formatted = []
    for i, step in enumerate(next_steps, 1):
        formatted.append(f"{i}. {step}")
    return "\n".join(formatted)

def get_now_formatted() -> str:
    """현재 시간을 기본 포맷으로 반환합니다."""
    return datetime.now().strftime(settings.reporting.timestamp_format)

def generate_report_header(company_name: str | None = None) -> str:
    """
    보고서 헤더를 생성합니다.
    """
    if company_name is None:
        company_name = settings.reporting.default_company_name
    
    formatted_time = datetime.now().strftime(settings.reporting.timestamp_format)
    
    return f"""
## {company_name} - 자동 생성 보고서
## Generated: {formatted_time}
"""

__all__ = [
    "setup_agent_app", "ensure_output_directory", "configure_filesystem_server",
    "create_executive_summary", "create_kpi_template", "save_deliverables",
    "save_report"
] 