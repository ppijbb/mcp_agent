"""
Common Utilities Module

Shared utility functions used across all agents for common operations.
"""

import os
import json
from datetime import datetime
from .config import get_timestamp, SUMMARY_TIMESTAMP_FORMAT, DEFAULT_COMPANY_NAME
from .imports import MCPApp, get_settings

def setup_agent_app(app_name, config_path="configs/mcp_agent.config.yaml"):
    """Setup and configure MCP app with standard settings"""
    return MCPApp(
        name=app_name,
        settings=get_settings(config_path),
        human_input_callback=None
    )

def ensure_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def configure_filesystem_server(context, logger):
    """Configure filesystem server with current working directory"""
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
        timestamp = get_timestamp()
    if not company_name:
        company_name = DEFAULT_COMPANY_NAME
    if not title:
        title = f"{agent_name.title()} Executive Summary"
    
    dashboard_path = os.path.join(output_dir, f"{agent_name}_executive_summary_{timestamp}.md")
    
    with open(dashboard_path, 'w') as f:
        f.write(f"""# {title} - {company_name}
## Generated: {datetime.now().strftime(SUMMARY_TIMESTAMP_FORMAT)}

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
        timestamp = get_timestamp()
    
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

__all__ = [
    "setup_agent_app", "ensure_output_directory", "configure_filesystem_server",
    "create_executive_summary", "create_kpi_template", "save_deliverables"
] 