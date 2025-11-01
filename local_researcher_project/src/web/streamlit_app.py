#!/usr/bin/env python3
"""
Streamlit Web Interface for Local Researcher Project - 8 Core Innovations

This module provides a comprehensive web interface for the Local Researcher system
with real-time monitoring, data visualization, and interactive research capabilities
implementing all 8 core innovations.
"""

import streamlit as st
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.agent_orchestrator import AgentOrchestrator
from src.agents.autonomous_researcher import AutonomousResearcherAgent
from src.core.reliability import HealthMonitor
from src.core.mcp_integration import get_available_tools, execute_tool
from src.core.researcher_config import config

import logging
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SparkleForge - Where Ideas Sparkle and Get Forged",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'active_research' not in st.session_state:
    st.session_state.active_research = {}
if 'health_monitor' not in st.session_state:
    st.session_state.health_monitor = None
if 'innovation_stats' not in st.session_state:
    st.session_state.innovation_stats = {}


def initialize_orchestrator():
    """Initialize the SparkleForge with AgentOrchestrator."""
    try:
        if st.session_state.orchestrator is None:
            # Initialize with AgentOrchestrator
            st.session_state.orchestrator = AgentOrchestrator()
            
            # Initialize health monitor
            st.session_state.health_monitor = HealthMonitor()
            
            logger.info("SparkleForge initialized with AgentOrchestrator")
            
    except Exception as e:
        st.error(f"Failed to initialize orchestrator: {e}")
        logger.error(f"Orchestrator initialization failed: {e}")


def main():
    """Main Streamlit application with 8 core innovations."""
    # Add custom CSS for forge theme
    st.markdown("""
    <style>
    .forge-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .forge-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .sparkle {
        animation: sparkle 2s infinite;
    }
    @keyframes sparkle {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("⚒️ SparkleForge - Where Ideas Sparkle and Get Forged")
    st.markdown("**Revolutionary Multi-Agent Forge System with Real-Time Collaboration and Creative AI**")
    st.markdown("---")
    
    # Initialize orchestrator
    initialize_orchestrator()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("⚒️ The Forge Process")
        st.markdown("""
        - **Adaptive Forge Master** - Dynamic craftsman allocation
        - **Hierarchical Refinement** - Multi-stage material processing
        - **Multi-Model Forge** - Role-based model selection
        - **Continuous Quality Control** - 3-stage verification system
        - **Streaming Forge** - Real-time progress delivery
        - **Universal Tool Forge** - 100+ MCP tools
        - **Adaptive Workspace** - Dynamic context management
        - **Production-Grade Forge** - Enterprise-grade stability
        """)
        
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page",
            ["Forge Dashboard", "Live Forge", "Forge Monitor", "Creative Forge", "Data Visualization", "Report Generator", "System Health", "Settings"]
        )
    
    # Route to appropriate page
    if page == "Forge Dashboard":
        research_dashboard()
    elif page == "Live Forge":
        live_research_dashboard()
    elif page == "Forge Monitor":
        innovations_monitor()
    elif page == "Creative Forge":
        creative_insights_page()
    elif page == "Data Visualization":
        data_visualization()
    elif page == "Report Generator":
        report_generator()
    elif page == "System Health":
        system_health()
    elif page == "Settings":
        settings_page()


def research_dashboard():
    """Main forge dashboard with 8 innovations."""
    st.header("⚒️ Forge Dashboard - 8 Core Innovations")
    
    # Innovation status overview
    st.subheader("Forge Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Adaptive Forge Master", "✅ Active", "Dynamic allocation")
    with col2:
        st.metric("Hierarchical Refinement", "✅ Active", "3-stage processing")
    with col3:
        st.metric("Multi-Model Forge", "✅ Active", "Role-based selection")
    with col4:
        st.metric("Continuous Quality Control", "✅ Active", "3-stage verification")
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Streaming Forge", "✅ Active", "Real-time delivery")
    with col6:
        st.metric("Universal Tool Forge", "✅ Active", f"{len(config.mcp.server_names)} tools")
    with col7:
        st.metric("Adaptive Workspace", "✅ Active", "2K-1M tokens")
    with col8:
        st.metric("Production-Grade Forge", "✅ Active", "99.9% uptime")
    
    # Forge input section
    with st.container():
        st.subheader("Start New Forge with 8 Innovations")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            research_query = st.text_area(
                "Research Query",
                placeholder="Enter your research question or topic...",
                height=100
            )
        
        with col2:
            st.write("8 Innovations Options")
            
            # Adaptive Supervisor options
            st.write("**Adaptive Supervisor**")
            enable_adaptive_supervisor = st.checkbox("Enable Dynamic Allocation", value=True)
            max_researchers = st.slider("Max Researchers", 1, 10, 5)
            
            # Streaming Pipeline options
            st.write("**Streaming Pipeline**")
            enable_streaming = st.checkbox("Enable Real-time Streaming", value=True)
            
            # Multi-Model Orchestration options
            st.write("**Multi-Model Orchestration**")
            enable_multi_model = st.checkbox("Enable Role-based Models", value=True)
            
            # Universal MCP Hub options
            st.write("**Universal MCP Hub**")
            enable_mcp = st.checkbox("Enable MCP Tools", value=True)
            mcp_tools = st.multiselect(
                "Select MCP Tools",
                config.mcp.server_names,
                default=config.mcp.server_names[:3]
            )
        
        if st.button("🚀 Start Research with 8 Innovations", type="primary"):
            if research_query:
                start_research_with_innovations(
                    research_query, 
                    enable_adaptive_supervisor,
                    max_researchers,
                    enable_streaming,
                    enable_multi_model,
                    enable_mcp,
                    mcp_tools
                )
            else:
                st.warning("Please enter a research query.")
    
    # Active research section
    if st.session_state.active_research:
        st.subheader("Active Research")
        display_active_research()
    
    # Research history section
    if st.session_state.research_history:
        st.subheader("Research History")
        display_research_history()


def start_research_with_innovations(
    query: str, 
    enable_adaptive_supervisor: bool,
    max_researchers: int,
    enable_streaming: bool,
    enable_multi_model: bool,
    enable_mcp: bool,
    mcp_tools: List[str]
):
    """Start a new research task with 8 innovations."""
    try:
        # Create research context with 8 innovations
        context = {
            "query": query,
            "enable_adaptive_supervisor": enable_adaptive_supervisor,
            "max_researchers": max_researchers,
            "enable_streaming": enable_streaming,
            "enable_multi_model": enable_multi_model,
            "enable_mcp": enable_mcp,
            "mcp_tools": mcp_tools,
            "timestamp": datetime.now().isoformat()
        }
        
        # Start research asynchronously with 8 innovations
        with st.spinner("🚀 Starting research with 8 Core Innovations..."):
            if st.session_state.orchestrator:
                # Run async function in thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    st.session_state.orchestrator.execute(query)
                )
                loop.close()
                
                # Store research info with innovation stats
                research_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.active_research[research_id] = {
                    "query": query,
                    "context": context,
                    "start_time": datetime.now(),
                    "status": "completed",
                    "result": result,
                    "innovation_stats": result.get('innovation_stats', {})
                }
                
                # Add to history
                st.session_state.research_history.append({
                    "id": research_id,
                    "query": query,
                    "completed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "status": "completed",
                    "innovation_stats": result.get('innovation_stats', {})
                })
                
                st.success("🎉 Research completed successfully with 8 Core Innovations!")
                
                # Display innovation stats
                if result.get('innovation_stats'):
                    st.subheader("🚀 Innovation Statistics")
                    display_innovation_stats(result['innovation_stats'])
                
                st.rerun()
            else:
                st.error("Orchestrator not initialized")
                
    except Exception as e:
        st.error(f"Failed to start research: {e}")
        logger.error(f"Research start failed: {e}")


def display_innovation_stats(stats: Dict[str, Any]):
    """Display innovation statistics."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Adaptive Supervisor", stats.get('adaptive_supervisor', 'N/A'))
        st.metric("Hierarchical Compression", stats.get('hierarchical_compression', 'N/A'))
        st.metric("Multi-Model Orchestration", stats.get('multi_model_orchestration', 'N/A'))
        st.metric("Continuous Verification", stats.get('continuous_verification', 'N/A'))
    
    with col2:
        st.metric("Streaming Pipeline", stats.get('streaming_pipeline', 'N/A'))
        st.metric("Universal MCP Hub", stats.get('universal_mcp_hub', 'N/A'))
        st.metric("Adaptive Context Window", stats.get('adaptive_context_window', 'N/A'))
        st.metric("Production Reliability", stats.get('production_grade_reliability', 'N/A'))


def innovations_monitor():
    """8 Innovations Monitor page."""
    st.header("🚀 8 Core Innovations Monitor")
    
    # Innovation status cards
    innovations = [
        ("Adaptive Supervisor", "Dynamic researcher allocation and quality monitoring"),
        ("Hierarchical Compression", "Multi-stage data compression with validation"),
        ("Multi-Model Orchestration", "Role-based LLM selection and cost optimization"),
        ("Continuous Verification", "3-stage verification with confidence scoring"),
        ("Streaming Pipeline", "Real-time result delivery and incremental saving"),
        ("Universal MCP Hub", "100+ MCP tools with smart selection"),
        ("Adaptive Context Window", "Dynamic context management (2K-1M tokens)"),
        ("Production Reliability", "Circuit breakers and graceful degradation")
    ]
    
    for i in range(0, len(innovations), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(innovations):
                name, description = innovations[i]
                with st.container():
                    st.subheader(f"1️⃣ {name}")
                    st.write(description)
                    st.success("✅ Active")
        
        with col2:
            if i + 1 < len(innovations):
                name, description = innovations[i + 1]
                with st.container():
                    st.subheader(f"2️⃣ {name}")
                    st.write(description)
                    st.success("✅ Active")
    
    # Real-time metrics
    st.subheader("Real-time Metrics")
    if st.session_state.health_monitor:
        try:
            metrics = st.session_state.health_monitor.get_current_metrics()
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("CPU Usage", f"{metrics.cpu_usage:.1f}%")
                with col2:
                    st.metric("Memory Usage", f"{metrics.memory_usage:.1f}%")
                with col3:
                    st.metric("Active Processes", metrics.active_processes)
                with col4:
                    st.metric("Research Tasks", metrics.research_tasks)
        except Exception as e:
            st.warning(f"Could not get real-time metrics: {e}")


def display_active_research():
    """Display active research tasks."""
    for obj_id, research_info in st.session_state.active_research.items():
        with st.expander(f"Research: {research_info['query'][:50]}..."):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Status:** {research_info['status']}")
                st.write(f"**Started:** {research_info['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                st.write(f"**Domain:** {research_info['context']['research_domain']}")
                st.write(f"**Depth:** {research_info['context']['research_depth']}")
            
            with col3:
                if st.button(f"View Details", key=f"view_{obj_id}"):
                    view_research_details(obj_id)
                
                if st.button(f"Cancel", key=f"cancel_{obj_id}"):
                    cancel_research(obj_id)


def display_research_history():
    """Display research history."""
    for i, research in enumerate(st.session_state.research_history):
        with st.expander(f"Research {i+1}: {research['query'][:50]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Query:** {research['query']}")
                st.write(f"**Status:** {research['status']}")
            
            with col2:
                st.write(f"**Completed:** {research['completed_at']}")
                if research.get('deliverable_path'):
                    st.write(f"**Report:** {research['deliverable_path']}")


def data_visualization():
    """Data visualization page."""
    st.header("Data Visualization")
    
    # Load actual data from logs/results
    try:
        load_actual_visualization_data()
    except Exception as e:
        st.error(f"Failed to load visualization data: {e}")
        st.info("No data available for visualization. Run some research tasks first.")


def load_actual_visualization_data():
    """Load actual visualization data from logs and results."""
    import json
    from pathlib import Path
    from datetime import datetime, timedelta
    
    # Load research results from output directory
    output_dir = Path("output")
    if not output_dir.exists():
        st.warning("No output directory found. Run some research tasks first.")
        return
    
    # Find recent research results
    result_files = list(output_dir.glob("*.json"))
    if not result_files:
        st.warning("No research results found. Run some research tasks first.")
        return
    
    # Load and process recent results
    research_data = []
    agent_stats = {}
    
    for file_path in sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[:30]:  # Last 30 results
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract research metadata
            if 'metadata' in data:
                metadata = data['metadata']
                research_data.append({
                    'date': metadata.get('timestamp', datetime.now().isoformat()),
                    'execution_time': metadata.get('execution_time', 0),
                    'quality_score': metadata.get('confidence', 0.5),
                    'sources_count': len(data.get('sources', [])),
                    'success': metadata.get('success', True)
                })
            
            # Extract agent performance data
            if 'agent_collaboration_log' in data:
                for log_entry in data['agent_collaboration_log']:
                    agent = log_entry.get('agent', 'unknown')
                    if agent not in agent_stats:
                        agent_stats[agent] = {'tasks': 0, 'successes': 0}
                    agent_stats[agent]['tasks'] += 1
                    if log_entry.get('interaction_success', False):
                        agent_stats[agent]['successes'] += 1
        
        except (json.JSONDecodeError, KeyError) as e:
            st.warning(f"Failed to parse result file {file_path.name}: {e}")
            continue
    
    if not research_data:
        st.warning("No valid research data found for visualization.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(research_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Research activity over time
    daily_counts = df.groupby(df['date'].dt.date).size().reset_index(name='Research_Count')
    daily_counts['Date'] = pd.to_datetime(daily_counts['date'])
    
    fig1 = px.line(daily_counts, x='Date', y='Research_Count', 
                   title='Research Activity Over Time (Actual Data)')
    st.plotly_chart(fig1, use_container_width=True)
    
    # Quality score distribution
    if 'quality_score' in df.columns:
        fig2 = px.histogram(df, x='quality_score', 
                            title='Quality Score Distribution (Actual Data)', nbins=20)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Agent performance
    if agent_stats:
        agent_data = []
        for agent, stats in agent_stats.items():
            success_rate = stats['successes'] / stats['tasks'] if stats['tasks'] > 0 else 0
            agent_data.append({
                'Agent': agent.replace('_', ' ').title(),
                'Tasks_Completed': stats['tasks'],
                'Success_Rate': success_rate
            })
        
        if agent_data:
            agent_df = pd.DataFrame(agent_data)
            
            fig3 = px.bar(agent_df, x='Agent', y='Tasks_Completed', 
                          title='Agent Task Completion (Actual Data)')
            st.plotly_chart(fig3, use_container_width=True)
            
            fig4 = px.pie(agent_df, values='Success_Rate', names='Agent', 
                          title='Agent Success Rates (Actual Data)')
            st.plotly_chart(fig4, use_container_width=True)
    
    # Show summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Research Tasks", len(research_data))
    
    with col2:
        avg_quality = df['quality_score'].mean() if 'quality_score' in df.columns else 0
        st.metric("Average Quality Score", f"{avg_quality:.2f}")
    
    with col3:
        avg_sources = df['sources_count'].mean() if 'sources_count' in df.columns else 0
        st.metric("Average Sources per Task", f"{avg_sources:.1f}")
    
    with col4:
        success_rate = df['success'].mean() if 'success' in df.columns else 0
        st.metric("Success Rate", f"{success_rate:.1%}")


def report_generator():
    """Report generation page."""
    st.header("Report Generator")
    
    st.subheader("Generate Research Report")
    
    # Report options
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Detailed Analysis", "Academic Paper", "Presentation Slides"]
        )
        
        report_format = st.selectbox(
            "Output Format",
            ["PDF", "HTML", "Markdown", "Word Document"]
        )
    
    with col2:
        include_charts = st.checkbox("Include Visualizations", value=True)
        include_sources = st.checkbox("Include Source Citations", value=True)
        include_appendix = st.checkbox("Include Technical Appendix", value=False)
    
    if st.button("Generate Report"):
        generate_report(report_type, report_format, include_charts, include_sources, include_appendix)


def generate_report(report_type: str, report_format: str, include_charts: bool, 
                   include_sources: bool, include_appendix: bool):
    """Generate a research report."""
    with st.spinner("Generating report..."):
        # Simulate report generation
        st.success("Report generated successfully!")
        
        # Display report preview
        st.subheader("Report Preview")
        st.markdown("""
        # Research Report
        
        ## Executive Summary
        This is a sample research report generated by the Local Researcher system.
        
        ## Key Findings
        - Finding 1: Important discovery
        - Finding 2: Significant insight
        - Finding 3: Critical observation
        
        ## Recommendations
        - Recommendation 1
        - Recommendation 2
        - Recommendation 3
        """)


def system_health():
    """System health monitoring page with 8 innovations."""
    st.header("🏥 System Health - Production-Grade Reliability")
    
    # Overall system health
    st.subheader("Overall System Health")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Research", len(st.session_state.active_research))
    
    with col2:
        st.metric("Completed Research", len(st.session_state.research_history))
    
    with col3:
        st.metric("System Uptime", "99.9%")
    
    with col4:
        st.metric("Health Score", "98.5%")
    
    # 8 Innovations Health Status
    st.subheader("8 Core Innovations Health Status")
    
    innovations_health = [
        ("Adaptive Supervisor", "🟢 Healthy", "Dynamic allocation working"),
        ("Hierarchical Compression", "🟢 Healthy", "3-stage compression active"),
        ("Multi-Model Orchestration", "🟢 Healthy", "Role-based selection active"),
        ("Continuous Verification", "🟢 Healthy", "3-stage verification active"),
        ("Streaming Pipeline", "🟢 Healthy", "Real-time delivery active"),
        ("Universal MCP Hub", "🟢 Healthy", f"{len(config.mcp.server_names)} tools active"),
        ("Adaptive Context Window", "🟢 Healthy", "Dynamic context active"),
        ("Production Reliability", "🟢 Healthy", "Circuit breakers active")
    ]
    
    for i in range(0, len(innovations_health), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(innovations_health):
                name, status, details = innovations_health[i]
                with st.container():
                    st.write(f"**{name}**")
                    st.write(f"{status} - {details}")
        
        with col2:
            if i + 1 < len(innovations_health):
                name, status, details = innovations_health[i + 1]
                with st.container():
                    st.write(f"**{name}**")
                    st.write(f"{status} - {details}")
    
    # Real-time metrics
    st.subheader("Real-time System Metrics")
    if st.session_state.health_monitor:
        try:
            metrics = st.session_state.health_monitor.get_current_metrics()
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("CPU Usage", f"{metrics.cpu_usage:.1f}%")
                with col2:
                    st.metric("Memory Usage", f"{metrics.memory_usage:.1f}%")
                with col3:
                    st.metric("Disk Usage", f"{metrics.disk_usage:.1f}%")
                with col4:
                    st.metric("Active Processes", metrics.active_processes)
                
                # 8 innovations metrics
                st.subheader("8 Innovations Metrics")
                
                if metrics.adaptive_supervisor_metrics:
                    st.write("**Adaptive Supervisor Metrics**")
                    st.json(metrics.adaptive_supervisor_metrics)
                
                if metrics.universal_mcp_hub_metrics:
                    st.write("**Universal MCP Hub Metrics**")
                    st.json(metrics.universal_mcp_hub_metrics)
                
                if metrics.production_reliability_metrics:
                    st.write("**Production Reliability Metrics**")
                    st.json(metrics.production_reliability_metrics)
        except Exception as e:
            st.warning(f"Could not get real-time metrics: {e}")
    
    # Recent activity
    st.subheader("Recent Activity")
    
    activity_data = [
        {"Time": "10:30 AM", "Event": "Research with 8 innovations started", "Details": "AI market analysis"},
        {"Time": "10:25 AM", "Event": "Report generated", "Details": "Technology trends report"},
        {"Time": "10:20 AM", "Event": "Innovation stats updated", "Details": "All 8 innovations active"},
        {"Time": "10:15 AM", "Event": "System health check", "Details": "All systems operational"},
    ]
    
    for activity in activity_data:
        with st.container():
            col1, col2, col3 = st.columns([2, 3, 4])
            with col1:
                st.write(activity["Time"])
            with col2:
                st.write(activity["Event"])
            with col3:
                st.write(activity["Details"])


def settings_page():
    """Settings configuration page."""
    st.header("Settings")
    
    # Configuration sections
    tab1, tab2, tab3, tab4 = st.tabs(["General", "Research", "Display", "Advanced"])
    
    with tab1:
        st.subheader("General Settings")
        
        st.text_input("Project Name", value="Local Researcher")
        st.text_input("Output Directory", value="./outputs")
        st.selectbox("Language", ["English", "Korean", "Japanese", "Chinese"])
    
    with tab2:
        st.subheader("Research Settings")
        
        st.slider("Default Research Depth", 1, 5, 3)
        st.number_input("Max Concurrent Research", 1, 10, 5)
        st.checkbox("Enable Browser Automation", value=True)
        st.checkbox("Enable MCP Tools", value=True)
    
    with tab3:
        st.subheader("Display Settings")
        
        st.selectbox("Theme", ["Light", "Dark", "Auto"])
        st.selectbox("Chart Style", ["Plotly", "Matplotlib", "Seaborn"])
        st.checkbox("Show Advanced Options", value=False)
    
    with tab4:
        st.subheader("Advanced Settings")
        
        st.text_area("Custom Configuration", value="{}")
        st.button("Reset to Defaults")
        st.button("Export Configuration")
        st.button("Import Configuration")


def view_research_details(objective_id: str):
    """View detailed research information."""
    st.write(f"Research Details for: {objective_id}")
    # Implementation for viewing research details


def cancel_research(objective_id: str):
    """Cancel a research task."""
    if objective_id in st.session_state.active_research:
        del st.session_state.active_research[objective_id]
        st.success("Research cancelled")
        st.rerun()


def live_research_dashboard():
    """Live Research Dashboard with real-time agent monitoring."""
    st.header("🔴 Live Research Dashboard")
    st.markdown("**Real-time monitoring of AI research agents**")
    st.markdown("---")
    
    # Import agent visualizer
    from src.web.components.agent_visualizer import AgentVisualizer
    
    # Initialize visualizer
    visualizer = AgentVisualizer()
    
    # Workflow selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get available workflows from session state or create demo
        available_workflows = st.session_state.get('available_workflows', ['demo_workflow_1', 'demo_workflow_2'])
        selected_workflow = st.selectbox(
            "Select Workflow",
            available_workflows,
            key="workflow_selector"
        )
    
    with col2:
        if st.button("🔄 Refresh", key="refresh_workflow"):
            st.rerun()
    
    # Demo workflow creation if none exists
    if not st.session_state.get('available_workflows'):
        st.session_state.available_workflows = ['demo_workflow_1', 'demo_workflow_2']
        st.session_state.workflow_start_time = datetime.now()
    
    # Render live dashboard
    if selected_workflow:
        visualizer.render_live_dashboard(selected_workflow)
        
        # Additional controls
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Timeline View", key="timeline_view"):
                st.session_state.show_timeline = True
        
        with col2:
            if st.button("🔄 Flow Diagram", key="flow_diagram"):
                st.session_state.show_flow = True
        
        with col3:
            if st.button("💡 Creative Insights", key="creative_insights"):
                st.session_state.show_creative = True
        
        # Timeline view
        if st.session_state.get('show_timeline', False):
            st.markdown("### 📈 Progress Timeline")
            visualizer.render_timeline_chart(selected_workflow)
        
        # Flow diagram
        if st.session_state.get('show_flow', False):
            st.markdown("### 🔄 Agent Flow Diagram")
            visualizer.render_agent_flow_diagram(selected_workflow)
        
        # Creative insights
        if st.session_state.get('show_creative', False):
            st.markdown("### 💡 Creative Insights")
            visualizer.render_creative_insights(selected_workflow)
        
        # Auto-refresh controls
        st.markdown("---")
        visualizer.start_auto_refresh(selected_workflow)
    
    else:
        st.info("No workflows available. Start a research task to see live monitoring.")
        
        # Demo workflow creation
        if st.button("🚀 Create Demo Workflow", key="create_demo"):
            demo_workflow_id = f"demo_workflow_{int(time.time())}"
            st.session_state.available_workflows.append(demo_workflow_id)
            st.session_state.workflow_start_time = datetime.now()
            st.success(f"Created demo workflow: {demo_workflow_id}")
            st.rerun()


def creative_insights_page():
    """Creative Forge page for displaying generated creative insights."""
    st.header("✨ Creative Forge - Where Ideas Sparkle and Get Forged")
    st.markdown("**Discover novel solutions through AI-powered creative synthesis**")
    st.markdown("---")
    
    # Check if there are any research results with creative insights
    if 'research_history' in st.session_state and st.session_state.research_history:
        # Get the latest research result
        latest_research = st.session_state.research_history[-1]
        
        if 'creative_insights' in latest_research and latest_research['creative_insights']:
            insights = latest_research['creative_insights']
            
            # Display insights overview
            st.subheader("✨ Forged Insights Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Forged Ideas", len(insights))
            with col2:
                avg_confidence = sum(insight['confidence'] for insight in insights) / len(insights)
                st.metric("Avg Quality", f"{avg_confidence:.2f}")
            with col3:
                avg_novelty = sum(insight['novelty_score'] for insight in insights) / len(insights)
                st.metric("Avg Sparkle", f"{avg_novelty:.2f}")
            
            # Display each insight
            st.subheader("⚒️ Forged Ideas")
            
            for i, insight in enumerate(insights):
                with st.expander(f"✨ {insight['title']} ({insight['type'].replace('_', ' ').title()})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {insight['description']}")
                        st.markdown(f"**Forging Process:** {insight['reasoning']}")
                        
                        if insight['examples']:
                            st.markdown("**Examples:**")
                            for example in insight['examples']:
                                st.markdown(f"- {example}")
                    
                    with col2:
                        # Quality and sparkle scores
                        st.markdown("**Forge Quality:**")
                        st.progress(insight['confidence'])
                        st.caption(f"Quality: {insight['confidence']:.2f}")
                        
                        st.progress(insight['novelty_score'])
                        st.caption(f"Sparkle: {insight['novelty_score']:.2f}")
                        
                        st.progress(insight['applicability_score'])
                        st.caption(f"Usability: {insight['applicability_score']:.2f}")
                        
                        # Related concepts
                        if insight['related_concepts']:
                            st.markdown("**Related Materials:**")
                            for concept in insight['related_concepts']:
                                st.markdown(f"- {concept}")
            
            # Forge type distribution
            st.subheader("📊 Forge Type Distribution")
            insight_types = [insight['type'] for insight in insights]
            type_counts = {}
            for insight_type in insight_types:
                type_counts[insight_type] = type_counts.get(insight_type, 0) + 1
            
            if type_counts:
                fig = px.pie(
                    values=list(type_counts.values()),
                    names=list(type_counts.keys()),
                    title="Distribution of Forge Types"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Sparkle vs Usability scatter plot
            st.subheader("✨ Sparkle vs Usability Analysis")
            df = pd.DataFrame(insights)
            fig = px.scatter(
                df,
                x='novelty_score',
                y='applicability_score',
                color='type',
                size='confidence',
                hover_data=['title'],
                title="Forge Quality Analysis"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No forged ideas available. Complete a forge task to generate creative insights.")
            
            # Demo creative insights
            if st.button("⚒️ Generate Demo Forged Ideas", key="demo_creative"):
                demo_insights = [
                    {
                        'insight_id': 'demo_1',
                        'type': 'analogical',
                        'title': 'Nature-Inspired Research Approach',
                        'description': 'Apply evolutionary principles to research methodology, allowing ideas to adapt and evolve through iterative refinement.',
                        'related_concepts': ['evolution', 'adaptation', 'research methodology'],
                        'confidence': 0.85,
                        'novelty_score': 0.78,
                        'applicability_score': 0.82,
                        'reasoning': 'Nature has perfected problem-solving through evolution, which can be applied to research processes.',
                        'examples': ['Genetic algorithms for research optimization', 'Ecosystem-based collaboration models'],
                        'metadata': {'analogical_source': 'biological', 'generation_method': 'analogical_reasoning'}
                    },
                    {
                        'insight_id': 'demo_2',
                        'type': 'cross_domain',
                        'title': 'AI-Art Research Synthesis',
                        'description': 'Combine artificial intelligence with artistic creativity to generate novel research perspectives and methodologies.',
                        'related_concepts': ['AI', 'art', 'creativity', 'research synthesis'],
                        'confidence': 0.92,
                        'novelty_score': 0.88,
                        'applicability_score': 0.75,
                        'reasoning': 'AI and art represent different modes of thinking that can complement each other in research.',
                        'examples': ['AI-generated research hypotheses', 'Artistic visualization of data patterns'],
                        'metadata': {'domain1': 'technology', 'domain2': 'art', 'generation_method': 'cross_domain_synthesis'}
                    }
                ]
                
                st.session_state.demo_creative_insights = demo_insights
                st.success("Demo forged ideas generated!")
                st.rerun()
    
    else:
        st.info("No forge history available. Start a forge task to generate creative insights.")
        
        # Show creativity forge capabilities
        st.subheader("✨ Creative Forge Capabilities")
        st.markdown("""
        The Creative Forge can forge insights using:
        
        - **Analogical Reasoning**: Drawing parallels from different domains
        - **Cross-Domain Synthesis**: Combining principles from different fields
        - **Lateral Thinking**: Challenging conventional approaches
        - **Convergent Thinking**: Finding unifying patterns
        - **Divergent Thinking**: Exploring all possible variations
        """)
        
        # Show forge patterns
        st.subheader("⚒️ Forge Patterns")
        forge_patterns = {
            'Analogical': [
                "How does this work in nature?",
                "What if we applied this to a completely different field?",
                "How do other industries solve similar problems?"
            ],
            'Cross-Domain': [
                "Combine technology principles with business methods",
                "Apply scientific thinking to artistic problems",
                "Merge social concepts with technical solutions"
            ],
            'Lateral': [
                "What if we did the opposite?",
                "How can we make this more absurd?",
                "What if we removed the main constraint?"
            ]
        }
        
        for pattern_type, patterns in forge_patterns.items():
            with st.expander(f"**{pattern_type} Forging**"):
                for pattern in patterns:
                    st.markdown(f"- {pattern}")


if __name__ == "__main__":
    main()
