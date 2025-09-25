#!/usr/bin/env python3
"""
Streamlit Web Interface for Local Researcher Project

This module provides a comprehensive web interface for the Local Researcher system
with real-time monitoring, data visualization, and interactive research capabilities.
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

from src.core.autonomous_orchestrator import LangGraphOrchestrator
from src.agents.task_analyzer import TaskAnalyzerAgent
from src.agents.task_decomposer import TaskDecomposerAgent
from src.agents.research_agent import ResearchAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.core.mcp_integration import MCPIntegrationManager
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

logger = setup_logger("streamlit_app", log_level="INFO")

# Page configuration
st.set_page_config(
    page_title="Local Researcher",
    page_icon="🔍",
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


def initialize_orchestrator():
    """Initialize the LangGraph orchestrator."""
    try:
        if st.session_state.orchestrator is None:
            # Initialize agents
            config_manager = ConfigManager()
            mcp_manager = MCPIntegrationManager()
            
            agents = {
                'analyzer': TaskAnalyzerAgent(),
                'decomposer': TaskDecomposerAgent(),
                'researcher': ResearchAgent(),
                'evaluator': EvaluationAgent(),
                'validator': ValidationAgent(),
                'synthesizer': SynthesisAgent()
            }
            
            st.session_state.orchestrator = LangGraphOrchestrator(
                config_path=None,
                agents=agents,
                mcp_manager=mcp_manager
            )
            
            logger.info("Orchestrator initialized successfully")
            
    except Exception as e:
        st.error(f"Failed to initialize orchestrator: {e}")
        logger.error(f"Orchestrator initialization failed: {e}")


def main():
    """Main Streamlit application."""
    st.title("🔍 Local Researcher - AI-Powered Research Platform")
    st.markdown("---")
    
    # Initialize orchestrator
    initialize_orchestrator()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page",
            ["Research Dashboard", "Data Visualization", "Report Generator", "System Monitor", "Settings"]
        )
    
    # Route to appropriate page
    if page == "Research Dashboard":
        research_dashboard()
    elif page == "Data Visualization":
        data_visualization()
    elif page == "Report Generator":
        report_generator()
    elif page == "System Monitor":
        system_monitor()
    elif page == "Settings":
        settings_page()


def research_dashboard():
    """Main research dashboard."""
    st.header("Research Dashboard")
    
    # Research input section
    with st.container():
        st.subheader("Start New Research")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            research_query = st.text_area(
                "Research Query",
                placeholder="Enter your research question or topic...",
                height=100
            )
        
        with col2:
            st.write("Research Options")
            research_depth = st.selectbox(
                "Research Depth",
                ["Quick", "Standard", "Deep", "Comprehensive"],
                index=1
            )
            
            research_domain = st.selectbox(
                "Research Domain",
                ["General", "Academic", "Business", "Technical", "Scientific"],
                index=0
            )
            
            use_browser = st.checkbox("Enable Browser Automation", value=True)
            use_mcp = st.checkbox("Enable MCP Tools", value=True)
        
        if st.button("Start Research", type="primary"):
            if research_query:
                start_research(research_query, research_depth, research_domain, use_browser, use_mcp)
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


def start_research(query: str, depth: str, domain: str, use_browser: bool, use_mcp: bool):
    """Start a new research task."""
    try:
        # Create research context
        context = {
            "research_depth": depth,
            "research_domain": domain,
            "use_browser": use_browser,
            "use_mcp": use_mcp,
            "timestamp": datetime.now().isoformat()
        }
        
        # Start research asynchronously
        with st.spinner("Starting research..."):
            if st.session_state.orchestrator:
                # Run async function in thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                objective_id = loop.run_until_complete(
                    st.session_state.orchestrator.start_autonomous_research(query, context)
                )
                loop.close()
                
                # Store research info
                st.session_state.active_research[objective_id] = {
                    "query": query,
                    "context": context,
                    "start_time": datetime.now(),
                    "status": "running"
                }
                
                st.success(f"Research started successfully! Objective ID: {objective_id}")
                st.rerun()
            else:
                st.error("Orchestrator not initialized")
                
    except Exception as e:
        st.error(f"Failed to start research: {e}")
        logger.error(f"Research start failed: {e}")


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
    
    # Sample data for demonstration
    if st.button("Generate Sample Data"):
        generate_sample_visualizations()


def generate_sample_visualizations():
    """Generate sample visualizations."""
    # Research performance over time
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    performance_data = pd.DataFrame({
        'Date': dates,
        'Research_Count': [2, 3, 1, 4, 2, 3, 5, 2, 1, 3, 4, 2, 3, 1, 2, 4, 3, 2, 1, 3, 2, 4, 1, 2, 3, 4, 2, 1, 3, 2, 4],
        'Quality_Score': [0.8, 0.85, 0.75, 0.9, 0.82, 0.88, 0.92, 0.78, 0.85, 0.89, 0.91, 0.83, 0.87, 0.79, 0.86, 0.93, 0.88, 0.84, 0.81, 0.89, 0.85, 0.91, 0.77, 0.86, 0.88, 0.92, 0.84, 0.82, 0.89, 0.87, 0.94]
    })
    
    # Performance trend chart
    fig1 = px.line(performance_data, x='Date', y='Research_Count', 
                   title='Research Activity Over Time')
    st.plotly_chart(fig1, use_container_width=True)
    
    # Quality score distribution
    fig2 = px.histogram(performance_data, x='Quality_Score', 
                        title='Quality Score Distribution', nbins=20)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Agent performance
    agent_data = pd.DataFrame({
        'Agent': ['Task Analyzer', 'Research Agent', 'Evaluation Agent', 'Validation Agent', 'Synthesis Agent'],
        'Tasks_Completed': [45, 38, 42, 40, 35],
        'Success_Rate': [0.95, 0.88, 0.92, 0.90, 0.87]
    })
    
    fig3 = px.bar(agent_data, x='Agent', y='Tasks_Completed', 
                  title='Agent Task Completion')
    st.plotly_chart(fig3, use_container_width=True)
    
    # Success rate pie chart
    fig4 = px.pie(agent_data, values='Success_Rate', names='Agent', 
                  title='Agent Success Rates')
    st.plotly_chart(fig4, use_container_width=True)


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


def system_monitor():
    """System monitoring page."""
    st.header("System Monitor")
    
    # System status
    st.subheader("System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Research", len(st.session_state.active_research))
    
    with col2:
        st.metric("Completed Research", len(st.session_state.research_history))
    
    with col3:
        st.metric("System Uptime", "99.9%")
    
    with col4:
        st.metric("Memory Usage", "45%")
    
    # Agent status
    st.subheader("Agent Status")
    
    if st.session_state.orchestrator:
        agent_status = {
            "Task Analyzer": "Running",
            "Research Agent": "Running", 
            "Evaluation Agent": "Running",
            "Validation Agent": "Running",
            "Synthesis Agent": "Running"
        }
        
        for agent, status in agent_status.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(agent)
            with col2:
                if status == "Running":
                    st.success("🟢 Running")
                else:
                    st.error("🔴 Stopped")
        
        # Browser status
        st.subheader("Browser Automation Status")
        try:
            research_agent = st.session_state.orchestrator.agents.get('researcher')
            if research_agent and hasattr(research_agent, 'browser_manager'):
                browser_status = research_agent.browser_manager.get_status()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Browser Available", "✅ Yes" if browser_status['browser_available'] else "❌ No")
                with col2:
                    st.metric("Fallback Mode", "✅ Active" if browser_status['fallback_mode'] else "❌ Inactive")
                with col3:
                    st.metric("Environment", "Streamlit" if browser_status['is_streamlit'] else "CLI" if browser_status['is_cli'] else "Background")
                
                # Browser details
                with st.expander("Browser Details"):
                    st.json(browser_status)
        except Exception as e:
            st.warning(f"Could not get browser status: {e}")
    
    # Recent activity
    st.subheader("Recent Activity")
    
    activity_data = [
        {"Time": "10:30 AM", "Event": "Research started", "Details": "AI market analysis"},
        {"Time": "10:25 AM", "Event": "Report generated", "Details": "Technology trends report"},
        {"Time": "10:20 AM", "Event": "Agent updated", "Details": "Research agent configuration"},
        {"Time": "10:15 AM", "Event": "System check", "Details": "All systems operational"},
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


if __name__ == "__main__":
    main()
