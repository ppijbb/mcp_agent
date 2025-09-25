#!/usr/bin/env python3
"""
Data Visualization Module for Local Researcher

This module provides comprehensive data visualization capabilities including
interactive charts, research analytics, and real-time monitoring dashboards.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

logger = setup_logger("data_visualizer", log_level="INFO")


class DataVisualizer:
    """Advanced data visualization for research analytics."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the data visualizer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        
        # Visualization settings
        self.chart_theme = self.config_manager.get('visualization.theme', 'plotly_white')
        self.color_palette = self.config_manager.get('visualization.color_palette', 'viridis')
        self.default_width = self.config_manager.get('visualization.default_width', 800)
        self.default_height = self.config_manager.get('visualization.default_height', 600)
        
        logger.info("Data Visualizer initialized")
    
    def create_research_timeline(self, research_data: List[Dict[str, Any]]) -> go.Figure:
        """Create a timeline visualization of research activities.
        
        Args:
            research_data: List of research activities with timestamps
            
        Returns:
            Plotly figure object
        """
        try:
            if not research_data:
                return self._create_empty_chart("No research data available")
            
            # Prepare data
            df = pd.DataFrame(research_data)
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])
            df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 3600  # hours
            
            # Create Gantt chart
            fig = px.timeline(
                df, 
                x_start="start_time", 
                x_end="end_time", 
                y="objective_id",
                color="status",
                title="Research Activity Timeline",
                hover_data=["duration", "agent_count", "quality_score"]
            )
            
            fig.update_layout(
                height=self.default_height,
                showlegend=True,
                xaxis_title="Time",
                yaxis_title="Research Objectives"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create research timeline: {e}")
            return self._create_error_chart(f"Timeline creation failed: {e}")
    
    def create_agent_performance_chart(self, performance_data: Dict[str, Any]) -> go.Figure:
        """Create agent performance visualization.
        
        Args:
            performance_data: Agent performance metrics
            
        Returns:
            Plotly figure object
        """
        try:
            agents = list(performance_data.keys())
            tasks_completed = [data.get('tasks_completed', 0) for data in performance_data.values()]
            success_rates = [data.get('success_rate', 0) for data in performance_data.values()]
            avg_quality = [data.get('avg_quality', 0) for data in performance_data.values()]
            
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Tasks Completed", "Success Rate & Quality"),
                specs=[[{"secondary_y": False}, {"secondary_y": True}]]
            )
            
            # Tasks completed bar chart
            fig.add_trace(
                go.Bar(
                    x=agents,
                    y=tasks_completed,
                    name="Tasks Completed",
                    marker_color=px.colors.qualitative.Set3[0]
                ),
                row=1, col=1
            )
            
            # Success rate line chart
            fig.add_trace(
                go.Scatter(
                    x=agents,
                    y=success_rates,
                    mode='lines+markers',
                    name="Success Rate",
                    line=dict(color=px.colors.qualitative.Set3[1], width=3)
                ),
                row=1, col=2
            )
            
            # Quality score line chart (secondary y-axis)
            fig.add_trace(
                go.Scatter(
                    x=agents,
                    y=avg_quality,
                    mode='lines+markers',
                    name="Avg Quality",
                    line=dict(color=px.colors.qualitative.Set3[2], width=3),
                    yaxis="y2"
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Agent Performance Analysis",
                height=self.default_height,
                showlegend=True
            )
            
            # Update axes
            fig.update_xaxes(title_text="Agents", row=1, col=1)
            fig.update_xaxes(title_text="Agents", row=1, col=2)
            fig.update_yaxes(title_text="Tasks Completed", row=1, col=1)
            fig.update_yaxes(title_text="Success Rate", row=1, col=2)
            fig.update_yaxes(title_text="Quality Score", secondary_y=True, row=1, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create agent performance chart: {e}")
            return self._create_error_chart(f"Performance chart creation failed: {e}")
    
    def create_research_quality_distribution(self, quality_data: List[float]) -> go.Figure:
        """Create research quality distribution visualization.
        
        Args:
            quality_data: List of quality scores
            
        Returns:
            Plotly figure object
        """
        try:
            if not quality_data:
                return self._create_empty_chart("No quality data available")
            
            # Create histogram
            fig = px.histogram(
                x=quality_data,
                nbins=20,
                title="Research Quality Score Distribution",
                labels={'x': 'Quality Score', 'y': 'Frequency'},
                color_discrete_sequence=[px.colors.qualitative.Set3[3]]
            )
            
            # Add mean line
            mean_score = sum(quality_data) / len(quality_data)
            fig.add_vline(
                x=mean_score,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_score:.2f}"
            )
            
            fig.update_layout(
                height=self.default_height,
                xaxis_title="Quality Score (0-1)",
                yaxis_title="Frequency"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create quality distribution: {e}")
            return self._create_error_chart(f"Quality distribution creation failed: {e}")
    
    def create_research_trends(self, trend_data: List[Dict[str, Any]]) -> go.Figure:
        """Create research trends over time.
        
        Args:
            trend_data: Time series data of research metrics
            
        Returns:
            Plotly figure object
        """
        try:
            if not trend_data:
                return self._create_empty_chart("No trend data available")
            
            df = pd.DataFrame(trend_data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Research Volume", "Quality Trends", "Agent Utilization", "Success Rate"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                     [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Research volume
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['research_count'],
                    mode='lines+markers',
                    name="Research Count",
                    line=dict(color=px.colors.qualitative.Set3[0])
                ),
                row=1, col=1
            )
            
            # Quality trends
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['avg_quality'],
                    mode='lines+markers',
                    name="Avg Quality",
                    line=dict(color=px.colors.qualitative.Set3[1])
                ),
                row=1, col=2
            )
            
            # Agent utilization
            fig.add_trace(
                go.Bar(
                    x=df['date'],
                    y=df['agent_utilization'],
                    name="Agent Utilization",
                    marker_color=px.colors.qualitative.Set3[2]
                ),
                row=2, col=1
            )
            
            # Success rate
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['success_rate'],
                    mode='lines+markers',
                    name="Success Rate",
                    line=dict(color=px.colors.qualitative.Set3[3])
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Research Trends Analysis",
                height=self.default_height * 1.5,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create research trends: {e}")
            return self._create_error_chart(f"Trends creation failed: {e}")
    
    def create_domain_analysis(self, domain_data: Dict[str, Any]) -> go.Figure:
        """Create research domain analysis visualization.
        
        Args:
            domain_data: Research domain statistics
            
        Returns:
            Plotly figure object
        """
        try:
            domains = list(domain_data.keys())
            research_counts = [data.get('research_count', 0) for data in domain_data.values()]
            avg_quality = [data.get('avg_quality', 0) for data in domain_data.values()]
            success_rates = [data.get('success_rate', 0) for data in domain_data.values()]
            
            # Create bubble chart
            fig = px.scatter(
                x=research_counts,
                y=avg_quality,
                size=success_rates,
                color=domains,
                title="Research Domain Analysis",
                labels={'x': 'Research Count', 'y': 'Average Quality'},
                hover_name=domains,
                hover_data={'size': 'Success Rate'}
            )
            
            fig.update_layout(
                height=self.default_height,
                xaxis_title="Number of Research Projects",
                yaxis_title="Average Quality Score"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create domain analysis: {e}")
            return self._create_error_chart(f"Domain analysis creation failed: {e}")
    
    def create_system_health_dashboard(self, system_data: Dict[str, Any]) -> go.Figure:
        """Create system health monitoring dashboard.
        
        Args:
            system_data: System health metrics
            
        Returns:
            Plotly figure object
        """
        try:
            # Create subplots for different metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("CPU Usage", "Memory Usage", "Active Research", "Error Rate"),
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                     [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # CPU usage gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=system_data.get('cpu_usage', 0),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "CPU Usage (%)"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "red"}]}
                ),
                row=1, col=1
            )
            
            # Memory usage gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=system_data.get('memory_usage', 0),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Memory Usage (%)"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkgreen"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "red"}]}
                ),
                row=1, col=2
            )
            
            # Active research bar chart
            research_status = system_data.get('research_status', {})
            statuses = list(research_status.keys())
            counts = list(research_status.values())
            
            fig.add_trace(
                go.Bar(
                    x=statuses,
                    y=counts,
                    name="Research Status",
                    marker_color=px.colors.qualitative.Set3[4]
                ),
                row=2, col=1
            )
            
            # Error rate bar chart
            error_data = system_data.get('error_rates', {})
            error_types = list(error_data.keys())
            error_counts = list(error_data.values())
            
            fig.add_trace(
                go.Bar(
                    x=error_types,
                    y=error_counts,
                    name="Error Types",
                    marker_color=px.colors.qualitative.Set3[5]
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="System Health Dashboard",
                height=self.default_height * 1.5,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create system health dashboard: {e}")
            return self._create_error_chart(f"System health dashboard creation failed: {e}")
    
    def create_interactive_research_map(self, research_locations: List[Dict[str, Any]]) -> go.Figure:
        """Create interactive map of research activities.
        
        Args:
            research_locations: Research activities with geographic data
            
        Returns:
            Plotly figure object
        """
        try:
            if not research_locations:
                return self._create_empty_chart("No location data available")
            
            df = pd.DataFrame(research_locations)
            
            # Create scatter mapbox
            fig = px.scatter_mapbox(
                df,
                lat="latitude",
                lon="longitude",
                color="research_type",
                size="research_count",
                hover_name="location",
                hover_data=["quality_score", "success_rate"],
                title="Research Activities Map",
                mapbox_style="open-street-map",
                zoom=2
            )
            
            fig.update_layout(
                height=self.default_height,
                margin={"r": 0, "t": 0, "l": 0, "b": 0}
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create research map: {e}")
            return self._create_error_chart(f"Research map creation failed: {e}")
    
    def export_chart(self, fig: go.Figure, filename: str, format: str = "html") -> str:
        """Export chart to file.
        
        Args:
            fig: Plotly figure object
            filename: Output filename
            format: Export format (html, png, svg, pdf)
            
        Returns:
            Path to exported file
        """
        try:
            output_dir = Path(self.config_manager.get('output.directory', './outputs'))
            output_dir.mkdir(exist_ok=True)
            
            file_path = output_dir / f"{filename}.{format}"
            
            if format == "html":
                fig.write_html(str(file_path))
            elif format == "png":
                fig.write_image(str(file_path))
            elif format == "svg":
                fig.write_image(str(file_path))
            elif format == "pdf":
                fig.write_image(str(file_path))
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Chart exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to export chart: {e}")
            raise
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            plot_bgcolor="white"
        )
        return fig
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create an error chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            plot_bgcolor="white"
        )
        return fig
