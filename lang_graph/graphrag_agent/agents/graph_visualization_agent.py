"""
Graph Visualization Agent

Knowledge Graph를 시각화하고 분석하는 전문 에이전트
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from pydantic import BaseModel, Field, validator
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo


class GraphVisualizationConfig(BaseModel):
    """Configuration for Graph Visualization Agent"""
    output_directory: str = Field(default="./graph_visualizations", description="Output directory for visualizations")
    chart_theme: str = Field(default="plotly_white", description="Chart theme for visualizations")
    default_width: int = Field(default=1200, description="Default chart width")
    default_height: int = Field(default=800, description="Default chart height")
    save_formats: List[str] = Field(default=["html", "png"], description="Formats to save charts")
    enable_interactive: bool = Field(default=True, description="Enable interactive charts")
    
    @validator('save_formats')
    def validate_save_formats(cls, v):
        allowed_formats = ["html", "png", "svg", "pdf", "jpg"]
        for fmt in v:
            if fmt not in allowed_formats:
                raise ValueError(f'Unsupported format: {fmt}. Allowed: {allowed_formats}')
        return v


class GraphVisualizationAgent:
    """Knowledge Graph 시각화 및 분석 전문 에이전트"""
    
    def __init__(self, config: GraphVisualizationConfig):
        self.config = config
        self._setup_logging()
        self._setup_output_directory()
        self.logger.info("GraphVisualizationAgent initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
    
    def _setup_output_directory(self):
        """Setup output directory for visualizations"""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory ready: {output_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
            raise
    
    async def create_comprehensive_visualization(self, knowledge_graph: Any, graph_name: str = "knowledge_graph") -> Dict[str, Any]:
        """
        Create comprehensive visualizations for the knowledge graph
        
        Args:
            knowledge_graph: The knowledge graph to visualize
            graph_name: Name for the visualization files
            
        Returns:
            Dict containing status and visualization paths
        """
        self.logger.info(f"Creating comprehensive visualizations for {graph_name}")
        
        try:
            # Convert to NetworkX for analysis
            nx_graph = self._convert_to_networkx(knowledge_graph)
            
            # Generate various visualizations
            viz_results = {}
            
            # 1. Network Graph Visualization
            network_viz = await self._create_network_visualization(nx_graph, graph_name)
            viz_results["network"] = network_viz
            
            # 2. Entity Type Distribution
            entity_dist_viz = await self._create_entity_distribution_chart(knowledge_graph, graph_name)
            viz_results["entity_distribution"] = entity_dist_viz
            
            # 3. Relationship Type Analysis
            relationship_viz = await self._create_relationship_analysis(knowledge_graph, graph_name)
            viz_results["relationship_analysis"] = relationship_viz
            
            # 4. Graph Metrics Dashboard
            metrics_viz = await self._create_metrics_dashboard(nx_graph, graph_name)
            viz_results["metrics_dashboard"] = metrics_viz
            
            # 5. Interactive Graph Explorer
            if self.config.enable_interactive:
                interactive_viz = await self._create_interactive_explorer(nx_graph, graph_name)
                viz_results["interactive_explorer"] = interactive_viz
            
            # 6. Graph Statistics Report
            stats_report = await self._generate_statistics_report(nx_graph, knowledge_graph, graph_name)
            viz_results["statistics_report"] = stats_report
            
            return {
                "status": "completed",
                "visualizations": viz_results,
                "graph_name": graph_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Visualization creation failed: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg}
    
    def _convert_to_networkx(self, knowledge_graph: Any) -> nx.Graph:
        """Convert knowledge graph to NetworkX format for analysis"""
        try:
            G = nx.Graph()
            
            # Add nodes
            for node in knowledge_graph.nodes:
                node_id = getattr(node, 'id', str(node))
                node_type = getattr(node, 'type', 'unknown')
                node_title = getattr(node, 'title', getattr(node, 'name', str(node)))
                
                G.add_node(node_id, 
                          type=node_type, 
                          title=node_title,
                          description=getattr(node, 'description', ''),
                          properties=getattr(node, 'properties', {}))
            
            # Add edges
            for edge in knowledge_graph.edges:
                source_id = getattr(edge.source, 'id', str(edge.source))
                target_id = getattr(edge.target, 'id', str(edge.target))
                edge_type = getattr(edge, 'type', 'unknown')
                edge_description = getattr(edge, 'description', '')
                
                G.add_edge(source_id, target_id, 
                          type=edge_type, 
                          description=edge_description)
            
            self.logger.info(f"Converted to NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            self.logger.error(f"Failed to convert to NetworkX: {e}")
            raise
    
    async def _create_network_visualization(self, nx_graph: nx.Graph, graph_name: str) -> Dict[str, Any]:
        """Create interactive network graph visualization"""
        try:
            # Calculate layout
            pos = nx.spring_layout(nx_graph, k=1, iterations=50)
            
            # Create Plotly network graph
            edge_trace = go.Scatter(
                x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
            
            for edge in nx_graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += tuple([x0, x1, None])
                edge_trace['y'] += tuple([y0, y1, None])
            
            # Create node traces by type
            node_traces = {}
            colors = px.colors.qualitative.Set3
            
            for node_type in set(nx.get_node_attributes(nx_graph, 'type').values()):
                node_list = [node for node in nx_graph.nodes() if nx_graph.nodes[node]['type'] == node_type]
                node_traces[node_type] = go.Scatter(
                    x=[pos[node][0] for node in node_list],
                    y=[pos[node][1] for node in node_list],
                    mode='markers+text',
                    hoverinfo='text',
                    text=[nx_graph.nodes[node]['title'] for node in node_list],
                    textposition="top center",
                    marker=dict(
                        size=20,
                        color=colors[len(node_traces) % len(colors)],
                        line=dict(width=2, color='white')
                    ),
                    name=node_type
                )
            
            # Create layout
            layout = go.Layout(
                title=f'{graph_name} - Network Visualization',
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=self.config.default_width,
                height=self.config.default_height,
                template=self.config.chart_theme
            )
            
            # Create figure
            fig = go.Figure(data=[edge_trace] + list(node_traces.values()), layout=layout)
            
            # Save visualization
            output_paths = await self._save_visualization(fig, f"{graph_name}_network", "network")
            
            return {
                "type": "network_visualization",
                "paths": output_paths,
                "node_count": nx_graph.number_of_nodes(),
                "edge_count": nx_graph.number_of_edges()
            }
            
        except Exception as e:
            self.logger.error(f"Network visualization failed: {e}")
            raise
    
    async def _create_entity_distribution_chart(self, knowledge_graph: Any, graph_name: str) -> Dict[str, Any]:
        """Create entity type distribution chart"""
        try:
            # Count entity types
            entity_counts = {}
            for node in knowledge_graph.nodes:
                node_type = getattr(node, 'type', 'unknown')
                entity_counts[node_type] = entity_counts.get(node_type, 0) + 1
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(entity_counts.keys()),
                values=list(entity_counts.values()),
                hole=0.3,
                textinfo='label+percent+value'
            )])
            
            fig.update_layout(
                title=f'{graph_name} - Entity Type Distribution',
                width=self.config.default_width,
                height=self.config.default_height,
                template=self.config.chart_theme
            )
            
            # Save visualization
            output_paths = await self._save_visualization(fig, f"{graph_name}_entity_distribution", "entity_distribution")
            
            return {
                "type": "entity_distribution",
                "paths": output_paths,
                "entity_types": len(entity_counts),
                "total_entities": sum(entity_counts.values())
            }
            
        except Exception as e:
            self.logger.error(f"Entity distribution chart failed: {e}")
            raise
    
    async def _create_relationship_analysis(self, knowledge_graph: Any, graph_name: str) -> Dict[str, Any]:
        """Create relationship type analysis chart"""
        try:
            # Count relationship types
            relationship_counts = {}
            for edge in knowledge_graph.edges:
                edge_type = getattr(edge, 'type', 'unknown')
                relationship_counts[edge_type] = relationship_counts.get(edge_type, 0) + 1
            
            # Create bar chart
            fig = go.Figure(data=[go.Bar(
                x=list(relationship_counts.keys()),
                y=list(relationship_counts.values()),
                marker_color='lightblue'
            )])
            
            fig.update_layout(
                title=f'{graph_name} - Relationship Type Analysis',
                xaxis_title="Relationship Type",
                yaxis_title="Count",
                width=self.config.default_width,
                height=self.config.default_height,
                template=self.config.chart_theme
            )
            
            # Save visualization
            output_paths = await self._save_visualization(fig, f"{graph_name}_relationship_analysis", "relationship_analysis")
            
            return {
                "type": "relationship_analysis",
                "paths": output_paths,
                "relationship_types": len(relationship_counts),
                "total_relationships": sum(relationship_counts.values())
            }
            
        except Exception as e:
            self.logger.error(f"Relationship analysis failed: {e}")
            raise
    
    async def _create_metrics_dashboard(self, nx_graph: nx.Graph, graph_name: str) -> Dict[str, Any]:
        """Create comprehensive metrics dashboard"""
        try:
            # Calculate graph metrics
            metrics = self._calculate_graph_metrics(nx_graph)
            
            # Create subplots for different metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Graph Density', 'Node Degree Distribution', 'Connected Components', 'Centrality Metrics'),
                specs=[[{"type": "indicator"}, {"type": "histogram"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # 1. Graph Density Indicator
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=metrics['density'],
                    title={'text': "Graph Density"},
                    gauge={'axis': {'range': [None, 1]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 0.3], 'color': "lightgray"},
                                   {'range': [0.3, 0.7], 'color': "yellow"},
                                   {'range': [0.7, 1], 'color': "green"}]},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=1, col=1
            )
            
            # 2. Node Degree Distribution
            degrees = [d for n, d in nx_graph.degree()]
            fig.add_trace(
                go.Histogram(x=degrees, nbinsx=20, name="Degree Distribution"),
                row=1, col=2
            )
            
            # 3. Connected Components
            component_sizes = [len(c) for c in sorted(nx.connected_components(nx_graph), key=len, reverse=True)]
            fig.add_trace(
                go.Bar(x=list(range(1, len(component_sizes) + 1)), y=component_sizes, name="Component Sizes"),
                row=2, col=1
            )
            
            # 4. Centrality Metrics
            centrality = nx.degree_centrality(nx_graph)
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            fig.add_trace(
                go.Scatter(x=[node for node, _ in top_nodes], y=[cent for _, cent in top_nodes], 
                          mode='markers+text', text=[node for node, _ in top_nodes], name="Top Centrality"),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'{graph_name} - Metrics Dashboard',
                width=self.config.default_width,
                height=self.config.default_height,
                template=self.config.chart_theme,
                showlegend=True
            )
            
            # Save visualization
            output_paths = await self._save_visualization(fig, f"{graph_name}_metrics_dashboard", "metrics_dashboard")
            
            return {
                "type": "metrics_dashboard",
                "paths": output_paths,
                "metrics": metrics
            }
            
        except Exception as e:
            self.logger.error(f"Metrics dashboard failed: {e}")
            raise
    
    def _calculate_graph_metrics(self, nx_graph: nx.Graph) -> Dict[str, Any]:
        """Calculate comprehensive graph metrics"""
        try:
            metrics = {
                'nodes': nx_graph.number_of_nodes(),
                'edges': nx_graph.number_of_edges(),
                'density': nx.density(nx_graph),
                'connected_components': nx.number_connected_components(nx_graph),
                'average_clustering': nx.average_clustering(nx_graph),
                'average_shortest_path': nx.average_shortest_path_length(nx_graph) if nx.is_connected(nx_graph) else None,
                'diameter': nx.diameter(nx_graph) if nx.is_connected(nx_graph) else None,
                'average_degree': sum(dict(nx_graph.degree()).values()) / nx_graph.number_of_nodes(),
                'max_degree': max(dict(nx_graph.degree()).values()),
                'min_degree': min(dict(nx_graph.degree()).values())
            }
            
            # Calculate centrality measures
            centrality_measures = {
                'degree_centrality': nx.degree_centrality(nx_graph),
                'betweenness_centrality': nx.betweenness_centrality(nx_graph),
                'closeness_centrality': nx.closeness_centrality(nx_graph),
                'eigenvector_centrality': nx.eigenvector_centrality(nx_graph, max_iter=1000)
            }
            
            metrics['centrality_measures'] = centrality_measures
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
            return {}
    
    async def _create_interactive_explorer(self, nx_graph: nx.Graph, graph_name: str) -> Dict[str, Any]:
        """Create interactive graph explorer"""
        try:
            # Create interactive network graph with Plotly
            pos = nx.spring_layout(nx_graph, k=1, iterations=50)
            
            # Node trace
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            
            for node in nx_graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"Node: {node}<br>Type: {nx_graph.nodes[node]['type']}<br>Title: {nx_graph.nodes[node]['title']}")
                node_colors.append(hash(nx_graph.nodes[node]['type']) % 20)
            
            # Edge trace
            edge_x = []
            edge_y = []
            edge_text = []
            
            for edge in nx_graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_text.extend([f"Edge: {edge[0]} -> {edge[1]}<br>Type: {nx_graph.edges[edge]['type']}", "", ""])
            
            # Create traces
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='text',
                text=edge_text,
                mode='lines',
                name='Edges'
            )
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[nx_graph.nodes[node]['title'] for node in nx_graph.nodes()],
                textposition="top center",
                marker=dict(
                    size=20,
                    color=node_colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Node Type"),
                    line=dict(width=2, color='white')
                ),
                name='Nodes'
            )
            
            # Layout
            layout = go.Layout(
                title=f'{graph_name} - Interactive Explorer',
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=self.config.default_width,
                height=self.config.default_height,
                template=self.config.chart_theme
            )
            
            fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
            
            # Save visualization
            output_paths = await self._save_visualization(fig, f"{graph_name}_interactive_explorer", "interactive_explorer")
            
            return {
                "type": "interactive_explorer",
                "paths": output_paths,
                "interactive_features": ["hover", "zoom", "pan", "select"]
            }
            
        except Exception as e:
            self.logger.error(f"Interactive explorer failed: {e}")
            raise
    
    async def _generate_statistics_report(self, nx_graph: nx.Graph, knowledge_graph: Any, graph_name: str) -> Dict[str, Any]:
        """Generate comprehensive statistics report"""
        try:
            # Calculate all metrics
            metrics = self._calculate_graph_metrics(nx_graph)
            
            # Entity analysis
            entity_analysis = {}
            for node in knowledge_graph.nodes:
                node_type = getattr(node, 'type', 'unknown')
                if node_type not in entity_analysis:
                    entity_analysis[node_type] = []
                entity_analysis[node_type].append({
                    'id': getattr(node, 'id', str(node)),
                    'title': getattr(node, 'title', getattr(node, 'name', str(node))),
                    'description': getattr(node, 'description', '')
                })
            
            # Relationship analysis
            relationship_analysis = {}
            for edge in knowledge_graph.edges:
                edge_type = getattr(edge, 'type', 'unknown')
                if edge_type not in relationship_analysis:
                    relationship_analysis[edge_type] = []
                relationship_analysis[edge_type].append({
                    'source': getattr(edge.source, 'id', str(edge.source)),
                    'target': getattr(edge.target, 'id', str(edge.target)),
                    'description': getattr(edge, 'description', '')
                })
            
            # Create report
            report = {
                'graph_name': graph_name,
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_nodes': metrics.get('nodes', 0),
                    'total_edges': metrics.get('edges', 0),
                    'entity_types': len(entity_analysis),
                    'relationship_types': len(relationship_analysis),
                    'graph_density': metrics.get('density', 0),
                    'connected_components': metrics.get('connected_components', 0)
                },
                'entity_analysis': entity_analysis,
                'relationship_analysis': relationship_analysis,
                'graph_metrics': metrics,
                'recommendations': self._generate_recommendations(metrics, entity_analysis, relationship_analysis)
            }
            
            # Save report as JSON
            report_path = Path(self.config.output_directory) / f"{graph_name}_statistics_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            return {
                "type": "statistics_report",
                "path": str(report_path),
                "report_summary": report['summary']
            }
            
        except Exception as e:
            self.logger.error(f"Statistics report failed: {e}")
            raise
    
    def _generate_recommendations(self, metrics: Dict, entity_analysis: Dict, relationship_analysis: Dict) -> List[str]:
        """Generate recommendations based on graph analysis"""
        recommendations = []
        
        # Density recommendations
        density = metrics.get('density', 0)
        if density < 0.1:
            recommendations.append("Graph density is very low. Consider adding more relationships between entities.")
        elif density > 0.8:
            recommendations.append("Graph density is very high. Consider if all relationships are necessary.")
        
        # Entity type recommendations
        if len(entity_analysis) < 3:
            recommendations.append("Consider diversifying entity types for richer knowledge representation.")
        
        # Relationship type recommendations
        if len(relationship_analysis) < 2:
            recommendations.append("Consider adding more relationship types for better entity connections.")
        
        # Connectivity recommendations
        if metrics.get('connected_components', 1) > 1:
            recommendations.append("Graph has multiple disconnected components. Consider adding bridging relationships.")
        
        # Centrality recommendations
        if metrics.get('max_degree', 0) > metrics.get('average_degree', 0) * 3:
            recommendations.append("Some nodes have very high degree. Consider if this represents information overload.")
        
        return recommendations
    
    async def _save_visualization(self, fig: go.Figure, filename: str, viz_type: str) -> Dict[str, str]:
        """Save visualization in multiple formats"""
        output_paths = {}
        
        for fmt in self.config.save_formats:
            try:
                if fmt == "html":
                    output_path = Path(self.config.output_directory) / f"{filename}.html"
                    fig.write_html(str(output_path))
                    output_paths["html"] = str(output_path)
                    
                elif fmt == "png":
                    output_path = Path(self.config.output_directory) / f"{filename}.png"
                    fig.write_image(str(output_path), width=self.config.default_width, height=self.config.default_height)
                    output_paths["png"] = str(output_path)
                    
                elif fmt == "svg":
                    output_path = Path(self.config.output_directory) / f"{filename}.svg"
                    fig.write_image(str(output_path), format="svg")
                    output_paths["svg"] = str(output_path)
                    
                elif fmt == "pdf":
                    output_path = Path(self.config.output_directory) / f"{filename}.pdf"
                    fig.write_image(str(output_path), format="pdf")
                    output_paths["pdf"] = str(output_path)
                    
            except Exception as e:
                self.logger.warning(f"Failed to save {fmt} format: {e}")
        
        return output_paths
    
    async def export_graph_data(self, knowledge_graph: Any, graph_name: str, export_format: str = "json") -> Dict[str, Any]:
        """Export graph data in various formats"""
        try:
            if export_format == "json":
                return await self._export_as_json(knowledge_graph, graph_name)
            elif export_format == "csv":
                return await self._export_as_csv(knowledge_graph, graph_name)
            elif export_format == "graphml":
                return await self._export_as_graphml(knowledge_graph, graph_name)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
                
        except Exception as e:
            error_msg = f"Export failed: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg}
    
    async def _export_as_json(self, knowledge_graph: Any, graph_name: str) -> Dict[str, Any]:
        """Export graph as JSON"""
        try:
            export_data = {
                'metadata': {
                    'name': graph_name,
                    'export_timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                },
                'nodes': [],
                'edges': []
            }
            
            # Export nodes
            for node in knowledge_graph.nodes:
                node_data = {
                    'id': getattr(node, 'id', str(node)),
                    'type': getattr(node, 'type', 'unknown'),
                    'title': getattr(node, 'title', getattr(node, 'name', str(node))),
                    'description': getattr(node, 'description', ''),
                    'properties': getattr(node, 'properties', {})
                }
                export_data['nodes'].append(node_data)
            
            # Export edges
            for edge in knowledge_graph.edges:
                edge_data = {
                    'id': getattr(edge, 'id', f"{edge.source}-{edge.target}"),
                    'source': getattr(edge.source, 'id', str(edge.source)),
                    'target': getattr(edge.target, 'id', str(edge.target)),
                    'type': getattr(edge, 'type', 'unknown'),
                    'description': getattr(edge, 'description', '')
                }
                export_data['edges'].append(edge_data)
            
            # Save JSON file
            output_path = Path(self.config.output_directory) / f"{graph_name}_export.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            return {
                "status": "completed",
                "format": "json",
                "path": str(output_path),
                "node_count": len(export_data['nodes']),
                "edge_count": len(export_data['edges'])
            }
            
        except Exception as e:
            self.logger.error(f"JSON export failed: {e}")
            raise
    
    async def _export_as_csv(self, knowledge_graph: Any, graph_name: str) -> Dict[str, Any]:
        """Export graph as CSV files"""
        try:
            # Export nodes
            nodes_data = []
            for node in knowledge_graph.nodes:
                node_data = {
                    'id': getattr(node, 'id', str(node)),
                    'type': getattr(node, 'type', 'unknown'),
                    'title': getattr(node, 'title', getattr(node, 'name', str(node))),
                    'description': getattr(node, 'description', ''),
                    'properties': json.dumps(getattr(node, 'properties', {}), ensure_ascii=False)
                }
                nodes_data.append(node_data)
            
            nodes_df = pd.DataFrame(nodes_data)
            nodes_path = Path(self.config.output_directory) / f"{graph_name}_nodes.csv"
            nodes_df.to_csv(nodes_path, index=False, encoding='utf-8')
            
            # Export edges
            edges_data = []
            for edge in knowledge_graph.edges:
                edge_data = {
                    'id': getattr(edge, 'id', f"{edge.source}-{edge.target}"),
                    'source': getattr(edge.source, 'id', str(edge.source)),
                    'target': getattr(edge.target, 'id', str(edge.target)),
                    'type': getattr(edge, 'type', 'unknown'),
                    'description': getattr(edge, 'description', '')
                }
                edges_data.append(edge_data)
            
            edges_df = pd.DataFrame(edges_data)
            edges_path = Path(self.config.output_directory) / f"{graph_name}_edges.csv"
            edges_df.to_csv(edges_path, index=False, encoding='utf-8')
            
            return {
                "status": "completed",
                "format": "csv",
                "paths": {
                    "nodes": str(nodes_path),
                    "edges": str(edges_path)
                },
                "node_count": len(nodes_data),
                "edge_count": len(edges_data)
            }
            
        except Exception as e:
            self.logger.error(f"CSV export failed: {e}")
            raise
    
    async def _export_as_graphml(self, knowledge_graph: Any, graph_name: str) -> Dict[str, Any]:
        """Export graph as GraphML format"""
        try:
            # Convert to NetworkX and export as GraphML
            nx_graph = self._convert_to_networkx(knowledge_graph)
            
            output_path = Path(self.config.output_directory) / f"{graph_name}_export.graphml"
            nx.write_graphml(nx_graph, output_path)
            
            return {
                "status": "completed",
                "format": "graphml",
                "path": str(output_path),
                "node_count": nx_graph.number_of_nodes(),
                "edge_count": nx_graph.number_of_edges()
            }
            
        except Exception as e:
            self.logger.error(f"GraphML export failed: {e}")
            raise
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent configuration"""
        return {
            "output_directory": self.config.output_directory,
            "chart_theme": self.config.chart_theme,
            "save_formats": self.config.save_formats,
            "enable_interactive": self.config.enable_interactive,
            "status": "ready"
        }
