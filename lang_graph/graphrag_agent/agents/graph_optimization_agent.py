"""
Graph Optimization Agent

Knowledge Graph의 품질을 최적화하고 검증하는 전문 에이전트
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import networkx as nx
from pathlib import Path
import json
import logging
from pydantic import BaseModel, Field, validator
import numpy as np
from datetime import datetime
import re
from collections import defaultdict, Counter


class GraphOptimizationConfig(BaseModel):
    """Configuration for Graph Optimization Agent"""
    enable_color_optimization: bool = Field(default=True, description="Enable color scheme optimization")
    enable_layout_optimization: bool = Field(default=True, description="Enable layout optimization")
    enable_typography_optimization: bool = Field(default=True, description="Enable typography optimization")
    quality_threshold: float = Field(default=0.8, description="Minimum quality score threshold")
    max_colors: int = Field(default=12, description="Maximum number of colors in palette")
    min_node_distance: float = Field(default=0.1, description="Minimum distance between nodes")
    
    @validator('quality_threshold')
    def validate_quality_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('quality_threshold must be between 0.0 and 1.0')
        return v
    
    @validator('max_colors')
    def validate_max_colors(cls, v):
        if v < 3 or v > 20:
            raise ValueError('max_colors must be between 3 and 20')
        return v


class GraphOptimizationAgent:
    """Knowledge Graph 품질 최적화 및 검증 전문 에이전트"""
    
    def __init__(self, config: GraphOptimizationConfig):
        self.config = config
        self._setup_logging()
        self._initialize_optimization_tools()
        self.logger.info("GraphOptimizationAgent initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
    
    def _initialize_optimization_tools(self):
        """Initialize optimization tools and color palettes"""
        # Professional color palettes for different entity types
        self.entity_color_palettes = {
            'person': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
            'organization': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261'],
            'location': ['#6B4423', '#8B4513', '#A0522D', '#CD853F'],
            'concept': ['#4B0082', '#8A2BE2', '#9370DB', '#BA55D3'],
            'event': ['#DC143C', '#B22222', '#8B0000', '#FF4500'],
            'product': ['#228B22', '#32CD32', '#00FF00', '#90EE90'],
            'technology': ['#4169E1', '#1E90FF', '#00BFFF', '#87CEEB'],
            'default': ['#2F4F4F', '#708090', '#778899', '#B0C4DE']
        }
        
        # Typography optimization settings
        self.typography_settings = {
            'font_family': ['Arial', 'Helvetica', 'sans-serif'],
            'font_size_range': (10, 24),
            'font_weight_range': (300, 700),
            'line_height_range': (1.2, 1.8)
        }
        
        # Layout optimization parameters
        self.layout_parameters = {
            'spring_k': 1.0,
            'spring_iterations': 50,
            'force_atlas_iterations': 100,
            'kamada_kawai_iterations': 200
        }
    
    async def optimize_graph_quality(self, knowledge_graph: Any, graph_name: str = "knowledge_graph") -> Dict[str, Any]:
        """
        Comprehensive graph quality optimization
        
        Args:
            knowledge_graph: The knowledge graph to optimize
            graph_name: Name for the optimization process
            
        Returns:
            Dict containing optimization results and quality metrics
        """
        self.logger.info(f"Starting quality optimization for {graph_name}")
        
        try:
            # Convert to NetworkX for optimization
            nx_graph = self._convert_to_networkx(knowledge_graph)
            
            optimization_results = {}
            quality_metrics = {}
            
            # 1. Graph Structure Optimization
            if self.config.enable_layout_optimization:
                structure_opt = await self._optimize_graph_structure(nx_graph)
                optimization_results["structure"] = structure_opt
                quality_metrics["structure_quality"] = structure_opt["quality_score"]
            
            # 2. Color Scheme Optimization
            if self.config.enable_color_optimization:
                color_opt = await self._optimize_color_scheme(nx_graph)
                optimization_results["color"] = color_opt
                quality_metrics["color_quality"] = color_opt["quality_score"]
            
            # 3. Typography Optimization
            if self.config.enable_typography_optimization:
                typography_opt = await self._optimize_typography(nx_graph)
                optimization_results["typography"] = typography_opt
                quality_metrics["typography_quality"] = typography_opt["quality_score"]
            
            # 4. Overall Quality Assessment
            overall_quality = self._calculate_overall_quality(quality_metrics)
            quality_metrics["overall_quality"] = overall_quality
            
            # 5. Generate Optimization Report
            optimization_report = await self._generate_optimization_report(
                optimization_results, quality_metrics, graph_name
            )
            
            return {
                "status": "completed",
                "optimization_results": optimization_results,
                "quality_metrics": quality_metrics,
                "overall_quality": overall_quality,
                "optimization_report": optimization_report,
                "graph_name": graph_name,
                "timestamp": datetime.now().isoformat(),
                "meets_threshold": overall_quality >= self.config.quality_threshold
            }
            
        except Exception as e:
            error_msg = f"Graph optimization failed: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg}
    
    def _convert_to_networkx(self, knowledge_graph: Any) -> nx.Graph:
        """Convert knowledge graph to NetworkX format for optimization"""
        try:
            G = nx.Graph()
            
            # Add nodes with enhanced attributes
            for node in knowledge_graph.nodes:
                node_id = getattr(node, 'id', str(node))
                node_type = getattr(node, 'type', 'unknown')
                node_title = getattr(node, 'title', getattr(node, 'name', str(node)))
                
                G.add_node(node_id, 
                          type=node_type, 
                          title=node_title,
                          description=getattr(node, 'description', ''),
                          properties=getattr(node, 'properties', {}),
                          original_node=node)
            
            # Add edges with enhanced attributes
            for edge in knowledge_graph.edges:
                source_id = getattr(edge.source, 'id', str(edge.source))
                target_id = getattr(edge.target, 'id', str(edge.target))
                edge_type = getattr(edge, 'type', 'unknown')
                edge_description = getattr(edge, 'description', '')
                
                G.add_edge(source_id, target_id, 
                          type=edge_type, 
                          description=edge_description,
                          original_edge=edge)
            
            self.logger.info(f"Converted to NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            self.logger.error(f"Failed to convert to NetworkX: {e}")
            raise
    
    async def _optimize_graph_structure(self, nx_graph: nx.Graph) -> Dict[str, Any]:
        """Optimize graph structure and layout"""
        try:
            original_metrics = self._calculate_structure_metrics(nx_graph)
            
            # Apply multiple layout algorithms and select the best
            layouts = {}
            layout_scores = {}
            
            # 1. Spring Layout
            try:
                spring_pos = nx.spring_layout(nx_graph, k=self.layout_parameters['spring_k'], 
                                           iterations=self.layout_parameters['spring_iterations'])
                layouts['spring'] = spring_pos
                layout_scores['spring'] = self._evaluate_layout_quality(nx_graph, spring_pos)
            except Exception as e:
                self.logger.warning(f"Spring layout failed: {e}")
            
            # 2. Force Atlas Layout
            try:
                force_pos = nx.spring_layout(nx_graph, k=2.0, iterations=self.layout_parameters['force_atlas_iterations'])
                layouts['force_atlas'] = force_pos
                layout_scores['force_atlas'] = self._evaluate_layout_quality(nx_graph, force_pos)
            except Exception as e:
                self.logger.warning(f"Force Atlas layout failed: {e}")
            
            # 3. Kamada-Kawai Layout
            try:
                kamada_pos = nx.kamada_kawai_layout(nx_graph, iterations=self.layout_parameters['kamada_kawai_iterations'])
                layouts['kamada_kawai'] = kamada_pos
                layout_scores['kamada_kawai'] = self._evaluate_layout_quality(nx_graph, kamada_pos)
            except Exception as e:
                self.logger.warning(f"Kamada-Kawai layout failed: {e}")
            
            # Select best layout
            if layout_scores:
                best_layout_name = max(layout_scores, key=layout_scores.get)
                best_layout = layouts[best_layout_name]
                best_score = layout_scores[best_layout_name]
            else:
                best_layout = nx.spring_layout(nx_graph)
                best_score = 0.5
                best_layout_name = "fallback_spring"
            
            # Apply layout to graph
            nx.set_node_attributes(nx_graph, best_layout, 'pos')
            
            # Calculate optimized metrics
            optimized_metrics = self._calculate_structure_metrics(nx_graph)
            
            # Calculate improvement
            improvement = self._calculate_improvement(original_metrics, optimized_metrics)
            
            return {
                "type": "structure_optimization",
                "best_layout": best_layout_name,
                "layout_score": best_score,
                "original_metrics": original_metrics,
                "optimized_metrics": optimized_metrics,
                "improvement": improvement,
                "quality_score": best_score,
                "recommendations": self._generate_structure_recommendations(optimized_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Structure optimization failed: {e}")
            raise
    
    def _calculate_structure_metrics(self, nx_graph: nx.Graph) -> Dict[str, Any]:
        """Calculate comprehensive structure metrics"""
        try:
            metrics = {
                'node_count': nx_graph.number_of_nodes(),
                'edge_count': nx_graph.number_of_edges(),
                'density': nx.density(nx_graph),
                'connected_components': nx.number_connected_components(nx_graph),
                'average_clustering': nx.average_clustering(nx_graph),
                'average_degree': sum(dict(nx_graph.degree()).values()) / nx_graph.number_of_nodes(),
                'max_degree': max(dict(nx_graph.degree()).values()),
                'min_degree': min(dict(nx_graph.degree()).values()),
                'degree_variance': np.var(list(dict(nx_graph.degree()).values())),
                'assortativity': nx.degree_assortativity_coefficient(nx_graph),
                'transitivity': nx.transitivity(nx_graph)
            }
            
            # Calculate path-based metrics if graph is connected
            if nx.is_connected(nx_graph):
                metrics.update({
                    'average_shortest_path': nx.average_shortest_path_length(nx_graph),
                    'diameter': nx.diameter(nx_graph),
                    'radius': nx.radius(nx_graph),
                    'center': len(nx.center(nx_graph)),
                    'periphery': len(nx.periphery(nx_graph))
                })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Structure metrics calculation failed: {e}")
            return {}
    
    def _evaluate_layout_quality(self, nx_graph: nx.Graph, pos: Dict) -> float:
        """Evaluate the quality of a layout"""
        try:
            if not pos:
                return 0.0
            
            # Calculate node overlap
            overlap_penalty = 0
            node_positions = list(pos.values())
            
            for i in range(len(node_positions)):
                for j in range(i + 1, len(node_positions)):
                    distance = np.linalg.norm(np.array(node_positions[i]) - np.array(node_positions[j]))
                    if distance < self.config.min_node_distance:
                        overlap_penalty += 1
            
            # Calculate edge length variance
            edge_lengths = []
            for edge in nx_graph.edges():
                if edge[0] in pos and edge[1] in pos:
                    length = np.linalg.norm(np.array(pos[edge[0]]) - np.array(pos[edge[1]]))
                    edge_lengths.append(length)
            
            if edge_lengths:
                edge_length_variance = np.var(edge_lengths)
            else:
                edge_length_variance = 0
            
            # Calculate overall score
            overlap_score = max(0, 1 - overlap_penalty / len(node_positions))
            edge_score = max(0, 1 - min(edge_length_variance, 1))
            
            # Weighted combination
            final_score = 0.6 * overlap_score + 0.4 * edge_score
            
            return max(0, min(1, final_score))
            
        except Exception as e:
            self.logger.error(f"Layout quality evaluation failed: {e}")
            return 0.5
    
    async def _optimize_color_scheme(self, nx_graph: nx.Graph) -> Dict[str, Any]:
        """Optimize color scheme for the graph"""
        try:
            # Analyze entity types and their distribution
            entity_types = nx.get_node_attributes(nx_graph, 'type')
            type_counts = Counter(entity_types.values())
            
            # Generate optimized color palette
            optimized_colors = self._generate_optimized_color_palette(type_counts)
            
            # Apply colors to nodes
            node_colors = {}
            for node, node_type in entity_types.items():
                if node_type in optimized_colors:
                    node_colors[node] = optimized_colors[node_type]
                else:
                    node_colors[node] = optimized_colors['default'][0]
            
            # Calculate color quality metrics
            color_quality = self._evaluate_color_quality(type_counts, optimized_colors)
            
            # Store colors in graph attributes
            nx.set_node_attributes(nx_graph, node_colors, 'color')
            
            return {
                "type": "color_optimization",
                "color_palette": optimized_colors,
                "node_colors": node_colors,
                "entity_type_distribution": dict(type_counts),
                "color_quality": color_quality,
                "quality_score": color_quality["overall_score"],
                "recommendations": self._generate_color_recommendations(color_quality, type_counts)
            }
            
        except Exception as e:
            self.logger.error(f"Color optimization failed: {e}")
            raise
    
    def _generate_optimized_color_palette(self, type_counts: Counter) -> Dict[str, str]:
        """Generate an optimized color palette based on entity type distribution"""
        try:
            optimized_colors = {}
            
            # Sort entity types by frequency
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Assign colors based on frequency and semantic meaning
            for i, (entity_type, count) in enumerate(sorted_types):
                if entity_type in self.entity_color_palettes:
                    # Use predefined palette for known types
                    palette = self.entity_color_palettes[entity_type]
                    color_index = min(i, len(palette) - 1)
                    optimized_colors[entity_type] = palette[color_index]
                else:
                    # Generate color for unknown types
                    if i < len(self.entity_color_palettes['default']):
                        optimized_colors[entity_type] = self.entity_color_palettes['default'][i]
                    else:
                        # Generate additional colors if needed
                        hue = (i * 137.508) % 360  # Golden angle approximation
                        saturation = 70 + (i % 20)
                        lightness = 45 + (i % 20)
                        optimized_colors[entity_type] = f"hsl({hue}, {saturation}%, {lightness}%)"
            
            return optimized_colors
            
        except Exception as e:
            self.logger.error(f"Color palette generation failed: {e}")
            return self.entity_color_palettes['default']
    
    def _evaluate_color_quality(self, type_counts: Counter, color_palette: Dict[str, str]) -> Dict[str, Any]:
        """Evaluate the quality of the color scheme"""
        try:
            quality_metrics = {
                'color_diversity': len(set(color_palette.values())),
                'type_coverage': len(color_palette) / len(type_counts) if type_counts else 0,
                'semantic_appropriateness': 0,
                'accessibility_score': 0,
                'overall_score': 0
            }
            
            # Evaluate semantic appropriateness
            semantic_score = 0
            for entity_type in color_palette:
                if entity_type in self.entity_color_palettes:
                    semantic_score += 1
            quality_metrics['semantic_appropriateness'] = semantic_score / len(color_palette) if color_palette else 0
            
            # Evaluate accessibility (contrast and distinctiveness)
            colors = list(color_palette.values())
            contrast_score = 0
            for i in range(len(colors)):
                for j in range(i + 1, len(colors)):
                    # Simple contrast calculation (can be enhanced with proper color theory)
                    contrast_score += 1 if colors[i] != colors[j] else 0
            
            if len(colors) > 1:
                quality_metrics['accessibility_score'] = contrast_score / (len(colors) * (len(colors) - 1) / 2)
            
            # Calculate overall score
            weights = {
                'color_diversity': 0.2,
                'type_coverage': 0.3,
                'semantic_appropriateness': 0.3,
                'accessibility_score': 0.2
            }
            
            quality_metrics['overall_score'] = sum(
                quality_metrics[key] * weights[key] for key in weights
            )
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Color quality evaluation failed: {e}")
            return {'overall_score': 0.5}
    
    async def _optimize_typography(self, nx_graph: nx.Graph) -> Dict[str, Any]:
        """Optimize typography for the graph"""
        try:
            # Analyze text content and length
            text_analysis = self._analyze_text_content(nx_graph)
            
            # Generate optimized typography settings
            optimized_typography = self._generate_optimized_typography(text_analysis)
            
            # Apply typography to graph attributes
            nx.set_node_attributes(nx_graph, optimized_typography['node_settings'], 'typography')
            nx.set_edge_attributes(nx_graph, optimized_typography['edge_settings'], 'typography')
            
            # Calculate typography quality
            typography_quality = self._evaluate_typography_quality(text_analysis, optimized_typography)
            
            return {
                "type": "typography_optimization",
                "text_analysis": text_analysis,
                "optimized_settings": optimized_typography,
                "typography_quality": typography_quality,
                "quality_score": typography_quality["overall_score"],
                "recommendations": self._generate_typography_recommendations(typography_quality, text_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Typography optimization failed: {e}")
            raise
    
    def _analyze_text_content(self, nx_graph: nx.Graph) -> Dict[str, Any]:
        """Analyze text content in the graph"""
        try:
            analysis = {
                'node_titles': [],
                'node_descriptions': [],
                'edge_descriptions': [],
                'text_lengths': [],
                'special_characters': 0,
                'language_patterns': {}
            }
            
            # Analyze node text
            for node in nx_graph.nodes():
                node_data = nx_graph.nodes[node]
                title = node_data.get('title', '')
                description = node_data.get('description', '')
                
                if title:
                    analysis['node_titles'].append(title)
                    analysis['text_lengths'].append(len(title))
                
                if description:
                    analysis['node_descriptions'].append(description)
                    analysis['text_lengths'].append(len(description))
            
            # Analyze edge text
            for edge in nx_graph.edges():
                edge_data = nx_graph.edges[edge]
                description = edge_data.get('description', '')
                
                if description:
                    analysis['edge_descriptions'].append(description)
                    analysis['text_lengths'].append(len(description))
            
            # Calculate statistics
            if analysis['text_lengths']:
                analysis['avg_text_length'] = np.mean(analysis['text_lengths'])
                analysis['max_text_length'] = max(analysis['text_lengths'])
                analysis['min_text_length'] = min(analysis['text_lengths'])
                analysis['text_length_variance'] = np.var(analysis['text_lengths'])
            else:
                analysis['avg_text_length'] = 0
                analysis['max_text_length'] = 0
                analysis['min_text_length'] = 0
                analysis['text_length_variance'] = 0
            
            # Analyze special characters
            all_text = ' '.join(analysis['node_titles'] + analysis['node_descriptions'] + analysis['edge_descriptions'])
            analysis['special_characters'] = len(re.findall(r'[^a-zA-Z0-9\s]', all_text))
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Text content analysis failed: {e}")
            return {}
    
    def _generate_optimized_typography(self, text_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized typography settings"""
        try:
            avg_length = text_analysis.get('avg_text_length', 10)
            max_length = text_analysis.get('max_text_length', 20)
            
            # Calculate optimal font sizes based on text length
            base_font_size = max(10, min(24, 16 - (avg_length - 15) * 0.5))
            title_font_size = min(24, base_font_size + 4)
            description_font_size = max(10, base_font_size - 2)
            
            # Generate node typography settings
            node_settings = {}
            for node in text_analysis.get('node_titles', []):
                node_settings[node] = {
                    'font_family': self.typography_settings['font_family'],
                    'font_size': title_font_size if len(node) <= avg_length else base_font_size,
                    'font_weight': 600 if len(node) <= avg_length else 400,
                    'line_height': 1.3,
                    'text_overflow': 'ellipsis' if len(node) > max_length else 'visible'
                }
            
            # Generate edge typography settings
            edge_settings = {}
            for edge_desc in text_analysis.get('edge_descriptions', []):
                edge_settings[edge_desc] = {
                    'font_family': self.typography_settings['font_family'],
                    'font_size': description_font_size,
                    'font_weight': 400,
                    'line_height': 1.2,
                    'text_overflow': 'ellipsis' if len(edge_desc) > avg_length else 'visible'
                }
            
            return {
                'node_settings': node_settings,
                'edge_settings': edge_settings,
                'global_settings': {
                    'base_font_size': base_font_size,
                    'title_font_size': title_font_size,
                    'description_font_size': description_font_size,
                    'font_family': self.typography_settings['font_family']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Typography generation failed: {e}")
            return {'node_settings': {}, 'edge_settings': {}, 'global_settings': {}}
    
    def _evaluate_typography_quality(self, text_analysis: Dict[str, Any], typography: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate typography quality"""
        try:
            quality_metrics = {
                'readability_score': 0,
                'consistency_score': 0,
                'accessibility_score': 0,
                'overall_score': 0
            }
            
            # Readability score based on font sizes and text lengths
            avg_length = text_analysis.get('avg_text_length', 10)
            base_font_size = typography.get('global_settings', {}).get('base_font_size', 16)
            
            if avg_length > 0 and base_font_size > 0:
                # Optimal font size for readability
                optimal_size = max(12, min(18, 16 - (avg_length - 15) * 0.3))
                size_diff = abs(base_font_size - optimal_size)
                quality_metrics['readability_score'] = max(0, 1 - size_diff / 10)
            
            # Consistency score
            font_sizes = []
            for node_settings in typography.get('node_settings', {}).values():
                font_sizes.append(node_settings.get('font_size', 16))
            
            if font_sizes:
                size_variance = np.var(font_sizes)
                quality_metrics['consistency_score'] = max(0, 1 - size_variance / 100)
            
            # Accessibility score
            quality_metrics['accessibility_score'] = 0.8  # Base score for standard fonts
            
            # Calculate overall score
            weights = {
                'readability_score': 0.4,
                'consistency_score': 0.3,
                'accessibility_score': 0.3
            }
            
            quality_metrics['overall_score'] = sum(
                quality_metrics[key] * weights[key] for key in weights
            )
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Typography quality evaluation failed: {e}")
            return {'overall_score': 0.5}
    
    def _calculate_overall_quality(self, quality_metrics: Dict[str, float]) -> float:
        """Calculate overall quality score"""
        try:
            if not quality_metrics:
                return 0.0
            
            # Weighted average of all quality metrics
            weights = {
                'structure_quality': 0.4,
                'color_quality': 0.3,
                'typography_quality': 0.3
            }
            
            total_score = 0
            total_weight = 0
            
            for metric, weight in weights.items():
                if metric in quality_metrics:
                    total_score += quality_metrics[metric] * weight
                    total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Overall quality calculation failed: {e}")
            return 0.0
    
    def _calculate_improvement(self, original: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, float]:
        """Calculate improvement between original and optimized metrics"""
        try:
            improvement = {}
            
            for key in original:
                if key in optimized and isinstance(original[key], (int, float)) and isinstance(optimized[key], (int, float)):
                    if original[key] != 0:
                        improvement[key] = (optimized[key] - original[key]) / original[key]
                    else:
                        improvement[key] = 0.0 if optimized[key] == 0 else 1.0
            
            return improvement
            
        except Exception as e:
            self.logger.error(f"Improvement calculation failed: {e}")
            return {}
    
    def _generate_structure_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate structure optimization recommendations"""
        recommendations = []
        
        density = metrics.get('density', 0)
        if density < 0.1:
            recommendations.append("Graph density is very low. Consider adding more relationships between entities.")
        elif density > 0.8:
            recommendations.append("Graph density is very high. Consider if all relationships are necessary.")
        
        connected_components = metrics.get('connected_components', 1)
        if connected_components > 1:
            recommendations.append(f"Graph has {connected_components} disconnected components. Consider adding bridging relationships.")
        
        clustering = metrics.get('average_clustering', 0)
        if clustering < 0.1:
            recommendations.append("Low clustering coefficient suggests sparse local structure. Consider adding local connections.")
        
        return recommendations
    
    def _generate_color_recommendations(self, quality: Dict[str, Any], type_counts: Counter) -> List[str]:
        """Generate color optimization recommendations"""
        recommendations = []
        
        if quality.get('color_diversity', 0) < len(type_counts) * 0.8:
            recommendations.append("Consider increasing color diversity for better entity type distinction.")
        
        if quality.get('semantic_appropriateness', 0) < 0.7:
            recommendations.append("Some entity types could benefit from more semantically appropriate colors.")
        
        if quality.get('accessibility_score', 0) < 0.6:
            recommendations.append("Consider improving color contrast for better accessibility.")
        
        return recommendations
    
    def _generate_typography_recommendations(self, quality: Dict[str, Any], text_analysis: Dict[str, Any]) -> List[str]:
        """Generate typography optimization recommendations"""
        recommendations = []
        
        if quality.get('readability_score', 0) < 0.7:
            recommendations.append("Consider adjusting font sizes for better readability.")
        
        if quality.get('consistency_score', 0) < 0.6:
            recommendations.append("Font size consistency could be improved for better visual hierarchy.")
        
        avg_length = text_analysis.get('avg_text_length', 0)
        if avg_length > 50:
            recommendations.append("Long text descriptions may benefit from improved typography settings.")
        
        return recommendations
    
    async def _generate_optimization_report(self, optimization_results: Dict[str, Any], 
                                         quality_metrics: Dict[str, float], 
                                         graph_name: str) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        try:
            report = {
                'graph_name': graph_name,
                'timestamp': datetime.now().isoformat(),
                'optimization_summary': {
                    'structure_optimized': 'structure' in optimization_results,
                    'color_optimized': 'color' in optimization_results,
                    'typography_optimized': 'typography' in optimization_results
                },
                'quality_metrics': quality_metrics,
                'detailed_results': optimization_results,
                'recommendations': []
            }
            
            # Collect all recommendations
            for result_type, result in optimization_results.items():
                if 'recommendations' in result:
                    report['recommendations'].extend(result['recommendations'])
            
            # Save report
            report_path = Path(f"./graph_optimizations/{graph_name}_optimization_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            return {
                "path": str(report_path),
                "summary": report['optimization_summary']
            }
            
        except Exception as e:
            self.logger.error(f"Optimization report generation failed: {e}")
            raise
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent configuration"""
        return {
            "enable_color_optimization": self.config.enable_color_optimization,
            "enable_layout_optimization": self.config.enable_layout_optimization,
            "enable_typography_optimization": self.config.enable_typography_optimization,
            "quality_threshold": self.config.quality_threshold,
            "max_colors": self.config.max_colors,
            "status": "ready"
        }
