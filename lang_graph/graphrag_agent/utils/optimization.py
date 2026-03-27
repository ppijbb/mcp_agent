"""
LangGraph-optimized Optimization Node

This module provides a LangGraph node for optimizing knowledge graphs by calculating
quality scores, identifying improvements, and generating recommendations.

Classes:
    OptimizationNode: LangGraph node for optimizing knowledge graphs

Example:
    >>> from lang_graph.graphrag_agent.utils.optimization import OptimizationNode
    >>> from lang_graph.graphrag_agent.config import AgentConfig
    >>> config = AgentConfig()
    >>> node = OptimizationNode(config)
    >>> result = node(state)
"""

import logging
from typing import Dict, Any, Optional

import networkx as nx

try:
    from models.types import GraphRAGState, OptimizationResult
except ImportError:
    from ..models.types import GraphRAGState, OptimizationResult

try:
    from config import AgentConfig
except ImportError:
    from ..config import AgentConfig


class OptimizationNode:
    """
    LangGraph node for optimizing knowledge graphs.
    
    This node evaluates the quality of a knowledge graph and provides
    recommendations for improvements based on various metrics including
    density, clustering, and connectivity.
    
    Attributes:
        config: Agent configuration containing optimization settings
        logger: Optional logger instance for recording operations
    
    Example:
        >>> config = AgentConfig(quality_threshold=0.8)
        >>> node = OptimizationNode(config)
        >>> optimized_state = node(current_state)
    """
    
    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize the OptimizationNode.
        
        Args:
            config: Agent configuration containing optimization settings
        """
        self.config = config
        self.logger: Optional[logging.Logger] = None
    
    def __call__(self, state: GraphRAGState) -> GraphRAGState:
        """
        Optimize the knowledge graph quality.
        
        This method is the main entry point for the LangGraph node. It validates
        the state, checks if optimization is enabled, and performs the optimization.
        
        Args:
            state: Current GraphRAG state containing knowledge_graph
            
        Returns:
            Updated state with optimization results or error information
        """
        try:
            if not state.get("knowledge_graph"):
                state["error"] = "No knowledge graph available for optimization"
                state["status"] = "error"
                return state
            
            if not self.config.enable_optimization:
                state["status"] = "skipped"
                return state
            
            optimization_result = self._optimize_graph(state["knowledge_graph"])
            
            state["optimization_results"] = optimization_result
            state["status"] = "completed"
            
            if self.logger:
                self.logger.info(f"Graph optimization completed. Quality: {optimization_result.quality_score:.3f}")
            
        except Exception as e:
            state["error"] = f"Optimization failed: {str(e)}"
            state["status"] = "error"
            if self.logger:
                self.logger.error(f"Optimization error: {e}")
        
        return state
    
    def _optimize_graph(self, graph: nx.Graph) -> OptimizationResult:
        """
        Optimize the knowledge graph by calculating quality metrics.
        
        Analyzes the graph structure and generates improvements and recommendations
        based on various quality metrics.
        
        Args:
            graph: The knowledge graph to optimize
            
        Returns:
            OptimizationResult containing quality score and recommendations
        """
        quality_score = self._calculate_quality_score(graph)
        improvements = self._identify_improvements(graph)
        recommendations = self._generate_recommendations(graph, quality_score)
        meets_threshold = quality_score >= self.config.quality_threshold
        
        return OptimizationResult(
            quality_score=quality_score,
            improvements=improvements,
            recommendations=recommendations,
            meets_threshold=meets_threshold
        )
    
    def _calculate_quality_score(self, graph: nx.Graph) -> float:
        """
        Calculate overall quality score for the graph.
        
        Computes a quality score based on graph density, clustering coefficient,
        and isolation penalty. The score is a weighted average of these metrics.
        
        Args:
            graph: The knowledge graph to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if graph.number_of_nodes() == 0:
            return 0.0
        
        density = nx.density(graph)
        clustering = nx.average_clustering(graph) if graph.number_of_nodes() > 2 else 0.0
        
        isolated_nodes = list(nx.isolates(graph))
        isolation_penalty = len(isolated_nodes) / graph.number_of_nodes()
        
        score = (density * 0.4 + clustering * 0.4 + (1 - isolation_penalty) * 0.2)
        
        return min(score, 1.0)
    
    def _identify_improvements(self, graph: nx.Graph) -> list:
        """
        Identify potential improvements for the graph.
        
        Analyzes the graph structure to find issues such as isolated nodes,
        low connectivity, and disconnected components.
        
        Args:
            graph: The knowledge graph to analyze
            
        Returns:
            List of improvement suggestions as strings
        """
        improvements = []
        
        isolated_nodes = list(nx.isolates(graph))
        if isolated_nodes:
            improvements.append(f"Remove or connect {len(isolated_nodes)} isolated nodes")
        
        if graph.number_of_nodes() > 1:
            density = nx.density(graph)
            if density < 0.1:
                improvements.append("Increase graph connectivity by adding more relationships")
        
        components = list(nx.connected_components(graph))
        if len(components) > 1:
            improvements.append(f"Connect {len(components)} disconnected components")
        
        return improvements
    
    def _generate_recommendations(self, graph: nx.Graph, quality_score: float) -> list:
        """
        Generate recommendations for improving the graph.
        
        Provides actionable recommendations based on the current quality score
        and graph characteristics.
        
        Args:
            graph: The knowledge graph to improve
            quality_score: Current quality score of the graph
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if quality_score < 0.5:
            recommendations.append("Consider adding more entities and relationships")
        
        if graph.number_of_nodes() < 10:
            recommendations.append("Expand the knowledge base with more data")
        
        if nx.density(graph) < 0.05:
            recommendations.append("Increase relationship density between entities")
        
        return recommendations
