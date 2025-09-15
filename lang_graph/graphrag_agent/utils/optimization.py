"""
LangGraph-optimized Optimization Node
"""

import networkx as nx
from typing import Dict, Any
from typing import Callable
from models.types import GraphRAGState, OptimizationResult
from config import AgentConfig


class OptimizationNode:
    """LangGraph node for optimizing knowledge graphs"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = None
    
    def __call__(self, state: GraphRAGState) -> GraphRAGState:
        """Optimize the knowledge graph quality"""
        try:
            if not state.get("knowledge_graph"):
                state["error"] = "No knowledge graph available for optimization"
                state["status"] = "error"
                return state
            
            if not self.config.enable_optimization:
                state["status"] = "skipped"
                return state
            
            # Optimize graph
            optimization_result = self._optimize_graph(state["knowledge_graph"])
            
            # Update state
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
        """Optimize the knowledge graph"""
        # Calculate current quality metrics
        quality_score = self._calculate_quality_score(graph)
        
        # Identify improvements
        improvements = self._identify_improvements(graph)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(graph, quality_score)
        
        # Check if meets threshold
        meets_threshold = quality_score >= self.config.quality_threshold
        
        return OptimizationResult(
            quality_score=quality_score,
            improvements=improvements,
            recommendations=recommendations,
            meets_threshold=meets_threshold
        )
    
    def _calculate_quality_score(self, graph: nx.Graph) -> float:
        """Calculate overall quality score for the graph"""
        if graph.number_of_nodes() == 0:
            return 0.0
        
        # Calculate various quality metrics
        density = nx.density(graph)
        clustering = nx.average_clustering(graph) if graph.number_of_nodes() > 2 else 0.0
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(graph))
        isolation_penalty = len(isolated_nodes) / graph.number_of_nodes()
        
        # Calculate overall score (weighted average)
        score = (density * 0.4 + clustering * 0.4 + (1 - isolation_penalty) * 0.2)
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _identify_improvements(self, graph: nx.Graph) -> list:
        """Identify potential improvements for the graph"""
        improvements = []
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(graph))
        if isolated_nodes:
            improvements.append(f"Remove or connect {len(isolated_nodes)} isolated nodes")
        
        # Check for low connectivity
        if graph.number_of_nodes() > 1:
            density = nx.density(graph)
            if density < 0.1:
                improvements.append("Increase graph connectivity by adding more relationships")
        
        # Check for disconnected components
        components = list(nx.connected_components(graph))
        if len(components) > 1:
            improvements.append(f"Connect {len(components)} disconnected components")
        
        return improvements
    
    def _generate_recommendations(self, graph: nx.Graph, quality_score: float) -> list:
        """Generate recommendations for improving the graph"""
        recommendations = []
        
        if quality_score < 0.5:
            recommendations.append("Consider adding more entities and relationships")
        
        if graph.number_of_nodes() < 10:
            recommendations.append("Expand the knowledge base with more data")
        
        if nx.density(graph) < 0.05:
            recommendations.append("Increase relationship density between entities")
        
        return recommendations
