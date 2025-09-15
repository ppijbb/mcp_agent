"""
Type definitions and data models for GraphRAG Agent

This module contains all the data models and type definitions used throughout the project.
"""

from typing import TypedDict, Optional, Dict, Any, List
from pydantic import BaseModel


class GraphRAGState(TypedDict, total=False):
    """State for GraphRAG workflow"""
    mode: str
    data_file: Optional[str]
    graph_path: Optional[str]
    output_path: Optional[str]
    query: Optional[str]
    knowledge_graph: Optional[Any]  # networkx.Graph
    result: Optional[str]
    stats: Optional[Dict[str, Any]]


# AgentConfig is defined in config.config_manager


class GraphStats(TypedDict):
    """Statistics about the generated graph"""
    num_nodes: int
    num_edges: int
    density: float
    clustering_coefficient: float
    average_degree: float


class OptimizationResult(TypedDict):
    """Result of graph optimization"""
    original_stats: GraphStats
    optimized_stats: GraphStats
    improvements: Dict[str, float]
    removed_nodes: List[str]
    removed_edges: List[tuple]