"""
Utility modules for GraphRAG Agent

This module contains utility functions and classes that are not core agents
but support the main workflow.
"""

from .visualization import VisualizationNode
from .optimization import OptimizationNode
from .sample_data import create_sample_data, create_tech_sample_data

try:
    from .neo4j_connector import Neo4jConnector, Neo4jConfig
    NEO4J_AVAILABLE = True
except ImportError:
    Neo4jConnector = None
    Neo4jConfig = None
    NEO4J_AVAILABLE = False

__all__ = [
    "VisualizationNode",
    "OptimizationNode",
    "create_sample_data",
    "create_tech_sample_data"
]

if NEO4J_AVAILABLE:
    __all__.extend(["Neo4jConnector", "Neo4jConfig"])
