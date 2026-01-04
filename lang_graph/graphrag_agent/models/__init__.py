"""
Data models and type definitions for GraphRAG Agent

This module contains all the data models and type definitions used throughout the project.
"""

from .types import GraphRAGState, GraphStats, OptimizationResult

try:
    from .ontology_schema import (
        OntologySchema,
        OntologySchemaGenerator,
        EntityType,
        RelationshipType,
        EntityCategory,
        RelationshipSemantic
    )
    ONTOLOGY_SCHEMA_AVAILABLE = True
except ImportError:
    ONTOLOGY_SCHEMA_AVAILABLE = False
    OntologySchema = None
    OntologySchemaGenerator = None
    EntityType = None
    RelationshipType = None
    EntityCategory = None
    RelationshipSemantic = None

__all__ = [
    "GraphRAGState",
    "GraphStats",
    "OptimizationResult"
]

if ONTOLOGY_SCHEMA_AVAILABLE:
    __all__.extend([
        "OntologySchema",
        "OntologySchemaGenerator",
        "EntityType",
        "RelationshipType",
        "EntityCategory",
        "RelationshipSemantic"
    ])
