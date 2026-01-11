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

try:
    from .system_ontology import (
        SystemOntology,
        Goal,
        Task,
        Precondition,
        Postcondition,
        State,
        StateTransition,
        Resource,
        ResourceRequirement,
        Constraint,
        GoalStatus,
        TaskStatus,
        StateType,
        ConstraintType,
        ConstraintSeverity
    )
    SYSTEM_ONTOLOGY_AVAILABLE = True
except ImportError:
    SYSTEM_ONTOLOGY_AVAILABLE = False
    SystemOntology = None
    Goal = None
    Task = None
    Precondition = None
    Postcondition = None
    State = None
    StateTransition = None
    Resource = None
    ResourceRequirement = None
    Constraint = None
    GoalStatus = None
    TaskStatus = None
    StateType = None
    ConstraintType = None
    ConstraintSeverity = None

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

if SYSTEM_ONTOLOGY_AVAILABLE:
    __all__.extend([
        "SystemOntology",
        "Goal",
        "Task",
        "Precondition",
        "Postcondition",
        "State",
        "StateTransition",
        "Resource",
        "ResourceRequirement",
        "Constraint",
        "GoalStatus",
        "TaskStatus",
        "StateType",
        "ConstraintType",
        "ConstraintSeverity"
    ])

try:
    from .ontology_integrator import OntologyIntegrator
    ONTOLOGY_INTEGRATOR_AVAILABLE = True
except ImportError:
    ONTOLOGY_INTEGRATOR_AVAILABLE = False
    OntologyIntegrator = None

if ONTOLOGY_INTEGRATOR_AVAILABLE:
    __all__.append("OntologyIntegrator")
