"""
Ontology Integrator

This module integrates knowledge ontology with system ontology:
- Combines entity types from both ontologies
- Merges relationship types
- Creates unified query interface
- Supports both knowledge queries and system execution queries
"""

import logging
from typing import Dict, Any, List, Optional

from .ontology_schema import OntologySchema, EntityType, RelationshipType, EntityCategory, RelationshipSemantic
from .system_ontology import SystemOntology, Goal, Task, GoalStatus, TaskStatus


class OntologyIntegrator:
    """
    Integrates knowledge ontology with system ontology
    
    Creates a unified ontology that supports both:
    - Knowledge queries (what is X, how are A and B related)
    - System execution queries (how to achieve goal Y, what tasks are executable)
    """
    
    def __init__(self):
        """Initialize ontology integrator"""
        self.logger = logging.getLogger(__name__)
    
    def integrate_ontologies(
        self,
        knowledge_schema: OntologySchema,
        system_ontology: SystemOntology
    ) -> OntologySchema:
        """
        Integrate knowledge schema with system ontology
        
        Args:
            knowledge_schema: Knowledge ontology schema
            system_ontology: System ontology
            
        Returns:
            Integrated OntologySchema
        """
        integrated = OntologySchema(
            name=f"{knowledge_schema.name}_integrated",
            description=f"Integrated ontology: {knowledge_schema.description}",
            domain=knowledge_schema.domain,
            version="2.0.0",
            metadata={
                **knowledge_schema.metadata,
                "system_ontology_integrated": True,
                "goals_count": len(system_ontology.goals),
                "tasks_count": len(system_ontology.tasks)
            }
        )
        
        # Copy knowledge entity types
        for entity_type in knowledge_schema.entity_types.values():
            integrated.add_entity_type(entity_type)
        
        # Copy knowledge relationship types
        for rel_type in knowledge_schema.relationship_types.values():
            integrated.add_relationship_type(rel_type)
        
        # Add system ontology entity types
        self._add_system_entity_types(integrated, system_ontology)
        
        # Add system relationship types
        self._add_system_relationship_types(integrated)
        
        return integrated
    
    def _add_system_entity_types(
        self,
        schema: OntologySchema,
        system_ontology: SystemOntology
    ):
        """Add system ontology entity types to schema"""
        
        # Goal entity type
        goal_type = EntityType(
            name="Goal",
            category=EntityCategory.GOAL,
            description="A goal that the system should achieve",
            properties={
                "id": "string",
                "name": "string",
                "description": "string",
                "priority": "float",
                "status": "string"
            },
            required_properties={"id", "name", "status"},
            optional_properties={"description", "priority"},
            examples=[g.name for g in list(system_ontology.goals.values())[:5]],
            importance_weight=1.0
        )
        schema.add_entity_type(goal_type)
        
        # Task entity type
        task_type = EntityType(
            name="Task",
            category=EntityCategory.TASK,
            description="An executable task that achieves goals",
            properties={
                "id": "string",
                "name": "string",
                "description": "string",
                "status": "string",
                "priority": "float",
                "execution_time": "float",
                "success_rate": "float"
            },
            required_properties={"id", "name", "status"},
            optional_properties={"description", "priority", "execution_time", "success_rate"},
            examples=[t.name for t in list(system_ontology.tasks.values())[:5]],
            importance_weight=1.0
        )
        schema.add_entity_type(task_type)
        
        # State entity type
        state_type = EntityType(
            name="State",
            category=EntityCategory.STATE,
            description="A system state",
            properties={
                "id": "string",
                "name": "string",
                "state_type": "string",
                "value": "any"
            },
            required_properties={"id", "name", "state_type"},
            optional_properties={"value"},
            examples=[s.name for s in list(system_ontology.states.values())[:5]],
            importance_weight=0.8
        )
        schema.add_entity_type(state_type)
        
        # Resource entity type
        resource_type = EntityType(
            name="Resource",
            category=EntityCategory.RESOURCE,
            description="A system resource",
            properties={
                "id": "string",
                "name": "string",
                "resource_type": "string",
                "availability": "float",
                "capacity": "float"
            },
            required_properties={"id", "name", "resource_type"},
            optional_properties={"availability", "capacity"},
            examples=[r.name for r in list(system_ontology.resources.values())[:5]],
            importance_weight=0.7
        )
        schema.add_entity_type(resource_type)
    
    def _add_system_relationship_types(self, schema: OntologySchema):
        """Add system ontology relationship types to schema"""
        
        # HAS_SUBGOAL relationship
        has_subgoal = RelationshipType(
            name="HAS_SUBGOAL",
            semantic_type=RelationshipSemantic.HIERARCHICAL,
            description="Goal has a sub-goal",
            source_entity_types=["Goal"],
            target_entity_types=["Goal"],
            properties={},
            required_properties=set(),
            optional_properties=set(),
            examples=["Goal 'System Optimization' HAS_SUBGOAL 'Performance Improvement'"],
            importance_weight=1.0,
            directed=True
        )
        schema.add_relationship_type(has_subgoal)
        
        # ACHIEVED_BY relationship
        achieved_by = RelationshipType(
            name="ACHIEVED_BY",
            semantic_type=RelationshipSemantic.ACHIEVEMENT,
            description="Goal is achieved by task",
            source_entity_types=["Goal"],
            target_entity_types=["Task"],
            properties={},
            required_properties=set(),
            optional_properties=set(),
            examples=["Goal 'System Optimization' ACHIEVED_BY Task 'Optimize Database'"],
            importance_weight=1.0,
            directed=True
        )
        schema.add_relationship_type(achieved_by)
        
        # DEPENDS_ON relationship
        depends_on = RelationshipType(
            name="DEPENDS_ON",
            semantic_type=RelationshipSemantic.DEPENDENCY,
            description="Task depends on another task",
            source_entity_types=["Task"],
            target_entity_types=["Task"],
            properties={},
            required_properties=set(),
            optional_properties=set(),
            examples=["Task 'Deploy Service' DEPENDS_ON Task 'Build Application'"],
            importance_weight=1.0,
            directed=True
        )
        schema.add_relationship_type(depends_on)
        
        # REQUIRES relationship
        requires = RelationshipType(
            name="REQUIRES",
            semantic_type=RelationshipSemantic.PRECONDITION,
            description="Task requires precondition",
            source_entity_types=["Task"],
            target_entity_types=["Precondition"],
            properties={},
            required_properties=set(),
            optional_properties=set(),
            examples=["Task 'Deploy Service' REQUIRES Precondition 'Application Built'"],
            importance_weight=1.0,
            directed=True
        )
        schema.add_relationship_type(requires)
        
        # PRODUCES relationship
        produces = RelationshipType(
            name="PRODUCES",
            semantic_type=RelationshipSemantic.POSTCONDITION,
            description="Task produces postcondition",
            source_entity_types=["Task"],
            target_entity_types=["Postcondition"],
            properties={},
            required_properties=set(),
            optional_properties=set(),
            examples=["Task 'Build Application' PRODUCES Postcondition 'Application Built'"],
            importance_weight=1.0,
            directed=True
        )
        schema.add_relationship_type(produces)
        
        # TRANSITIONS_TO relationship
        transitions_to = RelationshipType(
            name="TRANSITIONS_TO",
            semantic_type=RelationshipSemantic.STATE_TRANSITION,
            description="Task transitions to state",
            source_entity_types=["Task"],
            target_entity_types=["State"],
            properties={},
            required_properties=set(),
            optional_properties=set(),
            examples=["Task 'Initialize System' TRANSITIONS_TO State 'System Ready'"],
            importance_weight=0.9,
            directed=True
        )
        schema.add_relationship_type(transitions_to)
        
        # CONSUMES relationship
        consumes = RelationshipType(
            name="CONSUMES",
            semantic_type=RelationshipSemantic.RESOURCE_CONSUMPTION,
            description="Task consumes resource",
            source_entity_types=["Task"],
            target_entity_types=["Resource"],
            properties={"amount": "float"},
            required_properties=set(),
            optional_properties={"amount"},
            examples=["Task 'Process Data' CONSUMES Resource 'CPU'"],
            importance_weight=0.8,
            directed=True
        )
        schema.add_relationship_type(consumes)
        
        # CONSTRAINED_BY relationship
        constrained_by = RelationshipType(
            name="CONSTRAINED_BY",
            semantic_type=RelationshipSemantic.CONSTRAINT_APPLICATION,
            description="Task is constrained by constraint",
            source_entity_types=["Task"],
            target_entity_types=["Constraint"],
            properties={},
            required_properties=set(),
            optional_properties=set(),
            examples=["Task 'Deploy Service' CONSTRAINED_BY Constraint 'Must run during maintenance window'"],
            importance_weight=0.7,
            directed=True
        )
        schema.add_relationship_type(constrained_by)
