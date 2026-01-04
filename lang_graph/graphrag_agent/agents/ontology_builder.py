"""
Ontology Builder with Value-Based Filtering

This module implements value-based ontology construction:
- Semantic importance evaluation for relationships
- Filtering based on importance thresholds
- Service-level complexity control
- Value-centered graph structuring
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from config import AgentConfig
from .llm_processor import LLMProcessor, Entity, Relationship
from models.ontology_schema import (
    OntologySchema,
    OntologySchemaGenerator,
    EntityCategory,
    RelationshipSemantic
)


@dataclass
class RelationshipImportance:
    """Importance evaluation for a relationship"""
    relationship: Relationship
    semantic_importance: float  # 0.0-1.0
    query_value: float  # 0.0-1.0, value for querying
    service_relevance: float  # 0.0-1.0, relevance to service context
    overall_importance: float  # 0.0-1.0, combined importance score
    reasoning: str
    should_include: bool


@dataclass
class EntityImportance:
    """Importance evaluation for an entity"""
    entity: Entity
    semantic_importance: float  # 0.0-1.0
    centrality_potential: float  # 0.0-1.0, potential to be central in graph
    query_value: float  # 0.0-1.0
    overall_importance: float  # 0.0-1.0
    reasoning: str
    should_include: bool


class OntologyBuilder:
    """
    Value-based ontology builder
    
    Builds ontologies focusing on valuable relationships rather than
    trivial knowledge connections. Filters relationships based on
    semantic importance and service-level relevance.
    """
    
    def __init__(self, config: AgentConfig, llm_processor: LLMProcessor):
        """
        Initialize ontology builder
        
        Args:
            config: Agent configuration
            llm_processor: LLM processor for importance evaluation
        """
        self.config = config
        self.llm_processor = llm_processor
        self.logger = logging.getLogger(__name__)
        
        # Filtering parameters - can be overridden by config
        self.importance_threshold = getattr(config, 'ontology_importance_threshold', 0.5)
        self.complexity_control = getattr(config, 'ontology_complexity_control', True)
        self.max_relationships_per_entity = getattr(config, 'ontology_max_relationships', 20)
    
    def evaluate_relationship_importance(
        self,
        relationship: Relationship,
        source_entity: Entity,
        target_entity: Entity,
        context: Dict[str, Any],
        ontology_schema: Optional[OntologySchema] = None
    ) -> RelationshipImportance:
        """
        Evaluate semantic importance of a relationship
        
        Args:
            relationship: Relationship to evaluate
            source_entity: Source entity
            target_entity: Target entity
            context: Context information (user intent, domain, etc.)
            ontology_schema: Optional ontology schema for validation
            
        Returns:
            RelationshipImportance evaluation
        """
        prompt = self._build_importance_evaluation_prompt(
            relationship, source_entity, target_entity, context, ontology_schema
        )
        
        response = self.llm_processor._call_llm(prompt)
        evaluation_data = json.loads(response)
        
        semantic_importance = float(evaluation_data.get("semantic_importance", 0.5))
        query_value = float(evaluation_data.get("query_value", 0.5))
        service_relevance = float(evaluation_data.get("service_relevance", 0.5))
        
        # Calculate overall importance (weighted average)
        overall_importance = (
            semantic_importance * 0.4 +
            query_value * 0.4 +
            service_relevance * 0.2
        )
        
        # Apply ontology schema weights if available
        if ontology_schema:
            rel_type = ontology_schema.get_relationship_type(relationship.relationship_type)
            if rel_type:
                overall_importance *= rel_type.importance_weight
        
        should_include = overall_importance >= self.importance_threshold
        
        return RelationshipImportance(
            relationship=relationship,
            semantic_importance=semantic_importance,
            query_value=query_value,
            service_relevance=service_relevance,
            overall_importance=overall_importance,
            reasoning=evaluation_data.get("reasoning", ""),
            should_include=should_include
        )
    
    def evaluate_entity_importance(
        self,
        entity: Entity,
        context: Dict[str, Any],
        ontology_schema: Optional[OntologySchema] = None
    ) -> EntityImportance:
        """
        Evaluate importance of an entity
        
        Args:
            entity: Entity to evaluate
            context: Context information
            ontology_schema: Optional ontology schema
            
        Returns:
            EntityImportance evaluation
        """
        prompt = self._build_entity_importance_prompt(
            entity, context, ontology_schema
        )
        
        response = self.llm_processor._call_llm(prompt)
        evaluation_data = json.loads(response)
        
        semantic_importance = float(evaluation_data.get("semantic_importance", 0.5))
        centrality_potential = float(evaluation_data.get("centrality_potential", 0.5))
        query_value = float(evaluation_data.get("query_value", 0.5))
        
        overall_importance = (
            semantic_importance * 0.3 +
            centrality_potential * 0.4 +
            query_value * 0.3
        )
        
        # Apply ontology schema weights if available
        if ontology_schema:
            # Find matching entity type
            for et in ontology_schema.entity_types.values():
                if entity.name.lower() in [e.lower() for e in et.examples]:
                    overall_importance *= et.importance_weight
                    break
        
        should_include = overall_importance >= self.importance_threshold
        
        return EntityImportance(
            entity=entity,
            semantic_importance=semantic_importance,
            centrality_potential=centrality_potential,
            query_value=query_value,
            overall_importance=overall_importance,
            reasoning=evaluation_data.get("reasoning", ""),
            should_include=should_include
        )
    
    def filter_relationships(
        self,
        relationships: List[Relationship],
        entities: List[Entity],
        context: Dict[str, Any],
        ontology_schema: Optional[OntologySchema] = None
    ) -> List[RelationshipImportance]:
        """
        Filter relationships based on importance evaluation
        
        Args:
            relationships: List of candidate relationships
            entities: List of entities (for context)
            context: Context information
            ontology_schema: Optional ontology schema
            
        Returns:
            List of RelationshipImportance with should_include flag
        """
        entity_map = {e.name: e for e in entities}
        evaluations = []
        
        for relationship in relationships:
            source_entity = entity_map.get(relationship.source)
            target_entity = entity_map.get(relationship.target)
            
            if not source_entity or not target_entity:
                continue
            
            evaluation = self.evaluate_relationship_importance(
                relationship,
                source_entity,
                target_entity,
                context,
                ontology_schema
            )
            
            evaluations.append(evaluation)
        
        # Sort by importance
        evaluations.sort(key=lambda x: x.overall_importance, reverse=True)
        
        # Apply complexity control
        if self.complexity_control:
            evaluations = self._apply_complexity_control(evaluations, entity_map)
        
        return evaluations
    
    def filter_entities(
        self,
        entities: List[Entity],
        context: Dict[str, Any],
        ontology_schema: Optional[OntologySchema] = None
    ) -> List[EntityImportance]:
        """
        Filter entities based on importance evaluation
        
        Args:
            entities: List of candidate entities
            context: Context information
            ontology_schema: Optional ontology schema
            
        Returns:
            List of EntityImportance with should_include flag
        """
        evaluations = []
        
        for entity in entities:
            evaluation = self.evaluate_entity_importance(
                entity,
                context,
                ontology_schema
            )
            evaluations.append(evaluation)
        
        # Sort by importance
        evaluations.sort(key=lambda x: x.overall_importance, reverse=True)
        
        return evaluations
    
    def _apply_complexity_control(
        self,
        evaluations: List[RelationshipImportance],
        entity_map: Dict[str, Entity]
    ) -> List[RelationshipImportance]:
        """
        Apply service-level complexity control
        
        Limits relationships per entity to prevent over-complexity
        """
        entity_relationship_counts = {}
        filtered_evaluations = []
        
        for evaluation in evaluations:
            source = evaluation.relationship.source
            target = evaluation.relationship.target
            
            # Count relationships per entity
            source_count = entity_relationship_counts.get(source, 0)
            target_count = entity_relationship_counts.get(target, 0)
            
            # Include if both entities are under limit
            if (source_count < self.max_relationships_per_entity and
                target_count < self.max_relationships_per_entity):
                filtered_evaluations.append(evaluation)
                entity_relationship_counts[source] = source_count + 1
                entity_relationship_counts[target] = target_count + 1
            else:
                # Mark as excluded due to complexity control
                evaluation.should_include = False
        
        return filtered_evaluations
    
    def _build_importance_evaluation_prompt(
        self,
        relationship: Relationship,
        source_entity: Entity,
        target_entity: Entity,
        context: Dict[str, Any],
        ontology_schema: Optional[OntologySchema]
    ) -> str:
        """Build prompt for relationship importance evaluation"""
        
        schema_context = ""
        if ontology_schema:
            schema_context = f"""
Ontology Schema:
- Domain: {ontology_schema.domain}
- Entity Types: {list(ontology_schema.entity_types.keys())}
- Relationship Types: {list(ontology_schema.relationship_types.keys())}
"""
        
        prompt = f"""
You are an expert at evaluating the semantic importance of relationships in knowledge graphs.

Evaluate this relationship for inclusion in a service-oriented ontology:

Source Entity: {source_entity.name} ({source_entity.category})
Target Entity: {target_entity.name} ({target_entity.category})
Relationship Type: {relationship.relationship_type}
Relationship Context: {relationship.context}
Confidence: {relationship.confidence}

Context:
- User Intent: {context.get('user_intent', 'General knowledge graph')}
- Domain: {context.get('domain', 'general')}
- Service Level: Practical, queryable, not overly complex

{schema_context}

Evaluate the relationship and return JSON:
{{
    "semantic_importance": 0.0-1.0,  // How semantically meaningful is this relationship?
    "query_value": 0.0-1.0,  // How valuable is this for answering queries?
    "service_relevance": 0.0-1.0,  // How relevant is this to the service context?
    "reasoning": "explanation of the evaluation",
    "is_trivial": true/false,  // Is this a trivial connection that should be filtered?
    "provides_insight": true/false  // Does this relationship provide meaningful insight?
}}

Guidelines:
- Focus on relationships that provide VALUE, not just connections
- Filter out trivial relationships (e.g., "A is related to B" without meaning)
- Prioritize relationships that support querying and reasoning
- Consider service-level practicality (not academic ontology complexity)
- Evaluate based on semantic depth and query usefulness
"""
        
        return prompt
    
    def _build_entity_importance_prompt(
        self,
        entity: Entity,
        context: Dict[str, Any],
        ontology_schema: Optional[OntologySchema]
    ) -> str:
        """Build prompt for entity importance evaluation"""
        
        schema_context = ""
        if ontology_schema:
            schema_context = f"""
Ontology Schema:
- Domain: {ontology_schema.domain}
- Entity Types: {list(ontology_schema.entity_types.keys())}
"""
        
        prompt = f"""
You are an expert at evaluating the importance of entities in knowledge graphs.

Evaluate this entity for inclusion in a service-oriented ontology:

Entity: {entity.name}
Category: {entity.category}
Context: {entity.context}
Confidence: {entity.confidence}

Context:
- User Intent: {context.get('user_intent', 'General knowledge graph')}
- Domain: {context.get('domain', 'general')}
- Service Level: Practical, queryable, not overly complex

{schema_context}

Evaluate the entity and return JSON:
{{
    "semantic_importance": 0.0-1.0,  // How semantically important is this entity?
    "centrality_potential": 0.0-1.0,  // How likely is this to be central in the graph?
    "query_value": 0.0-1.0,  // How valuable is this for answering queries?
    "reasoning": "explanation of the evaluation"
}}

Guidelines:
- Focus on entities that provide VALUE for querying
- Consider centrality potential (entities that connect many others)
- Evaluate based on semantic importance and query usefulness
- Consider service-level practicality
"""
        
        return prompt

