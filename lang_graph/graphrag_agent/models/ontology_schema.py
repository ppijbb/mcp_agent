"""
Ontology Schema Models

This module defines the ontology schema structure for GraphRAG Agent:
- Entity types and their properties
- Relationship types and their semantics
- Schema generation and validation
- LLM-based automatic schema creation
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field
import json
import logging


class EntityCategory(str, Enum):
    """Standard entity categories"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    TIME = "time"
    EVENT = "event"
    CONCEPT = "concept"
    OBJECT = "object"
    DOCUMENT = "document"
    OTHER = "other"


class RelationshipSemantic(str, Enum):
    """Semantic types of relationships"""
    HIERARCHICAL = "hierarchical"  # parent-child, part-of
    TEMPORAL = "temporal"  # before, after, during
    CAUSAL = "causal"  # causes, influences
    SPATIAL = "spatial"  # located_in, contains
    SOCIAL = "social"  # collaborates_with, works_for
    OWNERSHIP = "ownership"  # owns, belongs_to
    FUNCTIONAL = "functional"  # uses, implements
    SEMANTIC = "semantic"  # related_to, similar_to
    OTHER = "other"


@dataclass
class EntityType:
    """Definition of an entity type in the ontology"""
    name: str
    category: EntityCategory
    description: str
    properties: Dict[str, str] = field(default_factory=dict)  # property_name -> property_type
    required_properties: Set[str] = field(default_factory=set)
    optional_properties: Set[str] = field(default_factory=set)
    examples: List[str] = field(default_factory=list)
    importance_weight: float = 1.0  # Weight for importance calculation


@dataclass
class RelationshipType:
    """Definition of a relationship type in the ontology"""
    name: str
    semantic_type: RelationshipSemantic
    description: str
    source_entity_types: List[str] = field(default_factory=list)  # Allowed source entity types
    target_entity_types: List[str] = field(default_factory=list)  # Allowed target entity types
    properties: Dict[str, str] = field(default_factory=dict)  # property_name -> property_type
    required_properties: Set[str] = field(default_factory=set)
    optional_properties: Set[str] = field(default_factory=set)
    examples: List[str] = field(default_factory=list)
    importance_weight: float = 1.0  # Weight for importance calculation
    directed: bool = True  # Whether relationship is directed


@dataclass
class OntologySchema:
    """
    Complete ontology schema definition
    
    Defines the structure of entities and relationships in the knowledge graph
    """
    name: str
    description: str
    domain: str = "general"
    entity_types: Dict[str, EntityType] = field(default_factory=dict)
    relationship_types: Dict[str, RelationshipType] = field(default_factory=dict)
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_entity_type(self, entity_type: EntityType):
        """Add an entity type to the schema"""
        self.entity_types[entity_type.name] = entity_type
    
    def add_relationship_type(self, relationship_type: RelationshipType):
        """Add a relationship type to the schema"""
        self.relationship_types[relationship_type.name] = relationship_type
    
    def get_entity_type(self, name: str) -> Optional[EntityType]:
        """Get entity type by name"""
        return self.entity_types.get(name)
    
    def get_relationship_type(self, name: str) -> Optional[RelationshipType]:
        """Get relationship type by name"""
        return self.relationship_types.get(name)
    
    def validate_entity(self, entity_name: str, entity_data: Dict[str, Any]) -> bool:
        """Validate entity data against schema"""
        # Find matching entity type
        entity_type = None
        for et in self.entity_types.values():
            if entity_name.lower() in [e.lower() for e in et.examples]:
                entity_type = et
                break
        
        if not entity_type:
            return True  # No specific type found, allow it
        
        # Check required properties
        for prop in entity_type.required_properties:
            if prop not in entity_data:
                return False
        
        return True
    
    def validate_relationship(
        self, 
        rel_type: str, 
        source_type: str, 
        target_type: str
    ) -> bool:
        """Validate relationship against schema"""
        relationship_type = self.get_relationship_type(rel_type)
        if not relationship_type:
            return True  # Unknown type, allow it
        
        # Check if source/target types are allowed
        if relationship_type.source_entity_types:
            if source_type not in relationship_type.source_entity_types:
                return False
        
        if relationship_type.target_entity_types:
            if target_type not in relationship_type.target_entity_types:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "version": self.version,
            "entity_types": {
                name: {
                    "name": et.name,
                    "category": et.category.value,
                    "description": et.description,
                    "properties": et.properties,
                    "required_properties": list(et.required_properties),
                    "optional_properties": list(et.optional_properties),
                    "examples": et.examples,
                    "importance_weight": et.importance_weight
                }
                for name, et in self.entity_types.items()
            },
            "relationship_types": {
                name: {
                    "name": rt.name,
                    "semantic_type": rt.semantic_type.value,
                    "description": rt.description,
                    "source_entity_types": rt.source_entity_types,
                    "target_entity_types": rt.target_entity_types,
                    "properties": rt.properties,
                    "required_properties": list(rt.required_properties),
                    "optional_properties": list(rt.optional_properties),
                    "examples": rt.examples,
                    "importance_weight": rt.importance_weight,
                    "directed": rt.directed
                }
                for name, rt in self.relationship_types.items()
            },
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologySchema":
        """Create schema from dictionary"""
        schema = cls(
            name=data["name"],
            description=data["description"],
            domain=data.get("domain", "general"),
            version=data.get("version", "1.0.0"),
            metadata=data.get("metadata", {})
        )
        
        # Load entity types
        for name, et_data in data.get("entity_types", {}).items():
            entity_type = EntityType(
                name=et_data["name"],
                category=EntityCategory(et_data["category"]),
                description=et_data["description"],
                properties=et_data.get("properties", {}),
                required_properties=set(et_data.get("required_properties", [])),
                optional_properties=set(et_data.get("optional_properties", [])),
                examples=et_data.get("examples", []),
                importance_weight=et_data.get("importance_weight", 1.0)
            )
            schema.add_entity_type(entity_type)
        
        # Load relationship types
        for name, rt_data in data.get("relationship_types", {}).items():
            relationship_type = RelationshipType(
                name=rt_data["name"],
                semantic_type=RelationshipSemantic(rt_data["semantic_type"]),
                description=rt_data["description"],
                source_entity_types=rt_data.get("source_entity_types", []),
                target_entity_types=rt_data.get("target_entity_types", []),
                properties=rt_data.get("properties", {}),
                required_properties=set(rt_data.get("required_properties", [])),
                optional_properties=set(rt_data.get("optional_properties", [])),
                examples=rt_data.get("examples", []),
                importance_weight=rt_data.get("importance_weight", 1.0),
                directed=rt_data.get("directed", True)
            )
            schema.add_relationship_type(relationship_type)
        
        return schema


class OntologySchemaGenerator:
    """
    LLM-based automatic ontology schema generator
    
    Generates ontology schemas based on data analysis and domain understanding
    """
    
    def __init__(self, llm_processor):
        """
        Initialize schema generator
        
        Args:
            llm_processor: LLMProcessor instance for LLM calls
        """
        self.llm_processor = llm_processor
        self.logger = logging.getLogger(__name__)
    
    def generate_schema(
        self,
        data_sample: str,
        domain: str = "general",
        user_intent: str = "",
        existing_schema: Optional[OntologySchema] = None
    ) -> OntologySchema:
        """
        Generate ontology schema from data sample
        
        Args:
            data_sample: Sample data to analyze
            domain: Domain context
            user_intent: User's intent for the graph
            existing_schema: Optional existing schema to extend
            
        Returns:
            Generated ontology schema
        """
        prompt = self._build_schema_generation_prompt(
            data_sample, domain, user_intent, existing_schema
        )
        
        response = self.llm_processor._call_llm(prompt)
        schema_data = json.loads(response)
        
        return self._parse_schema_response(schema_data, domain)
    
    def _build_schema_generation_prompt(
        self,
        data_sample: str,
        domain: str,
        user_intent: str,
        existing_schema: Optional[OntologySchema]
    ) -> str:
        """Build prompt for schema generation"""
        
        existing_context = ""
        if existing_schema:
            existing_context = f"""
Existing Schema:
{json.dumps(existing_schema.to_dict(), indent=2)}

Extend and refine this schema based on the new data.
"""
        
        prompt = f"""
You are an expert ontology engineer. Design a comprehensive but practical ontology schema for knowledge graph construction.

Domain: {domain}
User Intent: {user_intent if user_intent else "General knowledge graph generation"}

Data Sample:
{data_sample[:2000]}

{existing_context}

Design an ontology schema that:
1. Defines entity types relevant to the data and domain
2. Defines relationship types that capture valuable relationships (not just trivial connections)
3. Focuses on service-level practicality (not overly complex academic ontology)
4. Emphasizes relationships that provide meaningful insights
5. Includes properties that support querying and reasoning

Return the schema in JSON format:
{{
    "name": "schema_name",
    "description": "description of the schema",
    "domain": "{domain}",
    "entity_types": {{
        "EntityTypeName": {{
            "name": "EntityTypeName",
            "category": "person|organization|location|time|event|concept|object|document|other",
            "description": "description of this entity type",
            "properties": {{
                "property_name": "property_type"
            }},
            "required_properties": ["property1", "property2"],
            "optional_properties": ["property3"],
            "examples": ["example1", "example2"],
            "importance_weight": 0.0-1.0
        }}
    }},
    "relationship_types": {{
        "RelationshipTypeName": {{
            "name": "RelationshipTypeName",
            "semantic_type": "hierarchical|temporal|causal|spatial|social|ownership|functional|semantic|other",
            "description": "description of this relationship type",
            "source_entity_types": ["EntityType1", "EntityType2"],
            "target_entity_types": ["EntityType3", "EntityType4"],
            "properties": {{
                "property_name": "property_type"
            }},
            "required_properties": ["property1"],
            "optional_properties": ["property2"],
            "examples": ["example1", "example2"],
            "importance_weight": 0.0-1.0,
            "directed": true
        }}
    }},
    "metadata": {{
        "complexity_level": "simple|medium|complex",
        "focus_areas": ["area1", "area2"],
        "reasoning": "explanation of schema design decisions"
    }}
}}

Guidelines:
- Keep entity types practical and service-oriented
- Focus on relationship types that provide value (not trivial connections)
- Avoid over-engineering (20-year-old ontology complexity is not needed)
- Ensure relationships are queryable and meaningful
- Consider the user's intent and domain context
- Balance depth with practicality
"""
        
        return prompt
    
    def _parse_schema_response(
        self,
        schema_data: Dict[str, Any],
        domain: str
    ) -> OntologySchema:
        """Parse LLM response into OntologySchema"""
        
        schema = OntologySchema(
            name=schema_data.get("name", f"{domain}_schema"),
            description=schema_data.get("description", ""),
            domain=domain,
            version="1.0.0",
            metadata=schema_data.get("metadata", {})
        )
        
        # Parse entity types
        for name, et_data in schema_data.get("entity_types", {}).items():
            try:
                entity_type = EntityType(
                    name=et_data.get("name", name),
                    category=EntityCategory(et_data.get("category", "other")),
                    description=et_data.get("description", ""),
                    properties=et_data.get("properties", {}),
                    required_properties=set(et_data.get("required_properties", [])),
                    optional_properties=set(et_data.get("optional_properties", [])),
                    examples=et_data.get("examples", []),
                    importance_weight=float(et_data.get("importance_weight", 1.0))
                )
                schema.add_entity_type(entity_type)
            except Exception as e:
                self.logger.warning(f"Failed to parse entity type {name}: {e}")
        
        # Parse relationship types
        for name, rt_data in schema_data.get("relationship_types", {}).items():
            try:
                relationship_type = RelationshipType(
                    name=rt_data.get("name", name),
                    semantic_type=RelationshipSemantic(rt_data.get("semantic_type", "other")),
                    description=rt_data.get("description", ""),
                    source_entity_types=rt_data.get("source_entity_types", []),
                    target_entity_types=rt_data.get("target_entity_types", []),
                    properties=rt_data.get("properties", {}),
                    required_properties=set(rt_data.get("required_properties", [])),
                    optional_properties=set(rt_data.get("optional_properties", [])),
                    examples=rt_data.get("examples", []),
                    importance_weight=float(rt_data.get("importance_weight", 1.0)),
                    directed=rt_data.get("directed", True)
                )
                schema.add_relationship_type(relationship_type)
            except Exception as e:
                self.logger.warning(f"Failed to parse relationship type {name}: {e}")
        
        return schema

