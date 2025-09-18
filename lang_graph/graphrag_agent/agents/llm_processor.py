"""
LLM-based Dynamic Graph Processing

This module provides LLM-powered entity extraction, classification, and relationship detection
without any hardcoded patterns or keywords.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import openai
from config import AgentConfig


@dataclass
class Entity:
    """Represents an extracted entity"""
    name: str
    category: str
    confidence: float
    context: str
    attributes: Dict[str, Any] = None


@dataclass
class Relationship:
    """Represents a relationship between entities"""
    source: str
    target: str
    relationship_type: str
    confidence: float
    context: str
    attributes: Dict[str, Any] = None


class LLMProcessor:
    """LLM-powered dynamic graph processing"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = openai.OpenAI(api_key=config.openai_api_key)
        
    def extract_entities(self, text: str, user_intent: str = None) -> List[Entity]:
        """Extract entities from text using LLM"""
        try:
            prompt = self._build_entity_extraction_prompt(text, user_intent)
            response = self._call_llm(prompt)
            return self._parse_entities_response(response)
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return []
    
    def extract_relationships(self, text: str, entities: List[Entity], user_intent: str = None) -> List[Relationship]:
        """Extract relationships between entities using LLM"""
        try:
            prompt = self._build_relationship_extraction_prompt(text, entities, user_intent)
            response = self._call_llm(prompt)
            return self._parse_relationships_response(response)
        except Exception as e:
            self.logger.error(f"Relationship extraction failed: {e}")
            return []
    
    def classify_entities(self, entities: List[Entity], context: str) -> List[Entity]:
        """Classify entities into categories using LLM"""
        try:
            prompt = self._build_entity_classification_prompt(entities, context)
            response = self._call_llm(prompt)
            return self._parse_classification_response(entities, response)
        except Exception as e:
            self.logger.error(f"Entity classification failed: {e}")
            return entities
    
    def generate_graph_structure(self, text: str, user_intent: str) -> Dict[str, Any]:
        """Generate graph structure based on user intent"""
        try:
            prompt = self._build_graph_structure_prompt(text, user_intent)
            response = self._call_llm(prompt)
            return self._parse_graph_structure_response(response)
        except Exception as e:
            self.logger.error(f"Graph structure generation failed: {e}")
            return {}
    
    def _build_entity_extraction_prompt(self, text: str, user_intent: str = None) -> str:
        """Build prompt for entity extraction"""
        base_prompt = f"""
You are an expert at extracting entities from text. Analyze the following text and extract all relevant entities.

Text: "{text}"

Please extract entities and return them in the following JSON format:
{{
    "entities": [
        {{
            "name": "entity_name",
            "category": "person|organization|location|time|event|concept|object|other",
            "confidence": 0.0-1.0,
            "context": "brief context where entity appears",
            "attributes": {{"key": "value"}}
        }}
    ]
}}

Guidelines:
- Extract ALL relevant entities, not just obvious ones
- Use appropriate categories based on context
- Provide confidence scores based on certainty
- Include relevant attributes when available
- Be thorough but accurate
"""
        
        if user_intent:
            base_prompt += f"""

User Intent: "{user_intent}"
Focus on entities that are relevant to the user's intent and desired graph structure.
"""
        
        return base_prompt
    
    def _build_relationship_extraction_prompt(self, text: str, entities: List[Entity], user_intent: str = None) -> str:
        """Build prompt for relationship extraction"""
        entity_names = [entity.name for entity in entities]
        
        base_prompt = f"""
You are an expert at identifying relationships between entities. Analyze the following text and identify relationships between the given entities.

Text: "{text}"

Entities to analyze: {entity_names}

Please identify relationships and return them in the following JSON format:
{{
    "relationships": [
        {{
            "source": "entity1_name",
            "target": "entity2_name",
            "relationship_type": "descriptive_relationship_type",
            "confidence": 0.0-1.0,
            "context": "brief context of the relationship",
            "attributes": {{"key": "value"}}
        }}
    ]
}}

Guidelines:
- Identify ALL meaningful relationships between entities
- Use descriptive relationship types (e.g., "founded_by", "located_in", "studies", "collaborates_with")
- Provide confidence scores based on certainty
- Include relevant attributes when available
- Consider both direct and indirect relationships
"""
        
        if user_intent:
            base_prompt += f"""

User Intent: "{user_intent}"
Focus on relationships that are relevant to the user's intent and desired graph structure.
"""
        
        return base_prompt
    
    def _build_entity_classification_prompt(self, entities: List[Entity], context: str) -> str:
        """Build prompt for entity classification"""
        entity_list = [f"- {entity.name} (current: {entity.category})" for entity in entities]
        
        prompt = f"""
You are an expert at classifying entities. Review the following entities and their current classifications, then provide improved classifications based on the context.

Context: "{context}"

Entities to classify:
{chr(10).join(entity_list)}

Please return the improved classifications in the following JSON format:
{{
    "classifications": [
        {{
            "name": "entity_name",
            "category": "person|organization|location|time|event|concept|object|other",
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation for the classification"
        }}
    ]
}}

Guidelines:
- Use the most appropriate category for each entity
- Consider the context and domain
- Provide confidence scores based on certainty
- Include reasoning for your classifications
"""
        
        return prompt
    
    def _build_graph_structure_prompt(self, text: str, user_intent: str) -> str:
        """Build prompt for graph structure generation"""
        prompt = f"""
You are an expert at designing knowledge graph structures. Based on the user's intent and the text content, design an appropriate graph structure.

Text: "{text}"

User Intent: "{user_intent}"

Please design a graph structure and return it in the following JSON format:
{{
    "graph_type": "hierarchical|network|timeline|taxonomy|other",
    "main_entities": ["entity1", "entity2", "entity3"],
    "entity_categories": {{
        "person": ["entity1", "entity2"],
        "organization": ["entity3", "entity4"]
    }},
    "key_relationships": ["relationship1", "relationship2"],
    "focus_areas": ["area1", "area2"],
    "visualization_suggestions": ["suggestion1", "suggestion2"]
}}

Guidelines:
- Design a structure that matches the user's intent
- Identify the most important entities and relationships
- Suggest appropriate visualization approaches
- Consider the domain and content type
"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert knowledge graph analyst. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return "{}"
    
    def _parse_entities_response(self, response: str) -> List[Entity]:
        """Parse LLM response for entities"""
        try:
            data = json.loads(response)
            entities = []
            for entity_data in data.get("entities", []):
                entity = Entity(
                    name=entity_data.get("name", ""),
                    category=entity_data.get("category", "other"),
                    confidence=entity_data.get("confidence", 0.5),
                    context=entity_data.get("context", ""),
                    attributes=entity_data.get("attributes", {})
                )
                entities.append(entity)
            return entities
        except Exception as e:
            self.logger.error(f"Failed to parse entities response: {e}")
            return []
    
    def _parse_relationships_response(self, response: str) -> List[Relationship]:
        """Parse LLM response for relationships"""
        try:
            data = json.loads(response)
            relationships = []
            for rel_data in data.get("relationships", []):
                relationship = Relationship(
                    source=rel_data.get("source", ""),
                    target=rel_data.get("target", ""),
                    relationship_type=rel_data.get("relationship_type", "related_to"),
                    confidence=rel_data.get("confidence", 0.5),
                    context=rel_data.get("context", ""),
                    attributes=rel_data.get("attributes", {})
                )
                relationships.append(relationship)
            return relationships
        except Exception as e:
            self.logger.error(f"Failed to parse relationships response: {e}")
            return []
    
    def _parse_classification_response(self, entities: List[Entity], response: str) -> List[Entity]:
        """Parse LLM response for entity classification"""
        try:
            data = json.loads(response)
            classifications = {item["name"]: item for item in data.get("classifications", [])}
            
            updated_entities = []
            for entity in entities:
                if entity.name in classifications:
                    classification = classifications[entity.name]
                    entity.category = classification.get("category", entity.category)
                    entity.confidence = classification.get("confidence", entity.confidence)
                updated_entities.append(entity)
            
            return updated_entities
        except Exception as e:
            self.logger.error(f"Failed to parse classification response: {e}")
            return entities
    
    def _parse_graph_structure_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for graph structure"""
        try:
            data = json.loads(response)
            return data
        except Exception as e:
            self.logger.error(f"Failed to parse graph structure response: {e}")
            return {}
