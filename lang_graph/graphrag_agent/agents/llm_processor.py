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
import google.generativeai as genai
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
        
        # Initialize OpenAI client
        if config.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=config.openai_api_key)
        else:
            self.openai_client = None
        
        # Initialize Gemini client
        if config.gemini_api_key:
            genai.configure(api_key=config.gemini_api_key)
            self.gemini_model = genai.GenerativeModel(config.model_name)
        else:
            self.gemini_model = None
        
    def extract_entities(self, text: str, user_intent: str = None) -> List[Entity]:
        """Extract entities from text using LLM"""
        try:
            prompt = self._build_entity_extraction_prompt(text, user_intent)
            response = self._call_llm(prompt)
            return self._parse_entities_response(response)
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return []
    
    def extract_entities_batch(self, texts: List[str], user_intent: str = None) -> List[List[Entity]]:
        """Extract entities from multiple texts in batch for better performance"""
        try:
            if not texts:
                return []
            
            # Combine texts for batch processing
            combined_text = "\n\n---TEXT_SEPARATOR---\n\n".join(texts)
            prompt = self._build_batch_entity_extraction_prompt(combined_text, user_intent, len(texts))
            
            response = self._call_llm(prompt)
            return self._parse_batch_entities_response(response, len(texts))
            
        except Exception as e:
            self.logger.error(f"Batch entity extraction failed: {e}")
            # Fallback to individual processing
            return [self.extract_entities(text, user_intent) for text in texts]
    
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
    
    def _build_batch_entity_extraction_prompt(self, combined_text: str, user_intent: str = None, text_count: int = 1) -> str:
        """Build prompt for batch entity extraction"""
        base_prompt = f"""
You are an expert at extracting entities from multiple texts. Analyze the following {text_count} texts and extract all relevant entities from each.

Texts (separated by ---TEXT_SEPARATOR---):
{combined_text}

Please extract entities for each text and return them in the following JSON format:
{{
    "texts": [
        {{
            "text_index": 0,
            "entities": [
                {{
                    "name": "entity_name",
                    "category": "person|organization|location|time|event|concept|object|other",
                    "confidence": 0.0-1.0,
                    "context": "brief context where entity appears",
                    "attributes": {{"key": "value"}}
                }}
            ]
        }},
        {{
            "text_index": 1,
            "entities": [...]
        }}
    ]
}}

Guidelines:
- Extract ALL relevant entities from each text
- Use appropriate categories based on context
- Provide confidence scores based on certainty
- Include relevant attributes when available
- Be thorough but accurate
- Process each text independently
"""
        
        if user_intent:
            base_prompt += f"""

User Intent: "{user_intent}"
Focus on entities that are relevant to the user's intent and desired graph structure.
"""
        
        return base_prompt
    
    def _parse_batch_entities_response(self, response: str, text_count: int) -> List[List[Entity]]:
        """Parse LLM response for batch entities"""
        try:
            data = json.loads(response)
            results = [[] for _ in range(text_count)]
            
            for text_data in data.get("texts", []):
                text_index = text_data.get("text_index", 0)
                if 0 <= text_index < text_count:
                    entities = []
                    for entity_data in text_data.get("entities", []):
                        entity = Entity(
                            name=entity_data.get("name", ""),
                            category=entity_data.get("category", "other"),
                            confidence=entity_data.get("confidence", 0.5),
                            context=entity_data.get("context", ""),
                            attributes=entity_data.get("attributes", {})
                        )
                        entities.append(entity)
                    results[text_index] = entities
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to parse batch entities response: {e}")
            return [[] for _ in range(text_count)]
    
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
            # Use Gemini 2.5 if available, otherwise fallback to OpenAI
            if self.gemini_model and "gemini" in self.config.model_name.lower():
                return self._call_gemini(prompt)
            elif self.openai_client:
                return self._call_openai(prompt)
            else:
                self.logger.error("No LLM client available")
                return "{}"
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return "{}"
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini 2.5 with optimized settings"""
        try:
            system_prompt = "You are an expert knowledge graph analyst. Always respond with valid JSON."
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Optimized generation config for Gemini 2.5
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=min(self.config.max_tokens, 4000),  # Gemini 2.5 limit
                top_p=0.8,
                top_k=40,
                candidate_count=1,
                stop_sequences=None
            )
            
            # Use safety settings for better performance
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
            
            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Handle response properly
            if response.text:
                return response.text
            else:
                self.logger.warning("Empty response from Gemini")
                return "{}"
                
        except Exception as e:
            self.logger.error(f"Gemini call failed: {e}")
            return "{}"
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI with the given prompt"""
        try:
            response = self.openai_client.chat.completions.create(
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
            self.logger.error(f"OpenAI call failed: {e}")
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
