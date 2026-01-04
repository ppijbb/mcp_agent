"""
Natural Language to Cypher Query Translator

This module provides translation from natural language queries to Neo4j Cypher queries:
- Natural language understanding
- Cypher query generation
- Query optimization
- Query result validation
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from config import AgentConfig
from .llm_processor import LLMProcessor


@dataclass
class QueryTranslation:
    """Translation result from natural language to Cypher"""
    natural_language: str
    cypher_query: str
    query_type: str  # match, relationship, path, aggregation, etc.
    entities: List[str]
    relationships: List[str]
    confidence: float
    reasoning: str
    optimized: bool = False


@dataclass
class QueryResult:
    """Result from executing a Cypher query"""
    records: List[Dict[str, Any]]
    summary: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    execution_time: Optional[float] = None


class QueryTranslator:
    """
    Natural language to Cypher query translator
    
    Translates user queries into optimized Neo4j Cypher queries
    """
    
    def __init__(self, config: AgentConfig, llm_processor: LLMProcessor):
        """
        Initialize query translator
        
        Args:
            config: Agent configuration
            llm_processor: LLM processor for query generation
        """
        self.config = config
        self.llm_processor = llm_processor
        self.logger = logging.getLogger(__name__)
    
    def translate_query(
        self,
        natural_language_query: str,
        context: Optional[Dict[str, Any]] = None,
        graph_schema: Optional[Dict[str, Any]] = None
    ) -> QueryTranslation:
        """
        Translate natural language query to Cypher
        
        Args:
            natural_language_query: User's natural language query
            context: Optional context information
            graph_schema: Optional graph schema information
            
        Returns:
            QueryTranslation with Cypher query
        """
        prompt = self._build_translation_prompt(
            natural_language_query, context, graph_schema
        )
        
        response = self.llm_processor._call_llm(prompt)
        translation_data = json.loads(response)
        
        return QueryTranslation(
            natural_language=natural_language_query,
            cypher_query=translation_data.get("cypher_query", ""),
            query_type=translation_data.get("query_type", "match"),
            entities=translation_data.get("entities", []),
            relationships=translation_data.get("relationships", []),
            confidence=float(translation_data.get("confidence", 0.5)),
            reasoning=translation_data.get("reasoning", "")
        )
    
    def optimize_query(
        self,
        translation: QueryTranslation,
        graph_stats: Optional[Dict[str, Any]] = None
    ) -> QueryTranslation:
        """
        Optimize a Cypher query for better performance
        
        Args:
            translation: Query translation to optimize
            graph_stats: Optional graph statistics for optimization
            
        Returns:
            Optimized QueryTranslation
        """
        prompt = self._build_optimization_prompt(translation, graph_stats)
        
        response = self.llm_processor._call_llm(prompt)
        optimization_data = json.loads(response)
        
        optimized_query = optimization_data.get("optimized_query", translation.cypher_query)
        
        return QueryTranslation(
            natural_language=translation.natural_language,
            cypher_query=optimized_query,
            query_type=translation.query_type,
            entities=translation.entities,
            relationships=translation.relationships,
            confidence=translation.confidence,
            reasoning=optimization_data.get("optimization_reasoning", translation.reasoning),
            optimized=True
        )
    
    def validate_query_result(
        self,
        query: QueryTranslation,
        result: QueryResult,
        original_query: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate query result for correctness
        
        Args:
            query: Query translation
            result: Query execution result
            original_query: Original natural language query
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not result.success:
            return False, result.error
        
        if not result.records:
            # Check if empty result is expected
            prompt = f"""
Is an empty result expected for this query?

Original Query: "{original_query}"
Cypher Query: {query.cypher_query}
Result: Empty (no records returned)

Return JSON:
{{
    "empty_expected": true/false,
    "reasoning": "explanation"
}}
"""
            response = self.llm_processor._call_llm(prompt)
            validation_data = json.loads(response)
            
            if not validation_data.get("empty_expected", False):
                return False, "Query returned no results, but results were expected"
        
        # Validate result relevance
        prompt = f"""
Validate if the query results are relevant to the original question.

Original Query: "{original_query}"
Cypher Query: {query.cypher_query}
Result Count: {len(result.records)}
Sample Results: {json.dumps(result.records[:3], indent=2) if result.records else "None"}

Return JSON:
{{
    "is_relevant": true/false,
    "relevance_score": 0.0-1.0,
    "reasoning": "explanation",
    "suggestions": ["suggestions for improvement if not relevant"]
}}
"""
        response = self.llm_processor._call_llm(prompt)
        validation_data = json.loads(response)
        
        is_relevant = validation_data.get("is_relevant", True)
        relevance_score = float(validation_data.get("relevance_score", 1.0))
        
        if not is_relevant or relevance_score < 0.5:
            return False, f"Query results may not be relevant (score: {relevance_score})"
        
        return True, None
    
    def _build_translation_prompt(
        self,
        natural_language_query: str,
        context: Optional[Dict[str, Any]],
        graph_schema: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for query translation"""
        
        schema_context = ""
        if graph_schema:
            schema_context = f"""
Graph Schema:
- Entity Types: {graph_schema.get('entity_types', [])}
- Relationship Types: {graph_schema.get('relationship_types', [])}
- Node Labels: Entity, TextUnit
"""
        
        context_info = ""
        if context:
            context_info = f"""
Context:
- User Intent: {context.get('user_intent', '')}
- Domain: {context.get('domain', 'general')}
"""
        
        prompt = f"""
You are an expert at translating natural language queries to Neo4j Cypher queries.

Translate this natural language query to a Cypher query:

Query: "{natural_language_query}"

{schema_context}
{context_info}

Neo4j Graph Structure:
- Nodes: Entity (with properties: id, name, category, importance, confidence, context)
- Nodes: TextUnit (with properties: id, content, document_id)
- Relationships: Various types (with properties: confidence, weight, context)

Return the translation in JSON format:
{{
    "cypher_query": "MATCH ... RETURN ...",
    "query_type": "match|relationship|path|aggregation|complex",
    "entities": ["entity1", "entity2"],
    "relationships": ["relationship_type1"],
    "confidence": 0.0-1.0,
    "reasoning": "explanation of the translation"
}}

Guidelines:
- Generate valid, executable Cypher queries
- Use appropriate MATCH, WHERE, RETURN clauses
- Include proper node labels (Entity, TextUnit)
- Use relationship types from the schema when available
- Optimize for clarity and performance
- Handle edge cases (missing entities, ambiguous queries)
- Use parameters when appropriate for security
"""
        
        return prompt
    
    def _build_optimization_prompt(
        self,
        translation: QueryTranslation,
        graph_stats: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for query optimization"""
        
        stats_info = ""
        if graph_stats:
            stats_info = f"""
Graph Statistics:
- Entity Count: {graph_stats.get('entity_count', 0)}
- Relationship Count: {graph_stats.get('relationship_count', 0)}
- Relationship Types: {graph_stats.get('relationship_types', {})}
"""
        
        prompt = f"""
Optimize this Cypher query for better performance and clarity.

Original Query: "{translation.natural_language}"
Current Cypher: {translation.cypher_query}
Query Type: {translation.query_type}

{stats_info}

Return the optimized query in JSON format:
{{
    "optimized_query": "MATCH ... RETURN ...",
    "optimization_reasoning": "explanation of optimizations",
    "performance_improvements": ["improvement1", "improvement2"],
    "changes_made": ["change1", "change2"]
}}

Guidelines:
- Use indexes when available (Entity.name, Entity.category, etc.)
- Limit result sets appropriately
- Use WHERE clauses to filter early
- Avoid unnecessary traversals
- Use appropriate relationship directions
- Consider query execution plan
"""
        
        return prompt

