"""
True GraphRAG Intelligent Retrieval Agent Node

This node implements genuine GraphRAG retrieval capabilities:
- Autonomous query understanding and analysis
- Intelligent graph traversal and reasoning
- Context-aware information synthesis
- Multi-hop reasoning and inference
- Self-directed learning from queries
"""

import networkx as nx
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from typing import Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from models.types import GraphRAGState
from config import AgentConfig
from .llm_processor import LLMProcessor


class QueryType(Enum):
    """Types of queries that can be processed"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"
    RELATIONSHIP = "relationship"
    COMPLEX = "complex"


class RetrievalStrategy(Enum):
    """Strategies for information retrieval"""
    DIRECT_MATCH = "direct_match"
    GRAPH_TRAVERSAL = "graph_traversal"
    MULTI_HOP = "multi_hop"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CONTEXT_AWARE = "context_aware"
    REASONING_BASED = "reasoning_based"


@dataclass
class QueryAnalysis:
    """Analysis of a user query"""
    query_type: QueryType
    entities: List[str]
    relationships: List[str]
    intent: str
    complexity: str
    retrieval_strategy: RetrievalStrategy
    confidence: float
    reasoning: str


@dataclass
class RetrievalResult:
    """Result of information retrieval"""
    content: str
    source_nodes: List[str]
    confidence: float
    reasoning_path: List[str]
    context: Dict[str, Any]
    metadata: Dict[str, Any]


class RAGAgentNode:
    """
    True GraphRAG Intelligent Retrieval Agent
    
    This node embodies genuine GraphRAG principles:
    - Autonomous query understanding and analysis
    - Intelligent graph traversal and reasoning
    - Context-aware information synthesis
    - Multi-hop reasoning and inference
    - Self-directed learning from queries
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm_processor = LLMProcessor(config)
        
        # GraphRAG learning state
        self.query_patterns = []
        self.successful_retrievals = []
        self.domain_knowledge = {}
        self.retrieval_strategies = {}
        
        # Autonomous capabilities
        self.learning_enabled = True
        self.adaptation_threshold = 0.7
        self.reasoning_depth = 3
    
    def __call__(self, state: GraphRAGState) -> GraphRAGState:
        """
        Process query using true GraphRAG principles
        
        This method implements autonomous query processing:
        1. Autonomous query understanding and analysis
        2. Intelligent graph traversal and reasoning
        3. Context-aware information synthesis
        4. Multi-hop reasoning and inference
        5. Learning from the query process
        """
        try:
            if not state.get("query"):
                state["error"] = "No query provided"
                state["status"] = "error"
                return state
            
            if not state.get("knowledge_graph"):
                state["error"] = "No knowledge graph available"
                state["status"] = "error"
                return state
            
            # Step 1: Autonomous query analysis
            query_analysis = self._analyze_query_autonomously(state["query"], state.get("user_intent", ""))
            
            # Step 2: Intelligent information retrieval
            retrieval_results = self._retrieve_information_intelligently(
                state["knowledge_graph"], 
                query_analysis,
                state.get("user_intent", "")
            )
            
            # Step 3: Context-aware response generation
            response = self._generate_intelligent_response(
                state["query"],
                query_analysis,
                retrieval_results,
                state.get("user_intent", "")
            )
            
            # Step 4: Learning from the query process
            await self._learn_from_query_processing(query_analysis, retrieval_results, state["query"])
            
            # Update state
            state["query_analysis"] = query_analysis
            state["retrieval_results"] = retrieval_results
            state["response"] = response
            state["status"] = "completed"
            
            if self.logger:
                self.logger.info(f"GraphRAG query processed: {len(retrieval_results)} results found, strategy: {query_analysis.retrieval_strategy.value}")
            
        except Exception as e:
            state["error"] = f"GraphRAG processing failed: {str(e)}"
            state["status"] = "error"
            if self.logger:
                self.logger.error(f"GraphRAG processing error: {e}")
        
        return state
    
    def _analyze_query_autonomously(self, query: str, user_intent: str = "") -> QueryAnalysis:
        """Autonomous query analysis using LLM"""
        analysis_prompt = f"""
You are an expert at analyzing queries for knowledge graph retrieval. Analyze this query autonomously.

Query: "{query}"
User Intent: {user_intent if user_intent else "General information retrieval"}

Provide analysis in JSON format:
{{
    "query_type": "factual|analytical|comparative|causal|temporal|hierarchical|relationship|complex",
    "entities": ["entities mentioned in the query"],
    "relationships": ["relationships the user is asking about"],
    "intent": "what the user wants to know",
    "complexity": "simple|medium|complex",
    "retrieval_strategy": "direct_match|graph_traversal|multi_hop|semantic_similarity|context_aware|reasoning_based",
    "confidence": 0.0-1.0,
    "reasoning": "explanation for the analysis"
}}
"""
        
        response = self.llm_processor._call_llm(analysis_prompt)
        analysis_data = json.loads(response)
        
        query_type = QueryType(analysis_data.get("query_type", "factual"))
        retrieval_strategy = RetrievalStrategy(analysis_data.get("retrieval_strategy", "direct_match"))
        
        return QueryAnalysis(
            query_type=query_type,
            entities=analysis_data.get("entities", []),
            relationships=analysis_data.get("relationships", []),
            intent=analysis_data.get("intent", ""),
            complexity=analysis_data.get("complexity", "simple"),
            retrieval_strategy=retrieval_strategy,
            confidence=analysis_data.get("confidence", 0.5),
            reasoning=analysis_data.get("reasoning", "")
        )
    
    def _retrieve_information_intelligently(self, graph: nx.Graph, query_analysis: QueryAnalysis, user_intent: str = "") -> List[RetrievalResult]:
        """Intelligent information retrieval based on query analysis"""
        retrieval_results = []
        
        # Choose retrieval strategy based on query analysis
        if query_analysis.retrieval_strategy == RetrievalStrategy.DIRECT_MATCH:
            retrieval_results = self._direct_match_retrieval(graph, query_analysis)
        elif query_analysis.retrieval_strategy == RetrievalStrategy.GRAPH_TRAVERSAL:
            retrieval_results = self._graph_traversal_retrieval(graph, query_analysis)
        elif query_analysis.retrieval_strategy == RetrievalStrategy.MULTI_HOP:
            retrieval_results = self._multi_hop_retrieval(graph, query_analysis)
        else:
            # Default to direct match
            retrieval_results = self._direct_match_retrieval(graph, query_analysis)
        
        # Sort by confidence and relevance
        retrieval_results.sort(key=lambda x: x.confidence, reverse=True)
        return retrieval_results[:self.config.max_search_results]
    
    def _direct_match_retrieval(self, graph: nx.Graph, query_analysis: QueryAnalysis) -> List[RetrievalResult]:
        """Direct match retrieval for simple factual queries"""
        results = []
        query_entities = query_analysis.entities
        
        for node_id, data in graph.nodes(data=True):
            if data.get("type") == "entity":
                entity_name = data.get("name", "").lower()
                for entity in query_entities:
                    if entity.lower() in entity_name or entity_name in entity.lower():
                        # Find related information
                        related_info = self._gather_related_information(graph, node_id)
                        
                        result = RetrievalResult(
                            content=related_info["content"],
                            source_nodes=[node_id] + related_info["related_nodes"],
                            confidence=query_analysis.confidence,
                            reasoning_path=[f"Direct match for entity: {entity}"],
                            context=related_info["context"],
                            metadata=data
                        )
                        results.append(result)
        
        return results
    
    def _graph_traversal_retrieval(self, graph: nx.Graph, query_analysis: QueryAnalysis) -> List[RetrievalResult]:
        """Graph traversal retrieval for relationship queries"""
        results = []
        query_entities = query_analysis.entities
        
        for entity in query_entities:
            # Find entity nodes
            entity_nodes = [n for n in graph.nodes() if n.startswith("entity_") and 
                          entity.lower() in graph.nodes[n].get("name", "").lower()]
            
            for entity_node in entity_nodes:
                # Traverse relationships
                traversal_results = self._traverse_relationships(graph, entity_node, query_analysis)
                results.extend(traversal_results)
        
        return results
    
    def _multi_hop_retrieval(self, graph: nx.Graph, query_analysis: QueryAnalysis) -> List[RetrievalResult]:
        """Multi-hop reasoning retrieval for complex queries"""
        results = []
        query_entities = query_analysis.entities
        
        for entity in query_entities:
            # Find entity nodes
            entity_nodes = [n for n in graph.nodes() if n.startswith("entity_") and 
                          entity.lower() in graph.nodes[n].get("name", "").lower()]
            
            for entity_node in entity_nodes:
                # Perform multi-hop reasoning
                reasoning_results = self._perform_multi_hop_reasoning(graph, entity_node, query_analysis)
                results.extend(reasoning_results)
        
        return results
    
    def _gather_related_information(self, graph: nx.Graph, node_id: str) -> Dict[str, Any]:
        """Gather information related to a node"""
        data = graph.nodes[node_id]
        related_nodes = []
        content_parts = []
        
        # Add node's own information
        if data.get("type") == "entity":
            content_parts.append(f"Entity: {data.get('name', '')}")
            if data.get("context"):
                content_parts.append(f"Context: {data['context']}")
        
        # Add information from connected nodes
        for neighbor in graph.neighbors(node_id):
            neighbor_data = graph.nodes[neighbor]
            related_nodes.append(neighbor)
            
            if neighbor_data.get("type") == "text_unit":
                content_parts.append(f"Related text: {neighbor_data.get('content', '')}")
            elif neighbor_data.get("type") == "entity":
                content_parts.append(f"Related entity: {neighbor_data.get('name', '')}")
        
        return {
            "content": "\n".join(content_parts),
            "related_nodes": related_nodes,
            "context": {"node_type": data.get("type"), "node_name": data.get("name")}
        }
    
    def _traverse_relationships(self, graph: nx.Graph, start_node: str, query_analysis: QueryAnalysis) -> List[RetrievalResult]:
        """Traverse relationships from a starting node"""
        results = []
        visited = set()
        queue = [(start_node, 0)]  # (node, depth)
        
        while queue:
            current_node, depth = queue.pop(0)
            if current_node in visited or depth > self.reasoning_depth:
                continue
            
            visited.add(current_node)
            
            # Get information from current node
            related_info = self._gather_related_information(graph, current_node)
            
            result = RetrievalResult(
                content=related_info["content"],
                source_nodes=[current_node] + related_info["related_nodes"],
                confidence=query_analysis.confidence * (1.0 - depth * 0.1),  # Decrease with depth
                reasoning_path=[f"Traversal depth {depth} from {start_node}"],
                context=related_info["context"],
                metadata={"depth": depth, "start_node": start_node}
            )
            results.append(result)
            
            # Add neighbors to queue
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return results
    
    def _perform_multi_hop_reasoning(self, graph: nx.Graph, start_node: str, query_analysis: QueryAnalysis) -> List[RetrievalResult]:
        """Perform multi-hop reasoning from a starting node"""
        results = []
        
        # Use LLM for multi-hop reasoning
        reasoning_prompt = f"""
Perform multi-hop reasoning starting from this node to answer the query.

Starting Node: {start_node}
Node Data: {graph.nodes[start_node]}

Query: "{query_analysis.intent}"

Available Graph Structure:
{self._get_graph_summary(graph)}

Provide multi-hop reasoning in JSON format:
{{
    "reasoning_paths": [
        {{
            "path": ["node1", "node2", "node3"],
            "reasoning": "logical reasoning for this path",
            "conclusion": "conclusion from this path",
            "confidence": 0.0-1.0
        }}
    ],
    "final_answer": "synthesized answer from all paths",
    "confidence": 0.0-1.0
}}
"""
        
        response = self.llm_processor._call_llm(reasoning_prompt)
        reasoning_data = json.loads(response)
        
        # Create results from reasoning paths
        for path_info in reasoning_data.get("reasoning_paths", []):
            result = RetrievalResult(
                content=path_info["conclusion"],
                source_nodes=path_info["path"],
                confidence=path_info["confidence"],
                reasoning_path=[path_info["reasoning"]],
                context={"reasoning_type": "multi_hop"},
                metadata={"path": path_info["path"]}
            )
            results.append(result)
        
        return results
    
    def _get_graph_summary(self, graph: nx.Graph) -> str:
        """Get a summary of the graph structure"""
        entity_nodes = [n for n in graph.nodes() if n.startswith("entity_")]
        text_nodes = [n for n in graph.nodes() if n.startswith("text_")]
        
        summary = f"Graph has {len(entity_nodes)} entity nodes and {len(text_nodes)} text nodes.\n"
        
        # Add sample entities
        sample_entities = []
        for node in entity_nodes[:5]:  # First 5 entities
            data = graph.nodes[node]
            sample_entities.append(f"- {data.get('name', node)} ({data.get('category', 'unknown')})")
        
        if sample_entities:
            summary += "Sample entities:\n" + "\n".join(sample_entities)
        
        return summary
    
    def _generate_intelligent_response(self, query: str, query_analysis: QueryAnalysis, retrieval_results: List[RetrievalResult], user_intent: str = "") -> str:
        """Generate intelligent response based on retrieval results"""
        if not retrieval_results:
            return "I couldn't find relevant information to answer your query in the knowledge graph."
        
        # Prepare context for response generation
        context_parts = []
        for i, result in enumerate(retrieval_results[:5]):  # Top 5 results
            context_parts.append(f"{i+1}. {result.content}")
        
        context = "\n".join(context_parts)
        
        # Use LLM to generate intelligent response
        response_prompt = f"""
Generate an intelligent response to this query based on the retrieved information.

Query: "{query}"
Query Type: {query_analysis.query_type.value}
Query Intent: {query_analysis.intent}
User Intent: {user_intent if user_intent else "General information"}

Retrieved Information:
{context}

Query Analysis:
- Complexity: {query_analysis.complexity}
- Retrieval Strategy: {query_analysis.retrieval_strategy.value}
- Confidence: {query_analysis.confidence}

Provide a comprehensive, well-structured response that:
1. Directly answers the query
2. Provides relevant context and details
3. Explains the reasoning when appropriate
4. Cites sources when relevant
5. Adapts to the query type and complexity

Response:
"""
        
        response = self.llm_processor._call_llm(response_prompt)
        return response
    
    async def _learn_from_query_processing(self, query_analysis: QueryAnalysis, retrieval_results: List[RetrievalResult], query: str):
        """Learn from the query processing for future improvements"""
        if not self.learning_enabled:
            return
        
        # Record successful query pattern
        pattern = {
            "query_type": query_analysis.query_type.value,
            "retrieval_strategy": query_analysis.retrieval_strategy.value,
            "complexity": query_analysis.complexity,
            "success": len(retrieval_results) > 0,
            "confidence": query_analysis.confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        self.query_patterns.append(pattern)
        
        # Update successful retrievals
        if len(retrieval_results) > 0:
            self.successful_retrievals.append(pattern)
        
        # Update retrieval strategies
        strategy = query_analysis.retrieval_strategy.value
        if strategy not in self.retrieval_strategies:
            self.retrieval_strategies[strategy] = {"success": 0, "total": 0}
        
        self.retrieval_strategies[strategy]["total"] += 1
        if len(retrieval_results) > 0:
            self.retrieval_strategies[strategy]["success"] += 1
