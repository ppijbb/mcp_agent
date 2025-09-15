"""
LangGraph-optimized RAG Agent Node
"""

import networkx as nx
from typing import Dict, Any, List
from typing import Callable
from models.types import GraphRAGState
from config import AgentConfig


class RAGAgentNode:
    """LangGraph node for RAG-based querying of knowledge graphs"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = None
    
    def __call__(self, state: GraphRAGState) -> GraphRAGState:
        """Process query using RAG techniques"""
        try:
            if not state.get("query"):
                state["error"] = "No query provided"
                state["status"] = "error"
                return state
            
            if not state.get("knowledge_graph"):
                state["error"] = "No knowledge graph available"
                state["status"] = "error"
                return state
            
            # Perform search
            search_results = self._search_graph(
                state["knowledge_graph"], 
                state["query"]
            )
            
            # Generate response
            response = self._generate_response(
                state["query"],
                search_results
            )
            
            # Update state
            state["search_results"] = search_results
            state["response"] = response
            state["status"] = "completed"
            
            if self.logger:
                self.logger.info(f"Query processed: {len(search_results)} results found")
            
        except Exception as e:
            state["error"] = f"RAG processing failed: {str(e)}"
            state["status"] = "error"
            if self.logger:
                self.logger.error(f"RAG processing error: {e}")
        
        return state
    
    def _search_graph(self, graph: nx.Graph, query: str) -> List[Dict[str, Any]]:
        """Search the knowledge graph for relevant information"""
        results = []
        query_lower = query.lower()
        
        # Search through all nodes
        for node_id, data in graph.nodes(data=True):
            content = data.get("content", "")
            if content and query_lower in content.lower():
                score = self._calculate_relevance_score(query, content)
                results.append({
                    "node_id": node_id,
                    "content": content,
                    "score": score,
                    "metadata": data
                })
        
        # Sort by relevance score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:self.config.max_search_results]
    
    def _calculate_relevance_score(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        overlap = len(query_words.intersection(content_words))
        total_words = len(query_words.union(content_words))
        return overlap / total_words if total_words > 0 else 0.0
    
    def _generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate response based on search results"""
        if not search_results:
            return "I couldn't find relevant information to answer your query."
        
        context_parts = [f"{i+1}. {result['content']}" for i, result in enumerate(search_results)]
        context = "\n".join(context_parts)
        
        return f"""Based on the available information:

{context}

This information was retrieved from the knowledge graph for your question: "{query}"
"""
