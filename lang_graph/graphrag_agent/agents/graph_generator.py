"""
LangGraph-optimized Graph Generator Node

This node handles knowledge graph generation from text data using LangGraph patterns.
"""

import pandas as pd
import networkx as nx
from typing import Dict, Any, List
from typing import Callable
from models.types import GraphRAGState, GraphStats
from config import AgentConfig


class GraphGeneratorNode:
    """LangGraph node for generating knowledge graphs from text data"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = None
    
    def __call__(self, state: GraphRAGState) -> GraphRAGState:
        """Generate knowledge graph from text units"""
        try:
            # Load data if not already loaded
            if not state.get("text_units"):
                state = self._load_data(state)
            
            if not state.get("text_units"):
                state["error"] = "No text units available for graph generation"
                state["status"] = "error"
                return state
            
            # Generate knowledge graph
            knowledge_graph = self._generate_graph(state["text_units"])
            
            # Calculate statistics
            stats = self._calculate_stats(knowledge_graph)
            
            # Update state
            state["knowledge_graph"] = knowledge_graph
            state["status"] = "completed"
            
            if self.logger:
                self.logger.info(f"Graph generated: {stats.nodes} nodes, {stats.edges} edges")
            
        except Exception as e:
            state["error"] = f"Graph generation failed: {str(e)}"
            state["status"] = "error"
            if self.logger:
                self.logger.error(f"Graph generation error: {e}")
        
        return state
    
    def _load_data(self, state: GraphRAGState) -> GraphRAGState:
        """Load data from file"""
        try:
            if not state.get("data_file"):
                state["error"] = "No data file specified"
                return state
            
            df = pd.read_csv(state["data_file"])
            
            # Validate required columns
            required_columns = ["id", "document_id", "text_unit"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                state["error"] = f"Missing required columns: {missing_columns}"
                return state
            
            # Convert to text units
            text_units = []
            for _, row in df.iterrows():
                if pd.notna(row['text_unit']):
                    text_units.append({
                        "id": row["id"],
                        "document_id": row["document_id"],
                        "text": row["text_unit"]
                    })
            
            state["text_units"] = text_units
            
        except Exception as e:
            state["error"] = f"Data loading failed: {str(e)}"
        
        return state
    
    def _generate_graph(self, text_units: List[Dict[str, Any]]) -> nx.Graph:
        """Generate knowledge graph from text units"""
        # Create empty graph
        graph = nx.Graph()
        
        # Add nodes for each text unit
        for unit in text_units:
            node_id = f"text_{unit['id']}"
            graph.add_node(node_id, **{
                "type": "text_unit",
                "content": unit["text"],
                "document_id": unit["document_id"],
                "original_id": unit["id"]
            })
        
        # Simple entity extraction and relationship creation
        # In a real implementation, this would use LLM for entity extraction
        entities = self._extract_entities(text_units)
        
        # Add entity nodes
        for entity in entities:
            entity_id = f"entity_{entity['name'].replace(' ', '_').lower()}"
            if not graph.has_node(entity_id):
                graph.add_node(entity_id, **{
                    "type": "entity",
                    "name": entity["name"],
                    "category": entity["category"]
                })
        
        # Add relationships
        for unit in text_units:
            unit_id = f"text_{unit['id']}"
            for entity in entities:
                if entity["name"].lower() in unit["text"].lower():
                    entity_id = f"entity_{entity['name'].replace(' ', '_').lower()}"
                    graph.add_edge(unit_id, entity_id, relationship="mentions")
        
        return graph
    
    def _extract_entities(self, text_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract entities from text units (simplified version)"""
        entities = []
        entity_names = set()
        
        # Simple keyword-based entity extraction
        keywords = [
            "Apple", "Microsoft", "Google", "CEO", "company", "technology",
            "iPhone", "iPad", "Windows", "Azure", "founded", "headquartered"
        ]
        
        for unit in text_units:
            text = unit["text"]
            for keyword in keywords:
                if keyword.lower() in text.lower() and keyword not in entity_names:
                    entities.append({
                        "name": keyword,
                        "category": "organization" if keyword in ["Apple", "Microsoft", "Google"] else "concept"
                    })
                    entity_names.add(keyword)
        
        return entities
    
    def _calculate_stats(self, graph: nx.Graph) -> GraphStats:
        """Calculate graph statistics"""
        nodes = graph.number_of_nodes()
        edges = graph.number_of_edges()
        
        # Get entity types
        entity_types = list(set(
            data.get("type", "unknown") 
            for _, data in graph.nodes(data=True)
        ))
        
        # Get relationship types
        relationship_types = list(set(
            data.get("relationship", "unknown")
            for _, _, data in graph.edges(data=True)
        ))
        
        # Calculate density
        density = nx.density(graph) if nodes > 1 else 0.0
        
        # Calculate clustering coefficient
        clustering_coeff = nx.average_clustering(graph) if nodes > 2 else 0.0
        
        return GraphStats(
            nodes=nodes,
            edges=edges,
            entity_types=entity_types,
            relationship_types=relationship_types,
            density=density,
            clustering_coefficient=clustering_coeff
        )
