"""
LLM-powered Dynamic Graph Generator Node

This node handles knowledge graph generation from text data using LLM-based processing
without any hardcoded patterns or keywords.
"""

import pandas as pd
import networkx as nx
from typing import Dict, Any, List
from typing import Callable
from models.types import GraphRAGState, GraphStats
from config import AgentConfig
from .llm_processor import LLMProcessor, Entity, Relationship


class GraphGeneratorNode:
    """LLM-powered node for generating knowledge graphs from text data"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = None
        self.llm_processor = LLMProcessor(config)
    
    def __call__(self, state: GraphRAGState) -> GraphRAGState:
        """Generate knowledge graph from text units using LLM"""
        try:
            # Load data if not already loaded
            if not state.get("text_units"):
                state = self._load_data(state)
            
            if not state.get("text_units"):
                state["error"] = "No text units available for graph generation"
                state["status"] = "error"
                return state
            
            # Get user intent if available
            user_intent = state.get("user_intent", "")
            
            # Generate knowledge graph using LLM
            knowledge_graph = self._generate_graph_with_llm(state["text_units"], user_intent)
            
            # Calculate statistics
            stats = self._calculate_stats(knowledge_graph)
            
            # Update state
            state["knowledge_graph"] = knowledge_graph
            state["status"] = "completed"
            
            if self.logger:
                self.logger.info(f"Graph generated with LLM: {stats.nodes} nodes, {stats.edges} edges")
            
        except Exception as e:
            state["error"] = f"Graph generation failed: {str(e)}"
            state["status"] = "error"
            if self.logger:
                self.logger.error(f"Graph generation error: {e}")
        
        return state
    
    def _load_data(self, state: GraphRAGState) -> GraphRAGState:
        """Load data from file with flexible column mapping"""
        try:
            if not state.get("data_file"):
                state["error"] = "No data file specified"
                return state
            
            df = pd.read_csv(state["data_file"])
            
            # Auto-detect column mapping
            column_mapping = self._detect_column_mapping(df.columns)
            
            if not column_mapping:
                state["error"] = "Could not detect required columns in CSV file"
                return state
            
            # Convert to text units
            text_units = []
            for idx, row in df.iterrows():
                text_content = None
                
                # Try different text columns
                for text_col in column_mapping.get('text_columns', []):
                    if pd.notna(row[text_col]) and str(row[text_col]).strip():
                        text_content = str(row[text_col]).strip()
                        break
                
                if text_content:
                    text_units.append({
                        "id": row.get(column_mapping.get('id_column', 'id'), idx),
                        "document_id": row.get(column_mapping.get('document_id_column', 'document_id'), f"doc_{idx}"),
                        "text": text_content,
                        "metadata": self._extract_metadata(row, column_mapping)
                    })
            
            if not text_units:
                state["error"] = "No valid text content found in CSV file"
                return state
            
            state["text_units"] = text_units
            state["column_mapping"] = column_mapping
            
        except Exception as e:
            state["error"] = f"Data loading failed: {str(e)}"
        
        return state
    
    def _detect_column_mapping(self, columns: List[str]) -> Dict[str, Any]:
        """Auto-detect column mapping for different CSV formats"""
        columns_lower = [col.lower() for col in columns]
        mapping = {}
        
        # Detect ID column
        id_candidates = ['id', 'index', 'idx', 'key', 'pk']
        for candidate in id_candidates:
            if candidate in columns_lower:
                mapping['id_column'] = columns[columns_lower.index(candidate)]
                break
        
        # Detect document ID column
        doc_id_candidates = ['document_id', 'doc_id', 'document', 'doc', 'file_id', 'source_id']
        for candidate in doc_id_candidates:
            if candidate in columns_lower:
                mapping['document_id_column'] = columns[columns_lower.index(candidate)]
                break
        
        # Detect text columns (multiple possible names)
        text_candidates = [
            'text', 'content', 'text_unit', 'sentence', 'paragraph', 'description',
            'summary', 'abstract', 'body', 'message', 'comment', 'note'
        ]
        text_columns = []
        for candidate in text_candidates:
            if candidate in columns_lower:
                text_columns.append(columns[columns_lower.index(candidate)])
        
        if text_columns:
            mapping['text_columns'] = text_columns
        else:
            # If no standard text columns found, try to find any column with substantial text
            for col in columns:
                if col.lower() not in ['id', 'index', 'idx', 'key', 'pk', 'document_id', 'doc_id']:
                    text_columns.append(col)
            if text_columns:
                mapping['text_columns'] = text_columns
        
        # Detect metadata columns
        metadata_columns = []
        for col in columns:
            if col.lower() not in [mapping.get('id_column', '').lower(), 
                                 mapping.get('document_id_column', '').lower()] and col not in text_columns:
                metadata_columns.append(col)
        mapping['metadata_columns'] = metadata_columns
        
        return mapping if mapping.get('text_columns') else None
    
    def _extract_metadata(self, row: pd.Series, column_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from row based on column mapping"""
        metadata = {}
        
        for col in column_mapping.get('metadata_columns', []):
            if pd.notna(row[col]):
                metadata[col] = row[col]
        
        return metadata
    
    def _generate_graph_with_llm(self, text_units: List[Dict[str, Any]], user_intent: str = "") -> nx.Graph:
        """Generate knowledge graph from text units using LLM"""
        # Create empty graph
        graph = nx.Graph()
        
        # Combine all text for analysis
        combined_text = " ".join([unit["text"] for unit in text_units])
        
        # Extract entities using LLM with batch processing for better performance
        if len(text_units) > 1:
            # Use batch processing for multiple text units
            texts = [unit["text"] for unit in text_units]
            entities_list = self.llm_processor.extract_entities_batch(texts, user_intent)
            # Flatten entities from all texts
            entities = []
            for entity_list in entities_list:
                entities.extend(entity_list)
        else:
            # Single text processing
            entities = self.llm_processor.extract_entities(combined_text, user_intent)
        
        # Classify entities using LLM
        entities = self.llm_processor.classify_entities(entities, combined_text)
        
        # Extract relationships using LLM
        relationships = self.llm_processor.extract_relationships(combined_text, entities, user_intent)
        
        # Add text unit nodes
        for unit in text_units:
            node_id = f"text_{unit['id']}"
            graph.add_node(node_id, **{
                "type": "text_unit",
                "content": unit["text"],
                "document_id": unit["document_id"],
                "original_id": unit["id"]
            })
        
        # Add entity nodes
        for entity in entities:
            entity_id = f"entity_{entity.name.replace(' ', '_').lower()}"
            if not graph.has_node(entity_id):
                graph.add_node(entity_id, **{
                    "type": "entity",
                    "name": entity.name,
                    "category": entity.category,
                    "confidence": entity.confidence,
                    "context": entity.context,
                    "attributes": entity.attributes or {}
                })
        
        # Add relationships
        for relationship in relationships:
            source_id = f"entity_{relationship.source.replace(' ', '_').lower()}"
            target_id = f"entity_{relationship.target.replace(' ', '_').lower()}"
            
            if graph.has_node(source_id) and graph.has_node(target_id):
                graph.add_edge(source_id, target_id, **{
                    "relationship_type": relationship.relationship_type,
                    "confidence": relationship.confidence,
                    "context": relationship.context,
                    "attributes": relationship.attributes or {}
                })
        
        # Add entity-text relationships
        for unit in text_units:
            unit_id = f"text_{unit['id']}"
            text = unit["text"].lower()
            
            for entity in entities:
                if entity.name.lower() in text:
                    entity_id = f"entity_{entity.name.replace(' ', '_').lower()}"
                    if graph.has_node(entity_id):
                        graph.add_edge(unit_id, entity_id, **{
                            "relationship_type": "mentions",
                            "confidence": entity.confidence,
                            "context": entity.context
                        })
        
        return graph
    
    
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
