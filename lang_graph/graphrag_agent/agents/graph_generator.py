"""
True GraphRAG Dynamic Graph Generator Node

This node implements genuine GraphRAG capabilities:
- Autonomous data understanding and analysis
- Dynamic graph structure generation based on data patterns
- Intelligent entity and relationship discovery
- Context-aware graph construction
- Self-directed learning and adaptation
"""

import pandas as pd
import networkx as nx
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from typing import Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from models.types import GraphRAGState, GraphStats
from config import AgentConfig
from .llm_processor import LLMProcessor, Entity, Relationship


class GraphStructureType(Enum):
    """Types of graph structures that can be generated"""
    HIERARCHICAL = "hierarchical"
    NETWORK = "network"
    TIMELINE = "timeline"
    TAXONOMY = "taxonomy"
    KNOWLEDGE_WEB = "knowledge_web"
    CONCEPT_MAP = "concept_map"
    RELATIONSHIP_NETWORK = "relationship_network"


@dataclass
class GraphConstructionPlan:
    """Plan for constructing a knowledge graph"""
    structure_type: GraphStructureType
    main_entities: List[str]
    entity_categories: Dict[str, List[str]]
    relationship_hierarchy: List[Dict[str, Any]]
    focus_areas: List[str]
    visualization_strategy: str
    confidence: float
    reasoning: str


class GraphGeneratorNode:
    """
    True GraphRAG Dynamic Graph Generator
    
    This node embodies genuine GraphRAG principles:
    - Autonomous data analysis and understanding
    - Dynamic graph structure generation
    - Intelligent entity and relationship discovery
    - Context-aware graph construction
    - Self-directed learning and adaptation
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm_processor = LLMProcessor(config)
        
        # GraphRAG learning state
        self.construction_patterns = []
        self.domain_knowledge = {}
        self.successful_structures = []
        self.adaptation_history = []
        
        # Autonomous capabilities
        self.learning_enabled = True
        self.adaptation_threshold = 0.7
        self.quality_threshold = 0.6
    
    def __call__(self, state: GraphRAGState) -> GraphRAGState:
        """
        Generate knowledge graph using true GraphRAG principles
        
        This method implements autonomous graph generation:
        1. Autonomous data analysis and understanding
        2. Dynamic graph structure planning
        3. Intelligent entity and relationship discovery
        4. Context-aware graph construction
        5. Learning from the process
        """
        try:
            # Step 1: Autonomous data analysis
            if not state.get("text_units"):
                state = self._autonomous_data_analysis(state)
            
            if not state.get("text_units"):
                state["error"] = "No text units available for graph generation"
                state["status"] = "error"
                return state
            
            # Step 2: Dynamic graph structure planning
            user_intent = state.get("user_intent", "")
            construction_plan = self._create_dynamic_construction_plan(state["text_units"], user_intent)
            
            # Step 3: Intelligent graph generation
            knowledge_graph = self._generate_intelligent_graph(state["text_units"], construction_plan, user_intent)
            
            # Step 4: Graph optimization and validation
            optimized_graph = self._optimize_and_validate_graph(knowledge_graph, construction_plan)
            
            # Step 5: Learning from the process
            self._learn_from_graph_construction(construction_plan, optimized_graph, user_intent)
            
            # Calculate statistics
            stats = self._calculate_advanced_stats(optimized_graph)
            
            # Update state
            state["knowledge_graph"] = optimized_graph
            state["construction_plan"] = construction_plan
            state["graph_stats"] = stats
            state["status"] = "completed"
            
            if self.logger:
                self.logger.info(f"GraphRAG graph generated: {stats['nodes']} nodes, {stats['edges']} edges, structure: {construction_plan.structure_type.value}")
            
        except Exception as e:
            state["error"] = f"GraphRAG generation failed: {str(e)}"
            state["status"] = "error"
            if self.logger:
                self.logger.error(f"GraphRAG generation error: {e}")
        
        return state
    
    def _autonomous_data_analysis(self, state: GraphRAGState) -> GraphRAGState:
        """
        Autonomous data analysis and understanding
        
        This method implements the core GraphRAG principle of autonomous data understanding:
        - Analyzes data structure and content intelligently
        - Infers data patterns and relationships
        - Determines optimal processing strategy
        - Learns from data characteristics
        """
        try:
            if not state.get("data_file"):
                state["error"] = "No data file specified"
                return state
            
            # Load and analyze data autonomously
            df = pd.read_csv(state["data_file"])
            
            # Use LLM to understand data structure and content
            data_analysis = self._analyze_data_structure_autonomously(df, state.get("user_intent", ""))
            
            # Convert to text units based on analysis
            text_units = self._create_intelligent_text_units(df, data_analysis)
            
            if not text_units:
                state["error"] = "No valid text content found after autonomous analysis"
                return state
            
            state["text_units"] = text_units
            state["data_analysis"] = data_analysis
            state["column_mapping"] = data_analysis.get("column_mapping", {})
            
        except Exception as e:
            state["error"] = f"Autonomous data analysis failed: {str(e)}"
        
        return state
    
    def _analyze_data_structure_autonomously(self, df: pd.DataFrame, user_intent: str = "") -> Dict[str, Any]:
        """Use LLM to autonomously analyze data structure and content"""
        # Prepare data sample for analysis
        sample_data = df.head(10).to_string()
        column_info = {
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "unique_counts": df.nunique().to_dict()
        }
        
        analysis_prompt = f"""
You are an expert data analyst with GraphRAG capabilities. Analyze this dataset autonomously to understand its structure, content, and potential for knowledge graph generation.

Dataset Information:
- Shape: {df.shape}
- Columns: {df.columns.tolist()}
- Column Types: {df.dtypes.astype(str).to_dict()}

Sample Data:
{sample_data}

User Intent: {user_intent if user_intent else "General knowledge graph generation"}

Provide comprehensive analysis in JSON format:
{{
    "data_understanding": {{
        "primary_purpose": "What this dataset represents",
        "domain_context": "Domain or field this data belongs to",
        "key_entities": ["main entities that could be graph nodes"],
        "data_patterns": ["patterns observed in the data"],
        "complexity_level": "simple|medium|complex"
    }},
    "column_analysis": {{
        "text_columns": ["columns containing text content"],
        "entity_columns": ["columns containing entity names"],
        "relationship_columns": ["columns indicating relationships"],
        "metadata_columns": ["columns with metadata"],
        "id_columns": ["columns that could serve as identifiers"]
    }},
    "graph_potential": {{
        "entity_candidates": ["entities that should be graph nodes"],
        "relationship_opportunities": ["potential relationships to extract"],
        "graph_structure_suggestion": "hierarchical|network|timeline|taxonomy|knowledge_web",
        "visualization_approach": "how to visualize this data as a graph"
    }},
    "processing_strategy": {{
        "text_extraction_method": "how to extract text from this data",
        "entity_extraction_approach": "strategy for finding entities",
        "relationship_detection_strategy": "how to find relationships",
        "quality_considerations": ["data quality issues to address"]
    }},
    "column_mapping": {{
        "id_column": "column to use as ID",
        "text_column": "column containing main text content",
        "entity_columns": ["columns containing entities"],
        "metadata_columns": ["columns with additional metadata"]
    }},
    "confidence": 0.0-1.0,
    "recommendations": ["specific recommendations for graph generation"]
}}

Consider:
- The data's potential for knowledge graph generation
- Relationships and connections between entities
- Data quality and completeness
- Domain-specific patterns and insights
- User intent and requirements
- GraphRAG best practices
"""
        
        response = self.llm_processor._call_llm(analysis_prompt)
        return json.loads(response)
    
    def _create_intelligent_text_units(self, df: pd.DataFrame, data_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create text units based on intelligent data analysis"""
        text_units = []
        column_mapping = data_analysis.get("column_mapping", {})
        
        # Determine text extraction strategy
        text_columns = column_mapping.get("text_columns", [])
        if not text_columns:
            # Fallback to auto-detection
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
        
        for idx, row in df.iterrows():
            # Extract text content intelligently
            text_content = None
            for text_col in text_columns:
                if pd.notna(row[text_col]) and str(row[text_col]).strip():
                    text_content = str(row[text_col]).strip()
                    break
            
            if text_content:
                # Extract metadata intelligently
                metadata = {}
                for col in column_mapping.get("metadata_columns", []):
                    if pd.notna(row[col]):
                        metadata[col] = row[col]
                
                text_units.append({
                    "id": row.get(column_mapping.get("id_column", "id"), idx),
                    "document_id": f"doc_{idx}",
                    "text": text_content,
                    "metadata": metadata,
                    "source_column": text_col if 'text_col' in locals() else None
                })
        
        return text_units
    
    def _create_dynamic_construction_plan(self, text_units: List[Dict[str, Any]], user_intent: str = "") -> GraphConstructionPlan:
        """
        Create dynamic graph construction plan based on data analysis
        
        This method implements the core GraphRAG principle of autonomous planning:
        - Analyzes data patterns to determine optimal graph structure
        - Plans entity extraction and relationship discovery strategy
        - Adapts to user intent and domain context
        - Learns from previous successful constructions
        """
        # Combine all text for analysis
        combined_text = " ".join([unit["text"] for unit in text_units])
        
        # Use LLM to create construction plan
        planning_prompt = f"""
You are an expert at designing knowledge graph construction plans. Based on the data and user intent, create an optimal plan for building a knowledge graph.

Data Analysis:
- Number of text units: {len(text_units)}
- Sample text: {combined_text[:1000]}...
- User Intent: {user_intent if user_intent else "General knowledge graph generation"}

Previous Successful Patterns: {json.dumps(self.successful_structures[-3:], indent=2) if self.successful_structures else "None"}

Create a comprehensive construction plan in JSON format:
{{
    "structure_type": "hierarchical|network|timeline|taxonomy|knowledge_web|concept_map|relationship_network",
    "main_entities": ["primary entities that should be central to the graph"],
    "entity_categories": {{
        "person": ["person entities"],
        "organization": ["organization entities"],
        "concept": ["conceptual entities"],
        "location": ["location entities"],
        "event": ["event entities"]
    }},
    "relationship_hierarchy": [
        {{
            "level": 1,
            "relationship_type": "primary relationship type",
            "description": "description of this relationship level",
            "examples": ["example relationships"]
        }}
    ],
    "focus_areas": ["key areas to focus on in the graph"],
    "visualization_strategy": "strategy for visualizing this graph",
    "confidence": 0.0-1.0,
    "reasoning": "explanation for why this structure is optimal",
    "extraction_strategy": {{
        "entity_extraction_approach": "strategy for extracting entities",
        "relationship_detection_method": "method for finding relationships",
        "quality_validation": "how to validate graph quality"
    }},
    "adaptation_notes": ["how to adapt based on findings"]
}}

Consider:
- The data's domain and context
- User intent and requirements
- Optimal graph structure for the data type
- Previous successful patterns
- GraphRAG best practices
- Visualization and usability
"""
        
        response = self.llm_processor._call_llm(planning_prompt)
        plan_data = json.loads(response)
        
        # Convert to GraphConstructionPlan object
        structure_type = GraphStructureType(plan_data.get("structure_type", "network"))
        
        return GraphConstructionPlan(
            structure_type=structure_type,
            main_entities=plan_data.get("main_entities", []),
            entity_categories=plan_data.get("entity_categories", {}),
            relationship_hierarchy=plan_data.get("relationship_hierarchy", []),
            focus_areas=plan_data.get("focus_areas", []),
            visualization_strategy=plan_data.get("visualization_strategy", "network"),
            confidence=plan_data.get("confidence", 0.5),
            reasoning=plan_data.get("reasoning", "")
        )
    
    def _generate_intelligent_graph(self, text_units: List[Dict[str, Any]], construction_plan: GraphConstructionPlan, user_intent: str = "") -> nx.Graph:
        """
        Generate knowledge graph using intelligent, adaptive approach
        
        This method implements the core GraphRAG principle of intelligent graph construction:
        - Uses construction plan to guide graph building
        - Adapts entity extraction based on data patterns
        - Discovers relationships intelligently
        - Learns from the construction process
        """
        # Create empty graph
        graph = nx.Graph()
        
        # Add metadata about construction
        graph.graph["construction_plan"] = {
            "structure_type": construction_plan.structure_type.value,
            "main_entities": construction_plan.main_entities,
            "confidence": construction_plan.confidence,
            "reasoning": construction_plan.reasoning
        }
        
        # Step 1: Intelligent entity extraction
        entities = self._extract_entities_intelligently(text_units, construction_plan, user_intent)
        
        # Step 2: Dynamic relationship discovery
        relationships = self._discover_relationships_intelligently(text_units, entities, construction_plan, user_intent)
        
        # Step 3: Build graph structure
        graph = self._build_graph_structure(graph, text_units, entities, relationships, construction_plan)
        
        # Step 4: Apply structure-specific optimizations
        graph = self._apply_structure_optimizations(graph, construction_plan)
        
        return graph
    
    def _extract_entities_intelligently(self, text_units: List[Dict[str, Any]], construction_plan: GraphConstructionPlan, user_intent: str = "") -> List[Entity]:
        """Extract entities using intelligent, adaptive approach"""
        # Combine all text for analysis
        combined_text = " ".join([unit["text"] for unit in text_units])
        
        # Create intelligent extraction prompt
        extraction_prompt = f"""
You are an expert at extracting entities for knowledge graph construction. Extract entities based on the construction plan and data characteristics.

Construction Plan:
- Structure Type: {construction_plan.structure_type.value}
- Main Entities: {construction_plan.main_entities}
- Entity Categories: {construction_plan.entity_categories}
- Focus Areas: {construction_plan.focus_areas}

Data:
{combined_text}

User Intent: {user_intent if user_intent else "General knowledge graph generation"}

Extract entities intelligently and return in JSON format:
{{
    "entities": [
        {{
            "name": "entity_name",
            "category": "person|organization|location|time|event|concept|object|other",
            "confidence": 0.0-1.0,
            "context": "brief context where entity appears",
            "attributes": {{"key": "value"}},
            "importance": "high|medium|low",
            "relationships_hint": ["potential relationship types"]
        }}
    ]
}}

Guidelines:
- Focus on entities relevant to the construction plan
- Prioritize main entities and focus areas
- Consider the graph structure type
- Extract entities that can form meaningful relationships
- Adapt extraction strategy to data characteristics
- Use domain knowledge when available
"""
        
        response = self.llm_processor._call_llm(extraction_prompt)
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
    
    def _discover_relationships_intelligently(self, text_units: List[Dict[str, Any]], entities: List[Entity], construction_plan: GraphConstructionPlan, user_intent: str = "") -> List[Relationship]:
        """Discover relationships using intelligent, adaptive approach"""
        # Combine all text for analysis
        combined_text = " ".join([unit["text"] for unit in text_units])
        entity_names = [entity.name for entity in entities]
        
        # Create intelligent relationship discovery prompt
        relationship_prompt = f"""
You are an expert at discovering relationships for knowledge graph construction. Find relationships based on the construction plan and entity context.

Construction Plan:
- Structure Type: {construction_plan.structure_type.value}
- Relationship Hierarchy: {construction_plan.relationship_hierarchy}
- Focus Areas: {construction_plan.focus_areas}

Entities: {entity_names}

Data:
{combined_text}

User Intent: {user_intent if user_intent else "General knowledge graph generation"}

Discover relationships intelligently and return in JSON format:
{{
    "relationships": [
        {{
            "source": "entity1_name",
            "target": "entity2_name",
            "relationship_type": "descriptive_relationship_type",
            "confidence": 0.0-1.0,
            "context": "brief context of the relationship",
            "attributes": {{"key": "value"}},
            "strength": "strong|medium|weak",
            "direction": "directed|undirected"
        }}
    ]
}}

Guidelines:
- Focus on relationships that support the graph structure
- Consider the relationship hierarchy in the construction plan
- Prioritize relationships between main entities
- Look for both explicit and implicit relationships
- Consider relationship strength and direction
- Adapt to the data characteristics and domain
"""
        
        response = self.llm_processor._call_llm(relationship_prompt)
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
    
    def _build_graph_structure(self, graph: nx.Graph, text_units: List[Dict[str, Any]], entities: List[Entity], relationships: List[Relationship], construction_plan: GraphConstructionPlan) -> nx.Graph:
        """Build graph structure based on construction plan"""
        # Add text unit nodes
        for unit in text_units:
            node_id = f"text_{unit['id']}"
            graph.add_node(node_id, **{
                "type": "text_unit",
                "content": unit["text"],
                "document_id": unit["document_id"],
                "original_id": unit["id"],
                "metadata": unit.get("metadata", {})
            })
        
        # Add entity nodes with intelligent categorization
        for entity in entities:
            entity_id = f"entity_{entity.name.replace(' ', '_').lower()}"
            if not graph.has_node(entity_id):
                graph.add_node(entity_id, **{
                    "type": "entity",
                    "name": entity.name,
                    "category": entity.category,
                    "confidence": entity.confidence,
                    "context": entity.context,
                    "attributes": entity.attributes or {},
                    "importance": self._calculate_entity_importance(entity, construction_plan)
                })
        
        # Add relationships with intelligent weighting
        for relationship in relationships:
            source_id = f"entity_{relationship.source.replace(' ', '_').lower()}"
            target_id = f"entity_{relationship.target.replace(' ', '_').lower()}"
            
            if graph.has_node(source_id) and graph.has_node(target_id):
                graph.add_edge(source_id, target_id, **{
                    "relationship_type": relationship.relationship_type,
                    "confidence": relationship.confidence,
                    "context": relationship.context,
                    "attributes": relationship.attributes or {},
                    "weight": self._calculate_relationship_weight(relationship, construction_plan)
                })
        
        # Add entity-text relationships intelligently
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
                            "context": entity.context,
                            "weight": 1.0
                        })
        
        return graph
    
    def _apply_structure_optimizations(self, graph: nx.Graph, construction_plan: GraphConstructionPlan) -> nx.Graph:
        """Apply structure-specific optimizations"""
        if construction_plan.structure_type == GraphStructureType.HIERARCHICAL:
            graph = self._optimize_hierarchical_structure(graph)
        elif construction_plan.structure_type == GraphStructureType.NETWORK:
            graph = self._optimize_network_structure(graph)
        elif construction_plan.structure_type == GraphStructureType.TIMELINE:
            graph = self._optimize_timeline_structure(graph)
        elif construction_plan.structure_type == GraphStructureType.KNOWLEDGE_WEB:
            graph = self._optimize_knowledge_web_structure(graph)
        
        return graph
    
    def _optimize_hierarchical_structure(self, graph: nx.Graph) -> nx.Graph:
        """Optimize graph for hierarchical structure"""
        # Add hierarchical relationships
        # This would implement hierarchical optimization logic
        return graph
    
    def _optimize_network_structure(self, graph: nx.Graph) -> nx.Graph:
        """Optimize graph for network structure"""
        # Add network optimization logic
        return graph
    
    def _optimize_timeline_structure(self, graph: nx.Graph) -> nx.Graph:
        """Optimize graph for timeline structure"""
        # Add timeline optimization logic
        return graph
    
    def _optimize_knowledge_web_structure(self, graph: nx.Graph) -> nx.Graph:
        """Optimize graph for knowledge web structure"""
        # Add knowledge web optimization logic
        return graph
    
    def _calculate_entity_importance(self, entity: Entity, construction_plan: GraphConstructionPlan) -> float:
        """Calculate entity importance based on construction plan"""
        importance = entity.confidence
        
        # Boost importance if entity is in main entities
        if entity.name in construction_plan.main_entities:
            importance += 0.3
        
        # Boost importance if entity is in focus areas
        for focus_area in construction_plan.focus_areas:
            if focus_area.lower() in entity.name.lower() or focus_area.lower() in entity.context.lower():
                importance += 0.2
                break
        
        return min(1.0, importance)
    
    def _calculate_relationship_weight(self, relationship: Relationship, construction_plan: GraphConstructionPlan) -> float:
        """Calculate relationship weight based on construction plan"""
        weight = relationship.confidence
        
        # Boost weight for relationships in hierarchy
        for level in construction_plan.relationship_hierarchy:
            if relationship.relationship_type in level.get("examples", []):
                weight += 0.2
                break
        
        return min(1.0, weight)
    
    def _optimize_and_validate_graph(self, graph: nx.Graph, construction_plan: GraphConstructionPlan) -> nx.Graph:
        """Optimize and validate the generated graph"""
        # Remove isolated nodes (except text units)
        isolated_nodes = [node for node in graph.nodes() if graph.degree(node) == 0 and not node.startswith("text_")]
        graph.remove_nodes_from(isolated_nodes)
        
        # Validate graph quality
        quality_score = self._calculate_graph_quality(graph, construction_plan)
        graph.graph["quality_score"] = quality_score
        
        return graph
    
    def _calculate_graph_quality(self, graph: nx.Graph, construction_plan: GraphConstructionPlan) -> float:
        """Calculate overall graph quality"""
        if graph.number_of_nodes() == 0:
            return 0.0
        
        # Basic quality metrics
        density = nx.density(graph)
        clustering = nx.average_clustering(graph) if graph.number_of_nodes() > 2 else 0.0
        
        # Entity coverage
        entity_nodes = [n for n in graph.nodes() if n.startswith("entity_")]
        entity_coverage = len(entity_nodes) / max(len(construction_plan.main_entities), 1)
        
        # Relationship quality
        relationship_edges = [e for e in graph.edges() if not e[0].startswith("text_") and not e[1].startswith("text_")]
        relationship_quality = len(relationship_edges) / max(len(entity_nodes), 1)
        
        # Combine metrics
        quality_score = (density * 0.3 + clustering * 0.2 + entity_coverage * 0.3 + relationship_quality * 0.2)
        return min(1.0, quality_score)
    
    def _learn_from_graph_construction(self, construction_plan: GraphConstructionPlan, graph: nx.Graph, user_intent: str):
        """Learn from the graph construction process"""
        if not self.learning_enabled:
            return
        
        # Record successful construction pattern
        pattern = {
            "structure_type": construction_plan.structure_type.value,
            "entity_count": len([n for n in graph.nodes() if n.startswith("entity_")]),
            "relationship_count": len([e for e in graph.edges() if not e[0].startswith("text_")]),
            "quality_score": graph.graph.get("quality_score", 0.0),
            "user_intent": user_intent,
            "timestamp": datetime.now().isoformat()
        }
        
        self.construction_patterns.append(pattern)
        
        # Update successful structures
        if graph.graph.get("quality_score", 0.0) > self.quality_threshold:
            self.successful_structures.append(pattern)
        
        # Update domain knowledge
        if user_intent:
            domain = user_intent.split()[0] if user_intent else "general"
            if domain not in self.domain_knowledge:
                self.domain_knowledge[domain] = []
            self.domain_knowledge[domain].append({
                "structure_type": construction_plan.structure_type.value,
                "success_factors": pattern
            })
    
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
    
    def _calculate_advanced_stats(self, graph: nx.Graph) -> Dict[str, Any]:
        """Calculate advanced graph statistics for GraphRAG"""
        if graph.number_of_nodes() == 0:
            return {
                "nodes": 0,
                "edges": 0,
                "density": 0.0,
                "clustering_coefficient": 0.0,
                "quality_score": 0.0,
                "structure_type": "unknown"
            }
        
        # Basic metrics
        nodes = graph.number_of_nodes()
        edges = graph.number_of_edges()
        density = nx.density(graph)
        clustering_coeff = nx.average_clustering(graph) if nodes > 2 else 0.0
        
        # Entity and relationship metrics
        entity_nodes = [n for n in graph.nodes() if n.startswith("entity_")]
        text_nodes = [n for n in graph.nodes() if n.startswith("text_")]
        relationship_edges = [e for e in graph.edges() if not e[0].startswith("text_") and not e[1].startswith("text_")]
        
        # Quality score
        quality_score = graph.graph.get("quality_score", 0.0)
        
        # Structure type
        structure_type = graph.graph.get("construction_plan", {}).get("structure_type", "unknown")
        
        # Entity categories
        entity_categories = {}
        for node in entity_nodes:
            category = graph.nodes[node].get("category", "other")
            entity_categories[category] = entity_categories.get(category, 0) + 1
        
        # Relationship types
        relationship_types = {}
        for edge in relationship_edges:
            rel_type = graph.edges[edge].get("relationship_type", "unknown")
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        return {
            "nodes": nodes,
            "edges": edges,
            "entity_nodes": len(entity_nodes),
            "text_nodes": len(text_nodes),
            "relationship_edges": len(relationship_edges),
            "density": density,
            "clustering_coefficient": clustering_coeff,
            "quality_score": quality_score,
            "structure_type": structure_type,
            "entity_categories": entity_categories,
            "relationship_types": relationship_types,
            "average_degree": sum(dict(graph.degree()).values()) / nodes if nodes > 0 else 0.0
        }
    
