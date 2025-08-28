"""
Graph Generator Agent

Knowledge Graph를 생성하는 전문 에이전트
"""

import asyncio
from typing import Dict, Any, Optional, List
import pandas as pd
from langchain_core.runnables import RunnableConfig
from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI
from langchain_graphrag.indexing.graph_generation import (
    EntityRelationshipExtractor,
    GraphsMerger,
    EntityRelationshipDescriptionSummarizer,
    GraphGenerator,
)
from pydantic import BaseModel, Field, validator
import logging
from pathlib import Path


class GraphGeneratorConfig(BaseModel):
    """Configuration for Graph Generator Agent"""
    openai_api_key: str = Field(..., description="OpenAI API key")
    model_name: str = Field(default="gemini-2.5-flash-lite-preview-06-07", description="LLM model name")
    temperature: float = Field(default=0.0, description="LLM temperature")
    cache_file: str = Field(default="graph_cache.db", description="Cache file path")
    max_concurrency: int = Field(default=1, description="Max concurrency for processing")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError('temperature must be between 0.0 and 2.0')
        return v
    
    @validator('max_concurrency')
    def validate_max_concurrency(cls, v):
        if v < 1 or v > 10:
            raise ValueError('max_concurrency must be between 1 and 10')
        return v


class GraphGeneratorAgent:
    """Knowledge Graph 생성 전문 에이전트"""
    
    def __init__(self, config: GraphGeneratorConfig):
        self.config = config
        self._setup_logging()
        self._initialize_components()
        self.logger.info("GraphGeneratorAgent initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize LLM and GraphRAG components"""
        try:
            # Setup cache directory
            cache_dir = Path(self.config.cache_file).parent
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            cache = SQLiteCache(database_path=self.config.cache_file)
            
            # Initialize Entity Relationship Extractor
            er_llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                api_key=self.config.openai_api_key,
                cache=cache,
                max_retries=3,
                timeout=60
            )
            extractor = EntityRelationshipExtractor.build_default(llm=er_llm)
            self.logger.info("EntityRelationshipExtractor initialized")

            # Initialize Entity Relationship Description Summarizer
            es_llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                api_key=self.config.openai_api_key,
                cache=cache,
                max_retries=3,
                timeout=60
            )
            summarizer = EntityRelationshipDescriptionSummarizer.build_default(llm=es_llm)
            self.logger.info("EntityRelationshipDescriptionSummarizer initialized")
            
            # Initialize Graph Generator
            self.graph_generator = GraphGenerator(
                er_extractor=extractor,
                graphs_merger=GraphsMerger(),
                er_description_summarizer=summarizer,
            )
            self.logger.info("GraphGenerator initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise RuntimeError(f"Component initialization failed: {e}")
    
    async def process_text_units(self, text_units: pd.DataFrame) -> Dict[str, Any]:
        """
        Process text units and generate knowledge graph.
        This is a simplified, non-graph-based execution.
        
        Args:
            text_units: DataFrame containing text units with columns: id, document_id, text_unit
            
        Returns:
            Dict containing status and knowledge graph
        """
        self.logger.info(f"Processing {len(text_units)} text units")
        
        # 1. Validate Input
        validation_result = self._validate_input(text_units)
        if validation_result["status"] != "valid":
            return validation_result
        
        # 2. Preprocess Data
        try:
            processed_data = self._preprocess_data(text_units)
            self.logger.info(f"Data preprocessing completed: {len(processed_data)} valid units")
        except Exception as e:
            error_msg = f"Data preprocessing failed: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg}
        
        # 3. Generate Graph
        try:
            self.logger.info("Invoking underlying GraphGenerator...")
            knowledge_graph = await self._generate_graph(processed_data)
            
            if not knowledge_graph:
                return {"status": "error", "error": "Graph generation returned empty result"}
                
            self.logger.info("Graph generation completed successfully")
            
        except Exception as e:
            error_msg = f"Error during graph generation: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg}

        # 4. Validate Generated Graph
        graph_validation = self._validate_generated_graph(knowledge_graph)
        if graph_validation["status"] != "valid":
            return graph_validation

        # 5. Finalize and Return
        stats = self._calculate_graph_stats(knowledge_graph)
        
        return {
            "status": "completed",
            "knowledge_graph": knowledge_graph,
            "stats": stats,
            "processing_info": {
                "input_units": len(text_units),
                "processed_units": len(processed_data),
                "model_used": self.config.model_name
            }
        }
    
    def _validate_input(self, text_units: pd.DataFrame) -> Dict[str, Any]:
        """Validate input data"""
        if text_units.empty:
            return {"status": "error", "error": "No text units provided"}
        
        required_columns = ["id", "document_id", "text_unit"]
        missing_columns = [col for col in required_columns if col not in text_units.columns]
        if missing_columns:
            return {"status": "error", "error": f"Missing required columns: {missing_columns}"}
        
        # Check for empty text units
        empty_texts = text_units['text_unit'].isna().sum()
        if empty_texts > 0:
            self.logger.warning(f"Found {empty_texts} empty text units")
        
        return {"status": "valid"}
    
    def _preprocess_data(self, text_units: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and clean the input data"""
        # Remove rows with empty text units
        cleaned_data = text_units.dropna(subset=['text_unit'])
        
        # Remove duplicate text units
        cleaned_data = cleaned_data.drop_duplicates(subset=['text_unit'])
        
        # Ensure text units are strings
        cleaned_data['text_unit'] = cleaned_data['text_unit'].astype(str)
        
        # Filter out very short text units (less than 10 characters)
        cleaned_data = cleaned_data[cleaned_data['text_unit'].str.len() >= 10]
        
        return cleaned_data
    
    async def _generate_graph(self, processed_data: pd.DataFrame):
        """Generate the knowledge graph using the underlying GraphGenerator"""
        def run_sync():
            config = RunnableConfig(
                configurable={},
                max_concurrency=self.config.max_concurrency
            )
            return self.graph_generator.invoke(processed_data, config)

        return await asyncio.to_thread(run_sync)
    
    def _validate_generated_graph(self, knowledge_graph: Any) -> Dict[str, Any]:
        """Validate the generated knowledge graph"""
        if not knowledge_graph:
            return {"status": "error", "error": "Generated graph is None"}
        
        if not hasattr(knowledge_graph, 'nodes'):
            return {"status": "error", "error": "Generated graph has no nodes attribute"}
        
        if not knowledge_graph.nodes:
            return {"status": "error", "error": "Generated graph has no nodes"}
        
        if not hasattr(knowledge_graph, 'edges'):
            return {"status": "error", "error": "Generated graph has no edges attribute"}
        
        return {"status": "valid"}
    
    def _calculate_graph_stats(self, knowledge_graph: Any) -> Dict[str, Any]:
        """Calculate statistics for the generated graph"""
        try:
            stats = {
                "nodes": len(knowledge_graph.nodes),
                "edges": len(knowledge_graph.edges),
            }
            
            # Calculate entity types if available
            if hasattr(knowledge_graph.nodes[0], 'type'):
                entity_types = set(node.type for node in knowledge_graph.nodes if hasattr(node, 'type'))
                stats["entity_types"] = len(entity_types)
                stats["entity_type_list"] = list(entity_types)
            
            # Calculate relationship types if available
            if hasattr(knowledge_graph.edges[0], 'type'):
                relationship_types = set(edge.type for edge in knowledge_graph.edges if hasattr(edge, 'type'))
                stats["relationship_types"] = len(relationship_types)
                stats["relationship_type_list"] = list(relationship_types)
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Could not calculate detailed stats: {e}")
            return {
                "nodes": len(knowledge_graph.nodes),
                "edges": len(knowledge_graph.edges),
                "note": "Limited statistics available"
            }
    
    async def test_connectivity(self) -> bool:
        """Test if the agent can connect to the LLM service"""
        try:
            # Simple test to verify API connectivity
            test_llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=0.0,
                api_key=self.config.openai_api_key,
                max_retries=1,
                timeout=10
            )
            
            # Try a simple completion
            response = await test_llm.ainvoke("Hello")
            return bool(response and response.content)
            
        except Exception as e:
            self.logger.error(f"Connectivity test failed: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent configuration"""
        return {
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_concurrency": self.config.max_concurrency,
            "cache_file": self.config.cache_file,
            "status": "ready"
        }
