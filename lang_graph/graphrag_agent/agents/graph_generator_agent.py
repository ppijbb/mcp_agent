"""
Graph Generator Agent

Knowledge Graph를 생성하는 전문 에이전트
"""

import asyncio
from typing import Dict, Any, Optional
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
from pydantic import BaseModel, Field
import logging

class GraphGeneratorConfig(BaseModel):
    """Configuration for Graph Generator Agent"""
    openai_api_key: str = Field(..., description="OpenAI API key")
    model_name: str = Field(default="gemini-2.5-flash-lite-preview-06-07", description="LLM model name")
    temperature: float = Field(default=0.0, description="LLM temperature")
    cache_file: str = Field(default="graph_cache.db", description="Cache file path")
    max_concurrency: int = Field(default=1, description="Max concurrency for processing")

class GraphGeneratorAgent:
    """Knowledge Graph 생성 전문 에이전트"""
    
    def __init__(self, config: GraphGeneratorConfig):
        self.config = config
        self._initialize_components()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize LLM and GraphRAG components"""
        cache = SQLiteCache(database_path=self.config.cache_file)
        
        er_llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            api_key=self.config.openai_api_key,
            cache=cache,
        )
        extractor = EntityRelationshipExtractor.build_default(llm=er_llm)

        es_llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            api_key=self.config.openai_api_key,
            cache=cache,
        )
        summarizer = EntityRelationshipDescriptionSummarizer.build_default(llm=es_llm)
        
        self.graph_generator = GraphGenerator(
            er_extractor=extractor,
            graphs_merger=GraphsMerger(),
            er_description_summarizer=summarizer,
        )
    
    async def process_text_units(self, text_units: pd.DataFrame) -> Dict[str, Any]:
        """
        Process text units and generate knowledge graph.
        This is a simplified, non-graph-based execution.
        """
        self.logger.info(f"Processing {len(text_units)} text units.")
        
        # 1. Validate Input
        if text_units.empty:
            return {"status": "error", "error": "No text units provided"}
        
        required_columns = ["id", "document_id", "text_unit"]
        if not all(col in text_units.columns for col in required_columns):
            return {"status": "error", "error": f"Missing required columns: {required_columns}"}
        
        # 2. Generate Graph
        try:
            self.logger.info("Invoking underlying GraphGenerator...")
            def run_sync():
                config = RunnableConfig(
                    configurable={},
                    max_concurrency=self.config.max_concurrency
                )
                return self.graph_generator.invoke(text_units, config)

            knowledge_graph = await asyncio.to_thread(run_sync)
            self.logger.info("Graph generation complete.")
            
        except Exception as e:
            self.logger.error(f"Error during graph generation: {e}")
            return {"status": "error", "error": f"Knowledge graph generation failed: {e}"}

        # 3. Validate Graph
        if not knowledge_graph or not knowledge_graph.nodes:
            return {"status": "error", "error": "Generated graph is empty or invalid"}

        # 4. Finalize and Return
        stats = {
            "nodes": len(knowledge_graph.nodes),
            "edges": len(knowledge_graph.edges),
            "entity_types": len(set(node.type for node in knowledge_graph.nodes if hasattr(node, 'type'))),
            "relationship_types": len(set(edge.type for edge in knowledge_graph.edges if hasattr(edge, 'type')))
        }
        
        return {
            "status": "completed",
            "knowledge_graph": knowledge_graph,
            "stats": stats
        }
