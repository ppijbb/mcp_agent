"""
Multi-Agent Coordinator

Graph Generator Agent와 RAG Agent를 조율하는 코디네이터
"""

import asyncio
import pickle
from typing import Dict, Any, Optional
import pandas as pd
import logging

from .agents.graph_generator_agent import GraphGeneratorAgent, GraphGeneratorConfig
from .agents.rag_agent import RAGAgent, RAGAgentConfig
from pydantic import BaseModel, Field


class MultiAgentConfig(BaseModel):
    """Configuration for Multi-Agent System"""
    openai_api_key: str = Field(..., description="OpenAI API key")
    graph_model_name: str = Field(default="gemini-2.5-flash-lite-preview-06-07", description="Model for graph generation")
    rag_model_name: str = Field(default="gemini-2.5-flash-lite-preview-06-07", description="Model for RAG responses")
    max_search_results: int = Field(default=5, description="Max search results for RAG")
    context_window_size: int = Field(default=4000, description="Context window size")


class MultiAgentCoordinator:
    """멀티 에이전트 시스템 코디네이터"""

    def __init__(self, config: MultiAgentConfig):
        """
        Initialize Multi-Agent Coordinator
        
        Args:
            config: MultiAgentConfig object with all necessary parameters
        """
        self.config = config
        self._initialize_agents()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _initialize_agents(self):
        """Initialize the specialized agents"""
        graph_config = GraphGeneratorConfig(
            openai_api_key=self.config.openai_api_key,
            model_name=self.config.graph_model_name,
        )
        self.graph_generator = GraphGeneratorAgent(graph_config)
        
        rag_config = RAGAgentConfig(
            openai_api_key=self.config.openai_api_key,
            model_name=self.config.rag_model_name,
            max_search_results=self.config.max_search_results,
            context_window_size=self.config.context_window_size
        )
        self.rag_agent = RAGAgent(rag_config)

    async def create_knowledge_graph(self, data_file_path: str, output_path: str) -> Dict[str, Any]:
        """
        Loads data, generates a knowledge graph, and saves it to a file.
        """
        self.logger.info(f"Starting knowledge graph creation from {data_file_path}")
        
        try:
            df = pd.read_csv(data_file_path)
            required_columns = ["id", "document_id", "text_unit"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Data file must contain columns: {required_columns}")
            self.logger.info(f"Data loaded successfully: {len(df)} rows.")
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            return {"status": "error", "error": f"Data loading failed: {e}"}

        try:
            self.logger.info("Delegating to GraphGeneratorAgent...")
            result = await self.graph_generator.process_text_units(df)
            
            if result["status"] != "completed":
                error_msg = result.get("error", "Unknown error during graph generation")
                self.logger.error(error_msg)
                return {"status": "error", "error": error_msg}
            
            knowledge_graph = result["knowledge_graph"]
            self.logger.info(f"Graph generated successfully: {result.get('stats')}")
        except Exception as e:
            self.logger.error(f"Graph generation failed: {e}")
            return {"status": "error", "error": f"Graph generation failed: {e}"}

        try:
            with open(output_path, "wb") as f:
                pickle.dump(knowledge_graph, f)
            self.logger.info(f"Knowledge graph saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save knowledge graph: {e}")
            return {"status": "error", "error": f"Failed to save knowledge graph: {e}"}

        return {
            "status": "completed",
            "graph_path": output_path,
            "stats": result.get("stats")
        }

    async def query_knowledge_graph(self, user_query: str, graph_path: str) -> Dict[str, Any]:
        """
        Loads a knowledge graph and queries it using the RAG agent.
        """
        self.logger.info(f"Loading knowledge graph from {graph_path} for query: '{user_query}'")

        try:
            with open(graph_path, "rb") as f:
                knowledge_graph = pickle.load(f)
            self.logger.info("Knowledge graph loaded successfully.")
        except FileNotFoundError:
            error_msg = f"Knowledge graph file not found: {graph_path}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg, "response": "Could not find the necessary knowledge graph to answer."}
        except Exception as e:
            error_msg = f"Failed to load knowledge graph: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg, "response": "Failed to load the knowledge graph."}

        try:
            self.logger.info("Delegating to RAGAgent...")
            result = await self.rag_agent.query_knowledge_graph(
                user_query=user_query,
                knowledge_graph=knowledge_graph
            )
            self.logger.info("RAGAgent finished processing.")
            return result
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return {"status": "error", "error": f"Query processing failed: {e}", "response": "An error occurred while answering the question."}

    def get_agent_status(self) -> Dict[str, str]:
        """Get status of all agents"""
        return {
            "graph_generator": "ready",
            "rag_agent": "ready",
            "coordinator": "ready"
        }
