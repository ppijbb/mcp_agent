import asyncio
from qdrant_client import QdrantClient
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import get_settings
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from dataclasses import dataclass
from typing import Optional, Type, TypeVar, List, Dict, Any
import json
from datetime import datetime

SAMPLE_TEXTS = [
    "Today, we're open-sourcing the Model Context Protocol (MCP)...",
    # ... (other sample texts omitted for brevity)
    "Whether you're an AI tool developer... we invite you to build the future of context-aware AI together",
]

class RAGAgent:
    """A RAG agent that uses Qdrant for retrieval and an LLM for generation."""

    def __init__(self, collection_name: str = "my_collection"):
        self.collection_name = collection_name
        self.app = MCPApp(
            name="mcp_rag_agent",
            settings=get_settings("configs/mcp_agent.config.yaml")
        )
        self.qdrant_client = QdrantClient("http://localhost:6333")
        self.agent: Optional[Agent] = None
        self.llm: Optional[OpenAIAugmentedLLM] = None
        
    async def initialize(self):
        """Initializes the agent, LLM, and Qdrant collection."""
        await self.app.initialize()
        
        self._initialize_collection()
        
        self.agent = Agent(
            connection_persistence=False,
            name="rag_agent",
            instruction="""You are an intelligent assistant equipped with a "find memories" tool that allows you to access information about Model Context Protocol (MCP). Your primary role is to assist users with queries about MCP by actively using the "find memories" tool to retrieve and provide accurate responses. Always utilize the "find memories" tool whenever necessary to ensure accurate information.""",
            server_names=["qdrant"],
        )
        await self.agent.initialize()
        self.llm = await self.agent.attach_llm(OpenAIAugmentedLLM)

    def _initialize_collection(self):
        """Creates and populates the Qdrant collection if it doesn't exist."""
        self.qdrant_client.set_model("BAAI/bge-small-en-v1.5")
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.add(
                collection_name=self.collection_name,
                documents=SAMPLE_TEXTS,
            )
            print(f"Collection '{self.collection_name}' created and populated.")

    async def chat(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Handles a chat query, generates a response using RAG.
        
        Args:
            query: The user's query.
            history: The conversation history.
        
        Returns:
            The agent's response.
        """
        if not self.llm:
            await self.initialize()

        if not self.llm:
            return "Error: LLM could not be initialized."

        # Note: The 'history' parameter is not directly used in this simplified
        # 'generate_str' call, but the LLM is configured to use history from
        # its internal state if available. A more robust implementation might
        # format and pass the history explicitly.
        try:
            response = await self.llm.generate_str(
                message=query, 
                request_params=RequestParams(use_history=True)
            )
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

# --- Helper functions for Streamlit UI (can be moved later) ---

def get_qdrant_status() -> Dict[str, Any]:
    """Qdrant 서버 상태 확인"""
    try:
        client = QdrantClient("http://localhost:6333")
        collections = client.get_collections()
        return {
            "status": "connected",
            "server_url": "http://localhost:6333",
            "collections_count": len(collections.collections) if collections else 0,
            "timestamp": datetime.now().isoformat(),
            "message": "Qdrant server is running and accessible"
        }
    except Exception as e:
        return {
            "status": "disconnected",
            "server_url": "http://localhost:6333",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "message": "Failed to connect to Qdrant server."
        }
