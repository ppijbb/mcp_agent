from mcp_agent.context import AgentContext
from srcs.core.agent.base import BaseAgent
from srcs.core.errors import APIError, WorkflowError
from qdrant_client import QdrantClient, models
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from typing import Optional, List, Dict, Any
from datetime import datetime


SAMPLE_TEXTS = [
    "Today, we're open-sourcing the Model Context Protocol (MCP)...",
    # ... (other sample texts omitted for brevity)
    "Whether you're an AI tool developer... we invite you to build the future of context-aware AI together",
]

class RAGAgent(BaseAgent):
    """A RAG agent that uses Qdrant for retrieval and an LLM for generation."""

    def __init__(self, collection_name: str = "my_collection"):
        super().__init__("rag_agent")
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient("http://localhost:6333")
        self.llm: Optional[OpenAIAugmentedLLM] = None

    def _initialize_collection(self):
        """Creates and populates the Qdrant collection if it doesn't exist."""
        try:
            self.qdrant_client.set_model("BAAI/bge-small-en-v1.5")
            if not self.qdrant_client.collection_exists(self.collection_name):
                self.qdrant_client.add(
                    collection_name=self.collection_name,
                    documents=SAMPLE_TEXTS,
                )
                self.logger.info(f"Collection '{self.collection_name}' created and populated.")
        except Exception as e:
            raise APIError(f"Failed to initialize Qdrant collection: {e}") from e

    async def run_workflow(self, context: AgentContext):
        """Initializes the agent, LLM, and Qdrant collection, then runs the chat logic."""
        try:
            self.logger.info("Initializing RAG Agent...")
            self._initialize_collection()

            agent = await context.create_agent(
                name="rag_agent",
                instruction='''You are an intelligent assistant equipped with a "find memories" tool that allows you to access information about Model Context Protocol (MCP). Your primary role is to assist users with queries about MCP by actively using the "find memories" tool to retrieve and provide accurate responses. Always utilize the "find memories" tool whenever necessary to ensure accurate information.''',
                server_names=["qdrant"],
            )
            self.llm = await agent.attach_llm(OpenAIAugmentedLLM)
            self.logger.info("RAG Agent initialized.")

            # Example of how to use the agent. In a real scenario, this would
            # come from an external input, like a user query.
            query = context.get("query", "What is the Model Context Protocol?")
            self.logger.info(f"Received query: {query}")

            if not self.llm:
                raise WorkflowError("LLM could not be initialized.")

            response = await self.llm.generate_str(
                message=query,
                request_params=RequestParams(use_history=True)
            )
            self.logger.info(f"Agent response: {response}")
            context.set("response", response)
        except Exception as e:
            raise WorkflowError(f"RAG agent workflow failed: {e}") from e

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
        raise APIError(f"Failed to connect to Qdrant server: {e}") from e
