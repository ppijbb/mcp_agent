"""
RAG Agent

Knowledge Graph를 활용한 RAG(Retrieval-Augmented Generation) 전문 에이전트
"""

import asyncio
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import logging

class RAGAgentConfig(BaseModel):
    """Configuration for RAG Agent"""
    openai_api_key: str = Field(..., description="OpenAI API key")
    model_name: str = Field(default="gemini-2.5-flash-lite-preview-06-07", description="LLM model name")
    temperature: float = Field(default=0.1, description="LLM temperature for generation")
    max_search_results: int = Field(default=5, description="Maximum search results to retrieve")
    context_window_size: int = Field(default=4000, description="Maximum context window size")

class RAGAgent:
    """Knowledge Graph 기반 RAG 전문 에이전트"""
    
    def __init__(self, config: RAGAgentConfig):
        self.config = config
        self._initialize_components()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize LLM and embedding components"""
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            api_key=self.config.openai_api_key
        )
        self.embeddings = OpenAIEmbeddings(
            api_key=self.config.openai_api_key
        )
    
    async def query_knowledge_graph(self, user_query: str, knowledge_graph: Any) -> Dict[str, Any]:
        """
        Query the knowledge graph. This involves creating a vector store,
        retrieving context, and generating a response.
        """
        self.logger.info(f"Received query: '{user_query}'")

        # 1. Validate inputs
        if not user_query.strip():
            return {"status": "error", "error": "Empty query provided", "response": "Please provide a valid question."}
        if not knowledge_graph or not knowledge_graph.nodes:
            return {"status": "error", "error": "Knowledge graph is empty", "response": "The knowledge base is empty."}

        # 2. Create Vector Store from Knowledge Graph
        try:
            vector_store = self._create_vector_store(knowledge_graph)
            self.logger.info(f"Vector store created with {len(vector_store.get()['documents'])} documents.")
        except Exception as e:
            self.logger.error(f"Vector store creation failed: {e}")
            return {"status": "error", "error": f"Vector store creation failed: {e}", "response": "Could not process the knowledge base."}
            
        # 3. Retrieve Context
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": self.config.max_search_results})
            retrieved_docs = retriever.invoke(user_query)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            if len(context) > self.config.context_window_size:
                context = context[:self.config.context_window_size]
            
            self.logger.info(f"Retrieved {len(retrieved_docs)} documents for context.")
        except Exception as e:
            self.logger.error(f"Context retrieval failed: {e}")
            return {"status": "error", "error": f"Context retrieval failed: {e}", "response": "Could not find relevant information."}

        # 4. Generate Response
        if not context:
            response_text = "I could not find any relevant information in the knowledge graph to answer your question."
        else:
            generation_prompt = f"""
            You are a helpful AI assistant. Use the following context to answer the user's question.
            The context is derived from a knowledge graph.
            
            Context:
            {context}
            
            User Question: {user_query}
            
            Answer based only on the provided context.
            """
            try:
                response = await self.llm.ainvoke([HumanMessage(content=generation_prompt)])
                response_text = response.content
                self.logger.info("Response generated successfully.")
            except Exception as e:
                self.logger.error(f"Response generation failed: {e}")
                return {"status": "error", "error": f"Response generation failed: {e}", "response": "I encountered an error while trying to answer."}

        return {
            "status": "completed",
            "response": response_text,
            "context": context
        }

    def _create_vector_store(self, knowledge_graph: Any) -> Chroma:
        """Creates a Chroma vector store from the knowledge graph."""
        graph_docs = []
        for node in knowledge_graph.nodes:
            entity_content = f"Entity: {node.title}\n"
            if hasattr(node, 'description') and node.description:
                entity_content += f"Description: {node.description}\n"
            graph_docs.append(Document(page_content=entity_content, metadata={"id": node.id, "type": "entity"}))

        for edge in knowledge_graph.edges:
            relationship_content = f"Relationship: {edge.source.title} -> {edge.target.title}\n"
            if hasattr(edge, 'description') and edge.description:
                relationship_content += f"Description: {edge.description}\n"
            graph_docs.append(Document(page_content=relationship_content, metadata={"id": edge.id, "type": "relationship"}))
        
        if not graph_docs:
            raise ValueError("No documents could be created from the knowledge graph.")
            
        return Chroma.from_documents(graph_docs, self.embeddings)
