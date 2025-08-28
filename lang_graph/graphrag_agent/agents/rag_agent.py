"""
RAG Agent

Knowledge Graph를 활용한 RAG(Retrieval-Augmented Generation) 전문 에이전트
"""

import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, validator
import logging
import re
from pathlib import Path


class RAGAgentConfig(BaseModel):
    """Configuration for RAG Agent"""
    openai_api_key: str = Field(..., description="OpenAI API key")
    model_name: str = Field(default="gemini-2.5-flash-lite-preview-06-07", description="LLM model name")
    temperature: float = Field(default=0.1, description="LLM temperature for generation")
    max_search_results: int = Field(default=5, description="Maximum search results to retrieve")
    context_window_size: int = Field(default=4000, description="Maximum context window size")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError('temperature must be between 0.0 and 2.0')
        return v
    
    @validator('max_search_results')
    def validate_max_search_results(cls, v):
        if v < 1 or v > 20:
            raise ValueError('max_search_results must be between 1 and 20')
        return v
    
    @validator('context_window_size')
    def validate_context_window_size(cls, v):
        if v < 1000 or v > 32000:
            raise ValueError('context_window_size must be between 1000 and 32000')
        return v


class RAGAgent:
    """Knowledge Graph 기반 RAG 전문 에이전트"""
    
    def __init__(self, config: RAGAgentConfig):
        self.config = config
        self._setup_logging()
        self._initialize_components()
        self.logger.info("RAGAgent initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize LLM and embedding components"""
        try:
            self.llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                api_key=self.config.openai_api_key,
                max_retries=3,
                timeout=60
            )
            self.logger.info(f"LLM initialized with model: {self.config.model_name}")
            
            self.embeddings = OpenAIEmbeddings(
                api_key=self.config.openai_api_key,
                model="text-embedding-3-small"
            )
            self.logger.info("Embeddings initialized")
            
            # Initialize prompt template
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt()),
                ("human", "{user_question}")
            ])
            
            # Initialize output parser
            self.output_parser = StrOutputParser()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise RuntimeError(f"Component initialization failed: {e}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for RAG generation"""
        return """You are a helpful AI assistant that answers questions based on the provided context from a knowledge graph.

Your task is to:
1. Analyze the user's question carefully
2. Use ONLY the provided context to answer the question
3. If the context doesn't contain enough information to answer the question, say so clearly
4. Provide accurate, concise, and helpful responses
5. Cite specific entities or relationships from the context when relevant
6. If you're unsure about something, acknowledge the uncertainty

Remember: Only use information from the provided context. Do not make up or infer information that isn't explicitly stated in the context."""
    
    async def query_knowledge_graph(self, user_query: str, knowledge_graph: Any) -> Dict[str, Any]:
        """
        Query the knowledge graph. This involves creating a vector store,
        retrieving context, and generating a response.
        
        Args:
            user_query: The user's question
            knowledge_graph: The knowledge graph to query
            
        Returns:
            Dict containing status and response
        """
        self.logger.info(f"Received query: '{user_query}'")

        # 1. Validate inputs
        validation_result = self._validate_inputs(user_query, knowledge_graph)
        if validation_result["status"] != "valid":
            return validation_result

        # 2. Create Vector Store from Knowledge Graph
        try:
            vector_store = await self._create_vector_store(knowledge_graph)
            self.logger.info(f"Vector store created with {len(vector_store.get()['documents'])} documents")
        except Exception as e:
            error_msg = f"Vector store creation failed: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg, "response": "Could not process the knowledge base."}
            
        # 3. Retrieve Context
        try:
            context, retrieved_docs = await self._retrieve_context(user_query, vector_store)
            self.logger.info(f"Retrieved {len(retrieved_docs)} documents for context")
        except Exception as e:
            error_msg = f"Context retrieval failed: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg, "response": "Could not find relevant information."}

        # 4. Generate Response
        try:
            response_text = await self._generate_response(user_query, context)
            self.logger.info("Response generated successfully")
        except Exception as e:
            error_msg = f"Response generation failed: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg, "response": "I encountered an error while trying to answer."}

        return {
            "status": "completed",
            "response": response_text,
            "context": context,
            "metadata": {
                "retrieved_docs": len(retrieved_docs),
                "context_length": len(context),
                "model_used": self.config.model_name
            }
        }
    
    def _validate_inputs(self, user_query: str, knowledge_graph: Any) -> Dict[str, Any]:
        """Validate input parameters"""
        if not user_query or not user_query.strip():
            return {"status": "error", "error": "Empty query provided", "response": "Please provide a valid question."}
        
        if not knowledge_graph:
            return {"status": "error", "error": "Knowledge graph is None", "response": "The knowledge base is empty."}
        
        if not hasattr(knowledge_graph, 'nodes') or not knowledge_graph.nodes:
            return {"status": "error", "error": "Knowledge graph has no nodes", "response": "The knowledge base is empty."}
        
        return {"status": "valid"}
    
    async def _create_vector_store(self, knowledge_graph: Any) -> Chroma:
        """Creates a Chroma vector store from the knowledge graph."""
        try:
            graph_docs = []
            
            # Process nodes (entities)
            for node in knowledge_graph.nodes:
                entity_content = self._format_node_content(node)
                graph_docs.append(Document(
                    page_content=entity_content, 
                    metadata={
                        "id": getattr(node, 'id', 'unknown'),
                        "type": "entity",
                        "node_type": getattr(node, 'type', 'unknown'),
                        "source": "knowledge_graph"
                    }
                ))

            # Process edges (relationships)
            for edge in knowledge_graph.edges:
                relationship_content = self._format_edge_content(edge)
                graph_docs.append(Document(
                    page_content=relationship_content, 
                    metadata={
                        "id": getattr(edge, 'id', 'unknown'),
                        "type": "relationship",
                        "edge_type": getattr(edge, 'type', 'unknown'),
                        "source": "knowledge_graph"
                    }
                ))
            
            if not graph_docs:
                raise ValueError("No documents could be created from the knowledge graph")
                
            self.logger.info(f"Created {len(graph_docs)} documents from knowledge graph")
            return Chroma.from_documents(graph_docs, self.embeddings)
            
        except Exception as e:
            self.logger.error(f"Error creating vector store: {e}")
            raise
    
    def _format_node_content(self, node: Any) -> str:
        """Format a node's content for the vector store"""
        content_parts = []
        
        # Add title/name
        if hasattr(node, 'title') and node.title:
            content_parts.append(f"Entity: {node.title}")
        elif hasattr(node, 'name') and node.name:
            content_parts.append(f"Entity: {node.name}")
        elif hasattr(node, 'id') and node.id:
            content_parts.append(f"Entity ID: {node.id}")
        
        # Add description
        if hasattr(node, 'description') and node.description:
            content_parts.append(f"Description: {node.description}")
        
        # Add type
        if hasattr(node, 'type') and node.type:
            content_parts.append(f"Type: {node.type}")
        
        # Add properties if available
        if hasattr(node, 'properties') and node.properties:
            for key, value in node.properties.items():
                if value and str(value).strip():
                    content_parts.append(f"{key}: {value}")
        
        return "\n".join(content_parts) if content_parts else f"Entity: {str(node)}"
    
    def _format_edge_content(self, edge: Any) -> str:
        """Format an edge's content for the vector store"""
        content_parts = []
        
        # Add relationship description
        if hasattr(edge, 'description') and edge.description:
            content_parts.append(f"Relationship: {edge.description}")
        
        # Add source and target
        if hasattr(edge, 'source') and edge.source:
            source_name = getattr(edge.source, 'title', getattr(edge.source, 'name', str(edge.source)))
            content_parts.append(f"From: {source_name}")
        
        if hasattr(edge, 'target') and edge.target:
            target_name = getattr(edge.target, 'title', getattr(edge.target, 'name', str(edge.target)))
            content_parts.append(f"To: {target_name}")
        
        # Add type
        if hasattr(edge, 'type') and edge.type:
            content_parts.append(f"Type: {edge.type}")
        
        return "\n".join(content_parts) if content_parts else f"Relationship: {str(edge)}"
    
    async def _retrieve_context(self, user_query: str, vector_store: Chroma) -> tuple[str, List[Document]]:
        """Retrieve relevant context from the vector store"""
        try:
            retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": self.config.max_search_results,
                    "score_threshold": 0.7  # Only retrieve relevant documents
                }
            )
            
            retrieved_docs = await retriever.ainvoke(user_query)
            
            if not retrieved_docs:
                return "No relevant information found.", []
            
            # Combine document content
            context_parts = []
            for doc in retrieved_docs:
                context_parts.append(doc.page_content)
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Truncate if too long
            if len(context) > self.config.context_window_size:
                context = context[:self.config.context_window_size] + "... [truncated]"
            
            return context, retrieved_docs
            
        except Exception as e:
            self.logger.error(f"Error retrieving context: {e}")
            raise
    
    async def _generate_response(self, user_query: str, context: str) -> str:
        """Generate a response using the LLM"""
        try:
            if not context or context.strip() == "No relevant information found.":
                return "I could not find any relevant information in the knowledge graph to answer your question."
            
            # Create the full prompt with context
            full_prompt = f"""Context from Knowledge Graph:
{context}

User Question: {user_query}

Please answer the question based on the provided context."""
            
            # Generate response
            response = await self.llm.ainvoke([HumanMessage(content=full_prompt)])
            
            if not response or not response.content:
                return "I encountered an error while generating a response."
            
            return response.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise
    
    async def test_connectivity(self) -> bool:
        """Test if the agent can connect to the LLM service"""
        try:
            # Simple test to verify API connectivity
            test_response = await self.llm.ainvoke("Hello")
            return bool(test_response and test_response.content)
            
        except Exception as e:
            self.logger.error(f"Connectivity test failed: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent configuration"""
        return {
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_search_results": self.config.max_search_results,
            "context_window_size": self.config.context_window_size,
            "status": "ready"
        }
    
    async def batch_query(self, queries: List[str], knowledge_graph: Any) -> List[Dict[str, Any]]:
        """Process multiple queries in batch"""
        results = []
        
        for i, query in enumerate(queries):
            self.logger.info(f"Processing batch query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                result = await self.query_knowledge_graph(query, knowledge_graph)
                results.append({
                    "query": query,
                    "result": result
                })
            except Exception as e:
                self.logger.error(f"Batch query {i+1} failed: {e}")
                results.append({
                    "query": query,
                    "result": {
                        "status": "error",
                        "error": str(e),
                        "response": "Query processing failed"
                    }
                })
        
        return results
