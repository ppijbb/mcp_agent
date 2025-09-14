"""
Advanced RAG Agent - Unified Version

Modern Knowledge Graph-based RAG (Retrieval-Augmented Generation) agent
with advanced retrieval strategies, multi-modal support, and intelligent query processing.
"""

import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.llms import HuggingFacePipeline

from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .base_agent_simple import BaseAgent, BaseAgentConfig


class RAGAgentConfig(BaseAgentConfig):
    """Advanced Configuration for RAG Agent"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)
    
    # Retrieval Configuration
    max_search_results: int = Field(default=5, description="Maximum search results to retrieve")
    context_window_size: int = Field(default=4000, description="Maximum context window size")
    retrieval_strategy: str = Field(default="hybrid", description="Retrieval strategy: vector, bm25, hybrid, or graph")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold for retrieval")
    
    # Advanced Features
    enable_query_expansion: bool = Field(default=True, description="Enable query expansion")
    enable_reranking: bool = Field(default=True, description="Enable result reranking")
    enable_multi_modal: bool = Field(default=False, description="Enable multi-modal retrieval")
    enable_temporal_reasoning: bool = Field(default=True, description="Enable temporal reasoning in queries")
    
    # Vector Store Configuration
    vector_store_type: str = Field(default="chroma", description="Vector store type: chroma, faiss, or custom")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model name")
    chunk_size: int = Field(default=1000, description="Text chunk size for vectorization")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    
    # Response Generation
    response_mode: str = Field(default="structured", description="Response mode: structured, conversational, or detailed")
    include_sources: bool = Field(default=True, description="Include source citations in responses")
    max_response_length: int = Field(default=1000, description="Maximum response length")
    
    # Performance
    max_concurrent_queries: int = Field(default=5, description="Maximum concurrent queries")
    
    @field_validator('retrieval_strategy')
    @classmethod
    def validate_retrieval_strategy(cls, v: str) -> str:
        if v not in ['vector', 'bm25', 'hybrid', 'graph']:
            raise ValueError('retrieval_strategy must be one of: vector, bm25, hybrid, graph')
        return v
    
    @field_validator('response_mode')
    @classmethod
    def validate_response_mode(cls, v: str) -> str:
        if v not in ['structured', 'conversational', 'detailed']:
            raise ValueError('response_mode must be one of: structured, conversational, detailed')
        return v


class RAGAgent(BaseAgent):
    """Advanced Knowledge Graph-based RAG Agent with Modern Features"""
    
    def __init__(self, config: RAGAgentConfig):
        super().__init__(config)
        self._setup_cache()
        
        # Initialize RAG-specific components
        self._initialize_rag_components()
        
        self.logger.info("RAGAgent initialized successfully", 
                        config=self.config.model_dump())
    
    def _setup_metrics(self):
        """Setup performance metrics tracking"""
        super()._setup_metrics()
        # Add RAG-specific metrics
        self.metrics.update({
            'queries_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'retrieval_time': 0.0,
            'generation_time': 0.0,
            'successful_queries': 0,
            'failed_queries': 0
        })
    
    def _setup_cache(self):
        """Setup response caching"""
        self.cache = {} if self.config.enable_caching else None
        self.cache_timestamps = {} if self.config.enable_caching else None
    
    def _initialize_rag_components(self):
        """Initialize RAG-specific components"""
        try:
            # Initialize vector store (will be created when needed)
            self.vector_store = None
            self.bm25_retriever = None
            
            # Initialize prompt templates based on response mode
            self.prompt_templates = self._create_prompt_templates()
            
            # Initialize output parsers
            self.output_parsers = self._create_output_parsers()
            
            # Initialize query expansion and reranking components
            if self.config.enable_query_expansion:
                self.query_expander = self._create_query_expander()
            
            if self.config.enable_reranking:
                self.reranker = self._create_reranker()
            
            # Initialize retrieval strategies
            self.retrieval_strategies = self._create_retrieval_strategies()
            
            self.logger.info("RAG components initialized successfully")
            
        except Exception as e:
            self.logger.error("RAG component initialization failed", error=str(e))
            raise
    
    def _create_prompt_templates(self) -> Dict[str, ChatPromptTemplate]:
        """Create prompt templates for different response modes"""
        templates = {}
        
        # Structured response template
        structured_template = ChatPromptTemplate.from_messages([
            ("system", self._get_structured_system_prompt()),
            ("human", "Context: {context}\n\nQuestion: {user_question}")
        ])
        templates["structured"] = structured_template
        
        # Conversational response template
        conversational_template = ChatPromptTemplate.from_messages([
            ("system", self._get_conversational_system_prompt()),
            ("human", "Context: {context}\n\nQuestion: {user_question}")
        ])
        templates["conversational"] = conversational_template
        
        # Detailed response template
        detailed_template = ChatPromptTemplate.from_messages([
            ("system", self._get_detailed_system_prompt()),
            ("human", "Context: {context}\n\nQuestion: {user_question}")
        ])
        templates["detailed"] = detailed_template
        
        return templates
    
    def _create_output_parsers(self) -> Dict[str, Any]:
        """Create output parsers for different response modes"""
        parsers = {}
        
        # Structured response uses JSON parser
        parsers["structured"] = JsonOutputParser()
        
        # Conversational and detailed use string parser
        parsers["conversational"] = StrOutputParser()
        parsers["detailed"] = StrOutputParser()
        
        return parsers
    
    def _create_query_expander(self) -> Any:
        """Create query expansion component"""
        # Placeholder for query expansion logic
        return None
    
    def _create_reranker(self) -> Any:
        """Create reranking component"""
        # Placeholder for reranking logic
        return None
    
    def _create_retrieval_strategies(self) -> Dict[str, callable]:
        """Create retrieval strategy mappings"""
        return {
            'vector': self._vector_retrieval,
            'bm25': self._bm25_retrieval,
            'hybrid': self._hybrid_retrieval,
            'graph': self._graph_retrieval
        }
    
    def _get_structured_system_prompt(self) -> str:
        """Get system prompt for structured responses"""
        return """You are an expert knowledge graph assistant. Provide structured, accurate responses based on the given context.

Instructions:
1. Analyze the provided context carefully
2. Answer the user's question using only information from the context
3. Provide your response in JSON format with the following structure:
   {
     "answer": "Your detailed answer",
     "confidence": 0.0-1.0,
     "sources": ["source1", "source2"],
     "reasoning": "Brief explanation of your reasoning"
   }
4. If you cannot answer based on the context, set confidence to 0.0 and explain why
5. Be precise and factual in your responses"""
    
    def _get_conversational_system_prompt(self) -> str:
        """Get system prompt for conversational responses"""
        return """You are a helpful knowledge graph assistant. Provide natural, conversational responses based on the given context.

Instructions:
1. Use the provided context to answer the user's question
2. Be conversational and engaging in your response
3. If you're not sure about something, say so
4. Cite sources when relevant
5. Keep responses concise but informative"""
    
    def _get_detailed_system_prompt(self) -> str:
        """Get system prompt for detailed responses"""
        return """You are an expert knowledge graph assistant. Provide comprehensive, detailed responses based on the given context.

Instructions:
1. Thoroughly analyze the provided context
2. Provide a comprehensive answer to the user's question
3. Include relevant details, examples, and explanations
4. Cite specific sources and evidence
5. If applicable, provide additional context or related information
6. Structure your response clearly with headings or bullet points when helpful"""
    
    async def query_knowledge_graph(self, user_query: str, knowledge_graph: Any) -> Dict[str, Any]:
        """Query the knowledge graph and generate a response"""
        try:
            start_time = datetime.now()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Processing query...", total=100)
                
                # Validate inputs
                progress.update(task, advance=10, description="Validating inputs...")
                if not await self._validate_inputs_async(user_query, knowledge_graph):
                    return {
                        "response": "Invalid input provided",
                        "sources": [],
                        "confidence": 0.0,
                        "status": "error",
                        "processing_time": 0.0
                    }
                
                # Check cache
                progress.update(task, advance=10, description="Checking cache...")
                cached_response = await self._check_cache(user_query)
                if cached_response:
                    self.metrics['cache_hits'] += 1
                    return cached_response
                
                self.metrics['cache_misses'] += 1
                
                # Expand query if enabled
                progress.update(task, advance=10, description="Expanding query...")
                expanded_query = await self._expand_query(user_query)
                
                # Create or update vector store
                progress.update(task, advance=20, description="Creating vector store...")
                vector_store = await self._create_or_update_vector_store(knowledge_graph)
                
                # Retrieve context
                progress.update(task, advance=20, description="Retrieving context...")
                context, retrieved_docs, retrieval_metadata = await self._advanced_retrieve_context(
                    expanded_query, vector_store, knowledge_graph
                )
                
                # Rerank results if enabled
                progress.update(task, advance=10, description="Reranking results...")
                reranked_docs = await self._rerank_results(expanded_query, retrieved_docs)
                
                # Generate response
                progress.update(task, advance=15, description="Generating response...")
                response_data = await self._generate_advanced_response(
                    user_query, context, reranked_docs, retrieval_metadata
                )
                
                # Cache response if enabled
                progress.update(task, advance=5, description="Caching response...")
                if self.config.enable_caching:
                    await self._cache_response(user_query, response_data)
                
                # Update metrics
                processing_time = (datetime.now() - start_time).total_seconds()
                self._update_metrics(processing_time, len(retrieved_docs), True)
                
                progress.update(task, completed=100, description="Query completed!")
                
                return {
                    **response_data,
                    "processing_time": processing_time,
                    "status": "success"
                }
                
        except Exception as e:
            self.logger.error("Query processing failed", error=str(e))
            self.metrics['errors'] += 1
            self._update_metrics(0, 0, False)
            
            return {
                "response": f"Error processing query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "status": "error",
                "processing_time": 0.0
            }
    
    async def _validate_inputs_async(self, user_query: str, knowledge_graph: Any) -> bool:
        """Validate input parameters"""
        if not user_query or len(user_query.strip()) < 3:
            return False
        
        if not knowledge_graph:
            return False
        
        # Check for sensitive terms
        sensitive_terms = ['password', 'secret', 'private', 'confidential']
        if any(term in user_query.lower() for term in sensitive_terms):
            self.logger.warning("Query contains sensitive terms", query=user_query[:50])
        
        return True
    
    async def _check_cache(self, user_query: str) -> Optional[Dict[str, Any]]:
        """Check cache for existing response"""
        if not self.cache:
            return None
        
        cache_key = self._generate_cache_key(user_query)
        
        if cache_key in self.cache:
            # Check if cache entry is still valid
            if cache_key in self.cache_timestamps:
                cache_time = self.cache_timestamps[cache_key]
                if (datetime.now() - cache_time).total_seconds() < self.config.cache_ttl:
                    return self.cache[cache_key]
                else:
                    # Remove expired cache entry
                    del self.cache[cache_key]
                    del self.cache_timestamps[cache_key]
        
        return None
    
    def _generate_cache_key(self, user_query: str) -> str:
        """Generate cache key for user query"""
        return self._generate_cache_key(user_query)
    
    async def _expand_query(self, user_query: str) -> str:
        """Expand user query with related terms"""
        if not self.config.enable_query_expansion:
            return user_query
        
        # Simple keyword-based expansion
        expansion_keywords = {
            'what': ['definition', 'meaning', 'explanation'],
            'how': ['process', 'method', 'steps'],
            'why': ['reason', 'cause', 'purpose'],
            'when': ['time', 'date', 'schedule'],
            'where': ['location', 'place', 'position'],
            'who': ['person', 'individual', 'entity']
        }
        
        expanded_terms = [user_query]
        query_lower = user_query.lower()
        
        for keyword, expansions in expansion_keywords.items():
            if keyword in query_lower:
                expanded_terms.extend(expansions)
        
        return ' '.join(expanded_terms)
    
    async def _create_or_update_vector_store(self, knowledge_graph: Any) -> VectorStore:
        """Create or update vector store from knowledge graph"""
        if self.vector_store is None:
            self.vector_store = await self._create_vector_store(knowledge_graph)
        return self.vector_store
    
    async def _create_vector_store(self, knowledge_graph: Any) -> Chroma:
        """Create vector store from knowledge graph"""
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
                
            self.logger.info("Created documents from knowledge graph", doc_count=len(graph_docs))
            return Chroma.from_documents(graph_docs, self.embeddings)
            
        except Exception as e:
            self.logger.error("Error creating vector store", error=str(e))
            raise
    
    async def _advanced_retrieve_context(self, query: str, vector_store: VectorStore, 
                                       knowledge_graph: Any) -> Tuple[str, List[Document], Dict[str, Any]]:
        """Advanced context retrieval with multiple strategies"""
        try:
            retrieval_start = datetime.now()
            
            # Choose retrieval strategy
            if self.config.retrieval_strategy == "vector":
                docs, metadata = await self._vector_retrieval(query, vector_store)
            elif self.config.retrieval_strategy == "bm25":
                docs, metadata = await self._bm25_retrieval(query, vector_store)
            elif self.config.retrieval_strategy == "hybrid":
                docs, metadata = await self._hybrid_retrieval(query, vector_store)
            elif self.config.retrieval_strategy == "graph":
                docs, metadata = await self._graph_retrieval(query, knowledge_graph)
            else:
                docs, metadata = await self._vector_retrieval(query, vector_store)
            
            # Combine document content
            context_parts = []
            for doc in docs:
                context_parts.append(doc.page_content)
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Truncate if too long
            if len(context) > self.config.context_window_size:
                context = context[:self.config.context_window_size] + "... [truncated]"
            
            retrieval_time = (datetime.now() - retrieval_start).total_seconds()
            self.metrics['retrieval_time'] += retrieval_time
            
            metadata.update({
                "retrieval_time": retrieval_time,
                "strategy": self.config.retrieval_strategy
            })
            
            return context, docs, metadata
            
        except Exception as e:
            self.logger.error("Advanced context retrieval failed", error=str(e))
            raise
    
    async def _vector_retrieval(self, query: str, vector_store: VectorStore) -> Tuple[List[Document], Dict[str, Any]]:
        """Vector-based retrieval"""
        try:
            retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": self.config.max_search_results,
                    "score_threshold": self.config.similarity_threshold
                }
            )
            
            docs = await retriever.ainvoke(query)
            return docs, {"method": "vector", "doc_count": len(docs)}
            
        except Exception as e:
            self.logger.error("Vector retrieval failed", error=str(e))
            return [], {"method": "vector", "error": str(e)}
    
    async def _bm25_retrieval(self, query: str, vector_store: VectorStore) -> Tuple[List[Document], Dict[str, Any]]:
        """BM25-based retrieval"""
        try:
            # This would use BM25Retriever in practice
            # For now, fall back to vector retrieval
            return await self._vector_retrieval(query, vector_store)
            
        except Exception as e:
            self.logger.error("BM25 retrieval failed", error=str(e))
            return [], {"method": "bm25", "error": str(e)}
    
    async def _hybrid_retrieval(self, query: str, vector_store: VectorStore) -> Tuple[List[Document], Dict[str, Any]]:
        """Hybrid retrieval combining vector and BM25"""
        try:
            # Get results from both methods
            vector_docs, vector_metadata = await self._vector_retrieval(query, vector_store)
            bm25_docs, bm25_metadata = await self._bm25_retrieval(query, vector_store)
            
            # Combine and deduplicate
            all_docs = vector_docs + bm25_docs
            unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
            
            # Take top results
            final_docs = unique_docs[:self.config.max_search_results]
            
            return final_docs, {
                "method": "hybrid",
                "vector_docs": len(vector_docs),
                "bm25_docs": len(bm25_docs),
                "final_docs": len(final_docs)
            }
            
        except Exception as e:
            self.logger.error("Hybrid retrieval failed", error=str(e))
            return [], {"method": "hybrid", "error": str(e)}
    
    async def _graph_retrieval(self, query: str, knowledge_graph: Any) -> Tuple[List[Document], Dict[str, Any]]:
        """Graph-based retrieval using knowledge graph structure"""
        try:
            # This would use graph traversal algorithms
            # For now, convert graph to documents and use vector retrieval
            docs = []
            
            # Extract relevant nodes and edges based on query
            query_terms = query.lower().split()
            
            for node in knowledge_graph.nodes:
                node_text = getattr(node, 'title', '') + ' ' + getattr(node, 'description', '')
                if any(term in node_text.lower() for term in query_terms):
                    doc = Document(
                        page_content=node_text,
                        metadata={
                            "type": "node",
                            "id": getattr(node, 'id', ''),
                            "node_type": getattr(node, 'type', '')
                        }
                    )
                    docs.append(doc)
            
            return docs[:self.config.max_search_results], {
                "method": "graph",
                "doc_count": len(docs)
            }
            
        except Exception as e:
            self.logger.error("Graph retrieval failed", error=str(e))
            return [], {"method": "graph", "error": str(e)}
    
    async def _rerank_results(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank retrieved documents for better relevance"""
        try:
            if not self.reranker or len(docs) <= 1:
                return docs
            
            # Simple reranking based on query-document similarity
            scored_docs = []
            
            for doc in docs:
                # Simple scoring based on term overlap
                query_terms = set(query.lower().split())
                doc_terms = set(doc.page_content.lower().split())
                overlap = len(query_terms.intersection(doc_terms))
                score = overlap / len(query_terms) if query_terms else 0
                
                scored_docs.append((score, doc))
            
            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            return [doc for score, doc in scored_docs]
            
        except Exception as e:
            self.logger.error("Result reranking failed", error=str(e))
            return docs
    
    async def _generate_advanced_response(self, user_query: str, context: str, 
                                        retrieved_docs: List[Document], 
                                        retrieval_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advanced response with multiple modes"""
        try:
            generation_start = datetime.now()
            
            # Choose prompt template based on response mode
            prompt_template = self.prompt_templates.get(self.config.response_mode, 
                                                      self.prompt_templates["conversational"])
            
            # Prepare context with sources
            formatted_context = self._format_context_with_sources(context, retrieved_docs)
            
            # Generate response
            chain = prompt_template | self.llm | self.output_parsers[self.config.response_mode]
            
            response = await chain.ainvoke({
                "user_question": user_query,
                "context": formatted_context
            })
            
            generation_time = (datetime.now() - generation_start).total_seconds()
            self.metrics['generation_time'] += generation_time
            
            # Extract sources and confidence
            sources = self._extract_sources(retrieved_docs)
            confidence = self._calculate_confidence(response, context, user_query)
            
            return {
                "response": str(response),
                "sources": sources,
                "confidence": confidence,
                "generation_time": generation_time
            }
            
        except Exception as e:
            self.logger.error("Advanced response generation failed", error=str(e))
            raise
    
    def _format_context_with_sources(self, context: str, retrieved_docs: List[Document]) -> str:
        """Format context with source information"""
        if not self.config.include_sources:
            return context
        
        formatted_parts = [context]
        
        if retrieved_docs:
            formatted_parts.append("\n\nSources:")
            for i, doc in enumerate(retrieved_docs, 1):
                source_info = doc.metadata.get('source', f'Document {i}')
                formatted_parts.append(f"{i}. {source_info}")
        
        return "\n".join(formatted_parts)
    
    def _extract_sources(self, retrieved_docs: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from retrieved documents"""
        sources = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            source = {
                "id": i,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source)
        
        return sources
    
    def _calculate_confidence(self, response: str, context: str, query: str) -> float:
        """Calculate confidence score for the response"""
        try:
            # Simple confidence calculation based on response length and context relevance
            response_length = len(response)
            context_length = len(context)
            query_length = len(query)
            
            # Basic confidence factors
            length_factor = min(1.0, response_length / 100)  # Longer responses are generally better
            context_factor = min(1.0, context_length / 1000)  # More context is better
            query_factor = min(1.0, query_length / 50)  # Reasonable query length
            
            # Combine factors
            confidence = (length_factor * 0.4 + context_factor * 0.4 + query_factor * 0.2)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            self.logger.error("Confidence calculation failed", error=str(e))
            return 0.5
    
    async def _cache_response(self, user_query: str, response_data: Dict[str, Any]) -> None:
        """Cache response for future use"""
        try:
            if not self.cache:
                return
            
            cache_key = self._generate_cache_key(user_query)
            self.cache[cache_key] = response_data
            self.cache_timestamps[cache_key] = datetime.now()
            
            self.logger.info("Response cached", cache_key=cache_key)
            
        except Exception as e:
            self.logger.error("Response caching failed", error=str(e))
    
    def _update_metrics(self, processing_time: float, retrieved_docs: int, success: bool) -> None:
        """Update processing metrics"""
        super()._update_metrics(processing_time, success)
        
        if success:
            self.metrics['successful_queries'] += 1
        else:
            self.metrics['failed_queries'] += 1
    
    async def batch_query(self, queries: List[str], knowledge_graph: Any) -> List[Dict[str, Any]]:
        """Process multiple queries in batch with concurrency control"""
        results = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_queries)
        
        async def process_single_query(query: str, query_idx: int) -> Dict[str, Any]:
            async with semaphore:
                self.logger.info("Processing batch query", 
                               query_idx=query_idx + 1, 
                               total_queries=len(queries),
                               query=query[:50])
                
                try:
                    result = await self.query_knowledge_graph(query, knowledge_graph)
                    return {
                        "query": query,
                        "query_idx": query_idx,
                        "result": result
                    }
                except Exception as e:
                    self.logger.error("Batch query failed", 
                                    query_idx=query_idx + 1,
                                    error=str(e))
                    return {
                        "query": query,
                        "query_idx": query_idx,
                        "result": {
                            "status": "error",
                            "error": str(e),
                            "response": "Query processing failed"
                        }
                    }
        
        # Process all queries concurrently
        tasks = [process_single_query(query, idx) for idx, query in enumerate(queries)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        self.logger.info("Batch processing completed", 
                        total_queries=len(queries),
                        successful=len(valid_results))
        
        return valid_results
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get comprehensive configuration summary"""
        summary = super().get_config_summary()
        
        # Add RAG-specific fields
        summary.update({
            "max_search_results": self.config.max_search_results,
            "context_window_size": self.config.context_window_size,
            "retrieval_strategy": self.config.retrieval_strategy,
            "similarity_threshold": self.config.similarity_threshold,
            "enable_query_expansion": self.config.enable_query_expansion,
            "enable_reranking": self.config.enable_reranking,
            "enable_multi_modal": self.config.enable_multi_modal,
            "enable_temporal_reasoning": self.config.enable_temporal_reasoning,
            "vector_store_type": self.config.vector_store_type,
            "embedding_model": self.config.embedding_model,
            "response_mode": self.config.response_mode,
            "include_sources": self.config.include_sources,
            "max_response_length": self.config.max_response_length,
            "max_concurrent_queries": self.config.max_concurrent_queries
        })
        
        return summary
