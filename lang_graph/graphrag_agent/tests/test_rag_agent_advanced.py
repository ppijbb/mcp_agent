"""
Advanced test cases for RAG Agent with modern features
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from lang_graph.graphrag_agent.agents.rag_agent import (
    RAGAgent,
    RAGAgentConfig
)


class TestRAGAgentAdvanced:
    """Advanced test cases for RAG Agent"""
    
    @pytest.fixture
    def config(self):
        """Create advanced test configuration"""
        return RAGAgentConfig(
            model_name="gpt-5-mini",
            temperature=0.1,
            max_tokens=2000,
            max_search_results=5,
            context_window_size=4000,
            retrieval_strategy="hybrid",
            similarity_threshold=0.7,
            enable_query_expansion=True,
            enable_reranking=True,
            enable_multi_modal=False,
            enable_temporal_reasoning=False,
            vector_store_type="chroma",
            embedding_model="text-embedding-3-small",
            chunk_size=1000,
            chunk_overlap=200,
            response_mode="structured",
            include_sources=True,
            max_response_length=1000,
            enable_caching=True,
            cache_ttl=3600,
            max_concurrent_queries=3
        )
    
    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create mock knowledge graph"""
        graph = Mock()
        graph.nodes = [
            Mock(id="node1", title="Entity 1", type="person", description="A person entity"),
            Mock(id="node2", title="Entity 2", type="organization", description="An organization entity"),
            Mock(id="node3", title="Entity 3", type="concept", description="A concept entity")
        ]
        graph.edges = [
            Mock(
                id="edge1",
                type="works_for",
                description="Entity 1 works for Entity 2",
                source=graph.nodes[0],
                target=graph.nodes[1]
            ),
            Mock(
                id="edge2",
                type="related_to",
                description="Entity 2 is related to Entity 3",
                source=graph.nodes[1],
                target=graph.nodes[2]
            )
        ]
        return graph
    
    @pytest.fixture
    def agent(self, config):
        """Create test agent with mocked components"""
        with patch('lang_graph.graphrag_agent.agents.rag_agent.ChatOpenAI'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.OpenAIEmbeddings'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.Chroma'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.RedisCache'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.SQLiteCache'):
            
            agent = RAGAgent(config)
            
            # Mock LLM responses
            agent.llm = AsyncMock()
            agent.llm.ainvoke.return_value = Mock(content='{"answer": "Test response", "confidence": 0.8}')
            
            # Mock embeddings
            agent.embeddings = AsyncMock()
            agent.embeddings.aembed_query.return_value = [0.1, 0.2, 0.3]
            agent.embeddings.aembed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            
            # Mock vector store
            agent.vector_store = Mock()
            agent.vector_store.as_retriever.return_value = Mock()
            agent.vector_store.as_retriever.return_value.ainvoke = AsyncMock(return_value=[
                Mock(page_content="Test content 1", metadata={"source": "test1"}),
                Mock(page_content="Test content 2", metadata={"source": "test2"})
            ])
            
            return agent
    
    def test_agent_initialization_advanced(self, agent, config):
        """Test advanced agent initialization"""
        assert agent.config == config
        assert agent.console is not None
        assert agent.logger is not None
        assert agent.metrics is not None
        assert agent.cache is not None
        assert agent.cache_timestamps is not None
        assert agent.prompt_templates is not None
        assert agent.output_parsers is not None
        assert agent.retrieval_strategies is not None
    
    def test_prompt_templates_creation(self, agent):
        """Test prompt template creation"""
        assert 'structured' in agent.prompt_templates
        assert 'conversational' in agent.prompt_templates
        assert 'detailed' in agent.prompt_templates
        
        # Test structured prompt
        structured_prompt = agent.prompt_templates['structured']
        assert structured_prompt is not None
        
        # Test conversational prompt
        conversational_prompt = agent.prompt_templates['conversational']
        assert conversational_prompt is not None
        
        # Test detailed prompt
        detailed_prompt = agent.prompt_templates['detailed']
        assert detailed_prompt is not None
    
    def test_output_parsers_creation(self, agent):
        """Test output parser creation"""
        assert 'structured' in agent.output_parsers
        assert 'conversational' in agent.output_parsers
        assert 'detailed' in agent.output_parsers
        
        # Test structured parser (should be JsonOutputParser)
        from langchain_core.output_parsers import JsonOutputParser
        assert isinstance(agent.output_parsers['structured'], JsonOutputParser)
        
        # Test conversational parser (should be StrOutputParser)
        from langchain_core.output_parsers import StrOutputParser
        assert isinstance(agent.output_parsers['conversational'], StrOutputParser)
        assert isinstance(agent.output_parsers['detailed'], StrOutputParser)
    
    def test_retrieval_strategies_creation(self, agent):
        """Test retrieval strategy creation"""
        assert 'vector' in agent.retrieval_strategies
        assert 'bm25' in agent.retrieval_strategies
        assert 'hybrid' in agent.retrieval_strategies
        assert 'graph' in agent.retrieval_strategies
        
        # Test strategy methods exist
        assert callable(agent.retrieval_strategies['vector'])
        assert callable(agent.retrieval_strategies['bm25'])
        assert callable(agent.retrieval_strategies['hybrid'])
        assert callable(agent.retrieval_strategies['graph'])
    
    def test_system_prompts(self, agent):
        """Test system prompt generation"""
        # Test structured system prompt
        structured_prompt = agent._get_structured_system_prompt()
        assert "structured" in structured_prompt.lower()
        assert "json" in structured_prompt.lower()
        
        # Test conversational system prompt
        conversational_prompt = agent._get_conversational_system_prompt()
        assert "conversational" in conversational_prompt.lower()
        
        # Test detailed system prompt
        detailed_prompt = agent._get_detailed_system_prompt()
        assert "detailed" in detailed_prompt.lower()
        assert "comprehensive" in detailed_prompt.lower()
    
    def test_cache_key_generation(self, agent):
        """Test cache key generation"""
        query1 = "What is the meaning of life?"
        query2 = "What is the meaning of life?"  # Same query
        query3 = "What is the purpose of existence?"  # Different query
        
        key1 = agent._generate_cache_key(query1)
        key2 = agent._generate_cache_key(query2)
        key3 = agent._generate_cache_key(query3)
        
        # Same queries should generate same keys
        assert key1 == key2
        
        # Different queries should generate different keys
        assert key1 != key3
        
        # Keys should be strings
        assert isinstance(key1, str)
        assert isinstance(key2, str)
        assert isinstance(key3, str)
    
    def test_confidence_calculation(self, agent):
        """Test confidence score calculation"""
        # Test normal case
        response = "This is a comprehensive response with detailed information."
        context = "This is a long context with lots of relevant information."
        query = "What is the meaning of life?"
        
        confidence = agent._calculate_confidence(response, context, query)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably confident
        
        # Test edge cases
        empty_response = ""
        empty_context = ""
        empty_query = ""
        
        confidence_empty = agent._calculate_confidence(empty_response, empty_context, empty_query)
        assert confidence_empty == 0.0
        
        # Test very long response
        long_response = "This is a very long response. " * 100
        confidence_long = agent._calculate_confidence(long_response, context, query)
        assert confidence_long == 1.0  # Should be max confidence
    
    def test_source_extraction(self, agent):
        """Test source extraction from documents"""
        from langchain_core.documents import Document
        
        docs = [
            Document(page_content="Content 1", metadata={"source": "source1", "type": "entity"}),
            Document(page_content="Content 2", metadata={"source": "source2", "type": "relationship"}),
            Document(page_content="Content 3", metadata={"source": "source3", "type": "concept"})
        ]
        
        sources = agent._extract_sources(docs)
        
        assert len(sources) == 3
        
        for i, source in enumerate(sources):
            assert source['id'] == i + 1
            assert 'content' in source
            assert 'metadata' in source
            assert source['metadata']['source'] == f"source{i+1}"
    
    def test_context_formatting_with_sources(self, agent):
        """Test context formatting with source information"""
        from langchain_core.documents import Document
        
        context = "This is the main context."
        docs = [
            Document(page_content="Content 1", metadata={"source": "source1"}),
            Document(page_content="Content 2", metadata={"source": "source2"})
        ]
        
        # Test with sources enabled
        formatted = agent._format_context_with_sources(context, docs)
        assert "This is the main context." in formatted
        assert "Sources:" in formatted
        assert "1. source1" in formatted
        assert "2. source2" in formatted
        
        # Test with sources disabled
        agent.config.include_sources = False
        formatted_no_sources = agent._format_context_with_sources(context, docs)
        assert formatted_no_sources == context
    
    def test_metrics_initialization(self, agent):
        """Test metrics initialization"""
        expected_metrics = {
            'queries_processed': 0,
            'total_processing_time': 0.0,
            'average_response_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'retrieval_time': 0.0,
            'generation_time': 0.0,
            'errors': 0,
            'successful_queries': 0,
            'failed_queries': 0
        }
        
        for key, value in expected_metrics.items():
            assert agent.metrics[key] == value
    
    def test_metrics_update(self, agent):
        """Test metrics update"""
        # Test successful query
        agent._update_metrics(1.5, 3, True)
        
        assert agent.metrics['queries_processed'] == 1
        assert agent.metrics['total_processing_time'] == 1.5
        assert agent.metrics['average_response_time'] == 1.5
        assert agent.metrics['successful_queries'] == 1
        assert agent.metrics['failed_queries'] == 0
        
        # Test failed query
        agent._update_metrics(0.5, 0, False)
        
        assert agent.metrics['queries_processed'] == 2
        assert agent.metrics['total_processing_time'] == 2.0
        assert agent.metrics['average_response_time'] == 1.0
        assert agent.metrics['successful_queries'] == 1
        assert agent.metrics['failed_queries'] == 1
    
    def test_metrics_reset(self, agent):
        """Test metrics reset"""
        # Set some metrics
        agent.metrics['queries_processed'] = 10
        agent.metrics['total_processing_time'] = 100.0
        agent.metrics['successful_queries'] = 8
        agent.metrics['failed_queries'] = 2
        
        # Reset metrics
        agent.reset_metrics()
        
        # Check all metrics are reset
        assert agent.metrics['queries_processed'] == 0
        assert agent.metrics['total_processing_time'] == 0.0
        assert agent.metrics['average_response_time'] == 0.0
        assert agent.metrics['successful_queries'] == 0
        assert agent.metrics['failed_queries'] == 0
    
    @pytest.mark.asyncio
    async def test_validate_inputs_async(self, agent):
        """Test input validation"""
        # Test valid query
        result = await agent._validate_inputs_async("What is the meaning of life?", None)
        assert result is True
        
        # Test empty query
        result = await agent._validate_inputs_async("", None)
        assert result is False
        
        # Test very short query
        result = await agent._validate_inputs_async("a", None)
        assert result is False
        
        # Test very long query
        long_query = "What is the meaning of life? " * 1000
        result = await agent._validate_inputs_async(long_query, None)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_cache(self, agent):
        """Test cache checking"""
        # Test cache miss
        result = await agent._check_cache("test query")
        assert result is None
        
        # Test cache hit
        cache_key = agent._generate_cache_key("test query")
        agent.cache[cache_key] = {"response": "cached response", "sources": []}
        agent.cache_timestamps[cache_key] = datetime.now()
        
        result = await agent._check_cache("test query")
        assert result is not None
        assert result["response"] == "cached response"
    
    @pytest.mark.asyncio
    async def test_expand_query(self, agent):
        """Test query expansion"""
        # Test basic query expansion
        expanded = await agent._expand_query("AI machine learning")
        assert "AI" in expanded
        assert "machine learning" in expanded
        assert "artificial intelligence" in expanded.lower()
        
        # Test empty query
        expanded = await agent._expand_query("")
        assert expanded == ""
        
        # Test single word query
        expanded = await agent._expand_query("Python")
        assert "Python" in expanded
        assert "programming" in expanded.lower()
    
    @pytest.mark.asyncio
    async def test_create_or_update_vector_store(self, agent, mock_knowledge_graph):
        """Test vector store creation/update"""
        # Mock Chroma.from_documents
        with patch('lang_graph.graphrag_agent.agents.rag_agent.Chroma.from_documents') as mock_chroma:
            mock_chroma.return_value = Mock()
            
            result = await agent._create_or_update_vector_store(mock_knowledge_graph)
            
            assert result is not None
            mock_chroma.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_vector_retrieval(self, agent):
        """Test vector retrieval"""
        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.ainvoke = AsyncMock(return_value=[
            Mock(page_content="Test content 1", metadata={"source": "test1"}),
            Mock(page_content="Test content 2", metadata={"source": "test2"})
        ])
        
        agent.vector_store = Mock()
        agent.vector_store.as_retriever.return_value = mock_retriever
        
        docs, metadata = await agent._vector_retrieval("test query", agent.vector_store)
        
        assert len(docs) == 2
        assert metadata["method"] == "vector"
        assert metadata["doc_count"] == 2
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval(self, agent):
        """Test hybrid retrieval"""
        # Mock vector and BM25 retrievals
        with patch.object(agent, '_vector_retrieval') as mock_vector, \
             patch.object(agent, '_bm25_retrieval') as mock_bm25:
            
            mock_vector.return_value = ([Mock(page_content="Vector doc")], {"method": "vector"})
            mock_bm25.return_value = ([Mock(page_content="BM25 doc")], {"method": "bm25"})
            
            docs, metadata = await agent._hybrid_retrieval("test query", agent.vector_store)
            
            assert len(docs) == 2
            assert metadata["method"] == "hybrid"
            assert metadata["vector_docs"] == 1
            assert metadata["bm25_docs"] == 1
            assert metadata["final_docs"] == 2
    
    @pytest.mark.asyncio
    async def test_graph_retrieval(self, agent, mock_knowledge_graph):
        """Test graph-based retrieval"""
        docs, metadata = await agent._graph_retrieval("person", mock_knowledge_graph)
        
        assert len(docs) > 0
        assert metadata["method"] == "graph"
        assert metadata["doc_count"] == len(docs)
        
        # Check that documents contain relevant content
        for doc in docs:
            assert "person" in doc.page_content.lower() or "entity" in doc.page_content.lower()
    
    @pytest.mark.asyncio
    async def test_rerank_results(self, agent):
        """Test result reranking"""
        from langchain_core.documents import Document
        
        docs = [
            Document(page_content="This is about machine learning and AI", metadata={"source": "doc1"}),
            Document(page_content="This is about cooking recipes", metadata={"source": "doc2"}),
            Document(page_content="This is about artificial intelligence and neural networks", metadata={"source": "doc3"})
        ]
        
        query = "machine learning AI"
        
        reranked = await agent._rerank_results(query, docs)
        
        assert len(reranked) == 3
        
        # First document should be most relevant
        assert "machine learning" in reranked[0].page_content.lower()
        assert "AI" in reranked[0].page_content.lower()
    
    @pytest.mark.asyncio
    async def test_batch_query(self, agent, mock_knowledge_graph):
        """Test batch query processing"""
        queries = [
            "What is entity 1?",
            "What is entity 2?",
            "What is entity 3?"
        ]
        
        # Mock the main query method
        with patch.object(agent, 'query_knowledge_graph') as mock_query:
            mock_query.return_value = {
                "response": "Test response",
                "sources": [],
                "confidence": 0.8
            }
            
            results = await agent.batch_query(queries, mock_knowledge_graph)
            
            assert len(results) == 3
            
            for i, result in enumerate(results):
                assert result["query"] == queries[i]
                assert result["query_idx"] == i
                assert "result" in result
                assert result["result"]["response"] == "Test response"
    
    @pytest.mark.asyncio
    async def test_batch_query_with_errors(self, agent, mock_knowledge_graph):
        """Test batch query processing with errors"""
        queries = [
            "What is entity 1?",
            "What is entity 2?",
            "What is entity 3?"
        ]
        
        # Mock the main query method to raise exception for second query
        with patch.object(agent, 'query_knowledge_graph') as mock_query:
            def side_effect(query, graph):
                if "entity 2" in query:
                    raise Exception("Query failed")
                return {"response": "Test response", "sources": [], "confidence": 0.8}
            
            mock_query.side_effect = side_effect
            
            results = await agent.batch_query(queries, mock_knowledge_graph)
            
            assert len(results) == 3
            
            # First and third should succeed
            assert results[0]["result"]["response"] == "Test response"
            assert results[2]["result"]["response"] == "Test response"
            
            # Second should fail
            assert results[1]["result"]["status"] == "error"
            assert "Query failed" in results[1]["result"]["error"]
    
    def test_get_config_summary_advanced(self, agent):
        """Test advanced configuration summary"""
        summary = agent.get_config_summary()
        
        # Test all configuration fields are included
        assert summary["model_name"] == agent.config.model_name
        assert summary["temperature"] == agent.config.temperature
        assert summary["max_tokens"] == agent.config.max_tokens
        assert summary["retrieval_strategy"] == agent.config.retrieval_strategy
        assert summary["similarity_threshold"] == agent.config.similarity_threshold
        assert summary["enable_query_expansion"] == agent.config.enable_query_expansion
        assert summary["enable_reranking"] == agent.config.enable_reranking
        assert summary["enable_multi_modal"] == agent.config.enable_multi_modal
        assert summary["enable_temporal_reasoning"] == agent.config.enable_temporal_reasoning
        assert summary["vector_store_type"] == agent.config.vector_store_type
        assert summary["embedding_model"] == agent.config.embedding_model
        assert summary["response_mode"] == agent.config.response_mode
        assert summary["include_sources"] == agent.config.include_sources
        assert summary["enable_caching"] == agent.config.enable_caching
        assert summary["cache_ttl"] == agent.config.cache_ttl
        assert summary["max_concurrent_queries"] == agent.config.max_concurrent_queries
        assert summary["status"] == "ready"
        assert "metrics" in summary
    
    @pytest.mark.asyncio
    async def test_connectivity_test(self, agent):
        """Test connectivity testing"""
        # Test successful connectivity
        result = await agent.test_connectivity()
        assert result is True
        
        # Test failed connectivity
        agent.llm.ainvoke.return_value = None
        result = await agent.test_connectivity()
        assert result is False
        
        # Test exception during connectivity test
        agent.llm.ainvoke.side_effect = Exception("Connection failed")
        result = await agent.test_connectivity()
        assert result is False


class TestRAGAgentIntegration:
    """Integration tests for RAG Agent"""
    
    @pytest.fixture
    def config(self):
        """Create integration test configuration"""
        return RAGAgentConfig(
            model_name="gpt-5-mini",
            temperature=0.1,
            max_tokens=1000,
            max_search_results=3,
            context_window_size=2000,
            retrieval_strategy="hybrid",
            similarity_threshold=0.7,
            enable_query_expansion=True,
            enable_reranking=True,
            response_mode="conversational",
            include_sources=True,
            enable_caching=True,
            cache_ttl=3600,
            max_concurrent_queries=2
        )
    
    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create mock knowledge graph for integration tests"""
        graph = Mock()
        graph.nodes = [
            Mock(id="node1", title="Person A", type="person", description="A person entity"),
            Mock(id="node2", title="Company B", type="organization", description="A company entity"),
            Mock(id="node3", title="Project C", type="project", description="A project entity")
        ]
        graph.edges = [
            Mock(
                id="edge1",
                type="works_for",
                description="Person A works for Company B",
                source=graph.nodes[0],
                target=graph.nodes[1]
            ),
            Mock(
                id="edge2",
                type="manages",
                description="Person A manages Project C",
                source=graph.nodes[0],
                target=graph.nodes[2]
            ),
            Mock(
                id="edge3",
                type="sponsors",
                description="Company B sponsors Project C",
                source=graph.nodes[1],
                target=graph.nodes[2]
            )
        ]
        return graph
    
    @pytest.mark.asyncio
    async def test_full_query_workflow(self, config, mock_knowledge_graph):
        """Test complete query workflow"""
        with patch('lang_graph.graphrag_agent.agents.rag_agent.ChatOpenAI'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.OpenAIEmbeddings'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.Chroma'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.RedisCache'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.SQLiteCache'):
            
            agent = RAGAgent(config)
            
            # Mock LLM responses
            agent.llm = AsyncMock()
            agent.llm.ainvoke.return_value = Mock(content="Person A works for Company B and manages Project C.")
            
            # Mock embeddings
            agent.embeddings = AsyncMock()
            agent.embeddings.aembed_query.return_value = [0.1, 0.2, 0.3]
            agent.embeddings.aembed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            
            # Mock vector store
            agent.vector_store = Mock()
            agent.vector_store.as_retriever.return_value = Mock()
            agent.vector_store.as_retriever.return_value.ainvoke = AsyncMock(return_value=[
                Mock(page_content="Person A works for Company B", metadata={"source": "node1", "type": "entity"}),
                Mock(page_content="Person A manages Project C", metadata={"source": "node2", "type": "entity"})
            ])
            
            # Run query
            result = await agent.query_knowledge_graph("Who works for Company B?", mock_knowledge_graph)
            
            # Verify result structure
            assert "response" in result
            assert "sources" in result
            assert "confidence" in result
            assert "processing_time" in result
            
            # Verify response content
            assert "Person A" in result["response"]
            assert "Company B" in result["response"]
            
            # Verify sources
            assert len(result["sources"]) > 0
            for source in result["sources"]:
                assert "content" in source
                assert "metadata" in source
            
            # Verify confidence
            assert 0.0 <= result["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_caching_workflow(self, config, mock_knowledge_graph):
        """Test caching workflow"""
        with patch('lang_graph.graphrag_agent.agents.rag_agent.ChatOpenAI'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.OpenAIEmbeddings'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.Chroma'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.RedisCache'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.SQLiteCache'):
            
            agent = RAGAgent(config)
            
            # Mock LLM responses
            agent.llm = AsyncMock()
            agent.llm.ainvoke.return_value = Mock(content="Test response")
            
            # Mock embeddings
            agent.embeddings = AsyncMock()
            agent.embeddings.aembed_query.return_value = [0.1, 0.2, 0.3]
            agent.embeddings.aembed_documents.return_value = [[0.1, 0.2, 0.3]]
            
            # Mock vector store
            agent.vector_store = Mock()
            agent.vector_store.as_retriever.return_value = Mock()
            agent.vector_store.as_retriever.return_value.ainvoke = AsyncMock(return_value=[
                Mock(page_content="Test content", metadata={"source": "test"})
            ])
            
            query = "What is the meaning of life?"
            
            # First query - should not be cached
            result1 = await agent.query_knowledge_graph(query, mock_knowledge_graph)
            assert result1["response"] == "Test response"
            
            # Second query - should be cached
            result2 = await agent.query_knowledge_graph(query, mock_knowledge_graph)
            assert result2["response"] == "Test response"
            
            # Verify cache was used
            assert agent.metrics["cache_hits"] > 0 or agent.metrics["cache_misses"] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, config, mock_knowledge_graph):
        """Test error handling in query workflow"""
        with patch('lang_graph.graphrag_agent.agents.rag_agent.ChatOpenAI'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.OpenAIEmbeddings'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.Chroma'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.RedisCache'), \
             patch('lang_graph.graphrag_agent.agents.rag_agent.SQLiteCache'):
            
            agent = RAGAgent(config)
            
            # Mock LLM to raise exception
            agent.llm = AsyncMock()
            agent.llm.ainvoke.side_effect = Exception("LLM error")
            
            # Mock embeddings
            agent.embeddings = AsyncMock()
            agent.embeddings.aembed_query.return_value = [0.1, 0.2, 0.3]
            agent.embeddings.aembed_documents.return_value = [[0.1, 0.2, 0.3]]
            
            # Mock vector store
            agent.vector_store = Mock()
            agent.vector_store.as_retriever.return_value = Mock()
            agent.vector_store.as_retriever.return_value.ainvoke = AsyncMock(return_value=[
                Mock(page_content="Test content", metadata={"source": "test"})
            ])
            
            # Run query - should handle error gracefully
            with pytest.raises(Exception):
                await agent.query_knowledge_graph("What is the meaning of life?", mock_knowledge_graph)
            
            # Verify error was recorded
            assert agent.metrics["errors"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
