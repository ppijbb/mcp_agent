"""
Tests for Graph Generator Agent
"""

import pytest
import pandas as pd
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from graphrag_agent.agents.graph_generator_agent import (
    GraphGeneratorAgent,
    GraphGeneratorConfig
)


@pytest.fixture
def sample_data():
    """Sample data for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'document_id': ['doc_1', 'doc_1', 'doc_2', 'doc_2', 'doc_3'],
        'text_unit': [
            'Apple Inc. is a technology company based in Cupertino, California.',
            'Tim Cook is the current CEO of Apple Inc.',
            'Microsoft Corporation is an American multinational technology corporation.',
            'Satya Nadella is the CEO of Microsoft.',
            'Google LLC is an American multinational technology company.'
        ]
    })


@pytest.fixture
def config():
    """Test configuration"""
    return GraphGeneratorConfig(
        openai_api_key="test-key",
        model_name="gpt-5-mini",
        temperature=0.0,
        max_concurrency=2,
        batch_size=3,
        enable_graph_optimization=False,
        save_intermediate_results=False
    )


@pytest.fixture
def mock_knowledge_graph():
    """Mock knowledge graph for testing"""
    mock_graph = Mock()
    mock_graph.nodes = [
        Mock(id="1", type="organization", title="Apple Inc.", description="Technology company"),
        Mock(id="2", type="person", title="Tim Cook", description="CEO of Apple"),
        Mock(id="3", type="organization", title="Microsoft", description="Technology corporation"),
        Mock(id="4", type="person", title="Satya Nadella", description="CEO of Microsoft")
    ]
    mock_graph.edges = [
        Mock(source=Mock(id="2"), target=Mock(id="1"), type="works_for", description="Tim Cook works for Apple"),
        Mock(source=Mock(id="4"), target=Mock(id="3"), type="works_for", description="Satya Nadella works for Microsoft")
    ]
    return mock_graph


class TestGraphGeneratorConfig:
    """Test GraphGeneratorConfig validation"""
    
    def test_valid_config(self):
        """Test valid configuration"""
        config = GraphGeneratorConfig(
            openai_api_key="test-key",
            model_name="gpt-5-mini",
            temperature=0.5,
            max_concurrency=4
        )
        assert config.openai_api_key == "test-key"
        assert config.model_name == "gpt-5-mini"
        assert config.temperature == 0.5
        assert config.max_concurrency == 4
    
    def test_invalid_temperature(self):
        """Test invalid temperature validation"""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            GraphGeneratorConfig(
                openai_api_key="test-key",
                temperature=3.0
            )
    
    def test_invalid_max_concurrency(self):
        """Test invalid max_concurrency validation"""
        with pytest.raises(ValueError, match="max_concurrency must be between 1 and 20"):
            GraphGeneratorConfig(
                openai_api_key="test-key",
                max_concurrency=25
            )
    
    def test_invalid_cache_type(self):
        """Test invalid cache_type validation"""
        with pytest.raises(ValueError, match="cache_type must be one of"):
            GraphGeneratorConfig(
                openai_api_key="test-key",
                cache_type="invalid"
            )


class TestGraphGeneratorAgent:
    """Test GraphGeneratorAgent functionality"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, config):
        """Test agent initialization"""
        with patch('graphrag_agent.agents.graph_generator_agent.ChatOpenAI'), \
             patch('graphrag_agent.agents.graph_generator_agent.OpenAIEmbeddings'), \
             patch('graphrag_agent.agents.graph_generator_agent.EntityRelationshipExtractor'), \
             patch('graphrag_agent.agents.graph_generator_agent.GraphGenerator'):
            
            agent = GraphGeneratorAgent(config)
            assert agent.config == config
            assert agent.metrics['processing_time'] == 0.0
            assert agent.metrics['entities_extracted'] == 0
    
    @pytest.mark.asyncio
    async def test_validate_input_async_valid(self, config, sample_data):
        """Test input validation with valid data"""
        with patch('graphrag_agent.agents.graph_generator_agent.ChatOpenAI'), \
             patch('graphrag_agent.agents.graph_generator_agent.OpenAIEmbeddings'), \
             patch('graphrag_agent.agents.graph_generator_agent.EntityRelationshipExtractor'), \
             patch('graphrag_agent.agents.graph_generator_agent.GraphGenerator'):
            
            agent = GraphGeneratorAgent(config)
            result = await agent._validate_input_async(sample_data)
            
            assert result['status'] == 'valid'
            assert 'stats' in result
    
    @pytest.mark.asyncio
    async def test_validate_input_async_empty(self, config):
        """Test input validation with empty data"""
        with patch('graphrag_agent.agents.graph_generator_agent.ChatOpenAI'), \
             patch('graphrag_agent.agents.graph_generator_agent.OpenAIEmbeddings'), \
             patch('graphrag_agent.agents.graph_generator_agent.EntityRelationshipExtractor'), \
             patch('graphrag_agent.agents.graph_generator_agent.GraphGenerator'):
            
            agent = GraphGeneratorAgent(config)
            empty_data = pd.DataFrame()
            result = await agent._validate_input_async(empty_data)
            
            assert result['status'] == 'error'
            assert 'No text units provided' in result['error']
    
    @pytest.mark.asyncio
    async def test_validate_input_async_missing_columns(self, config):
        """Test input validation with missing columns"""
        with patch('graphrag_agent.agents.graph_generator_agent.ChatOpenAI'), \
             patch('graphrag_agent.agents.graph_generator_agent.OpenAIEmbeddings'), \
             patch('graphrag_agent.agents.graph_generator_agent.EntityRelationshipExtractor'), \
             patch('graphrag_agent.agents.graph_generator_agent.GraphGenerator'):
            
            agent = GraphGeneratorAgent(config)
            invalid_data = pd.DataFrame({'id': [1, 2], 'text': ['text1', 'text2']})
            result = await agent._validate_input_async(invalid_data)
            
            assert result['status'] == 'error'
            assert 'Missing required columns' in result['error']
    
    @pytest.mark.asyncio
    async def test_preprocess_data_async(self, config, sample_data):
        """Test data preprocessing"""
        with patch('graphrag_agent.agents.graph_generator_agent.ChatOpenAI'), \
             patch('graphrag_agent.agents.graph_generator_agent.OpenAIEmbeddings'), \
             patch('graphrag_agent.agents.graph_generator_agent.EntityRelationshipExtractor'), \
             patch('graphrag_agent.agents.graph_generator_agent.GraphGenerator'):
            
            agent = GraphGeneratorAgent(config)
            result = await agent._preprocess_data_async(sample_data)
            
            assert len(result) == len(sample_data)
            assert 'text_length' in result.columns
            assert 'word_count' in result.columns
            assert 'sentence_count' in result.columns
            assert 'complexity_score' in result.columns
    
    @pytest.mark.asyncio
    async def test_validate_generated_graph_async_valid(self, config, mock_knowledge_graph):
        """Test graph validation with valid graph"""
        with patch('graphrag_agent.agents.graph_generator_agent.ChatOpenAI'), \
             patch('graphrag_agent.agents.graph_generator_agent.OpenAIEmbeddings'), \
             patch('graphrag_agent.agents.graph_generator_agent.EntityRelationshipExtractor'), \
             patch('graphrag_agent.agents.graph_generator_agent.GraphGenerator'):
            
            agent = GraphGeneratorAgent(config)
            result = await agent._validate_generated_graph_async(mock_knowledge_graph)
            
            assert result['status'] in ['valid', 'valid_with_warnings']
    
    @pytest.mark.asyncio
    async def test_validate_generated_graph_async_none(self, config):
        """Test graph validation with None graph"""
        with patch('graphrag_agent.agents.graph_generator_agent.ChatOpenAI'), \
             patch('graphrag_agent.agents.graph_generator_agent.OpenAIEmbeddings'), \
             patch('graphrag_agent.agents.graph_generator_agent.EntityRelationshipExtractor'), \
             patch('graphrag_agent.agents.graph_generator_agent.GraphGenerator'):
            
            agent = GraphGeneratorAgent(config)
            result = await agent._validate_generated_graph_async(None)
            
            assert result['status'] == 'error'
            assert 'Generated graph is None' in result['error']
    
    @pytest.mark.asyncio
    async def test_calculate_advanced_graph_stats(self, config, mock_knowledge_graph):
        """Test advanced graph statistics calculation"""
        with patch('graphrag_agent.agents.graph_generator_agent.ChatOpenAI'), \
             patch('graphrag_agent.agents.graph_generator_agent.OpenAIEmbeddings'), \
             patch('graphrag_agent.agents.graph_generator_agent.EntityRelationshipExtractor'), \
             patch('graphrag_agent.agents.graph_generator_agent.GraphGenerator'), \
             patch('graphrag_agent.agents.graph_generator_agent.nx') as mock_nx:
            
            # Mock NetworkX graph
            mock_G = Mock()
            mock_G.number_of_nodes.return_value = 4
            mock_G.number_of_edges.return_value = 2
            mock_G.degree.return_value = [(1, 2), (2, 1), (3, 1), (4, 1)]
            mock_nx.density.return_value = 0.5
            mock_nx.number_connected_components.return_value = 1
            mock_nx.average_clustering.return_value = 0.3
            mock_nx.is_connected.return_value = True
            mock_nx.degree_assortativity_coefficient.return_value = 0.1
            mock_nx.transitivity.return_value = 0.2
            
            agent = GraphGeneratorAgent(config)
            agent._convert_to_networkx = AsyncMock(return_value=mock_G)
            
            result = await agent._calculate_advanced_graph_stats(mock_knowledge_graph)
            
            assert 'nodes' in result
            assert 'edges' in result
            assert 'density' in result
            assert 'connected_components' in result
            assert 'overall_quality_score' in result
    
    @pytest.mark.asyncio
    async def test_test_connectivity_success(self, config):
        """Test successful connectivity test"""
        with patch('graphrag_agent.agents.graph_generator_agent.ChatOpenAI') as mock_chat:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Hello"
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_chat.return_value = mock_llm
            
            agent = GraphGeneratorAgent(config)
            result = await agent.test_connectivity()
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_test_connectivity_failure(self, config):
        """Test failed connectivity test"""
        with patch('graphrag_agent.agents.graph_generator_agent.ChatOpenAI') as mock_chat:
            mock_llm = Mock()
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))
            mock_chat.return_value = mock_llm
            
            agent = GraphGeneratorAgent(config)
            result = await agent.test_connectivity()
            
            assert result is False
    
    def test_get_config_summary(self, config):
        """Test configuration summary"""
        with patch('graphrag_agent.agents.graph_generator_agent.ChatOpenAI'), \
             patch('graphrag_agent.agents.graph_generator_agent.OpenAIEmbeddings'), \
             patch('graphrag_agent.agents.graph_generator_agent.EntityRelationshipExtractor'), \
             patch('graphrag_agent.agents.graph_generator_agent.GraphGenerator'):
            
            agent = GraphGeneratorAgent(config)
            summary = agent.get_config_summary()
            
            assert 'model_name' in summary
            assert 'temperature' in summary
            assert 'max_concurrency' in summary
            assert 'status' in summary
            assert summary['status'] == 'ready'
    
    def test_get_metrics(self, config):
        """Test metrics retrieval"""
        with patch('graphrag_agent.agents.graph_generator_agent.ChatOpenAI'), \
             patch('graphrag_agent.agents.graph_generator_agent.OpenAIEmbeddings'), \
             patch('graphrag_agent.agents.graph_generator_agent.EntityRelationshipExtractor'), \
             patch('graphrag_agent.agents.graph_generator_agent.GraphGenerator'):
            
            agent = GraphGeneratorAgent(config)
            metrics = agent.get_metrics()
            
            assert 'processing_time' in metrics
            assert 'entities_extracted' in metrics
            assert 'api_calls' in metrics
            assert 'errors' in metrics
    
    def test_reset_metrics(self, config):
        """Test metrics reset"""
        with patch('graphrag_agent.agents.graph_generator_agent.ChatOpenAI'), \
             patch('graphrag_agent.agents.graph_generator_agent.OpenAIEmbeddings'), \
             patch('graphrag_agent.agents.graph_generator_agent.EntityRelationshipExtractor'), \
             patch('graphrag_agent.agents.graph_generator_agent.GraphGenerator'):
            
            agent = GraphGeneratorAgent(config)
            agent.metrics['processing_time'] = 10.0
            agent.metrics['api_calls'] = 5
            
            agent.reset_metrics()
            
            assert agent.metrics['processing_time'] == 0.0
            assert agent.metrics['api_calls'] == 0
            assert agent.metrics['errors'] == 0


@pytest.mark.integration
class TestGraphGeneratorAgentIntegration:
    """Integration tests for GraphGeneratorAgent"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_processing_pipeline(self, config, sample_data):
        """Test the full processing pipeline with mocked components"""
        with patch('graphrag_agent.agents.graph_generator_agent.ChatOpenAI'), \
             patch('graphrag_agent.agents.graph_generator_agent.OpenAIEmbeddings'), \
             patch('graphrag_agent.agents.graph_generator_agent.EntityRelationshipExtractor'), \
             patch('graphrag_agent.agents.graph_generator_agent.GraphGenerator') as mock_generator, \
             patch('graphrag_agent.agents.graph_generator_agent.nx') as mock_nx:
            
            # Mock the graph generator to return a mock graph
            mock_graph = Mock()
            mock_graph.nodes = [Mock(id="1", type="organization", title="Apple Inc.")]
            mock_graph.edges = [Mock(source=Mock(id="1"), target=Mock(id="2"), type="related_to")]
            mock_generator.return_value.invoke.return_value = mock_graph
            
            # Mock NetworkX operations
            mock_G = Mock()
            mock_G.number_of_nodes.return_value = 1
            mock_G.number_of_edges.return_value = 1
            mock_G.degree.return_value = [(1, 1)]
            mock_nx.density.return_value = 0.5
            mock_nx.number_connected_components.return_value = 1
            mock_nx.average_clustering.return_value = 0.0
            mock_nx.is_connected.return_value = True
            mock_nx.degree_assortativity_coefficient.return_value = 0.0
            mock_nx.transitivity.return_value = 0.0
            
            agent = GraphGeneratorAgent(config)
            agent._convert_to_networkx = AsyncMock(return_value=mock_G)
            
            result = await agent.process_text_units(sample_data)
            
            assert result['status'] == 'completed'
            assert 'knowledge_graph' in result
            assert 'stats' in result
            assert 'metrics' in result
            assert 'processing_info' in result


if __name__ == "__main__":
    pytest.main([__file__])
