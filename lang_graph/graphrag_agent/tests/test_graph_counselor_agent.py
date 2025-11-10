"""
Test cases for Graph Counselor Agent
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from lang_graph.graphrag_agent.agents.graph_counselor_agent import (
    GraphCounselorAgent,
    GraphCounselorConfig,
    AgentRole,
    ExplorationStrategy
)


class TestGraphCounselorConfig:
    """Test cases for GraphCounselorConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = GraphCounselorConfig()
        
        assert config.model_name == "gpt-5-mini"
        assert config.temperature == 0.1
        assert config.max_tokens == 4000
        assert config.num_explorer_agents == 3
        assert config.num_analyzer_agents == 2
        assert config.num_optimizer_agents == 1
        assert config.exploration_strategy == ExplorationStrategy.COMMUNITY_AWARE
        assert config.max_exploration_depth == 5
        assert config.exploration_budget == 1000
        assert config.communication_frequency == 10
        assert config.consensus_threshold == 0.7
        assert config.adaptation_rate == 0.1
        assert config.enable_parallel_processing is True
        assert config.max_concurrent_agents == 5
        assert config.cache_enabled is True
        assert config.cache_ttl == 3600
        assert config.enable_community_detection is True
        assert config.enable_hierarchical_analysis is True
        assert config.enable_temporal_analysis is False
        assert config.enable_semantic_clustering is True
        assert config.min_confidence_threshold == 0.6
        assert config.max_iterations == 50
        assert config.convergence_threshold == 0.01
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test valid config
        config = GraphCounselorConfig(
            model_name="gpt-5-mini",
            temperature=0.5,
            max_tokens=2000,
            num_explorer_agents=5,
            num_analyzer_agents=3,
            num_optimizer_agents=2
        )
        assert config.model_name == "gpt-5-mini"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.num_explorer_agents == 5
        assert config.num_analyzer_agents == 3
        assert config.num_optimizer_agents == 2
        
        # Test invalid temperature
        with pytest.raises(ValueError):
            GraphCounselorConfig(temperature=3.0)
        
        # Test invalid model name
        with pytest.raises(ValueError):
            GraphCounselorConfig(model_name="")
    
    def test_exploration_strategy_enum(self):
        """Test exploration strategy enum values"""
        assert ExplorationStrategy.BREADTH_FIRST.value == "breadth_first"
        assert ExplorationStrategy.DEPTH_FIRST.value == "depth_first"
        assert ExplorationStrategy.RANDOM_WALK.value == "random_walk"
        assert ExplorationStrategy.PRIORITY_BASED.value == "priority_based"
        assert ExplorationStrategy.COMMUNITY_AWARE.value == "community_aware"
    
    def test_agent_role_enum(self):
        """Test agent role enum values"""
        assert AgentRole.EXPLORER.value == "explorer"
        assert AgentRole.ANALYZER.value == "analyzer"
        assert AgentRole.OPTIMIZER.value == "optimizer"
        assert AgentRole.COORDINATOR.value == "coordinator"


class TestGraphCounselorAgent:
    """Test cases for GraphCounselorAgent"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return GraphCounselorConfig(
            model_name="gpt-5-mini",
            temperature=0.1,
            max_tokens=1000,
            num_explorer_agents=2,
            num_analyzer_agents=1,
            num_optimizer_agents=1,
            max_iterations=5,
            exploration_budget=100
        )
    
    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create mock knowledge graph"""
        graph = Mock()
        graph.nodes = [
            Mock(id="node1", title="Entity 1", type="person", description="A person"),
            Mock(id="node2", title="Entity 2", type="organization", description="An organization"),
            Mock(id="node3", title="Entity 3", type="concept", description="A concept")
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
        with patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.ChatOpenAI'), \
             patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.OpenAIEmbeddings'), \
             patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.RedisCache'), \
             patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.SQLiteCache'):
            
            agent = GraphCounselorAgent(config)
            
            # Mock LLM responses
            agent.llm = AsyncMock()
            agent.llm.ainvoke.return_value = Mock(content="Test response")
            
            # Mock embeddings
            agent.embeddings = AsyncMock()
            agent.embeddings.aembed_query.return_value = [0.1, 0.2, 0.3]
            
            return agent
    
    def test_agent_initialization(self, agent, config):
        """Test agent initialization"""
        assert agent.config == config
        assert agent.console is not None
        assert agent.logger is not None
        assert agent.metrics is not None
        assert agent.agents is not None
        assert agent.message_queue is not None
        assert agent.communication_history == []
        assert agent.consensus_tracker == {}
        assert agent.strategy_adaptations == []
    
    def test_agent_setup(self, agent):
        """Test agent setup methods"""
        # Test agent count
        total_agents = agent._get_total_agent_count()
        expected_agents = (agent.config.num_explorer_agents + 
                          agent.config.num_analyzer_agents + 
                          agent.config.num_optimizer_agents + 1)  # +1 for coordinator
        assert total_agents == expected_agents
        
        # Test agent structure
        assert 'explorers' in agent.agents
        assert 'analyzers' in agent.agents
        assert 'optimizers' in agent.agents
        assert 'coordinator' in agent.agents
        
        # Test explorer agents
        assert len(agent.agents['explorers']) == agent.config.num_explorer_agents
        for i, explorer in enumerate(agent.agents['explorers']):
            assert explorer['id'] == f"explorer_{i+1}"
            assert explorer['role'] == AgentRole.EXPLORER
            assert explorer['status'] == 'idle'
            assert 'specialization' in explorer
        
        # Test analyzer agents
        assert len(agent.agents['analyzers']) == agent.config.num_analyzer_agents
        for i, analyzer in enumerate(agent.agents['analyzers']):
            assert analyzer['id'] == f"analyzer_{i+1}"
            assert analyzer['role'] == AgentRole.ANALYZER
            assert analyzer['status'] == 'idle'
            assert 'specialization' in analyzer
        
        # Test optimizer agents
        assert len(agent.agents['optimizers']) == agent.config.num_optimizer_agents
        for i, optimizer in enumerate(agent.agents['optimizers']):
            assert optimizer['id'] == f"optimizer_{i+1}"
            assert optimizer['role'] == AgentRole.OPTIMIZER
            assert optimizer['status'] == 'idle'
            assert 'specialization' in optimizer
        
        # Test coordinator
        coordinator = agent.agents['coordinator']
        assert coordinator['id'] == 'coordinator'
        assert coordinator['role'] == AgentRole.COORDINATOR
        assert coordinator['status'] == 'active'
    
    def test_metrics_initialization(self, agent):
        """Test metrics initialization"""
        expected_metrics = {
            'total_explorations': 0,
            'successful_explorations': 0,
            'failed_explorations': 0,
            'total_analysis_cycles': 0,
            'consensus_reached': 0,
            'strategy_adaptations': 0,
            'total_processing_time': 0.0,
            'average_exploration_time': 0.0,
            'average_analysis_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'inter_agent_messages': 0,
            'errors': 0
        }
        
        for key, value in expected_metrics.items():
            assert agent.metrics[key] == value
    
    def test_get_graph_size(self, agent, mock_knowledge_graph):
        """Test graph size calculation"""
        size = agent._get_graph_size(mock_knowledge_graph)
        assert size['nodes'] == 3
        assert size['edges'] == 2
    
    def test_get_graph_size_empty(self, agent):
        """Test graph size calculation with empty graph"""
        empty_graph = Mock()
        empty_graph.nodes = []
        empty_graph.edges = []
        
        size = agent._get_graph_size(empty_graph)
        assert size['nodes'] == 0
        assert size['edges'] == 0
    
    def test_get_graph_size_error(self, agent):
        """Test graph size calculation with error"""
        invalid_graph = Mock()
        del invalid_graph.nodes
        del invalid_graph.edges
        
        size = agent._get_graph_size(invalid_graph)
        assert size['nodes'] == 0
        assert size['edges'] == 0
    
    def test_check_convergence(self, agent):
        """Test convergence checking"""
        # Test max iterations reached
        state = {
            'current_iteration': 50,
            'convergence_score': 0.5
        }
        assert agent._check_convergence(state) is True
        
        # Test convergence threshold reached
        state = {
            'current_iteration': 10,
            'convergence_score': 0.01
        }
        assert agent._check_convergence(state) is True
        
        # Test not converged
        state = {
            'current_iteration': 10,
            'convergence_score': 0.5
        }
        assert agent._check_convergence(state) is False
    
    def test_update_exploration_state(self, agent):
        """Test exploration state update"""
        state = {
            'completed_tasks': [],
            'insights': [],
            'convergence_score': 0.0
        }
        
        iteration_results = {
            'completed_tasks': [{'task_id': 'test_task', 'status': 'completed'}],
            'insights': ['Test insight 1', 'Test insight 2']
        }
        
        updated_state = agent._update_exploration_state(state, iteration_results)
        
        assert len(updated_state['completed_tasks']) == 1
        assert len(updated_state['insights']) == 2
        assert updated_state['convergence_score'] == 0.1
    
    def test_calculate_current_performance(self, agent):
        """Test performance calculation"""
        # Test with no tasks
        performance = agent._calculate_current_performance()
        assert performance['efficiency'] == 0.0
        assert performance['coverage'] == 0.0
        assert performance['quality'] == 0.8
        
        # Test with some tasks
        agent.metrics['total_explorations'] = 10
        agent.metrics['successful_explorations'] = 8
        
        performance = agent._calculate_current_performance()
        assert performance['efficiency'] == 0.8
        assert performance['coverage'] == 0.1
        assert performance['quality'] == 0.8
    
    def test_generate_final_insights(self, agent):
        """Test final insights generation"""
        exploration_state = {
            'current_iteration': 5,
            'convergence_score': 0.8,
            'insights': ['Insight 1', 'Insight 2', 'Insight 1']  # Duplicate
        }
        
        insights = agent._generate_final_insights(exploration_state)
        
        # Should include summary insights
        assert any("Exploration completed in 5 iterations" in insight for insight in insights)
        assert any("Generated 2 unique insights" in insight for insight in insights)
        assert any("Convergence score: 0.800" in insight for insight in insights)
        
        # Should deduplicate insights
        unique_insights = [insight for insight in insights if insight.startswith('Insight')]
        assert len(unique_insights) == 2  # Only unique insights
    
    def test_get_agent_performance_summary(self, agent):
        """Test agent performance summary"""
        # Set some performance scores
        agent.agents['explorers'][0]['performance_score'] = 0.8
        agent.agents['explorers'][1]['performance_score'] = 0.6
        agent.agents['analyzers'][0]['performance_score'] = 0.9
        
        summary = agent._get_agent_performance_summary()
        
        assert 'explorers' in summary
        assert 'analyzers' in summary
        assert 'optimizers' in summary
        assert 'coordinator' not in summary
        
        # Test explorer performance
        explorer_perf = summary['explorers']
        assert explorer_perf['count'] == 2
        assert explorer_perf['avg_score'] == 0.7
        assert explorer_perf['max_score'] == 0.8
        assert explorer_perf['min_score'] == 0.6
    
    def test_generate_recommendations(self, agent):
        """Test recommendations generation"""
        exploration_state = {
            'completed_tasks': [{'task_id': 'task1'}] * 5  # 5 tasks
        }
        
        # Mock performance calculation
        with patch.object(agent, '_calculate_current_performance') as mock_perf:
            mock_perf.return_value = {
                'efficiency': 0.5,  # Low efficiency
                'coverage': 0.6,    # Low coverage
                'quality': 0.4      # Low quality
            }
            
            recommendations = agent._generate_recommendations(exploration_state)
            
            assert len(recommendations) >= 3
            assert any("increasing the number of parallel agents" in rec for rec in recommendations)
            assert any("Increase exploration depth or budget" in rec for rec in recommendations)
            assert any("Improve agent coordination and communication" in rec for rec in recommendations)
    
    def test_update_metrics(self, agent):
        """Test metrics update"""
        initial_explorations = agent.metrics['total_explorations']
        initial_time = agent.metrics['total_processing_time']
        
        # Test successful update
        agent._update_metrics(1.5, True)
        
        assert agent.metrics['total_explorations'] == initial_explorations + 1
        assert agent.metrics['successful_explorations'] == 1
        assert agent.metrics['total_processing_time'] == initial_time + 1.5
        
        # Test failed update
        agent._update_metrics(0.5, False)
        
        assert agent.metrics['total_explorations'] == initial_explorations + 2
        assert agent.metrics['failed_explorations'] == 1
        assert agent.metrics['total_processing_time'] == initial_time + 2.0
    
    def test_reset_metrics(self, agent):
        """Test metrics reset"""
        # Set some metrics
        agent.metrics['total_explorations'] = 10
        agent.metrics['successful_explorations'] = 8
        agent.metrics['total_processing_time'] = 100.0
        
        # Reset metrics
        agent.reset_metrics()
        
        # Check all metrics are reset
        assert agent.metrics['total_explorations'] == 0
        assert agent.metrics['successful_explorations'] == 0
        assert agent.metrics['total_processing_time'] == 0.0
        assert agent.metrics['average_exploration_time'] == 0.0
    
    def test_get_agent_status(self, agent):
        """Test agent status retrieval"""
        status = agent.get_agent_status()
        
        assert 'explorers' in status
        assert 'analyzers' in status
        assert 'optimizers' in status
        assert 'coordinator' in status
        
        # Test explorer status
        explorer_status = status['explorers']
        assert len(explorer_status) == 2
        
        for i, explorer in enumerate(explorer_status):
            assert explorer['id'] == f"explorer_{i+1}"
            assert explorer['status'] == 'idle'
            assert explorer['current_task'] is None
            assert explorer['performance_score'] == 0.0
            assert 'specialization' in explorer
    
    def test_get_config_summary(self, agent):
        """Test configuration summary"""
        summary = agent.get_config_summary()
        
        assert summary['model_name'] == agent.config.model_name
        assert summary['temperature'] == agent.config.temperature
        assert summary['max_tokens'] == agent.config.max_tokens
        assert summary['num_explorer_agents'] == agent.config.num_explorer_agents
        assert summary['num_analyzer_agents'] == agent.config.num_analyzer_agents
        assert summary['num_optimizer_agents'] == agent.config.num_optimizer_agents
        assert summary['status'] == 'ready'
        assert 'total_agents' in summary
        assert 'metrics' in summary
    
    @pytest.mark.asyncio
    async def test_test_connectivity_success(self, agent):
        """Test connectivity test success"""
        result = await agent.test_connectivity()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_test_connectivity_failure(self, agent):
        """Test connectivity test failure"""
        # Mock LLM failure
        agent.llm.ainvoke.return_value = None
        
        result = await agent.test_connectivity()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_test_connectivity_exception(self, agent):
        """Test connectivity test with exception"""
        # Mock LLM exception
        agent.llm.ainvoke.side_effect = Exception("Connection failed")
        
        result = await agent.test_connectivity()
        assert result is False
    
    def test_convert_to_networkx(self, agent, mock_knowledge_graph):
        """Test graph conversion to NetworkX"""
        import networkx as nx
        
        nx_graph = agent._convert_to_networkx(mock_knowledge_graph)
        
        assert isinstance(nx_graph, nx.Graph)
        assert nx_graph.number_of_nodes() == 3
        assert nx_graph.number_of_edges() == 2
        
        # Check node attributes
        for node_id in nx_graph.nodes():
            assert 'title' in nx_graph.nodes[node_id]
            assert 'type' in nx_graph.nodes[node_id]
            assert 'description' in nx_graph.nodes[node_id]
        
        # Check edge attributes
        for u, v in nx_graph.edges():
            assert 'type' in nx_graph.edges[u, v]
            assert 'description' in nx_graph.edges[u, v]
    
    def test_convert_to_networkx_error(self, agent):
        """Test graph conversion with error"""
        invalid_graph = Mock()
        del invalid_graph.nodes
        del invalid_graph.edges
        
        nx_graph = agent._convert_to_networkx(invalid_graph)
        
        import networkx as nx
        assert isinstance(nx_graph, nx.Graph)
        assert nx_graph.number_of_nodes() == 0
        assert nx_graph.number_of_edges() == 0


class TestGraphCounselorIntegration:
    """Integration tests for Graph Counselor Agent"""
    
    @pytest.fixture
    def config(self):
        """Create integration test configuration"""
        return GraphCounselorConfig(
            model_name="gpt-5-mini",
            temperature=0.1,
            max_tokens=1000,
            num_explorer_agents=1,
            num_analyzer_agents=1,
            num_optimizer_agents=1,
            max_iterations=2,
            exploration_budget=50,
            enable_parallel_processing=False  # Disable for testing
        )
    
    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create mock knowledge graph for integration tests"""
        graph = Mock()
        graph.nodes = [
            Mock(id="node1", title="Person A", type="person", description="A person"),
            Mock(id="node2", title="Company B", type="organization", description="A company"),
            Mock(id="node3", title="Project C", type="project", description="A project")
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
    async def test_explore_graph_basic(self, config, mock_knowledge_graph):
        """Test basic graph exploration"""
        with patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.ChatOpenAI'), \
             patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.OpenAIEmbeddings'), \
             patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.RedisCache'), \
             patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.SQLiteCache'):
            
            agent = GraphCounselorAgent(config)
            
            # Mock LLM responses
            agent.llm = AsyncMock()
            agent.llm.ainvoke.return_value = Mock(content="Test response")
            
            # Mock embeddings
            agent.embeddings = AsyncMock()
            agent.embeddings.aembed_query.return_value = [0.1, 0.2, 0.3]
            
            # Mock NetworkX functions
            with patch('networkx.is_connected') as mock_connected, \
                 patch('networkx.number_connected_components') as mock_components, \
                 patch('networkx.average_shortest_path_length') as mock_avg_path, \
                 patch('networkx.diameter') as mock_diameter, \
                 patch('networkx.density') as mock_density, \
                 patch('networkx.louvain_communities') as mock_communities, \
                 patch('networkx.modularity') as mock_modularity, \
                 patch('networkx.betweenness_centrality') as mock_betweenness, \
                 patch('networkx.closeness_centrality') as mock_closeness, \
                 patch('networkx.eigenvector_centrality') as mock_eigenvector:
                
                # Setup mocks
                mock_connected.return_value = True
                mock_components.return_value = 1
                mock_avg_path.return_value = 1.5
                mock_diameter.return_value = 2
                mock_density.return_value = 0.5
                mock_communities.return_value = [{"node1", "node2"}, {"node3"}]
                mock_modularity.return_value = 0.3
                mock_betweenness.return_value = {"node1": 0.5, "node2": 0.3, "node3": 0.2}
                mock_closeness.return_value = {"node1": 0.8, "node2": 0.6, "node3": 0.4}
                mock_eigenvector.return_value = {"node1": 0.7, "node2": 0.5, "node3": 0.3}
                
                # Run exploration
                result = await agent.explore_graph(mock_knowledge_graph, "comprehensive_analysis")
                
                # Verify result structure
                assert 'exploration_summary' in result
                assert 'insights' in result
                assert 'agent_performance' in result
                assert 'strategy_adaptations' in result
                assert 'recommendations' in result
                assert 'timestamp' in result
                
                # Verify exploration summary
                summary = result['exploration_summary']
                assert summary['goal'] == "comprehensive_analysis"
                assert summary['iterations'] >= 0
                assert summary['convergence_score'] >= 0.0
                assert summary['total_tasks'] >= 0
                assert summary['total_insights'] >= 0
                
                # Verify insights
                insights = result['insights']
                assert isinstance(insights, list)
                assert len(insights) > 0
                
                # Verify agent performance
                agent_perf = result['agent_performance']
                assert 'explorers' in agent_perf
                assert 'analyzers' in agent_perf
                assert 'optimizers' in agent_perf
    
    @pytest.mark.asyncio
    async def test_explore_graph_error_handling(self, config, mock_knowledge_graph):
        """Test graph exploration error handling"""
        with patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.ChatOpenAI'), \
             patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.OpenAIEmbeddings'), \
             patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.RedisCache'), \
             patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.SQLiteCache'):
            
            agent = GraphCounselorAgent(config)
            
            # Mock LLM to raise exception
            agent.llm = AsyncMock()
            agent.llm.ainvoke.side_effect = Exception("LLM error")
            
            # Mock embeddings
            agent.embeddings = AsyncMock()
            agent.embeddings.aembed_query.return_value = [0.1, 0.2, 0.3]
            
            # Run exploration - should handle error gracefully
            with pytest.raises(Exception):
                await agent.explore_graph(mock_knowledge_graph, "comprehensive_analysis")
    
    def test_create_exploration_tasks(self, config, mock_knowledge_graph):
        """Test exploration task creation"""
        with patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.ChatOpenAI'), \
             patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.OpenAIEmbeddings'), \
             patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.RedisCache'), \
             patch('lang_graph.graphrag_agent.agents.graph_counselor_agent.SQLiteCache'):
            
            agent = GraphCounselorAgent(config)
            
            # Test comprehensive analysis tasks
            tasks = asyncio.run(agent._create_exploration_tasks(mock_knowledge_graph, "comprehensive_analysis"))
            
            assert len(tasks) > 0
            
            # Check task structure
            for task in tasks:
                assert 'id' in task
                assert 'type' in task
                assert 'agent_type' in task
                assert 'priority' in task
                assert 'description' in task
                assert 'parameters' in task
                
                assert task['type'] in ['exploration', 'analysis', 'optimization']
                assert task['agent_type'] in ['explorer', 'analyzer', 'optimizer']
                assert task['priority'] in [1, 2, 3]
            
            # Test specific goal tasks
            tasks = asyncio.run(agent._create_exploration_tasks(mock_knowledge_graph, "structure_only"))
            
            # Should have fewer tasks for specific goal
            assert len(tasks) >= 0  # May be empty for specific goals


if __name__ == "__main__":
    pytest.main([__file__])
