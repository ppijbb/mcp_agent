"""
Graph Counselor Agent - Multi-Agent Collaboration for Adaptive Graph Exploration

This agent implements the Graph Counselor approach from recent research,
enabling multi-agent collaboration for complex graph structure modeling
and dynamic information extraction strategy adjustment.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import networkx as nx
from pydantic import BaseModel, Field, field_validator, ConfigDict
from rich.progress import Progress, TaskID

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma, FAISS

from .base_agent import BaseAgent, BaseAgentConfig


class AgentRole(Enum):
    """Roles for different agents in the collaboration"""
    EXPLORER = "explorer"  # Explores graph structure
    ANALYZER = "analyzer"  # Analyzes patterns and relationships
    OPTIMIZER = "optimizer"  # Optimizes extraction strategies
    COORDINATOR = "coordinator"  # Coordinates between agents


class ExplorationStrategy(Enum):
    """Different graph exploration strategies"""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    RANDOM_WALK = "random_walk"
    PRIORITY_BASED = "priority_based"
    COMMUNITY_AWARE = "community_aware"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    sender: str
    receiver: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=low, 2=medium, 3=high


class GraphCounselorConfig(BaseAgentConfig):
    """Advanced Configuration for Graph Counselor Agent"""
    
    model_config = ConfigDict(extra='forbid', validate_assignment=True)
    
    # Multi-agent settings
    num_explorer_agents: int = Field(default=3, ge=1, le=10, description="Number of explorer agents")
    num_analyzer_agents: int = Field(default=2, ge=1, le=5, description="Number of analyzer agents")
    num_optimizer_agents: int = Field(default=1, ge=1, le=3, description="Number of optimizer agents")
    
    # Exploration settings
    exploration_strategy: ExplorationStrategy = Field(
        default=ExplorationStrategy.COMMUNITY_AWARE,
        description="Graph exploration strategy"
    )
    max_exploration_depth: int = Field(default=5, ge=1, le=10, description="Maximum exploration depth")
    exploration_budget: int = Field(default=1000, ge=100, le=10000, description="Exploration budget")
    
    # Collaboration settings
    communication_frequency: int = Field(default=10, ge=1, le=100, description="Communication frequency")
    consensus_threshold: float = Field(default=0.7, ge=0.1, le=1.0, description="Consensus threshold")
    adaptation_rate: float = Field(default=0.1, ge=0.01, le=1.0, description="Strategy adaptation rate")
    
    # Performance settings
    max_concurrent_agents: int = Field(default=5, ge=1, le=20, description="Max concurrent agents")
    
    # Advanced features
    enable_community_detection: bool = Field(default=True, description="Enable community detection")
    enable_hierarchical_analysis: bool = Field(default=True, description="Enable hierarchical analysis")
    enable_temporal_analysis: bool = Field(default=False, description="Enable temporal analysis")
    enable_semantic_clustering: bool = Field(default=True, description="Enable semantic clustering")
    
    # Quality control
    min_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Min confidence threshold")
    max_iterations: int = Field(default=50, ge=1, le=200, description="Maximum iterations")
    convergence_threshold: float = Field(default=0.01, ge=0.001, le=0.1, description="Convergence threshold")
    


class GraphCounselorAgent(BaseAgent):
    """
    Advanced Multi-Agent Graph Counselor for Adaptive Graph Exploration
    
    This agent implements a multi-agent system where different specialized agents
    collaborate to explore, analyze, and optimize graph structures dynamically.
    """
    
    def __init__(self, config: GraphCounselorConfig):
        super().__init__(config)
        self._setup_agents()
        self._setup_communication()
        
        self.logger.info("Graph Counselor Agent initialized", 
                        config=config.model_dump(),
                        agent_count=self._get_total_agent_count())
    
    def _setup_metrics(self):
        """Setup performance metrics tracking"""
        super()._setup_metrics()
        # Add Graph Counselor specific metrics
        self.metrics.update({
            'total_explorations': 0,
            'successful_explorations': 0,
            'failed_explorations': 0,
            'total_analysis_cycles': 0,
            'consensus_reached': 0,
            'strategy_adaptations': 0,
            'average_exploration_time': 0.0,
            'average_analysis_time': 0.0,
            'inter_agent_messages': 0
        })
    
    def _setup_agents(self):
        """Setup specialized agent instances"""
        self.agents = {
            'explorers': [],
            'analyzers': [],
            'optimizers': [],
            'coordinator': None
        }
        
        # Create explorer agents
        for i in range(self.config.num_explorer_agents):
            agent = {
                'id': f"explorer_{i+1}",
                'role': AgentRole.EXPLORER,
                'status': 'idle',
                'current_task': None,
                'performance_score': 0.0,
                'specialization': self._get_explorer_specialization(i)
            }
            self.agents['explorers'].append(agent)
        
        # Create analyzer agents
        for i in range(self.config.num_analyzer_agents):
            agent = {
                'id': f"analyzer_{i+1}",
                'role': AgentRole.ANALYZER,
                'status': 'idle',
                'current_task': None,
                'performance_score': 0.0,
                'specialization': self._get_analyzer_specialization(i)
            }
            self.agents['analyzers'].append(agent)
        
        # Create optimizer agents
        for i in range(self.config.num_optimizer_agents):
            agent = {
                'id': f"optimizer_{i+1}",
                'role': AgentRole.OPTIMIZER,
                'status': 'idle',
                'current_task': None,
                'performance_score': 0.0,
                'specialization': self._get_optimizer_specialization(i)
            }
            self.agents['optimizers'].append(agent)
        
        # Create coordinator
        self.agents['coordinator'] = {
            'id': 'coordinator',
            'role': AgentRole.COORDINATOR,
            'status': 'active',
            'current_task': None,
            'performance_score': 0.0,
            'specialization': 'coordination'
        }
    
    def _setup_communication(self):
        """Setup inter-agent communication system"""
        self.message_queue = asyncio.Queue()
        self.communication_history = []
        self.consensus_tracker = {}
        self.strategy_adaptations = []
    
    def _get_explorer_specialization(self, index: int) -> str:
        """Get specialization for explorer agent"""
        specializations = [
            'structure_exploration',
            'community_detection',
            'path_analysis',
            'centrality_analysis',
            'clustering_analysis'
        ]
        return specializations[index % len(specializations)]
    
    def _get_analyzer_specialization(self, index: int) -> str:
        """Get specialization for analyzer agent"""
        specializations = [
            'pattern_recognition',
            'relationship_analysis',
            'semantic_analysis',
            'temporal_analysis',
            'quality_assessment'
        ]
        return specializations[index % len(specializations)]
    
    def _get_optimizer_specialization(self, index: int) -> str:
        """Get specialization for optimizer agent"""
        specializations = [
            'strategy_optimization',
            'performance_tuning',
            'resource_allocation',
            'quality_improvement'
        ]
        return specializations[index % len(specializations)]
    
    def _get_total_agent_count(self) -> int:
        """Get total number of agents"""
        return (self.config.num_explorer_agents + 
                self.config.num_analyzer_agents + 
                self.config.num_optimizer_agents + 1)  # +1 for coordinator
    
    def _setup_components(self):
        """Setup Graph Counselor specific components"""
        super()._setup_components()
        # Initialize vector store
        self.vector_store = None
    
    async def explore_graph(self, knowledge_graph: Any, 
                          exploration_goal: str = "comprehensive_analysis") -> Dict[str, Any]:
        """
        Main method for multi-agent graph exploration
        
        Args:
            knowledge_graph: The knowledge graph to explore
            exploration_goal: Goal for the exploration process
            
        Returns:
            Dictionary containing exploration results and insights
        """
        try:
            start_time = datetime.now()
            self.logger.info("Starting multi-agent graph exploration", 
                           goal=exploration_goal,
                           graph_size=self._get_graph_size(knowledge_graph))
            
            # Initialize exploration state
            exploration_state = {
                'goal': exploration_goal,
                'current_iteration': 0,
                'convergence_score': 0.0,
                'active_agents': [],
                'completed_tasks': [],
                'insights': [],
                'strategy_adaptations': []
            }
            
            # Create exploration tasks
            tasks = await self._create_exploration_tasks(knowledge_graph, exploration_goal)
            
            # Execute multi-agent exploration
            with Progress() as progress:
                main_task = progress.add_task("Multi-Agent Exploration", total=len(tasks))
                
                for iteration in range(self.config.max_iterations):
                    # Check convergence
                    if self._check_convergence(exploration_state):
                        self.logger.info("Exploration converged", iteration=iteration)
                        break
                    
                    # Execute agent tasks
                    iteration_results = await self._execute_agent_iteration(
                        knowledge_graph, tasks, exploration_state, progress, main_task
                    )
                    
                    # Update exploration state
                    exploration_state = self._update_exploration_state(
                        exploration_state, iteration_results
                    )
                    
                    # Adapt strategies based on results
                    if iteration % self.config.communication_frequency == 0:
                        await self._adapt_strategies(exploration_state)
                    
                    exploration_state['current_iteration'] = iteration + 1
                
                # Finalize exploration
                final_results = await self._finalize_exploration(
                    knowledge_graph, exploration_state
                )
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(processing_time, True)
            
            self.logger.info("Multi-agent exploration completed", 
                           processing_time=processing_time,
                           iterations=exploration_state['current_iteration'],
                           insights_count=len(final_results.get('insights', [])))
            
            return final_results
            
        except Exception as e:
            self.logger.error("Multi-agent exploration failed", error=str(e))
            self._update_metrics(0, False)
            raise
    
    async def _create_exploration_tasks(self, knowledge_graph: Any, 
                                      exploration_goal: str) -> List[Dict[str, Any]]:
        """Create exploration tasks for agents"""
        tasks = []
        
        # Structure exploration tasks
        if 'structure' in exploration_goal or 'comprehensive' in exploration_goal:
            tasks.extend([
                {
                    'id': 'structure_analysis',
                    'type': 'exploration',
                    'agent_type': 'explorer',
                    'priority': 3,
                    'description': 'Analyze graph structure and connectivity',
                    'parameters': {'focus': 'connectivity', 'depth': self.config.max_exploration_depth}
                },
                {
                    'id': 'community_detection',
                    'type': 'analysis',
                    'agent_type': 'analyzer',
                    'priority': 2,
                    'description': 'Detect communities and clusters',
                    'parameters': {'algorithm': 'louvain', 'min_community_size': 3}
                }
            ])
        
        # Relationship analysis tasks
        if 'relationships' in exploration_goal or 'comprehensive' in exploration_goal:
            tasks.extend([
                {
                    'id': 'relationship_analysis',
                    'type': 'analysis',
                    'agent_type': 'analyzer',
                    'priority': 2,
                    'description': 'Analyze relationship patterns',
                    'parameters': {'focus': 'patterns', 'include_weights': True}
                },
                {
                    'id': 'centrality_analysis',
                    'type': 'analysis',
                    'agent_type': 'analyzer',
                    'priority': 1,
                    'description': 'Calculate centrality measures',
                    'parameters': {'measures': ['betweenness', 'closeness', 'eigenvector']}
                }
            ])
        
        # Optimization tasks
        if 'optimization' in exploration_goal or 'comprehensive' in exploration_goal:
            tasks.extend([
                {
                    'id': 'strategy_optimization',
                    'type': 'optimization',
                    'agent_type': 'optimizer',
                    'priority': 1,
                    'description': 'Optimize exploration strategies',
                    'parameters': {'focus': 'efficiency', 'target_metric': 'coverage'}
                }
            ])
        
        return tasks
    
    async def _execute_agent_iteration(self, knowledge_graph: Any, 
                                     tasks: List[Dict[str, Any]], 
                                     exploration_state: Dict[str, Any],
                                     progress: Progress, 
                                     main_task: TaskID) -> Dict[str, Any]:
        """Execute one iteration of agent tasks"""
        iteration_results = {
            'completed_tasks': [],
            'insights': [],
            'agent_performance': {},
            'strategy_updates': []
        }
        
        # Select active agents for this iteration
        active_agents = self._select_active_agents(tasks, exploration_state)
        
        # Execute tasks in parallel
        if self.config.enable_parallel_processing:
            semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)
            task_coroutines = []
            
            for agent in active_agents:
                coro = self._execute_agent_task(
                    agent, knowledge_graph, tasks, semaphore
                )
                task_coroutines.append(coro)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error("Agent task failed", error=str(result))
                    continue
                
                if result:
                    iteration_results['completed_tasks'].append(result)
                    if 'insights' in result:
                        iteration_results['insights'].extend(result['insights'])
        
        # Update progress
        progress.update(main_task, advance=1)
        
        return iteration_results
    
    def _select_active_agents(self, tasks: List[Dict[str, Any]], 
                            exploration_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select which agents should be active for this iteration"""
        active_agents = []
        
        # Select explorers
        for explorer in self.agents['explorers']:
            if explorer['status'] == 'idle':
                # Find suitable tasks for this explorer
                suitable_tasks = [t for t in tasks if t['agent_type'] == 'explorer']
                if suitable_tasks:
                    explorer['current_task'] = suitable_tasks[0]
                    explorer['status'] = 'active'
                    active_agents.append(explorer)
        
        # Select analyzers
        for analyzer in self.agents['analyzers']:
            if analyzer['status'] == 'idle':
                suitable_tasks = [t for t in tasks if t['agent_type'] == 'analyzer']
                if suitable_tasks:
                    analyzer['current_task'] = suitable_tasks[0]
                    analyzer['status'] = 'active'
                    active_agents.append(analyzer)
        
        # Select optimizers
        for optimizer in self.agents['optimizers']:
            if optimizer['status'] == 'idle':
                suitable_tasks = [t for t in tasks if t['agent_type'] == 'optimizer']
                if suitable_tasks:
                    optimizer['current_task'] = suitable_tasks[0]
                    optimizer['status'] = 'active'
                    active_agents.append(optimizer)
        
        return active_agents
    
    async def _execute_agent_task(self, agent: Dict[str, Any], 
                                knowledge_graph: Any, 
                                tasks: List[Dict[str, Any]],
                                semaphore: asyncio.Semaphore) -> Optional[Dict[str, Any]]:
        """Execute a task for a specific agent"""
        async with semaphore:
            try:
                task = agent['current_task']
                if not task:
                    return None
                
                self.logger.info("Executing agent task", 
                               agent_id=agent['id'],
                               task_id=task['id'],
                               task_type=task['type'])
                
                # Execute based on agent role
                if agent['role'] == AgentRole.EXPLORER:
                    result = await self._execute_explorer_task(agent, knowledge_graph, task)
                elif agent['role'] == AgentRole.ANALYZER:
                    result = await self._execute_analyzer_task(agent, knowledge_graph, task)
                elif agent['role'] == AgentRole.OPTIMIZER:
                    result = await self._execute_optimizer_task(agent, knowledge_graph, task)
                else:
                    return None
                
                # Update agent status
                agent['status'] = 'idle'
                agent['current_task'] = None
                
                if result and 'performance_score' in result:
                    agent['performance_score'] = result['performance_score']
                
                return result
                
            except Exception as e:
                self.logger.error("Agent task execution failed", 
                                agent_id=agent['id'],
                                error=str(e))
                agent['status'] = 'idle'
                agent['current_task'] = None
                return None
    
    async def _execute_explorer_task(self, agent: Dict[str, Any], 
                                   knowledge_graph: Any, 
                                   task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute explorer agent task"""
        try:
            task_id = task['id']
            parameters = task.get('parameters', {})
            
            if task_id == 'structure_analysis':
                return await self._analyze_graph_structure(knowledge_graph, parameters)
            elif task_id == 'community_detection':
                return await self._detect_communities(knowledge_graph, parameters)
            else:
                return {'task_id': task_id, 'status': 'unsupported', 'insights': []}
                
        except Exception as e:
            self.logger.error("Explorer task failed", 
                            agent_id=agent['id'],
                            task_id=task['id'],
                            error=str(e))
            return {'task_id': task['id'], 'status': 'error', 'insights': []}
    
    async def _execute_analyzer_task(self, agent: Dict[str, Any], 
                                   knowledge_graph: Any, 
                                   task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analyzer agent task"""
        try:
            task_id = task['id']
            parameters = task.get('parameters', {})
            
            if task_id == 'relationship_analysis':
                return await self._analyze_relationships(knowledge_graph, parameters)
            elif task_id == 'centrality_analysis':
                return await self._analyze_centrality(knowledge_graph, parameters)
            else:
                return {'task_id': task_id, 'status': 'unsupported', 'insights': []}
                
        except Exception as e:
            self.logger.error("Analyzer task failed", 
                            agent_id=agent['id'],
                            task_id=task['id'],
                            error=str(e))
            return {'task_id': task['id'], 'status': 'error', 'insights': []}
    
    async def _execute_optimizer_task(self, agent: Dict[str, Any], 
                                    knowledge_graph: Any, 
                                    task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimizer agent task"""
        try:
            task_id = task['id']
            parameters = task.get('parameters', {})
            
            if task_id == 'strategy_optimization':
                return await self._optimize_strategies(knowledge_graph, parameters)
            else:
                return {'task_id': task_id, 'status': 'unsupported', 'insights': []}
                
        except Exception as e:
            self.logger.error("Optimizer task failed", 
                            agent_id=agent['id'],
                            task_id=task['id'],
                            error=str(e))
            return {'task_id': task['id'], 'status': 'error', 'insights': []}
    
    async def _analyze_graph_structure(self, knowledge_graph: Any, 
                                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze graph structure and connectivity"""
        try:
            # Convert to NetworkX for analysis
            nx_graph = self._convert_to_networkx(knowledge_graph)
            
            # Basic structure metrics
            num_nodes = nx_graph.number_of_nodes()
            num_edges = nx_graph.number_of_edges()
            density = nx.density(nx_graph)
            
            # Connectivity analysis
            is_connected = nx.is_connected(nx_graph)
            num_components = nx.number_connected_components(nx_graph)
            
            # Path analysis
            avg_path_length = 0
            diameter = 0
            if is_connected and num_nodes > 1:
                avg_path_length = nx.average_shortest_path_length(nx_graph)
                diameter = nx.diameter(nx_graph)
            
            insights = [
                f"Graph has {num_nodes} nodes and {num_edges} edges",
                f"Density: {density:.3f}",
                f"Connected: {is_connected}, Components: {num_components}",
                f"Average path length: {avg_path_length:.3f}",
                f"Diameter: {diameter}"
            ]
            
            return {
                'task_id': 'structure_analysis',
                'status': 'completed',
                'insights': insights,
                'metrics': {
                    'num_nodes': num_nodes,
                    'num_edges': num_edges,
                    'density': density,
                    'is_connected': is_connected,
                    'num_components': num_components,
                    'avg_path_length': avg_path_length,
                    'diameter': diameter
                },
                'performance_score': 0.8
            }
            
        except Exception as e:
            self.logger.error("Graph structure analysis failed", error=str(e))
            return {
                'task_id': 'structure_analysis',
                'status': 'error',
                'insights': [],
                'performance_score': 0.0
            }
    
    async def _detect_communities(self, knowledge_graph: Any, 
                                parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect communities in the graph"""
        try:
            nx_graph = self._convert_to_networkx(knowledge_graph)
            
            # Use Louvain algorithm for community detection
            import networkx.algorithms.community as nx_comm
            
            communities = list(nx_comm.louvain_communities(nx_graph))
            num_communities = len(communities)
            
            # Calculate modularity
            modularity = nx_comm.modularity(nx_graph, communities)
            
            # Community size distribution
            community_sizes = [len(c) for c in communities]
            avg_community_size = np.mean(community_sizes) if community_sizes else 0
            
            insights = [
                f"Found {num_communities} communities",
                f"Modularity: {modularity:.3f}",
                f"Average community size: {avg_community_size:.1f}",
                f"Community size range: {min(community_sizes)}-{max(community_sizes)}"
            ]
            
            return {
                'task_id': 'community_detection',
                'status': 'completed',
                'insights': insights,
                'metrics': {
                    'num_communities': num_communities,
                    'modularity': modularity,
                    'avg_community_size': avg_community_size,
                    'community_sizes': community_sizes
                },
                'performance_score': 0.9
            }
            
        except Exception as e:
            self.logger.error("Community detection failed", error=str(e))
            return {
                'task_id': 'community_detection',
                'status': 'error',
                'insights': [],
                'performance_score': 0.0
            }
    
    async def _analyze_relationships(self, knowledge_graph: Any, 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationship patterns in the graph"""
        try:
            nx_graph = self._convert_to_networkx(knowledge_graph)
            
            # Edge type analysis
            edge_types = {}
            for u, v, data in nx_graph.edges(data=True):
                edge_type = data.get('type', 'unknown')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            # Relationship patterns
            total_edges = nx_graph.number_of_edges()
            unique_edge_types = len(edge_types)
            
            insights = [
                f"Total relationships: {total_edges}",
                f"Unique relationship types: {unique_edge_types}",
                f"Most common type: {max(edge_types.items(), key=lambda x: x[1])[0] if edge_types else 'N/A'}"
            ]
            
            return {
                'task_id': 'relationship_analysis',
                'status': 'completed',
                'insights': insights,
                'metrics': {
                    'total_edges': total_edges,
                    'unique_edge_types': unique_edge_types,
                    'edge_type_distribution': edge_types
                },
                'performance_score': 0.7
            }
            
        except Exception as e:
            self.logger.error("Relationship analysis failed", error=str(e))
            return {
                'task_id': 'relationship_analysis',
                'status': 'error',
                'insights': [],
                'performance_score': 0.0
            }
    
    async def _analyze_centrality(self, knowledge_graph: Any, 
                                parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze centrality measures"""
        try:
            nx_graph = self._convert_to_networkx(knowledge_graph)
            
            # Calculate centrality measures
            betweenness = nx.betweenness_centrality(nx_graph)
            closeness = nx.closeness_centrality(nx_graph)
            eigenvector = nx.eigenvector_centrality(nx_graph, max_iter=1000)
            
            # Find most central nodes
            most_between = max(betweenness.items(), key=lambda x: x[1]) if betweenness else (None, 0)
            most_close = max(closeness.items(), key=lambda x: x[1]) if closeness else (None, 0)
            most_eigen = max(eigenvector.items(), key=lambda x: x[1]) if eigenvector else (None, 0)
            
            insights = [
                f"Most betweenness central: {most_between[0]} ({most_between[1]:.3f})",
                f"Most closeness central: {most_close[0]} ({most_close[1]:.3f})",
                f"Most eigenvector central: {most_eigen[0]} ({most_eigen[1]:.3f})"
            ]
            
            return {
                'task_id': 'centrality_analysis',
                'status': 'completed',
                'insights': insights,
                'metrics': {
                    'betweenness_centrality': betweenness,
                    'closeness_centrality': closeness,
                    'eigenvector_centrality': eigenvector
                },
                'performance_score': 0.8
            }
            
        except Exception as e:
            self.logger.error("Centrality analysis failed", error=str(e))
            return {
                'task_id': 'centrality_analysis',
                'status': 'error',
                'insights': [],
                'performance_score': 0.0
            }
    
    async def _optimize_strategies(self, knowledge_graph: Any, 
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize exploration strategies"""
        try:
            # Analyze current performance
            current_performance = self._calculate_current_performance()
            
            # Suggest optimizations
            optimizations = []
            if current_performance['efficiency'] < 0.7:
                optimizations.append("Consider increasing parallel processing")
            if current_performance['coverage'] < 0.8:
                optimizations.append("Expand exploration depth")
            if current_performance['quality'] < 0.6:
                optimizations.append("Improve agent coordination")
            
            insights = [
                f"Current efficiency: {current_performance['efficiency']:.2f}",
                f"Current coverage: {current_performance['coverage']:.2f}",
                f"Current quality: {current_performance['quality']:.2f}",
                f"Optimizations: {len(optimizations)} suggestions"
            ]
            
            return {
                'task_id': 'strategy_optimization',
                'status': 'completed',
                'insights': insights,
                'optimizations': optimizations,
                'performance_score': 0.6
            }
            
        except Exception as e:
            self.logger.error("Strategy optimization failed", error=str(e))
            return {
                'task_id': 'strategy_optimization',
                'status': 'error',
                'insights': [],
                'performance_score': 0.0
            }
    
    
    def _check_convergence(self, exploration_state: Dict[str, Any]) -> bool:
        """Check if exploration has converged"""
        if exploration_state['current_iteration'] >= self.config.max_iterations:
            return True
        
        if exploration_state['convergence_score'] >= self.config.convergence_threshold:
            return True
        
        return False
    
    def _update_exploration_state(self, state: Dict[str, Any], 
                                iteration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Update exploration state with iteration results"""
        state['completed_tasks'].extend(iteration_results.get('completed_tasks', []))
        state['insights'].extend(iteration_results.get('insights', []))
        
        # Update convergence score
        if iteration_results.get('completed_tasks'):
            state['convergence_score'] = min(1.0, state['convergence_score'] + 0.1)
        
        return state
    
    async def _adapt_strategies(self, exploration_state: Dict[str, Any]) -> None:
        """Adapt exploration strategies based on results"""
        try:
            # Analyze current performance
            performance = self._calculate_current_performance()
            
            # Adapt based on performance
            if performance['efficiency'] < 0.5:
                # Increase parallel processing
                self.config.max_concurrent_agents = min(
                    self.config.max_concurrent_agents + 1, 20
                )
                self.logger.info("Increased parallel processing", 
                               new_max_concurrent=self.config.max_concurrent_agents)
            
            if performance['coverage'] < 0.6:
                # Increase exploration depth
                self.config.max_exploration_depth = min(
                    self.config.max_exploration_depth + 1, 10
                )
                self.logger.info("Increased exploration depth", 
                               new_depth=self.config.max_exploration_depth)
            
            # Record adaptation
            adaptation = {
                'timestamp': datetime.now(),
                'performance': performance,
                'changes': {
                    'max_concurrent_agents': self.config.max_concurrent_agents,
                    'max_exploration_depth': self.config.max_exploration_depth
                }
            }
            self.strategy_adaptations.append(adaptation)
            self.metrics['strategy_adaptations'] += 1
            
        except Exception as e:
            self.logger.error("Strategy adaptation failed", error=str(e))
    
    def _calculate_current_performance(self) -> Dict[str, float]:
        """Calculate current performance metrics"""
        try:
            total_tasks = self.metrics['total_explorations']
            successful_tasks = self.metrics['successful_explorations']
            
            efficiency = successful_tasks / total_tasks if total_tasks > 0 else 0.0
            coverage = min(1.0, total_tasks / 100)  # Assume 100 tasks = full coverage
            quality = 0.8  # Placeholder for quality metric
            
            return {
                'efficiency': efficiency,
                'coverage': coverage,
                'quality': quality
            }
        except Exception as e:
            self.logger.error("Performance calculation failed", error=str(e))
            return {'efficiency': 0.0, 'coverage': 0.0, 'quality': 0.0}
    
    async def _finalize_exploration(self, knowledge_graph: Any, 
                                  exploration_state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize exploration and generate comprehensive results"""
        try:
            # Generate final insights
            final_insights = self._generate_final_insights(exploration_state)
            
            # Create comprehensive report
            report = {
                'exploration_summary': {
                    'goal': exploration_state['goal'],
                    'iterations': exploration_state['current_iteration'],
                    'convergence_score': exploration_state['convergence_score'],
                    'total_tasks': len(exploration_state['completed_tasks']),
                    'total_insights': len(exploration_state['insights'])
                },
                'insights': final_insights,
                'agent_performance': self._get_agent_performance_summary(),
                'strategy_adaptations': self.strategy_adaptations,
                'recommendations': self._generate_recommendations(exploration_state),
                'timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error("Exploration finalization failed", error=str(e))
            return {'error': str(e)}
    
    def _generate_final_insights(self, exploration_state: Dict[str, Any]) -> List[str]:
        """Generate final insights from exploration"""
        insights = exploration_state.get('insights', [])
        
        # Deduplicate and prioritize insights
        unique_insights = list(set(insights))
        
        # Add summary insights
        summary_insights = [
            f"Exploration completed in {exploration_state['current_iteration']} iterations",
            f"Generated {len(unique_insights)} unique insights",
            f"Convergence score: {exploration_state['convergence_score']:.3f}"
        ]
        
        return summary_insights + unique_insights[:10]  # Limit to top 10 insights
    
    def _get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get summary of agent performance"""
        performance = {}
        
        for agent_type, agents in self.agents.items():
            if agent_type == 'coordinator':
                continue
            
            scores = [agent['performance_score'] for agent in agents]
            performance[agent_type] = {
                'count': len(agents),
                'avg_score': np.mean(scores) if scores else 0.0,
                'max_score': max(scores) if scores else 0.0,
                'min_score': min(scores) if scores else 0.0
            }
        
        return performance
    
    def _generate_recommendations(self, exploration_state: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on exploration results"""
        recommendations = []
        
        # Performance-based recommendations
        performance = self._calculate_current_performance()
        
        if performance['efficiency'] < 0.7:
            recommendations.append("Consider increasing the number of parallel agents")
        
        if performance['coverage'] < 0.8:
            recommendations.append("Increase exploration depth or budget")
        
        if performance['quality'] < 0.6:
            recommendations.append("Improve agent coordination and communication")
        
        # Task-based recommendations
        if len(exploration_state['completed_tasks']) < 10:
            recommendations.append("Consider adding more exploration tasks")
        
        return recommendations
    
    def _update_metrics(self, processing_time: float, success: bool) -> None:
        """Update performance metrics"""
        super()._update_metrics(processing_time, success)
        
        # Update Graph Counselor specific metrics
        if success:
            self.metrics['successful_explorations'] += 1
        else:
            self.metrics['failed_explorations'] += 1
        
        # Update averages
        if self.metrics['total_explorations'] > 0:
            self.metrics['average_exploration_time'] = (
                self.metrics['total_processing_time'] / self.metrics['total_explorations']
            )
    
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get comprehensive configuration summary"""
        summary = super().get_config_summary()
        
        # Add Graph Counselor specific fields
        summary.update({
            "num_explorer_agents": self.config.num_explorer_agents,
            "num_analyzer_agents": self.config.num_analyzer_agents,
            "num_optimizer_agents": self.config.num_optimizer_agents,
            "exploration_strategy": self.config.exploration_strategy.value,
            "max_exploration_depth": self.config.max_exploration_depth,
            "exploration_budget": self.config.exploration_budget,
            "communication_frequency": self.config.communication_frequency,
            "consensus_threshold": self.config.consensus_threshold,
            "adaptation_rate": self.config.adaptation_rate,
            "max_concurrent_agents": self.config.max_concurrent_agents,
            "enable_community_detection": self.config.enable_community_detection,
            "enable_hierarchical_analysis": self.config.enable_hierarchical_analysis,
            "enable_temporal_analysis": self.config.enable_temporal_analysis,
            "enable_semantic_clustering": self.config.enable_semantic_clustering,
            "min_confidence_threshold": self.config.min_confidence_threshold,
            "max_iterations": self.config.max_iterations,
            "convergence_threshold": self.config.convergence_threshold,
            "total_agents": self._get_total_agent_count()
        })
        
        return summary
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current status of all agents"""
        status = {}
        
        for agent_type, agents in self.agents.items():
            status[agent_type] = []
            for agent in agents:
                status[agent_type].append({
                    'id': agent['id'],
                    'status': agent['status'],
                    'current_task': agent['current_task'],
                    'performance_score': agent['performance_score'],
                    'specialization': agent.get('specialization', 'general')
                })
        
        return status
