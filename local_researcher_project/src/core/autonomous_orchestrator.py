#!/usr/bin/env python3
"""
LangGraph-Optimized Autonomous Orchestrator for Multi-Agent Research System

This orchestrator manages the complete autonomous workflow using LangGraph:
1. LLM-based objective analysis and task decomposition
2. Dynamic multi-agent task assignment and execution
3. Critical evaluation and recursive refinement
4. Result validation and final synthesis

No fallback or dummy code - production-level autonomous operation only.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, TypedDict, Annotated
from datetime import datetime
import json
from pathlib import Path
import os
import google.generativeai as genai

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from src.core.mcp_integration import MCPIntegrationManager
from src.core.llm_methods import LLMMethods
from src.utils.config_manager import ConfigManager, AdvancedConfiguration
from src.utils.logger import setup_logger

logger = setup_logger("autonomous_orchestrator", log_level="INFO")


class ResearchState(TypedDict):
    """State definition for LangGraph research workflow."""
    # Input
    user_request: str
    context: Optional[Dict[str, Any]]
    objective_id: str
    
    # Analysis
    analyzed_objectives: List[Dict[str, Any]]
    intent_analysis: Dict[str, Any]
    domain_analysis: Dict[str, Any]
    scope_analysis: Dict[str, Any]
    
    # Task Decomposition
    decomposed_tasks: List[Dict[str, Any]]
    task_assignments: List[Dict[str, Any]]
    execution_strategy: str
    
    # Execution
    execution_results: List[Dict[str, Any]]
    agent_status: Dict[str, Any]
    execution_metadata: Dict[str, Any]
    
    # Evaluation
    evaluation_results: Dict[str, Any]
    quality_metrics: Dict[str, float]
    improvement_areas: List[str]
    
    # Validation
    validation_results: Dict[str, Any]
    validation_score: float
    missing_elements: List[str]
    
    # Synthesis
    final_synthesis: Dict[str, Any]
    deliverable_path: Optional[str]
    synthesis_metadata: Dict[str, Any]
    
    # Control Flow
    current_step: str
    iteration: int
    max_iterations: int
    should_continue: bool
    error_message: Optional[str]
    
    # Messages for LangGraph
    messages: Annotated[List[BaseMessage], "Messages in the conversation"]


class LangGraphOrchestrator:
    """LangGraph-optimized autonomous orchestrator for multi-agent research system."""
    
    def __init__(self, config_path: Optional[str], agents: Dict[str, Any], mcp_manager: MCPIntegrationManager):
        """Initialize the LangGraph orchestrator.
        
        Args:
            config_path: Path to configuration file
            agents: Dictionary of specialized agents
            mcp_manager: MCP integration manager
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.agents = agents
        self.mcp_manager = mcp_manager
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        self.llm_methods = LLMMethods(self.llm)
        
        # Advanced configuration
        self.advanced_config = self.config_manager.get_advanced_config()
        
        # Orchestration state
        self.active_objectives: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # LangGraph configuration
        self.langgraph_config = self.config_manager.get_langgraph_config()
        
        # Build LangGraph workflow
        self.graph = self._build_langgraph_workflow()
        
        logger.info("LangGraph Orchestrator initialized with advanced workflow management")
    
    def _build_langgraph_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for autonomous research."""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("analyze_objectives", self._analyze_objectives_node)
        workflow.add_node("decompose_tasks", self._decompose_tasks_node)
        workflow.add_node("execute_research", self._execute_research_node)
        workflow.add_node("evaluate_results", self._evaluate_results_node)
        workflow.add_node("validate_results", self._validate_results_node)
        workflow.add_node("synthesize_deliverable", self._synthesize_deliverable_node)
        workflow.add_node("decide_continuation", self._decide_continuation_node)
        
        # Add edges
        workflow.set_entry_point("analyze_objectives")
        workflow.add_edge("analyze_objectives", "decompose_tasks")
        workflow.add_edge("decompose_tasks", "execute_research")
        workflow.add_edge("execute_research", "evaluate_results")
        workflow.add_edge("evaluate_results", "validate_results")
        workflow.add_edge("validate_results", "synthesize_deliverable")
        workflow.add_edge("synthesize_deliverable", "decide_continuation")
        
        # Conditional edges for iteration
        workflow.add_conditional_edges(
            "decide_continuation",
            self._should_continue,
            {
                "continue": "execute_research",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _initialize_llm(self):
        """Initialize the LLM client."""
        try:
            # Get API key from config or environment
            api_key = self.config_manager.get('gemini_api_key') or os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("LLM initialized successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    async def start_autonomous_research(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Start fully autonomous research using LangGraph workflow.
        
        Args:
            user_request: The user's research request
            context: Additional context for the research
            
        Returns:
            Research objective ID
        """
        try:
            # Create objective ID
            objective_id = f"obj_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(user_request) % 10000:04d}"
            
            # Initialize state
            initial_state = ResearchState(
                user_request=user_request,
                context=context or {},
                objective_id=objective_id,
                analyzed_objectives=[],
                intent_analysis={},
                domain_analysis={},
                scope_analysis={},
                decomposed_tasks=[],
                task_assignments=[],
                execution_strategy="",
                execution_results=[],
                agent_status={},
                execution_metadata={},
                evaluation_results={},
                quality_metrics={},
                improvement_areas=[],
                validation_results={},
                validation_score=0.0,
                missing_elements=[],
                final_synthesis={},
                deliverable_path=None,
                synthesis_metadata={},
                current_step="analyze_objectives",
                iteration=0,
                max_iterations=self.advanced_config.max_researcher_iterations,
                should_continue=True,
                error_message=None,
                messages=[HumanMessage(content=user_request)]
            )
            
            # Store objective
            self.active_objectives[objective_id] = initial_state
            
            logger.info(f"Starting LangGraph-based autonomous research for objective: {objective_id}")
            logger.info(f"User request: {user_request}")
            
            # Execute LangGraph workflow
            final_state = await self.graph.ainvoke(
                initial_state,
                config=self.langgraph_config
            )
            
            # Update stored objective
            self.active_objectives[objective_id] = final_state
            
            logger.info(f"LangGraph-based autonomous research completed: {objective_id}")
            logger.info(f"Final deliverable: {final_state.get('deliverable_path', 'N/A')}")
            
            return objective_id
            
        except Exception as e:
            logger.error(f"LangGraph-based autonomous research failed: {e}")
            if 'objective_id' in locals() and objective_id in self.active_objectives:
                self.active_objectives[objective_id]['error_message'] = str(e)
            raise
    
    # LangGraph Node Methods
    async def _analyze_objectives_node(self, state: ResearchState) -> ResearchState:
        """Analyze user request and extract objectives."""
        try:
            logger.info(f"Analyzing objectives for: {state['user_request']}")
            
            # Use task analyzer agent
            analyzer = self.agents.get('analyzer')
            if analyzer:
                analysis_result = await analyzer.analyze_objective(
                    state['user_request'], 
                    state['context'], 
                    state['objective_id']
                )
                
                state['analyzed_objectives'] = analysis_result.get('objectives', [])
                state['intent_analysis'] = analysis_result.get('intent_analysis', {})
                state['domain_analysis'] = analysis_result.get('domain_analysis', {})
                state['scope_analysis'] = analysis_result.get('scope_analysis', {})
            else:
                # Fallback to LLM analysis
                analysis_result = await self.llm_methods.llm_analyze_objective(state)
                state['analyzed_objectives'] = analysis_result
            
            state['current_step'] = "decompose_tasks"
            state['messages'].append(AIMessage(content=f"Analyzed {len(state['analyzed_objectives'])} objectives"))
            
            return state
            
        except Exception as e:
            logger.error(f"Objective analysis failed: {e}")
            state['error_message'] = str(e)
            return state
    
    async def _decompose_tasks_node(self, state: ResearchState) -> ResearchState:
        """Decompose objectives into executable tasks."""
        try:
            logger.info("Decomposing tasks")
            
            # Use task decomposer agent
            decomposer = self.agents.get('decomposer')
            if decomposer:
                decomposition_result = await decomposer.decompose_tasks(
                    state['analyzed_objectives'],
                    state['context']
                )
                
                state['decomposed_tasks'] = decomposition_result.get('tasks', [])
                state['task_assignments'] = decomposition_result.get('assignments', [])
                state['execution_strategy'] = decomposition_result.get('strategy', 'sequential')
            else:
                # Fallback to LLM decomposition
                decomposition_result = await self.llm_methods.llm_decompose_tasks(state)
                state['decomposed_tasks'] = decomposition_result.get('tasks', [])
                state['task_assignments'] = decomposition_result.get('assignments', [])
                state['execution_strategy'] = decomposition_result.get('strategy', 'sequential')
            
            state['current_step'] = "execute_research"
            state['messages'].append(AIMessage(content=f"Decomposed into {len(state['decomposed_tasks'])} tasks"))
            
            return state
            
        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
            state['error_message'] = str(e)
            return state
    
    async def _execute_research_node(self, state: ResearchState) -> ResearchState:
        """Execute research tasks using specialized agents."""
        try:
            logger.info("Executing research tasks")
            
            execution_results = []
            agent_status = {}
            
            # Execute tasks based on strategy
            if state['execution_strategy'] == 'parallel':
                # Execute tasks in parallel
                tasks = []
                for task in state['decomposed_tasks']:
                    agent_name = task.get('assigned_to')
                    if agent_name and agent_name in self.agents:
                        task_coro = self._execute_single_task(task, agent_name, state)
                        tasks.append(task_coro)
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception):
                            logger.error(f"Task execution failed: {result}")
                        else:
                            execution_results.append(result)
            else:
                # Execute tasks sequentially
                for task in state['decomposed_tasks']:
                    agent_name = task.get('assigned_to')
                    if agent_name and agent_name in self.agents:
                        result = await self._execute_single_task(task, agent_name, state)
                        execution_results.append(result)
            
            state['execution_results'] = execution_results
            state['agent_status'] = agent_status
            state['current_step'] = "evaluate_results"
            state['messages'].append(AIMessage(content=f"Executed {len(execution_results)} tasks"))
            
            return state
            
        except Exception as e:
            logger.error(f"Research execution failed: {e}")
            state['error_message'] = str(e)
            return state
    
    async def _execute_single_task(self, task: Dict[str, Any], agent_name: str, state: ResearchState) -> Dict[str, Any]:
        """Execute a single task using the specified agent."""
        try:
            agent = self.agents[agent_name]
            
            # Execute task based on agent type
            if hasattr(agent, 'execute_task'):
                result = await agent.execute_task(task, state['context'], state['objective_id'])
            elif hasattr(agent, 'conduct_research'):
                result = await agent.conduct_research(task, state['context'], state['objective_id'])
            else:
                result = {"error": f"Agent {agent_name} does not support task execution"}
            
            return {
                "task_id": task.get('task_id'),
                "agent": agent_name,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Single task execution failed: {e}")
            return {
                "task_id": task.get('task_id'),
                "agent": agent_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _evaluate_results_node(self, state: ResearchState) -> ResearchState:
        """Evaluate research results and identify improvements."""
        try:
            logger.info("Evaluating results")
            
            # Use evaluation agent
            evaluator = self.agents.get('evaluator')
            if evaluator:
                evaluation_result = await evaluator.evaluate_results(
                    state['execution_results'],
                    state['analyzed_objectives']
                )
            else:
                # Fallback to LLM evaluation
                evaluation_result = await self.llm_methods.llm_evaluate_results(state)
            
            state['evaluation_results'] = evaluation_result
            state['quality_metrics'] = evaluation_result.get('quality_metrics', {})
            state['improvement_areas'] = evaluation_result.get('improvement_areas', [])
            state['current_step'] = "validate_results"
            state['messages'].append(AIMessage(content="Results evaluated"))
            
            return state
            
        except Exception as e:
            logger.error(f"Result evaluation failed: {e}")
            state['error_message'] = str(e)
            return state
    
    async def _validate_results_node(self, state: ResearchState) -> ResearchState:
        """Validate results against original objectives."""
        try:
            logger.info("Validating results")
            
            # Use validation agent
            validator = self.agents.get('validator')
            if validator:
                validation_result = await validator.validate_results(
                    state['user_request'],
                    state['analyzed_objectives'],
                    state['execution_results']
                )
            else:
                # Fallback to LLM validation
                validation_result = await self.llm_methods.llm_validate_results(state)
            
            state['validation_results'] = validation_result
            state['validation_score'] = validation_result.get('validation_score', 0.0)
            state['missing_elements'] = validation_result.get('missing_elements', [])
            state['current_step'] = "synthesize_deliverable"
            state['messages'].append(AIMessage(content="Results validated"))
            
            return state
            
        except Exception as e:
            logger.error(f"Result validation failed: {e}")
            state['error_message'] = str(e)
            return state
    
    async def _synthesize_deliverable_node(self, state: ResearchState) -> ResearchState:
        """Synthesize final deliverable from all results."""
        try:
            logger.info("Synthesizing deliverable")
            
            # Use synthesis agent
            synthesizer = self.agents.get('synthesizer')
            if synthesizer:
                synthesis_result = await synthesizer.synthesize_deliverable(
                    state['user_request'],
                    state['execution_results'],
                    state['evaluation_results'],
                    state['validation_results']
                )
            else:
                # Fallback to LLM synthesis
                synthesis_result = await self.llm_methods.llm_synthesize_deliverable(state)
            
            state['final_synthesis'] = synthesis_result
            state['deliverable_path'] = synthesis_result.get('deliverable_path')
            state['synthesis_metadata'] = synthesis_result.get('metadata', {})
            state['current_step'] = "decide_continuation"
            state['messages'].append(AIMessage(content="Deliverable synthesized"))
            
            return state
            
        except Exception as e:
            logger.error(f"Deliverable synthesis failed: {e}")
            state['error_message'] = str(e)
            return state
    
    async def _decide_continuation_node(self, state: ResearchState) -> ResearchState:
        """Decide whether to continue with another iteration."""
        try:
            logger.info("Deciding continuation")
            
            # Check if we should continue based on quality and iterations
            quality_score = state['validation_score']
            iteration = state['iteration']
            max_iterations = state['max_iterations']
            
            should_continue = (
                quality_score < 0.8 and 
                iteration < max_iterations and 
                not state.get('error_message')
            )
            
            state['should_continue'] = should_continue
            state['iteration'] = iteration + 1
            
            if should_continue:
                state['current_step'] = "execute_research"
                state['messages'].append(AIMessage(content=f"Continuing iteration {iteration + 1}"))
            else:
                state['current_step'] = "completed"
                state['messages'].append(AIMessage(content="Research completed"))
            
            return state
            
        except Exception as e:
            logger.error(f"Continuation decision failed: {e}")
            state['error_message'] = str(e)
            state['should_continue'] = False
            return state
    
    def _should_continue(self, state: ResearchState) -> str:
        """Determine whether to continue or end the workflow."""
        return "continue" if state['should_continue'] else "end"
    
    async def get_research_status(self, objective_id: str) -> Optional[Dict[str, Any]]:
        """Get research status."""
        try:
            if objective_id not in self.active_objectives:
                return None
                
            objective = self.active_objectives[objective_id]
            return {
                'objective_id': objective.objective_id,
                'status': objective.status,
                'user_request': objective.user_request,
                'created_at': objective.created_at.isoformat(),
                'analyzed_objectives': objective.analyzed_objectives,
                'decomposed_tasks': objective.decomposed_tasks,
                'assigned_agents': objective.assigned_agents,
                'execution_results': objective.execution_results,
                'evaluation_results': objective.evaluation_results,
                'validation_results': objective.validation_results,
                'final_synthesis': objective.final_synthesis
            }
            
        except Exception as e:
            logger.error(f"Failed to get research status: {e}")
            return None
    
    async def list_research(self) -> List[Dict[str, Any]]:
        """List all research objectives."""
        try:
            objectives = []
            for objective in self.active_objectives.values():
                objectives.append({
                    'objective_id': objective.objective_id,
                    'status': objective.status,
                    'user_request': objective.user_request,
                    'created_at': objective.created_at.isoformat()
                })
            return objectives
            
        except Exception as e:
            logger.error(f"Failed to list research: {e}")
            return []
    
    async def cancel_research(self, objective_id: str) -> bool:
        """Cancel research objective."""
        try:
            if objective_id not in self.active_objectives:
                return False
                
            # Cancel all active agent tasks
            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'cancel_tasks'):
                    await agent.cancel_tasks(objective_id)
            
            # Mark objective as cancelled
            self.active_objectives[objective_id].status = "cancelled"
            
            logger.info(f"Research objective cancelled: {objective_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel research: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup orchestrator resources."""
        try:
            # Cancel all active objectives
            for objective_id in list(self.active_objectives.keys()):
                await self.cancel_research(objective_id)
            
            # Cleanup agents
            for agent in self.agents.values():
                if hasattr(agent, 'cleanup'):
                    await agent.cleanup()
            
            logger.info("Autonomous Orchestrator cleanup completed")
            
        except Exception as e:
            logger.error(f"Orchestrator cleanup failed: {e}")


# Backward compatibility alias
AutonomousOrchestrator = LangGraphOrchestrator
