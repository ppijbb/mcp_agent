#!/usr/bin/env python3
"""
Autonomous Orchestrator for Multi-Agent Research System

This orchestrator manages the complete autonomous workflow:
1. Objective analysis and task decomposition
2. Multi-agent task assignment and execution
3. Critical evaluation and recursive refinement
4. Result validation and final synthesis

No fallback or dummy code - production-level autonomous operation only.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

from src.core.mcp_integration import MCPIntegrationManager
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

logger = setup_logger("autonomous_orchestrator", log_level="INFO")


class AutonomousOrchestrator:
    """Fully autonomous orchestrator for multi-agent research system."""
    
    def __init__(self, config_path: Optional[str], agents: Dict[str, Any], mcp_manager: MCPIntegrationManager):
        """Initialize the autonomous orchestrator.
        
        Args:
            config_path: Path to configuration file
            agents: Dictionary of specialized agents
            mcp_manager: MCP integration manager
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.agents = agents
        self.mcp_manager = mcp_manager
        
        # Orchestration state
        self.active_objectives: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        logger.info("Autonomous Orchestrator initialized with full agent coordination")
    
    async def analyze_objective(self, objective: Any) -> List[Dict[str, Any]]:
        """Phase 1: Autonomous analysis of research objectives.
        
        Args:
            objective: Research objective to analyze
            
        Returns:
            List of analyzed objectives with priorities and requirements
        """
        try:
            logger.info(f"Starting autonomous analysis for objective: {objective.objective_id}")
            
            # Use task analyzer agent for autonomous analysis
            analysis_result = await self.agents['analyzer'].analyze_objective(
                user_request=objective.user_request,
                context=objective.context,
                objective_id=objective.objective_id
            )
            
            # Process analysis results
            analyzed_objectives = analysis_result.get('objectives', [])
            
            # Enhance with MCP capabilities if available
            if self.mcp_manager.is_available():
                enhanced_analysis = await self.mcp_manager.enhance_analysis(analyzed_objectives)
                analyzed_objectives = enhanced_analysis
            
            logger.info(f"Analysis completed: {len(analyzed_objectives)} objectives identified")
            return analyzed_objectives
            
        except Exception as e:
            logger.error(f"Objective analysis failed: {e}")
            raise
    
    async def decompose_tasks(self, objective: Any) -> Dict[str, Any]:
        """Phase 2: Dynamic task decomposition and agent assignment.
        
        Args:
            objective: Research objective with analyzed objectives
            
        Returns:
            Dictionary containing decomposed tasks and agent assignments
        """
        try:
            logger.info(f"Starting task decomposition for objective: {objective.objective_id}")
            
            # Use task decomposer agent for dynamic decomposition
            decomposition_result = await self.agents['decomposer'].decompose_tasks(
                analyzed_objectives=objective.analyzed_objectives,
                context=objective.context,
                objective_id=objective.objective_id
            )
            
            # Process decomposition results
            decomposed_tasks = decomposition_result.get('tasks', [])
            agent_assignments = decomposition_result.get('assignments', [])
            
            # Optimize assignments based on agent capabilities
            optimized_assignments = await self._optimize_agent_assignments(
                decomposed_tasks, agent_assignments
            )
            
            logger.info(f"Task decomposition completed: {len(decomposed_tasks)} tasks assigned to {len(optimized_assignments)} agents")
            
            return {
                'tasks': decomposed_tasks,
                'assignments': optimized_assignments
            }
            
        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
            raise
    
    async def execute_tasks(self, objective: Any) -> List[Dict[str, Any]]:
        """Phase 3: Multi-agent task execution with MCP integration.
        
        Args:
            objective: Research objective with decomposed tasks and assignments
            
        Returns:
            List of execution results from all agents
        """
        try:
            logger.info(f"Starting multi-agent execution for objective: {objective.objective_id}")
            
            execution_results = []
            
            # Execute tasks in parallel where possible
            execution_tasks = []
            
            for assignment in objective.assigned_agents:
                agent_name = assignment['agent']
                task = assignment['task']
                
                # Create execution task
                execution_task = self._execute_agent_task(
                    agent_name, task, objective.objective_id
                )
                execution_tasks.append(execution_task)
            
            # Execute all tasks concurrently
            if execution_tasks:
                results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Agent execution failed: {result}")
                        execution_results.append({
                            'agent': objective.assigned_agents[i]['agent'],
                            'task_id': objective.assigned_agents[i]['task']['task_id'],
                            'status': 'failed',
                            'error': str(result),
                            'timestamp': datetime.now().isoformat()
                        })
                    else:
                        execution_results.append(result)
            
            # Integrate MCP capabilities if available
            if self.mcp_manager.is_available():
                enhanced_results = await self.mcp_manager.enhance_execution_results(execution_results)
                execution_results = enhanced_results
            
            logger.info(f"Multi-agent execution completed: {len(execution_results)} results generated")
            return execution_results
            
        except Exception as e:
            logger.error(f"Multi-agent execution failed: {e}")
            raise
    
    async def evaluate_results(self, objective: Any) -> Dict[str, Any]:
        """Phase 4: Critical evaluation and recursive execution assessment.
        
        Args:
            objective: Research objective with execution results
            
        Returns:
            Evaluation results including recursion recommendations
        """
        try:
            logger.info(f"Starting critical evaluation for objective: {objective.objective_id}")
            
            # Use evaluation agent for critical assessment
            evaluation_result = await self.agents['evaluator'].evaluate_results(
                execution_results=objective.execution_results,
                original_objectives=objective.analyzed_objectives,
                context=objective.context,
                objective_id=objective.objective_id
            )
            
            # Determine if recursive execution is needed
            needs_recursion = evaluation_result.get('needs_recursion', False)
            recursion_reason = evaluation_result.get('recursion_reason', '')
            
            if needs_recursion:
                logger.info(f"Recursive execution recommended: {recursion_reason}")
            else:
                logger.info("Evaluation completed - no recursion needed")
            
            # Store evaluation in history
            self.execution_history.append({
                'objective_id': objective.objective_id,
                'phase': 'evaluation',
                'result': evaluation_result,
                'timestamp': datetime.now().isoformat()
            })
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Critical evaluation failed: {e}")
            raise
    
    async def execute_recursive_refinement(self, objective: Any) -> List[Dict[str, Any]]:
        """Execute recursive refinement based on evaluation results.
        
        Args:
            objective: Research objective with evaluation results
            
        Returns:
            Additional execution results from recursive refinement
        """
        try:
            logger.info(f"Starting recursive refinement for objective: {objective.objective_id}")
            
            # Get refinement recommendations from evaluation
            evaluation = objective.evaluation_results
            refinement_tasks = evaluation.get('refinement_tasks', [])
            
            if not refinement_tasks:
                logger.info("No refinement tasks identified")
                return []
            
            # Execute refinement tasks
            refinement_results = []
            
            for refinement_task in refinement_tasks:
                agent_name = refinement_task['agent']
                task = refinement_task['task']
                
                # Execute refinement task
                result = await self._execute_agent_task(
                    agent_name, task, objective.objective_id, is_refinement=True
                )
                refinement_results.append(result)
            
            logger.info(f"Recursive refinement completed: {len(refinement_results)} additional results")
            return refinement_results
            
        except Exception as e:
            logger.error(f"Recursive refinement failed: {e}")
            raise
    
    async def validate_results(self, objective: Any) -> Dict[str, Any]:
        """Phase 5: Result validation against original objectives.
        
        Args:
            objective: Research objective with all execution results
            
        Returns:
            Validation results with alignment scores
        """
        try:
            logger.info(f"Starting result validation for objective: {objective.objective_id}")
            
            # Use validation agent for objective alignment check
            validation_result = await self.agents['validator'].validate_results(
                execution_results=objective.execution_results,
                original_objectives=objective.analyzed_objectives,
                user_request=objective.user_request,
                context=objective.context,
                objective_id=objective.objective_id
            )
            
            # Calculate overall validation score
            validation_score = validation_result.get('validation_score', 0)
            alignment_issues = validation_result.get('alignment_issues', [])
            
            logger.info(f"Validation completed: {validation_score:.2f}% alignment with original objectives")
            
            if alignment_issues:
                logger.warning(f"Alignment issues identified: {len(alignment_issues)}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Result validation failed: {e}")
            raise
    
    async def synthesize_final_deliverable(self, objective: Any) -> Dict[str, Any]:
        """Phase 6: Final synthesis and deliverable generation.
        
        Args:
            objective: Research objective with all results and validation
            
        Returns:
            Final synthesis result with deliverable information
        """
        try:
            logger.info(f"Starting final synthesis for objective: {objective.objective_id}")
            
            # Use synthesis agent for final deliverable generation
            synthesis_result = await self.agents['synthesizer'].synthesize_deliverable(
                execution_results=objective.execution_results,
                evaluation_results=objective.evaluation_results,
                validation_results=objective.validation_results,
                original_objectives=objective.analyzed_objectives,
                user_request=objective.user_request,
                context=objective.context,
                objective_id=objective.objective_id
            )
            
            # Generate deliverable file
            deliverable_path = synthesis_result.get('deliverable_path')
            if deliverable_path:
                logger.info(f"Final deliverable generated: {deliverable_path}")
            else:
                logger.warning("No deliverable path generated")
            
            return synthesis_result
            
        except Exception as e:
            logger.error(f"Final synthesis failed: {e}")
            raise
    
    async def cancel_objective(self, objective_id: str) -> bool:
        """Cancel an active research objective.
        
        Args:
            objective_id: Objective ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        try:
            logger.info(f"Cancelling objective: {objective_id}")
            
            # Cancel all active agent tasks
            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'cancel_tasks'):
                    await agent.cancel_tasks(objective_id)
            
            # Remove from active objectives
            if objective_id in self.active_objectives:
                del self.active_objectives[objective_id]
            
            logger.info(f"Objective cancelled: {objective_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel objective: {e}")
            return False
    
    async def _execute_agent_task(self, agent_name: str, task: Dict[str, Any], 
                                objective_id: str, is_refinement: bool = False) -> Dict[str, Any]:
        """Execute a single agent task.
        
        Args:
            agent_name: Name of the agent to execute
            task: Task to execute
            objective_id: Objective ID
            is_refinement: Whether this is a refinement task
            
        Returns:
            Execution result
        """
        try:
            agent = self.agents.get(agent_name)
            if not agent:
                raise ValueError(f"Agent not found: {agent_name}")
            
            # Execute task
            result = await agent.execute_task(
                task=task,
                objective_id=objective_id,
                is_refinement=is_refinement
            )
            
            # Add metadata
            result.update({
                'agent': agent_name,
                'task_id': task.get('task_id'),
                'objective_id': objective_id,
                'is_refinement': is_refinement,
                'timestamp': datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Agent task execution failed: {agent_name} - {e}")
            return {
                'agent': agent_name,
                'task_id': task.get('task_id'),
                'objective_id': objective_id,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _optimize_agent_assignments(self, tasks: List[Dict[str, Any]], 
                                        assignments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize agent assignments based on capabilities and workload.
        
        Args:
            tasks: List of decomposed tasks
            assignments: Initial agent assignments
            
        Returns:
            Optimized agent assignments
        """
        try:
            # Simple optimization - can be enhanced with more sophisticated algorithms
            optimized_assignments = []
            
            for assignment in assignments:
                agent_name = assignment['agent']
                task = assignment['task']
                
                # Check agent capabilities
                agent = self.agents.get(agent_name)
                if agent and hasattr(agent, 'can_handle_task'):
                    if await agent.can_handle_task(task):
                        optimized_assignments.append(assignment)
                    else:
                        # Find alternative agent
                        alternative_agent = await self._find_alternative_agent(task)
                        if alternative_agent:
                            optimized_assignments.append({
                                'agent': alternative_agent,
                                'task': task,
                                'reason': 'capability_optimization'
                            })
                        else:
                            # Keep original assignment as fallback
                            optimized_assignments.append(assignment)
                else:
                    optimized_assignments.append(assignment)
            
            return optimized_assignments
            
        except Exception as e:
            logger.error(f"Agent assignment optimization failed: {e}")
            return assignments
    
    async def _find_alternative_agent(self, task: Dict[str, Any]) -> Optional[str]:
        """Find alternative agent for a task.
        
        Args:
            task: Task to find agent for
            
        Returns:
            Alternative agent name or None
        """
        try:
            task_type = task.get('type', 'general')
            
            # Simple mapping - can be enhanced with more sophisticated matching
            agent_mapping = {
                'research': 'researcher',
                'analysis': 'analyzer',
                'evaluation': 'evaluator',
                'validation': 'validator',
                'synthesis': 'synthesizer'
            }
            
            return agent_mapping.get(task_type)
            
        except Exception as e:
            logger.error(f"Alternative agent search failed: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup orchestrator resources."""
        try:
            # Cancel all active objectives
            for objective_id in list(self.active_objectives.keys()):
                await self.cancel_objective(objective_id)
            
            # Cleanup agents
            for agent in self.agents.values():
                if hasattr(agent, 'cleanup'):
                    await agent.cleanup()
            
            logger.info("Autonomous Orchestrator cleanup completed")
            
        except Exception as e:
            logger.error(f"Orchestrator cleanup failed: {e}")
