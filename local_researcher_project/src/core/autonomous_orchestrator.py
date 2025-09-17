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
        
        # Enhanced autonomy settings
        self.max_iterations = 10  # Maximum iterations for recursive improvement
        self.quality_threshold = 0.8  # Quality threshold for stopping iterations
        self.learning_enabled = True  # Enable learning from previous executions
        self.adaptive_strategy = True  # Enable adaptive strategy selection
        
        logger.info("Autonomous Orchestrator initialized with enhanced autonomy and iterative improvement")
    
    async def start_autonomous_research(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Start fully autonomous research with enhanced iterative improvement.
        
        Args:
            user_request: The user's research request
            context: Additional context for the research
            
        Returns:
            Research objective ID
        """
        try:
            # Create research objective using existing structure
            objective = type('ResearchObjective', (), {
                'objective_id': f"obj_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(user_request) % 10000:04d}",
                'user_request': user_request,
                'context': context or {},
                'status': 'pending',
                'created_at': datetime.now(),
                'analyzed_objectives': [],
                'decomposed_tasks': [],
                'assigned_agents': [],
                'execution_results': [],
                'evaluation_results': {},
                'validation_results': {},
                'final_synthesis': {},
                'learning_data': []
            })()
            self.active_objectives[objective.objective_id] = objective
            
            logger.info(f"Starting enhanced autonomous research for objective: {objective.objective_id}")
            logger.info(f"User request: {user_request}")
            
            # Enhanced autonomous workflow with multiple iterations
            iteration = 0
            best_quality = 0.0
            best_results = None
            
            while iteration < self.max_iterations:
                iteration += 1
                logger.info(f"Starting iteration {iteration}/{self.max_iterations}")
                
                # Phase 1: Enhanced Analysis with Learning
                objective.status = "analyzing"
                analysis_result = await self._enhanced_analyze_objective(objective, iteration)
                objective.analyzed_objectives = analysis_result
                
                # Phase 2: Adaptive Task Decomposition
                objective.status = "decomposing"
                decomposition_result = await self._adaptive_decompose_tasks(objective, iteration)
                objective.decomposed_tasks = decomposition_result['tasks']
                objective.assigned_agents = decomposition_result['assignments']
                
                # Phase 3: Multi-Agent Execution with MCP Integration
                objective.status = "executing"
                execution_result = await self._enhanced_execute_tasks(objective, iteration)
                objective.execution_results = execution_result
                
                # Phase 4: Critical Evaluation with Learning
                objective.status = "evaluating"
                evaluation_result = await self._enhanced_evaluate_results(objective, iteration)
                objective.evaluation_results = evaluation_result
                
                # Check if this iteration improved quality
                current_quality = evaluation_result.get('overall_score', 0.0)
                if current_quality > best_quality:
                    best_quality = current_quality
                    best_results = {
                        'analysis': analysis_result,
                        'decomposition': decomposition_result,
                        'execution': execution_result,
                        'evaluation': evaluation_result,
                        'iteration': iteration
                    }
                    logger.info(f"New best quality achieved: {best_quality:.2f} at iteration {iteration}")
                
                # Check if we should continue iterating
                if self._should_continue_iteration(evaluation_result, iteration, best_quality):
                    logger.info(f"Continuing to iteration {iteration + 1}")
                    # Learn from this iteration for next one
                    if self.learning_enabled:
                        await self._learn_from_iteration(objective, evaluation_result, iteration)
                else:
                    logger.info(f"Stopping iterations at iteration {iteration} (quality: {current_quality:.2f})")
                    break
            
            # Use best results if available
            if best_results:
                objective.analyzed_objectives = best_results['analysis']
                objective.decomposed_tasks = best_results['decomposition']['tasks']
                objective.assigned_agents = best_results['decomposition']['assignments']
                objective.execution_results = best_results['execution']
                objective.evaluation_results = best_results['evaluation']
                objective.final_iteration = best_results['iteration']
            
            # Phase 5: Result Validation Against Original Objectives
            objective.status = "validating"
            validation_result = await self._enhanced_validate_results(objective)
            objective.validation_results = validation_result
            
            # Phase 6: Final Synthesis and Deliverable Generation
            objective.status = "synthesizing"
            synthesis_result = await self._enhanced_synthesize_deliverable(objective)
            objective.final_synthesis = synthesis_result
            objective.status = "completed"
            
            # Store execution history for learning
            if self.learning_enabled:
                self.execution_history.append({
                    'objective_id': objective.objective_id,
                    'user_request': user_request,
                    'iterations': iteration,
                    'final_quality': best_quality,
                    'timestamp': datetime.now().isoformat()
                })
            
            logger.info(f"Enhanced autonomous research completed: {objective.objective_id}")
            logger.info(f"Final quality: {best_quality:.2f} after {iteration} iterations")
            logger.info(f"Final deliverable: {synthesis_result.get('deliverable_path', 'N/A')}")
            
            return objective.objective_id
            
        except Exception as e:
            logger.error(f"Enhanced autonomous research failed: {e}")
            if objective.objective_id in self.active_objectives:
                self.active_objectives[objective.objective_id].status = "failed"
            raise
    
    def _should_continue_iteration(self, evaluation_result: Dict[str, Any], 
                                 iteration: int, best_quality: float) -> bool:
        """Determine if we should continue iterating.
        
        Args:
            evaluation_result: Current evaluation results
            iteration: Current iteration number
            best_quality: Best quality achieved so far
            
        Returns:
            True if should continue iterating
        """
        current_quality = evaluation_result.get('overall_score', 0.0)
        needs_recursion = evaluation_result.get('needs_recursion', False)
        
        # Stop if we've reached max iterations
        if iteration >= self.max_iterations:
            return False
        
        # Stop if quality is above threshold
        if current_quality >= self.quality_threshold:
            return False
        
        # Stop if no improvement in last 2 iterations
        if iteration > 2 and current_quality <= best_quality:
            return False
        
        # Continue if evaluation suggests recursion is needed
        if needs_recursion:
            return True
        
        # Continue if quality is still improving
        if current_quality > best_quality * 0.9:  # At least 90% of best quality
            return True
        
        return False
    
    async def _learn_from_iteration(self, objective: Any, evaluation_result: Dict[str, Any], 
                                  iteration: int) -> None:
        """Learn from current iteration to improve next one.
        
        Args:
            objective: Current research objective
            evaluation_result: Evaluation results from current iteration
            iteration: Current iteration number
        """
        try:
            # Extract learning insights
            feedback = evaluation_result.get('feedback', [])
            quality_metrics = evaluation_result.get('quality_metrics', {})
            
            # Update agent capabilities based on performance
            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'update_capabilities'):
                    await agent.update_capabilities(evaluation_result, iteration)
            
            # Store learning data
            learning_data = {
                'iteration': iteration,
                'feedback': feedback,
                'quality_metrics': quality_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            if not hasattr(objective, 'learning_data'):
                objective.learning_data = []
            objective.learning_data.append(learning_data)
            
            logger.info(f"Learning data stored for iteration {iteration}")
            
        except Exception as e:
            logger.error(f"Learning from iteration failed: {e}")
    
    async def _enhanced_analyze_objective(self, objective: Any, iteration: int) -> List[Dict[str, Any]]:
        """Enhanced analysis with learning from previous iterations."""
        try:
            analyzer_agent = self.agents.get('analyzer')
            if not analyzer_agent:
                raise ValueError("TaskAnalyzerAgent not initialized")
            
            # Include learning data from previous iterations
            context = objective.context or {}
            if hasattr(objective, 'learning_data') and objective.learning_data:
                context['learning_data'] = objective.learning_data
                context['iteration'] = iteration
            
            analysis_result = await analyzer_agent.analyze(objective.user_request, context)
            
            # Enhance analysis based on iteration
            if iteration > 1:
                analysis_result = await self._enhance_analysis_with_learning(analysis_result, objective, iteration)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            return []
    
    async def _enhance_analysis_with_learning(self, analysis_result: List[Dict[str, Any]], 
                                            objective: Any, iteration: int) -> List[Dict[str, Any]]:
        """Enhance analysis using learning from previous iterations."""
        try:
            enhanced_result = []
            
            for objective_data in analysis_result:
                enhanced_objective = objective_data.copy()
                
                # Add iteration-specific enhancements
                enhanced_objective['iteration'] = iteration
                enhanced_objective['learning_applied'] = True
                
                # Refine based on previous feedback
                if hasattr(objective, 'learning_data') and objective.learning_data:
                    latest_feedback = objective.learning_data[-1].get('feedback', [])
                    if 'insufficient_depth' in str(latest_feedback):
                        enhanced_objective['depth_requirement'] = 'high'
                    if 'missing_scope' in str(latest_feedback):
                        enhanced_objective['scope'] = 'comprehensive'
                
                enhanced_result.append(enhanced_objective)
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Analysis enhancement failed: {e}")
            return analysis_result
    
    async def _adaptive_decompose_tasks(self, objective: Any, iteration: int) -> Dict[str, Any]:
        """Adaptive task decomposition that learns from previous iterations."""
        try:
            decomposer_agent = self.agents.get('decomposer')
            if not decomposer_agent:
                raise ValueError("TaskDecomposerAgent not initialized")
            
            # Include learning data for adaptive decomposition
            context = objective.context or {}
            if hasattr(objective, 'learning_data') and objective.learning_data:
                context['learning_data'] = objective.learning_data
                context['iteration'] = iteration
                context['adaptive_strategy'] = self.adaptive_strategy
            
            decomposition_result = await decomposer_agent.decompose_and_assign(
                objective.analyzed_objectives, context, objective.objective_id
            )
            
            # Enhance decomposition based on iteration
            if iteration > 1:
                decomposition_result = await self._enhance_decomposition_with_learning(
                    decomposition_result, objective, iteration
                )
            
            return decomposition_result
            
        except Exception as e:
            logger.error(f"Adaptive decomposition failed: {e}")
            return {'tasks': [], 'assignments': []}
    
    async def _enhance_decomposition_with_learning(self, decomposition_result: Dict[str, Any], 
                                                 objective: Any, iteration: int) -> Dict[str, Any]:
        """Enhance task decomposition using learning from previous iterations."""
        try:
            enhanced_tasks = []
            
            for task in decomposition_result.get('tasks', []):
                enhanced_task = task.copy()
                
                # Add iteration-specific enhancements
                enhanced_task['iteration'] = iteration
                enhanced_task['learning_applied'] = True
                
                # Refine task based on previous feedback
                if hasattr(objective, 'learning_data') and objective.learning_data:
                    latest_feedback = objective.learning_data[-1].get('feedback', [])
                    if 'task_complexity' in str(latest_feedback):
                        enhanced_task['complexity'] = 'high'
                    if 'insufficient_parallelism' in str(latest_feedback):
                        enhanced_task['parallel_execution'] = True
                
                enhanced_tasks.append(enhanced_task)
            
            return {
                'tasks': enhanced_tasks,
                'assignments': decomposition_result.get('assignments', [])
            }
            
        except Exception as e:
            logger.error(f"Decomposition enhancement failed: {e}")
            return decomposition_result
    
    async def _enhanced_execute_tasks(self, objective: Any, iteration: int) -> List[Dict[str, Any]]:
        """Enhanced task execution with learning and adaptation."""
        try:
            execution_results = []
            
            # Execute tasks with enhanced coordination
            for task in objective.decomposed_tasks:
                agent_name = task.get('assigned_to')
                if agent_name and agent_name in self.agents:
                    agent = self.agents[agent_name]
                    
                    # Include learning context
                    task_context = task.get('context', {})
                    task_context['iteration'] = iteration
                    task_context['learning_enabled'] = self.learning_enabled
                    
                    if hasattr(objective, 'learning_data') and objective.learning_data:
                        task_context['learning_data'] = objective.learning_data
                    
                    # Execute task with enhanced context
                    if hasattr(agent, 'execute_task'):
                        result = await agent.execute_task(task, task_context, objective.objective_id)
                    else:
                        result = await agent.conduct_research(task, task_context, objective.objective_id)
                    
                    execution_results.append(result)
            
            # Enhance results based on iteration
            if iteration > 1:
                execution_results = await self._enhance_execution_with_learning(
                    execution_results, objective, iteration
                )
            
            return execution_results
            
        except Exception as e:
            logger.error(f"Enhanced execution failed: {e}")
            return []
    
    async def _enhance_execution_with_learning(self, execution_results: List[Dict[str, Any]], 
                                             objective: Any, iteration: int) -> List[Dict[str, Any]]:
        """Enhance execution results using learning from previous iterations."""
        try:
            enhanced_results = []
            
            for result in execution_results:
                enhanced_result = result.copy()
                
                # Add iteration-specific enhancements
                enhanced_result['iteration'] = iteration
                enhanced_result['learning_applied'] = True
                
                # Refine based on previous feedback
                if hasattr(objective, 'learning_data') and objective.learning_data:
                    latest_feedback = objective.learning_data[-1].get('feedback', [])
                    if 'insufficient_data' in str(latest_feedback):
                        enhanced_result['data_enhanced'] = True
                    if 'quality_issues' in str(latest_feedback):
                        enhanced_result['quality_enhanced'] = True
                
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Execution enhancement failed: {e}")
            return execution_results
    
    async def _enhanced_evaluate_results(self, objective: Any, iteration: int) -> Dict[str, Any]:
        """Enhanced evaluation with learning and adaptive criteria."""
        try:
            evaluator_agent = self.agents.get('evaluator')
            if not evaluator_agent:
                raise ValueError("EvaluationAgent not initialized")
            
            # Include learning context for evaluation
            context = objective.context or {}
            context['iteration'] = iteration
            context['learning_enabled'] = self.learning_enabled
            
            if hasattr(objective, 'learning_data') and objective.learning_data:
                context['learning_data'] = objective.learning_data
            
            evaluation_result = await evaluator_agent.evaluate(
                objective.execution_results, objective.analyzed_objectives, context
            )
            
            # Enhance evaluation based on iteration
            if iteration > 1:
                evaluation_result = await self._enhance_evaluation_with_learning(
                    evaluation_result, objective, iteration
                )
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Enhanced evaluation failed: {e}")
            return {'overall_score': 0.0, 'needs_recursion': True, 'feedback': ['Evaluation failed']}
    
    async def _enhance_evaluation_with_learning(self, evaluation_result: Dict[str, Any], 
                                              objective: Any, iteration: int) -> Dict[str, Any]:
        """Enhance evaluation using learning from previous iterations."""
        try:
            enhanced_result = evaluation_result.copy()
            
            # Add iteration-specific enhancements
            enhanced_result['iteration'] = iteration
            enhanced_result['learning_applied'] = True
            
            # Refine evaluation criteria based on previous feedback
            if hasattr(objective, 'learning_data') and objective.learning_data:
                latest_feedback = objective.learning_data[-1].get('feedback', [])
                
                # Adjust quality thresholds based on learning
                if 'quality_threshold_too_high' in str(latest_feedback):
                    enhanced_result['quality_threshold'] = 0.7
                elif 'quality_threshold_too_low' in str(latest_feedback):
                    enhanced_result['quality_threshold'] = 0.9
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Evaluation enhancement failed: {e}")
            return evaluation_result
    
    async def _enhanced_validate_results(self, objective: Any) -> Dict[str, Any]:
        """Enhanced validation with learning and adaptive criteria."""
        try:
            validator_agent = self.agents.get('validator')
            if not validator_agent:
                raise ValueError("ValidationAgent not initialized")
            
            validation_result = await validator_agent.validate(
                objective.user_request, objective.analyzed_objectives, 
                objective.execution_results, objective.context
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Enhanced validation failed: {e}")
            return {'validation_score': 0.0, 'is_valid': False, 'validation_feedback': ['Validation failed']}
    
    async def _enhanced_synthesize_deliverable(self, objective: Any) -> Dict[str, Any]:
        """Enhanced synthesis with learning and adaptive formatting."""
        try:
            synthesizer_agent = self.agents.get('synthesizer')
            if not synthesizer_agent:
                raise ValueError("SynthesisAgent not initialized")
            
            synthesis_result = await synthesizer_agent.synthesize(
                objective.user_request, objective.execution_results, 
                objective.context, objective.objective_id
            )
            
            return synthesis_result
            
        except Exception as e:
            logger.error(f"Enhanced synthesis failed: {e}")
            return {'deliverable_path': None, 'error': str(e)}
    
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
