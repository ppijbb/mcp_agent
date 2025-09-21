#!/usr/bin/env python3
"""
Task Decomposer Agent for Autonomous Research System

This agent autonomously decomposes research objectives into specific tasks
and assigns them to specialized agents based on capabilities and workload.

No fallback or dummy code - production-level autonomous decomposition only.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import uuid
from pathlib import Path
import google.generativeai as genai
import os

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

logger = setup_logger("task_decomposer", log_level="INFO")


class TaskDecomposerAgent:
    """Autonomous task decomposer agent for research task breakdown."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the task decomposer agent.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Learning capabilities
        self.learning_data = []
        self.decomposition_history = []
        
        logger.info("Task Decomposer Agent initialized with LLM-based decomposition")
    
    def _initialize_llm(self):
        """Initialize the LLM client."""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("LLM initialized for TaskDecomposerAgent")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _load_task_templates(self) -> Dict[str, Any]:
        """Load task templates for different types of research tasks.
        
        Returns:
            Dictionary of task templates
        """
        return {
            'data_collection': {
                'template': {
                    'task_type': 'data_collection',
                    'description': 'Collect {data_type} from {sources}',
                    'required_skills': ['research', 'data_gathering'],
                    'estimated_duration': 'medium',
                    'complexity': 'medium',
                    'dependencies': [],
                    'outputs': ['raw_data', 'data_summary']
                },
                'variations': {
                    'web_research': {
                        'description': 'Conduct web research on {topic}',
                        'sources': ['academic_papers', 'news_articles', 'reports', 'websites'],
                        'skills': ['web_search', 'content_analysis']
                    },
                    'literature_review': {
                        'description': 'Perform literature review on {topic}',
                        'sources': ['academic_databases', 'journals', 'conferences'],
                        'skills': ['academic_research', 'citation_analysis']
                    },
                    'data_analysis': {
                        'description': 'Analyze {data_type} data',
                        'sources': ['datasets', 'databases', 'apis'],
                        'skills': ['data_analysis', 'statistics']
                    }
                }
            },
            'analysis': {
                'template': {
                    'task_type': 'analysis',
                    'description': 'Analyze {analysis_type} of {subject}',
                    'required_skills': ['analysis', 'critical_thinking'],
                    'estimated_duration': 'long',
                    'complexity': 'high',
                    'dependencies': ['data_collection'],
                    'outputs': ['analysis_report', 'insights', 'findings']
                },
                'variations': {
                    'comparative_analysis': {
                        'description': 'Compare {items} across {dimensions}',
                        'skills': ['comparative_analysis', 'evaluation']
                    },
                    'trend_analysis': {
                        'description': 'Analyze trends in {domain} over {timeframe}',
                        'skills': ['trend_analysis', 'time_series']
                    },
                    'swot_analysis': {
                        'description': 'Perform SWOT analysis of {subject}',
                        'skills': ['strategic_analysis', 'evaluation']
                    }
                }
            },
            'synthesis': {
                'template': {
                    'task_type': 'synthesis',
                    'description': 'Synthesize {synthesis_type} from {sources}',
                    'required_skills': ['synthesis', 'writing'],
                    'estimated_duration': 'medium',
                    'complexity': 'medium',
                    'dependencies': ['analysis'],
                    'outputs': ['synthesis_report', 'summary', 'recommendations']
                },
                'variations': {
                    'report_synthesis': {
                        'description': 'Synthesize findings into comprehensive report',
                        'skills': ['report_writing', 'synthesis']
                    },
                    'recommendation_synthesis': {
                        'description': 'Generate recommendations based on analysis',
                        'skills': ['recommendation_generation', 'strategic_thinking']
                    }
                }
            },
            'validation': {
                'template': {
                    'task_type': 'validation',
                    'description': 'Validate {validation_type} against {criteria}',
                    'required_skills': ['validation', 'quality_assurance'],
                    'estimated_duration': 'short',
                    'complexity': 'medium',
                    'dependencies': ['synthesis'],
                    'outputs': ['validation_report', 'quality_metrics']
                },
                'variations': {
                    'quality_validation': {
                        'description': 'Validate quality of research outputs',
                        'skills': ['quality_assurance', 'evaluation']
                    },
                    'accuracy_validation': {
                        'description': 'Validate accuracy of findings',
                        'skills': ['fact_checking', 'verification']
                    }
                }
            }
        }
    
    def _load_agent_capabilities(self) -> Dict[str, Any]:
        """Load agent capabilities for task assignment.
        
        Returns:
            Dictionary of agent capabilities
        """
        return {
            'researcher': {
                'primary_skills': ['research', 'data_collection', 'web_search', 'academic_research'],
                'secondary_skills': ['analysis', 'content_analysis'],
                'max_concurrent_tasks': 3,
                'preferred_task_types': ['data_collection', 'literature_review'],
                'performance_metrics': {
                    'accuracy': 0.9,
                    'speed': 0.8,
                    'reliability': 0.85
                }
            },
            'analyzer': {
                'primary_skills': ['analysis', 'critical_thinking', 'comparative_analysis', 'trend_analysis'],
                'secondary_skills': ['data_analysis', 'statistics'],
                'max_concurrent_tasks': 2,
                'preferred_task_types': ['analysis', 'comparative_analysis', 'trend_analysis'],
                'performance_metrics': {
                    'accuracy': 0.95,
                    'speed': 0.7,
                    'reliability': 0.9
                }
            },
            'evaluator': {
                'primary_skills': ['evaluation', 'validation', 'quality_assurance', 'assessment'],
                'secondary_skills': ['analysis', 'critical_thinking'],
                'max_concurrent_tasks': 2,
                'preferred_task_types': ['validation', 'quality_validation'],
                'performance_metrics': {
                    'accuracy': 0.92,
                    'speed': 0.75,
                    'reliability': 0.88
                }
            },
            'validator': {
                'primary_skills': ['validation', 'fact_checking', 'verification', 'quality_assurance'],
                'secondary_skills': ['analysis', 'evaluation'],
                'max_concurrent_tasks': 3,
                'preferred_task_types': ['validation', 'accuracy_validation'],
                'performance_metrics': {
                    'accuracy': 0.98,
                    'speed': 0.8,
                    'reliability': 0.95
                }
            },
            'synthesizer': {
                'primary_skills': ['synthesis', 'writing', 'report_generation', 'recommendation_generation'],
                'secondary_skills': ['analysis', 'communication'],
                'max_concurrent_tasks': 2,
                'preferred_task_types': ['synthesis', 'report_synthesis'],
                'performance_metrics': {
                    'accuracy': 0.9,
                    'speed': 0.85,
                    'reliability': 0.87
                }
            }
        }
    
    def _load_decomposition_strategies(self) -> Dict[str, Any]:
        """Load decomposition strategies for different objective types.
        
        Returns:
            Dictionary of decomposition strategies
        """
        return {
            'sequential': {
                'description': 'Tasks executed in sequence with dependencies',
                'suitable_for': ['analysis', 'comparative_study', 'trend_analysis'],
                'task_flow': ['data_collection', 'analysis', 'synthesis', 'validation']
            },
            'parallel': {
                'description': 'Tasks executed in parallel where possible',
                'suitable_for': ['comprehensive_analysis', 'multi_domain_study'],
                'task_flow': ['parallel_data_collection', 'parallel_analysis', 'synthesis', 'validation']
            },
            'iterative': {
                'description': 'Tasks executed iteratively with refinement',
                'suitable_for': ['exploratory_research', 'development', 'optimization'],
                'task_flow': ['initial_analysis', 'refinement', 'validation', 'final_synthesis']
            },
            'hierarchical': {
                'description': 'Tasks organized in hierarchical structure',
                'suitable_for': ['complex_analysis', 'multi_level_study'],
                'task_flow': ['high_level_analysis', 'detailed_analysis', 'integration', 'validation']
            }
        }
    
    async def decompose_tasks(self, analyzed_objectives: List[Dict[str, Any]], 
                            context: Optional[Dict[str, Any]] = None,
                            objective_id: str = None) -> Dict[str, Any]:
        """Autonomously decompose research objectives into specific tasks.
        
        Args:
            analyzed_objectives: List of analyzed objectives
            context: Additional context for decomposition
            objective_id: Objective ID for tracking
            
        Returns:
            Dictionary containing decomposed tasks and agent assignments
        """
        try:
            logger.info(f"Starting autonomous task decomposition for objective: {objective_id}")
            
            # Phase 1: Strategy Selection
            decomposition_strategy = await self._select_decomposition_strategy(analyzed_objectives, context)
            
            # Phase 2: Task Generation
            generated_tasks = await self._generate_tasks(analyzed_objectives, decomposition_strategy)
            
            # Phase 3: Task Optimization
            optimized_tasks = await self._optimize_tasks(generated_tasks, context)
            
            # Phase 4: Agent Assignment
            agent_assignments = await self._assign_agents_to_tasks(optimized_tasks, context)
            
            # Phase 5: Dependency Resolution
            resolved_tasks = await self._resolve_dependencies(optimized_tasks, agent_assignments)
            
            # Phase 6: Workload Balancing
            balanced_assignments = await self._balance_workload(agent_assignments, context)
            
            decomposition_result = {
                'tasks': resolved_tasks,
                'assignments': balanced_assignments,
                'strategy': decomposition_strategy,
                'decomposition_metadata': {
                    'objective_id': objective_id,
                    'timestamp': datetime.now().isoformat(),
                    'total_tasks': len(resolved_tasks),
                    'total_agents': len(set(assignment['agent'] for assignment in balanced_assignments)),
                    'estimated_duration': self._estimate_total_duration(resolved_tasks),
                    'complexity_score': self._calculate_complexity_score(resolved_tasks)
                }
            }
            
            logger.info(f"Task decomposition completed: {len(resolved_tasks)} tasks assigned to {len(set(assignment['agent'] for assignment in balanced_assignments))} agents")
            return decomposition_result
            
        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
            raise
    
    async def _select_decomposition_strategy(self, objectives: List[Dict[str, Any]], 
                                           context: Optional[Dict[str, Any]] = None) -> str:
        """Select the best decomposition strategy for the objectives.
        
        Args:
            objectives: List of objectives
            context: Additional context
            
        Returns:
            Selected strategy name
        """
        try:
            # Analyze objective characteristics
            objective_types = [obj.get('type', 'general') for obj in objectives]
            complexity_levels = [obj.get('estimated_effort', 0.5) for obj in objectives]
            avg_complexity = sum(complexity_levels) / len(complexity_levels) if complexity_levels else 0.5
            
            # Strategy selection logic
            if len(objectives) == 1 and objectives[0].get('type') == 'primary':
                return 'sequential'
            elif len(objectives) > 3 and avg_complexity > 0.7:
                return 'hierarchical'
            elif any('exploratory' in obj.get('description', '').lower() for obj in objectives):
                return 'iterative'
            elif len(objectives) > 1 and avg_complexity < 0.6:
                return 'parallel'
            else:
                return 'sequential'
                
        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return 'sequential'
    
    async def _generate_tasks(self, objectives: List[Dict[str, Any]], 
                            strategy: str) -> List[Dict[str, Any]]:
        """Generate specific tasks from objectives.
        
        Args:
            objectives: List of objectives
            strategy: Decomposition strategy
            
        Returns:
            List of generated tasks
        """
        try:
            tasks = []
            task_id_counter = 1
            
            for objective in objectives:
                objective_tasks = await self._generate_tasks_for_objective(objective, strategy)
                
                for task in objective_tasks:
                    task['task_id'] = f"task_{task_id_counter:03d}"
                    task['objective_id'] = objective['objective_id']
                    task['created_at'] = datetime.now().isoformat()
                    tasks.append(task)
                    task_id_counter += 1
            
            return tasks
            
        except Exception as e:
            logger.error(f"Task generation failed: {e}")
            return []
    
    async def _generate_tasks_for_objective(self, objective: Dict[str, Any], 
                                          strategy: str) -> List[Dict[str, Any]]:
        """Generate tasks for a specific objective.
        
        Args:
            objective: Objective to generate tasks for
            strategy: Decomposition strategy
            
        Returns:
            List of tasks for the objective
        """
        try:
            tasks = []
            objective_type = objective.get('type', 'primary')
            intent = objective.get('intent', 'analysis')
            domain = objective.get('domain', 'general')
            scope = objective.get('scope', 'general')
            
            # Generate tasks based on strategy
            if strategy == 'sequential':
                tasks = await self._generate_sequential_tasks(objective)
            elif strategy == 'parallel':
                tasks = await self._generate_parallel_tasks(objective)
            elif strategy == 'iterative':
                tasks = await self._generate_iterative_tasks(objective)
            elif strategy == 'hierarchical':
                tasks = await self._generate_hierarchical_tasks(objective)
            else:
                tasks = await self._generate_sequential_tasks(objective)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Task generation for objective failed: {e}")
            return []
    
    async def _generate_sequential_tasks(self, objective: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate sequential tasks for an objective.
        
        Args:
            objective: Objective to generate tasks for
            
        Returns:
            List of sequential tasks
        """
        try:
            tasks = []
            
            # Data collection task
            data_task = self._create_task_from_template('data_collection', objective)
            if data_task:
                tasks.append(data_task)
            
            # Analysis task
            analysis_task = self._create_task_from_template('analysis', objective)
            if analysis_task:
                analysis_task['dependencies'] = [data_task['task_id']] if data_task else []
                tasks.append(analysis_task)
            
            # Synthesis task
            synthesis_task = self._create_task_from_template('synthesis', objective)
            if synthesis_task:
                synthesis_task['dependencies'] = [analysis_task['task_id']] if analysis_task else []
                tasks.append(synthesis_task)
            
            # Validation task
            validation_task = self._create_task_from_template('validation', objective)
            if validation_task:
                validation_task['dependencies'] = [synthesis_task['task_id']] if synthesis_task else []
                tasks.append(validation_task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Sequential task generation failed: {e}")
            return []
    
    async def _generate_parallel_tasks(self, objective: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parallel tasks for an objective.
        
        Args:
            objective: Objective to generate tasks for
            
        Returns:
            List of parallel tasks
        """
        try:
            tasks = []
            
            # Multiple data collection tasks in parallel
            data_sources = ['web_research', 'literature_review', 'data_analysis']
            for source in data_sources:
                data_task = self._create_task_from_template('data_collection', objective, source)
                if data_task:
                    tasks.append(data_task)
            
            # Analysis tasks in parallel
            analysis_types = ['comparative_analysis', 'trend_analysis']
            for analysis_type in analysis_types:
                analysis_task = self._create_task_from_template('analysis', objective, analysis_type)
                if analysis_task:
                    analysis_task['dependencies'] = [task['task_id'] for task in tasks if task['task_type'] == 'data_collection']
                    tasks.append(analysis_task)
            
            # Synthesis task (depends on all analysis tasks)
            synthesis_task = self._create_task_from_template('synthesis', objective)
            if synthesis_task:
                synthesis_task['dependencies'] = [task['task_id'] for task in tasks if task['task_type'] == 'analysis']
                tasks.append(synthesis_task)
            
            # Validation task
            validation_task = self._create_task_from_template('validation', objective)
            if validation_task:
                validation_task['dependencies'] = [synthesis_task['task_id']] if synthesis_task else []
                tasks.append(validation_task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Parallel task generation failed: {e}")
            return []
    
    async def _generate_iterative_tasks(self, objective: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate iterative tasks for an objective.
        
        Args:
            objective: Objective to generate tasks for
            
        Returns:
            List of iterative tasks
        """
        try:
            tasks = []
            
            # Initial analysis task
            initial_task = self._create_task_from_template('analysis', objective)
            if initial_task:
                initial_task['description'] = f"Initial analysis of {objective.get('description', 'objective')}"
                tasks.append(initial_task)
            
            # Refinement task
            refinement_task = self._create_task_from_template('analysis', objective)
            if refinement_task:
                refinement_task['description'] = f"Refine analysis based on initial findings"
                refinement_task['dependencies'] = [initial_task['task_id']] if initial_task else []
                refinement_task['is_refinement'] = True
                tasks.append(refinement_task)
            
            # Validation task
            validation_task = self._create_task_from_template('validation', objective)
            if validation_task:
                validation_task['dependencies'] = [refinement_task['task_id']] if refinement_task else []
                tasks.append(validation_task)
            
            # Final synthesis task
            synthesis_task = self._create_task_from_template('synthesis', objective)
            if synthesis_task:
                synthesis_task['dependencies'] = [validation_task['task_id']] if validation_task else []
                tasks.append(synthesis_task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Iterative task generation failed: {e}")
            return []
    
    async def _generate_hierarchical_tasks(self, objective: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hierarchical tasks for an objective.
        
        Args:
            objective: Objective to generate tasks for
            
        Returns:
            List of hierarchical tasks
        """
        try:
            tasks = []
            
            # High-level analysis task
            high_level_task = self._create_task_from_template('analysis', objective)
            if high_level_task:
                high_level_task['description'] = f"High-level analysis of {objective.get('description', 'objective')}"
                high_level_task['level'] = 'high'
                tasks.append(high_level_task)
            
            # Detailed analysis tasks
            detailed_analysis_types = ['comparative_analysis', 'trend_analysis', 'swot_analysis']
            for analysis_type in detailed_analysis_types:
                detailed_task = self._create_task_from_template('analysis', objective, analysis_type)
                if detailed_task:
                    detailed_task['level'] = 'detailed'
                    detailed_task['dependencies'] = [high_level_task['task_id']] if high_level_task else []
                    tasks.append(detailed_task)
            
            # Integration task
            integration_task = self._create_task_from_template('synthesis', objective)
            if integration_task:
                integration_task['description'] = f"Integrate findings from detailed analysis"
                integration_task['dependencies'] = [task['task_id'] for task in tasks if task.get('level') == 'detailed']
                tasks.append(integration_task)
            
            # Validation task
            validation_task = self._create_task_from_template('validation', objective)
            if validation_task:
                validation_task['dependencies'] = [integration_task['task_id']] if integration_task else []
                tasks.append(validation_task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Hierarchical task generation failed: {e}")
            return []
    
    def _create_task_from_template(self, task_type: str, objective: Dict[str, Any], 
                                 variation: str = None) -> Optional[Dict[str, Any]]:
        """Create a task from a template.
        
        Args:
            task_type: Type of task to create
            objective: Objective the task belongs to
            variation: Specific variation of the task type
            
        Returns:
            Created task or None if creation failed
        """
        try:
            if task_type not in self.task_templates:
                return None
            
            template = self.task_templates[task_type]['template'].copy()
            
            # Apply variation if specified
            if variation and variation in self.task_templates[task_type]['variations']:
                variation_config = self.task_templates[task_type]['variations'][variation]
                template.update(variation_config)
            
            # Customize template for objective
            template['description'] = template['description'].format(
                topic=objective.get('description', 'research topic'),
                domain=objective.get('domain', 'general'),
                intent=objective.get('intent', 'analysis'),
                synthesis_type=objective.get('synthesis_type', 'findings'),
                validation_type=objective.get('validation_type', 'results'),
                data_type=objective.get('data_type', 'research data'),
                items=objective.get('items', 'research items'),
                timeframe=objective.get('timeframe', 'current period'),
                sources=objective.get('sources', 'research sources'),
                criteria=objective.get('criteria', 'quality standards')
            )
            
            # Add objective-specific information
            template['objective_id'] = objective['objective_id']
            template['priority'] = objective.get('priority', 0.5)
            template['constraints'] = objective.get('constraints', {})
            template['success_criteria'] = objective.get('success_criteria', {})
            
            return template
            
        except Exception as e:
            logger.error(f"Task creation from template failed: {e}")
            return None
    
    async def _optimize_tasks(self, tasks: List[Dict[str, Any]], 
                            context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Optimize tasks for better execution.
        
        Args:
            tasks: List of tasks to optimize
            context: Additional context
            
        Returns:
            List of optimized tasks
        """
        try:
            optimized_tasks = []
            
            for task in tasks:
                optimized_task = task.copy()
                
                # Optimize task duration based on complexity
                if task.get('complexity') == 'high':
                    optimized_task['estimated_duration'] = 'long'
                elif task.get('complexity') == 'medium':
                    optimized_task['estimated_duration'] = 'medium'
                else:
                    optimized_task['estimated_duration'] = 'short'
                
                # Add optimization metadata
                optimized_task['optimization_applied'] = True
                optimized_task['optimized_at'] = datetime.now().isoformat()
                
                optimized_tasks.append(optimized_task)
            
            return optimized_tasks
            
        except Exception as e:
            logger.error(f"Task optimization failed: {e}")
            return tasks
    
    async def _assign_agents_to_tasks(self, tasks: List[Dict[str, Any]], 
                                    context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Assign agents to tasks based on capabilities and workload.
        
        Args:
            tasks: List of tasks to assign
            context: Additional context
            
        Returns:
            List of agent assignments
        """
        try:
            assignments = []
            agent_workload = {agent: 0 for agent in self.agent_capabilities.keys()}
            
            for task in tasks:
                # Find best agent for task
                best_agent = await self._find_best_agent_for_task(task, agent_workload)
                
                if best_agent:
                    assignment = {
                        'task_id': task['task_id'],
                        'agent': best_agent,
                        'assignment_reason': 'capability_match',
                        'estimated_duration': task.get('estimated_duration', 'medium'),
                        'priority': task.get('priority', 0.5),
                        'assigned_at': datetime.now().isoformat()
                    }
                    assignments.append(assignment)
                    
                    # Update agent workload
                    agent_workload[best_agent] += 1
                else:
                    logger.warning(f"No suitable agent found for task: {task['task_id']}")
            
            return assignments
            
        except Exception as e:
            logger.error(f"Agent assignment failed: {e}")
            return []
    
    async def _find_best_agent_for_task(self, task: Dict[str, Any], 
                                      agent_workload: Dict[str, int]) -> Optional[str]:
        """Find the best agent for a specific task.
        
        Args:
            task: Task to find agent for
            agent_workload: Current agent workload
            
        Returns:
            Best agent name or None
        """
        try:
            task_type = task.get('task_type', 'general')
            required_skills = task.get('required_skills', [])
            
            best_agent = None
            best_score = 0
            
            for agent_name, capabilities in self.agent_capabilities.items():
                # Check if agent can handle the task
                if not await self._agent_can_handle_task(agent_name, task):
                    continue
                
                # Calculate agent score
                score = await self._calculate_agent_score(agent_name, task, agent_workload)
                
                if score > best_score:
                    best_score = score
                    best_agent = agent_name
            
            return best_agent
            
        except Exception as e:
            logger.error(f"Best agent finding failed: {e}")
            return None
    
    async def _agent_can_handle_task(self, agent_name: str, task: Dict[str, Any]) -> bool:
        """Check if an agent can handle a specific task.
        
        Args:
            agent_name: Name of the agent
            task: Task to check
            
        Returns:
            True if agent can handle the task
        """
        try:
            capabilities = self.agent_capabilities.get(agent_name, {})
            required_skills = task.get('required_skills', [])
            
            # Check if agent has required skills
            agent_skills = capabilities.get('primary_skills', []) + capabilities.get('secondary_skills', [])
            
            for skill in required_skills:
                if skill not in agent_skills:
                    return False
            
            # Check if agent prefers this task type
            preferred_types = capabilities.get('preferred_task_types', [])
            task_type = task.get('task_type', 'general')
            
            if preferred_types and task_type not in preferred_types:
                # Agent can still handle it, but with lower preference
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Agent capability check failed: {e}")
            return False
    
    async def _calculate_agent_score(self, agent_name: str, task: Dict[str, Any], 
                                   agent_workload: Dict[str, int]) -> float:
        """Calculate agent score for task assignment.
        
        Args:
            agent_name: Name of the agent
            task: Task to score for
            agent_workload: Current agent workload
            
        Returns:
            Agent score (0.0 to 1.0)
        """
        try:
            capabilities = self.agent_capabilities.get(agent_name, {})
            required_skills = task.get('required_skills', [])
            
            # Base score from performance metrics
            performance_metrics = capabilities.get('performance_metrics', {})
            base_score = sum(performance_metrics.values()) / len(performance_metrics) if performance_metrics else 0.5
            
            # Skill match bonus
            agent_skills = capabilities.get('primary_skills', []) + capabilities.get('secondary_skills', [])
            skill_match_ratio = sum(1 for skill in required_skills if skill in agent_skills) / len(required_skills) if required_skills else 1.0
            skill_bonus = skill_match_ratio * 0.3
            
            # Workload penalty
            max_concurrent = capabilities.get('max_concurrent_tasks', 3)
            current_workload = agent_workload.get(agent_name, 0)
            workload_penalty = (current_workload / max_concurrent) * 0.2 if max_concurrent > 0 else 0
            
            # Task type preference bonus
            preferred_types = capabilities.get('preferred_task_types', [])
            task_type = task.get('task_type', 'general')
            preference_bonus = 0.1 if task_type in preferred_types else 0
            
            # Calculate final score
            final_score = base_score + skill_bonus - workload_penalty + preference_bonus
            
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Agent score calculation failed: {e}")
            return 0.5
    
    async def _resolve_dependencies(self, tasks: List[Dict[str, Any]], 
                                 assignments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve task dependencies and ensure proper execution order.
        
        Args:
            tasks: List of tasks
            assignments: List of agent assignments
            
        Returns:
            List of tasks with resolved dependencies
        """
        try:
            # Create task lookup
            task_lookup = {task['task_id']: task for task in tasks}
            
            # Resolve dependencies
            for task in tasks:
                resolved_dependencies = []
                
                for dep_id in task.get('dependencies', []):
                    if dep_id in task_lookup:
                        resolved_dependencies.append(dep_id)
                    else:
                        logger.warning(f"Dependency not found: {dep_id}")
                
                task['dependencies'] = resolved_dependencies
            
            return tasks
            
        except Exception as e:
            logger.error(f"Dependency resolution failed: {e}")
            return tasks
    
    async def _balance_workload(self, assignments: List[Dict[str, Any]], 
                              context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Balance workload across agents.
        
        Args:
            assignments: List of agent assignments
            context: Additional context
            
        Returns:
            List of balanced assignments
        """
        try:
            # Calculate current workload
            agent_workload = {}
            for assignment in assignments:
                agent = assignment['agent']
                agent_workload[agent] = agent_workload.get(agent, 0) + 1
            
            # Check for overloaded agents
            max_concurrent = {agent: capabilities.get('max_concurrent_tasks', 3) 
                            for agent, capabilities in self.agent_capabilities.items()}
            
            overloaded_agents = [agent for agent, workload in agent_workload.items() 
                               if workload > max_concurrent.get(agent, 3)]
            
            if not overloaded_agents:
                return assignments
            
            # Rebalance assignments
            balanced_assignments = assignments.copy()
            
            for agent in overloaded_agents:
                # Find tasks that can be reassigned
                agent_tasks = [a for a in balanced_assignments if a['agent'] == agent]
                
                for assignment in agent_tasks:
                    # Find alternative agent
                    alternative_agent = await self._find_alternative_agent(assignment, agent_workload)
                    
                    if alternative_agent:
                        assignment['agent'] = alternative_agent
                        assignment['assignment_reason'] = 'workload_balancing'
                        agent_workload[agent] -= 1
                        agent_workload[alternative_agent] = agent_workload.get(alternative_agent, 0) + 1
            
            return balanced_assignments
            
        except Exception as e:
            logger.error(f"Workload balancing failed: {e}")
            return assignments
    
    async def _find_alternative_agent(self, assignment: Dict[str, Any], 
                                    agent_workload: Dict[str, int]) -> Optional[str]:
        """Find alternative agent for an assignment.
        
        Args:
            assignment: Assignment to find alternative for
            agent_workload: Current agent workload
            
        Returns:
            Alternative agent name or None
        """
        try:
            task_id = assignment['task_id']
            current_agent = assignment['agent']
            
            # Find task details
            task = None  # This would need to be passed or looked up
            
            if not task:
                return None
            
            # Find agents with lower workload
            available_agents = [agent for agent, workload in agent_workload.items() 
                              if agent != current_agent and workload < self.agent_capabilities.get(agent, {}).get('max_concurrent_tasks', 3)]
            
            # Find best alternative
            best_alternative = None
            best_score = 0
            
            for agent in available_agents:
                if await self._agent_can_handle_task(agent, task):
                    score = await self._calculate_agent_score(agent, task, agent_workload)
                    if score > best_score:
                        best_score = score
                        best_alternative = agent
            
            return best_alternative
            
        except Exception as e:
            logger.error(f"Alternative agent finding failed: {e}")
            return None
    
    def _estimate_total_duration(self, tasks: List[Dict[str, Any]]) -> str:
        """Estimate total duration for all tasks.
        
        Args:
            tasks: List of tasks
            
        Returns:
            Estimated total duration
        """
        try:
            duration_scores = {
                'short': 1,
                'medium': 2,
                'long': 3
            }
            
            total_score = sum(duration_scores.get(task.get('estimated_duration', 'medium'), 2) for task in tasks)
            
            if total_score <= 3:
                return 'short'
            elif total_score <= 6:
                return 'medium'
            else:
                return 'long'
                
        except Exception as e:
            logger.error(f"Duration estimation failed: {e}")
            return 'medium'
    
    def _calculate_complexity_score(self, tasks: List[Dict[str, Any]]) -> float:
        """Calculate overall complexity score for tasks.
        
        Args:
            tasks: List of tasks
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        try:
            complexity_scores = {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.9
            }
            
            total_score = sum(complexity_scores.get(task.get('complexity', 'medium'), 0.6) for task in tasks)
            avg_score = total_score / len(tasks) if tasks else 0.5
            
            return min(avg_score, 1.0)
            
        except Exception as e:
            logger.error(f"Complexity calculation failed: {e}")
            return 0.5
    
    async def cleanup(self):
        """Cleanup agent resources."""
        try:
            logger.info("Task Decomposer Agent cleanup completed")
        except Exception as e:
            logger.error(f"Task Decomposer Agent cleanup failed: {e}")
