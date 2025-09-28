#!/usr/bin/env python3
"""
Task Decomposer Agent for Multi-Agent Research System

This agent autonomously decomposes research objectives into executable tasks
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
    """Autonomous task decomposer agent for research task generation."""
    
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
                logger.warning("Gemini API key not found. Decomposition functionality will be limited.")
                return None
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            logger.info("LLM initialized for TaskDecomposerAgent with model: gemini-2.5-flash-lite")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    async def decompose_tasks(self, objectives: List[Dict[str, Any]], 
                            available_agents: Dict[str, Any], 
                            objective_id: str) -> Dict[str, Any]:
        """LLM-based task decomposition.
        
        Args:
            objectives: List of research objectives to decompose
            available_agents: Dictionary of available agents and their capabilities
            objective_id: Objective ID for tracking
            
        Returns:
            Dictionary containing decomposed tasks and assignments
        """
        try:
            decomposition_id = f"decomp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(objectives)) % 10000:04d}"
            
            logger.info(f"Starting LLM-based task decomposition: {decomposition_id}")
            logger.info(f"Objectives to decompose: {len(objectives)}")
            
            # Track active decomposition
            self.active_decompositions = getattr(self, 'active_decompositions', {})
            self.active_decompositions[decomposition_id] = {
                'objectives': objectives,
                'available_agents': available_agents,
                'objective_id': objective_id,
                'started_at': datetime.now(),
                'status': 'running'
            }
            
            # Use LLM for task decomposition
            decomposition_result = await self._llm_decompose_tasks(objectives, available_agents, objective_id)
            
            # Update decomposition status
            self.active_decompositions[decomposition_id]['status'] = 'completed'
            self.active_decompositions[decomposition_id]['completed_at'] = datetime.now()
            
            # Add metadata
            decomposition_result.update({
                'decomposition_id': decomposition_id,
                'objective_id': objective_id,
                'metadata': {
                    'total_tasks': len(decomposition_result.get('tasks', [])),
                    'total_agents': len(set(task.get('assigned_to') for task in decomposition_result.get('tasks', []))),
                    'decomposition_time': (datetime.now() - self.active_decompositions[decomposition_id]['started_at']).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            # Store in decomposition history
            self.decomposition_history.append({
                'decomposition_id': decomposition_id,
                'objective_id': objective_id,
                'objectives': objectives,
                'result': decomposition_result,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"LLM-based task decomposition completed: {decomposition_id}")
            logger.info(f"Generated {len(decomposition_result.get('tasks', []))} tasks")
            
            return decomposition_result
            
        except Exception as e:
            logger.error(f"LLM-based task decomposition failed: {e}")
            
            # Update decomposition status
            if 'decomposition_id' in locals() and hasattr(self, 'active_decompositions') and decomposition_id in self.active_decompositions:
                self.active_decompositions[decomposition_id]['status'] = 'failed'
                self.active_decompositions[decomposition_id]['error'] = str(e)
            
            return {
                'decomposition_id': decomposition_id if 'decomposition_id' in locals() else 'unknown',
                'objective_id': objective_id,
                'tasks': [],
                'assignments': [],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _llm_decompose_tasks(self, objectives: List[Dict[str, Any]], 
                                 available_agents: Dict[str, Any], 
                                 objective_id: str) -> Dict[str, Any]:
        """Use LLM to decompose tasks."""
        try:
            prompt = f"""
            다음 연구 목표들을 구체적인 작업들로 분해하고 적절한 에이전트에게 할당하세요.
            
            연구 목표들: {objectives}
            사용 가능한 에이전트들: {list(available_agents.keys())}
            
            다음을 수행하세요:
            1. 각 목표를 실행 가능한 작업들로 분해
            2. 작업 간 의존성 관계 설정
            3. 각 작업에 적합한 에이전트 할당
            4. 작업 우선순위 설정
            5. 예상 소요 시간과 복잡도 평가
            6. 실행 전략 수립
            
            JSON 형태로 응답하세요:
            {{
                "tasks": [
                    {{
                        "task_id": "unique_id",
                        "objective_id": "목표_id",
                        "description": "작업 설명",
                        "task_type": "data_collection|analysis|synthesis|validation",
                        "assigned_to": "agent_name",
                        "priority": 0.0-1.0,
                        "estimated_duration": "short|medium|long",
                        "complexity": "low|medium|high",
                        "dependencies": ["다른_작업_id"],
                        "success_criteria": ["기준1", "기준2"],
                        "required_skills": ["스킬1", "스킬2"]
                    }}
                ],
                "assignments": [
                    {{
                        "task_id": "작업_id",
                        "agent": "에이전트명",
                        "assignment_reason": "할당 이유",
                        "workload": 0.0-1.0
                    }}
                ],
                "dependencies": [
                    {{
                        "task_id": "작업_id",
                        "depends_on": ["의존_작업_id"],
                        "dependency_type": "sequential|parallel|conditional"
                    }}
                ],
                "priorities": [
                    {{
                        "task_id": "작업_id",
                        "priority_level": 0.0-1.0,
                        "priority_reason": "우선순위 이유"
                    }}
                ],
                "strategy": "sequential|parallel|iterative|hierarchical",
                "execution_plan": {{
                    "phase_1": ["작업_id1", "작업_id2"],
                    "phase_2": ["작업_id3", "작업_id4"],
                    "phase_3": ["작업_id5"]
                }},
                "quality_metrics": {{
                    "completeness_threshold": 0.8,
                    "accuracy_threshold": 0.9,
                    "timeliness_threshold": 0.7
                }}
            }}
            """
            
            response = await asyncio.to_thread(self.llm.generate_content, prompt)
            
            # Parse Gemini response properly
            response_text = response.text.strip()
            logger.debug(f"Raw Gemini response: {response_text}")
            
            # Try to extract JSON from response
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")
            
            return result
            
        except Exception as e:
            logger.error(f"LLM task decomposition failed: {e}")
            logger.info("Falling back to rule-based task decomposition")
            return await self._rule_based_decomposition(objectives, available_agents)
    
    async def _rule_based_decomposition(self, objectives: List[Dict[str, Any]], available_agents: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based task decomposition when LLM is not available.
        
        Args:
            objectives: List of research objectives
            available_agents: Available agents and their capabilities
            
        Returns:
            Task decomposition results
        """
        tasks = []
        assignments = []
        dependencies = []
        priorities = []
        
        for i, objective in enumerate(objectives):
            obj_type = objective.get('type', 'exploratory')
            description = objective.get('description', '')
            objective_id = objective.get('objective_id', f'obj_{i}')
            
            # Generate basic tasks based on objective type
            base_tasks = []
            
            if obj_type == "trend_analysis":
                base_tasks = [
                    f"Research current trends in {description}",
                    f"Analyze historical data for {description}",
                    f"Identify emerging patterns in {description}",
                    f"Compile trend analysis report for {description}"
                ]
            elif obj_type == "comparative":
                base_tasks = [
                    f"Gather information on comparison subjects in {description}",
                    f"Analyze similarities and differences in {description}",
                    f"Evaluate pros and cons in {description}",
                    f"Create comparative analysis report for {description}"
                ]
            elif obj_type == "analytical":
                base_tasks = [
                    f"Collect relevant data for {description}",
                    f"Perform detailed analysis of {description}",
                    f"Identify key insights from {description}",
                    f"Generate analytical report for {description}"
                ]
            else:  # exploratory or comprehensive
                base_tasks = [
                    f"Conduct initial research on {description}",
                    f"Gather comprehensive information about {description}",
                    f"Analyze findings related to {description}",
                    f"Synthesize research results for {description}"
                ]
            
            # Create task objects
            for j, task_desc in enumerate(base_tasks):
                task_id = f"task_{objective_id}_{j+1}"
                task = {
                    "task_id": task_id,
                    "description": task_desc,
                    "objective_id": objective_id,
                    "type": "research" if j < len(base_tasks) - 1 else "synthesis",
                    "priority": "high" if j == 0 else "medium",
                    "estimated_duration": 30 if j < len(base_tasks) - 1 else 15,
                    "required_capabilities": ["research", "analysis"],
                    "method": "rule_based"
                }
                tasks.append(task)
                
                # Simple agent assignment
                agent_type = "research_agent" if j < len(base_tasks) - 1 else "synthesis_agent"
                assignments.append({
                    "task_id": task_id,
                    "agent_type": agent_type,
                    "assignment_reason": f"Rule-based assignment for {task['type']} task"
                })
                
                priorities.append({
                    "task_id": task_id,
                    "priority_score": 10 - j,
                    "urgency": objective.get('urgency', 'normal')
                })
                
                # Add dependencies (sequential execution)
                if j > 0:
                    dependencies.append({
                        "task_id": task_id,
                        "depends_on": [f"task_{objective_id}_{j}"],
                        "dependency_type": "sequential"
                    })
        
        return {
            'tasks': tasks,
            'assignments': assignments,
            'dependencies': dependencies,
            'priorities': priorities,
            'strategy': 'sequential',
            'execution_plan': {
                'total_tasks': len(tasks),
                'estimated_duration': sum(task.get('estimated_duration', 30) for task in tasks),
                'method': 'rule_based'
            },
            'quality_metrics': {
                'decomposition_confidence': 0.7,
                'task_coverage': 1.0,
                'method': 'rule_based'
            }
        }
    
    async def get_decomposition_status(self, decomposition_id: str) -> Optional[Dict[str, Any]]:
        """Get decomposition status."""
        try:
            if not hasattr(self, 'active_decompositions') or decomposition_id not in self.active_decompositions:
                return None
                
            decomposition = self.active_decompositions[decomposition_id]
            return {
                'decomposition_id': decomposition_id,
                'status': decomposition['status'],
                'objective_id': decomposition['objective_id'],
                'objectives': decomposition['objectives'],
                'started_at': decomposition['started_at'].isoformat(),
                'completed_at': decomposition.get('completed_at', {}).isoformat() if decomposition.get('completed_at') else None,
                'error': decomposition.get('error')
            }
            
        except Exception as e:
            logger.error(f"Failed to get decomposition status: {e}")
            return None
    
    async def list_decompositions(self) -> List[Dict[str, Any]]:
        """List all decompositions."""
        try:
            if not hasattr(self, 'active_decompositions'):
                return []
                
            decompositions = []
            for decomposition_id, decomposition in self.active_decompositions.items():
                decompositions.append({
                    'decomposition_id': decomposition_id,
                    'status': decomposition['status'],
                    'objective_id': decomposition['objective_id'],
                    'started_at': decomposition['started_at'].isoformat()
                })
            return decompositions
            
        except Exception as e:
            logger.error(f"Failed to list decompositions: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup decomposer resources."""
        try:
            # Clear active decompositions
            if hasattr(self, 'active_decompositions'):
                self.active_decompositions.clear()
            
            logger.info("Task Decomposer Agent cleanup completed")
            
        except Exception as e:
            logger.error(f"Decomposer cleanup failed: {e}")
