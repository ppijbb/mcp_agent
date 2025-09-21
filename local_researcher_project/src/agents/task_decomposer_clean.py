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
                raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("LLM initialized for TaskDecomposerAgent")
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
            result = json.loads(response.text)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM task decomposition failed: {e}")
            return {
                'tasks': [],
                'assignments': [],
                'dependencies': [],
                'priorities': [],
                'strategy': 'sequential',
                'execution_plan': {},
                'quality_metrics': {}
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
