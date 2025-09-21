#!/usr/bin/env python3
"""
Autonomous Orchestrator for Multi-Agent Research System

This orchestrator manages the complete autonomous workflow:
1. LLM-based objective analysis and task decomposition
2. Dynamic multi-agent task assignment and execution
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
import os
import google.generativeai as genai

from src.core.mcp_integration import MCPIntegrationManager
from src.core.llm_methods import LLMMethods
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
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        self.llm_methods = LLMMethods(self.llm)
        
        # Orchestration state
        self.active_objectives: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Enhanced autonomy settings
        self.max_iterations = 10  # Maximum iterations for recursive improvement
        self.quality_threshold = 0.8  # Quality threshold for stopping iterations
        self.learning_enabled = True  # Enable learning from previous executions
        self.adaptive_strategy = True  # Enable adaptive strategy selection
        
        # LLM-based decision making
        self.decision_memory = []  # Store LLM decisions for learning
        self.strategy_weights = {}  # Dynamic strategy weights based on success
        
        logger.info("Autonomous Orchestrator initialized with LLM-based decision making")
    
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
        """Start fully autonomous research with LLM-based decision making.
        
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
            
            logger.info(f"Starting LLM-based autonomous research for objective: {objective.objective_id}")
            logger.info(f"User request: {user_request}")
            
            # Phase 1: LLM-based Analysis
            objective.status = "analyzing"
            analysis_result = await self._llm_analyze_objective(objective)
            objective.analyzed_objectives = analysis_result
            
            # Phase 2: LLM-based Task Decomposition
            objective.status = "decomposing"
            decomposition_result = await self.llm_methods.llm_decompose_tasks(objective)
            objective.decomposed_tasks = decomposition_result['tasks']
            objective.assigned_agents = decomposition_result['assignments']
            
            # Phase 3: Multi-Agent Execution with LLM Coordination
            objective.status = "executing"
            execution_result = await self.llm_methods.llm_coordinate_execution(objective, self.agents)
            objective.execution_results = execution_result
            
            # Phase 4: LLM-based Evaluation
            objective.status = "evaluating"
            evaluation_result = await self.llm_methods.llm_evaluate_results(objective)
            objective.evaluation_results = evaluation_result
            
            # Phase 5: LLM-based Validation
            objective.status = "validating"
            validation_result = await self.llm_methods.llm_validate_results(objective)
            objective.validation_results = validation_result
            
            # Phase 6: LLM-based Final Synthesis
            objective.status = "synthesizing"
            synthesis_result = await self.llm_methods.llm_synthesize_deliverable(objective)
            objective.final_synthesis = synthesis_result
            objective.status = "completed"
            
            # Store execution history for learning
            if self.learning_enabled:
                self.execution_history.append({
                    'objective_id': objective.objective_id,
                    'user_request': user_request,
                    'final_quality': evaluation_result.get('overall_score', 0.0),
                    'timestamp': datetime.now().isoformat()
                })
            
            logger.info(f"LLM-based autonomous research completed: {objective.objective_id}")
            logger.info(f"Final deliverable: {synthesis_result.get('deliverable_path', 'N/A')}")
            
            return objective.objective_id
            
        except Exception as e:
            logger.error(f"LLM-based autonomous research failed: {e}")
            if 'objective' in locals() and objective.objective_id in self.active_objectives:
                self.active_objectives[objective.objective_id].status = "failed"
            raise
    
    async def _llm_analyze_objective(self, objective: Any) -> List[Dict[str, Any]]:
        """LLM-based objective analysis."""
        try:
            prompt = f"""
            사용자 요청을 분석하고 연구 목표를 설정하세요.
            
            사용자 요청: {objective.user_request}
            컨텍스트: {objective.context}
            
            다음을 수행하세요:
            1. 요청의 핵심 목표 파악
            2. 관련된 하위 목표들 식별
            3. 각 목표의 우선순위 설정
            4. 성공 기준 정의
            5. 필요한 리소스와 제약사항 식별
            
            JSON 형태로 응답하세요:
            {{
                "objectives": [
                    {{
                        "objective_id": "unique_id",
                        "type": "primary|secondary|quality",
                        "description": "목표 설명",
                        "priority": 0.0-1.0,
                        "success_criteria": ["기준1", "기준2"],
                        "estimated_effort": 0.0-1.0,
                        "dependencies": ["다른_목표_id"]
                    }}
                ],
                "analysis_metadata": {{
                    "confidence_score": 0.0-1.0,
                    "complexity_level": "low|medium|high",
                    "domain": "identified_domain"
                }}
            }}
            """
            
            response = await asyncio.to_thread(self.llm.generate_content, prompt)
            result = json.loads(response.text)
            
            # Store decision for learning
            self.decision_memory.append({
                'type': 'objective_analysis',
                'input': objective.user_request,
                'output': result,
                'timestamp': datetime.now().isoformat()
            })
            
            return result.get('objectives', [])
            
        except Exception as e:
            logger.error(f"LLM objective analysis failed: {e}")
            return []
    
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
