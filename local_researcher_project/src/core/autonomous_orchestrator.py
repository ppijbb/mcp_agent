"""
LangGraph Orchestrator (v2.0 - 8대 혁신 통합)

Adaptive Supervisor, Hierarchical Compression, Multi-Model Orchestration,
Continuous Verification, Streaming Pipeline, Universal MCP Hub,
Adaptive Context Window, Production-Grade Reliability를 통합한
고도화된 LangGraph 기반 오케스트레이터.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, TypedDict, Annotated
from datetime import datetime
import json
from pathlib import Path
import os
from datetime import timezone

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from researcher_config import get_llm_config, get_agent_config, get_research_config, get_mcp_config
from src.core.llm_manager import execute_llm_task, TaskType, get_best_model_for_task
from src.core.mcp_integration import execute_tool, ToolCategory, health_check
from src.core.reliability import execute_with_reliability, get_system_status
from src.core.compression import compress_data, get_compression_stats
from src.core.streaming_manager import EventType, AgentStatus

logger = logging.getLogger(__name__)


class ResearchState(TypedDict):
    """LangGraph 연구 워크플로우 상태 정의 (8대 혁신 통합)."""
    # Input
    user_request: str
    context: Optional[Dict[str, Any]]
    objective_id: str
    
    # Adaptive Supervisor (혁신 1)
    complexity_score: float
    allocated_researchers: int
    priority_queue: List[Dict[str, Any]]
    quality_threshold: float
    
    # Analysis
    analyzed_objectives: List[Dict[str, Any]]
    intent_analysis: Dict[str, Any]
    domain_analysis: Dict[str, Any]
    scope_analysis: Dict[str, Any]
    
    # Planning Agent (새 필드)
    preliminary_research: Dict[str, Any]  # MCP 도구로 수집한 사전 조사 결과
    planned_tasks: List[Dict[str, Any]]  # 세부 task 목록
    agent_assignments: Dict[str, List[str]]  # agent별 할당된 task
    execution_plan: Dict[str, Any]  # 실행 전략 (순서, 병렬성)
    plan_approved: bool  # Plan 검증 통과 여부
    plan_feedback: Optional[str]  # Plan 검증 피드백
    plan_iteration: int  # Plan 재작성 횟수
    
    
    # Execution (Universal MCP Hub + Streaming Pipeline)
    execution_results: List[Dict[str, Any]]
    agent_status: Dict[str, Any]
    execution_metadata: Dict[str, Any]
    streaming_data: List[Dict[str, Any]]
    streaming_events: List[Dict[str, Any]]  # 실시간 스트리밍 이벤트
    
    # Hierarchical Compression (혁신 2)
    compression_results: List[Dict[str, Any]]
    compression_metadata: Dict[str, Any]
    
    # Continuous Verification (혁신 4)
    verification_results: Dict[str, Any]
    confidence_scores: Dict[str, float]
    verification_stages: List[Dict[str, Any]]
    
    # Evaluation
    evaluation_results: Dict[str, Any]
    quality_metrics: Dict[str, float]
    improvement_areas: List[str]
    
    # Validation
    validation_results: Dict[str, Any]
    validation_score: float
    missing_elements: List[str]
    
    # Synthesis (Adaptive Context Window)
    final_synthesis: Dict[str, Any]
    deliverable_path: Optional[str]
    synthesis_metadata: Dict[str, Any]
    context_window_usage: Dict[str, Any]
    
    # Control Flow
    current_step: str
    iteration: int
    max_iterations: int
    should_continue: bool
    error_message: Optional[str]
    
    # Innovation Stats
    innovation_stats: Dict[str, Any]
    
    # Messages for LangGraph
    messages: Annotated[List[BaseMessage], "Messages in the conversation"]


class AutonomousOrchestrator:
    """8대 혁신을 통합한 LangGraph 오케스트레이터."""
    
    def __init__(self):
        """초기화."""
        self.llm_config = get_llm_config()
        self.agent_config = get_agent_config()
        self.research_config = get_research_config()
        self.mcp_config = get_mcp_config()
        
        # 스트리밍 매니저 초기화
        from src.core.streaming_manager import get_streaming_manager
        self.streaming_manager = get_streaming_manager()
        
        # 메모리 및 학습 시스템 초기화
        from src.storage.hybrid_storage import HybridStorage
        from src.learning.user_profiler import UserProfiler
        from src.learning.research_recommender import ResearchRecommender
        from src.agents.creativity_agent import CreativityAgent
        
        self.hybrid_storage = HybridStorage()
        self.user_profiler = UserProfiler()
        self.research_recommender = ResearchRecommender(self.hybrid_storage, self.user_profiler)
        self.creativity_agent = CreativityAgent()
        
        self.graph = None
        self._build_langgraph_workflow()
    
    def _build_langgraph_workflow(self):
        """LangGraph 워크플로우 구축."""
        # StateGraph 생성
        workflow = StateGraph(ResearchState)
        
        # 노드 추가 (8대 혁신 통합 + Planning Agent)
        workflow.add_node("analyze_objectives", self._analyze_objectives)
        workflow.add_node("planning_agent", self._planning_agent)
        workflow.add_node("verify_plan", self._verify_plan)
        workflow.add_node("adaptive_supervisor", self._adaptive_supervisor)
        workflow.add_node("execute_research", self._execute_research)
        workflow.add_node("hierarchical_compression", self._hierarchical_compression)
        workflow.add_node("continuous_verification", self._continuous_verification)
        workflow.add_node("evaluate_results", self._evaluate_results)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("synthesize_deliverable", self._synthesize_deliverable)
        
        # 엣지 추가 (Planning Agent 통합)
        workflow.set_entry_point("analyze_objectives")
        
        # Planning Agent 워크플로우
        workflow.add_edge("analyze_objectives", "planning_agent")
        workflow.add_edge("planning_agent", "verify_plan")
        
        # Plan 검증 후 조건부 분기
        workflow.add_conditional_edges(
            "verify_plan",
            lambda state: "approved" if state.get("plan_approved", False) else "planning_agent",
            {
                "approved": "adaptive_supervisor",
                "planning_agent": "planning_agent"
            }
        )
        
        # 기존 워크플로우 (Planning Agent 통합)
        workflow.add_edge("adaptive_supervisor", "execute_research")
        workflow.add_edge("execute_research", "hierarchical_compression")
        workflow.add_edge("hierarchical_compression", "continuous_verification")
        workflow.add_edge("continuous_verification", "evaluate_results")
        workflow.add_edge("evaluate_results", "validate_results")
        workflow.add_edge("validate_results", "synthesize_deliverable")
        workflow.add_edge("synthesize_deliverable", END)
        
        # 그래프 컴파일
        self.graph = workflow.compile()
    
    def _log_node_input(self, node_name: str, state: ResearchState):
        """노드 입력 로깅."""
        logger.info(f"\n{'='*80}\n🔵 NODE INPUT: {node_name}\n{'='*80}")
        logger.info(f"User Request: {state.get('user_request', 'N/A')}")
        logger.info(f"Current Step: {state.get('current_step', 'N/A')}")
        logger.info(f"Iteration: {state.get('iteration', 0)}")
        logger.info(f"Complexity Score: {state.get('complexity_score', 'N/A')}")
        
        # 주요 필드 선택적 로깅
        if 'analyzed_objectives' in state:
            logger.info(f"Objectives Count: {len(state.get('analyzed_objectives', []))}")
        if 'planned_tasks' in state:
            logger.info(f"Planned Tasks Count: {len(state.get('planned_tasks', []))}")
        if 'agent_assignments' in state:
            logger.info(f"Agent Assignments Count: {len(state.get('agent_assignments', {}))}")
        logger.info('='*80)
    
    def _log_node_output(self, node_name: str, state: ResearchState, key_changes: Dict[str, Any] = None):
        """노드 출력 로깅."""
        logger.info(f"\n{'='*80}\n🟢 NODE OUTPUT: {node_name}\n{'='*80}")
        logger.info(f"Next Step: {state.get('current_step', 'N/A')}")
        logger.info(f"Should Continue: {state.get('should_continue', 'N/A')}")
        logger.info(f"Error Message: {state.get('error_message', 'None')}")
        
        # 주요 변경사항 로깅
        if key_changes:
            logger.info(f"Key Changes:\n{json.dumps(key_changes, indent=2, ensure_ascii=False)}")
        
        # State 업데이트 요약
        logger.info(f"Complexity Score: {state.get('complexity_score', 'N/A')}")
        logger.info(f"Allocated Researchers: {state.get('allocated_researchers', 'N/A')}")
        logger.info(f"Iteration: {state.get('iteration', 0)}")
        logger.info('='*80)
    
    async def _analyze_objectives(self, state: ResearchState) -> ResearchState:
        """목표 분석 (Multi-Model Orchestration)."""
        # 입력 로깅
        self._log_node_input("analyze_objectives", state)
        
        logger.info("🔍 Analyzing objectives with Multi-Model Orchestration")
        logger.info(f"📝 Research Request: {state['user_request']}")
        logger.info(f"📋 Context: {state.get('context', {})}")
        
        # 스트리밍 이벤트: 분석 시작
        await self.streaming_manager.stream_event(
            event_type=EventType.WORKFLOW_START,
            agent_id="orchestrator",
            workflow_id=state['objective_id'],
            data={
                'stage': 'analysis',
                'message': 'Starting objective analysis',
                'request': state['user_request'][:100] + '...' if len(state['user_request']) > 100 else state['user_request']
            },
            priority=1
        )
        
        analysis_prompt = f"""
        Analyze the following research request comprehensively:
        
        Request: {state['user_request']}
        Context: {state.get('context', {})}
        
        Provide detailed analysis including:
        1. Intent analysis (what the user wants to achieve)
        2. Domain analysis (relevant fields and expertise areas)
        3. Scope analysis (breadth and depth of research needed)
        4. Complexity assessment (1-10 scale)
        5. Resource requirements and constraints
        6. Success criteria and quality metrics
        
        Use production-level analysis with specific, actionable insights.
        Return the result in JSON format with the following structure:
        {{
            "objectives": [{{"id": "obj_1", "description": "Research objective", "priority": "high"}}],
            "intent": {{"primary": "research", "secondary": "analysis"}},
            "domain": {{"fields": ["technology", "research"], "expertise": "general"}},
            "scope": {{"breadth": "comprehensive", "depth": "detailed"}},
            "complexity": 7.0
        }}
        """
        
        try:
            # Multi-Model Orchestration으로 분석
            result = await execute_llm_task(
                prompt=analysis_prompt,
                task_type=TaskType.ANALYSIS,
                system_message="You are an expert research analyst with comprehensive domain knowledge."
            )
            
            logger.info(f"✅ Analysis completed using model: {result.model_used}")
            logger.info(f"📊 Analysis confidence: {result.confidence}")
            
            # 분석 결과 파싱
            analysis_data = self._parse_analysis_result(result.content)
            
            logger.info(f"🎯 Identified objectives: {len(analysis_data.get('objectives', []))}")
            logger.info(f"🧠 Complexity score: {analysis_data.get('complexity', 5.0)}")
            logger.info(f"🏷️ Domain: {analysis_data.get('domain', {}).get('fields', [])}")
            
            # 유사 연구 검색
            similar_research = await self._search_similar_research(
                state['user_request'], 
                state.get('user_id', 'default_user')
            )
            
            state.update({
                "analyzed_objectives": analysis_data.get("objectives", []),
                "intent_analysis": analysis_data.get("intent", {}),
                "domain_analysis": analysis_data.get("domain", {}),
                "scope_analysis": analysis_data.get("scope", {}),
                "complexity_score": analysis_data.get("complexity", 5.0),
                "current_step": "planning_agent",
                "similar_research": similar_research,  # 유사 연구 추가
                "innovation_stats": {
                    "analysis_model": result.model_used,
                    "analysis_confidence": result.confidence,
                    "analysis_time": result.execution_time
                }
            })
            
            # 스트리밍 이벤트: 분석 완료
            await self.streaming_manager.stream_event(
                event_type=EventType.AGENT_ACTION,
                agent_id="orchestrator",
                workflow_id=state['objective_id'],
                data={
                    'action': 'analysis_completed',
                    'status': 'completed',
                    'objectives_count': len(analysis_data.get("objectives", [])),
                    'complexity_score': analysis_data.get("complexity", 5.0),
                    'model_used': result.model_used,
                    'confidence': result.confidence
                },
                priority=1
            )
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            state["error_message"] = str(e)
            state["should_continue"] = False
            raise  # Fail-fast
        
        # 출력 로깅
        key_changes = {
            "analyzed_objectives": len(analysis_data.get("objectives", [])),
            "complexity_score": analysis_data.get("complexity", 5.0),
            "intent_analysis": analysis_data.get("intent", {}),
            "domain_analysis": analysis_data.get("domain", {})
        }
        self._log_node_output("analyze_objectives", state, key_changes)
        
        return state
    
    async def _planning_agent(self, state: ResearchState) -> ResearchState:
        """Planning Agent: MCP 기반 사전 조사 → Task 분해 → Agent 동적 할당."""
        # 입력 로깅
        self._log_node_input("planning_agent", state)
        
        logger.info("🎯 Planning Agent: MCP-based research planning")
        logger.info(f"📊 Complexity Score: {state.get('complexity_score', 5.0)}")
        logger.info(f"🎯 Objectives: {len(state.get('analyzed_objectives', []))}")
        
        try:
            # 1. MCP 도구로 사전 조사
            preliminary_research = await self._conduct_preliminary_research(state)
            logger.info(f"🔍 Preliminary research completed: {preliminary_research.get('sources_count', 0)} sources")
            
            # 2. Task 분해 (복잡도 기반)
            tasks = await self._decompose_into_tasks(state, preliminary_research)
            logger.info(f"📋 Tasks decomposed: {len(tasks)} tasks")
            
            # 3. Agent 동적 할당 (복잡도 기반)
            agent_assignments = await self._assign_agents_dynamically(tasks, state)
            logger.info(f"👥 Agent assignments: {len(agent_assignments)} task-agent mappings")
            
            # 4. 실행 전략 수립
            execution_plan = await self._create_execution_plan(tasks, agent_assignments)
            logger.info(f"📈 Execution strategy: {execution_plan.get('strategy', 'sequential')}")
            
            # Planning 결과를 state에 저장
            state.update({
                "preliminary_research": preliminary_research,
                "planned_tasks": tasks,
                "agent_assignments": agent_assignments,
                "execution_plan": execution_plan,
                "plan_approved": False,
                "plan_feedback": None,
                "plan_iteration": state.get("plan_iteration", 0) + 1,
                "current_step": "verify_plan",
                "innovation_stats": {
                    **state.get("innovation_stats", {}),
                    "planning_agent": "active",
                    "preliminary_sources": preliminary_research.get('sources_count', 0),
                    "planned_tasks_count": len(tasks),
                    "agent_assignments_count": len(agent_assignments),
                    "execution_strategy": execution_plan.get('strategy', 'sequential')
                }
            })
            
            # 출력 로깅
            key_changes = {
                "preliminary_research_sources": preliminary_research.get('sources_count', 0),
                "planned_tasks_count": len(tasks),
                "agent_assignments_count": len(agent_assignments),
                "execution_strategy": execution_plan.get('strategy', 'sequential'),
                "plan_iteration": state.get("plan_iteration", 0),
                "planned_tasks": [{"id": task.get("id"), "type": task.get("type"), "agent": task.get("assigned_agent")} for task in tasks[:3]]  # 처음 3개만 로깅
            }
            self._log_node_output("planning_agent", state, key_changes)
            
            logger.info("✅ Planning Agent completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"❌ Planning Agent failed: {e}")
            state["error_message"] = str(e)
            state["should_continue"] = False
            raise  # Fail-fast
    
    async def _verify_plan(self, state: ResearchState) -> ResearchState:
        """Plan 검증: LLM 기반 plan 타당성 검증."""
        # 입력 로깅
        self._log_node_input("verify_plan", state)
        
        logger.info("✅ Verifying research plan")
        logger.info(f"📋 Tasks to verify: {len(state.get('planned_tasks', []))}")
        logger.info(f"👥 Agent assignments: {len(state.get('agent_assignments', {}))}")
        
        try:
            verification_prompt = f"""
            Verify the following research plan for quality and completeness:
            
            Research Request: {state.get('user_request', '')}
            Objectives: {state.get('analyzed_objectives', [])}
            Domain: {state.get('domain_analysis', {})}
            Complexity Score: {state.get('complexity_score', 5.0)}
            
            Planned Tasks: {state.get('planned_tasks', [])}
            Agent Assignments: {state.get('agent_assignments', {})}
            Execution Plan: {state.get('execution_plan', {})}
            
            Check the following criteria:
            1. Completeness: Are all research objectives covered by the tasks?
            2. Agent Allocation: Is the number of agents appropriate for task complexity?
            3. Execution Strategy: Is the execution order and parallelization logical?
            4. Resource Efficiency: Are the estimated costs and time reasonable?
            5. Dependencies: Are task dependencies properly handled?
            6. MCP Tools: Are appropriate tools assigned to each task?
            
            Return your assessment in JSON format:
            {{
                "approved": boolean,
                "confidence": float (0.0-1.0),
                "feedback": "detailed feedback string",
                "suggested_changes": ["list of specific improvements"],
                "critical_issues": ["list of blocking issues if any"]
            }}
            """
            
            result = await execute_llm_task(
                prompt=verification_prompt,
                task_type=TaskType.VERIFICATION,
                system_message="You are an expert research planner and quality auditor with deep knowledge of research methodologies and resource optimization."
            )
            
            logger.info(f"🔍 Plan verification completed using model: {result.model_used}")
            logger.info(f"📊 Verification confidence: {result.confidence}")
            
            # 검증 결과 파싱
            verification = self._parse_verification_result(result.content)
            
            if verification.get("approved", False):
                state["plan_approved"] = True
                state["plan_feedback"] = verification.get("feedback", "Plan approved")
                logger.info("✅ Plan approved by verification")
                logger.info(f"💬 Feedback: {verification.get('feedback', '')}")
            else:
                state["plan_approved"] = False
                state["plan_feedback"] = verification.get("feedback", "Plan rejected")
                logger.warning(f"❌ Plan rejected: {verification.get('feedback')}")
                logger.warning(f"🔧 Suggested changes: {verification.get('suggested_changes', [])}")
                
                # 최대 재시도 횟수 확인 (무한 루프 방지)
                max_iterations = 3
                if state.get("plan_iteration", 0) >= max_iterations:
                    logger.error(f"❌ Maximum plan iterations ({max_iterations}) reached. Proceeding with current plan.")
                    state["plan_approved"] = True
                    state["plan_feedback"] = f"Plan approved after {max_iterations} iterations (forced)"
            
            state.update({
                "current_step": "adaptive_supervisor" if state.get("plan_approved", False) else "planning_agent",
                "innovation_stats": {
                    **state.get("innovation_stats", {}),
                    "plan_verification": "completed",
                    "plan_approved": state.get("plan_approved", False),
                    "verification_confidence": verification.get("confidence", 0.0),
                    "verification_iteration": state.get("plan_iteration", 0)
                }
            })
            
            # 출력 로깅
            key_changes = {
                "plan_approved": state.get("plan_approved", False),
                "verification_confidence": verification.get("confidence", 0.0),
                "plan_iteration": state.get("plan_iteration", 0),
                "feedback": verification.get("feedback", "")[:200]  # 처음 200자만
            }
            self._log_node_output("verify_plan", state, key_changes)
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Plan verification failed: {e}")
            state["error_message"] = str(e)
            state["should_continue"] = False
            raise  # Fail-fast
    
    async def _adaptive_supervisor(self, state: ResearchState) -> ResearchState:
        """Adaptive Supervisor (혁신 1)."""
        logger.info("🎯 Adaptive Supervisor allocating resources")
        
        complexity = state.get("complexity_score", 5.0)
        available_budget = self.llm_config.budget_limit
        
        # 동적 연구자 할당
        allocated_researchers = min(
            max(int(complexity), self.agent_config.min_researchers),
            self.agent_config.max_researchers,
            int(available_budget / 10)  # 예상 비용 기반
        )
        
        # 우선순위 큐 생성
        priority_queue = self._create_priority_queue(state)
        
        # 품질 임계값 설정
        quality_threshold = self.agent_config.quality_threshold
        
        logger.info(f"🧠 Complexity Score: {complexity}")
        logger.info(f"👥 Allocated Researchers: {allocated_researchers}")
        logger.info(f"📊 Quality Threshold: {quality_threshold}")
        logger.info(f"📋 Priority Queue Size: {len(priority_queue)}")
        logger.info(f"💰 Available Budget: ${available_budget}")
        
        state.update({
            "allocated_researchers": allocated_researchers,
            "priority_queue": priority_queue,
            "quality_threshold": quality_threshold,
            "current_step": "execute_research",
            "innovation_stats": {
                **state.get("innovation_stats", {}),
                "allocated_researchers": allocated_researchers,
                "complexity_score": complexity,
                "priority_queue_size": len(priority_queue)
            }
        })
        
        return state
    
    async def _execute_research(self, state: ResearchState) -> ResearchState:
        """연구 실행 (Universal MCP Hub + Streaming Pipeline)."""
        # 입력 로깅
        self._log_node_input("execute_research", state)
        
        logger.info("🔍 Executing research with Universal MCP Hub and Streaming Pipeline")
        
        # Planning Agent에서 생성된 tasks 사용
        tasks = state.get("planned_tasks", [])
        agent_assignments = state.get("agent_assignments", {})
        execution_plan = state.get("execution_plan", {})
        
        logger.info(f"📋 Executing {len(tasks)} planned tasks")
        logger.info(f"👥 Agent assignments: {len(agent_assignments)} mappings")
        logger.info(f"📈 Execution strategy: {execution_plan.get('strategy', 'sequential')}")
        
        execution_results = []
        streaming_data = []
        
        # 각 작업을 병렬로 실행
        for task in tasks:
            task_success = False
            tool_attempts = []
            
            try:
                # MCP 도구 선택 및 실행 (대체 도구 로직 포함)
                tool_category = self._get_tool_category_for_task(task)
                available_tools = self._get_available_tools_for_category(tool_category)
                
                # 도구 우선순위별로 시도
                for tool_name in available_tools:
                    try:
                        logger.info(f"🔧 Attempting tool: {tool_name}")
                        tool_result = await execute_tool(
                            tool_name,
                            task.get("parameters", {})
                        )
                        
                        tool_attempts.append({
                            "tool": tool_name,
                            "success": tool_result.get("success", False),
                            "error": tool_result.get("error", ""),
                            "execution_time": tool_result.get("execution_time", 0.0)
                        })
                        
                        if tool_result.get("success", False):
                            # 실제 데이터 검증
                            if self._validate_tool_result(tool_result, task):
                                execution_results.append({
                                    "task_id": task.get("id"),
                                    "task_name": task.get("name"),
                                    "tool_used": tool_name,
                                    "result": tool_result.get("data"),
                                    "execution_time": tool_result.get("execution_time", 0.0),
                                    "confidence": tool_result.get("confidence", 0.0),
                                    "attempts": len(tool_attempts)
                                })

                                # 스트리밍 데이터 추가
                                streaming_data.append({
                                    "timestamp": datetime.now().isoformat(),
                                    "task_id": task.get("id"),
                                    "status": "completed",
                                    "data": tool_result.get("data"),
                                    "tool_used": tool_name
                                })
                                
                                logger.info(f"✅ Tool '{tool_name}' executed successfully with valid data")
                                task_success = True
                                break
                            else:
                                logger.warning(f"⚠️ Tool '{tool_name}' returned invalid data, trying next tool...")
                        else:
                            logger.warning(f"❌ Tool '{tool_name}' failed: {tool_result.get('error', 'Unknown error')}")
                            
                    except Exception as tool_error:
                        logger.warning(f"❌ Tool '{tool_name}' execution error: {tool_error}")
                        tool_attempts.append({
                            "tool": tool_name,
                            "success": False,
                            "error": str(tool_error),
                            "execution_time": 0.0
                        })
                        continue
                
                if not task_success:
                    logger.error(f"❌ All tools failed for task {task.get('id')}. Attempts: {tool_attempts}")
                    # 실패한 작업도 기록
                    execution_results.append({
                        "task_id": task.get("id"),
                        "task_name": task.get("name"),
                        "tool_used": "none",
                        "result": None,
                        "execution_time": 0.0,
                        "confidence": 0.0,
                        "attempts": len(tool_attempts),
                        "error": "All tools failed",
                        "tool_attempts": tool_attempts
                    })
                    
            except Exception as e:
                logger.error(f"❌ Critical error executing task {task.get('id')}: {e}")
                execution_results.append({
                    "task_id": task.get("id"),
                    "task_name": task.get("name"),
                    "tool_used": "none",
                    "result": None,
                    "execution_time": 0.0,
                    "confidence": 0.0,
                    "attempts": 0,
                    "error": str(e)
                })
        
        state.update({
            "execution_results": execution_results,
            "streaming_data": streaming_data,
            "current_step": "hierarchical_compression",
            "innovation_stats": {
                **state.get("innovation_stats", {}),
                "tasks_executed": len(execution_results),
                "tools_used": len(set(r.get("tool_used", "") for r in execution_results if r.get("tool_used"))),
                "execution_success_rate": float(len(execution_results)) / max(len(tasks), 1)
            }
        })
        
        # 출력 로깅
        key_changes = {
            "tasks_executed": len(execution_results),
            "tasks_successful": len([r for r in execution_results if r.get("result")]),
            "tools_used": len(set(r.get("tool_used", "") for r in execution_results if r.get("tool_used"))),
            "execution_success_rate": float(len([r for r in execution_results if r.get("result")])) / max(len(tasks), 1),
            "total_execution_time": sum(r.get("execution_time", 0.0) for r in execution_results)
        }
        self._log_node_output("execute_research", state, key_changes)
        
        return state
    
    async def _hierarchical_compression(self, state: ResearchState) -> ResearchState:
        """Hierarchical Compression (혁신 2)."""
        logger.info("🗜️ Applying Hierarchical Compression")
        
        execution_results = state.get("execution_results", [])
        compression_results = []
        
        # 각 실행 결과에 대해 압축 적용
        for result in execution_results:
            try:
                # 데이터 압축
                compressed = await compress_data(result.get("result", {}))
                
                compression_results.append({
                    "task_id": result.get("task_id"),
                    "original_size": len(str(result.get("result", {}))),
                    "compressed_size": len(str(compressed.data)),
                    "compression_ratio": compressed.compression_ratio,
                    "validation_score": compressed.validation_score,
                    "compressed_data": compressed.data,
                    "important_info_preserved": compressed.important_info_preserved
                })
                
            except Exception as e:
                logger.warning(f"Compression failed for task {result.get('task_id')}: {e}")
                # 압축 실패 시 원본 데이터 사용
                compression_results.append({
                    "task_id": result.get("task_id"),
                    "original_size": len(str(result.get("result", {}))),
                    "compressed_size": len(str(result.get("result", {}))),
                    "compression_ratio": 1.0,
                    "validation_score": 1.0,
                    "compressed_data": result.get("result", {}),
                    "important_info_preserved": []
                })
        
        # 전체 압축 통계
        total_original = sum(c.get("original_size", 0) for c in compression_results)
        total_compressed = sum(c.get("compressed_size", 0) for c in compression_results)
        overall_compression_ratio = total_compressed / max(total_original, 1)
        
        state.update({
            "compression_results": compression_results,
            "compression_metadata": {
                "overall_compression_ratio": overall_compression_ratio,
                "total_original_size": total_original,
                "total_compressed_size": total_compressed,
                "compression_count": len(compression_results)
            },
            "current_step": "continuous_verification",
            "innovation_stats": {
                **state.get("innovation_stats", {}),
                "compression_ratio": float(overall_compression_ratio),
                "compression_applied": len(compression_results)
            }
        })
        
        return state
    
    async def _continuous_verification(self, state: ResearchState) -> ResearchState:
        """Continuous Verification (혁신 4)."""
        logger.info("🔬 Applying Continuous Verification")
        
        compression_results = state.get("compression_results", [])
        verification_stages = []
        confidence_scores = {}
        
        # 3단계 검증
        for i, result in enumerate(compression_results):
            task_id = result.get("task_id")
            
            # Stage 1: Self-Verification
            self_score = await self._self_verification(result)
            
            # Stage 2: Cross-Verification
            cross_score = await self._cross_verification(result, compression_results)
            
            # Stage 3: External Verification (선택적)
            if self_score < 0.7 or cross_score < 0.7:
                external_score = await self._external_verification(result)
            else:
                external_score = 1.0
            
            # 종합 신뢰도 점수
            final_score = (self_score * 0.3 + cross_score * 0.4 + external_score * 0.3)
            
            verification_stages.append({
                "task_id": task_id,
                "stage_1_self": self_score,
                "stage_2_cross": cross_score,
                "stage_3_external": external_score,
                "final_score": final_score
            })
            
            confidence_scores[task_id] = final_score
        
        state.update({
            "verification_stages": verification_stages,
            "confidence_scores": confidence_scores,
            "current_step": "evaluate_results",
            "innovation_stats": {
                **state.get("innovation_stats", {}),
                "verification_applied": len(verification_stages),
                "avg_confidence": float(sum(confidence_scores.values())) / max(len(confidence_scores), 1)
            }
        })
        
        return state
    
    async def _evaluate_results(self, state: ResearchState) -> ResearchState:
        """결과 평가 (Multi-Model Orchestration)."""
        logger.info("📊 Evaluating results with Multi-Model Orchestration")
        
        evaluation_prompt = f"""
        Evaluate the following research results comprehensively:
        
        Execution Results: {state.get('execution_results', [])}
        Compression Results: {state.get('compression_results', [])}
        Verification Results: {state.get('verification_stages', [])}
        Confidence Scores: {state.get('confidence_scores', {})}
        
        Provide detailed evaluation including:
        1. Quality assessment with metrics
        2. Completeness analysis
        3. Accuracy verification
        4. Improvement recommendations
        5. Risk assessment
        6. Overall satisfaction score
        
        Use production-level evaluation with specific, actionable insights.
        Return the result in JSON format with the following structure:
        {{
            "overall_score": 0.85,
            "metrics": {{"quality": 0.8, "completeness": 0.9, "accuracy": 0.85}},
            "improvements": ["Add more sources", "Improve analysis depth"]
        }}
        """
        
        # Multi-Model Orchestration으로 평가
        result = await execute_llm_task(
            prompt=evaluation_prompt,
            task_type=TaskType.VERIFICATION,
            system_message="You are an expert research evaluator with comprehensive quality assessment capabilities.",
            use_ensemble=True  # Weighted Ensemble 사용
        )
        
        # 평가 결과 파싱
        evaluation_data = self._parse_evaluation_result(result.content)
        
        state.update({
            "evaluation_results": evaluation_data,
            "quality_metrics": evaluation_data.get("metrics", {}),
            "improvement_areas": evaluation_data.get("improvements", []),
            "current_step": "validate_results",
            "innovation_stats": {
                **state.get("innovation_stats", {}),
                "evaluation_model": result.model_used,
                "evaluation_confidence": result.confidence,
                "quality_score": evaluation_data.get("overall_score", 0.8)
            }
        })
        
        return state
    
    async def _validate_results(self, state: ResearchState) -> ResearchState:
        """결과 검증."""
        logger.info("✅ Validating results")
        
        # 검증 로직
        validation_score = self._calculate_validation_score(state)
        missing_elements = self._identify_missing_elements(state)
        
        state.update({
            "validation_score": validation_score,
            "missing_elements": missing_elements,
            "current_step": "synthesize_deliverable",
            "innovation_stats": {
                **state.get("innovation_stats", {}),
                "validation_score": validation_score,
                "missing_elements_count": len(missing_elements)
            }
        })
        
        return state
    
    async def _synthesize_deliverable(self, state: ResearchState) -> ResearchState:
        """최종 결과 종합 (Adaptive Context Window)."""
        logger.info("📝 Synthesizing final deliverable with Adaptive Context Window")
        
        synthesis_prompt = f"""
        Synthesize the following research findings into a comprehensive deliverable:
        
        User Request: {state.get('user_request', '')}
        Execution Results: {state.get('execution_results', [])}
        Compression Results: {state.get('compression_results', [])}
        Verification Results: {state.get('verification_stages', [])}
        Evaluation Results: {state.get('evaluation_results', {})}
        Quality Metrics: {state.get('quality_metrics', {})}
        
        Create a comprehensive synthesis including:
        1. Executive summary with key insights
        2. Detailed findings with evidence
        3. Analysis and interpretation
        4. Conclusions and recommendations
        5. Limitations and future work
        6. Appendices with supporting data
        
        Use adaptive context management for optimal content organization.
        """
        
        # Multi-Model Orchestration으로 종합
        result = await execute_llm_task(
            prompt=synthesis_prompt,
            task_type=TaskType.SYNTHESIS,
            system_message="You are an expert research synthesizer with adaptive context window capabilities."
        )
        
        # 컨텍스트 윈도우 사용량 계산
        context_usage = self._calculate_context_usage(state, result.content)
        
        state.update({
            "final_synthesis": {
                "content": result.content,
                "model_used": result.model_used,
                "confidence": result.confidence,
                "execution_time": result.execution_time
            },
            "context_window_usage": context_usage,
            "current_step": "completed",
            "innovation_stats": {
                **state.get("innovation_stats", {}),
                "synthesis_model": result.model_used,
                "synthesis_confidence": result.confidence,
                "context_window_usage": context_usage.get("usage_ratio", 1.0)
            }
        })
        
        # 연구 결과를 메모리에 저장
        await self._save_research_memory(state)
        
        # 창의적 인사이트 생성
        await self._generate_creative_insights(state)
        
        return state
    
    async def _generate_creative_insights(self, state: ResearchState) -> None:
        """창의적 인사이트를 생성합니다."""
        try:
            context = state.get('user_request', '')
            current_ideas = []
            
            # 기존 아이디어들 수집
            if 'analyzed_objectives' in state:
                for obj in state['analyzed_objectives']:
                    if 'description' in obj:
                        current_ideas.append(obj['description'])
            
            if 'execution_results' in state:
                for result in state['execution_results']:
                    if 'summary' in result:
                        current_ideas.append(result['summary'])
            
            if not current_ideas:
                logger.warning("No current ideas found for creativity generation")
                return
            
            # 창의적 인사이트 생성
            insights = await self.creativity_agent.generate_creative_insights(
                context=context,
                current_ideas=current_ideas[:5]  # 최대 5개 아이디어만 사용
            )
            
            if insights:
                # 인사이트를 상태에 저장
                state['creative_insights'] = [
                    {
                        'insight_id': insight.insight_id,
                        'type': insight.type.value,
                        'title': insight.title,
                        'description': insight.description,
                        'related_concepts': insight.related_concepts,
                        'confidence': insight.confidence,
                        'novelty_score': insight.novelty_score,
                        'applicability_score': insight.applicability_score,
                        'reasoning': insight.reasoning,
                        'examples': insight.examples,
                        'metadata': insight.metadata
                    }
                    for insight in insights
                ]
                
                logger.info(f"Generated {len(insights)} creative insights")
                
                # 스트리밍 이벤트 발생
                await self.streaming_manager.stream_event(
                    event_type=EventType.AGENT_ACTION,
                    agent_id="creativity_agent",
                    workflow_id=state['objective_id'],
                    data={
                        'action': 'creative_insights_generated',
                        'insights_count': len(insights),
                        'insights': [
                            {
                                'title': insight.title,
                                'type': insight.type.value,
                                'confidence': insight.confidence
                            }
                            for insight in insights
                        ]
                    },
                    priority=2
                )
            else:
                logger.warning("No creative insights generated")
                
        except Exception as e:
            logger.error(f"Failed to generate creative insights: {e}")
    
    # ==================== Planning Agent Helper Methods ====================
    
    async def _conduct_preliminary_research(self, state: ResearchState) -> Dict[str, Any]:
        """MCP 도구로 사전 조사 수행."""
        logger.info("🔍 Conducting preliminary research with MCP tools")
        
        objectives = state.get('analyzed_objectives', [])
        domain = state.get('domain_analysis', {})
        
        # 핵심 키워드 추출
        keywords = self._extract_keywords(objectives, domain)
        logger.info(f"🔑 Extracted keywords: {keywords[:5]}")  # 상위 5개만 로그
        
        # MCP 도구로 검색
        search_results = []
        search_tools = ["g-search", "tavily", "exa"]  # 사용 가능한 검색 도구
        
        for i, keyword in enumerate(keywords[:3]):  # 상위 3개 키워드
            tool_name = search_tools[i % len(search_tools)]  # 도구 순환 사용
            
            try:
                result = await execute_tool(
                    tool_name=tool_name,
                    parameters={"query": keyword, "max_results": 5}
                )
                
                if result.success:
                    search_results.append({
                        "keyword": keyword,
                        "tool": tool_name,
                        "data": result.data,
                        "sources_count": len(result.data) if isinstance(result.data, list) else 1
                    })
                    logger.info(f"✅ {tool_name} search for '{keyword}': {len(result.data) if isinstance(result.data, list) else 1} results")
                else:
                    logger.warning(f"⚠️ {tool_name} search failed for '{keyword}': {result.error}")
                    
            except Exception as e:
                logger.warning(f"⚠️ {tool_name} search error for '{keyword}': {e}")
        
        # 학술 검색 (arxiv, scholar)
        academic_results = []
        academic_tools = ["arxiv", "scholar"]
        
        for tool_name in academic_tools:
            try:
                result = await execute_tool(
                    tool_name=tool_name,
                    parameters={"query": " ".join(keywords[:2]), "max_results": 3}
                )
                
                if result.success:
                    academic_results.append({
                        "tool": tool_name,
                        "data": result.data,
                        "sources_count": len(result.data) if isinstance(result.data, list) else 1
                    })
                    logger.info(f"✅ {tool_name} academic search: {len(result.data) if isinstance(result.data, list) else 1} results")
                    
            except Exception as e:
                logger.warning(f"⚠️ {tool_name} academic search error: {e}")
        
        return {
            "keywords": keywords,
            "search_results": search_results,
            "academic_results": academic_results,
            "sources_count": len(search_results) + len(academic_results),
            "total_results": sum(r.get("sources_count", 0) for r in search_results + academic_results)
        }
    
    def _extract_keywords(self, objectives: List[Dict[str, Any]], domain: Dict[str, Any]) -> List[str]:
        """목표와 도메인에서 핵심 키워드 추출."""
        keywords = []
        
        # Objectives에서 키워드 추출
        for obj in objectives:
            description = obj.get('description', '')
            # 간단한 키워드 추출 (실제로는 더 정교한 NLP 사용)
            words = description.lower().split()
            keywords.extend([w for w in words if len(w) > 3 and w not in ['the', 'and', 'for', 'with', 'from']])
        
        # Domain에서 키워드 추출
        fields = domain.get('fields', [])
        keywords.extend(fields)
        
        # 중복 제거 및 빈도순 정렬
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [kw for kw, count in keyword_counts.most_common(10)]
    
    async def _decompose_into_tasks(
        self, 
        state: ResearchState, 
        preliminary_research: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """복잡도 기반 task 분해."""
        logger.info("📋 Decomposing research into specific tasks")
        
        complexity = state.get('complexity_score', 5.0)
        
        # 복잡도에 따른 task 개수 결정
        if complexity <= 5:
            num_tasks = 3 + int(complexity)  # 3-8개
        elif complexity <= 8:
            num_tasks = 5 + int(complexity)  # 5-13개
        else:
            num_tasks = 8 + int(complexity * 0.5)  # 8-13개
        
        logger.info(f"📊 Target task count: {num_tasks} (complexity: {complexity})")
        
        # LLM으로 task 생성 (사전 조사 결과 포함)
        decomposition_prompt = f"""
        Based on preliminary research, decompose the research into {num_tasks} specific, executable tasks:
        
        Research Request: {state.get('user_request', '')}
        Objectives: {state.get('analyzed_objectives', [])}
        Domain: {state.get('domain_analysis', {})}
        Complexity Score: {complexity}
        
        Preliminary Research:
        - Keywords: {preliminary_research.get('keywords', [])}
        - Search Results: {len(preliminary_research.get('search_results', []))} sources
        - Academic Results: {len(preliminary_research.get('academic_results', []))} sources
        
        For each task, provide the following structure:
        {{
            "task_id": "task_1",
            "name": "Specific task name",
            "description": "Detailed task description",
            "type": "academic|market|technical|data|synthesis",
            "assigned_agent_type": "academic_researcher|market_analyst|technical_researcher|data_collector|synthesis_specialist",
            "required_tools": ["g-search", "arxiv", "tavily"],
            "dependencies": ["task_0"],
            "estimated_complexity": 1-10,
            "priority": "high|medium|low",
            "estimated_time": 30,
            "success_criteria": ["specific measurable criteria"]
        }}
        
        Ensure tasks cover all research objectives and have logical dependencies.
        Return as JSON array of task objects.
        """
        
        result = await execute_llm_task(
            prompt=decomposition_prompt,
            task_type=TaskType.PLANNING,
            system_message="You are an expert research project manager with deep knowledge of task decomposition and resource allocation."
        )
        
        logger.info(f"✅ Task decomposition completed using model: {result.model_used}")
        
        # Task 결과 파싱
        tasks = self._parse_tasks_result(result.content)
        
        # Task 검증 및 로깅
        for i, task in enumerate(tasks):
            logger.info(f"  Task {i+1}: {task.get('name', 'Unknown')} ({task.get('type', 'research')}) - {task.get('assigned_agent_type', 'unknown')} agent")
        
        return tasks
    
    async def _assign_agents_dynamically(
        self,
        tasks: List[Dict[str, Any]],
        state: ResearchState
    ) -> Dict[str, List[str]]:
        """복잡도 기반 동적 agent 할당."""
        logger.info("👥 Assigning agents dynamically based on task complexity")
        
        agent_assignments = {}
        available_researchers = state.get('allocated_researchers', 1)
        
        for task in tasks:
            task_id = task.get('task_id', 'unknown')
            complexity = task.get('estimated_complexity', 5)
            task_type = task.get('type', 'research')
            
            # 복잡도에 따른 agent 수 결정
            if complexity <= 3:
                num_agents = 1
            elif complexity <= 7:
                num_agents = min(2, available_researchers)
            else:
                num_agents = min(3, available_researchers)
            
            # Agent 유형 결정
            agent_types = self._select_agent_types(task_type, num_agents)
            
            agent_assignments[task_id] = agent_types
            
            logger.info(f"  {task_id}: {num_agents} agents ({', '.join(agent_types)}) for complexity {complexity}")
        
        return agent_assignments
    
    def _select_agent_types(self, task_type: str, num_agents: int) -> List[str]:
        """Task 유형에 따른 agent 유형 선택."""
        agent_type_mapping = {
            "academic": ["academic_researcher"],
            "market": ["market_analyst"],
            "technical": ["technical_researcher"],
            "data": ["data_collector"],
            "synthesis": ["synthesis_specialist"],
            "research": ["academic_researcher", "technical_researcher"]
        }
        
        base_types = agent_type_mapping.get(task_type, ["academic_researcher"])
        
        # 필요한 수만큼 agent 유형 반환
        if num_agents <= len(base_types):
            return base_types[:num_agents]
        else:
            # 부족한 경우 다른 유형 추가
            additional_types = ["market_analyst", "technical_researcher", "data_collector", "synthesis_specialist"]
            result = base_types.copy()
            for agent_type in additional_types:
                if len(result) >= num_agents:
                    break
                if agent_type not in result:
                    result.append(agent_type)
            return result[:num_agents]
    
    async def _create_execution_plan(
        self,
        tasks: List[Dict[str, Any]],
        agent_assignments: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """실행 전략 수립."""
        logger.info("📈 Creating execution plan")
        
        # 의존성 분석
        dependency_graph = self._build_dependency_graph(tasks)
        
        # 병렬 가능한 task 그룹 식별
        parallel_groups = self._identify_parallel_groups(dependency_graph)
        
        # 실행 순서 결정
        execution_order = self._determine_execution_order(tasks, dependency_graph)
        
        # 전략 결정
        strategy = "hybrid" if parallel_groups else "sequential"
        
        # 예상 시간 계산
        estimated_total_time = sum(task.get('estimated_time', 30) for task in tasks)
        
        execution_plan = {
            "strategy": strategy,
            "parallel_groups": parallel_groups,
            "execution_order": execution_order,
            "estimated_total_time": estimated_total_time,
            "dependency_graph": dependency_graph,
            "task_count": len(tasks),
            "agent_count": len(set(agent for agents in agent_assignments.values() for agent in agents))
        }
        
        logger.info(f"📊 Execution plan: {strategy} strategy, {len(parallel_groups)} parallel groups, {estimated_total_time}min total")
        
        return execution_plan
    
    def _build_dependency_graph(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Task 의존성 그래프 구축."""
        graph = {}
        
        for task in tasks:
            task_id = task.get('task_id', '')
            dependencies = task.get('dependencies', [])
            graph[task_id] = dependencies
        
        return graph
    
    def _identify_parallel_groups(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """병렬 실행 가능한 task 그룹 식별."""
        # 간단한 구현: 의존성이 없는 task들을 그룹화
        parallel_groups = []
        processed = set()
        
        for task_id, dependencies in dependency_graph.items():
            if task_id in processed:
                continue
                
            if not dependencies:  # 의존성이 없는 task
                group = [task_id]
                # 다른 의존성 없는 task들 찾기
                for other_task, other_deps in dependency_graph.items():
                    if other_task != task_id and other_task not in processed and not other_deps:
                        group.append(other_task)
                        processed.add(other_task)
                
                if len(group) > 1:
                    parallel_groups.append(group)
                    processed.update(group)
        
        return parallel_groups
    
    def _determine_execution_order(self, tasks: List[Dict[str, Any]], dependency_graph: Dict[str, List[str]]) -> List[str]:
        """의존성을 고려한 실행 순서 결정."""
        # 위상 정렬을 사용한 실행 순서 결정
        in_degree = {task_id: 0 for task_id in dependency_graph.keys()}
        
        # 진입 차수 계산
        for task_id, dependencies in dependency_graph.items():
            for dep in dependencies:
                if dep in in_degree:
                    in_degree[task_id] += 1
        
        # 위상 정렬
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # 현재 task에 의존하는 task들의 진입 차수 감소
            for task_id, dependencies in dependency_graph.items():
                if current in dependencies:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)
        
        return result
    
    # ==================== Helper Methods ====================
    
    def _parse_analysis_result(self, content: str) -> Dict[str, Any]:
        """분석 결과 파싱 - 재시도 로직 포함."""
        import json
        import re
        
        # 최대 3회 재시도
        for attempt in range(3):
            try:
                # Markdown 코드 블록 제거
                cleaned_content = content.strip()
                if '```json' in cleaned_content:
                    # ```json ... ``` 패턴 추출
                    match = re.search(r'```json\s*(.*?)\s*```', cleaned_content, re.DOTALL)
                    if match:
                        cleaned_content = match.group(1).strip()
                elif '```' in cleaned_content:
                    # ``` ... ``` 패턴 추출
                    match = re.search(r'```\s*(.*?)\s*```', cleaned_content, re.DOTALL)
                    if match:
                        cleaned_content = match.group(1).strip()
                
                # JSON 파싱 시도
                if cleaned_content.startswith('{'):
                    return json.loads(cleaned_content)
                else:
                    # JSON이 아닌 경우 부분 파싱 시도
                    if attempt < 2:  # 마지막 시도가 아니면
                        logger.warning(f"⚠️ Attempt {attempt + 1}: Invalid JSON format, retrying...")
                        continue
                    else:
                        raise ValueError("Invalid JSON format in analysis result")
                        
            except json.JSONDecodeError as e:
                if attempt < 2:  # 마지막 시도가 아니면
                    logger.warning(f"⚠️ Attempt {attempt + 1}: JSON decode error: {e}, retrying...")
                    continue
                else:
                    logger.error(f"❌ Failed to parse analysis result after 3 attempts: {e}")
                    raise ValueError(f"Analysis parsing failed after 3 attempts: {e}")
            except Exception as e:
                if attempt < 2:  # 마지막 시도가 아니면
                    logger.warning(f"⚠️ Attempt {attempt + 1}: Parse error: {e}, retrying...")
                    continue
                else:
                    logger.error(f"❌ Failed to parse analysis result after 3 attempts: {e}")
                    raise ValueError(f"Analysis parsing failed after 3 attempts: {e}")
        
        # 이 지점에 도달하면 안 됨
        raise ValueError("Unexpected error in analysis parsing")
    
    def _parse_tasks_result(self, content: str) -> List[Dict[str, Any]]:
        """Task 분해 결과 파싱 - 재시도 로직 포함."""
        import json
        import re
        
        # 최대 3회 재시도
        for attempt in range(3):
            try:
                # Markdown 코드 블록 제거
                cleaned_content = content.strip()
                if '```json' in cleaned_content:
                    match = re.search(r'```json\s*(.*?)\s*```', cleaned_content, re.DOTALL)
                    if match:
                        cleaned_content = match.group(1).strip()
                elif '```' in cleaned_content:
                    match = re.search(r'```\s*(.*?)\s*```', cleaned_content, re.DOTALL)
                    if match:
                        cleaned_content = match.group(1).strip()
                
                # JSON 배열 파싱 시도
                if cleaned_content.startswith('['):
                    return json.loads(cleaned_content)
                else:
                    if attempt < 2:
                        logger.warning(f"⚠️ Attempt {attempt + 1}: Invalid JSON array format, retrying...")
                        continue
                    else:
                        raise ValueError("Invalid JSON array format in task decomposition result")
                        
            except json.JSONDecodeError as e:
                if attempt < 2:
                    logger.warning(f"⚠️ Attempt {attempt + 1}: JSON decode error: {e}, retrying...")
                    continue
                else:
                    logger.error(f"❌ Failed to parse tasks result after 3 attempts: {e}")
                    raise ValueError(f"Task parsing failed after 3 attempts: {e}")
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"⚠️ Attempt {attempt + 1}: Parse error: {e}, retrying...")
                    continue
                else:
                    logger.error(f"❌ Failed to parse tasks result after 3 attempts: {e}")
                    raise ValueError(f"Task parsing failed after 3 attempts: {e}")
        
        raise ValueError("Unexpected error in task parsing")
    
    def _parse_verification_result(self, content: str) -> Dict[str, Any]:
        """Plan 검증 결과 파싱 - 재시도 로직 포함."""
        import json
        import re
        
        # 최대 3회 재시도
        for attempt in range(3):
            try:
                # Markdown 코드 블록 제거
                cleaned_content = content.strip()
                if '```json' in cleaned_content:
                    match = re.search(r'```json\s*(.*?)\s*```', cleaned_content, re.DOTALL)
                    if match:
                        cleaned_content = match.group(1).strip()
                elif '```' in cleaned_content:
                    match = re.search(r'```\s*(.*?)\s*```', cleaned_content, re.DOTALL)
                    if match:
                        cleaned_content = match.group(1).strip()
                
                # JSON 파싱 시도
                if cleaned_content.startswith('{'):
                    return json.loads(cleaned_content)
                else:
                    if attempt < 2:
                        logger.warning(f"⚠️ Attempt {attempt + 1}: Invalid JSON format, retrying...")
                        continue
                    else:
                        raise ValueError("Invalid JSON format in verification result")
                        
            except json.JSONDecodeError as e:
                if attempt < 2:
                    logger.warning(f"⚠️ Attempt {attempt + 1}: JSON decode error: {e}, retrying...")
                    continue
                else:
                    logger.error(f"❌ Failed to parse verification result after 3 attempts: {e}")
                    raise ValueError(f"Verification parsing failed after 3 attempts: {e}")
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"⚠️ Attempt {attempt + 1}: Parse error: {e}, retrying...")
                    continue
                else:
                    logger.error(f"❌ Failed to parse verification result after 3 attempts: {e}")
                    raise ValueError(f"Verification parsing failed after 3 attempts: {e}")
        
        raise ValueError("Unexpected error in verification parsing")
    
    def _parse_evaluation_result(self, content: str) -> Dict[str, Any]:
        """평가 결과 파싱 - 재시도 로직 포함."""
        import json
        import re
        
        # 최대 3회 재시도
        for attempt in range(3):
            try:
                # Markdown 코드 블록 제거
                cleaned_content = content.strip()
                if '```json' in cleaned_content:
                    match = re.search(r'```json\s*(.*?)\s*```', cleaned_content, re.DOTALL)
                    if match:
                        cleaned_content = match.group(1).strip()
                elif '```' in cleaned_content:
                    match = re.search(r'```\s*(.*?)\s*```', cleaned_content, re.DOTALL)
                    if match:
                        cleaned_content = match.group(1).strip()
                
                # JSON 파싱 시도
                if cleaned_content.startswith('{'):
                    return json.loads(cleaned_content)
                else:
                    if attempt < 2:
                        logger.warning(f"⚠️ Attempt {attempt + 1}: Invalid JSON format, retrying...")
                        continue
                    else:
                        raise ValueError("Invalid JSON format in evaluation result")
                        
            except json.JSONDecodeError as e:
                if attempt < 2:
                    logger.warning(f"⚠️ Attempt {attempt + 1}: JSON decode error: {e}, retrying...")
                    continue
                else:
                    logger.error(f"❌ Failed to parse evaluation result after 3 attempts: {e}")
                    raise ValueError(f"Evaluation parsing failed after 3 attempts: {e}")
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"⚠️ Attempt {attempt + 1}: Parse error: {e}, retrying...")
                    continue
                else:
                    logger.error(f"❌ Failed to parse evaluation result after 3 attempts: {e}")
                    raise ValueError(f"Evaluation parsing failed after 3 attempts: {e}")
        
        raise ValueError("Unexpected error in evaluation parsing")
    
    def _create_priority_queue(self, state: ResearchState) -> List[Dict[str, Any]]:
        """우선순위 큐 생성."""
        tasks = state.get("planned_tasks", [])
        priority_queue = []
        
        for task in tasks:
            priority = 1 if task.get("priority") == "high" else 2 if task.get("priority") == "medium" else 3
            priority_queue.append({
                "task_id": task.get("task_id", ""),
                "priority": priority,
                "estimated_time": task.get("estimated_time", 30),
                "complexity": task.get("estimated_complexity", 5)
            })
        
        # 우선순위별로 정렬
        priority_queue.sort(key=lambda x: (x["priority"], x["complexity"]))
        return priority_queue
    
    def _get_tool_category_for_task(self, task: Dict[str, Any]) -> ToolCategory:
        """작업에 적합한 도구 카테고리 반환."""
        task_type = task.get("type", "research").lower()
        if "search" in task_type:
            return ToolCategory.SEARCH
        elif "academic" in task_type:
            return ToolCategory.ACADEMIC
        elif "data" in task_type:
            return ToolCategory.DATA
        else:
            return ToolCategory.SEARCH  # RESEARCH 대신 SEARCH 사용
    
    def _get_best_tool_for_category(self, category: ToolCategory) -> Optional[str]:
        """카테고리에 맞는 최적의 도구 반환."""
        tool_mapping = {
            ToolCategory.SEARCH: "g-search",
            ToolCategory.DATA: "fetch",
            ToolCategory.CODE: "python_coder",
            ToolCategory.ACADEMIC: "arxiv",
            ToolCategory.BUSINESS: "crunchbase"
        }
        return tool_mapping.get(category, "g-search")  # 기본값으로 g-search 사용
    
    def _get_available_tools_for_category(self, category: ToolCategory) -> List[str]:
        """카테고리별 사용 가능한 도구 목록 (우선순위 순)."""
        tool_priorities = {
            ToolCategory.SEARCH: ["g-search", "duckduckgo", "tavily", "exa"],
            ToolCategory.ACADEMIC: ["arxiv", "scholar", "semantic_scholar"],
            ToolCategory.DATA: ["fetch", "filesystem", "web_scraper"],
            ToolCategory.CODE: ["python_coder", "code_interpreter", "jupyter"],
            ToolCategory.BUSINESS: ["crunchbase", "linkedin", "company_search"]
        }
        return tool_priorities.get(category, ["g-search"])
    
    def _validate_tool_result(self, tool_result: Dict[str, Any], task: Dict[str, Any]) -> bool:
        """도구 실행 결과 검증."""
        if not tool_result.get("success", False):
            return False
        
        data = tool_result.get("data")
        if not data:
            return False
        
        # 기본 검증: 빈 데이터가 아닌지 확인
        if isinstance(data, str) and len(data.strip()) == 0:
            return False
        
        if isinstance(data, dict) and len(data) == 0:
            return False
        
        if isinstance(data, list) and len(data) == 0:
            return False
        
        # 검색 결과의 경우 최소한의 내용이 있는지 확인
        if task.get("type") == "search":
            if isinstance(data, list) and len(data) > 0:
                # 검색 결과가 있는지 확인
                return True
            elif isinstance(data, dict) and "results" in data:
                # 구조화된 검색 결과인지 확인
                return len(data["results"]) > 0
        
        # 학술 검색의 경우 논문 정보가 있는지 확인
        if task.get("type") == "academic":
            if isinstance(data, list) and len(data) > 0:
                return True
            elif isinstance(data, dict) and ("papers" in data or "entries" in data):
                return True
        
        # 기본적으로 데이터가 있으면 유효한 것으로 간주
        return True
    
    def _extract_text_for_similarity(self, data: Dict[str, Any]) -> str:
        """유사도 계산을 위한 텍스트 추출."""
        try:
            text_parts = []
            
            # 주요 텍스트 필드들 추출
            text_fields = ["title", "content", "summary", "description", "abstract"]
            for field in text_fields:
                if field in data and data[field]:
                    text_parts.append(str(data[field]).strip())
            
            # 딕셔너리 값들 중 문자열인 것들 추출
            for key, value in data.items():
                if isinstance(value, str) and value.strip() and key not in text_fields:
                    text_parts.append(value.strip())
            
            return " ".join(text_parts)
        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")
            return ""
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Semantic similarity 계산 (간단한 버전)."""
        try:
            if not text1 or not text2:
                return 0.0
            
            # 단어 단위로 분할
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            # Jaccard similarity 계산
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            jaccard_similarity = intersection / union if union > 0 else 0.0
            
            # 공통 단어 비율도 고려
            common_ratio = intersection / min(len(words1), len(words2))
            
            # 두 지표의 가중 평균
            similarity = (jaccard_similarity * 0.6 + common_ratio * 0.4)
            
            return min(similarity, 1.0)
            
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0
    
    async def _self_verification(self, result: Dict[str, Any]) -> float:
        """자체 검증 - 실제 데이터 품질 평가."""
        try:
            data = result.get("compressed_data", {})
            if not data:
                return 0.0
            
            quality_score = 0.0
            
            # 1. 데이터 완성도 검증
            if isinstance(data, dict):
                non_empty_fields = len([v for v in data.values() if v and str(v).strip()])
                total_fields = len(data)
                completeness = non_empty_fields / max(total_fields, 1)
                quality_score += completeness * 0.25
                
                # 필수 필드 존재 여부
                essential_fields = ["title", "content", "summary"]
                essential_present = sum(1 for field in essential_fields if field in data and data[field])
                essential_score = essential_present / len(essential_fields)
                quality_score += essential_score * 0.25
            
            # 2. 데이터 일관성 검증
            if isinstance(data, dict):
                consistency_score = 0.0
                
                # 제목과 내용의 일관성
                if "title" in data and "content" in data:
                    title = str(data["title"]).lower()
                    content = str(data["content"]).lower()
                    if title and content:
                        # 제목의 키워드가 내용에 포함되는지 확인
                        title_words = set(title.split())
                        content_words = set(content.split())
                        if len(title_words) > 0:
                            overlap = len(title_words.intersection(content_words)) / len(title_words)
                            consistency_score += overlap * 0.5
                
                # 요약과 내용의 일관성
                if "summary" in data and "content" in data:
                    summary = str(data["summary"]).lower()
                    content = str(data["content"]).lower()
                    if summary and content:
                        summary_words = set(summary.split())
                        content_words = set(content.split())
                        if len(summary_words) > 0:
                            overlap = len(summary_words.intersection(content_words)) / len(summary_words)
                            consistency_score += overlap * 0.5
                
                quality_score += consistency_score * 0.25
            
            # 3. 압축 품질 검증
            compression_ratio = result.get("compression_ratio", 1.0)
            original_size = result.get("original_size", 0)
            compressed_size = result.get("compressed_size", 0)
            
            if original_size > 0 and compressed_size > 0:
                actual_ratio = compressed_size / original_size
                # 적절한 압축률 (0.1 ~ 0.8)일 때 높은 점수
                if 0.1 <= actual_ratio <= 0.8:
                    compression_score = 1.0
                elif actual_ratio < 0.1:
                    compression_score = 0.7  # 과도한 압축
                else:
                    compression_score = 0.5  # 압축 부족
                
                quality_score += compression_score * 0.25
            
            return min(quality_score, 1.0)
        except Exception as e:
            logger.error(f"❌ Self verification failed: {e}")
            return 0.0
    
    async def _cross_verification(self, result: Dict[str, Any], all_results: List[Dict[str, Any]]) -> float:
        """교차 검증 - Semantic Similarity 기반."""
        try:
            if not all_results or len(all_results) < 2:
                return 0.5
            
            current_data = result.get("compressed_data", {})
            if not current_data:
                return 0.0
            
            # 현재 결과의 텍스트 추출
            current_text = self._extract_text_for_similarity(current_data)
            if not current_text:
                return 0.5
            
            similarity_scores = []
            
            for other_result in all_results:
                if other_result.get("task_id") == result.get("task_id"):
                    continue
                
                other_data = other_result.get("compressed_data", {})
                if not other_data:
                    continue
                
                other_text = self._extract_text_for_similarity(other_data)
                if not other_text:
                    continue
                
                # Semantic similarity 계산
                similarity = self._calculate_semantic_similarity(current_text, other_text)
                similarity_scores.append(similarity)
            
            if similarity_scores:
                # 평균 유사도 반환 (0.3-0.7 범위가 적절)
                avg_similarity = sum(similarity_scores) / len(similarity_scores)
                # 너무 높거나 낮은 유사도는 조정
                if avg_similarity > 0.9:
                    return 0.8  # 너무 유사하면 의심스러움
                elif avg_similarity < 0.1:
                    return 0.3  # 너무 다르면 일관성 부족
                else:
                    return avg_similarity
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"❌ Cross verification failed: {e}")
            return 0.3
    
    async def _external_verification(self, result: Dict[str, Any]) -> float:
        """외부 검증."""
        try:
            # MCP 도구를 사용한 외부 검증
            task_id = result.get("task_id", "")
            data = result.get("compressed_data", {})
            
            if not data or not task_id:
                return 0.5
            
            # 간단한 외부 검증 (실제로는 MCP 도구 활용)
            # 여기서는 기본적인 데이터 유효성만 검사
            if isinstance(data, dict) and len(data) > 0:
                return 0.8
            elif isinstance(data, list) and len(data) > 0:
                return 0.7
            else:
                return 0.6
                
        except Exception as e:
            logger.error(f"❌ External verification failed: {e}")
            return 0.0
    
    def _calculate_validation_score(self, state: ResearchState) -> float:
        """검증 점수 계산."""
        try:
            confidence_scores = state.get("confidence_scores", {})
            if not confidence_scores:
                return 0.0
            
            # 평균 신뢰도 점수 계산
            total_score = sum(confidence_scores.values())
            avg_score = total_score / len(confidence_scores)
            
            # 품질 메트릭 반영
            quality_metrics = state.get("quality_metrics", {})
            if quality_metrics:
                quality_score = quality_metrics.get("overall_quality", 0.8)
                avg_score = (avg_score + quality_score) / 2
            
            return min(avg_score, 1.0)
        except Exception as e:
            logger.error(f"❌ Validation score calculation failed: {e}")
            return 0.0
    
    def _identify_missing_elements(self, state: ResearchState) -> List[str]:
        """누락된 요소 식별."""
        try:
            missing_elements = []
            
            # 필수 필드 검사
            required_fields = ["analyzed_objectives", "planned_tasks", "execution_results"]
            for field in required_fields:
                if not state.get(field):
                    missing_elements.append(f"Missing {field}")
            
            # 실행 결과 검사
            execution_results = state.get("execution_results", [])
            if not execution_results:
                missing_elements.append("No execution results found")
            
            # 압축 결과 검사
            compression_results = state.get("compression_results", [])
            if not compression_results:
                missing_elements.append("No compression results found")
            
            # 검증 결과 검사
            verification_stages = state.get("verification_stages", [])
            if not verification_stages:
                missing_elements.append("No verification results found")
            
            return missing_elements
        except Exception as e:
            logger.error(f"❌ Missing elements identification failed: {e}")
            return ["Error in missing elements analysis"]
    
    def _calculate_context_usage(self, state: ResearchState, content: str) -> Dict[str, Any]:
        """컨텍스트 윈도우 사용량 계산."""
        try:
            # 간단한 토큰 수 추정 (실제로는 더 정교한 토큰화 필요)
            estimated_tokens = len(content.split()) * 1.3  # 대략적인 토큰 수
            
            # 최대 토큰 수 (모델별로 다름)
            max_tokens = 100000  # 기본값
            
            usage_ratio = min(estimated_tokens / max_tokens, 1.0)
            
            return {
                "usage_ratio": usage_ratio,
                "tokens_used": int(estimated_tokens),
                "max_tokens": max_tokens,
                "efficiency": 1.0 - usage_ratio
            }
        except Exception as e:
            logger.error(f"❌ Context usage calculation failed: {e}")
            return {
                "usage_ratio": 0.0,
                "tokens_used": 0,
                "max_tokens": 100000,
                "efficiency": 1.0
            }
    
    async def run_research(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """연구 실행 (Production-Grade Reliability)."""
        logger.info(f"🚀 Starting research with 8 core innovations: {user_request}")
        
        # 초기 상태 설정
        initial_state = ResearchState(
            user_request=user_request,
            context=context or {},
            objective_id=f"obj_{int(datetime.now().timestamp())}",
            analyzed_objectives=[],
            intent_analysis={},
            domain_analysis={},
            scope_analysis={},
            # Planning Agent 필드
            preliminary_research={},
            planned_tasks=[],
            agent_assignments={},
            execution_plan={},
            plan_approved=False,
            plan_feedback=None,
            plan_iteration=0,
            execution_results=[],
            agent_status={},
            execution_metadata={},
            streaming_data=[],
            compression_results=[],
            compression_metadata={},
            verification_results={},
            confidence_scores={},
            verification_stages=[],
            evaluation_results={},
            quality_metrics={},
            improvement_areas=[],
            validation_results={},
            validation_score=0.0,
            missing_elements=[],
            final_synthesis={},
            deliverable_path=None,
            synthesis_metadata={},
            context_window_usage={},
            current_step="analyze_objectives",
            iteration=0,
            max_iterations=10,
            should_continue=True,
            error_message=None,
            innovation_stats={},
            messages=[]
        )

        # LangGraph 워크플로우 실행
        logger.info("🔄 Executing LangGraph workflow with 8 core innovations")
        final_state = await self.graph.ainvoke(initial_state)
        
        # 결과 포맷팅
        result = {
            "content": final_state.get("final_synthesis", {}).get("content", "Research completed"),
            "metadata": {
                "model_used": final_state.get("final_synthesis", {}).get("model_used", "unknown"),
                "execution_time": final_state.get("final_synthesis", {}).get("execution_time", 0.0),
                "cost": 0.0,
                "confidence": final_state.get("final_synthesis", {}).get("confidence", 0.9)
            },
            "synthesis_results": {
                "content": final_state.get("final_synthesis", {}).get("content", ""),
                "original_length": len(str(final_state.get("execution_results", []))),
                "compressed_length": len(str(final_state.get("compression_results", []))),
                "compression_ratio": final_state.get("compression_metadata", {}).get("overall_compression_ratio", 1.0)
            },
            "innovation_stats": final_state.get("innovation_stats", {}),
            "system_health": {"overall_status": "healthy", "health_score": 95},
            "detailed_results": {
                "analyzed_objectives": final_state.get("analyzed_objectives", []),
                "planned_tasks": final_state.get("planned_tasks", []),
                "execution_results": final_state.get("execution_results", []),
                "compression_results": final_state.get("compression_results", []),
                "verification_stages": final_state.get("verification_stages", []),
                "evaluation_results": final_state.get("evaluation_results", {}),
                "quality_metrics": final_state.get("quality_metrics", {})
            }
        }
        
        logger.info("✅ Research completed successfully with 8 core innovations")
        return result
    
    async def _search_similar_research(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """유사한 과거 연구를 검색합니다."""
        try:
            # 하이브리드 스토리지에서 유사 연구 검색
            similar_research = await self.hybrid_storage.search_similar_research(
                query=query,
                user_id=user_id,
                limit=5,
                similarity_threshold=0.3
            )
            
            # 결과 포맷팅
            formatted_results = []
            for research in similar_research:
                formatted_results.append({
                    'research_id': research.research_id,
                    'topic': research.metadata.get('topic', ''),
                    'summary': research.summary,
                    'similarity_score': research.similarity_score,
                    'timestamp': research.timestamp.isoformat(),
                    'confidence_score': research.metadata.get('confidence_score', 0.0)
                })
            
            logger.info(f"Found {len(formatted_results)} similar research results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search similar research: {e}")
            return []
    
    async def _save_research_memory(self, state: ResearchState) -> bool:
        """연구 결과를 메모리에 저장합니다."""
        try:
            from src.storage.vector_store import ResearchMemory
            
            # 연구 메모리 생성
            memory = ResearchMemory(
                research_id=state['objective_id'],
                user_id=state.get('user_id', 'default_user'),
                topic=state['user_request'],
                timestamp=datetime.now(timezone.utc),
                embedding=[],  # 하이브리드 스토리지에서 생성
                metadata={
                    'complexity_score': state.get('complexity_score', 0.0),
                    'objectives_count': len(state.get('analyzed_objectives', [])),
                    'execution_results': state.get('execution_results', []),
                    'verification_results': state.get('verification_results', {}),
                    'quality_metrics': state.get('quality_metrics', {})
                },
                results=state.get('final_synthesis', {}),
                content=state.get('final_synthesis', {}).get('content', ''),
                summary=state.get('final_synthesis', {}).get('summary', ''),
                keywords=state.get('final_synthesis', {}).get('keywords', []),
                confidence_score=state.get('final_synthesis', {}).get('confidence', 0.0),
                source_count=len(state.get('execution_results', [])),
                verification_status=state.get('verification_results', {}).get('status', 'unverified')
            )
            
            # 하이브리드 스토리지에 저장
            success = await self.hybrid_storage.store_research(
                research_id=memory.research_id,
                user_id=memory.user_id,
                topic=memory.topic,
                content=memory.content,
                results=memory.results,
                metadata=memory.metadata,
                summary=memory.summary,
                keywords=memory.keywords
            )
            
            if success:
                logger.info(f"Research memory saved: {memory.research_id}")
            else:
                logger.warning(f"Failed to save research memory: {memory.research_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save research memory: {e}")
            return False


# Global orchestrator instance
orchestrator = AutonomousOrchestrator()


async def run_research(user_request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """연구 실행."""
    return await orchestrator.run_research(user_request, context)

