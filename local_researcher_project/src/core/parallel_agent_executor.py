#!/usr/bin/env python3
"""
Parallel Agent Executor for Local Researcher Project

병렬 agent 실행 관리 시스템.
여러 agent를 동시에 실행하고, 작업을 분할하여 병렬 처리하며, 결과를 수집 통합합니다.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from collections import defaultdict

from src.core.task_queue import TaskQueue, TaskQueueItem
from src.core.agent_pool import AgentPool
from src.core.researcher_config import get_agent_config, get_research_config, get_mcp_config
from src.core.mcp_integration import execute_tool
from src.core.streaming_manager import get_streaming_manager, EventType
from src.core.agent_result_sharing import SharedResultsManager, AgentDiscussionManager
from src.core.task_validator import TaskValidator, ValidationResult
from src.core.result_cache import get_result_cache
from src.core.error_handler import get_error_handler
from src.core.concurrency_manager import get_concurrency_manager

logger = logging.getLogger(__name__)


class ParallelAgentExecutor:
    """병렬 agent 실행 관리자."""
    
    def __init__(self):
        """초기화."""
        self.agent_config = get_agent_config()
        self.research_config = get_research_config()
        self.mcp_config = get_mcp_config()
        self.concurrency_manager = get_concurrency_manager()
        # Use dynamic concurrency if available, otherwise use config value
        self.max_concurrent = self.concurrency_manager.get_current_concurrency() or self.agent_config.max_concurrent_research_units
        
        # 컴포넌트 초기화
        self.task_queue = TaskQueue()
        self.agent_pool = AgentPool(max_pool_size=self.max_concurrent * 2)
        self.streaming_manager = get_streaming_manager()
        self.task_validator = TaskValidator()
        self.result_cache = get_result_cache()
        self.error_handler = get_error_handler()
        
        # 실행 상태
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_results: List[Dict[str, Any]] = []
        self.failed_tasks: List[Dict[str, Any]] = []
        
        # 결과 공유 및 토론 (objective_id가 설정되면 초기화)
        self.shared_results_manager: Optional[SharedResultsManager] = None
        self.discussion_manager: Optional[AgentDiscussionManager] = None
        
        logger.info(f"ParallelAgentExecutor initialized with max_concurrent={self.max_concurrent}")
        
        # Start concurrency monitoring
        asyncio.create_task(self.concurrency_manager.start_monitoring())
    
    async def execute_parallel_tasks(
        self,
        tasks: List[Dict[str, Any]],
        agent_assignments: Dict[str, List[str]],
        execution_plan: Dict[str, Any],
        objective_id: str
    ) -> Dict[str, Any]:
        """병렬 작업 실행."""
        logger.info(f"Starting parallel execution of {len(tasks)} tasks")
        
        # 결과 공유 및 토론 시스템 초기화
        if self.agent_config.enable_agent_communication:
            self.shared_results_manager = SharedResultsManager(objective_id=objective_id)
            self.discussion_manager = AgentDiscussionManager(
                objective_id=objective_id,
                shared_results_manager=self.shared_results_manager
            )
            logger.info("✅ Agent result sharing and discussion enabled")
        
        # 작업 큐 초기화
        self.task_queue.add_tasks(tasks)
        
        # 스트리밍 이벤트: 실행 시작
        await self.streaming_manager.stream_event(
            event_type=EventType.WORKFLOW_START,
            agent_id="parallel_executor",
            workflow_id=objective_id,
            data={
                'stage': 'parallel_execution',
                'message': 'Starting parallel task execution',
                'total_tasks': len(tasks),
                'max_concurrent': self.max_concurrent,
                'strategy': execution_plan.get('strategy', 'sequential')
            },
            priority=1
        )
        
        # 병렬 그룹 정보 활용
        parallel_groups = execution_plan.get('parallel_groups', [])
        if parallel_groups:
            logger.info(f"Using {len(parallel_groups)} parallel groups from execution plan")
            self.task_queue.parallel_groups = parallel_groups
        
        # 병렬 실행 시작
        execution_start = datetime.now()
        results = await self._execute_with_parallel_groups(
            agent_assignments,
            execution_plan,
            objective_id
        )
        execution_time = (datetime.now() - execution_start).total_seconds()
        
        # Record performance for concurrency optimization
        tasks_completed = len([r for r in results if r.get("status") == "completed"])
        self.concurrency_manager.record_performance(tasks_completed, execution_time)
        
        # 결과 통합
        final_results = await self._collect_results(results, objective_id)
        
        # 스트리밍 이벤트: 실행 완료
        await self.streaming_manager.stream_event(
            event_type=EventType.WORKFLOW_COMPLETE,
            agent_id="parallel_executor",
            workflow_id=objective_id,
            data={
                'stage': 'parallel_execution',
                'message': 'Parallel task execution completed',
                'total_tasks': len(tasks),
                'completed_tasks': len(final_results.get('execution_results', [])),
                'failed_tasks': len(self.failed_tasks),
                'execution_time': execution_time,
                'success_rate': len(final_results.get('execution_results', [])) / max(len(tasks), 1)
            },
            priority=1
        )
        
        logger.info(f"Parallel execution completed: {len(final_results.get('execution_results', []))} tasks completed in {execution_time:.2f}s")
        
        return final_results
    
    async def _execute_with_parallel_groups(
        self,
        agent_assignments: Dict[str, List[str]],
        execution_plan: Dict[str, Any],
        objective_id: str
    ) -> List[Dict[str, Any]]:
        """병렬 그룹 기반 실행."""
        results = []
        
        # Get dynamic concurrency
        current_concurrency = self.concurrency_manager.get_current_concurrency()
        semaphore = asyncio.Semaphore(current_concurrency)
        
        while self.task_queue.has_pending_tasks():
            # 다음 작업 그룹 가져오기 (dynamic concurrency)
            current_concurrency = self.concurrency_manager.get_current_concurrency()
            task_group = self.task_queue.get_next_task_group(max_group_size=current_concurrency)
            
            if not task_group:
                # 더 이상 실행 가능한 작업이 없으면 대기
                await asyncio.sleep(0.1)
                continue
            
            logger.info(f"Executing task group with {len(task_group)} tasks")
            
            # 그룹 내 작업들을 병렬로 실행
            group_tasks = []
            for task_id in task_group:
                task = self.task_queue.get_task(task_id)
                if task:
                    group_tasks.append(
                        self._execute_single_task(
                            task_id,
                            task,
                            agent_assignments,
                            semaphore,
                            objective_id
                        )
                    )
            
            # 그룹 실행 완료 대기
            group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
            
            # 결과 처리
            for i, result in enumerate(group_results):
                task_id = task_group[i]
                if isinstance(result, Exception):
                    logger.error(f"Task {task_id} failed with exception: {result}")
                    self.failed_tasks.append({
                        'task_id': task_id,
                        'error': str(result)
                    })
                    self.task_queue.mark_completed(task_id)
                else:
                    results.append(result)
                    self.task_queue.mark_completed(task_id)
        
        return results
    
    async def _execute_single_task(
        self,
        task_id: str,
        task: Dict[str, Any],
        agent_assignments: Dict[str, List[str]],
        semaphore: asyncio.Semaphore,
        objective_id: str
    ) -> Dict[str, Any]:
        """단일 작업 실행."""
        # Track active tasks for concurrency manager
        self.concurrency_manager.increment_active_tasks()
        
        try:
            async with semaphore:
                task_start = datetime.now()
                
                # 스트리밍 이벤트: 작업 시작
                await self.streaming_manager.stream_event(
                    event_type=EventType.AGENT_ACTION,
                    agent_id="parallel_executor",
                    workflow_id=objective_id,
                    data={
                        'action': 'task_started',
                        'task_id': task_id,
                        'task_name': task.get('name', ''),
                        'status': 'running'
                    },
                    priority=0
                )
                
                try:
                    # 사전 검증: 작업 실행 전 검증
                    tool_category = self._get_tool_category_for_task(task)
                    available_tools = self._get_available_tools_for_category(tool_category)
                    
                    pre_validation = await self.task_validator.validate_task_before_execution(
                        task=task,
                        task_id=task_id,
                        task_queue=self.task_queue,
                        available_tools=available_tools
                    )
                    
                    if not pre_validation.is_valid:
                        logger.warning(f"Task {task_id} failed pre-execution validation: {pre_validation.errors}")
                        # 검증 실패해도 계속 진행 (경고만)
                        if pre_validation.confidence < 0.5:
                            # 신뢰도가 너무 낮으면 실패 처리
                            raise ValueError(f"Task validation failed: {', '.join(pre_validation.errors)}")
                    
                    if pre_validation.warnings:
                        logger.debug(f"Task {task_id} pre-execution validation warnings: {pre_validation.warnings}")
                    
                    tool_result = None
                    tool_attempts = []
                    
                    # 도구 우선순위별로 시도
                    for tool_name in available_tools:
                        try:
                            logger.debug(f"Task {task_id}: Attempting tool {tool_name}")
                            
                            # 파라미터 생성
                            tool_parameters = self._generate_tool_parameters(task, tool_name)
                            
                            # 캐시 확인
                            cached_result = await self.result_cache.get(
                                tool_name=tool_name,
                                parameters=tool_parameters,
                                task_id=task_id,
                                check_similarity=True
                            )
                            
                            if cached_result:
                                logger.info(f"Task {task_id}: Cache hit for tool {tool_name}")
                                tool_result = cached_result
                            else:
                                # 도구 실행
                                tool_result = await execute_tool(
                                    tool_name,
                                    tool_parameters
                                )
                                
                                # 성공한 결과만 캐시에 저장
                                if tool_result.get("success", False):
                                    # TTL 결정: 검색 도구는 1시간, 다른 도구는 30분
                                    ttl = 3600 if 'search' in tool_name.lower() else 1800
                                    await self.result_cache.set(
                                        tool_name=tool_name,
                                        parameters=tool_parameters,
                                        value=tool_result,
                                        ttl=ttl,
                                        task_id=task_id
                                    )
                                    logger.debug(f"Task {task_id}: Cached result for tool {tool_name}")
                            
                            tool_attempts.append({
                                "tool": tool_name,
                                "success": tool_result.get("success", False),
                                "error": tool_result.get("error", ""),
                                "execution_time": tool_result.get("execution_time", 0.0)
                            })
                            
                            # 실행 중 검증
                            execution_time_so_far = (datetime.now() - task_start).total_seconds()
                            during_validation = await self.task_validator.validate_task_during_execution(
                                task_id=task_id,
                                intermediate_result=tool_result,
                                task=task,
                                execution_time=execution_time_so_far
                            )
                            
                            if during_validation.warnings:
                                logger.debug(f"Task {task_id} during-execution validation warnings: {during_validation.warnings}")
                            
                            # 결과 검증 (강화된 버전)
                            result_validation = await self.task_validator.validate_task_result(
                                tool_result=tool_result,
                                task=task
                            )
                            
                            # 성공 조건: 기본 성공 + 검증 통과
                            is_success = (
                                tool_result.get("success", False) and
                                result_validation.is_valid and
                                result_validation.confidence >= 0.5
                            )
                            
                            if is_success:
                                logger.info(
                                    f"Task {task_id}: Tool {tool_name} executed successfully "
                                    f"(confidence: {result_validation.confidence:.2f})"
                                )
                                break
                            else:
                                if not tool_result.get("success", False):
                                    logger.warning(f"Task {task_id}: Tool {tool_name} execution failed")
                                elif not result_validation.is_valid:
                                    logger.warning(
                                        f"Task {task_id}: Tool {tool_name} result validation failed: "
                                        f"{result_validation.errors}"
                                    )
                                elif result_validation.confidence < 0.5:
                                    logger.warning(
                                        f"Task {task_id}: Tool {tool_name} result confidence too low: "
                                        f"{result_validation.confidence:.2f}"
                                    )
                                tool_result = None
                            
                        except Exception as tool_error:
                            logger.warning(f"Task {task_id}: Tool {tool_name} error: {tool_error}")
                            
                            # Try error handler recovery
                            try:
                                recovery_result, recovery_success = await self.error_handler.handle_error(
                                    tool_error,
                                    execute_tool,
                                    tool_name,
                                    tool_parameters
                                )
                                
                                if recovery_success and recovery_result:
                                    logger.info(f"Task {task_id}: Tool {tool_name} recovered from error")
                                    tool_result = recovery_result
                                    
                                    tool_attempts.append({
                                        "tool": tool_name,
                                        "success": True,
                                        "error": "",
                                        "execution_time": recovery_result.get("execution_time", 0.0),
                                        "recovered": True
                                    })
                                    
                                    # Continue with recovered result
                                    execution_time_so_far = (datetime.now() - task_start).total_seconds()
                                    during_validation = await self.task_validator.validate_task_during_execution(
                                        task_id=task_id,
                                        intermediate_result=tool_result,
                                        task=task,
                                        execution_time=execution_time_so_far
                                    )
                                    
                                    result_validation = await self.task_validator.validate_task_result(
                                        tool_result=tool_result,
                                        task=task
                                    )
                                    
                                    is_success = (
                                        tool_result.get("success", False) and
                                        result_validation.is_valid and
                                        result_validation.confidence >= 0.5
                                    )
                                    
                                    if is_success:
                                        logger.info(f"Task {task_id}: Tool {tool_name} recovered and validated successfully")
                                        break
                            except Exception as recovery_error:
                                logger.debug(f"Error recovery failed: {recovery_error}")
                            
                            # If recovery failed or not attempted, record failure
                            tool_attempts.append({
                                "tool": tool_name,
                                "success": False,
                                "error": str(tool_error),
                                "execution_time": 0.0
                            })
                            continue
                    
                    execution_time = (datetime.now() - task_start).total_seconds()
                    
                    # Agent ID 생성 (작업별 고유 agent)
                    agent_id = f"agent_{task_id}"
                    
                    # 결과 생성
                    if tool_result and tool_result.get("success", False):
                        result_data = tool_result.get("data")
                        confidence = tool_result.get("confidence", 0.0)
                        
                        result = {
                        "task_id": task_id,
                        "task_name": task.get("name", ""),
                        "agent_id": agent_id,
                        "tool_used": tool_attempts[-1]["tool"] if tool_attempts else "none",
                        "result": result_data,
                        "execution_time": execution_time,
                        "confidence": confidence,
                        "attempts": len(tool_attempts),
                        "status": "completed"
                        }
                        
                        # 결과 공유 (agent communication이 활성화된 경우)
                        if self.shared_results_manager:
                            result_id = await self.shared_results_manager.share_result(
                                task_id=task_id,
                                agent_id=agent_id,
                                result=result_data,
                                metadata={
                                    "tool_used": result["tool_used"],
                                    "execution_time": execution_time
                                },
                                confidence=confidence
                            )
                            result["shared_result_id"] = result_id
                            
                            # 다른 agent들의 결과 가져오기 (동일한 작업에 대한)
                            other_results = await self.shared_results_manager.get_shared_results(
                                task_id=task_id,
                                exclude_agent_id=agent_id
                            )
                            
                            # 다른 agent들과 토론
                            if other_results and self.discussion_manager:
                                discussion = await self.discussion_manager.agent_discuss_result(
                                    result_id=result_id,
                                    agent_id=agent_id,
                                    other_agent_results=other_results
                                )
                                if discussion:
                                    result["discussion"] = discussion
                                    logger.info(f"Agent {agent_id} discussed result with {len(other_results)} other agents")
                    else:
                        result = {
                            "task_id": task_id,
                            "task_name": task.get("name", ""),
                            "agent_id": agent_id,
                            "tool_used": "none",
                            "result": None,
                            "execution_time": execution_time,
                            "confidence": 0.0,
                            "attempts": len(tool_attempts),
                            "error": "All tools failed",
                            "tool_attempts": tool_attempts,
                            "status": "failed"
                        }
                    
                    # 스트리밍 이벤트: 작업 완료
                    await self.streaming_manager.stream_event(
                        event_type=EventType.AGENT_ACTION,
                        agent_id="parallel_executor",
                        workflow_id=objective_id,
                        data={
                            'action': 'task_completed',
                            'task_id': task_id,
                            'status': result.get("status", "unknown"),
                            'execution_time': execution_time
                        },
                        priority=0
                    )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Task {task_id} execution failed: {e}")
                    execution_time = (datetime.now() - task_start).total_seconds()
                    
                    return {
                        "task_id": task_id,
                        "task_name": task.get("name", ""),
                        "tool_used": "none",
                        "result": None,
                        "execution_time": execution_time,
                        "confidence": 0.0,
                        "attempts": 0,
                        "error": str(e),
                        "status": "failed"
                    }
        finally:
            # Always decrement active tasks
            self.concurrency_manager.decrement_active_tasks()
    
    async def _collect_results(
        self,
        results: List[Dict[str, Any]],
        objective_id: str
    ) -> Dict[str, Any]:
        """결과 수집 및 통합."""
        # 진행 상황 업데이트
        progress = self.task_queue.get_progress()
        
        # 결과 공유 요약
        sharing_summary = None
        discussion_summary = None
        if self.shared_results_manager:
            sharing_summary = await self.shared_results_manager.get_result_summary()
            logger.info(f"✅ Result sharing summary: {sharing_summary['total_results']} results shared by {sharing_summary['agents_count']} agents")
        
        if self.discussion_manager:
            discussion_summary = await self.discussion_manager.get_discussion_summary()
            logger.info(f"✅ Discussion summary: {discussion_summary['total_topics']} discussion topics")
        
        # 스트리밍 이벤트: 진행 상황
        await self.streaming_manager.stream_event(
            event_type=EventType.PROGRESS_UPDATE,
            agent_id="parallel_executor",
            workflow_id=objective_id,
            data={
                'progress': progress,
                'completed_results': len(results),
                'failed_tasks': len(self.failed_tasks),
                'result_sharing': sharing_summary,
                'discussions': discussion_summary
            },
            priority=0
        )
        
        return {
            "execution_results": results,
            "failed_tasks": self.failed_tasks,
            "progress": progress,
            "total_execution_time": sum(r.get("execution_time", 0.0) for r in results),
            "success_count": len([r for r in results if r.get("status") == "completed"]),
            "failure_count": len([r for r in results if r.get("status") == "failed"]) + len(self.failed_tasks),
            "result_sharing": sharing_summary,
            "discussions": discussion_summary
        }
    
    def _get_tool_category_for_task(self, task: Dict[str, Any]) -> str:
        """작업에 적합한 도구 카테고리 결정."""
        task_type = task.get('task_type', 'general').lower()
        
        if 'search' in task_type or 'find' in task_type:
            return 'search'
        elif 'data' in task_type or 'analyze' in task_type:
            return 'data'
        elif 'code' in task_type or 'implement' in task_type:
            return 'code'
        elif 'academic' in task_type or 'paper' in task_type:
            return 'academic'
        elif 'business' in task_type or 'market' in task_type:
            return 'business'
        else:
            return 'utility'
    
    def _get_available_tools_for_category(self, category: str) -> List[str]:
        """카테고리별 사용 가능한 도구 목록."""
        from src.core.mcp_integration import UniversalMCPHub
        
        # 설정에서 도구 목록 가져오기
        config_tools = []
        if category == 'search':
            config_tools = self.mcp_config.search_tools
        elif category == 'data':
            config_tools = self.mcp_config.data_tools
        elif category == 'code':
            config_tools = self.mcp_config.code_tools
        elif category == 'academic':
            config_tools = self.mcp_config.academic_tools
        elif category == 'business':
            config_tools = self.mcp_config.business_tools
        
        # MCP Hub에서 실제 사용 가능한 도구 확인
        available_tools = []
        try:
            mcp_hub = UniversalMCPHub()
            if hasattr(mcp_hub, 'mcp_tools_map') and mcp_hub.mcp_tools_map:
                # 설정에서 지정된 도구가 실제로 사용 가능한지 확인
                for tool_name in config_tools:
                    # server_name::tool_name 형식인지 확인
                    if "::" in tool_name:
                        server_name, actual_tool_name = tool_name.split("::", 1)
                        if server_name in mcp_hub.mcp_tools_map:
                            if actual_tool_name in mcp_hub.mcp_tools_map[server_name]:
                                available_tools.append(tool_name)
                    else:
                        # tool_name만 있는 경우, 모든 서버에서 찾기
                        for server_name, server_tools in mcp_hub.mcp_tools_map.items():
                            if tool_name in server_tools:
                                available_tools.append(f"{server_name}::{tool_name}")
                                break
                
                # 설정된 도구가 없으면 카테고리 기반으로 찾기
                if not available_tools:
                    for server_name, server_tools in mcp_hub.mcp_tools_map.items():
                        for tool_name in server_tools.keys():
                            tool_lower = tool_name.lower()
                            if category == 'search' and ('search' in tool_lower or 'query' in tool_lower):
                                available_tools.append(f"{server_name}::{tool_name}")
                            elif category == 'data' and 'data' in tool_lower:
                                available_tools.append(f"{server_name}::{tool_name}")
                            elif category == 'code' and ('code' in tool_lower or 'exec' in tool_lower):
                                available_tools.append(f"{server_name}::{tool_name}")
                            elif category == 'academic' and ('academic' in tool_lower or 'paper' in tool_lower):
                                available_tools.append(f"{server_name}::{tool_name}")
                            elif category == 'business' and ('business' in tool_lower or 'market' in tool_lower):
                                available_tools.append(f"{server_name}::{tool_name}")
        except Exception as e:
            logger.warning(f"Failed to get tools from MCP Hub: {e}")
        
        # 설정된 도구가 있으면 반환, 없으면 빈 리스트 (에러 처리)
        return available_tools if available_tools else config_tools
    
    def _generate_tool_parameters(self, task: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """도구 파라미터 생성."""
        # 간단한 구현: 작업 정보를 도구 파라미터로 변환
        task_description = task.get('description', task.get('name', ''))
        
        # task_type 기반 파라미터 생성
        task_type = task.get('task_type', 'general').lower()
        
        if 'search' in tool_name.lower() or 'search' in task_type:
            query = task.get('query', task_description)
            if not query:
                query = task.get('name', '')
            return {
                'query': query,
                'max_results': task.get('max_results', 10)
            }
        elif 'fetch' in tool_name.lower() or 'fetch' in task_type:
            return {
                'url': task.get('url', ''),
                'timeout': task.get('timeout', 30)
            }
        else:
            # 기본 파라미터
            query = task.get('query', task_description)
            if not query:
                query = task.get('name', '')
            return {
                'query': query,
                'task': task
            }
    

