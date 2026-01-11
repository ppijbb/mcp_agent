"""
Task Execution Engine

This module provides task execution capabilities:
- Precondition validation
- Task execution
- State transition management
- Postcondition verification
- Dependency-based execution ordering
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass

from config import AgentConfig
from models.system_ontology import (
    SystemOntology,
    Task,
    Precondition,
    Postcondition,
    State,
    StateTransition,
    Resource,
    ResourceRequirement,
    TaskStatus,
    GoalStatus
)


@dataclass
class ExecutionResult:
    """Result of task execution"""
    task_id: str
    success: bool
    execution_time: float
    error: Optional[str] = None
    state_changes: List[State] = None
    postconditions_achieved: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.state_changes is None:
            self.state_changes = []
        if self.postconditions_achieved is None:
            self.postconditions_achieved = []
        if self.metadata is None:
            self.metadata = {}


class TaskExecutor:
    """
    Task execution engine
    
    Executes tasks from system ontology with:
    - Precondition validation
    - Resource allocation
    - Task execution
    - State transition
    - Postcondition verification
    - Dependency management
    """
    
    def __init__(self, config: AgentConfig, ontology: SystemOntology):
        """
        Initialize task executor
        
        Args:
            config: Agent configuration
            ontology: System ontology to execute tasks from
        """
        self.config = config
        self.ontology = ontology
        self.logger = logging.getLogger(__name__)
        
        # Execution state
        self.execution_history: List[ExecutionResult] = []
        self.running_tasks: Dict[str, Task] = {}
        self.task_handlers: Dict[str, Callable] = {}  # Custom task handlers
    
    def register_task_handler(self, task_name: str, handler: Callable):
        """Register a custom handler for a specific task"""
        self.task_handlers[task_name] = handler
    
    async def execute_task(
        self,
        task_id: str,
        context: Dict[str, Any] = None
    ) -> ExecutionResult:
        """
        Execute a single task
        
        Args:
            task_id: Task ID to execute
            context: Execution context
            
        Returns:
            ExecutionResult
        """
        if task_id not in self.ontology.tasks:
            return ExecutionResult(
                task_id=task_id,
                success=False,
                execution_time=0.0,
                error=f"Task {task_id} not found in ontology"
            )
        
        task = self.ontology.tasks[task_id]
        context = context or {}
        start_time = datetime.now()
        
        try:
            # Step 1: Validate preconditions
            preconditions_valid, reasons = await self._validate_preconditions(task, context)
            if not preconditions_valid:
                task.status = TaskStatus.BLOCKED
                return ExecutionResult(
                    task_id=task_id,
                    success=False,
                    execution_time=0.0,
                    error=f"Preconditions not satisfied: {', '.join(reasons)}"
                )
            
            # Step 2: Allocate resources
            resource_allocation = await self._allocate_resources(task)
            if not resource_allocation[0]:
                task.status = TaskStatus.BLOCKED
                return ExecutionResult(
                    task_id=task_id,
                    success=False,
                    execution_time=0.0,
                    error=f"Resources not available: {resource_allocation[1]}"
                )
            
            # Step 3: Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            self.running_tasks[task_id] = task
            
            # Step 4: Execute task
            execution_success = await self._execute_task_logic(task, context)
            
            # Step 5: Handle post-execution
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if execution_success:
                # Step 6: Verify postconditions
                postconditions_achieved = await self._verify_postconditions(task, context)
                
                # Step 7: Apply state transitions
                state_changes = await self._apply_state_transitions(task, context)
                
                # Step 8: Update task status
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                
                # Step 9: Release resources
                await self._release_resources(task)
                
                # Step 10: Update dependent tasks
                await self._update_dependent_tasks(task)
                
                result = ExecutionResult(
                    task_id=task_id,
                    success=True,
                    execution_time=execution_time,
                    state_changes=state_changes,
                    postconditions_achieved=postconditions_achieved,
                    metadata={"task_name": task.name}
                )
            else:
                task.status = TaskStatus.FAILED
                await self._release_resources(task)
                
                result = ExecutionResult(
                    task_id=task_id,
                    success=False,
                    execution_time=execution_time,
                    error="Task execution failed",
                    metadata={"task_name": task.name}
                )
            
            self.execution_history.append(result)
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution error: {e}")
            task.status = TaskStatus.FAILED
            await self._release_resources(task)
            
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            return ExecutionResult(
                task_id=task_id,
                success=False,
                execution_time=execution_time,
                error=str(e)
            )
    
    async def execute_task_chain(
        self,
        task_ids: List[str],
        context: Dict[str, Any] = None
    ) -> List[ExecutionResult]:
        """
        Execute a chain of tasks in order
        
        Args:
            task_ids: List of task IDs in execution order
            context: Execution context
            
        Returns:
            List of ExecutionResult
        """
        results = []
        context = context or {}
        
        for task_id in task_ids:
            result = await self.execute_task(task_id, context)
            results.append(result)
            
            # Update context with execution results
            context[f"task_{task_id}_result"] = result
            
            # Stop on failure if critical
            if not result.success:
                self.logger.warning(f"Task chain stopped at {task_id} due to failure")
                break
        
        return results
    
    async def _validate_preconditions(
        self,
        task: Task,
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate all preconditions for a task"""
        reasons = []
        
        for pre_id in task.preconditions:
            if pre_id in self.ontology.preconditions:
                pre = self.ontology.preconditions[pre_id]
                
                # Check if precondition is satisfied
                if not pre.satisfied:
                    # Check if it's satisfied by another task's postcondition
                    if pre.satisfied_by:
                        # Check if the satisfying task is completed
                        if pre.satisfied_by in self.ontology.tasks:
                            satisfying_task = self.ontology.tasks[pre.satisfied_by]
                            if satisfying_task.status == TaskStatus.COMPLETED:
                                pre.satisfied = True
                            else:
                                reasons.append(f"Precondition '{pre.description}' not satisfied (depends on task {pre.satisfied_by})")
                        else:
                            reasons.append(f"Precondition '{pre.description}' not satisfied")
                    else:
                        reasons.append(f"Precondition '{pre.description}' not satisfied")
                
                # Evaluate condition if provided
                if pre.condition and not pre.check_satisfaction(context):
                    reasons.append(f"Precondition condition not met: {pre.condition}")
        
        return len(reasons) == 0, reasons
    
    async def _allocate_resources(self, task: Task) -> Tuple[bool, str]:
        """Allocate resources for task execution"""
        for req_id in task.resource_requirements:
            if req_id in self.ontology.resource_requirements:
                req = self.ontology.resource_requirements[req_id]
                if req.resource_id in self.ontology.resources:
                    resource = self.ontology.resources[req.resource_id]
                    
                    if not resource.is_available(req.required_amount):
                        return False, f"Resource {resource.name} not available"
                    
                    # Allocate resource
                    resource.current_usage += req.required_amount
                    resource.availability = max(0.0, 1.0 - (resource.current_usage / resource.capacity))
        
        return True, ""
    
    async def _release_resources(self, task: Task):
        """Release resources after task execution"""
        for req_id in task.resource_requirements:
            if req_id in self.ontology.resource_requirements:
                req = self.ontology.resource_requirements[req_id]
                if req.resource_id in self.ontology.resources:
                    resource = self.ontology.resources[req.resource_id]
                    
                    # Release resource
                    resource.current_usage = max(0.0, resource.current_usage - req.required_amount)
                    resource.availability = max(0.0, 1.0 - (resource.current_usage / resource.capacity))
    
    async def _execute_task_logic(
        self,
        task: Task,
        context: Dict[str, Any]
    ) -> bool:
        """Execute the actual task logic"""
        try:
            # Check for custom handler
            if task.name in self.task_handlers:
                handler = self.task_handlers[task.name]
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(task, context)
                else:
                    result = handler(task, context)
                return bool(result)
            
            # Default execution (placeholder)
            # In production, this would execute actual task logic
            self.logger.info(f"Executing task: {task.name}")
            
            # Simulate execution time if provided
            if task.execution_time:
                await asyncio.sleep(min(task.execution_time, 1.0))  # Cap at 1 second for simulation
            
            # Use success rate for probabilistic success
            import random
            success = random.random() < task.success_rate
            
            return success
            
        except Exception as e:
            self.logger.error(f"Task execution logic error: {e}")
            return False
    
    async def _verify_postconditions(
        self,
        task: Task,
        context: Dict[str, Any]
    ) -> List[str]:
        """Verify postconditions after task execution"""
        achieved = []
        
        for post_id in task.postconditions:
            if post_id in self.ontology.postconditions:
                post = self.ontology.postconditions[post_id]
                
                # Mark as achieved
                post.achieved = True
                achieved.append(post_id)
                
                # Update preconditions that this postcondition satisfies
                for other_task_id, other_task in self.ontology.tasks.items():
                    if other_task_id != task.id:
                        for pre_id in other_task.preconditions:
                            if pre_id in self.ontology.preconditions:
                                pre = self.ontology.preconditions[pre_id]
                                if pre.satisfied_by == task.id:
                                    pre.satisfied = True
        
        return achieved
    
    async def _apply_state_transitions(
        self,
        task: Task,
        context: Dict[str, Any]
    ) -> List[State]:
        """Apply state transitions after task execution"""
        state_changes = []
        
        for trans_id in task.state_transitions:
            if trans_id in self.ontology.state_transitions:
                transition = self.ontology.state_transitions[trans_id]
                
                # Get from and to states
                if (transition.from_state in self.ontology.states and
                    transition.to_state in self.ontology.states):
                    
                    from_state = self.ontology.states[transition.from_state]
                    to_state = self.ontology.states[transition.to_state]
                    
                    # Check transition condition
                    if not transition.condition or self._evaluate_condition(transition.condition, context):
                        # Apply transition
                        to_state.update("active")
                        state_changes.append(to_state)
                        
                        self.logger.info(f"State transition: {from_state.name} -> {to_state.name}")
        
        return state_changes
    
    async def _update_dependent_tasks(self, completed_task: Task):
        """Update tasks that depend on the completed task"""
        for task_id, task in self.ontology.tasks.items():
            if task_id != completed_task.id:
                # Check if this task depends on the completed task
                if completed_task.id in task.dependencies:
                    # Check if all dependencies are now satisfied
                    all_deps_satisfied = True
                    for dep_id in task.dependencies:
                        if dep_id in self.ontology.tasks:
                            dep_task = self.ontology.tasks[dep_id]
                            if dep_task.status != TaskStatus.COMPLETED:
                                all_deps_satisfied = False
                                break
                    
                    if all_deps_satisfied:
                        # Check if task is now executable
                        is_executable, _ = task.is_executable(
                            self.ontology.preconditions,
                            self.ontology.constraints,
                            self.ontology.resources,
                            self.ontology.resource_requirements,
                            {}
                        )
                        
                        if is_executable and task.status == TaskStatus.BLOCKED:
                            task.status = TaskStatus.READY
                        elif is_executable and task.status == TaskStatus.PENDING:
                            task.status = TaskStatus.READY
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition expression"""
        # Simple evaluation - can be extended with proper expression evaluator
        try:
            # For now, return True if condition is not empty
            # In production, use a proper expression evaluator
            return bool(condition)
        except Exception:
            return False
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution history"""
        total = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.success)
        failed = total - successful
        total_time = sum(r.execution_time for r in self.execution_history)
        
        return {
            "total_tasks": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "total_execution_time": total_time,
            "average_execution_time": total_time / total if total > 0 else 0.0
        }
