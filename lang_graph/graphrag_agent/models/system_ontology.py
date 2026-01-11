"""
System Ontology Models

This module defines the system ontology for executable, goal-oriented knowledge:
- Goals and sub-goals hierarchy
- Tasks with preconditions and postconditions
- State transitions
- Dependencies and constraints
- Resource requirements
- Execution feasibility validation
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
import json
import logging


class GoalStatus(str, Enum):
    """Goal status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskStatus(str, Enum):
    """Task status"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class StateType(str, Enum):
    """State types"""
    SYSTEM = "system"
    RESOURCE = "resource"
    DATA = "data"
    PROCESS = "process"
    ENVIRONMENT = "environment"


class ConstraintType(str, Enum):
    """Constraint types"""
    TEMPORAL = "temporal"
    RESOURCE = "resource"
    LOGICAL = "logical"
    BUSINESS = "business"
    TECHNICAL = "technical"


class ConstraintSeverity(str, Enum):
    """Constraint severity"""
    HARD = "hard"  # Must be satisfied
    SOFT = "soft"  # Should be satisfied
    WARNING = "warning"  # Warning only


@dataclass
class Goal:
    """Represents a goal in the system ontology"""
    id: str
    name: str
    description: str
    priority: float = 1.0  # 0.0-1.0
    status: GoalStatus = GoalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    achieved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_achieved(self) -> bool:
        """Check if goal is achieved"""
        return self.status == GoalStatus.ACHIEVED
    
    def is_blocked(self) -> bool:
        """Check if goal is blocked"""
        return self.status == GoalStatus.BLOCKED


@dataclass
class Precondition:
    """Represents a precondition for task execution"""
    id: str
    description: str
    condition: str  # Logical condition expression
    satisfied: bool = False
    required_by: Optional[str] = None  # Task ID
    satisfied_by: Optional[str] = None  # Task ID that satisfies this
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def check_satisfaction(self, context: Dict[str, Any]) -> bool:
        """Check if precondition is satisfied in given context"""
        # Simple evaluation - can be extended with expression evaluator
        try:
            # For now, use the satisfied flag
            # In production, evaluate the condition expression
            return self.satisfied
        except Exception:
            return False


@dataclass
class Postcondition:
    """Represents a postcondition after task execution"""
    id: str
    description: str
    condition: str  # Logical condition expression
    achieved: bool = False
    produced_by: Optional[str] = None  # Task ID
    enables: List[str] = field(default_factory=list)  # Task IDs that can use this
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def check_achievement(self, context: Dict[str, Any]) -> bool:
        """Check if postcondition is achieved in given context"""
        try:
            return self.achieved
        except Exception:
            return False


@dataclass
class State:
    """Represents a system state"""
    id: str
    name: str
    state_type: StateType
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, new_value: Any):
        """Update state value"""
        self.value = new_value
        self.timestamp = datetime.now()


@dataclass
class StateTransition:
    """Represents a state transition"""
    id: str
    from_state: str  # State ID
    to_state: str  # State ID
    triggered_by: Optional[str] = None  # Task ID
    condition: Optional[str] = None  # Transition condition
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Resource:
    """Represents a system resource"""
    id: str
    name: str
    resource_type: str
    availability: float = 1.0  # 0.0-1.0
    capacity: float = 1.0
    current_usage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_available(self, required_amount: float = 0.0) -> bool:
        """Check if resource is available"""
        return (self.availability > 0.0 and 
                (self.capacity - self.current_usage) >= required_amount)


@dataclass
class ResourceRequirement:
    """Represents a resource requirement for a task"""
    id: str
    resource_id: str
    required_amount: float
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Constraint:
    """Represents a constraint"""
    id: str
    constraint_type: ConstraintType
    condition: str  # Constraint expression
    severity: ConstraintSeverity = ConstraintSeverity.HARD
    violated: bool = False
    applies_to: List[str] = field(default_factory=list)  # Task/Goal IDs
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def check_violation(self, context: Dict[str, Any]) -> bool:
        """Check if constraint is violated"""
        try:
            # Evaluate constraint condition
            return self.violated
        except Exception:
            return False


@dataclass
class Task:
    """Represents a task in the system ontology"""
    id: str
    name: str
    description: str
    executable: bool = True
    status: TaskStatus = TaskStatus.PENDING
    priority: float = 1.0  # 0.0-1.0
    execution_time: Optional[float] = None  # Estimated execution time
    success_rate: float = 1.0  # 0.0-1.0
    preconditions: List[str] = field(default_factory=list)  # Precondition IDs
    postconditions: List[str] = field(default_factory=list)  # Postcondition IDs
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    resource_requirements: List[str] = field(default_factory=list)  # ResourceRequirement IDs
    constraints: List[str] = field(default_factory=list)  # Constraint IDs
    state_transitions: List[str] = field(default_factory=list)  # StateTransition IDs
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def is_executable(self, 
                     preconditions_map: Dict[str, Precondition],
                     constraints_map: Dict[str, Constraint],
                     resources_map: Dict[str, Resource],
                     resource_requirements_map: Dict[str, ResourceRequirement],
                     context: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
        """
        Check if task is executable
        
        Returns:
            Tuple of (is_executable, reasons)
        """
        reasons = []
        context = context or {}
        
        # Check preconditions
        for pre_id in self.preconditions:
            if pre_id in preconditions_map:
                pre = preconditions_map[pre_id]
                if not pre.check_satisfaction(context):
                    reasons.append(f"Precondition not satisfied: {pre.description}")
                    return False, reasons
        
        # Check constraints
        for constraint_id in self.constraints:
            if constraint_id in constraints_map:
                constraint = constraints_map[constraint_id]
                if constraint.severity == ConstraintSeverity.HARD:
                    if constraint.check_violation(context):
                        reasons.append(f"Hard constraint violated: {constraint.condition}")
                        return False, reasons
        
        # Check resource requirements
        for req_id in self.resource_requirements:
            if req_id in resource_requirements_map:
                req = resource_requirements_map[req_id]
                if req.resource_id in resources_map:
                    resource = resources_map[req.resource_id]
                    if not resource.is_available(req.required_amount):
                        reasons.append(f"Resource not available: {resource.name}")
                        return False, reasons
        
        # Check dependencies
        for dep_id in self.dependencies:
            # Dependencies should be completed
            # This will be checked by the executor
            pass
        
        return True, reasons
    
    def can_transition_to(self, target_status: TaskStatus) -> bool:
        """Check if task can transition to target status"""
        valid_transitions = {
            TaskStatus.PENDING: [TaskStatus.READY, TaskStatus.BLOCKED, TaskStatus.CANCELLED],
            TaskStatus.READY: [TaskStatus.RUNNING, TaskStatus.BLOCKED, TaskStatus.CANCELLED],
            TaskStatus.RUNNING: [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED],
            TaskStatus.BLOCKED: [TaskStatus.READY, TaskStatus.CANCELLED],
            TaskStatus.COMPLETED: [],
            TaskStatus.FAILED: [TaskStatus.READY, TaskStatus.CANCELLED],
            TaskStatus.CANCELLED: []
        }
        return target_status in valid_transitions.get(self.status, [])


@dataclass
class SystemOntology:
    """
    Complete system ontology
    
    Represents the entire system ontology with goals, tasks, states, etc.
    """
    goals: Dict[str, Goal] = field(default_factory=dict)
    tasks: Dict[str, Task] = field(default_factory=dict)
    preconditions: Dict[str, Precondition] = field(default_factory=dict)
    postconditions: Dict[str, Postcondition] = field(default_factory=dict)
    states: Dict[str, State] = field(default_factory=dict)
    state_transitions: Dict[str, StateTransition] = field(default_factory=dict)
    resources: Dict[str, Resource] = field(default_factory=dict)
    resource_requirements: Dict[str, ResourceRequirement] = field(default_factory=dict)
    constraints: Dict[str, Constraint] = field(default_factory=dict)
    
    def add_goal(self, goal: Goal):
        """Add a goal to the ontology"""
        self.goals[goal.id] = goal
    
    def add_task(self, task: Task):
        """Add a task to the ontology"""
        self.tasks[task.id] = task
    
    def add_precondition(self, precondition: Precondition):
        """Add a precondition to the ontology"""
        self.preconditions[precondition.id] = precondition
    
    def add_postcondition(self, postcondition: Postcondition):
        """Add a postcondition to the ontology"""
        self.postconditions[postcondition.id] = postcondition
    
    def add_state(self, state: State):
        """Add a state to the ontology"""
        self.states[state.id] = state
    
    def add_state_transition(self, transition: StateTransition):
        """Add a state transition to the ontology"""
        self.state_transitions[transition.id] = transition
    
    def add_resource(self, resource: Resource):
        """Add a resource to the ontology"""
        self.resources[resource.id] = resource
    
    def add_resource_requirement(self, requirement: ResourceRequirement):
        """Add a resource requirement to the ontology"""
        self.resource_requirements[requirement.id] = requirement
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the ontology"""
        self.constraints[constraint.id] = constraint
    
    def get_executable_tasks(self, context: Dict[str, Any] = None) -> List[Task]:
        """Get all executable tasks"""
        executable = []
        context = context or {}
        
        for task in self.tasks.values():
            if task.status == TaskStatus.READY or task.status == TaskStatus.PENDING:
                is_executable, _ = task.is_executable(
                    self.preconditions,
                    self.constraints,
                    self.resources,
                    self.resource_requirements,
                    context
                )
                if is_executable:
                    executable.append(task)
        
        return sorted(executable, key=lambda t: t.priority, reverse=True)
    
    def get_goal_achievement_path(self, goal_id: str) -> List[Task]:
        """
        Get tasks needed to achieve a goal
        
        Returns:
            List of tasks in execution order
        """
        # This is a simplified version
        # Full implementation would do dependency resolution
        goal = self.goals.get(goal_id)
        if not goal:
            return []
        
        # Find tasks that achieve this goal
        # In a real implementation, we'd query the graph for ACHIEVED_BY relationships
        tasks = []
        for task in self.tasks.values():
            # Check if task is related to goal (would be in graph)
            if goal_id in task.metadata.get('achieves_goals', []):
                tasks.append(task)
        
        # Sort by dependencies
        return self._resolve_dependencies(tasks)
    
    def _resolve_dependencies(self, tasks: List[Task]) -> List[Task]:
        """Resolve task dependencies and return ordered list"""
        # Topological sort
        ordered = []
        visited = set()
        visiting = set()
        
        def visit(task: Task):
            if task.id in visiting:
                return  # Circular dependency
            if task.id in visited:
                return
            
            visiting.add(task.id)
            
            # Visit dependencies first
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    visit(self.tasks[dep_id])
            
            visiting.remove(task.id)
            visited.add(task.id)
            ordered.append(task)
        
        for task in tasks:
            if task.id not in visited:
                visit(task)
        
        return ordered
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ontology to dictionary"""
        return {
            "goals": {gid: {
                "id": g.id,
                "name": g.name,
                "description": g.description,
                "priority": g.priority,
                "status": g.status.value,
                "created_at": g.created_at.isoformat(),
                "achieved_at": g.achieved_at.isoformat() if g.achieved_at else None,
                "metadata": g.metadata
            } for gid, g in self.goals.items()},
            "tasks": {tid: {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "executable": t.executable,
                "status": t.status.value,
                "priority": t.priority,
                "execution_time": t.execution_time,
                "success_rate": t.success_rate,
                "preconditions": t.preconditions,
                "postconditions": t.postconditions,
                "dependencies": t.dependencies,
                "resource_requirements": t.resource_requirements,
                "constraints": t.constraints,
                "state_transitions": t.state_transitions,
                "metadata": t.metadata,
                "created_at": t.created_at.isoformat(),
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None
            } for tid, t in self.tasks.items()},
            "preconditions": {pid: {
                "id": p.id,
                "description": p.description,
                "condition": p.condition,
                "satisfied": p.satisfied,
                "required_by": p.required_by,
                "satisfied_by": p.satisfied_by,
                "metadata": p.metadata
            } for pid, p in self.preconditions.items()},
            "postconditions": {pid: {
                "id": p.id,
                "description": p.description,
                "condition": p.condition,
                "achieved": p.achieved,
                "produced_by": p.produced_by,
                "enables": p.enables,
                "metadata": p.metadata
            } for pid, p in self.postconditions.items()},
            "states": {sid: {
                "id": s.id,
                "name": s.name,
                "state_type": s.state_type.value,
                "value": str(s.value),
                "timestamp": s.timestamp.isoformat(),
                "metadata": s.metadata
            } for sid, s in self.states.items()},
            "resources": {rid: {
                "id": r.id,
                "name": r.name,
                "resource_type": r.resource_type,
                "availability": r.availability,
                "capacity": r.capacity,
                "current_usage": r.current_usage,
                "metadata": r.metadata
            } for rid, r in self.resources.items()},
            "constraints": {cid: {
                "id": c.id,
                "constraint_type": c.constraint_type.value,
                "condition": c.condition,
                "severity": c.severity.value,
                "violated": c.violated,
                "applies_to": c.applies_to,
                "metadata": c.metadata
            } for cid, c in self.constraints.items()}
        }
