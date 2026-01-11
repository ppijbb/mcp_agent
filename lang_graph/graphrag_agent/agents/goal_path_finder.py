"""
Goal Achievement Path Finder

This module provides algorithms for finding paths to achieve goals:
- Goal hierarchy traversal
- Task dependency resolution
- Executable task chain generation
- Constraint validation
- Optimal path selection
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import deque

from config import AgentConfig
from models.system_ontology import (
    SystemOntology,
    Goal,
    Task,
    GoalStatus,
    TaskStatus,
    ConstraintSeverity
)


@dataclass
class AchievementPath:
    """Represents a path to achieve a goal"""
    goal_id: str
    tasks: List[Task]  # Tasks in execution order
    subgoals: List[Goal]  # Subgoals in order
    total_cost: float  # Estimated total execution time
    confidence: float  # 0.0-1.0
    constraints_satisfied: bool
    reasoning: str


class GoalPathFinder:
    """
    Goal achievement path finder
    
    Finds optimal paths to achieve goals by:
    1. Traversing goal hierarchy
    2. Resolving task dependencies
    3. Generating executable task chains
    4. Validating constraints
    5. Selecting optimal paths
    """
    
    def __init__(self, config: AgentConfig, ontology: SystemOntology):
        """
        Initialize goal path finder
        
        Args:
            config: Agent configuration
            ontology: System ontology
        """
        self.config = config
        self.ontology = ontology
        self.logger = logging.getLogger(__name__)
    
    def find_achievement_path(
        self,
        goal_id: str,
        context: Dict[str, Any] = None
    ) -> Optional[AchievementPath]:
        """
        Find a path to achieve a goal
        
        Args:
            goal_id: Goal ID to achieve
            context: Optional context
            
        Returns:
            AchievementPath or None if no path found
        """
        if goal_id not in self.ontology.goals:
            self.logger.error(f"Goal {goal_id} not found")
            return None
        
        goal = self.ontology.goals[goal_id]
        context = context or {}
        
        # Step 1: Find all tasks that achieve this goal
        goal_tasks = self._find_tasks_for_goal(goal_id)
        
        if not goal_tasks:
            self.logger.warning(f"No tasks found for goal {goal_id}")
            return None
        
        # Step 2: Resolve dependencies and create execution order
        ordered_tasks = self._resolve_task_dependencies(goal_tasks)
        
        # Step 3: Validate constraints
        constraints_satisfied, constraint_reasons = self._validate_path_constraints(
            ordered_tasks, context
        )
        
        # Step 4: Calculate path metrics
        total_cost = sum(t.execution_time or 0.0 for t in ordered_tasks)
        confidence = self._calculate_path_confidence(ordered_tasks)
        
        # Step 5: Find subgoals
        subgoals = self._find_subgoals(goal_id)
        
        reasoning = f"Path found with {len(ordered_tasks)} tasks. "
        if not constraints_satisfied:
            reasoning += f"Constraints: {', '.join(constraint_reasons)}. "
        reasoning += f"Estimated time: {total_cost:.2f}s. Confidence: {confidence:.2f}"
        
        return AchievementPath(
            goal_id=goal_id,
            tasks=ordered_tasks,
            subgoals=subgoals,
            total_cost=total_cost,
            confidence=confidence,
            constraints_satisfied=constraints_satisfied,
            reasoning=reasoning
        )
    
    def find_all_achievement_paths(
        self,
        goal_id: str,
        context: Dict[str, Any] = None
    ) -> List[AchievementPath]:
        """
        Find all possible paths to achieve a goal
        
        Args:
            goal_id: Goal ID
            context: Optional context
            
        Returns:
            List of AchievementPath, sorted by confidence
        """
        if goal_id not in self.ontology.goals:
            return []
        
        # Find all task combinations that could achieve the goal
        goal_tasks = self._find_tasks_for_goal(goal_id)
        
        if not goal_tasks:
            return []
        
        # Generate different task orderings
        paths = []
        
        # Try different task combinations
        for task_subset in self._generate_task_combinations(goal_tasks):
            try:
                ordered_tasks = self._resolve_task_dependencies(task_subset)
                
                constraints_satisfied, _ = self._validate_path_constraints(
                    ordered_tasks, context or {}
                )
                
                total_cost = sum(t.execution_time or 0.0 for t in ordered_tasks)
                confidence = self._calculate_path_confidence(ordered_tasks)
                
                subgoals = self._find_subgoals(goal_id)
                
                path = AchievementPath(
                    goal_id=goal_id,
                    tasks=ordered_tasks,
                    subgoals=subgoals,
                    total_cost=total_cost,
                    confidence=confidence,
                    constraints_satisfied=constraints_satisfied,
                    reasoning=f"Alternative path with {len(ordered_tasks)} tasks"
                )
                
                paths.append(path)
            except Exception as e:
                self.logger.warning(f"Failed to create path: {e}")
                continue
        
        # Sort by confidence and cost
        paths.sort(key=lambda p: (p.confidence, -p.total_cost), reverse=True)
        
        return paths
    
    def find_executable_tasks(
        self,
        context: Dict[str, Any] = None
    ) -> List[Task]:
        """
        Find all currently executable tasks
        
        Args:
            context: Optional context
            
        Returns:
            List of executable tasks
        """
        return self.ontology.get_executable_tasks(context or {})
    
    def _find_tasks_for_goal(self, goal_id: str) -> List[Task]:
        """Find all tasks that achieve a goal"""
        tasks = []
        
        # Direct task-goal relationships (stored in task metadata)
        for task in self.ontology.tasks.values():
            if goal_id in task.metadata.get("achieves_goals", []):
                tasks.append(task)
        
        # Also check subgoals
        for goal in self.ontology.goals.values():
            if goal.metadata.get("parent_goal") == goal_id:
                subgoal_tasks = self._find_tasks_for_goal(goal.id)
                tasks.extend(subgoal_tasks)
        
        return tasks
    
    def _resolve_task_dependencies(self, tasks: List[Task]) -> List[Task]:
        """Resolve task dependencies and return ordered list"""
        # Build dependency graph
        task_map = {t.id: t for t in tasks}
        in_degree = {t.id: 0 for t in tasks}
        dependencies = {t.id: [] for t in tasks}
        
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    in_degree[task.id] += 1
                    dependencies[dep_id].append(task.id)
        
        # Topological sort
        queue = deque([t.id for t in tasks if in_degree[t.id] == 0])
        ordered = []
        
        while queue:
            task_id = queue.popleft()
            ordered.append(task_map[task_id])
            
            for dependent_id in dependencies[task_id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)
        
        # Check for circular dependencies
        if len(ordered) < len(tasks):
            self.logger.warning("Circular dependencies detected in task graph")
            # Add remaining tasks
            for task in tasks:
                if task not in ordered:
                    ordered.append(task)
        
        return ordered
    
    def _validate_path_constraints(
        self,
        tasks: List[Task],
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate constraints for a task path"""
        reasons = []
        
        for task in tasks:
            for constraint_id in task.constraints:
                if constraint_id in self.ontology.constraints:
                    constraint = self.ontology.constraints[constraint_id]
                    
                    if constraint.severity == ConstraintSeverity.HARD:
                        if constraint.check_violation(context):
                            reasons.append(f"Hard constraint violated: {constraint.condition} (task: {task.name})")
                            return False, reasons
        
        return True, reasons
    
    def _calculate_path_confidence(self, tasks: List[Task]) -> float:
        """Calculate confidence for a task path"""
        if not tasks:
            return 0.0
        
        # Average success rate of tasks
        success_rates = [t.success_rate for t in tasks if t.success_rate]
        if not success_rates:
            return 0.5
        
        # Product of success rates (all tasks must succeed)
        import math
        confidence = math.prod(success_rates)
        
        # Adjust based on number of tasks (more tasks = lower confidence)
        if len(tasks) > 1:
            confidence *= (0.95 ** (len(tasks) - 1))
        
        return min(1.0, max(0.0, confidence))
    
    def _find_subgoals(self, goal_id: str) -> List[Goal]:
        """Find subgoals of a goal"""
        subgoals = []
        
        for goal in self.ontology.goals.values():
            if goal.metadata.get("parent_goal") == goal_id:
                subgoals.append(goal)
                # Recursively find sub-subgoals
                subgoals.extend(self._find_subgoals(goal.id))
        
        return subgoals
    
    def _generate_task_combinations(self, tasks: List[Task]) -> List[List[Task]]:
        """Generate different task combinations"""
        # For now, return single combination (all tasks)
        # Can be extended to generate alternative paths
        if not tasks:
            return []
        
        return [tasks]
    
    def find_optimal_path(
        self,
        goal_id: str,
        optimization_criteria: str = "confidence",
        context: Dict[str, Any] = None
    ) -> Optional[AchievementPath]:
        """
        Find optimal path based on criteria
        
        Args:
            goal_id: Goal ID
            optimization_criteria: "confidence", "cost", or "balanced"
            context: Optional context
            
        Returns:
            Optimal AchievementPath
        """
        paths = self.find_all_achievement_paths(goal_id, context)
        
        if not paths:
            return None
        
        if optimization_criteria == "confidence":
            return paths[0]  # Already sorted by confidence
        elif optimization_criteria == "cost":
            paths.sort(key=lambda p: p.total_cost)
            return paths[0]
        elif optimization_criteria == "balanced":
            # Balance between confidence and cost
            paths.sort(key=lambda p: (p.confidence * 0.7 - p.total_cost * 0.3), reverse=True)
            return paths[0]
        else:
            return paths[0]
