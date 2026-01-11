"""
Goal-Oriented Ontology Builder

This module builds system ontologies from goals:
- Goal extraction and hierarchy construction
- Task decomposition
- Precondition and postcondition inference
- Dependency discovery
- Executable task chain generation
"""

import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from config import AgentConfig
from .llm_processor import LLMProcessor
from models.system_ontology import (
    SystemOntology,
    Goal,
    Task,
    Precondition,
    Postcondition,
    State,
    StateTransition,
    Resource,
    ResourceRequirement,
    Constraint,
    GoalStatus,
    TaskStatus,
    StateType,
    ConstraintType,
    ConstraintSeverity
)


@dataclass
class GoalDecomposition:
    """Result of goal decomposition"""
    goal: Goal
    subgoals: List[Goal]
    tasks: List[Task]
    preconditions: List[Precondition]
    postconditions: List[Postcondition]
    dependencies: List[Tuple[str, str]]  # (task_id, depends_on_task_id)
    constraints: List[Constraint]
    resources: List[Resource]
    state_transitions: List[StateTransition]


class GoalOntologyBuilder:
    """
    Goal-oriented ontology builder
    
    Builds executable system ontologies from high-level goals by:
    1. Extracting goals and creating hierarchy
    2. Decomposing goals into tasks
    3. Inferring preconditions and postconditions
    4. Discovering dependencies
    5. Creating executable task chains
    """
    
    def __init__(self, config: AgentConfig, llm_processor: LLMProcessor):
        """
        Initialize goal ontology builder
        
        Args:
            config: Agent configuration
            llm_processor: LLM processor for goal analysis
        """
        self.config = config
        self.llm_processor = llm_processor
        self.logger = logging.getLogger(__name__)
    
    def build_ontology_from_goals(
        self,
        goals: List[str],
        context: Dict[str, Any] = None
    ) -> SystemOntology:
        """
        Build system ontology from high-level goals
        
        Args:
            goals: List of goal descriptions
            context: Optional context information
            
        Returns:
            SystemOntology with goals, tasks, and relationships
        """
        ontology = SystemOntology()
        context = context or {}
        
        # Step 1: Extract and create goal hierarchy
        goal_objects = []
        for goal_desc in goals:
            goal = self._extract_goal(goal_desc, context)
            if goal:
                goal_objects.append(goal)
                ontology.add_goal(goal)
        
        # Step 2: Decompose goals into subgoals and tasks
        for goal in goal_objects:
            decomposition = self._decompose_goal(goal, context, ontology)
            
            # Add decomposed elements to ontology
            for subgoal in decomposition.subgoals:
                ontology.add_goal(subgoal)
            
            for task in decomposition.tasks:
                ontology.add_task(task)
            
            for precondition in decomposition.preconditions:
                ontology.add_precondition(precondition)
            
            for postcondition in decomposition.postconditions:
                ontology.add_postcondition(postcondition)
            
            for constraint in decomposition.constraints:
                ontology.add_constraint(constraint)
            
            for resource in decomposition.resources:
                ontology.add_resource(resource)
            
            for transition in decomposition.state_transitions:
                ontology.add_state_transition(transition)
            
            # Add dependencies to tasks
            for task_id, depends_on_id in decomposition.dependencies:
                if task_id in ontology.tasks and depends_on_id in ontology.tasks:
                    if depends_on_id not in ontology.tasks[task_id].dependencies:
                        ontology.tasks[task_id].dependencies.append(depends_on_id)
        
        # Step 3: Link preconditions and postconditions
        self._link_preconditions_postconditions(ontology)
        
        # Step 4: Update task statuses based on executability
        self._update_task_statuses(ontology, context)
        
        return ontology
    
    def _extract_goal(self, goal_description: str, context: Dict[str, Any]) -> Optional[Goal]:
        """Extract goal from description"""
        prompt = f"""
Extract a goal from this description and provide structured information.

Goal Description: "{goal_description}"

Context: {json.dumps(context, indent=2) if context else "None"}

Return JSON:
{{
    "name": "goal_name",
    "description": "detailed description",
    "priority": 0.0-1.0,
    "metadata": {{
        "domain": "domain context",
        "urgency": "high|medium|low"
    }}
}}
"""
        
        try:
            response = self.llm_processor._call_llm(prompt)
            goal_data = json.loads(response)
            
            goal = Goal(
                id=f"goal_{uuid.uuid4().hex[:8]}",
                name=goal_data.get("name", goal_description[:50]),
                description=goal_data.get("description", goal_description),
                priority=float(goal_data.get("priority", 0.5)),
                status=GoalStatus.PENDING,
                metadata=goal_data.get("metadata", {})
            )
            
            return goal
            
        except Exception as e:
            self.logger.error(f"Failed to extract goal: {e}")
            return None
    
    def _decompose_goal(
        self,
        goal: Goal,
        context: Dict[str, Any],
        ontology: SystemOntology
    ) -> GoalDecomposition:
        """Decompose goal into subgoals, tasks, and relationships"""
        
        prompt = f"""
Decompose this goal into a system ontology structure.

Goal: {goal.name}
Description: {goal.description}
Priority: {goal.priority}

Context: {json.dumps(context, indent=2) if context else "None"}

Decompose into:
1. Sub-goals (if needed)
2. Tasks that achieve the goal
3. Preconditions for each task
4. Postconditions for each task
5. Dependencies between tasks
6. Constraints
7. Required resources
8. State transitions

Return JSON:
{{
    "subgoals": [
        {{
            "name": "subgoal_name",
            "description": "description",
            "priority": 0.0-1.0
        }}
    ],
    "tasks": [
        {{
            "name": "task_name",
            "description": "detailed task description",
            "priority": 0.0-1.0,
            "execution_time": estimated_time_in_seconds,
            "success_rate": 0.0-1.0,
            "preconditions": [
                {{
                    "description": "precondition description",
                    "condition": "logical condition"
                }}
            ],
            "postconditions": [
                {{
                    "description": "postcondition description",
                    "condition": "logical condition"
                }}
            ],
            "dependencies": ["task_name_that_must_complete_first"],
            "resource_requirements": [
                {{
                    "resource_name": "resource_name",
                    "resource_type": "type",
                    "required_amount": 1.0
                }}
            ],
            "constraints": [
                {{
                    "type": "temporal|resource|logical|business|technical",
                    "condition": "constraint expression",
                    "severity": "hard|soft|warning"
                }}
            ],
            "state_transitions": [
                {{
                    "from_state": "state_name",
                    "to_state": "state_name",
                    "condition": "transition condition"
                }}
            ]
        }}
    ],
    "resources": [
        {{
            "name": "resource_name",
            "resource_type": "type",
            "capacity": 1.0,
            "availability": 1.0
        }}
    ]
}}

Guidelines:
- Focus on executable tasks, not just knowledge
- Identify clear preconditions and postconditions
- Discover real dependencies between tasks
- Consider system constraints
- Think about state changes
- Make tasks actionable and specific
"""
        
        try:
            response = self.llm_processor._call_llm(prompt)
            decomposition_data = json.loads(response)
            
            subgoals = []
            tasks = []
            preconditions = []
            postconditions = []
            dependencies = []
            constraints = []
            resources = []
            state_transitions = []
            
            # Create subgoals
            for sg_data in decomposition_data.get("subgoals", []):
                subgoal = Goal(
                    id=f"goal_{uuid.uuid4().hex[:8]}",
                    name=sg_data.get("name", ""),
                    description=sg_data.get("description", ""),
                    priority=float(sg_data.get("priority", 0.5)),
                    status=GoalStatus.PENDING,
                    metadata={"parent_goal": goal.id}
                )
                subgoals.append(subgoal)
            
            # Create resources first (tasks may reference them)
            resource_map = {}
            for res_data in decomposition_data.get("resources", []):
                resource = Resource(
                    id=f"resource_{uuid.uuid4().hex[:8]}",
                    name=res_data.get("name", ""),
                    resource_type=res_data.get("resource_type", "generic"),
                    capacity=float(res_data.get("capacity", 1.0)),
                    availability=float(res_data.get("availability", 1.0))
                )
                resources.append(resource)
                resource_map[res_data.get("name", "")] = resource.id
            
            # Create tasks
            task_map = {}
            for task_data in decomposition_data.get("tasks", []):
                task_id = f"task_{uuid.uuid4().hex[:8]}"
                task = Task(
                    id=task_id,
                    name=task_data.get("name", ""),
                    description=task_data.get("description", ""),
                    priority=float(task_data.get("priority", 0.5)),
                    execution_time=task_data.get("execution_time"),
                    success_rate=float(task_data.get("success_rate", 1.0)),
                    status=TaskStatus.PENDING,
                    metadata={"achieves_goals": [goal.id]}
                )
                
                # Create preconditions
                task_preconditions = []
                for pre_data in task_data.get("preconditions", []):
                    pre_id = f"pre_{uuid.uuid4().hex[:8]}"
                    precondition = Precondition(
                        id=pre_id,
                        description=pre_data.get("description", ""),
                        condition=pre_data.get("condition", ""),
                        satisfied=False,
                        required_by=task_id
                    )
                    preconditions.append(precondition)
                    task_preconditions.append(pre_id)
                
                task.preconditions = task_preconditions
                
                # Create postconditions
                task_postconditions = []
                for post_data in task_data.get("postconditions", []):
                    post_id = f"post_{uuid.uuid4().hex[:8]}"
                    postcondition = Postcondition(
                        id=post_id,
                        description=post_data.get("description", ""),
                        condition=post_data.get("condition", ""),
                        achieved=False,
                        produced_by=task_id
                    )
                    postconditions.append(postcondition)
                    task_postconditions.append(post_id)
                
                task.postconditions = task_postconditions
                
                # Create resource requirements
                task_resource_requirements = []
                for req_data in task_data.get("resource_requirements", []):
                    req_id = f"req_{uuid.uuid4().hex[:8]}"
                    resource_name = req_data.get("resource_name", "")
                    if resource_name in resource_map:
                        requirement = ResourceRequirement(
                            id=req_id,
                            resource_id=resource_map[resource_name],
                            required_amount=float(req_data.get("required_amount", 1.0)),
                            task_id=task_id
                        )
                        ontology.add_resource_requirement(requirement)
                        task_resource_requirements.append(req_id)
                
                task.resource_requirements = task_resource_requirements
                
                # Create constraints
                task_constraints = []
                for const_data in task_data.get("constraints", []):
                    const_id = f"const_{uuid.uuid4().hex[:8]}"
                    constraint = Constraint(
                        id=const_id,
                        constraint_type=ConstraintType(const_data.get("type", "logical")),
                        condition=const_data.get("condition", ""),
                        severity=ConstraintSeverity(const_data.get("severity", "hard")),
                        violated=False,
                        applies_to=[task_id]
                    )
                    constraints.append(constraint)
                    task_constraints.append(const_id)
                
                task.constraints = task_constraints
                
                # Create state transitions
                task_state_transitions = []
                for trans_data in task_data.get("state_transitions", []):
                    trans_id = f"trans_{uuid.uuid4().hex[:8]}"
                    from_state_name = trans_data.get("from_state", "")
                    to_state_name = trans_data.get("to_state", "")
                    
                    # Create states if they don't exist
                    from_state_id = f"state_{from_state_name.replace(' ', '_').lower()}"
                    to_state_id = f"state_{to_state_name.replace(' ', '_').lower()}"
                    
                    if from_state_id not in ontology.states:
                        from_state = State(
                            id=from_state_id,
                            name=from_state_name,
                            state_type=StateType.SYSTEM,
                            value="initial"
                        )
                        ontology.add_state(from_state)
                    
                    if to_state_id not in ontology.states:
                        to_state = State(
                            id=to_state_id,
                            name=to_state_name,
                            state_type=StateType.SYSTEM,
                            value="target"
                        )
                        ontology.add_state(to_state)
                    
                    transition = StateTransition(
                        id=trans_id,
                        from_state=from_state_id,
                        to_state=to_state_id,
                        triggered_by=task_id,
                        condition=trans_data.get("condition")
                    )
                    state_transitions.append(transition)
                    task_state_transitions.append(trans_id)
                
                task.state_transitions = task_state_transitions
                
                tasks.append(task)
                task_map[task_data.get("name", "")] = task_id
                
                # Record dependencies
                for dep_name in task_data.get("dependencies", []):
                    if dep_name in task_map:
                        dependencies.append((task_id, task_map[dep_name]))
            
            return GoalDecomposition(
                goal=goal,
                subgoals=subgoals,
                tasks=tasks,
                preconditions=preconditions,
                postconditions=postconditions,
                dependencies=dependencies,
                constraints=constraints,
                resources=resources,
                state_transitions=state_transitions
            )
            
        except Exception as e:
            self.logger.error(f"Failed to decompose goal: {e}")
            return GoalDecomposition(
                goal=goal,
                subgoals=[],
                tasks=[],
                preconditions=[],
                postconditions=[],
                dependencies=[],
                constraints=[],
                resources=[],
                state_transitions=[]
            )
    
    def _link_preconditions_postconditions(self, ontology: SystemOntology):
        """Link preconditions and postconditions between tasks"""
        # For each postcondition, find preconditions it can satisfy
        for post_id, postcondition in ontology.postconditions.items():
            if postcondition.produced_by:
                # Find tasks that need this postcondition as a precondition
                for task_id, task in ontology.tasks.items():
                    if task_id != postcondition.produced_by:
                        for pre_id in task.preconditions:
                            if pre_id in ontology.preconditions:
                                pre = ontology.preconditions[pre_id]
                                # Simple matching - can be improved with semantic similarity
                                if (pre.description.lower() in postcondition.description.lower() or
                                    postcondition.description.lower() in pre.description.lower()):
                                    pre.satisfied_by = postcondition.produced_by
                                    postcondition.enables.append(task_id)
    
    def _update_task_statuses(self, ontology: SystemOntology, context: Dict[str, Any]):
        """Update task statuses based on executability"""
        for task_id, task in ontology.tasks.items():
            is_executable, _ = task.is_executable(
                ontology.preconditions,
                ontology.constraints,
                ontology.resources,
                ontology.resource_requirements,
                context
            )
            
            # Check if dependencies are satisfied
            deps_satisfied = True
            for dep_id in task.dependencies:
                if dep_id in ontology.tasks:
                    dep_task = ontology.tasks[dep_id]
                    if dep_task.status != TaskStatus.COMPLETED:
                        deps_satisfied = False
                        break
            
            if is_executable and deps_satisfied:
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.READY
            elif not deps_satisfied:
                task.status = TaskStatus.BLOCKED
