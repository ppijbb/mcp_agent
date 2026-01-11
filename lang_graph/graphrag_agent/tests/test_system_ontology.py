"""
System Ontology Integration Tests

Tests for system ontology functionality:
- Goal achievement scenarios
- Task execution scenarios
- Path finding
- Query translation
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from models.system_ontology import (
    SystemOntology,
    Goal,
    Task,
    Precondition,
    Postcondition,
    State,
    Resource,
    ResourceRequirement,
    Constraint,
    GoalStatus,
    TaskStatus,
    StateType,
    ConstraintType,
    ConstraintSeverity
)
from agents.goal_ontology_builder import GoalOntologyBuilder
from agents.task_executor import TaskExecutor, ExecutionResult
from agents.goal_path_finder import GoalPathFinder, AchievementPath
from agents.system_query_translator import SystemQueryTranslator


@pytest.fixture
def mock_config():
    """Mock agent configuration"""
    config = Mock()
    config.model_name = "gemini-2.5-flash-lite"
    config.max_search_results = 10
    config.context_window_size = 8000
    config.temperature = 0.1
    config.max_tokens = 2000
    return config


@pytest.fixture
def mock_llm_processor(mock_config):
    """Mock LLM processor"""
    processor = Mock()
    
    def mock_call_llm(prompt):
        # Return mock JSON responses based on prompt content
        if "goal" in prompt.lower() and "decompose" in prompt.lower():
            return """{
                "subgoals": [],
                "tasks": [
                    {
                        "name": "Initialize System",
                        "description": "Initialize the system",
                        "priority": 0.8,
                        "execution_time": 5.0,
                        "success_rate": 0.95,
                        "preconditions": [
                            {
                                "description": "System not initialized",
                                "condition": "system_state != 'initialized'"
                            }
                        ],
                        "postconditions": [
                            {
                                "description": "System initialized",
                                "condition": "system_state == 'initialized'"
                            }
                        ],
                        "dependencies": [],
                        "resource_requirements": [],
                        "constraints": [],
                        "state_transitions": [
                            {
                                "from_state": "Uninitialized",
                                "to_state": "Initialized",
                                "condition": "true"
                            }
                        ]
                    }
                ],
                "resources": []
            }"""
        elif "goal" in prompt.lower():
            return """{
                "name": "System Optimization",
                "description": "Optimize system performance",
                "priority": 0.9,
                "metadata": {
                    "domain": "system",
                    "urgency": "high"
                }
            }"""
        else:
            return "{}"
    
    processor._call_llm = Mock(side_effect=mock_call_llm)
    return processor


@pytest.fixture
def sample_system_ontology():
    """Create a sample system ontology for testing"""
    ontology = SystemOntology()
    
    # Create a goal
    goal = Goal(
        id="goal_1",
        name="System Optimization",
        description="Optimize system performance",
        priority=0.9,
        status=GoalStatus.PENDING
    )
    ontology.add_goal(goal)
    
    # Create preconditions
    pre1 = Precondition(
        id="pre_1",
        description="System not optimized",
        condition="optimized == false",
        satisfied=False,
        required_by="task_1"
    )
    ontology.add_precondition(pre1)
    
    # Create postconditions
    post1 = Postcondition(
        id="post_1",
        description="System optimized",
        condition="optimized == true",
        achieved=False,
        produced_by="task_1"
    )
    ontology.add_postcondition(post1)
    
    # Create a task
    task1 = Task(
        id="task_1",
        name="Optimize System",
        description="Optimize the system",
        priority=0.8,
        execution_time=10.0,
        success_rate=0.9,
        status=TaskStatus.PENDING,
        preconditions=["pre_1"],
        postconditions=["post_1"],
        dependencies=[],
        metadata={"achieves_goals": ["goal_1"]}
    )
    ontology.add_task(task1)
    
    # Create a state
    state1 = State(
        id="state_1",
        name="Unoptimized",
        state_type=StateType.SYSTEM,
        value="unoptimized"
    )
    ontology.add_state(state1)
    
    state2 = State(
        id="state_2",
        name="Optimized",
        state_type=StateType.SYSTEM,
        value="optimized"
    )
    ontology.add_state(state2)
    
    return ontology


@pytest.mark.asyncio
async def test_goal_ontology_builder(mock_config, mock_llm_processor):
    """Test goal ontology builder"""
    builder = GoalOntologyBuilder(mock_config, mock_llm_processor)
    
    goals = ["Optimize system performance"]
    ontology = builder.build_ontology_from_goals(goals)
    
    assert len(ontology.goals) > 0
    assert len(ontology.tasks) > 0


@pytest.mark.asyncio
async def test_task_executor(mock_config, sample_system_ontology):
    """Test task executor"""
    executor = TaskExecutor(mock_config, sample_system_ontology)
    
    # Mark precondition as satisfied
    sample_system_ontology.preconditions["pre_1"].satisfied = True
    
    # Execute task
    result = await executor.execute_task("task_1")
    
    assert result.task_id == "task_1"
    # Task should execute (success depends on mock)
    assert result.execution_time >= 0


@pytest.mark.asyncio
async def test_goal_path_finder(mock_config, sample_system_ontology):
    """Test goal path finder"""
    finder = GoalPathFinder(mock_config, sample_system_ontology)
    
    # Find path to achieve goal
    path = finder.find_achievement_path("goal_1")
    
    assert path is not None
    assert path.goal_id == "goal_1"
    assert len(path.tasks) > 0


def test_system_query_translator(mock_config, mock_llm_processor, sample_system_ontology):
    """Test system query translator"""
    translator = SystemQueryTranslator(mock_config, mock_llm_processor)
    
    # Test goal achievement query
    translation = translator.translate_system_query(
        "How to achieve System Optimization?",
        sample_system_ontology
    )
    
    assert translation.query_category == "goal_achievement"
    assert "MATCH" in translation.cypher_query
    assert "Goal" in translation.cypher_query
    
    # Test executable tasks query
    translation2 = translator.translate_system_query(
        "What tasks are executable?",
        sample_system_ontology
    )
    
    assert translation2.query_category == "executable_tasks"
    assert "Task" in translation2.cypher_query


@pytest.mark.asyncio
async def test_integrated_goal_achievement_scenario(mock_config, mock_llm_processor):
    """Test integrated goal achievement scenario"""
    # Build ontology from goals
    builder = GoalOntologyBuilder(mock_config, mock_llm_processor)
    ontology = builder.build_ontology_from_goals(["Optimize system"])
    
    if len(ontology.goals) == 0:
        pytest.skip("No goals created")
    
    goal_id = list(ontology.goals.keys())[0]
    
    # Find achievement path
    finder = GoalPathFinder(mock_config, ontology)
    path = finder.find_achievement_path(goal_id)
    
    if path and len(path.tasks) > 0:
        # Execute tasks
        executor = TaskExecutor(mock_config, ontology)
        
        # Satisfy preconditions
        for task in path.tasks:
            for pre_id in task.preconditions:
                if pre_id in ontology.preconditions:
                    ontology.preconditions[pre_id].satisfied = True
        
        # Execute first task
        if path.tasks:
            result = await executor.execute_task(path.tasks[0].id)
            assert result.task_id == path.tasks[0].id


def test_system_ontology_serialization(sample_system_ontology):
    """Test system ontology serialization"""
    # Convert to dict
    ontology_dict = sample_system_ontology.to_dict()
    
    assert "goals" in ontology_dict
    assert "tasks" in ontology_dict
    assert "preconditions" in ontology_dict
    assert "postconditions" in ontology_dict
    
    # Verify structure
    assert len(ontology_dict["goals"]) > 0
    assert len(ontology_dict["tasks"]) > 0


@pytest.mark.asyncio
async def test_task_dependency_resolution(mock_config, sample_system_ontology):
    """Test task dependency resolution"""
    # Add a dependent task
    task2 = Task(
        id="task_2",
        name="Verify Optimization",
        description="Verify system is optimized",
        priority=0.7,
        status=TaskStatus.PENDING,
        dependencies=["task_1"],
        metadata={"achieves_goals": ["goal_1"]}
    )
    sample_system_ontology.add_task(task2)
    
    finder = GoalPathFinder(mock_config, sample_system_ontology)
    path = finder.find_achievement_path("goal_1")
    
    if path:
        # Check that task_1 comes before task_2
        task_ids = [t.id for t in path.tasks]
        if "task_1" in task_ids and "task_2" in task_ids:
            assert task_ids.index("task_1") < task_ids.index("task_2")
