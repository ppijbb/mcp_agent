"""
Simulation models for the Kimi-K2 Agentic Data Synthesis System
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime


class SimulationStatus(str, Enum):
    """Status of simulation sessions"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(str, Enum):
    """Status of simulation steps"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepType(str, Enum):
    """Types of simulation steps"""
    USER_INPUT = "user_input"
    AGENT_ACTION = "agent_action"
    TOOL_USAGE = "tool_usage"
    ENVIRONMENT_CHANGE = "environment_change"
    EVALUATION = "evaluation"


class SimulationState(TypedDict):
    """
    Represents the state of a Kimi-K2 simulation within LangGraph.
    This state will be passed between nodes in the graph.
    """
    simulation_id: Annotated[str, "Unique identifier for the simulation session"]
    user_query: Annotated[str, "The initial query or problem statement from the user"]
    messages: Annotated[List[Dict[str, Any]], "History of messages exchanged between agents and environment"]
    current_agents: Annotated[List[str], "IDs of agents currently active in the simulation"]
    environment_state: Annotated[Dict[str, Any], "Current state of the virtual environment"]
    tool_results: Annotated[List[Dict[str, Any]], "Results of tool usages in the current step/turn"]
    final_outcome: Annotated[Optional[Dict[str, Any]], "The final outcome or solution of the simulation"]
    status: Annotated[str, "Current status of the simulation (e.g., running, completed, failed)"]
    error_message: Annotated[Optional[str], "Any error message if the simulation fails"]
    max_turns: Annotated[int, "Maximum number of turns for the simulation"]
    timeout: Annotated[int, "Timeout in seconds for the simulation"]
    # You might also want to add fields for evaluation results, generated data, etc.
    # if they need to be passed as part of the *intermediate* graph state.


class SimulationStep(BaseModel):
    """Individual step in a simulation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int
    step_type: StepType
    description: str
    agent_id: Optional[str] = None
    tool_used: Optional[str] = None
    input_data: Dict[str, Any] = {}
    output_data: Dict[str, Any] = {}
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None  # in seconds
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "step_number": 1,
                "step_type": "user_input",
                "description": "User requests help with network issue",
                "input_data": {"message": "My internet is not working"},
                "status": "completed"
            }
        }
    
    def start(self) -> None:
        """Start the step execution"""
        self.status = StepStatus.IN_PROGRESS
        self.start_time = datetime.utcnow()
    
    def complete(self, output: Dict[str, Any] = None) -> None:
        """Complete the step execution"""
        self.status = StepStatus.COMPLETED
        self.end_time = datetime.utcnow()
        if output:
            self.output_data = output
        
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
    
    def fail(self, error: str) -> None:
        """Mark step as failed"""
        self.status = StepStatus.FAILED
        self.end_time = datetime.utcnow()
        self.error_message = error
        
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()


class EnvironmentState(BaseModel):
    """State of the simulation environment"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    current_step: int = 0
    environment_variables: Dict[str, Any] = {}
    available_tools: List[str] = []
    active_agents: List[str] = []
    user_context: Dict[str, Any] = {}
    system_state: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_123",
                "current_step": 3,
                "environment_variables": {"mode": "debug", "level": "intermediate"},
                "available_tools": ["web_search", "file_search"],
                "active_agents": ["agent_1", "agent_2"],
                "user_context": {"issue_type": "network", "urgency": "high"}
            }
        }
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update environment state"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.timestamp = datetime.utcnow()
    
    def add_tool(self, tool_id: str) -> None:
        """Add a tool to available tools"""
        if tool_id not in self.available_tools:
            self.available_tools.append(tool_id)
            self.timestamp = datetime.utcnow()
    
    def remove_tool(self, tool_id: str) -> None:
        """Remove a tool from available tools"""
        if tool_id in self.available_tools:
            self.available_tools.remove(tool_id)
            self.timestamp = datetime.utcnow()
    
    def add_agent(self, agent_id: str) -> None:
        """Add an agent to active agents"""
        if agent_id not in self.active_agents:
            self.active_agents.append(agent_id)
            self.timestamp = datetime.utcnow()
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from active agents"""
        if agent_id in self.active_agents:
            self.active_agents.remove(agent_id)
            self.timestamp = datetime.utcnow()


class EnvironmentConfig(BaseModel):
    """Configuration for simulation environment"""
    environment_type: str = "default"
    resources: Dict[str, str] = {}
    tools_available: List[str] = []
    constraints: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "environment_type": "development_workspace",
                "resources": {"memory": "8GB", "cpu": "4 cores"},
                "tools_available": ["code_editor", "terminal", "git"],
                "constraints": {"max_execution_time": 300}
            }
        }


class SimulationConfig(BaseModel):
    """Configuration for creating simulations"""
    simulation_id: str
    name: str
    description: str
    agent_configs: List[Any] = []  # List of AgentConfig
    environment_config: EnvironmentConfig
    user_agent_config: Optional[Any] = None  # Optional UserAgentConfig
    max_turns: int = 20
    timeout: int = 600  # seconds
    scenario: str = ""
    
    class Config:
        json_schema_extra = {
            "example": {
                "simulation_id": "web_dev_collaboration",
                "name": "Web Development Collaboration",
                "description": "Senior and junior developers collaborating on a React project",
                "agent_configs": [],
                "environment_config": {},
                "max_turns": 20,
                "timeout": 600,
                "scenario": "Create a responsive React component with proper error handling"
            }
        }


class SimulationSession(BaseModel):
    """Complete simulation session"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    domain_id: str
    scenario_id: str
    agent_ids: List[str] = []
    user_agent_id: Optional[str] = None
    status: SimulationStatus = SimulationStatus.PENDING
    steps: List[SimulationStep] = []
    environment_states: List[EnvironmentState] = []
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None  # in seconds
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "domain_id": "tech_support",
                "scenario_id": "network_troubleshooting",
                "agent_ids": ["agent_1", "agent_2"],
                "user_agent_id": "user_1",
                "status": "running",
                "total_steps": 5,
                "completed_steps": 3
            }
        }
    
    def start(self) -> None:
        """Start the simulation session"""
        self.status = SimulationStatus.RUNNING
        self.start_time = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def complete(self, quality_score: Optional[float] = None) -> None:
        """Complete the simulation session"""
        self.status = SimulationStatus.COMPLETED
        self.end_time = datetime.utcnow()
        if quality_score is not None:
            self.quality_score = quality_score
        
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        
        self.updated_at = datetime.utcnow()
    
    def fail(self, error_message: str) -> None:
        """Mark simulation as failed"""
        self.status = SimulationStatus.FAILED
        self.end_time = datetime.utcnow()
        self.metadata["error_message"] = error_message
        
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        
        self.updated_at = datetime.utcnow()
    
    def add_step(self, step: SimulationStep) -> None:
        """Add a step to the simulation"""
        step.step_number = len(self.steps) + 1
        self.steps.append(step)
        self.total_steps = len(self.steps)
        self.updated_at = datetime.utcnow()
    
    def add_environment_state(self, state: EnvironmentState) -> None:
        """Add an environment state to the simulation"""
        state.session_id = self.id
        self.environment_states.append(state)
        self.updated_at = datetime.utcnow()
    
    def get_current_state(self) -> Optional[EnvironmentState]:
        """Get the current environment state"""
        if self.environment_states:
            return self.environment_states[-1]
        return None
    
    def get_step_by_number(self, step_number: int) -> Optional[SimulationStep]:
        """Get a step by its number"""
        for step in self.steps:
            if step.step_number == step_number:
                return step
        return None
    
    def get_completed_steps(self) -> List[SimulationStep]:
        """Get all completed steps"""
        return [step for step in self.steps if step.status == StepStatus.COMPLETED]
    
    def get_failed_steps(self) -> List[SimulationStep]:
        """Get all failed steps"""
        return [step for step in self.steps if step.status == StepStatus.FAILED]
    
    def update_progress(self) -> None:
        """Update progress statistics"""
        self.completed_steps = len(self.get_completed_steps())
        self.failed_steps = len(self.get_failed_steps())
        self.updated_at = datetime.utcnow()
    
    def get_success_rate(self) -> float:
        """Calculate success rate of the simulation"""
        if self.total_steps == 0:
            return 0.0
        return self.completed_steps / self.total_steps 