"""
Training models for the Agentic Agent Trainer System

Defines data structures for online learning, including episodes, batches,
rewards, and model checkpoints.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime


class TrainingAlgorithm(str, Enum):
    """Supported training algorithms"""
    DPO = "dpo"
    GRPO = "grpo"
    PPO = "ppo"
    SUPERVISED = "supervised"


class RewardSource(str, Enum):
    """Sources of reward signals"""
    AGENT_SELF_EVALUATION = "agent_self_evaluation"
    LLM_JUDGE = "llm_judge"
    HUMAN_FEEDBACK = "human_feedback"
    EXTERNAL_METRIC = "external_metric"
    COMPOSITE = "composite"


class ModelStatus(str, Enum):
    """Model checkpoint status"""
    TRAINING = "training"
    EVALUATING = "evaluating"
    READY = "ready"
    ARCHIVED = "archived"


class ToolCallStep(BaseModel):
    """Single tool call step in an episode"""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int
    tool_name: str
    tool_parameters: Dict[str, Any] = {}
    tool_result: Optional[Dict[str, Any]] = None
    success: bool = False
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}


class PlanningStep(BaseModel):
    """Planning step in an episode"""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int
    plan_description: str
    sub_goals: List[str] = []
    reasoning: Optional[str] = None
    confidence: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}


class ResultSynthesis(BaseModel):
    """Result synthesis step"""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int
    synthesized_result: str
    source_tool_results: List[Dict[str, Any]] = []
    reasoning: Optional[str] = None
    completeness: float = 0.0
    accuracy: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}


class TrainingEpisode(BaseModel):
    """Single training episode containing tool calls, planning, and results"""
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    simulation_id: str
    agent_id: str
    user_query: str
    context: Dict[str, Any] = {}
    
    # Episode components
    planning_steps: List[PlanningStep] = []
    tool_call_steps: List[ToolCallStep] = []
    result_synthesis: Optional[ResultSynthesis] = None
    
    # Episode metadata
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    episode_duration: float = 0.0
    final_outcome: Optional[Dict[str, Any]] = None
    
    # Training metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = False
    processed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "episode_id": "ep_123",
                "simulation_id": "sim_456",
                "agent_id": "agent_789",
                "user_query": "Create a React component",
                "planning_steps": [],
                "tool_call_steps": [],
                "total_steps": 5,
                "successful_steps": 4,
                "failed_steps": 1
            }
        }
    
    def add_planning_step(self, step: PlanningStep) -> None:
        """Add a planning step to the episode"""
        step.step_number = len(self.planning_steps) + 1
        self.planning_steps.append(step)
        self.total_steps = len(self.planning_steps) + len(self.tool_call_steps)
        if self.result_synthesis:
            self.total_steps += 1
    
    def add_tool_call_step(self, step: ToolCallStep) -> None:
        """Add a tool call step to the episode"""
        step.step_number = len(self.tool_call_steps) + 1
        self.tool_call_steps.append(step)
        self.total_steps = len(self.planning_steps) + len(self.tool_call_steps)
        if self.result_synthesis:
            self.total_steps += 1
        
        if step.success:
            self.successful_steps += 1
        else:
            self.failed_steps += 1
    
    def set_result_synthesis(self, synthesis: ResultSynthesis) -> None:
        """Set the result synthesis for the episode"""
        synthesis.step_number = len(self.planning_steps) + len(self.tool_call_steps) + 1
        self.result_synthesis = synthesis
        self.total_steps = len(self.planning_steps) + len(self.tool_call_steps) + 1
    
    def calculate_success_rate(self) -> float:
        """Calculate success rate of the episode"""
        if self.total_steps == 0:
            return 0.0
        return self.successful_steps / self.total_steps
    
    def get_episode_sequence(self) -> List[Union[PlanningStep, ToolCallStep, ResultSynthesis]]:
        """Get all steps in chronological order"""
        sequence = []
        sequence.extend(self.planning_steps)
        sequence.extend(self.tool_call_steps)
        if self.result_synthesis:
            sequence.append(self.result_synthesis)
        return sorted(sequence, key=lambda x: x.step_number)


class RewardSignal(BaseModel):
    """Reward signal for training"""
    reward_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    episode_id: str
    agent_id: str
    
    # Reward components
    overall_reward: float = 0.0
    tool_selection_reward: float = 0.0
    planning_quality_reward: float = 0.0
    result_synthesis_reward: float = 0.0
    
    # Reward metadata
    reward_source: RewardSource
    reward_components: Dict[str, float] = {}
    reward_weights: Dict[str, float] = {}
    
    # Evaluation details
    evaluation_details: Dict[str, Any] = {}
    evaluator_id: Optional[str] = None
    evaluation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Normalization
    normalized_reward: float = 0.0
    reward_range: tuple = (-1.0, 1.0)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "episode_id": "ep_123",
                "agent_id": "agent_789",
                "overall_reward": 0.85,
                "tool_selection_reward": 0.9,
                "planning_quality_reward": 0.8,
                "result_synthesis_reward": 0.85,
                "reward_source": "agent_self_evaluation",
                "reward_components": {
                    "tool_accuracy": 0.9,
                    "planning_coherence": 0.8,
                    "synthesis_quality": 0.85
                }
            }
        }
    
    def calculate_overall_reward(self) -> float:
        """Calculate overall reward from components"""
        if not self.reward_weights:
            # Default equal weights
            self.reward_weights = {
                "tool_selection": 0.33,
                "planning_quality": 0.33,
                "result_synthesis": 0.34
            }
        
        self.overall_reward = (
            self.tool_selection_reward * self.reward_weights.get("tool_selection", 0.33) +
            self.planning_quality_reward * self.reward_weights.get("planning_quality", 0.33) +
            self.result_synthesis_reward * self.reward_weights.get("result_synthesis", 0.34)
        )
        return self.overall_reward
    
    def normalize_reward(self, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """Normalize reward to specified range"""
        self.reward_range = (min_val, max_val)
        # Simple linear normalization (can be enhanced)
        if self.overall_reward >= 0:
            self.normalized_reward = self.overall_reward * max_val
        else:
            self.normalized_reward = self.overall_reward * abs(min_val)
        return self.normalized_reward


class TrainingBatch(BaseModel):
    """Batch of training episodes for learning"""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    batch_name: str
    algorithm: TrainingAlgorithm
    
    # Batch data
    episodes: List[TrainingEpisode] = []
    rewards: List[RewardSignal] = []
    
    # Batch metadata
    batch_size: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = False
    processed_at: Optional[datetime] = None
    
    # Statistics
    average_reward: float = 0.0
    average_episode_length: float = 0.0
    success_rate: float = 0.0
    
    metadata: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "batch_name": "training_batch_001",
                "algorithm": "ppo",
                "batch_size": 32,
                "average_reward": 0.75,
                "success_rate": 0.85
            }
        }
    
    def add_episode(self, episode: TrainingEpisode) -> None:
        """Add an episode to the batch"""
        self.episodes.append(episode)
        self.batch_size = len(self.episodes)
    
    def add_reward(self, reward: RewardSignal) -> None:
        """Add a reward signal to the batch"""
        self.rewards.append(reward)
    
    def calculate_statistics(self) -> Dict[str, float]:
        """Calculate batch statistics"""
        if not self.episodes:
            return {}
        
        # Calculate average episode length
        total_steps = sum(ep.total_steps for ep in self.episodes)
        self.average_episode_length = total_steps / len(self.episodes)
        
        # Calculate success rate
        total_successful = sum(ep.successful_steps for ep in self.episodes)
        total_steps_all = sum(ep.total_steps for ep in self.episodes)
        if total_steps_all > 0:
            self.success_rate = total_successful / total_steps_all
        
        # Calculate average reward
        if self.rewards:
            self.average_reward = sum(r.overall_reward for r in self.rewards) / len(self.rewards)
        
        return {
            "average_episode_length": self.average_episode_length,
            "success_rate": self.success_rate,
            "average_reward": self.average_reward,
            "batch_size": self.batch_size
        }
    
    def get_episode_reward_pairs(self) -> List[tuple]:
        """Get (episode, reward) pairs for training"""
        reward_map = {r.episode_id: r for r in self.rewards}
        pairs = []
        for episode in self.episodes:
            reward = reward_map.get(episode.episode_id)
            if reward:
                pairs.append((episode, reward))
        return pairs


class ModelCheckpoint(BaseModel):
    """Model checkpoint metadata"""
    checkpoint_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str
    model_path: str
    model_version: str
    
    # Training metadata
    algorithm: TrainingAlgorithm
    training_step: int = 0
    epoch: int = 0
    
    # Performance metrics
    training_loss: float = 0.0
    validation_loss: Optional[float] = None
    average_reward: float = 0.0
    evaluation_metrics: Dict[str, float] = {}
    
    # Model state
    status: ModelStatus = ModelStatus.TRAINING
    is_best: bool = False
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    model_size_mb: Optional[float] = None
    parameters_count: Optional[int] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "agentic_agent_v1",
                "model_path": "/path/to/checkpoint",
                "model_version": "1.0.0",
                "algorithm": "ppo",
                "training_step": 1000,
                "epoch": 5,
                "training_loss": 0.15,
                "validation_loss": 0.18,
                "average_reward": 0.85,
                "status": "ready"
            }
        }
    
    def mark_as_best(self) -> None:
        """Mark this checkpoint as the best model"""
        self.is_best = True
        self.status = ModelStatus.READY
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update evaluation metrics"""
        self.evaluation_metrics.update(metrics)
        if "validation_loss" in metrics:
            self.validation_loss = metrics["validation_loss"]
        if "average_reward" in metrics:
            self.average_reward = metrics["average_reward"]


class TrainingConfig(BaseModel):
    """Configuration for training"""
    config_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    algorithm: TrainingAlgorithm
    
    # Model configuration
    model_name: str
    base_model: str
    model_type: str = "causal_lm"
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    warmup_steps: int = 100
    
    # Algorithm-specific parameters
    algorithm_params: Dict[str, Any] = {}
    
    # Reward configuration
    reward_weights: Dict[str, float] = {
        "tool_selection": 0.33,
        "planning_quality": 0.33,
        "result_synthesis": 0.34
    }
    reward_source: RewardSource = RewardSource.AGENT_SELF_EVALUATION
    
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Output configuration
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "algorithm": "ppo",
                "model_name": "agentic_agent",
                "base_model": "gemini-2.5-flash",
                "learning_rate": 1e-5,
                "batch_size": 32,
                "num_epochs": 3,
                "reward_source": "agent_self_evaluation"
            }
        }

