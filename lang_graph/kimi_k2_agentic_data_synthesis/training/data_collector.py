"""
Online Data Collector for Agentic Agent Trainer System

Collects real-time data from agent executions including tool calls, planning,
and result synthesis for online learning.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from collections import deque
from datetime import datetime
import uuid

from ..models.training import TrainingEpisode, ToolCallStep, PlanningStep, ResultSynthesis
from ..models.simulation import SimulationState, SimulationStep, StepType

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects real-time training data from agent executions.
    
    Responsibilities:
    - Monitor simulation engine for agent actions
    - Extract tool calls, planning steps, and results
    - Build training episodes from simulation data
    - Manage memory buffer for collected episodes
    """
    
    def __init__(
        self,
        buffer_size: int = 1000,
        max_episode_length: int = 100
    ):
        """
        Initialize the data collector.
        
        Args:
            buffer_size: Maximum number of episodes to keep in buffer
            max_episode_length: Maximum steps per episode
        """
        self.buffer_size = buffer_size
        self.max_episode_length = max_episode_length
        
        # Episode buffer (FIFO queue)
        self.episode_buffer: deque = deque(maxlen=buffer_size)
        
        # Active episodes being collected
        self.active_episodes: Dict[str, TrainingEpisode] = {}
        
        # Statistics
        self.total_episodes_collected = 0
        self.total_tool_calls_collected = 0
        self.total_planning_steps_collected = 0
        
        # Callbacks for data collection hooks
        self.on_tool_call_callbacks: List[Callable] = []
        self.on_planning_callbacks: List[Callable] = []
        self.on_episode_complete_callbacks: List[Callable] = []
        
        logger.info(f"DataCollector initialized with buffer_size={buffer_size}")
    
    def register_tool_call_callback(self, callback: Callable) -> None:
        """Register a callback for tool call events"""
        self.on_tool_call_callbacks.append(callback)
    
    def register_planning_callback(self, callback: Callable) -> None:
        """Register a callback for planning events"""
        self.on_planning_callbacks.append(callback)
    
    def register_episode_complete_callback(self, callback: Callable) -> None:
        """Register a callback for episode completion"""
        self.on_episode_complete_callbacks.append(callback)
    
    def start_episode(
        self,
        simulation_id: str,
        agent_id: str,
        user_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start collecting data for a new episode.
        
        Args:
            simulation_id: ID of the simulation
            agent_id: ID of the agent
            user_query: Initial user query
            context: Additional context
            
        Returns:
            Episode ID
        """
        episode = TrainingEpisode(
            episode_id=str(uuid.uuid4()),
            simulation_id=simulation_id,
            agent_id=agent_id,
            user_query=user_query,
            context=context or {}
        )
        
        self.active_episodes[simulation_id] = episode
        logger.debug(f"Started episode {episode.episode_id} for simulation {simulation_id}")
        
        return episode.episode_id
    
    def collect_tool_call(
        self,
        simulation_id: str,
        tool_name: str,
        tool_parameters: Dict[str, Any],
        tool_result: Optional[Dict[str, Any]] = None,
        success: bool = False,
        error_message: Optional[str] = None,
        execution_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Collect a tool call step.
        
        Args:
            simulation_id: ID of the simulation
            tool_name: Name of the tool called
            tool_parameters: Parameters passed to the tool
            tool_result: Result from tool execution
            success: Whether tool call was successful
            error_message: Error message if failed
            execution_time: Time taken for execution
            metadata: Additional metadata
            
        Returns:
            Step ID if collected, None otherwise
        """
        episode = self.active_episodes.get(simulation_id)
        if not episode:
            logger.warning(f"No active episode for simulation {simulation_id}")
            return None
        
        if episode.total_steps >= self.max_episode_length:
            logger.warning(f"Episode {episode.episode_id} exceeded max length")
            return None
        
        step = ToolCallStep(
            step_number=len(episode.tool_call_steps) + 1,
            tool_name=tool_name,
            tool_parameters=tool_parameters,
            tool_result=tool_result,
            success=success,
            error_message=error_message,
            execution_time=execution_time,
            metadata=metadata or {}
        )
        
        episode.add_tool_call_step(step)
        self.total_tool_calls_collected += 1
        
        # Notify callbacks
        for callback in self.on_tool_call_callbacks:
            try:
                callback(episode, step)
            except Exception as e:
                logger.error(f"Error in tool call callback: {e}")
        
        logger.debug(f"Collected tool call: {tool_name} for episode {episode.episode_id}")
        return step.step_id
    
    def collect_planning_step(
        self,
        simulation_id: str,
        plan_description: str,
        sub_goals: Optional[List[str]] = None,
        reasoning: Optional[str] = None,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Collect a planning step.
        
        Args:
            simulation_id: ID of the simulation
            plan_description: Description of the plan
            sub_goals: List of sub-goals
            reasoning: Reasoning behind the plan
            confidence: Confidence score
            metadata: Additional metadata
            
        Returns:
            Step ID if collected, None otherwise
        """
        episode = self.active_episodes.get(simulation_id)
        if not episode:
            logger.warning(f"No active episode for simulation {simulation_id}")
            return None
        
        if episode.total_steps >= self.max_episode_length:
            logger.warning(f"Episode {episode.episode_id} exceeded max length")
            return None
        
        step = PlanningStep(
            step_number=len(episode.planning_steps) + 1,
            plan_description=plan_description,
            sub_goals=sub_goals or [],
            reasoning=reasoning,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        episode.add_planning_step(step)
        self.total_planning_steps_collected += 1
        
        # Notify callbacks
        for callback in self.on_planning_callbacks:
            try:
                callback(episode, step)
            except Exception as e:
                logger.error(f"Error in planning callback: {e}")
        
        logger.debug(f"Collected planning step for episode {episode.episode_id}")
        return step.step_id
    
    def collect_result_synthesis(
        self,
        simulation_id: str,
        synthesized_result: str,
        source_tool_results: Optional[List[Dict[str, Any]]] = None,
        reasoning: Optional[str] = None,
        completeness: float = 0.0,
        accuracy: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Collect result synthesis step.
        
        Args:
            simulation_id: ID of the simulation
            synthesized_result: Final synthesized result
            source_tool_results: Results from tools used
            reasoning: Reasoning for synthesis
            completeness: Completeness score
            accuracy: Accuracy score
            metadata: Additional metadata
            
        Returns:
            Step ID if collected, None otherwise
        """
        episode = self.active_episodes.get(simulation_id)
        if not episode:
            logger.warning(f"No active episode for simulation {simulation_id}")
            return None
        
        synthesis = ResultSynthesis(
            step_number=len(episode.planning_steps) + len(episode.tool_call_steps) + 1,
            synthesized_result=synthesized_result,
            source_tool_results=source_tool_results or [],
            reasoning=reasoning,
            completeness=completeness,
            accuracy=accuracy,
            metadata=metadata or {}
        )
        
        episode.set_result_synthesis(synthesis)
        logger.debug(f"Collected result synthesis for episode {episode.episode_id}")
        return synthesis.step_id
    
    def complete_episode(
        self,
        simulation_id: str,
        final_outcome: Optional[Dict[str, Any]] = None,
        episode_duration: Optional[float] = None
    ) -> Optional[TrainingEpisode]:
        """
        Complete an episode and add it to the buffer.
        
        Args:
            simulation_id: ID of the simulation
            final_outcome: Final outcome of the episode
            episode_duration: Duration of the episode
            
        Returns:
            Completed episode or None if not found
        """
        episode = self.active_episodes.pop(simulation_id, None)
        if not episode:
            logger.warning(f"No active episode for simulation {simulation_id}")
            return None
        
        episode.final_outcome = final_outcome
        if episode_duration is not None:
            episode.episode_duration = episode_duration
        else:
            # Calculate duration from timestamps
            if episode.planning_steps or episode.tool_call_steps:
                start_time = min(
                    (step.timestamp for step in episode.planning_steps + episode.tool_call_steps),
                    default=datetime.utcnow()
                )
                end_time = max(
                    (step.timestamp for step in episode.planning_steps + episode.tool_call_steps),
                    default=datetime.utcnow()
                )
                episode.episode_duration = (end_time - start_time).total_seconds()
        
        # Add to buffer
        self.episode_buffer.append(episode)
        self.total_episodes_collected += 1
        
        # Notify callbacks
        for callback in self.on_episode_complete_callbacks:
            try:
                callback(episode)
            except Exception as e:
                logger.error(f"Error in episode complete callback: {e}")
        
        logger.info(f"Completed episode {episode.episode_id} with {episode.total_steps} steps")
        return episode
    
    def collect_from_simulation_state(
        self,
        simulation_state: Dict[str, Any],
        agent_id: str
    ) -> Optional[TrainingEpisode]:
        """
        Collect episode data from a simulation state dictionary.
        
        Args:
            simulation_state: LangGraph simulation state
            agent_id: ID of the agent
            
        Returns:
            Collected episode or None
        """
        simulation_id = simulation_state.get("simulation_id")
        user_query = simulation_state.get("user_query", "")
        
        if not simulation_id:
            logger.warning("Simulation state missing simulation_id")
            return None
        
        # Start episode if not already started
        if simulation_id not in self.active_episodes:
            self.start_episode(
                simulation_id=simulation_id,
                agent_id=agent_id,
                user_query=user_query,
                context=simulation_state.get("environment_state", {})
            )
        
        episode = self.active_episodes.get(simulation_id)
        if not episode:
            return None
        
        # Collect tool calls from tool_results
        tool_results = simulation_state.get("tool_results", [])
        for tool_result in tool_results:
            self.collect_tool_call(
                simulation_id=simulation_id,
                tool_name=tool_result.get("tool_name", "unknown"),
                tool_parameters=tool_result.get("parameters", {}),
                tool_result=tool_result.get("output", {}),
                success=tool_result.get("status") == "success",
                error_message=tool_result.get("error"),
                execution_time=tool_result.get("execution_time", 0.0),
                metadata=tool_result
            )
        
        # Collect planning steps from messages
        messages = simulation_state.get("messages", [])
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str) and any(keyword in content.lower() for keyword in ["plan", "strategy", "approach"]):
                self.collect_planning_step(
                    simulation_id=simulation_id,
                    plan_description=content,
                    reasoning=content,
                    confidence=0.5,
                    metadata={"message_index": messages.index(message)}
                )
        
        # Collect result synthesis from final_outcome
        final_outcome = simulation_state.get("final_outcome")
        if final_outcome:
            self.collect_result_synthesis(
                simulation_id=simulation_id,
                synthesized_result=str(final_outcome),
                source_tool_results=tool_results,
                completeness=0.8,
                accuracy=0.8,
                metadata=final_outcome
            )
        
        return episode
    
    def get_episodes(
        self,
        limit: Optional[int] = None,
        agent_id: Optional[str] = None,
        min_steps: int = 0
    ) -> List[TrainingEpisode]:
        """
        Get episodes from the buffer.
        
        Args:
            limit: Maximum number of episodes to return
            agent_id: Filter by agent ID
            min_steps: Minimum number of steps per episode
            
        Returns:
            List of episodes
        """
        episodes = list(self.episode_buffer)
        
        # Filter by agent_id
        if agent_id:
            episodes = [ep for ep in episodes if ep.agent_id == agent_id]
        
        # Filter by min_steps
        episodes = [ep for ep in episodes if ep.total_steps >= min_steps]
        
        # Limit results
        if limit:
            episodes = episodes[:limit]
        
        return episodes
    
    def clear_buffer(self) -> int:
        """
        Clear the episode buffer.
        
        Returns:
            Number of episodes cleared
        """
        count = len(self.episode_buffer)
        self.episode_buffer.clear()
        logger.info(f"Cleared {count} episodes from buffer")
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "total_episodes_collected": self.total_episodes_collected,
            "total_tool_calls_collected": self.total_tool_calls_collected,
            "total_planning_steps_collected": self.total_planning_steps_collected,
            "buffer_size": len(self.episode_buffer),
            "active_episodes": len(self.active_episodes),
            "buffer_capacity": self.buffer_size
        }


