"""
Reward System for Agentic Agent Trainer System

Calculates rewards for training episodes using agent self-evaluation,
LLM judges, or external metrics.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from ..models.training import (
    TrainingEpisode,
    RewardSignal,
    RewardSource
)
from ..evaluation.llm_judge import LLMJudgeSystem

logger = logging.getLogger(__name__)


class RewardSystem:
    """
    Calculates rewards for training episodes.
    
    Responsibilities:
    - Agent self-evaluation: Agent evaluates its own performance
    - LLM Judge evaluation: External LLM evaluates performance
    - Multi-dimensional rewards: Tool selection, planning, synthesis
    - Reward normalization and shaping
    """
    
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        default_reward_source: RewardSource = RewardSource.AGENT_SELF_EVALUATION,
        reward_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the reward system.
        
        Args:
            llm_config: Configuration for LLM judge
            default_reward_source: Default source for rewards
            reward_weights: Weights for different reward components
        """
        self.llm_config = llm_config
        self.default_reward_source = default_reward_source
        self.reward_weights = reward_weights or {
            "tool_selection": 0.33,
            "planning_quality": 0.33,
            "result_synthesis": 0.34
        }
        
        # Initialize LLM judge if config provided
        self.llm_judge = None
        if llm_config:
            try:
                self.llm_judge = LLMJudgeSystem()
            except Exception as e:
                logger.warning(f"Failed to initialize LLM judge: {e}")
        
        # Custom reward calculators
        self.custom_calculators: Dict[str, Callable] = {}
        
        logger.info(f"RewardSystem initialized with default source: {default_reward_source}")
    
    def register_custom_calculator(
        self,
        name: str,
        calculator: Callable[[TrainingEpisode], Dict[str, float]]
    ) -> None:
        """Register a custom reward calculator"""
        self.custom_calculators[name] = calculator
        logger.info(f"Registered custom reward calculator: {name}")
    
    def calculate_agent_self_evaluation(
        self,
        episode: TrainingEpisode
    ) -> Dict[str, float]:
        """
        Calculate reward using agent self-evaluation.
        
        The agent evaluates its own performance based on:
        - Tool selection accuracy
        - Planning quality
        - Result synthesis quality
        
        Args:
            episode: Training episode
            
        Returns:
            Dictionary of reward components
        """
        rewards = {
            "tool_selection": 0.0,
            "planning_quality": 0.0,
            "result_synthesis": 0.0
        }
        
        # Tool selection reward: based on success rate
        if episode.tool_call_steps:
            successful_tools = sum(1 for step in episode.tool_call_steps if step.success)
            total_tools = len(episode.tool_call_steps)
            rewards["tool_selection"] = successful_tools / total_tools if total_tools > 0 else 0.0
        else:
            rewards["tool_selection"] = 0.5  # Neutral if no tools used
        
        # Planning quality reward: based on planning steps and coherence
        if episode.planning_steps:
            # Average confidence of planning steps
            avg_confidence = sum(step.confidence for step in episode.planning_steps) / len(episode.planning_steps)
            # Bonus for having sub-goals
            sub_goals_bonus = 0.1 if any(step.sub_goals for step in episode.planning_steps) else 0.0
            rewards["planning_quality"] = min(1.0, avg_confidence + sub_goals_bonus)
        else:
            rewards["planning_quality"] = 0.3  # Lower score if no planning
        
        # Result synthesis reward: based on completeness and accuracy
        if episode.result_synthesis:
            completeness = episode.result_synthesis.completeness
            accuracy = episode.result_synthesis.accuracy
            rewards["result_synthesis"] = (completeness + accuracy) / 2.0
        elif episode.final_outcome:
            # If we have final outcome but no synthesis, give moderate score
            rewards["result_synthesis"] = 0.6
        else:
            rewards["result_synthesis"] = 0.2  # Low score if no result
        
        # Overall success rate bonus
        success_rate = episode.calculate_success_rate()
        if success_rate > 0.8:
            # Boost all rewards if high success rate
            for key in rewards:
                rewards[key] = min(1.0, rewards[key] * 1.1)
        
        return rewards
    
    async def calculate_llm_judge_evaluation(
        self,
        episode: TrainingEpisode
    ) -> Dict[str, float]:
        """
        Calculate reward using LLM judge evaluation.
        
        Args:
            episode: Training episode
            
        Returns:
            Dictionary of reward components
        """
        if not self.llm_judge:
            logger.warning("LLM judge not available, falling back to self-evaluation")
            return self.calculate_agent_self_evaluation(episode)
        
        rewards = {
            "tool_selection": 0.0,
            "planning_quality": 0.0,
            "result_synthesis": 0.0
        }
        
        try:
            # Format episode for evaluation
            evaluation_prompt = self._format_episode_for_evaluation(episode)
            
            # Use LLM judge to evaluate (simplified - would need proper integration)
            # For now, use a heuristic based on episode characteristics
            # In production, this would call the LLM judge system
            
            # Tool selection evaluation
            if episode.tool_call_steps:
                tool_success_rate = sum(1 for s in episode.tool_call_steps if s.success) / len(episode.tool_call_steps)
                rewards["tool_selection"] = tool_success_rate * 0.9  # Slightly more conservative than self-eval
            
            # Planning quality evaluation
            if episode.planning_steps:
                # LLM would evaluate planning coherence, but for now use confidence
                avg_confidence = sum(s.confidence for s in episode.planning_steps) / len(episode.planning_steps)
                rewards["planning_quality"] = avg_confidence * 0.85
            
            # Result synthesis evaluation
            if episode.result_synthesis:
                rewards["result_synthesis"] = (episode.result_synthesis.completeness + episode.result_synthesis.accuracy) / 2.0 * 0.9
            
        except Exception as e:
            logger.error(f"Error in LLM judge evaluation: {e}")
            # Fallback to self-evaluation
            return self.calculate_agent_self_evaluation(episode)
        
        return rewards
    
    def calculate_external_metric_reward(
        self,
        episode: TrainingEpisode,
        metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate reward from external metrics.
        
        Args:
            episode: Training episode
            metrics: External metrics dictionary
            
        Returns:
            Dictionary of reward components
        """
        rewards = {
            "tool_selection": metrics.get("tool_selection_score", 0.0),
            "planning_quality": metrics.get("planning_score", 0.0),
            "result_synthesis": metrics.get("synthesis_score", 0.0)
        }
        
        # Normalize to [0, 1] range
        for key in rewards:
            rewards[key] = max(0.0, min(1.0, rewards[key]))
        
        return rewards
    
    def calculate_composite_reward(
        self,
        episode: TrainingEpisode,
        sources: List[RewardSource],
        source_weights: Optional[Dict[RewardSource, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate reward from multiple sources.
        
        Args:
            episode: Training episode
            sources: List of reward sources to use
            source_weights: Weights for each source
            
        Returns:
            Dictionary of reward components
        """
        if not source_weights:
            # Equal weights by default
            source_weights = {source: 1.0 / len(sources) for source in sources}
        
        all_rewards = {}
        
        for source in sources:
            if source == RewardSource.AGENT_SELF_EVALUATION:
                rewards = self.calculate_agent_self_evaluation(episode)
            elif source == RewardSource.LLM_JUDGE:
                # Note: This is async, but for simplicity we'll use sync version
                rewards = self.calculate_agent_self_evaluation(episode)  # Fallback
            elif source == RewardSource.EXTERNAL_METRIC:
                # Would need external metrics passed in
                rewards = {"tool_selection": 0.0, "planning_quality": 0.0, "result_synthesis": 0.0}
            else:
                rewards = {"tool_selection": 0.0, "planning_quality": 0.0, "result_synthesis": 0.0}
            
            weight = source_weights.get(source, 0.0)
            for key, value in rewards.items():
                if key not in all_rewards:
                    all_rewards[key] = 0.0
                all_rewards[key] += value * weight
        
        return all_rewards
    
    def create_reward_signal(
        self,
        episode: TrainingEpisode,
        reward_source: Optional[RewardSource] = None,
        custom_rewards: Optional[Dict[str, float]] = None,
        evaluator_id: Optional[str] = None
    ) -> RewardSignal:
        """
        Create a reward signal for an episode.
        
        Args:
            episode: Training episode
            reward_source: Source of reward (uses default if None)
            custom_rewards: Custom reward components (overrides calculation)
            evaluator_id: ID of the evaluator
            
        Returns:
            Reward signal
        """
        if reward_source is None:
            reward_source = self.default_reward_source
        
        # Calculate rewards
        if custom_rewards:
            reward_components = custom_rewards
        elif reward_source == RewardSource.AGENT_SELF_EVALUATION:
            reward_components = self.calculate_agent_self_evaluation(episode)
        elif reward_source == RewardSource.LLM_JUDGE:
            # For async, would need to await
            reward_components = self.calculate_agent_self_evaluation(episode)
        elif reward_source == RewardSource.COMPOSITE:
            reward_components = self.calculate_composite_reward(
                episode,
                [RewardSource.AGENT_SELF_EVALUATION, RewardSource.LLM_JUDGE]
            )
        else:
            reward_components = {"tool_selection": 0.0, "planning_quality": 0.0, "result_synthesis": 0.0}
        
        # Create reward signal
        reward_signal = RewardSignal(
            episode_id=episode.episode_id,
            agent_id=episode.agent_id,
            tool_selection_reward=reward_components.get("tool_selection", 0.0),
            planning_quality_reward=reward_components.get("planning_quality", 0.0),
            result_synthesis_reward=reward_components.get("result_synthesis", 0.0),
            reward_source=reward_source,
            reward_components=reward_components,
            reward_weights=self.reward_weights,
            evaluator_id=evaluator_id or "system"
        )
        
        # Calculate overall reward
        reward_signal.calculate_overall_reward()
        
        # Normalize reward
        reward_signal.normalize_reward()
        
        return reward_signal
    
    def batch_calculate_rewards(
        self,
        episodes: List[TrainingEpisode],
        reward_source: Optional[RewardSource] = None
    ) -> List[RewardSignal]:
        """
        Calculate rewards for a batch of episodes.
        
        Args:
            episodes: List of training episodes
            reward_source: Source of reward (uses default if None)
            
        Returns:
            List of reward signals
        """
        rewards = []
        
        for episode in episodes:
            try:
                reward = self.create_reward_signal(episode, reward_source)
                rewards.append(reward)
            except Exception as e:
                logger.error(f"Error calculating reward for episode {episode.episode_id}: {e}")
                # Create a default reward signal with zero rewards
                reward = RewardSignal(
                    episode_id=episode.episode_id,
                    agent_id=episode.agent_id,
                    reward_source=reward_source or self.default_reward_source
                )
                rewards.append(reward)
        
        logger.info(f"Calculated rewards for {len(rewards)} episodes")
        return rewards
    
    def _format_episode_for_evaluation(
        self,
        episode: TrainingEpisode
    ) -> str:
        """Format episode for LLM evaluation"""
        parts = [
            f"User Query: {episode.user_query}",
            f"\nAgent Actions:"
        ]
        
        for tool_step in episode.tool_call_steps:
            parts.append(f"  - Called {tool_step.tool_name}: {'Success' if tool_step.success else 'Failed'}")
        
        if episode.result_synthesis:
            parts.append(f"\nFinal Result: {episode.result_synthesis.synthesized_result}")
        
        return "\n".join(parts)
    
    def get_reward_statistics(
        self,
        rewards: List[RewardSignal]
    ) -> Dict[str, Any]:
        """Get statistics about rewards"""
        if not rewards:
            return {}
        
        return {
            "total_rewards": len(rewards),
            "average_overall_reward": sum(r.overall_reward for r in rewards) / len(rewards),
            "average_tool_selection": sum(r.tool_selection_reward for r in rewards) / len(rewards),
            "average_planning_quality": sum(r.planning_quality_reward for r in rewards) / len(rewards),
            "average_result_synthesis": sum(r.result_synthesis_reward for r in rewards) / len(rewards),
            "min_reward": min(r.overall_reward for r in rewards),
            "max_reward": max(r.overall_reward for r in rewards)
        }

