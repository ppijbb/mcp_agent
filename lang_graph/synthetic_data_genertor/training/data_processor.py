"""
Training Data Processor for Agentic Agent Trainer System

Processes collected episodes into training formats for DPO, GRPO, and PPO algorithms.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..models.training import (
    TrainingEpisode,
    TrainingBatch,
    RewardSignal,
    TrainingAlgorithm
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessedTrainingSample:
    """Processed training sample ready for model training"""
    prompt: str
    response: str
    reward: float
    metadata: Dict[str, Any]


@dataclass
class PreferencePair:
    """Preference pair for DPO training"""
    prompt: str
    chosen: str
    rejected: str
    metadata: Dict[str, Any]


class DataProcessor:
    """
    Processes training episodes into formats suitable for different algorithms.
    
    Responsibilities:
    - Convert episodes to DPO format (preference pairs)
    - Convert episodes to GRPO format (group-based preferences)
    - Convert episodes to PPO format (sequences with rewards)
    - Format prompts with tool calling and planning context
    """
    
    def __init__(
        self,
        include_tool_calls: bool = True,
        include_planning: bool = True,
        include_results: bool = True,
        max_sequence_length: int = 2048
    ):
        """
        Initialize the data processor.
        
        Args:
            include_tool_calls: Include tool call information in prompts
            include_planning: Include planning steps in prompts
            include_results: Include result synthesis in prompts
            max_sequence_length: Maximum sequence length for training
        """
        self.include_tool_calls = include_tool_calls
        self.include_planning = include_planning
        self.include_results = include_results
        self.max_sequence_length = max_sequence_length
        
        logger.info("DataProcessor initialized")
    
    def format_episode_prompt(
        self,
        episode: TrainingEpisode,
        include_context: bool = True
    ) -> str:
        """
        Format an episode into a training prompt.
        
        Args:
            episode: Training episode
            include_context: Include context information
            
        Returns:
            Formatted prompt string
        """
        parts = []
        
        # User query
        parts.append(f"User Query: {episode.user_query}")
        
        # Context
        if include_context and episode.context:
            context_str = ", ".join(f"{k}: {v}" for k, v in episode.context.items())
            parts.append(f"Context: {context_str}")
        
        # Planning steps
        if self.include_planning and episode.planning_steps:
            parts.append("\nPlanning Steps:")
            for plan_step in episode.planning_steps:
                parts.append(f"  - {plan_step.plan_description}")
                if plan_step.sub_goals:
                    for goal in plan_step.sub_goals:
                        parts.append(f"    * {goal}")
                if plan_step.reasoning:
                    parts.append(f"    Reasoning: {plan_step.reasoning}")
        
        # Tool calls
        if self.include_tool_calls and episode.tool_call_steps:
            parts.append("\nTool Calls:")
            for tool_step in episode.tool_call_steps:
                tool_info = f"  - {tool_step.tool_name}("
                params_str = ", ".join(f"{k}={v}" for k, v in tool_step.tool_parameters.items())
                tool_info += params_str + ")"
                if tool_step.success:
                    tool_info += f" -> Success"
                else:
                    tool_info += f" -> Failed: {tool_step.error_message}"
                parts.append(tool_info)
        
        # Result synthesis
        if self.include_results and episode.result_synthesis:
            parts.append("\nResult Synthesis:")
            parts.append(f"  {episode.result_synthesis.synthesized_result}")
            if episode.result_synthesis.reasoning:
                parts.append(f"  Reasoning: {episode.result_synthesis.reasoning}")
        
        return "\n".join(parts)
    
    def format_episode_response(
        self,
        episode: TrainingEpisode
    ) -> str:
        """
        Format an episode into a response string.
        
        Args:
            episode: Training episode
            
        Returns:
            Formatted response string
        """
        parts = []
        
        # Tool calls as actions
        if episode.tool_call_steps:
            for tool_step in episode.tool_call_steps:
                action = f"Action: Call {tool_step.tool_name}"
                if tool_step.tool_parameters:
                    params_str = ", ".join(f"{k}={v}" for k, v in tool_step.tool_parameters.items())
                    action += f" with parameters: {params_str}"
                parts.append(action)
                
                if tool_step.tool_result:
                    result_str = f"Result: {tool_step.tool_result}"
                    parts.append(result_str)
        
        # Final result
        if episode.result_synthesis:
            parts.append(f"Final Answer: {episode.result_synthesis.synthesized_result}")
        elif episode.final_outcome:
            parts.append(f"Final Outcome: {episode.final_outcome}")
        
        return "\n".join(parts) if parts else "No response generated"
    
    def process_for_dpo(
        self,
        episodes: List[TrainingEpisode],
        rewards: List[RewardSignal]
    ) -> List[PreferencePair]:
        """
        Process episodes into DPO preference pairs.
        
        Args:
            episodes: List of training episodes
            rewards: List of reward signals
            
        Returns:
            List of preference pairs (chosen, rejected)
        """
        reward_map = {r.episode_id: r for r in rewards}
        preference_pairs = []
        
        # Group episodes by similar queries
        query_groups: Dict[str, List[Tuple[TrainingEpisode, RewardSignal]]] = {}
        for episode in episodes:
            reward = reward_map.get(episode.episode_id)
            if not reward:
                continue
            
            # Use a simplified query key (can be enhanced with semantic similarity)
            query_key = episode.user_query[:100]  # First 100 chars as key
            
            if query_key not in query_groups:
                query_groups[query_key] = []
            query_groups[query_key].append((episode, reward))
        
        # Create preference pairs from groups
        for query_key, episode_rewards in query_groups.items():
            if len(episode_rewards) < 2:
                continue
            
            # Sort by reward
            episode_rewards.sort(key=lambda x: x[1].overall_reward, reverse=True)
            
            # Create pairs: higher reward vs lower reward
            for i in range(len(episode_rewards) - 1):
                chosen_ep, chosen_reward = episode_rewards[i]
                rejected_ep, rejected_reward = episode_rewards[i + 1]
                
                # Only create pair if reward difference is significant
                if chosen_reward.overall_reward > rejected_reward.overall_reward + 0.1:
                    prompt = self.format_episode_prompt(chosen_ep)
                    chosen = self.format_episode_response(chosen_ep)
                    rejected = self.format_episode_response(rejected_ep)
                    
                    preference_pairs.append(PreferencePair(
                        prompt=prompt,
                        chosen=chosen,
                        rejected=rejected,
                        metadata={
                            "chosen_episode_id": chosen_ep.episode_id,
                            "rejected_episode_id": rejected_ep.episode_id,
                            "chosen_reward": chosen_reward.overall_reward,
                            "rejected_reward": rejected_reward.overall_reward,
                            "reward_diff": chosen_reward.overall_reward - rejected_reward.overall_reward
                        }
                    ))
        
        logger.info(f"Created {len(preference_pairs)} DPO preference pairs from {len(episodes)} episodes")
        return preference_pairs
    
    def process_for_grpo(
        self,
        episodes: List[TrainingEpisode],
        rewards: List[RewardSignal]
    ) -> List[Dict[str, Any]]:
        """
        Process episodes into GRPO format (group-based preferences).
        
        Args:
            episodes: List of training episodes
            rewards: List of reward signals
            
        Returns:
            List of group-based training samples
        """
        reward_map = {r.episode_id: r for r in rewards}
        groups: Dict[str, List[Dict[str, Any]]] = {}
        
        for episode in episodes:
            reward = reward_map.get(episode.episode_id)
            if not reward:
                continue
            
            # Group by agent_id and similar queries
            group_key = f"{episode.agent_id}_{episode.user_query[:50]}"
            
            if group_key not in groups:
                groups[group_key] = []
            
            prompt = self.format_episode_prompt(episode)
            response = self.format_episode_response(episode)
            
            groups[group_key].append({
                "prompt": prompt,
                "response": response,
                "reward": reward.overall_reward,
                "episode_id": episode.episode_id,
                "metadata": {
                    "agent_id": episode.agent_id,
                    "tool_selection_reward": reward.tool_selection_reward,
                    "planning_quality_reward": reward.planning_quality_reward,
                    "result_synthesis_reward": reward.result_synthesis_reward
                }
            })
        
        # Convert groups to GRPO format
        grpo_samples = []
        for group_key, samples in groups.items():
            if len(samples) < 2:
                continue
            
            # Sort by reward
            samples.sort(key=lambda x: x["reward"], reverse=True)
            
            grpo_samples.append({
                "group_id": group_key,
                "samples": samples,
                "group_size": len(samples),
                "average_reward": sum(s["reward"] for s in samples) / len(samples),
                "best_reward": samples[0]["reward"],
                "worst_reward": samples[-1]["reward"]
            })
        
        logger.info(f"Created {len(grpo_samples)} GRPO groups from {len(episodes)} episodes")
        return grpo_samples
    
    def process_for_ppo(
        self,
        episodes: List[TrainingEpisode],
        rewards: List[RewardSignal]
    ) -> List[ProcessedTrainingSample]:
        """
        Process episodes into PPO format (sequences with rewards).
        
        Args:
            episodes: List of training episodes
            rewards: List of reward signals
            
        Returns:
            List of processed training samples
        """
        reward_map = {r.episode_id: r for r in rewards}
        ppo_samples = []
        
        for episode in episodes:
            reward = reward_map.get(episode.episode_id)
            if not reward:
                continue
            
            prompt = self.format_episode_prompt(episode)
            response = self.format_episode_response(episode)
            
            # Truncate if too long
            if len(prompt) + len(response) > self.max_sequence_length:
                # Keep prompt, truncate response
                max_response_len = self.max_sequence_length - len(prompt) - 100
                response = response[:max_response_len] + "..."
            
            ppo_samples.append(ProcessedTrainingSample(
                prompt=prompt,
                response=response,
                reward=reward.normalized_reward if reward.normalized_reward != 0.0 else reward.overall_reward,
                metadata={
                    "episode_id": episode.episode_id,
                    "agent_id": episode.agent_id,
                    "total_steps": episode.total_steps,
                    "success_rate": episode.calculate_success_rate(),
                    "tool_selection_reward": reward.tool_selection_reward,
                    "planning_quality_reward": reward.planning_quality_reward,
                    "result_synthesis_reward": reward.result_synthesis_reward,
                    "reward_source": reward.reward_source.value
                }
            ))
        
        logger.info(f"Created {len(ppo_samples)} PPO samples from {len(episodes)} episodes")
        return ppo_samples
    
    def process_batch(
        self,
        batch: TrainingBatch,
        algorithm: TrainingAlgorithm
    ) -> Dict[str, Any]:
        """
        Process a training batch for a specific algorithm.
        
        Args:
            batch: Training batch
            algorithm: Training algorithm to use
            
        Returns:
            Processed data in algorithm-specific format
        """
        episodes = batch.episodes
        rewards = batch.rewards
        
        if algorithm == TrainingAlgorithm.DPO:
            preference_pairs = self.process_for_dpo(episodes, rewards)
            return {
                "format": "dpo",
                "preference_pairs": preference_pairs,
                "num_pairs": len(preference_pairs)
            }
        
        elif algorithm == TrainingAlgorithm.GRPO:
            grpo_groups = self.process_for_grpo(episodes, rewards)
            return {
                "format": "grpo",
                "groups": grpo_groups,
                "num_groups": len(grpo_groups)
            }
        
        elif algorithm == TrainingAlgorithm.PPO:
            ppo_samples = self.process_for_ppo(episodes, rewards)
            return {
                "format": "ppo",
                "samples": ppo_samples,
                "num_samples": len(ppo_samples)
            }
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def create_training_batch(
        self,
        episodes: List[TrainingEpisode],
        rewards: List[RewardSignal],
        algorithm: TrainingAlgorithm,
        batch_name: str = "training_batch"
    ) -> TrainingBatch:
        """
        Create a training batch from episodes and rewards.
        
        Args:
            episodes: List of episodes
            rewards: List of rewards
            algorithm: Training algorithm
            batch_name: Name for the batch
            
        Returns:
            Training batch
        """
        batch = TrainingBatch(
            batch_name=batch_name,
            algorithm=algorithm
        )
        
        for episode in episodes:
            batch.add_episode(episode)
        
        for reward in rewards:
            batch.add_reward(reward)
        
        batch.calculate_statistics()
        
        return batch


