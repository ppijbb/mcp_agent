"""
Online Trainer for Agentic Agent Trainer System

Main training engine that integrates data collection, processing, reward calculation,
and model training in an online learning loop.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..models.training import (
    TrainingConfig,
    TrainingAlgorithm,
    TrainingBatch,
    TrainingEpisode,
    RewardSignal,
    ModelCheckpoint
)
from .data_collector import DataCollector
from .data_processor import DataProcessor
from .reward_system import RewardSystem
from .model_manager import ModelManager
from .algorithms.dpo_trainer import DPOTrainer
from .algorithms.grpo_trainer import GRPOTrainer
from .algorithms.ppo_trainer import PPOTrainer

logger = logging.getLogger(__name__)


class OnlineTrainer:
    """
    Online trainer that performs real-time learning from agent executions.
    
    Responsibilities:
    - Coordinate data collection, processing, and training
    - Manage online learning loop
    - Handle model updates and checkpoints
    - Integrate with simulation engine
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        data_collector: Optional[DataCollector] = None,
        data_processor: Optional[DataProcessor] = None,
        reward_system: Optional[RewardSystem] = None,
        model_manager: Optional[ModelManager] = None
    ):
        """
        Initialize online trainer.
        
        Args:
            config: Training configuration
            data_collector: Data collector instance (optional)
            data_processor: Data processor instance (optional)
            reward_system: Reward system instance (optional)
            model_manager: Model manager instance (optional)
        """
        self.config = config
        
        # Initialize components
        self.data_collector = data_collector or DataCollector()
        self.data_processor = data_processor or DataProcessor()
        self.reward_system = reward_system or RewardSystem(
            llm_config=config.metadata.get("llm_config"),
            default_reward_source=config.reward_source,
            reward_weights=config.reward_weights
        )
        self.model_manager = model_manager or ModelManager(
            checkpoint_dir=config.output_dir
        )
        
        # Initialize algorithm-specific trainer
        self.algorithm_trainer = self._create_algorithm_trainer()
        
        # Training state
        self.current_training_step = 0
        self.current_epoch = 0
        self.is_training = False
        
        # Statistics
        self.total_episodes_processed = 0
        self.total_batches_trained = 0
        
        logger.info(f"OnlineTrainer initialized with algorithm: {config.algorithm}")
    
    def _create_algorithm_trainer(self):
        """Create algorithm-specific trainer"""
        if self.config.algorithm == TrainingAlgorithm.DPO:
            return DPOTrainer(self.config)
        elif self.config.algorithm == TrainingAlgorithm.GRPO:
            return GRPOTrainer(self.config)
        elif self.config.algorithm == TrainingAlgorithm.PPO:
            return PPOTrainer(self.config)
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
    
    async def train_on_episodes(
        self,
        episodes: List[TrainingEpisode],
        rewards: Optional[List[RewardSignal]] = None
    ) -> Dict[str, Any]:
        """
        Train on a batch of episodes.
        
        Args:
            episodes: List of training episodes
            rewards: Optional pre-calculated rewards
            
        Returns:
            Training results
        """
        if not episodes:
            logger.warning("No episodes provided for training")
            return {"status": "skipped", "reason": "no_episodes"}
        
        logger.info(f"Training on {len(episodes)} episodes")
        
        # Calculate rewards if not provided
        if rewards is None:
            rewards = self.reward_system.batch_calculate_rewards(
                episodes,
                reward_source=self.config.reward_source
            )
        
        # Create training batch
        batch = self.data_processor.create_training_batch(
            episodes=episodes,
            rewards=rewards,
            algorithm=self.config.algorithm,
            batch_name=f"batch_step_{self.current_training_step}"
        )
        
        # Process batch for algorithm
        processed_data = self.data_processor.process_batch(batch, self.config.algorithm)
        
        # Train using algorithm-specific trainer
        training_results = await self._train_with_algorithm(processed_data)
        
        # Update statistics
        self.total_episodes_processed += len(episodes)
        self.total_batches_trained += 1
        self.current_training_step += 1
        
        # Save checkpoint if needed
        if self.current_training_step % self.config.save_steps == 0:
            await self._save_checkpoint(training_results)
        
        return training_results
    
    async def _train_with_algorithm(
        self,
        processed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train using the configured algorithm"""
        try:
            if self.config.algorithm == TrainingAlgorithm.DPO:
                preference_pairs = processed_data.get("preference_pairs", [])
                results = self.algorithm_trainer.train(preference_pairs)
            
            elif self.config.algorithm == TrainingAlgorithm.GRPO:
                groups = processed_data.get("groups", [])
                results = self.algorithm_trainer.train(groups)
            
            elif self.config.algorithm == TrainingAlgorithm.PPO:
                samples = processed_data.get("samples", [])
                results = self.algorithm_trainer.train(samples)
            
            else:
                raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error in algorithm training: {e}")
            raise
    
    async def _save_checkpoint(
        self,
        training_results: Dict[str, Any]
    ) -> ModelCheckpoint:
        """Save a training checkpoint"""
        # Save model
        checkpoint_path = f"{self.config.output_dir}/checkpoint_step_{self.current_training_step}"
        self.algorithm_trainer.save_model(checkpoint_path)
        
        # Create checkpoint metadata
        checkpoint = self.model_manager.save_checkpoint(
            model_path=checkpoint_path,
            model_name=self.config.model_name,
            algorithm=self.config.algorithm,
            training_step=self.current_training_step,
            epoch=self.current_epoch,
            training_loss=training_results.get("average_loss", 0.0),
            average_reward=training_results.get("average_reward", 0.0),
            evaluation_metrics=training_results.get("evaluation_metrics", {}),
            metadata={
                "total_episodes": self.total_episodes_processed,
                "total_batches": self.total_batches_trained
            }
        )
        
        logger.info(f"Saved checkpoint at step {self.current_training_step}")
        return checkpoint
    
    async def online_learning_loop(
        self,
        min_episodes_per_batch: int = 32,
        max_wait_time: float = 60.0
    ) -> None:
        """
        Run online learning loop.
        
        Continuously collects episodes, processes them, and trains the model.
        
        Args:
            min_episodes_per_batch: Minimum episodes before training
            max_wait_time: Maximum time to wait for episodes (seconds)
        """
        self.is_training = True
        logger.info("Starting online learning loop")
        
        try:
            while self.is_training:
                # Collect episodes from buffer
                episodes = self.data_collector.get_episodes(
                    limit=min_episodes_per_batch * 2,
                    min_steps=1
                )
                
                if len(episodes) >= min_episodes_per_batch:
                    # Train on collected episodes
                    training_results = await self.train_on_episodes(episodes)
                    logger.info(f"Training completed: {training_results}")
                    
                    # Clear processed episodes from buffer
                    # (In practice, might want to keep some for validation)
                    self.data_collector.clear_buffer()
                else:
                    # Wait for more episodes
                    logger.debug(f"Waiting for more episodes ({len(episodes)}/{min_episodes_per_batch})")
                    await asyncio.sleep(5.0)
        
        except KeyboardInterrupt:
            logger.info("Online learning loop interrupted")
        except Exception as e:
            logger.error(f"Error in online learning loop: {e}")
            raise
        finally:
            self.is_training = False
    
    def stop_training(self) -> None:
        """Stop the online learning loop"""
        self.is_training = False
        logger.info("Stopping online learning loop")
    
    async def train_on_simulation_results(
        self,
        simulation_results: List[Dict[str, Any]],
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Train on simulation results directly.
        
        Args:
            simulation_results: List of simulation state dictionaries
            agent_id: ID of the agent
            
        Returns:
            Training results
        """
        # Collect episodes from simulation results
        episodes = []
        for sim_result in simulation_results:
            episode = self.data_collector.collect_from_simulation_state(
                sim_result,
                agent_id
            )
            if episode:
                # Complete the episode
                completed_episode = self.data_collector.complete_episode(
                    sim_result.get("simulation_id"),
                    final_outcome=sim_result.get("final_outcome"),
                    episode_duration=None
                )
                if completed_episode:
                    episodes.append(completed_episode)
        
        if not episodes:
            return {"status": "skipped", "reason": "no_episodes_collected"}
        
        # Train on episodes
        return await self.train_on_episodes(episodes)
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            "current_training_step": self.current_training_step,
            "current_epoch": self.current_epoch,
            "is_training": self.is_training,
            "total_episodes_processed": self.total_episodes_processed,
            "total_batches_trained": self.total_batches_trained,
            "data_collector_stats": self.data_collector.get_statistics(),
            "model_manager_stats": self.model_manager.get_statistics()
        }
    
    def load_best_model(self) -> Optional[str]:
        """Load the best model checkpoint"""
        best_checkpoint = self.model_manager.get_best_checkpoint()
        if best_checkpoint:
            self.algorithm_trainer.load_model(best_checkpoint.model_path)
            logger.info(f"Loaded best model from {best_checkpoint.model_path}")
            return best_checkpoint.model_path
        else:
            logger.warning("No best model checkpoint found")
            return None
    
    def evaluate_model(
        self,
        validation_episodes: List[TrainingEpisode]
    ) -> Dict[str, float]:
        """
        Evaluate model on validation episodes.
        
        Args:
            validation_episodes: List of validation episodes
            
        Returns:
            Evaluation metrics
        """
        # Calculate rewards for validation episodes
        rewards = self.reward_system.batch_calculate_rewards(validation_episodes)
        
        # Calculate average reward
        avg_reward = sum(r.overall_reward for r in rewards) / len(rewards) if rewards else 0.0
        
        metrics = {
            "average_reward": avg_reward,
            "num_episodes": len(validation_episodes),
            "average_tool_selection": sum(r.tool_selection_reward for r in rewards) / len(rewards) if rewards else 0.0,
            "average_planning_quality": sum(r.planning_quality_reward for r in rewards) / len(rewards) if rewards else 0.0,
            "average_result_synthesis": sum(r.result_synthesis_reward for r in rewards) / len(rewards) if rewards else 0.0
        }
        
        return metrics


