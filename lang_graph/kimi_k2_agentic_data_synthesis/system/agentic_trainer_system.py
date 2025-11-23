"""
Agentic Trainer System

Integrates the training system with the existing AgenticDataSynthesisSystem
to enable online learning from agent executions.
"""

import logging
from typing import Dict, Any, List, Optional
import asyncio

from ..system.agentic_data_synthesis_system import AgenticDataSynthesisSystem
from ..models.training import TrainingConfig, TrainingAlgorithm
from ..models.simulation import SimulationConfig
from ..training.online_trainer import OnlineTrainer
from ..training.data_collector import DataCollector

logger = logging.getLogger(__name__)


class AgenticTrainerSystem(AgenticDataSynthesisSystem):
    """
    Extended system that adds online learning capabilities.
    
    Extends AgenticDataSynthesisSystem with:
    - Real-time data collection from simulations
    - Online training loop
    - Model updates and deployment
    """
    
    def __init__(
        self,
        training_config: TrainingConfig,
        output_dir: str = "generated_data",
        log_level: str = "INFO",
        llm_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the agentic trainer system.
        
        Args:
            training_config: Training configuration
            output_dir: Directory for outputs
            log_level: Logging level
            llm_config: LLM configuration
        """
        # Initialize base system
        super().__init__(
            output_dir=output_dir,
            log_level=log_level,
            llm_config=llm_config
        )
        
        self.training_config = training_config
        
        # Initialize data collector
        self.data_collector = DataCollector()
        
        # Connect data collector to simulation engine
        self.simulation_engine.data_collector = self.data_collector
        
        # Initialize online trainer
        self.online_trainer = OnlineTrainer(
            config=training_config,
            data_collector=self.data_collector,
            model_manager=None  # Will be created by OnlineTrainer
        )
        
        logger.info("AgenticTrainerSystem initialized with online learning")
    
    def _register_data_collection_hooks(self) -> None:
        """Register hooks for data collection from simulations"""
        # Hook into simulation engine
        # This will be called when simulations complete
        pass  # Implementation will be in simulation engine hooks
    
    async def run_simulation_with_training(
        self,
        simulation_config: SimulationConfig,
        collect_data: bool = True
    ) -> Dict[str, Any]:
        """
        Run a simulation and collect training data.
        
        Args:
            simulation_config: Simulation configuration
            collect_data: Whether to collect training data
            
        Returns:
            Simulation results with training data info
        """
        # Start data collection
        if collect_data:
            agent_id = simulation_config.agent_configs[0].agent_id if simulation_config.agent_configs else "unknown"
            self.data_collector.start_episode(
                simulation_id=simulation_config.simulation_id,
                agent_id=agent_id,
                user_query=simulation_config.scenario,
                context=simulation_config.environment_config.model_dump() if simulation_config.environment_config else {}
            )
        
        # Run simulation
        simulation_result = await self.run_single_simulation(simulation_config)
        
        # Collect data from simulation result
        if collect_data and simulation_result:
            agent_id = simulation_config.agent_configs[0].agent_id if simulation_config.agent_configs else "unknown"
            episode = self.data_collector.collect_from_simulation_state(
                simulation_result,
                agent_id
            )
            
            # Complete episode
            if episode:
                completed_episode = self.data_collector.complete_episode(
                    simulation_config.simulation_id,
                    final_outcome=simulation_result.get("final_outcome"),
                    episode_duration=None
                )
        
        return {
            "simulation_result": simulation_result,
            "data_collected": collect_data,
            "episode_id": episode.episode_id if episode else None
        }
    
    async def run_training_cycle(
        self,
        simulation_configs: List[SimulationConfig],
        min_episodes: int = 32
    ) -> Dict[str, Any]:
        """
        Run a complete training cycle: simulations -> data collection -> training.
        
        Args:
            simulation_configs: List of simulation configurations
            min_episodes: Minimum episodes before training
            
        Returns:
            Training cycle results
        """
        logger.info(f"Starting training cycle with {len(simulation_configs)} simulations")
        
        # Run simulations and collect data
        simulation_results = []
        for config in simulation_configs:
            result = await self.run_simulation_with_training(config, collect_data=True)
            simulation_results.append(result)
        
        # Wait until we have enough episodes
        episodes = self.data_collector.get_episodes(min_steps=1)
        
        if len(episodes) >= min_episodes:
            # Train on collected episodes
            training_results = await self.online_trainer.train_on_episodes(episodes)
            
            return {
                "simulations_run": len(simulation_results),
                "episodes_collected": len(episodes),
                "training_results": training_results,
                "status": "completed"
            }
        else:
            return {
                "simulations_run": len(simulation_results),
                "episodes_collected": len(episodes),
                "training_results": None,
                "status": "insufficient_episodes",
                "message": f"Only {len(episodes)} episodes collected, need {min_episodes}"
            }
    
    async def start_online_learning(
        self,
        simulation_generator: callable,
        min_episodes_per_batch: int = 32
    ) -> None:
        """
        Start continuous online learning loop.
        
        Args:
            simulation_generator: Function that generates simulation configs
            min_episodes_per_batch: Minimum episodes per training batch
        """
        logger.info("Starting continuous online learning")
        
        # Start training loop in background
        training_task = asyncio.create_task(
            self.online_trainer.online_learning_loop(
                min_episodes_per_batch=min_episodes_per_batch
            )
        )
        
        # Generate and run simulations
        try:
            while True:
                # Generate simulation config
                sim_config = await simulation_generator()
                
                # Run simulation with data collection
                await self.run_simulation_with_training(sim_config, collect_data=True)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(1.0)
        
        except KeyboardInterrupt:
            logger.info("Stopping online learning")
            self.online_trainer.stop_training()
            training_task.cancel()
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        base_stats = self.get_system_stats()
        training_stats = self.online_trainer.get_training_statistics()
        
        return {
            **base_stats,
            "training": training_stats
        }
    
    def load_best_model(self) -> Optional[str]:
        """Load the best trained model"""
        return self.online_trainer.load_best_model()
    
    async def evaluate_trained_model(
        self,
        validation_simulations: List[SimulationConfig]
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model on validation simulations.
        
        Args:
            validation_simulations: List of validation simulation configs
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating model on {len(validation_simulations)} validation simulations")
        
        # Run validation simulations
        validation_episodes = []
        for config in validation_simulations:
            result = await self.run_simulation_with_training(config, collect_data=True)
            episodes = self.data_collector.get_episodes(
                limit=1,
                agent_id=config.agent_configs[0].agent_id if config.agent_configs else None
            )
            validation_episodes.extend(episodes)
        
        # Evaluate model
        evaluation_metrics = self.online_trainer.evaluate_model(validation_episodes)
        
        return {
            "num_validation_simulations": len(validation_simulations),
            "num_validation_episodes": len(validation_episodes),
            "evaluation_metrics": evaluation_metrics
        }

