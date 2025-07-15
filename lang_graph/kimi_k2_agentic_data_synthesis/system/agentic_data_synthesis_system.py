"""
Main system orchestration for Kimi-K2 Agentic Data Synthesis System.

This module provides the main system class that coordinates all components
for large-scale agentic data synthesis.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import time
from datetime import datetime

from ..core.domain_manager import DomainManager
from ..core.tool_registry import ToolRegistry
from ..core.agent_factory import AgentFactory
from ..core.simulation_engine import SimulationEngine
from ..core.environment_manager import EnvironmentManager
from ..core.user_agent_manager import UserAgentManager
from ..evaluation.llm_judge import LLMJudgeSystem
from ..evaluation.quality_filter import QualityFilter
from ..data.data_generator import DataGenerator
from ..models.domain import Domain, DomainConfig
from ..models.tool import ToolType, ToolParameter, ParameterType, ToolConfig
from ..models.agent import Agent, AgentConfig
from ..models.simulation import SimulationConfig, SimulationSession as SimulationResult # SimulationResult is now LangGraph state dict
from ..models.evaluation import EvaluationConfig, EvaluationResult
from ..models.data import TrainingData, DataExportConfig
from ..models.domain import ComplexityLevel


class AgenticDataSynthesisSystem:
    """
    Main orchestration system for Kimi-K2 Agentic Data Synthesis.
    
    This class coordinates all components to generate high-quality training data
    for tool usage learning through large-scale multi-agent simulations.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str = "generated_data",
        log_level: str = "INFO",
        llm_config: Optional[Dict[str, Any]] = None # Add llm_config parameter
    ):
        """
        Initialize the Kimi-K2 system.
        
        Args:
            config_path: Path to system configuration file
            output_dir: Directory for generated data output
            log_level: Logging level
            llm_config: Configuration for LLM models (e.g., API key, model name)
        """
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Store LLM config
        self.llm_config = llm_config

        # Initialize components
        self.domain_manager = DomainManager()
        self.tool_registry = ToolRegistry()
        # Pass llm_config and tool_registry to agent_factory
        self.agent_factory = AgentFactory(tool_registry=self.tool_registry, llm_config=self.llm_config)
        self.environment_manager = EnvironmentManager()
        self.user_agent_manager = UserAgentManager()
        # Pass dependencies to simulation_engine
        self.simulation_engine = SimulationEngine(
            domain_manager=self.domain_manager,
            tool_registry=self.tool_registry,
            agent_factory=self.agent_factory,
            llm_config=self.llm_config # Pass llm_config to simulation engine as well
        )
        self.llm_judge = LLMJudgeSystem()
        self.quality_filter = QualityFilter()
        self.data_generator = DataGenerator()
        
        # System state
        self.active_simulations: Dict[str, Dict[str, Any]] = {} # Changed type hint to Dict for LangGraph state
        self.generated_data: List[TrainingData] = []
        self.evaluation_results: List[EvaluationResult] = []
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """
        Load system configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load domains
            if 'domains' in config:
                for domain_config in config['domains']:
                    self.domain_manager.add_domain(DomainConfig(**domain_config))
            
            # Load tools
            if 'tools' in config:
                for tool_config_dict in config['tools']:
                    tool_config = ToolConfig(**tool_config_dict)
                    # Convert parameters dict to list of ToolParameter objects
                    tool_parameters = [
                        ToolParameter(name=key, type=ParameterType(value), description=f"Parameter {key}")
                        for key, value in tool_config.parameters.items()
                    ]
                    self.tool_registry.register_tool(
                        name=tool_config.name,
                        type=ToolType(tool_config.tool_type), # Convert string to ToolType enum
                        description=tool_config.description,
                        parameters=tool_parameters
                    )
            
            # Load agents
            if 'agents' in config:
                for agent_config_dict in config['agents']:
                    # Create AgentConfig from dict
                    agent_config = AgentConfig(**agent_config_dict)
                    # Create KimiK2ConversableAgent using the factory
                    self.agent_factory.create_agent(agent_config)
            
            self.logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def setup_domains(self, domains: List[DomainConfig]) -> None:
        """
        Setup domains for the system.
        
        Args:
            domains: List of domain configurations
        """
        for domain_config in domains:
            self.domain_manager.create_domain(
                name=domain_config.name,
                description=domain_config.description,
                category=domain_config.domain_type, # This is already DomainCategory enum
                complexity_level=domain_config.complexity_levels[0] if domain_config.complexity_levels else ComplexityLevel.INTERMEDIATE,
                required_tools=domain_config.required_tools
                # metadata is not in DomainConfig in example_usage.py for now
            )
        self.logger.info(f"Setup {len(domains)} domains")
    
    def setup_tools(self, tools: List[ToolConfig]) -> None:
        """
        Setup tools for the system.
        
        Args:
            tools: List of tool configurations
        """
        for tool_config in tools:
            # Convert parameters dict to list of ToolParameter objects
            tool_parameters = [
                ToolParameter(name=key, type=ParameterType(value), description=f"Parameter {key}")
                for key, value in tool_config.parameters.items()
            ]
            self.tool_registry.register_tool(
                name=tool_config.name,
                type=ToolType(tool_config.tool_type), # Convert string to ToolType enum
                description=tool_config.description,
                parameters=tool_parameters, # Pass the converted parameters
                # return_type and domain_compatibility are not in ToolConfig in example_usage.py
                # but can be added if needed, or default values will be used.
            )
        self.logger.info(f"Setup {len(tools)} tools")
    
    def setup_agents(self, agents: List[AgentConfig]) -> None:
        """
        Setup agents for the system.
        
        Args:
            agents: List of agent configurations
        """
        for agent_config in agents:
            # Use the factory to create the actual conversable agent
            self.agent_factory.create_agent(agent_config)
        self.logger.info(f"Setup {len(agents)} agents")
    
    async def run_simulation_batch(
        self,
        simulation_configs: List[SimulationConfig],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]: # Changed return type to Dict for LangGraph state
        """
        Run a batch of simulations concurrently.
        
        Args:
            simulation_configs: List of simulation configurations
            max_concurrent: Maximum number of concurrent simulations
            
        Returns:
            List of simulation results (LangGraph states)
        """
        self.logger.info(f"Starting batch of {len(simulation_configs)} simulations")
        
        # Create semaphore to limit concurrent simulations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_simulation_wrapper(config: SimulationConfig) -> Dict[str, Any]: # Wrapper to handle semaphore
            async with semaphore:
                return await self.run_single_simulation(config)
        
        # Run simulations concurrently
        tasks = [run_single_simulation_wrapper(config) for config in simulation_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Simulation {i} failed: {result}")
            else:
                valid_results.append(result)
        
        self.logger.info(f"Completed {len(valid_results)} simulations successfully")
        return valid_results
    
    async def run_single_simulation(self, config: SimulationConfig) -> Dict[str, Any]: # Changed return type to Dict for LangGraph state
        """
        Run a single simulation using the LangGraph engine.
        
        Args:
            config: Simulation configuration
            
        Returns:
            Simulation result (LangGraph state dict)
        """
        simulation_id = config.simulation_id or f"sim_{int(time.time() * 1000)}"
        self.logger.info(f"Starting simulation {simulation_id}")
        
        try:
            # Get agents (KimiK2ConversableAgent instances) from the factory
            agents_for_sim = []
            for agent_cfg in config.agent_configs:
                agent_instance = self.agent_factory.get_agent(agent_cfg.agent_id)
                if agent_instance:
                    agents_for_sim.append(agent_instance)
                else:
                    self.logger.warning(f"Agent {agent_cfg.agent_id} not found. Creating from config...")
                    # Attempt to create if not found (e.g., if config was not loaded from file)
                    agent_instance = self.agent_factory.create_agent(agent_cfg)
                    if agent_instance:
                        agents_for_sim.append(agent_instance)
                    else:
                        raise ValueError(f"Could not create agent {agent_cfg.agent_id} for simulation {simulation_id}")

            if not agents_for_sim:
                raise ValueError(f"No agents available for simulation {simulation_id}")

            # Create environment (if EnvironmentManager is still used for virtual envs)
            # For now, let's pass environment_config directly as part of the state input
            # or create a mock environment object if EnvironmentManager is too complex for LangGraph state.
            environment_data = config.environment_config.model_dump() # Pass as dict
            
            # Run simulation using the LangGraph-based simulation engine
            result = await self.simulation_engine.run_simulation(
                simulation_id=simulation_id,
                agents=agents_for_sim, # Pass KimiK2ConversableAgent instances
                environment=environment_data, # Pass environment data as dict
                user_agent=None, # UserAgentManager integration might need LangGraph node
                max_turns=config.max_turns,
                timeout=config.timeout,
                scenario_id=config.scenario_id, # Pass scenario_id and domain_id for context
                domain_id=config.domain_id,
                user_query=config.scenario # Using scenario as user_query for LangGraph initial input
            )
            
            # Store result (LangGraph state dict)
            self.active_simulations[simulation_id] = result
            
            self.logger.info(f"Simulation {simulation_id} completed successfully (LangGraph output)")
            return result
            
        except Exception as e:
            self.logger.error(f"Simulation {simulation_id} failed: {e}")
            # Return a failed state dictionary
            return {
                "simulation_id": simulation_id,
                "status": "failed",
                "error_message": str(e),
                "messages": [],
                "current_agents": [],
                "environment_state": {},
                "tool_results": [],
                "final_outcome": None,
                "sim_steps": []
            }
    
    async def evaluate_simulations(
        self,
        simulation_results: List[Dict[str, Any]], # Changed to List[Dict] for LangGraph states
        evaluation_config: EvaluationConfig
    ) -> List[EvaluationResult]:
        """
        Evaluate simulation results using LLM judges.
        
        Args:
            simulation_results: List of simulation results (LangGraph states) to evaluate
            evaluation_config: Evaluation configuration
            
        Returns:
            List of evaluation results
        """
        self.logger.info(f"Evaluating {len(simulation_results)} simulations")
        
        evaluation_results = []
        
        for simulation_result in simulation_results:
            try:
                # Evaluate simulation - llm_judge should be able to handle LangGraph state dict
                evaluation_result = await self.llm_judge.evaluate_simulation(
                    simulation_result=simulation_result,
                    config=evaluation_config
                )
                
                evaluation_results.append(evaluation_result)
                
            except Exception as e:
                self.logger.error(f"Evaluation failed for simulation {simulation_result.get('simulation_id', 'unknown')}: {e}")
        
        self.evaluation_results.extend(evaluation_results)
        self.logger.info(f"Completed evaluation of {len(evaluation_results)} simulations")
        
        return evaluation_results
    
    def filter_quality_data(
        self,
        simulation_results: List[Dict[str, Any]], # Changed to List[Dict] for LangGraph states
        evaluation_results: List[EvaluationResult],
        quality_threshold: float = 0.7
    ) -> List[TrainingData]:
        """
        Filter and select high-quality training data.
        
        Args:
            simulation_results: List of simulation results (LangGraph states)
            evaluation_results: List of evaluation results
            quality_threshold: Minimum quality score threshold
            
        Returns:
            List of high-quality training data
        """
        self.logger.info("Filtering high-quality training data")
        
        # Create mapping of simulation_id to evaluation_result
        eval_map = {eval_result.simulation_id: eval_result for eval_result in evaluation_results}
        
        high_quality_data = []
        
        for simulation_result in simulation_results:
            simulation_id = simulation_result.get("simulation_id")
            if not simulation_id:
                self.logger.warning(f"Simulation result missing ID: {simulation_result}")
                continue

            evaluation_result = eval_map.get(simulation_id)
            
            # Check if simulation status is not failed and meets quality threshold
            is_successful_sim = simulation_result.get("status") != "failed"
            if evaluation_result and evaluation_result.overall_score >= quality_threshold and is_successful_sim:
                # Convert simulation result (LangGraph state dict) to training data
                training_data = self.simulation_engine.convert_to_training_data(
                    simulation_result # Pass the LangGraph state dict
                )
                if training_data:
                    high_quality_data.append(TrainingData(**training_data)) # Convert dict to Pydantic model
                else:
                    self.logger.warning(f"Failed to convert simulation {simulation_id} to training data.")
        
        self.logger.info(f"Selected {len(high_quality_data)} high-quality training samples")
        return high_quality_data
    
    async def generate_training_data(
        self,
        training_data: List[TrainingData],
        export_config: DataExportConfig
    ) -> Dict[str, str]:
        """
        Generate and export training data.
        
        Args:
            training_data: List of training data to export
            export_config: Export configuration
            
        Returns:
            Dictionary mapping format to file path
        """
        self.logger.info(f"Generating training data for {len(training_data)} samples")
        
        # Generate data
        export_paths = await self.data_generator.generate_data(
            training_data=training_data,
            config=export_config,
            output_dir=self.output_dir
        )
        
        self.generated_data.extend(training_data)
        
        self.logger.info(f"Training data exported to: {export_paths}")
        return export_paths
    
    async def run_full_pipeline(
        self,
        simulation_configs: List[SimulationConfig],
        evaluation_config: EvaluationConfig,
        export_config: DataExportConfig,
        quality_threshold: float = 0.7,
        max_concurrent_simulations: int = 5
    ) -> Dict[str, Any]:
        """
        Run the complete Kimi-K2 pipeline.
        
        Args:
            simulation_configs: List of simulation configurations
            evaluation_config: Evaluation configuration
            export_config: Export configuration
            quality_threshold: Minimum quality score threshold
            max_concurrent_simulations: Maximum concurrent simulations
            
        Returns:
            Pipeline results summary
        """
        self.logger.info("Starting Kimi-K2 full pipeline")
        start_time = time.time()
        
        try:
            # Step 1: Run simulations
            # simulation_results will be List[Dict[str, Any]] (LangGraph states)
            simulation_results = await self.run_simulation_batch(
                simulation_configs,
                max_concurrent=max_concurrent_simulations
            )
            
            # Filter out failed simulations before evaluation
            successful_sim_results = [s for s in simulation_results if s.get("status") != "failed"]

            # Step 2: Evaluate simulations
            evaluation_results = await self.evaluate_simulations(
                successful_sim_results, # Only evaluate successful ones
                evaluation_config
            )
            
            # Step 3: Filter quality data
            high_quality_data = self.filter_quality_data(
                successful_sim_results, # Use successful simulations for filtering
                evaluation_results,
                quality_threshold
            )
            
            # Step 4: Generate training data
            export_paths = await self.generate_training_data(
                high_quality_data,
                export_config
            )
            
            # Calculate statistics
            total_time = time.time() - start_time
            success_rate = len(successful_sim_results) / len(simulation_configs) if simulation_configs else 0
            quality_rate = len(high_quality_data) / len(successful_sim_results) if successful_sim_results else 0
            
            results = {
                "total_simulations": len(simulation_configs),
                "successful_simulations": len(successful_sim_results),
                "success_rate": success_rate,
                "evaluated_simulations": len(evaluation_results),
                "high_quality_samples": len(high_quality_data),
                "quality_rate": quality_rate,
                "total_time_seconds": total_time,
                "export_paths": export_paths,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Pipeline completed successfully: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            System statistics dictionary
        """
        return {
            "domains": len(self.domain_manager.get_all_domains()),
            "tools": len(self.tool_registry.get_all_tools()),
            "agents": len(self.agent_factory.get_all_agents()),
            "active_simulations": len(self.active_simulations), # This now stores LangGraph states (dicts)
            "generated_data": len(self.generated_data),
            "evaluation_results": len(self.evaluation_results)
        }
    
    def cleanup(self) -> None:
        """
        Cleanup system resources.
        """
        self.logger.info("Cleaning up system resources")
        
        # Cleanup environments
        if self.environment_manager:
            active_envs = self.environment_manager.list_environments(status="active")
            for env in active_envs:
                self.environment_manager.destroy_environment(env["id"])
        
        # Clear active simulations
        self.active_simulations.clear()
        
        self.logger.info("System cleanup completed") 