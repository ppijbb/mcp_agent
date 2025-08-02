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
from ..evaluation.human_feedback_manager import HumanFeedbackManager
from ..data.data_generator import DataGenerator
from ..models.domain import Domain, DomainConfig
from ..models.tool import ToolType, ToolParameter, ParameterType, ToolConfig
from ..models.agent import Agent, AgentConfig
from ..models.simulation import SimulationConfig, SimulationState, SimulationStatus, SimulationSession as SimulationResult # SimulationResult is now LangGraph state dict
from ..models.evaluation import EvaluationConfig, EvaluationResult, HumanFeedback
from ..models.data import TrainingData, DataExportConfig, DataFormat
from ..models.domain import ComplexityLevel
from ..agents.kimi_k2_agent import KimiK2ConversableAgent # Added missing import


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
        self.human_feedback_manager = HumanFeedbackManager()
        
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
                for domain_config_dict in config['domains']:
                    domain_config = DomainConfig(**domain_config_dict)
                    self.domain_manager.create_domain(
                        name=domain_config.name,
                        description=domain_config.description,
                        category=domain_config.domain_type,
                        complexity_level=domain_config.complexity_levels[0] if domain_config.complexity_levels else ComplexityLevel.INTERMEDIATE,
                        required_tools=domain_config.required_tools
                    )
            
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
        """Setup tools in the system"""
        self.logger.info(f"Setting up {len(tools)} tools")
        
        for tool_config in tools:
            try:
                # Convert parameters dict to list of ToolParameter objects
                tool_parameters = []
                for param_name, param_type_str in tool_config.parameters.items():
                    param_type = ParameterType(param_type_str)
                    tool_parameters.append(
                        ToolParameter(
                            name=param_name,
                            type=param_type,
                            description=f"Parameter {param_name}",
                            required=True
                        )
                    )
                
                # Register tool with explicit tool_id
                self.tool_registry.register_tool(
                    tool_id=tool_config.tool_id,
                    name=tool_config.name,
                    type=ToolType(tool_config.tool_type),
                    description=tool_config.description,
                    parameters=tool_parameters,
                    domain_compatibility=["general"]  # Default compatibility
                )
                
                self.logger.info(f"Registered tool: {tool_config.name} (ID: {tool_config.tool_id})")
                
            except Exception as e:
                self.logger.error(f"Failed to register tool {tool_config.name}: {e}")
                raise
    
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
        Run a single simulation using the LangGraph workflow.
        """
        simulation_id = config.simulation_id or f"sim_{int(time.time() * 1000)}"
        self.logger.info(f"Starting LangGraph simulation {simulation_id} for scenario {config.scenario}")

        # Get agent instances for this simulation
        agents_for_this_sim: List[KimiK2ConversableAgent] = []
        for agent_cfg in config.agent_configs:
            agent_instance = self.agent_factory.get_agent(agent_cfg.agent_id)
            if agent_instance:
                agents_for_this_sim.append(agent_instance)
            else:
                self.logger.warning(f"Agent {agent_cfg.agent_id} not found in factory. It should have been setup.")
                # Fallback: try to create if not found (e.g., if config was not loaded from file)
                created_agent = self.agent_factory.create_agent(agent_cfg) # This also adds to factory
                if created_agent:
                    agents_for_this_sim.append(created_agent)
                else:
                    raise ValueError(f"Could not find or create agent {agent_cfg.agent_id} for simulation {simulation_id}")

        if not agents_for_this_sim:
            raise ValueError(f"No agents available for simulation {simulation_id}")

        # Initialize the state for the LangGraph workflow
        initial_state: SimulationState = {
            "simulation_id": simulation_id,
            "user_query": config.scenario, # Pass scenario as user_query
            "messages": [], # Start with empty messages for the graph
            "current_agents": [agent.name for agent in agents_for_this_sim], # Use agent.name (autogen's name)
            "environment_state": config.environment_config.model_dump(), # Initial environment state from config
            "tool_results": [],
            "final_outcome": None,
            "status": SimulationStatus.RUNNING.value,
            "error_message": None,
            "max_turns": config.max_turns, # Pass max turns and timeout to state
            "timeout": config.timeout,
            "scenario": config.scenario, # Pass scenario
            "domain_id": None, # Cannot derive directly from SimulationConfig. Leaving as None for now.
            "sim_steps": [] # List to store SimulationStep objects generated during the run
        }

        config_langgraph = {
            "configurable": {"thread_id": simulation_id},
            "recursion_limit": config.max_turns
        }

        try:
            print(f"DEBUG: In run_single_simulation, config.max_turns is {config.max_turns} (type: {type(config.max_turns)})")
            self.logger.debug(f"Invoking LangGraph app with initial_state: {initial_state.keys()}")
            # Invoke the LangGraph application
            final_state = await self.simulation_engine.app.ainvoke(initial_state, config=config_langgraph)
            self.logger.info(f"LangGraph simulation {simulation_id} finished with status: {final_state.get("status")}")
            return final_state
        except Exception as e:
            self.logger.error(f"LangGraph simulation {simulation_id} failed: {e}")
            initial_state["status"] = SimulationStatus.FAILED.value
            initial_state["error_message"] = str(e)
            return initial_state
    
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
        """
        self.logger.info(f"Generating training data for {len(training_data)} samples")
        
        export_paths = {}
        
        if not training_data:
            self.logger.warning("No training data to generate. Skipping export.")
            return export_paths

        # Create a single batch for all training data
        batch = self.data_generator.create_training_batch(
            name="generated_simulation_data",
            description="Training data from Kimi-K2 simulations",
            training_data=training_data
        )

        # Export data in specified formats
        for fmt in export_config.formats:
            # Convert string format to DataFormat enum
            data_format_enum = DataFormat(fmt) 
            output_path = self.data_generator.export_batch(batch.id, format=data_format_enum)
            if output_path:
                export_paths[fmt] = str(output_path)
            else:
                self.logger.error(f"Failed to export data in {fmt} format.")
        
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
            "domains": len(self.domain_manager.list_domains()),
            "tools": len(self.tool_registry.list_tools()),
            "agents": len(self.agent_factory.agents), # Use len of agents dictionary
            "active_simulations": len(self.active_simulations),
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

    def add_human_feedback(self, evaluation_id: str, reviewer_id: str, rating: float, feedback_text: str) -> None:
        """
        Adds human feedback to an evaluation result.

        Args:
            evaluation_id: The ID of the evaluation to add feedback to.
            reviewer_id: The ID of the human reviewer.
            rating: The rating given by the reviewer (0.0 to 1.0).
            feedback_text: The qualitative feedback.
        """
        feedback = HumanFeedback(
            reviewer_id=reviewer_id,
            rating=rating,
            qualitative_feedback=feedback_text
        )
        self.human_feedback_manager.add_feedback(evaluation_id, feedback)

        # Optionally link it to the evaluation result object if it exists
        for ev_res in self.evaluation_results:
            if ev_res.id == evaluation_id:
                ev_res.human_feedback = feedback
                self.logger.info(f"Successfully linked feedback to evaluation result {evaluation_id}")
                break

    def get_human_feedback(self, evaluation_id: str) -> Optional[HumanFeedback]:
        """
        Retrieves human feedback for a specific evaluation.

        Args:
            evaluation_id: The ID of the evaluation.

        Returns:
            The HumanFeedback object or None if not found.
        """
        return self.human_feedback_manager.get_feedback(evaluation_id)
 