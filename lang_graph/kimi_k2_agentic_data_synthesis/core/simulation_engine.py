"""
Simulation Engine for the Kimi-K2 Agentic Data Synthesis System

Manages multi-agent simulations using LangGraph for complex scenarios.
"""

from typing import Dict, Any, List, Optional, Annotated
from models.simulation import SimulationState, SimulationStep, StepType, StepStatus, SimulationStatus, EnvironmentState
from models.agent import Agent
from agents.kimi_k2_agent import KimiK2ConversableAgent
from langchain_core.messages import HumanMessage, AIMessage
import logging
import random
import re
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Engine for executing large-scale agentic simulations using LangGraph.
    
    Responsibilities:
    - Building and managing the LangGraph StateGraph for simulations
    - Executing multi-turn scenarios
    - Coordinating agent interactions and tool usage
    - Managing environment state changes within the graph
    - Logging and monitoring simulation progress
    """
    
    def __init__(self, domain_manager: Any = None, tool_registry: Any = None, agent_factory: Any = None, llm_config: Optional[Dict[str, Any]] = None):
        self.domain_manager = domain_manager
        self.tool_registry = tool_registry
        self.agent_factory = agent_factory
        self.llm_config = llm_config
        
        # LangGraph workflow setup
        self.workflow = StateGraph(SimulationState)
        
        # Define nodes
        self.workflow.add_node("initialize_simulation", self._initialize_simulation_node)
        self.workflow.add_node("execute_agent_turn", self._execute_agent_turn_node)
        self.workflow.add_node("simulate_tool_usage", self._simulate_tool_usage_node)
        self.workflow.add_node("evaluate_step", self._evaluate_step_node)
        self.workflow.add_node("finalize_simulation", self._finalize_simulation_node)
        
        # Define edges
        self.workflow.add_edge(START, "initialize_simulation")
        self.workflow.add_edge("initialize_simulation", "execute_agent_turn")
        
        # Conditional transitions from execute_agent_turn
        self.workflow.add_conditional_edges(
            "execute_agent_turn",
            self._decide_next_step_from_agent_turn,
            {
                "tool_call": "simulate_tool_usage",
                "continue_turn": "execute_agent_turn", # Agent decides to continue its turn without tool
                "end_turn": "evaluate_step",
            },
        )

        # Conditional transitions from simulate_tool_usage
        self.workflow.add_conditional_edges(
            "simulate_tool_usage",
            self._decide_next_step_from_tool_usage,
            {
                "agent_turn": "execute_agent_turn", # Tool result returns to agent for next action
                "evaluate": "evaluate_step", # Tool usage completes a sub-task, go to evaluation
            },
        )

        # Conditional transitions from evaluate_step
        self.workflow.add_conditional_edges(
            "evaluate_step",
            self._decide_next_step_from_evaluation,
            {
                "continue_simulation": "execute_agent_turn", # Continue with more agent turns
                "end_simulation": "finalize_simulation", # End the simulation
            },
        )

        # Removed the direct edge from evaluate_step as it's now handled by conditional edges
        # self.workflow.add_edge("evaluate_step", "execute_agent_turn") 
        self.workflow.add_edge("finalize_simulation", END)

        # Compile the workflow
        self.app = self.workflow.compile(checkpointer=MemorySaver())
        
        logger.info("SimulationEngine initialized with LangGraph workflow.")


    async def run_simulation(
        self,
        simulation_id: str,
        agents: List[KimiK2ConversableAgent], # Now expects ConversableAgents
        environment: Dict[str, Any],
        user_agent: Optional[Any] = None,
        max_turns: int = 20,
        timeout: int = 600, # seconds
        scenario: Optional[str] = None, # Changed from scenario_id
        domain_id: Optional[str] = None,
        user_query: str = ""
    ) -> Dict[str, Any]: # Returns SimulationState dict
        """
        Run a single simulation using the LangGraph workflow.
        """
        logger.info(f"Starting LangGraph simulation {simulation_id} for scenario {scenario}")

        # Initialize the state for the LangGraph workflow
        initial_state: SimulationState = {
            "simulation_id": simulation_id,
            "user_query": user_query,
            "messages": [], # Start with empty messages for the graph
            "current_agents": [agent.name for agent in agents], # Use agent.name (autogen's name)
            "environment_state": environment, # Initial environment state
            "tool_results": [],
            "final_outcome": None,
            "status": SimulationStatus.RUNNING.value,
            "error_message": None,
            "max_turns": max_turns, # Pass max turns and timeout to state
            "timeout": timeout,
            "scenario": scenario, # Changed from scenario_id
            "domain_id": domain_id,
            "sim_steps": [] # List to store SimulationStep objects generated during the run
        }

        config = {
            "configurable": {"thread_id": simulation_id},
            "recursion_limit": max_turns
        }

        try:
            # Invoke the LangGraph application
            final_state = await self.app.ainvoke(initial_state, config=config)
            logger.info(f"LangGraph simulation {simulation_id} finished with status: {final_state.get("status")}")
            return final_state
        except Exception as e:
            logger.error(f"LangGraph simulation {simulation_id} failed: {e}")
            initial_state["status"] = SimulationStatus.FAILED.value
            initial_state["error_message"] = str(e)
            return initial_state

    def _initialize_simulation_node(self, state: SimulationState) -> SimulationState:
        """
        LangGraph node to initialize the simulation state.
        This will prepare the initial environment and set up the first turn.
        """
        logger.info(f"Node: initialize_simulation for {state["simulation_id"]}")
        # Add initial user query as a message
        messages = state.get("messages", [])
        messages.append(HumanMessage(content=state["user_query"]).model_dump())

        # Log the start of the simulation
        sim_step = SimulationStep(
            step_number=1, # Add missing step_number
            description="Simulation initialized with user query",
            step_type=StepType.USER_INPUT,
            input_data={"user_query": state["user_query"]}
        )
        sim_step.start()
        sim_step.complete(output={"status": "initialized"})
        
        sim_steps = state.get("sim_steps", [])
        sim_steps.append(sim_step.model_dump())

        return {
            "messages": messages,
            "status": SimulationStatus.RUNNING.value,
            "sim_steps": sim_steps,
            "max_turns": state["max_turns"], # Pass max_turns through the state
            "timeout": state["timeout"] # Pass timeout through the state
        }

    def _execute_agent_turn_node(self, state: SimulationState) -> SimulationState:
        """
        LangGraph node to execute a turn for the current active agent.
        The agent will generate a response or tool call.
        """
        sim_id = state["simulation_id"]
        messages = state["messages"]
        current_agents = state["current_agents"]
        environment_state = state["environment_state"]
        max_turns = state["max_turns"]

        logger.info(f"Node: execute_agent_turn for {sim_id}, Turn: {len(messages) // 2 + 1}")

        # Determine current agent based on turn or round-robin for simplicity
        # In a real system, a more sophisticated agent selection logic might be here.
        # For now, let's pick the first agent for simplicity or a round-robin.
        if not current_agents:
            logger.error(f"No agents available for simulation {sim_id}.")
            return {"status": SimulationStatus.FAILED.value, "error_message": "No agents available"}

        # Get the actual ConversableAgent instance
        # This requires agent_factory to be available in the node context or passed explicitly
        # For now, we'll assume a global access or passed as part of the engine's state.
        # In LangGraph, nodes should be pure functions or methods of a class with dependencies injected.
        
        # Placeholder for agent selection - in reality, agent_factory will provide instances
        # For this example, let's just pick the first one from the list in state (which is agent ID/name)
        acting_agent_id = current_agents[0] # Simplistic: always the first agent
        acting_agent: Optional[KimiK2ConversableAgent] = self.agent_factory.get_agent(acting_agent_id) # Need agent factory here

        if not acting_agent:
            logger.error(f"Acting agent '{acting_agent_id}' not found in factory.")
            return {"status": SimulationStatus.FAILED.value, "error_message": f"Agent {acting_agent_id} not found"}

        # Simulate agent's thinking and response generation
        # This is where the LLM call and AutoGen logic would go.
        # For now, a simplified simulated response.
        try:
            # In a real AutoGen setup, you would use agent.initiate_chat or agent.generate_reply
            # to get the agent's response, which might include tool calls.
            # For this example, we simulate a response.
            ai_response = f"Agent {acting_agent.name} is working on the task. Current environment: {environment_state}."
            # This is where the actual LLM call for the agent would happen. For demonstration:
            # response = await acting_agent.a_run_task(state["user_query"]) # Or messages[-1].content
            # ai_response = response["content"]

            # Simulate a tool call sometimes
            if random.random() < 0.5: # 50% chance to simulate a tool call
                tool_name = random.choice(acting_agent.agent_config.tool_preferences or ["generic_tool"])
                tool_params = {"input": "simulated_input"}
                ai_response += f"\n<tool_code>\n{tool_name}({tool_params})\n</tool_code>"
                logger.info(f"Agent {acting_agent.name} simulated tool call: {tool_name}")

            messages.append(AIMessage(content=ai_response).model_dump())
            
            # Log the turn result
            logger.info(f"=== TURN {len(messages) // 2} RESULT ===")
            logger.info(f"Agent: {acting_agent.name}")
            logger.info(f"Response: {ai_response}")
            logger.info(f"================================")

            # Calculate current step number. Each full turn (user+agent) counts as 2 messages.
            # The current step number should correspond to the message being processed.
            current_step_number = len(state.get("sim_steps", [])) + 1 # Increment for the new step

            sim_step = SimulationStep(
                step_number=current_step_number, # Provide the step number
                description=f"Agent {acting_agent.name} executes turn",
                step_type=StepType.AGENT_ACTION,
                agent_id=acting_agent.name,
                input_data=state["messages"][-2] if len(state["messages"]) > 1 else {}, # Extract content as dict
                output_data={"response": ai_response}
            )
            sim_steps = state.get("sim_steps", [])
            sim_steps.append(sim_step.model_dump())
            sim_step.start()
            sim_step.complete(output={"response": ai_response})
            sim_steps[-1].update(sim_step.model_dump())

            return {"messages": messages, "sim_steps": sim_steps}

        except Exception as e:
            logger.error(f"Agent turn failed for {acting_agent_id}: {e}")
            
            current_step_number = len(state.get("sim_steps", [])) + 1 # Calculate step number for failed step
            input_data_for_failure = state["messages"][-2] if len(state["messages"]) > 1 else {}

            sim_step = SimulationStep(
                step_number=current_step_number, # Add step_number to failed step
                description=f"Agent {acting_agent.name} turn failed",
                step_type=StepType.AGENT_ACTION,
                agent_id=acting_agent.name,
                status=StepStatus.FAILED,
                error_message=str(e),
                input_data=input_data_for_failure # Provide valid dictionary for input_data
            )
            sim_steps = state.get("sim_steps", [])
            sim_steps.append(sim_step.model_dump())
            sim_step.start()
            sim_step.fail(str(e))
            sim_steps[-1].update(sim_step.model_dump())
            return {"messages": messages, "tool_results": state.get("tool_results", []), "status": SimulationStatus.FAILED.value, "error_message": str(e), "sim_steps": sim_steps} # Ensure tool_results is passed

    def _decide_next_step_from_agent_turn(self, state: SimulationState) -> str:
        """
        Decides the next step after an agent turn.
        Checks if the agent's message contains a tool call or if the simulation should end.
        """
        last_message_content = state["messages"][-1]["content"]
        sim_id = state["simulation_id"]
        current_turn = len(state["messages"]) // 2 # Number of full turns (user + agent)

        if "<tool_code>" in last_message_content and "</tool_code>" in last_message_content:
            logger.info(f"Simulation {sim_id}: Agent turn contains tool call. Transition to simulate_tool_usage.")
            return "tool_call"
        # Safely get max_turns, defaulting to 20 if not found in state
        max_turns_threshold = state.get("max_turns", 20) 
        if current_turn >= max_turns_threshold or SimulationStatus.FAILED.value == state.get("status"): # Check for max turns or failure
            logger.info(f"Simulation {sim_id}: Max turns reached or simulation failed. Transition to evaluate_step.")
            return "end_turn"
        else:
            logger.info(f"Simulation {sim_id}: Agent turn continues. Transition to execute_agent_turn.")
            return "continue_turn"

    def _simulate_tool_usage_node(self, state: SimulationState) -> SimulationState:
        """
        LangGraph node to simulate tool usage based on agent's tool call.
        """
        sim_id = state["simulation_id"]
        messages = state["messages"]
        tool_results = state.get("tool_results", [])

        last_message_content = messages[-1]["content"]
        tool_code_block = ""
        try:
            start_idx = last_message_content.find("<tool_code>") + len("<tool_code>")
            end_idx = last_message_content.find("</tool_code>")
            tool_code_block = last_message_content[start_idx:end_idx].strip()
            tool_name_match = re.match(r"^(\w+)\(.*\)", tool_code_block)
            tool_name = tool_name_match.group(1) if tool_name_match else "unknown_tool"
            
            # Parse tool parameters from the code block
            import ast
            try:
                # Extract the function call part
                func_call_match = re.match(r"(\w+)\((.*)\)", tool_code_block)
                if func_call_match:
                    tool_name = func_call_match.group(1)
                    args_str = func_call_match.group(2)
                    
                    # Parse arguments (simple parsing for demonstration)
                    parameters = {}
                    if args_str.strip():
                        # Handle keyword arguments
                        if "=" in args_str:
                            for arg in args_str.split(","):
                                if "=" in arg:
                                    key, value = arg.split("=", 1)
                                    parameters[key.strip()] = value.strip().strip('"\'')
                        else:
                            # Handle positional arguments
                            args = [arg.strip().strip('"\'') for arg in args_str.split(",")]
                            if args:
                                parameters["input"] = args[0]  # Assume first arg is input
                    
                    # Execute the tool through the registry
                    if self.tool_registry:
                        result = self.tool_registry.execute_tool(tool_name, parameters)
                        if "error" in result:
                            simulated_output = f"Tool execution failed: {result['error']}"
                            result_status = "failed"
                        else:
                            simulated_output = result.get("result", f"Tool {tool_name} executed successfully")
                            result_status = "success"
                    else:
                        simulated_output = f"Simulated output of {tool_name}: Operation successful."
                        result_status = "success"
                else:
                    simulated_output = f"Could not parse tool call: {tool_code_block}"
                    result_status = "failed"
                    
            except Exception as parse_error:
                simulated_output = f"Error parsing tool call: {parse_error}"
                result_status = "failed"
            
            result = {"tool_name": tool_name, "output": simulated_output, "status": result_status}
            tool_results.append(result)
            logger.info(f"Node: simulate_tool_usage for {sim_id}. Tool '{tool_name}' executed with status: {result_status}.")

            # Add tool output back to messages for the agent to see
            messages.append(HumanMessage(content=f"Tool {tool_name} output: {simulated_output}").model_dump())
            
            # Log the tool usage result
            logger.info(f"=== TOOL USAGE RESULT ===")
            logger.info(f"Tool: {tool_name}")
            logger.info(f"Input: {tool_code_block}")
            logger.info(f"Output: {simulated_output}")
            logger.info(f"Status: {result_status}")
            logger.info(f"==========================")
            
        except Exception as e:
            logger.error(f"Error in tool usage simulation: {e}")
            result = {"tool_name": "unknown", "output": f"Error: {str(e)}", "status": "failed"}
            tool_results.append(result)
            messages.append(HumanMessage(content=f"Tool execution error: {str(e)}").model_dump())
        
        current_step_number = len(state.get("sim_steps", [])) + 1

        sim_step = SimulationStep(
            step_number=current_step_number,
            description=f"Tool {tool_name} executed",
            step_type=StepType.TOOL_USAGE,
            input_data={"tool_call": tool_code_block},
            output_data=result
        )
        sim_steps = state.get("sim_steps", [])
        sim_steps.append(sim_step.model_dump())
        sim_step.start()
        sim_step.complete(output=result)
        sim_steps[-1].update(sim_step.model_dump())

        return {"messages": messages, "tool_results": tool_results, "sim_steps": sim_steps}

    def _decide_next_step_from_tool_usage(self, state: SimulationState) -> str:
        """
        Decides the next step after tool usage.
        If tool usage was successful, return to agent turn. Otherwise, evaluate (failure).
        """
        sim_id = state["simulation_id"]
        if state.get("status") == SimulationStatus.FAILED.value:
            logger.info(f"Simulation {sim_id}: Tool usage failed. Transition to evaluate_step.")
            return "evaluate"
        else:
            # Assuming successful tool usage leads back to agent to process output
            logger.info(f"Simulation {sim_id}: Tool usage successful. Transition to execute_agent_turn.")
            return "agent_turn"

    def _evaluate_step_node(self, state: SimulationState) -> SimulationState:
        """
        LangGraph node to evaluate the current step or overall simulation.
        """
        sim_id = state["simulation_id"]
        logger.info(f"Node: evaluate_step for {sim_id}")

        # This is where the LLMJudgeSystem and QualityFilter would be used.
        # For now, simulate a simple evaluation.
        is_successful_turn = random.random() > 0.1 # 90% chance of success
        
        current_step_status = StepStatus.COMPLETED.value if is_successful_turn else StepStatus.FAILED.value
        feedback_message = "Turn successful." if is_successful_turn else "Turn failed due to simulated issue."
        overall_score = 0.9 if is_successful_turn else 0.2

        messages = state["messages"]
        messages.append(AIMessage(content=f"Evaluation: {feedback_message} Score: {overall_score:.2f}").model_dump())

        current_step_number = len(state.get("sim_steps", [])) + 1
        sim_step = SimulationStep(
            step_number=current_step_number,
            description=f"Evaluated turn with score {overall_score:.2f}",
            step_type=StepType.EVALUATION,
            input_data={
                "last_human_message": state["messages"][-2] if len(state["messages"]) >= 2 else {},
                "last_ai_message": state["messages"][-1] if len(state["messages"]) >= 1 else {}
            },
            output_data={"score": overall_score, "feedback": feedback_message}
        )
        sim_steps = state.get("sim_steps", [])
        sim_steps.append(sim_step.model_dump())
        sim_step.start()
        sim_step.complete(output={"score": overall_score})
        sim_steps[-1].update(sim_step.model_dump())
        
        new_status = state["status"]
        if not is_successful_turn: # If current turn simulation failed, mark overall sim as failed
            new_status = SimulationStatus.FAILED.value
            state["error_message"] = feedback_message

        return {
            "messages": messages,
            "status": new_status,
            "final_outcome": {"score": overall_score, "feedback": feedback_message},
            "sim_steps": sim_steps
        }

    def _decide_next_step_from_evaluation(self, state: SimulationState) -> str:
        """
        Decides the next step after an evaluation.
        If the simulation is complete or failed, transition to finalize. Otherwise, continue agent turns.
        """
        sim_id = state["simulation_id"]
        current_turn = len(state["messages"]) // 2 # Number of full turns (user + agent)
        max_turns = state.get("max_turns", 20)

        if state.get("status") == SimulationStatus.FAILED.value:
            logger.info(f"Simulation {sim_id}: Evaluation indicates failure. Transition to finalize_simulation.")
            return "end_simulation"
        elif current_turn >= max_turns:
            logger.info(f"Simulation {sim_id}: Max turns ({max_turns}) reached. Transition to finalize_simulation.")
            return "end_simulation"
        else:
            logger.info(f"Simulation {sim_id}: Evaluation complete, continuing agent turns. Transition to execute_agent_turn.")
            return "continue_simulation"

    def _finalize_simulation_node(self, state: SimulationState) -> SimulationState:
        """
        LangGraph node to finalize the simulation, gather results, and mark as completed.
        This is where DataGenerator would be used to export data.
        """
        sim_id = state["simulation_id"]
        logger.info(f"Node: finalize_simulation for {sim_id}. Status: {state["status"]}")

        # Convert raw simulation steps (dicts) back to SimulationStep objects if needed for processing
        # sim_steps_objects = [SimulationStep(**s) for s in state.get("sim_steps", [])]

        # Create a final SimulationSession object for comprehensive storage/export
        # This step maps the LangGraph state back to our original Pydantic model
        final_status = SimulationStatus.COMPLETED.value if state["status"] != SimulationStatus.FAILED.value else SimulationStatus.FAILED.value
        simulation_session = SimulationSession(
            id=sim_id,
            domain_id=state.get("domain_id", "unknown"),
            scenario_id=state.get("scenario", "unknown"), # Use scenario from state
            agent_ids=state["current_agents"],
            user_agent_id=None, # UserAgentManager would handle this
            status=SimulationStatus(final_status),
            steps=state["sim_steps"],
            environment_states=[], # Collect from sim_steps if needed
            start_time=None, # Need to track start time in state
            end_time=datetime.utcnow(),
            duration=state["timeout"] - state.get("time_remaining", state["timeout"]), # Placeholder for duration
            total_steps=len(state["sim_steps"]),
            completed_steps=sum(1 for step in state["sim_steps"] if step.get("status") == StepStatus.COMPLETED.value),
            failed_steps=sum(1 for step in state["sim_steps"] if step.get("status") == StepStatus.FAILED.value),
            quality_score=state.get("final_outcome", {}).get("score"),
            metadata={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # DataGenerator integration would go here
        # self.data_generator.generate_data(simulation_session, ...)

        state["status"] = simulation_session.status.value
        logger.info(f"Simulation {sim_id} finalized.")
        return state

    def convert_to_training_data(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a LangGraph simulation result (dict) into a TrainingData format.
        This function should be called by the DataGenerator or by the main system.
        """
        # This is a simplified conversion. In reality, you'd parse messages, tool_results, etc.
        # to construct the conversation_history and tool_usage_log more accurately.
        training_data_id = str(uuid.uuid4())
        metadata = {
            "simulation_id": simulation_result["simulation_id"],
            "domain_id": simulation_result.get("domain_id", "unknown"),
            "scenario_id": simulation_result.get("scenario", "unknown"), # Use scenario from state
            "agent_ids": simulation_result["current_agents"],
            "quality_score": simulation_result["final_outcome"].get("score") if simulation_result["final_outcome"] else 0.0,
            "data_format": "json",
            "version": "1.0.0",
            "tags": ["simulated", simulation_result.get("status").lower()],
            "description": f"Training data from simulation {simulation_result["simulation_id"]}"
        }

        conversation_history = []
        for msg in simulation_result["messages"]:
            conversation_history.append({"role": msg["type"], "content": msg["content"]})
        
        tool_usage_log = simulation_result.get("tool_results", [])

        final_outcome = simulation_result.get("final_outcome", {})

        # Create a dummy TrainingData model (as Pydantic models require full data, this is simplified)
        # In a real scenario, you'd create the Pydantic TrainingData object
        return {
            "id": training_data_id,
            "metadata": metadata,
            "conversation_history": conversation_history,
            "tool_usage_log": tool_usage_log,
            "final_outcome": final_outcome,
            "quality_metrics": final_outcome, # Simplified
            "is_valid": True # Assuming valid for now
        } 