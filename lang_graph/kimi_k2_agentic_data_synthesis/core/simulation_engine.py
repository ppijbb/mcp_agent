"""
Simulation Engine for the Kimi-K2 Agentic Data Synthesis System

Executes large-scale simulations with multiple agents and scenarios.
"""

from typing import List, Dict, Any, Optional, Callable
from ..models.simulation import SimulationState, SimulationSession, SimulationStep, EnvironmentState, SimulationStatus, StepStatus, StepType, SimulationConfig
from ..models.domain import Domain, Scenario
from ..models.agent import Agent, AgentConfig # Agent is now KimiK2ConversableAgent
from ..models.tool import Tool
from ..agents.kimi_k2_agent import KimiK2ConversableAgent

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver # For in-memory checkpointing
from langchain_core.messages import AIMessage, HumanMessage

import asyncio
import logging
from datetime import datetime
import time
import random
import uuid
import re

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

        self.workflow.add_edge("evaluate_step", "execute_agent_turn") # For multi-turn scenarios, go back to agent
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
        scenario_id: Optional[str] = None,
        domain_id: Optional[str] = None,
        user_query: str = ""
    ) -> Dict[str, Any]: # Returns SimulationState dict
        """
        Run a single simulation using the LangGraph workflow.
        """
        logger.info(f"Starting LangGraph simulation {simulation_id} for scenario {scenario_id}")

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
            "scenario_id": scenario_id,
            "domain_id": domain_id,
            "sim_steps": [] # List to store SimulationStep objects generated during the run
        }

        config = {"configurable": {"thread_id": simulation_id}}

        try:
            # Invoke the LangGraph application
            final_state = await self.app.ainvoke(initial_state, config=config, recursion_limit=max_turns)
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
        messages.append(HumanMessage(content=state["user_query"]))

        # Log the start of the simulation
        sim_step = SimulationStep(
            description="Simulation initialized with user query",
            step_type=StepType.USER_INPUT,
            input_data={"user_query": state["user_query"]}
        )
        sim_steps = state.get("sim_steps", [])
        sim_steps.append(sim_step.model_dump())
        sim_step.start()
        sim_step.complete(output={"status": "initialized"})
        sim_steps[-1].update(sim_step.model_dump())

        return {
            "messages": messages,
            "status": SimulationStatus.RUNNING.value,
            "sim_steps": sim_steps
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

            messages.append(AIMessage(content=ai_response))

            sim_step = SimulationStep(
                description=f"Agent {acting_agent.name} executes turn",
                step_type=StepType.AGENT_ACTION,
                agent_id=acting_agent.name,
                input_data=state["messages"][-2] if len(state["messages"]) > 1 else {}, # Previous HumanMessage
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
            sim_step = SimulationStep(
                description=f"Agent {acting_agent.name} turn failed",
                step_type=StepType.AGENT_ACTION,
                agent_id=acting_agent.name,
                status=StepStatus.FAILED,
                error_message=str(e)
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
        last_message_content = state["messages"][-1].content
        sim_id = state["simulation_id"]
        current_turn = len(state["messages"]) // 2 # Number of full turns (user + agent)

        if "<tool_code>" in last_message_content and "</tool_code>" in last_message_content:
            logger.info(f"Simulation {sim_id}: Agent turn contains tool call. Transition to simulate_tool_usage.")
            return "tool_call"
        elif current_turn >= state["max_turns"] or SimulationStatus.FAILED.value == state.get("status"): # Check for max turns or failure
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

        last_message_content = messages[-1].content
        tool_code_block = ""
        try:
            start_idx = last_message_content.find("<tool_code>") + len("<tool_code>")
            end_idx = last_message_content.find("</tool_code>")
            tool_code_block = last_message_content[start_idx:end_idx].strip()
            tool_name_match = re.match(r"^(\w+)\(.*\)", tool_code_block)
            tool_name = tool_name_match.group(1) if tool_name_match else "unknown_tool"
            
            # Simulate tool execution (replace with actual tool_registry.execute_tool in real impl)
            simulated_output = f"Simulated output of {tool_name}: Operation successful."
            result = {"tool_name": tool_name, "output": simulated_output, "status": "success"}
            tool_results.append(result)
            logger.info(f"Node: simulate_tool_usage for {sim_id}. Tool '{tool_name}' executed.")

            # Add tool output back to messages for the agent to see
            messages.append(HumanMessage(content=f"Tool {tool_name} output: {simulated_output}"))
            
            sim_step = SimulationStep(
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

        except Exception as e:
            logger.error(f"Tool usage simulation failed for {sim_id}: {e}")
            error_result = {"tool_name": "parse_error", "output": str(e), "status": "failed"}
            tool_results.append(error_result)
            messages.append(HumanMessage(content=f"Tool execution error: {e}"))
            
            sim_step = SimulationStep(
                description=f"Tool usage failed: {tool_code_block}",
                step_type=StepType.TOOL_USAGE,
                status=StepStatus.FAILED,
                error_message=str(e),
                input_data={"tool_call": tool_code_block},
                output_data=error_result
            )
            sim_steps = state.get("sim_steps", [])
            sim_steps.append(sim_step.model_dump())
            sim_step.start()
            sim_step.fail(str(e))
            sim_steps[-1].update(sim_step.model_dump())

            return {"messages": messages, "tool_results": tool_results, "status": SimulationStatus.FAILED.value, "error_message": str(e), "sim_steps": sim_steps}

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
        messages.append(AIMessage(content=f"Evaluation: {feedback_message} Score: {overall_score:.2f}"))

        sim_step = SimulationStep(
            description=f"Evaluated turn with score {overall_score:.2f}",
            step_type=StepType.EVALUATION,
            input_data=state.get("messages")[-2:] if len(state.get("messages",[]))>=2 else {},
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
        final_session = SimulationSession(
            id=sim_id,
            domain_id=state.get("domain_id", "unknown"),
            scenario_id=state.get("scenario_id", "unknown"),
            agent_ids=state["current_agents"],
            user_agent_id=None, # UserAgentManager would handle this
            status=SimulationStatus.COMPLETED if state["status"] != SimulationStatus.FAILED.value else SimulationStatus.FAILED,
            steps=[SimulationStep(**s) for s in state.get("sim_steps", [])], # Convert back to Pydantic models
            environment_states=[], # Not directly stored in graph state currently, would need to be passed
            start_time=datetime.utcnow(), # Placeholder, ideally captured at start
            end_time=datetime.utcnow(),
            quality_score=state["final_outcome"].get("score") if state["final_outcome"] else 0.0,
            metadata={"user_query": state["user_query"], "error_message": state.get("error_message")}
        )
        
        # DataGenerator integration would go here
        # self.data_generator.generate_data(final_session, ...)

        state["status"] = final_session.status.value
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
            "scenario_id": simulation_result.get("scenario_id", "unknown"),
            "agent_ids": simulation_result["current_agents"],
            "quality_score": simulation_result["final_outcome"].get("score") if simulation_result["final_outcome"] else 0.0,
            "data_format": "json",
            "version": "1.0.0",
            "tags": ["simulated", simulation_result.get("status").lower()],
            "description": f"Training data from simulation {simulation_result["simulation_id"]}"
        }

        conversation_history = []
        for msg in simulation_result["messages"]:
            conversation_history.append({"role": msg.type, "content": msg.content})
        
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