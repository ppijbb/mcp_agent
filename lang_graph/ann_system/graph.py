from typing import List, TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from .nodes import planner_node_logic, executor_node_logic, critique_node_logic
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """
    Represents the state of our ANN-inspired graph.
    It holds all the information that flows between the nodes.
    """
    initial_task: str
    plan: Optional[str]
    code: Optional[str]
    critique: Optional[str]
    execution_result: Optional[str]
    history: List[Dict[str, str]]
    revision_number: int
    error: Optional[str]
    status: str

class AnnWorkflow:
    """
    Production-ready ANN-inspired workflow for autonomous code generation and refinement.
    """
    
    def __init__(self, max_revisions: int = 5):
        self.max_revisions = max_revisions
        self.graph = StateGraph(AgentState)
        self._build_graph()
    
    def _build_graph(self):
        """Build the workflow graph with all nodes and edges."""
        # Add nodes
        self.graph.add_node("planner", self.plan_node)
        self.graph.add_node("executor", self.execute_node)
        self.graph.add_node("critiquer", self.critique_node)
        self.graph.add_node("error_handler", self.error_handler_node)
        
        # Define the main workflow edges
        self.graph.set_entry_point("planner")
        self.graph.add_edge("planner", "executor")
        self.graph.add_edge("executor", "critiquer")
        
        # Add conditional edges for the refinement loop
        self.graph.add_conditional_edges(
            "critiquer",
            self.should_continue,
            {
                "continue": "planner",
                "end": END,
                "error": "error_handler"
            }
        )
        
        # Error handling edges
        self.graph.add_edge("error_handler", END)
        
        # Compile the graph
        self.app = self.graph.compile()
    
    def should_continue(self, state: AgentState) -> str:
        """
        Decision node to determine workflow continuation.
        
        Args:
            state: Current workflow state
            
        Returns:
            str: Next step in the workflow
        """
        logger.info("---CHECKING COMPLETION---")
        
        # Check for errors
        if state.get("error"):
            logger.error(f"Workflow error detected: {state['error']}")
            return "error"
        
        # Check revision limit
        revision_number = state.get("revision_number", 0)
        if revision_number >= self.max_revisions:
            logger.info(f"-> Stopping due to reaching revision limit ({revision_number})")
            return "end"
        
        # Check if task is complete
        critique = state.get("critique", "").strip().upper()
        if critique == "DONE":
            logger.info("-> Task is complete")
            return "end"
        
        logger.info("-> Needs refinement. Continuing")
        return "continue"
    
    def plan_node(self, state: AgentState) -> AgentState:
        """
        Execute the planning node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with new plan
        """
        logger.info("---PLANNING---")
        
        try:
            # Update history with previous cycle results
            if state.get("critique"):
                full_cycle_summary = {
                    "plan": state.get("plan", ""),
                    "code": state.get("code", ""),
                    "execution_result": state.get("execution_result", ""),
                    "critique": state.get("critique", ""),
                }
                state["history"].append(full_cycle_summary)
            
            # Increment revision number
            revision_number = state.get("revision_number", 0) + 1
            
            # Execute planning logic
            plan_result = planner_node_logic(state)
            
            return {
                **state,
                **plan_result,
                "revision_number": revision_number,
                "history": state["history"],
                "error": None,
                "status": "planning_complete"
            }
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return {
                **state,
                "error": f"Planning failed: {str(e)}",
                "status": "error"
            }
    
    def execute_node(self, state: AgentState) -> AgentState:
        """
        Execute the code generation and execution node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with generated code and execution results
        """
        logger.info("---EXECUTING---")
        
        try:
            # Execute code generation and execution logic
            execution_result = executor_node_logic(state)
            
            return {
                **state,
                **execution_result,
                "error": None,
                "status": "execution_complete"
            }
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return {
                **state,
                "error": f"Execution failed: {str(e)}",
                "status": "error"
            }
    
    def critique_node(self, state: AgentState) -> AgentState:
        """
        Execute the critique node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with critique results
        """
        logger.info("---CRITIQUING---")
        
        try:
            # Execute critique logic
            critique_result = critique_node_logic(state)
            
            return {
                **state,
                **critique_result,
                "error": None,
                "status": "critique_complete"
            }
            
        except Exception as e:
            logger.error(f"Critique failed: {e}")
            return {
                **state,
                "error": f"Critique failed: {str(e)}",
                "status": "error"
            }
    
    def error_handler_node(self, state: AgentState) -> AgentState:
        """
        Handle workflow errors gracefully.
        
        Args:
            state: Current workflow state
            
        Returns:
            State with error information
        """
        logger.error("---ERROR HANDLING---")
        
        error_msg = state.get("error", "Unknown error occurred")
        logger.error(f"Workflow terminated due to error: {error_msg}")
        
        return {
            **state,
            "status": "error",
            "error": error_msg
        }
    
    def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the workflow with the given initial state.
        
        Args:
            initial_state: Initial workflow state
            
        Returns:
            Final workflow state
        """
        try:
            # Ensure required fields are present
            required_fields = ["initial_task"]
            for field in required_fields:
                if field not in initial_state:
                    raise ValueError(f"Missing required field: {field}")
            
            # Set default values
            default_state = {
                "plan": None,
                "code": None,
                "critique": None,
                "execution_result": None,
                "history": [],
                "revision_number": 0,
                "error": None,
                "status": "initialized"
            }
            
            # Merge with provided state
            complete_state = {**default_state, **initial_state}
            
            logger.info(f"Starting workflow with task: {complete_state['initial_task']}")
            
            # Run the workflow
            final_state = None
            for event in self.app.stream(complete_state, {"recursion_limit": self.max_revisions + 2}):
                final_state = event
                logger.info(f"Workflow step completed: {event.get('status', 'unknown')}")
            
            return final_state if final_state else complete_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                **initial_state,
                "error": f"Workflow execution failed: {str(e)}",
                "status": "error"
            }
