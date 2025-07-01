from typing import List, TypedDict, Optional, Dict
from langgraph.graph import StateGraph, END
from .nodes import planner_node_logic, executor_node_logic, critique_node_logic

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

class AnnWorkflow:
    def __init__(self):
        self.graph = StateGraph(AgentState)
        self._build_graph()

    def _build_graph(self):
        # All nodes and edges will be defined here.
        # This is the blueprint of our agent system.
        
        self.graph.add_node("planner", self.plan_node)
        self.graph.add_node("executor", self.execute_node)
        self.graph.add_node("critiquer", self.critique_node)

        # Define the edges that connect these nodes for the forward pass.
        self.graph.set_entry_point("planner")
        self.graph.add_edge("planner", "executor")
        self.graph.add_edge("executor", "critiquer")
        
        # This is the conditional edge that forms the "refinement loop".
        self.graph.add_conditional_edges(
            "critiquer",
            self.should_continue,
            {
                "continue": "planner",
                "end": END,
            },
        )

        # Compile the graph
        self.app = self.graph.compile()

    def should_continue(self, state: AgentState) -> str:
        """
        Decision node to determine whether to continue with refinement or end the process.
        Includes a safety net to prevent infinite loops.
        """
        print("---CHECKING COMPLETION---")
        
        # Safety net: Stop after a certain number of revisions to prevent infinite loops.
        if state["revision_number"] >= 5:
            print(f"-> Stopping due to reaching revision limit ({state['revision_number']}).")
            return "end"

        # Check if the critique is "DONE"
        critique = state.get("critique", "").strip().upper()
        if critique == "DONE":
            print("-> Task is complete.")
            return "end"
        else:
            print("-> Needs refinement. Continuing.")
            return "continue"

    def plan_node(self, state: AgentState):
        print("---PLANNING---")
        
        # Add the result of the last full loop to history before planning the next one.
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
        
        # Call the planner logic
        plan_result = planner_node_logic(state)
        return {**plan_result, "revision_number": revision_number, "history": state["history"]}

    def execute_node(self, state: AgentState):
        print("---EXECUTING---")
        return executor_node_logic(state)

    def critique_node(self, state: AgentState):
        print("---CRITIQUING---")
        return critique_node_logic(state)
