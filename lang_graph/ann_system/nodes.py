from .llm_client import call_llm
from .mcp_tool_executor import execute_python_code
import json # For pretty printing the history

# --- PROMPTS ---

PLANNER_PROMPT = """
You are a master planner. Your role is to take a user's task and create a clear, step-by-step plan for execution by a programmer.
You must learn from past attempts and refine the plan based on the provided history of plans, code, execution results, and critiques.

User's Task: {initial_task}

Here is the history of previous attempts:
{history}

Based on the user's task and the history, create a new, improved plan. If the history is empty, create the first plan.
Your output should be a concise plan.
"""

EXECUTOR_PROMPT = """
You are an expert Python programmer. Your task is to write Python code based on the given plan.
You must adhere to the plan and write clean, efficient code.

Plan: {plan}

Your output must be only the Python code inside a markdown block.
"""

CRITIQUE_PROMPT = """
You are a code critic. Your role is to review the generated code and its execution result to provide constructive feedback.
Assess the code for correctness, efficiency, and adherence to the plan. Also, check the execution result for any errors.

Plan: {plan}
Code:
{code}

Execution Result:
{execution_result}

If the code is perfect, has been successfully executed, and meets all requirements from the plan, respond with only the single word "DONE".
Your response MUST be "DONE" and nothing else if the task is complete.
Otherwise, provide a concise critique of what needs to be improved.
"""

# --- NODE LOGIC ---

def planner_node_logic(state):
    # This logic will be imported and used by the AnnWorkflow's plan_node
    # Pretty print the history for the LLM
    history_str = json.dumps(state.get("history", []), indent=2)
    prompt = PLANNER_PROMPT.format(
        initial_task=state["initial_task"],
        history=history_str
    )
    response = call_llm(prompt)
    return {"plan": response}

def executor_node_logic(state):
    # This logic will be imported and used by the AnnWorkflow's execute_node
    prompt = EXECUTOR_PROMPT.format(plan=state["plan"])
    code_generation = call_llm(prompt)
    
    # Execute the generated code
    execution_result = execute_python_code(code_generation)
    
    return {"code": code_generation, "execution_result": execution_result}

def critique_node_logic(state):
    # This logic will be imported and used by the AnnWorkflow's critique_node
    prompt = CRITIQUE_PROMPT.format(
        plan=state["plan"], 
        code=state["code"],
        execution_result=state["execution_result"]
    )
    response = call_llm(prompt)
    return {"critique": response}
