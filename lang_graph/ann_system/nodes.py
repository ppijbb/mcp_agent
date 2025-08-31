from .llm_client import call_llm
from .mcp_tool_executor import execute_python_code
import json
from typing import Dict, Any, Optional

# --- PROMPTS ---

PLANNER_PROMPT = """
You are an expert software architect and project planner. Your role is to analyze user tasks and create comprehensive, actionable plans for implementation.

TASK: {initial_task}

HISTORY OF PREVIOUS ATTEMPTS:
{history}

INSTRUCTIONS:
1. Analyze the task requirements carefully
2. Review the history to understand what has been attempted and what failed
3. Create a detailed, step-by-step plan that addresses previous failures
4. Ensure the plan is specific, measurable, and implementable
5. If this is the first attempt, create a comprehensive initial plan

OUTPUT FORMAT:
Provide a clear, structured plan with:
- Specific steps to accomplish the task
- Expected outcomes for each step
- Any constraints or considerations
- Success criteria

Your response should be the plan only, no additional commentary.
"""

EXECUTOR_PROMPT = """
You are a senior Python developer with expertise in writing clean, efficient, and production-ready code.

PLAN TO IMPLEMENT:
{plan}

REQUIREMENTS:
1. Write Python code that implements the plan exactly
2. Ensure code is well-structured and follows PEP 8 standards
3. Include appropriate error handling and validation
4. Make the code reusable and maintainable
5. Add clear comments explaining complex logic

OUTPUT FORMAT:
Provide ONLY the Python code wrapped in a markdown code block:

```python
# Your Python code here
```

No additional text, explanations, or commentary outside the code block.
"""

CRITIQUE_PROMPT = """
You are a senior software engineer and code reviewer. Your task is to evaluate the generated code against the original plan and execution results.

ORIGINAL PLAN:
{plan}

GENERATED CODE:
{code}

EXECUTION RESULTS:
{execution_result}

EVALUATION CRITERIA:
1. Does the code correctly implement the plan?
2. Are there any syntax errors or runtime issues?
3. Does the code meet the original task requirements?
4. Is the code production-ready (clean, efficient, well-documented)?

DECISION:
- If the code successfully implements the plan, meets all requirements, and executes without errors, respond with exactly: "DONE"
- Otherwise, provide specific, actionable feedback on what needs to be improved

Your response must be either "DONE" or specific improvement suggestions.
"""

# --- NODE LOGIC ---

def planner_node_logic(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the planning node logic.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dict containing the generated plan
    """
    try:
        # Format history for better LLM understanding
        history = state.get("history", [])
        if history:
            history_str = json.dumps(history, indent=2, ensure_ascii=False)
        else:
            history_str = "No previous attempts"
        
        prompt = PLANNER_PROMPT.format(
            initial_task=state["initial_task"],
            history=history_str
        )
        
        response = call_llm(prompt, temperature=0.3, max_tokens=1500)
        
        if not response or response.strip() == "":
            raise RuntimeError("Planner returned empty response")
        
        return {"plan": response.strip()}
        
    except Exception as e:
        raise RuntimeError(f"Planning failed: {str(e)}")

def executor_node_logic(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the code generation and execution node logic.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dict containing generated code and execution results
    """
    try:
        plan = state.get("plan")
        if not plan:
            raise RuntimeError("No plan available for execution")
        
        # Generate code based on plan
        prompt = EXECUTOR_PROMPT.format(plan=plan)
        code_generation = call_llm(prompt, temperature=0.2, max_tokens=2000)
        
        if not code_generation or code_generation.strip() == "":
            raise RuntimeError("Code generator returned empty response")
        
        # Execute the generated code
        execution_result = execute_python_code(code_generation)
        
        return {
            "code": code_generation.strip(),
            "execution_result": execution_result
        }
        
    except Exception as e:
        raise RuntimeError(f"Execution failed: {str(e)}")

def critique_node_logic(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the critique node logic.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dict containing the critique or completion status
    """
    try:
        plan = state.get("plan")
        code = state.get("code")
        execution_result = state.get("execution_result")
        
        if not all([plan, code, execution_result]):
            raise RuntimeError("Missing required state for critique")
        
        prompt = CRITIQUE_PROMPT.format(
            plan=plan,
            code=code,
            execution_result=execution_result
        )
        
        response = call_llm(prompt, temperature=0.1, max_tokens=1000)
        
        if not response or response.strip() == "":
            raise RuntimeError("Critique returned empty response")
        
        return {"critique": response.strip()}
        
    except Exception as e:
        raise RuntimeError(f"Critique failed: {str(e)}")
