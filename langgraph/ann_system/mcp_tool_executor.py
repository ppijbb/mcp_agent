import subprocess
import os
import re

def execute_python_code(code: str) -> str:
    """
    Executes a string of Python code and returns the output or error.
    
    Args:
        code (str): The Python code to execute.
        
    Returns:
        str: A string containing the execution result (stdout/stderr).
    """
    # 1. Extract code from markdown block if present
    code_match = re.search(r"```python\n(.*)```", code, re.DOTALL)
    if code_match:
        code_to_execute = code_match.group(1).strip()
    else:
        code_to_execute = code.strip()

    if not code_to_execute:
        return "Execution Result: FAILURE\nError: No code found to execute."

    # 2. Create a temporary file to run the code
    filepath = "temp_mcp_execution.py"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code_to_execute)

    # 3. Execute the code in a subprocess
    try:
        result = subprocess.run(
            ["python", filepath],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        output = f"Execution Result: SUCCESS\n---Output---\n{result.stdout}"
        if result.stderr:
            output += f"\n---Stderr---\n{result.stderr}"
        return output
    except subprocess.CalledProcessError as e:
        return f"Execution Result: FAILURE\n---Error---\n{e.stderr}"
    except subprocess.TimeoutExpired:
        return "Execution Result: FAILURE\nError: Code execution timed out after 15 seconds."
    except Exception as e:
        return f"Execution Result: FAILURE\nAn unexpected error occurred: {e}"
    finally:
        # 4. Clean up the temporary file
        if os.path.exists(filepath):
            os.remove(filepath) 