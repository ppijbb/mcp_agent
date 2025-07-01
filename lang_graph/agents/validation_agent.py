from autogen import ConversableAgent
from typing import Dict, Any
import subprocess
import os

class ValidationAgent(ConversableAgent):
    def __init__(self, name="Validator", **kwargs):
        super().__init__(
            name=name,
            system_message="""You are a Validator. Your role is to execute Python code and report the result.
You will be given a Python code snippet to execute.
You should report whether the code ran successfully or if it produced any errors.
Your response should start with 'Validation Result:' followed by 'SUCCESS' or 'FAILURE'.
If it fails, provide the error message.
""",
            **kwargs,
        )

    def validate_code(self, code: str) -> str:
        # Create a temporary file to run the code
        filepath = "temp_validation_code.py"
        with open(filepath, "w") as f:
            f.write(code)

        try:
            # Execute the code in a subprocess
            result = subprocess.run(
                ["python", filepath],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return f"Validation Result: SUCCESS\nOutput:\n{result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Validation Result: FAILURE\nError:\n{e.stderr}"
        except subprocess.TimeoutExpired:
            return "Validation Result: FAILURE\nError: Code execution timed out."
        finally:
            # Clean up the temporary file
            if os.path.exists(filepath):
                os.remove(filepath)

    def run(self, message: str) -> Dict[str, Any]:
        # Assumes the message is the code to validate.
        # The code might be inside a markdown block.
        code_to_validate = message.strip().replace("```python", "").replace("```", "").strip()
        result = self.validate_code(code_to_validate)
        return {"content": result} 