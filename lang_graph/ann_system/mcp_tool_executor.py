import subprocess
import os
import re
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import ast

class CodeExecutor:
    """
    Production-ready Python code executor with security measures.
    """
    
    def __init__(self, timeout: int = 30, max_output_size: int = 10000):
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.temp_dir = None
    
    def _validate_python_code(self, code: str) -> bool:
        """
        Validate Python code syntax and check for potentially dangerous operations.
        
        Args:
            code (str): Python code to validate
            
        Returns:
            bool: True if code is safe and valid
        """
        try:
            # Parse the code to check syntax
            ast.parse(code)
            
            # Check for potentially dangerous operations
            dangerous_patterns = [
                r'import\s+os\s*$',
                r'import\s+subprocess\s*$',
                r'import\s+sys\s*$',
                r'__import__\s*\(',
                r'eval\s*\(',
                r'exec\s*\(',
                r'open\s*\(',
                r'file\s*\(',
                r'input\s*\(',
                r'raw_input\s*\(',
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                    return False
            
            return True
        except SyntaxError:
            return False
    
    def _extract_code_from_markdown(self, content: str) -> str:
        """
        Extract Python code from markdown code blocks.
        
        Args:
            content (str): Content that may contain markdown code blocks
            
        Returns:
            str: Extracted Python code
        """
        # Look for Python code blocks
        python_patterns = [
            r"```python\n(.*?)```",
            r"```py\n(.*?)```",
            r"```\n(.*?)```"
        ]
        
        for pattern in python_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no markdown blocks found, treat entire content as code
        return content.strip()
    
    def execute_python_code(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code safely and return detailed results.
        
        Args:
            code (str): Python code to execute
            
        Returns:
            Dict[str, Any]: Execution results with status, output, and metadata
        """
        # Extract code from markdown if present
        extracted_code = self._extract_code_from_markdown(code)
        
        if not extracted_code:
            return {
                "status": "error",
                "error": "No executable code found",
                "output": "",
                "execution_time": 0
            }
        
        # Validate code safety
        if not self._validate_python_code(extracted_code):
            return {
                "status": "error",
                "error": "Code contains potentially dangerous operations",
                "output": "",
                "execution_time": 0
            }
        
        # Create temporary directory for execution
        self.temp_dir = tempfile.mkdtemp(prefix="code_exec_")
        temp_file = os.path.join(self.temp_dir, "execution.py")
        
        try:
            # Write code to temporary file
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(extracted_code)
            
            # Execute code
            start_time = os.times()
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.temp_dir,
                env=self._get_safe_environment()
            )
            end_time = os.times()
            
            execution_time = end_time[0] - start_time[0]
            
            # Process output
            stdout = result.stdout[:self.max_output_size]
            stderr = result.stderr[:self.max_output_size]
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "output": stdout,
                    "stderr": stderr,
                    "execution_time": execution_time,
                    "return_code": result.returncode
                }
            else:
                return {
                    "status": "error",
                    "error": f"Execution failed with return code {result.returncode}",
                    "output": stdout,
                    "stderr": stderr,
                    "execution_time": execution_time,
                    "return_code": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": f"Execution timed out after {self.timeout} seconds",
                "output": "",
                "stderr": "",
                "execution_time": self.timeout,
                "return_code": None
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Execution failed: {str(e)}",
                "output": "",
                "stderr": "",
                "execution_time": 0,
                "return_code": None
            }
        finally:
            self._cleanup()
    
    def _get_safe_environment(self) -> Dict[str, str]:
        """
        Create a safe environment for code execution.
        
        Returns:
            Dict[str, str]: Safe environment variables
        """
        env = os.environ.copy()
        
        # Remove potentially dangerous environment variables
        dangerous_vars = [
            'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP',
            'PYTHONEXECUTABLE', 'PYTHONUSERBASE'
        ]
        
        for var in dangerous_vars:
            env.pop(var, None)
        
        # Set safe defaults
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        
        return env
    
    def _cleanup(self):
        """Clean up temporary files and directories."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
            except Exception:
                pass

# Global instance
code_executor = CodeExecutor()

def execute_python_code(code: str) -> str:
    """
    Convenience function for backward compatibility.
    
    Args:
        code (str): Python code to execute
        
    Returns:
        str: Formatted execution result
    """
    result = code_executor.execute_python_code(code)
    
    if result["status"] == "success":
        output = f"Execution Result: SUCCESS\n"
        if result["output"]:
            output += f"---Output---\n{result['output']}\n"
        if result["stderr"]:
            output += f"---Stderr---\n{result['stderr']}\n"
        output += f"---Execution Time---\n{result['execution_time']:.3f}s"
    else:
        output = f"Execution Result: FAILURE\n"
        output += f"---Error---\n{result['error']}\n"
        if result["stderr"]:
            output += f"---Stderr---\n{result['stderr']}\n"
        if result["execution_time"] > 0:
            output += f"---Execution Time---\n{result['execution_time']:.3f}s"
    
    return output 