"""
Secure Code Execution MCP Server

Provides safe Python code execution with proper sandboxing and security controls.
Only allows basic mathematical operations and safe built-in functions.
"""

import ast
import asyncio
import logging
from typing import Any, Dict
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)
server = FastMCP("code_executor")

# Safe built-ins that are allowed for execution
SAFE_BUILTINS: Dict[str, Any] = {
    'abs': abs,
    'all': all,
    'any': any,
    'bool': bool,
    'dict': dict,
    'enumerate': enumerate,
    'float': float,
    'int': int,
    'len': len,
    'list': list,
    'max': max,
    'min': min,
    'pow': pow,
    'range': range,
    'reversed': reversed,
    'round': round,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'tuple': tuple,
    'zip': zip,
}

# Safe math functions
SAFE_MATH = {
    'pi': 3.141592653589793,
    'e': 2.718281828459045,
}

class SafeCodeValidator(ast.NodeVisitor):
    """AST visitor to validate code for safe execution."""
    
    ALLOWED_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
        ast.Name, ast.Load, ast.Call, ast.Compare, ast.BoolOp,
        ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
        ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
        ast.Pow, ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr,
        ast.BitXor, ast.Invert, ast.UAdd, ast.USub, ast.List,
        ast.Tuple, ast.Dict, ast.Set, ast.comprehension, ast.GeneratorExp,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.IfExp,
        ast.Subscript, ast.Index, ast.Slice, ast.For, ast.While,
        ast.If, ast.Break, ast.Continue, ast.Pass, ast.Return,
    }
    
    def __init__(self):
        self.errors = []
        
    def generic_visit(self, node):
        if type(node) not in self.ALLOWED_NODES:
            self.errors.append(f"Unsafe node type: {type(node).__name__}")
        super().generic_visit(node)
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id not in SAFE_BUILTINS:
                self.errors.append(f"Function '{node.func.id}' is not allowed")
        elif isinstance(node.func, ast.Attribute):
            self.errors.append(f"Attribute access '{ast.unparse(node.func)}' is not allowed")
        super().generic_visit(node)

@server.tool()
async def execute_python(code: str) -> str:
    """Execute basic Python code safely with proper sandboxing.
    
    Args:
        code: Python code to execute (expression mode only)
        
    Returns:
        Execution result or error message
        
    Security features:
        - AST validation to prevent unsafe operations
        - Restricted built-in functions
        - No file system or network access
        - No import statements allowed
        - No attribute access on objects
    """
    try:
        # Input validation
        if not code or not code.strip():
            return "Error: Empty code provided"
        
        code = code.strip()
        
        # Parse and validate the AST
        try:
            tree = ast.parse(code, mode='eval')
        except SyntaxError as e:
            return f"Syntax error: {str(e)}"
        
        # Validate for safety
        validator = SafeCodeValidator()
        validator.visit(tree)
        
        if validator.errors:
            return f"Security error: {'; '.join(validator.errors)}"
        
        # Prepare safe execution environment
        safe_globals = {
            '__builtins__': SAFE_BUILTINS,
            **SAFE_MATH,
        }
        
        # Execute the code
        result = eval(compile(tree, '<string>', 'eval'), safe_globals)
        
        # Format result safely
        if result is None:
            return "Execution completed successfully (no return value)"
        elif isinstance(result, (int, float, str, bool)):
            return f"Execution result: {result}"
        elif isinstance(result, (list, tuple, set)):
            # Limit output size for collections
            str_result = str(result)
            if len(str_result) > 1000:
                str_result = str_result[:1000] + "... (truncated)"
            return f"Execution result: {str_result}"
        elif isinstance(result, dict):
            # Limit output size for dictionaries
            str_result = str(result)
            if len(str_result) > 1000:
                str_result = str_result[:1000] + "... (truncated)"
            return f"Execution result: {str_result}"
        else:
            return f"Execution result: {type(result).__name__} object (type not fully supported for display)"
            
    except MemoryError:
        return "Execution error: Memory limit exceeded"
    except RecursionError:
        return "Execution error: Recursion limit exceeded"
    except OverflowError:
        return "Execution error: Numeric overflow"
    except ZeroDivisionError:
        return "Execution error: Division by zero"
    except ValueError as e:
        return f"Execution error: Invalid value - {str(e)}"
    except TypeError as e:
        return f"Execution error: Type error - {str(e)}"
    except Exception as e:
        logger.warning(f"Unexpected error in code execution: {e}")
        return f"Execution error: {type(e).__name__}: {str(e)}"

@server.tool()
async def validate_code(code: str) -> str:
    """Validate Python code for safe execution without running it.
    
    Args:
        code: Python code to validate
        
    Returns:
        Validation result indicating if code is safe to execute
    """
    try:
        if not code or not code.strip():
            return "Error: Empty code provided"
            
        tree = ast.parse(code.strip(), mode='eval')
        validator = SafeCodeValidator()
        validator.visit(tree)
        
        if validator.errors:
            return f"Unsafe: {'; '.join(validator.errors)}"
        
        return "Safe: Code passes security validation"
        
    except SyntaxError as e:
        return f"Syntax error: {str(e)}"
    except Exception as e:
        return f"Validation error: {type(e).__name__}: {str(e)}"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    server.run()