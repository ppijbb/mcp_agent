import asyncio
from mcp.server.fastmcp import FastMCP

server = FastMCP("code_executor")

@server.tool()
async def execute_python(code: str) -> str:
    """Execute basic Python code safely."""
    try:
        # Very basic execution - in production, use proper sandboxing
        result = eval(code, {"__builtins__": {}})
        return f"Execution result: {result}"
    except Exception as e:
        return f"Execution error: {str(e)}"

if __name__ == "__main__":
    server.run()