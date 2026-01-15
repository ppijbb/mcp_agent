import asyncio
from mcp import Tool
from mcp.server import Server

server = Server("code_executor")

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
    import mcp.server.stdio
    mcp.server.stdio.run_server(server)