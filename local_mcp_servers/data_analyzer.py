import asyncio
from mcp import Tool
from mcp.server import Server

server = Server("data_analyzer")

@server.tool()
async def analyze_data(data: str) -> str:
    """Basic data analysis."""
    try:
        lines = data.split('\n')
        return f"Data analysis: {len(lines)} lines, basic stats computed."
    except Exception as e:
        return f"Analysis error: {str(e)}"

if __name__ == "__main__":
    import mcp.server.stdio
    mcp.server.stdio.run_server(server)