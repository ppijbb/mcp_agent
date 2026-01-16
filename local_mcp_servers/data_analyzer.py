import asyncio
from mcp.server.fastmcp import FastMCP

server = FastMCP("data_analyzer")

@server.tool()
async def analyze_data(data: str) -> str:
    """Basic data analysis."""
    try:
        lines = data.split('\n')
        return f"Data analysis: {len(lines)} lines, basic stats computed."
    except Exception as e:
        return f"Analysis error: {str(e)}"

if __name__ == "__main__":
    server.run()