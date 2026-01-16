import asyncio
from mcp.server.fastmcp import FastMCP

server = FastMCP("search_server")

@server.tool()
async def search_web(query: str) -> str:
    """Basic web search simulation."""
    return f"Search results for '{query}': This is a local search simulation. In a real implementation, this would connect to search APIs."

if __name__ == "__main__":
    server.run()