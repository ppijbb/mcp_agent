"""
Auto-generated MCP server for github_api
Generated at: 2025-12-31T11:12:20.225133
"""

import os
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path

try:
    from fastmcp import FastMCP
    from pydantic import BaseModel, Field
    import httpx
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}. Install with: pip install fastmcp pydantic httpx")

logger = logging.getLogger(__name__)

mcp = FastMCP("github_api")

class Github::SearchCodeInput(BaseModel):
    """Input schema for github::search_code"""
    query: str = Field(..., description="The search keywords, as well as any qualifiers (e.g., 'user:openai MCP').")
    limit: Optional[int] = Field(default=5, description="The maximum number of code results to return. Maps to the 'per_page' API parameter.")

@mcp.tool()
async def github::search_code(input: Github::SearchCodeInput) -> str:
    """
    github::search_code tool
    """
    try:
        api_key = os.getenv("GITHUB_API_TOKEN")
        if not api_key:
            return json.dumps({"error": "API key not found. Set GITHUB_API_TOKEN environment variable."})
        headers = {"Authorization": f"Bearer {api_key}"}

        url = f"https://api.github.com/search/code"
        params = input.model_dump(exclude_none=True)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()

            result = response.json()
            return json.dumps(result, ensure_ascii=False, indent=2)

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})
    except Exception as e:
        error_msg = f"Error in github::search_code: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

if __name__ == "__main__":
    mcp.run()
