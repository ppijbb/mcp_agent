"""
Auto-generated MCP server for fetch_service
Generated at: 2025-12-22T13:32:02.574661
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

mcp = FastMCP("fetch_service")

class Fetch::FetchUrlInput(BaseModel):
    """Input schema for fetch::fetch_url"""
    url: str = Field(..., description="The URL to be fetched")

@mcp.tool()
async def fetch::fetch_url(input: Fetch::FetchUrlInput) -> str:
    """
    fetch::fetch_url tool
    """
    try:
        headers = {}

        url = f"variablevariable"
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
        error_msg = f"Error in fetch::fetch_url: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

if __name__ == "__main__":
    mcp.run()
