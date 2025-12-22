"""
Auto-generated MCP server for fetch::fetch_url
Generated at: 2025-12-22T13:33:24.190459
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

mcp = FastMCP("fetch::fetch_url")

class Fetch::FetchUrlInput(BaseModel):
    """Input schema for fetch::fetch_url"""
    url: str = Field(..., description="https://example.com")

@mcp.tool()
async def fetch::fetch_url(input: Fetch::FetchUrlInput) -> str:
    """
    fetch::fetch_url tool
    """
    try:
        api_key = os.getenv("FETCH::FETCH_URL_API_KEY")
        if not api_key:
            return json.dumps({"error": "API key not found. Set FETCH::FETCH_URL_API_KEY environment variable."})
        headers = {"Authorization": f"Bearer {api_key}"}

        url = f"https://api.example.com/api/v1/endpoint"
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
