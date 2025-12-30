"""
Auto-generated MCP server for github
Generated at: 2025-12-30T15:06:06.389060
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

mcp = FastMCP("github")

class Github::CreatePullRequestInput(BaseModel):
    """Input schema for github::create_pull_request"""
    owner: str = Field(..., description="The account owner of the repository. The name is not case sensitive.")
    repo: str = Field(..., description="The name of the repository. The name is not case sensitive.")
    title: str = Field(..., description="The title of the new pull request.")
    head: str = Field(..., description="The name of the branch where your changes are implemented (the source branch).")
    base: str = Field(..., description="The name of the branch you want the changes pulled into (the target branch).")
    body: Optional[str] = Field(default=None, description="The contents of the pull request.")
    draft: Optional[str] = Field(default=false, description="Indicates whether the pull request is a draft.")

@mcp.tool()
async def github::create_pull_request(input: Github::CreatePullRequestInput) -> str:
    """
    github::create_pull_request tool
    """
    try:
        api_key = os.getenv("GITHUB_TOKEN")
        if not api_key:
            return json.dumps({"error": "API key not found. Set GITHUB_TOKEN environment variable."})
        headers = {"Authorization": f"Bearer {api_key}"}

        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        data = input.model_dump(exclude_none=True)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()

            result = response.json()
            return json.dumps(result, ensure_ascii=False, indent=2)

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})
    except Exception as e:
        error_msg = f"Error in github::create_pull_request: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

if __name__ == "__main__":
    mcp.run()
