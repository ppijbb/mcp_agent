"""
Auto-generated MCP server for tavily_search_server
Generated at: 2025-12-27T13:12:00.299289
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

mcp = FastMCP("tavily_search_server")

class FinancialAgent::RunFinancialAnalysisInput(BaseModel):
    """Input schema for financial_agent::run_financial_analysis"""
    user_query: str = Field(..., description="The detailed financial analysis query to research. For example: '2024년 한국 주식 시장 전망과 투자 전략 분석'. This will be mapped to the 'query' parameter of the backend API.")
    search_depth: Optional[str] = Field(default="advanced", description="The depth of the search. Use 'basic' for a quick search or 'advanced' for a more in-depth, slower analysis that may consume more credits.")
    include_answer: Optional[str] = Field(default=true, description="Whether to include a synthesized answer to the query in the response.")
    max_results: Optional[str] = Field(default=7, description="The maximum number of search results to return.")

@mcp.tool()
async def financial_agent::run_financial_analysis(input: FinancialAgent::RunFinancialAnalysisInput) -> str:
    """
    financial_agent::run_financial_analysis tool
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return json.dumps({"error": "API key not found. Set TAVILY_API_KEY environment variable."})
        headers = {"Authorization": f"Bearer {api_key}"}

        url = f"https://api.tavily.com/search"
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
        error_msg = f"Error in financial_agent::run_financial_analysis: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

if __name__ == "__main__":
    mcp.run()
