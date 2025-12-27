"""
Auto-generated MCP server for alpha_vantage_service
Generated at: 2025-12-27T13:17:13.335853
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

mcp = FastMCP("alpha_vantage_service")

class FinancialAgent::RunFinancialAnalysisInput(BaseModel):
    """Input schema for financial_agent::run_financial_analysis"""
    user_query: str = Field(..., description="The natural language query for financial analysis. It must contain a recognizable company name or stock ticker (e.g., 'NVDA', 'Apple', 'MSFT'). The server will parse the ticker and perform a comprehensive analysis by fetching company overview, news sentiment, and recent performance.")

@mcp.tool()
async def financial_agent::run_financial_analysis(input: FinancialAgent::RunFinancialAnalysisInput) -> str:
    """
    financial_agent::run_financial_analysis tool
    """
    try:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            return json.dumps({"error": "API key not found. Set ALPHA_VANTAGE_API_KEY environment variable."})
        headers = {"Authorization": f"Bearer {api_key}"}

        url = f"https://www.alphavantage.co/query"
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
        error_msg = f"Error in financial_agent::run_financial_analysis: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

if __name__ == "__main__":
    mcp.run()
