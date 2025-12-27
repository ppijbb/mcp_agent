"""
Auto-generated MCP server for financial_analysis_service
Generated at: 2025-12-27T13:14:43.807573
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

mcp = FastMCP("financial_analysis_service")

class FinancialAgent::RunFinancialAnalysisInput(BaseModel):
    """Input schema for financial_agent::run_financial_analysis"""
    user_query: str = Field(..., description="The user's natural language query for financial analysis. Should include the company name or ticker symbol (e.g., 'NVDA 주식 투자 전략 분석').")
    ticker_symbols: Optional[str] = Field(default=None, description="An optional list of specific ticker symbols to focus the analysis on. If not provided, the service will attempt to extract them from the user_query.")
    output_language: Optional[str] = Field(default="en", description="The desired language for the analysis output, specified as an ISO 639-1 code (e.g., 'en', 'ko').")

@mcp.tool()
async def financial_agent::run_financial_analysis(input: FinancialAgent::RunFinancialAnalysisInput) -> str:
    """
    financial_agent::run_financial_analysis tool
    """
    try:
        api_key = os.getenv("FINANCIAL_ANALYSIS_API_KEY")
        if not api_key:
            return json.dumps({"error": "API key not found. Set FINANCIAL_ANALYSIS_API_KEY environment variable."})
        headers = {"Authorization": f"Bearer {api_key}"}

        url = f"https://api.internal.financial-analysis.com/v1/analysis"
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
