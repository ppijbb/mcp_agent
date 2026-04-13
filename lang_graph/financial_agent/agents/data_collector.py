"""
Market Data Collector Node

Collects technical analysis data from MCP servers for financial analysis workflows.
"""

from typing import Dict
from ..state import AgentState
from ..mcp_client import call_technical_indicators_tool


def market_data_collector_node(state: AgentState) -> Dict:
    """
    Market data collector node: Collects technical analysis data via MCP server.

    Fetches technical indicators for specified tickers in parallel and validates
    the returned data before passing it to the next node in the workflow.

    Args:
        state: AgentState containing target tickers and processing context

    Returns:
        Dict containing technical_analysis data or error_message

    Raises:
        ValueError: If data collection fails or validation errors occur
    """
    print("--- AGENT: Market Data Collector (MCP-Powered) ---")
    log_message = "Starting real-time market data collection via MCP."
    state["log"].append(log_message)
    print(log_message)

    tickers = state.get("target_tickers", [])
    if not tickers:
        error_message = "No tickers specified for analysis."
        print(error_message)
        state["log"].append(error_message)
        return {"error_message": error_message}

    # Concurrent call support: request all tickers at once
    print(f"Fetching technical data for {tickers} via MCP (concurrent)...")

    try:
        all_technicals = call_technical_indicators_tool(tickers)

        # Data validation
        if not all_technicals:
            raise ValueError("Technical analysis data is empty.")

        # Validate each ticker's data
        for ticker in tickers:
            if ticker not in all_technicals:
                raise ValueError(f"No technical analysis data for {ticker}.")

            ticker_data = all_technicals[ticker]
            if "error" in ticker_data:
                raise ValueError(f"{ticker} data collection error: {ticker_data['error']}")

        log_message = f"Technical analysis data collection complete for {len(tickers)} tickers."
        state["log"].append(log_message)
        print(log_message)

        return {"technical_analysis": all_technicals}
    except Exception as e:
        error_message = f"Error during technical analysis data collection: {e}"
        print(error_message)
        state["log"].append(error_message)
        raise ValueError(error_message)
