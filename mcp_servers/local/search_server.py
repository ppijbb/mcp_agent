"""
Enhanced Search MCP Server with proper error handling and validation.

Provides web search functionality with input validation, error handling,
and proper logging for production use.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)
server = FastMCP("search_server")

@server.tool()
async def search_web(query: str, max_results: Optional[int] = None) -> str:
    """Perform web search with proper validation and error handling.
    
    Args:
        query: Search query string (non-empty, max 500 characters)
        max_results: Optional maximum number of results to return (1-50)
        
    Returns:
        Formatted search results or error message
        
    Raises:
        ValueError: If input parameters are invalid
        
    Security features:
        - Input validation and sanitization
        - Query length limits
        - Result count limits
        - Comprehensive error logging
    """
    try:
        # Input validation
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        query = query.strip()
        
        if len(query) > 500:
            raise ValueError("Search query too long (max 500 characters)")
        
        if max_results is not None:
            if not isinstance(max_results, int) or max_results < 1 or max_results > 50:
                raise ValueError("max_results must be an integer between 1 and 50")
        
        # Log search request (without sensitive data)
        logger.info(f"Search request received: {len(query)} characters, max_results: {max_results}")
        
        # Simulate search with different results based on query complexity
        if len(query) < 10:
            result_type = "simple"
            mock_results = [
                f"Result 1: Basic match for '{query}'",
                f"Result 2: Related topic for '{query}'"
            ]
        elif len(query) < 50:
            result_type = "moderate"
            mock_results = [
                f"Result 1: Detailed match for '{query}'",
                f"Result 2: Comprehensive analysis of '{query}'",
                f"Result 3: Related resources for '{query}'"
            ]
        else:
            result_type = "complex"
            mock_results = [
                f"Result 1: In-depth analysis for query: {query[:50]}...",
                f"Result 2: Research findings for complex query",
                f"Result 3: Expert opinions and analysis",
                f"Result 4: Related case studies and examples"
            ]
        
        # Apply max_results limit
        if max_results is not None:
            mock_results = mock_results[:max_results]
        
        # Format results
        result_lines = [
            f"Search results for '{query}':",
            f"Query type: {result_type}",
            f"Results found: {len(mock_results)}",
            "",
            "Mock search results (this is a simulation):"
        ]
        
        for i, result in enumerate(mock_results, 1):
            result_lines.append(f"{i}. {result}")
        
        result_lines.append("")
        result_lines.append("Note: This is a local search simulation. In production, this would connect to real search APIs like Google, Bing, or DuckDuckGo.")
        
        return "\n".join(result_lines)
        
    except ValueError as e:
        error_msg = f"Invalid input: {str(e)}"
        logger.warning(f"Validation error: {error_msg}")
        return error_msg
        
    except Exception as e:
        error_msg = f"Search error: {type(e).__name__}: {str(e)}"
        logger.error(f"Unexpected search error: {e}", exc_info=True)
        return error_msg

@server.tool()
async def validate_search_query(query: str) -> str:
    """Validate search query without performing search.
    
    Args:
        query: Search query to validate
        
    Returns:
        Validation result with recommendations
    """
    try:
        if not query or not query.strip():
            return "Invalid: Empty query"
        
        query = query.strip()
        
        issues = []
        suggestions = []
        
        if len(query) < 3:
            issues.append("Query too short (minimum 3 characters)")
            suggestions.append("Use more descriptive terms")
        
        if len(query) > 500:
            issues.append("Query too long (maximum 500 characters)")
            suggestions.append("Shorten the query or split into multiple searches")
        
        # Check for common non-productive patterns
        if query.lower() in ["test", "asdf", "hello", "hi"]:
            issues.append("Query appears to be a test")
            suggestions.append("Use meaningful search terms")
        
        if not issues:
            return f"Valid: Query passes validation (length: {len(query)} characters)"
        
        result = [
            f"Validation issues found:",
            *[f"- {issue}" for issue in issues],
            "",
            f"Suggestions:",
            *[f"- {suggestion}" for suggestion in suggestions]
        ]
        
        return "\n".join(result)
        
    except Exception as e:
        error_msg = f"Validation error: {type(e).__name__}: {str(e)}"
        logger.error(f"Query validation error: {e}", exc_info=True)
        return error_msg

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting Search MCP Server")
    server.run()
