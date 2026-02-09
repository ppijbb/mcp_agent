"""
Enhanced Data Analyzer MCP Server with proper validation and error handling.

Provides data analysis capabilities with input validation, comprehensive error handling,
and detailed analysis results for various data formats.
"""

import asyncio
import logging
from typing import Dict, Any, List
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)
server = FastMCP("data_analyzer")

@server.tool()
async def analyze_data(data: str) -> str:
    """Analyze data with comprehensive validation and detailed statistics.
    
    Args:
        data: Input data string to analyze (non-empty, max 100KB)
        
    Returns:
        Detailed analysis results including line count, word count, character statistics,
        and basic data patterns
        
    Raises:
        ValueError: If input data is invalid or too large
        
    Security features:
        - Input validation and size limits
        - Safe data processing without execution
        - Comprehensive error logging
    """
    try:
        # Input validation
        if not data or not data.strip():
            raise ValueError("Input data cannot be empty")
        
        if len(data.encode('utf-8')) > 100 * 1024:  # 100KB limit
            raise ValueError("Input data too large (max 100KB)")
        
        # Basic statistics
        lines = data.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        words = data.split()
        characters = len(data)
        characters_no_spaces = len(data.replace(' ', ''))
        
        # Data patterns
        numeric_count = sum(1 for word in words if word.replace('.', '').replace('-', '').isdigit())
        alpha_count = sum(1 for word in words if word.isalpha())
        alnum_count = sum(1 for word in words if word.isalnum())
        
        # Line length statistics
        line_lengths = [len(line) for line in non_empty_lines]
        avg_line_length = sum(line_lengths) / len(line_lengths) if line_lengths else 0
        max_line_length = max(line_lengths) if line_lengths else 0
        
        # Format results
        result = [
            "📊 Data Analysis Results",
            "=" * 30,
            f"Total lines: {len(lines)}",
            f"Non-empty lines: {len(non_empty_lines)}",
            f"Total words: {len(words)}",
            f"Total characters: {characters}",
            f"Characters (no spaces): {characters_no_spaces}",
            "",
            "📈 Content Statistics:",
            f"Numeric words: {numeric_count}",
            f"Alphabetic words: {alpha_count}",
            f"Alphanumeric words: {alnum_count}",
            "",
            "📏 Line Statistics:",
            f"Average line length: {avg_line_length:.1f} characters",
            f"Maximum line length: {max_line_length} characters",
            "",
            "✅ Analysis completed successfully"
        ]
        
        logger.info(f"Data analysis completed: {len(lines)} lines, {len(words)} words")
        return "\n".join(result)
        
    except ValueError as e:
        error_msg = f"Validation error: {str(e)}"
        logger.warning(f"Data analysis validation failed: {error_msg}")
        return error_msg
        
    except Exception as e:
        error_msg = f"Analysis error: {type(e).__name__}: {str(e)}"
        logger.error(f"Unexpected data analysis error: {e}", exc_info=True)
        return error_msg

@server.tool()
async def validate_data_format(data: str, format_type: str = "text") -> str:
    """Validate data format and structure.
    
    Args:
        data: Input data string to validate
        format_type: Expected format type ('text', 'json', 'csv', 'xml')
        
    Returns:
        Validation result with format-specific checks and recommendations
    """
    try:
        if not data or not data.strip():
            return "Invalid: Empty data provided"
        
        format_type = format_type.lower().strip()
        valid_formats = {'text', 'json', 'csv', 'xml'}
        
        if format_type not in valid_formats:
            return f"Invalid: Unsupported format '{format_type}'. Supported formats: {', '.join(valid_formats)}"
        
        issues = []
        suggestions = []
        
        if format_type == 'json':
            import json
            try:
                json.loads(data)
                return "✅ Valid JSON format"
            except json.JSONDecodeError as e:
                issues.append(f"JSON syntax error: {str(e)}")
                suggestions.append("Check for missing commas, brackets, or quotes")
        
        elif format_type == 'csv':
            lines = [line for line in data.split('\n') if line.strip()]
            if len(lines) < 2:
                issues.append("CSV appears to have insufficient data (need header + data)")
                suggestions.append("Ensure CSV has header row and at least one data row")
            
            # Check for consistent column count
            header_cols = len(lines[0].split(','))
            for i, line in enumerate(lines[1:], 1):
                cols = len(line.split(','))
                if cols != header_cols:
                    issues.append(f"Row {i} has {cols} columns, expected {header_cols}")
                    suggestions.append("Ensure all rows have the same number of columns")
                    break
        
        elif format_type == 'xml':
            import xml.etree.ElementTree as ET
            try:
                ET.fromstring(data)
                return "✅ Valid XML format"
            except ET.ParseError as e:
                issues.append(f"XML syntax error: {str(e)}")
                suggestions.append("Check for missing closing tags or malformed elements")
        
        else:  # text
            if len(data) > 50000:
                issues.append("Text file is very large")
                suggestions.append("Consider splitting into smaller files for better performance")
        
        if not issues:
            return f"✅ Valid {format_type.upper()} format"
        
        result = [
            f"❌ {format_type.upper()} format validation issues:",
            *[f"- {issue}" for issue in issues],
            "",
            f"💡 Suggestions:",
            *[f"- {suggestion}" for suggestion in suggestions]
        ]
        
        return "\n".join(result)
        
    except Exception as e:
        error_msg = f"Validation error: {type(e).__name__}: {str(e)}"
        logger.error(f"Data format validation error: {e}", exc_info=True)
        return error_msg

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting Data Analyzer MCP Server")
    server.run()
