"""Utility helper functions for Product Planner Agent."""

from datetime import datetime
from typing import Any, Dict, Optional


def format_date(date: datetime, format_str: str = "%Y-%m-%d") -> str:
    """Format a datetime object as a string.

    Args:
        date: The datetime object to format.
        format_str: The format string to use.

    Returns:
        Formatted date string.
    """
    return date.strftime(format_str)


def parse_date(date_str: str, format_str: str = "%Y-%m-%d") -> Optional[datetime]:
    """Parse a date string into a datetime object.

    Args:
        date_str: The date string to parse.
        format_str: The expected format string.

    Returns:
        Parsed datetime object or None if parsing fails.
    """
    try:
        return datetime.strptime(date_str, format_str)
    except ValueError:
        return None


def sanitize_filename(name: str) -> str:
    """Convert a string to a safe filename.

    Args:
        name: The string to sanitize.

    Returns:
        A safe filename string.
    """
    import re
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.

    Args:
        base: The base dictionary.
        override: Dictionary with values to override.

    Returns:
        Merged dictionary with override values taking precedence.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
