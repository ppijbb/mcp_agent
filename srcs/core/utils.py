"""
Common utilities for MCP Agent system.

Provides shared utility functions including JSON encoding with support
for dataclasses, datetime objects, and custom value attributes.
"""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Enhanced JSON encoder that handles special types.
    
    Extends json.JSONEncoder to support:
    - dataclasses (converted to dict)
    - datetime objects (converted to ISO format strings)
    - objects with a 'value' attribute
    
    Usage:
        json.dumps(data, cls=EnhancedJSONEncoder)
    """
    
    def default(self, o):
        """
        Convert non-serializable objects to JSON-serializable format.

        Args:
            o: Object to serialize

        Returns:
            JSON-serializable representation of the object

        Handles:
            - dataclasses: Converted to dictionary using asdict()
            - datetime objects: Converted to ISO format strings
            - objects with 'value' attribute: Uses the value attribute
        """
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, 'value'):
            return o.value
        return super().default(o)
