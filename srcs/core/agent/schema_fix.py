"""
Schema fix for fetch-mcp and other MCP servers that use non-standard type formats.

This module provides a monkey patch for transform_mcp_tool_schema to handle
type fields that are lists (e.g., type: ['boolean', 'null']) instead of
the standard JSON Schema format (anyOf: [{type: 'boolean'}, {type: 'null'}]).
"""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def normalize_schema_type(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize schema types that are lists to standard JSON Schema anyOf format.

    Converts non-standard type formats like `type: ['boolean', 'null']`
    to the standard JSON Schema format using `anyOf`.

    Args:
        schema: The MCP tool schema dictionary to normalize

    Returns:
        Normalized schema dictionary with list-type fields converted to anyOf format

    Example:
        Input:  {"type": ["boolean", "null"]}
        Output: {"anyOf": [{"type": "boolean"}, {"type": "null"}]}
    """
    if not isinstance(schema, dict):
        return schema

    result = schema.copy()

    # Handle type field that is a list
    if "type" in result and isinstance(result["type"], list):
        type_list = result["type"]

        # Convert to anyOf format
        any_of = [{"type": t} for t in type_list if isinstance(t, str)]

        if any_of:
            # Remove the list type
            del result["type"]
            # Add anyOf
            result["anyOf"] = any_of
            logger.debug(f"Normalized type list {type_list} to anyOf format")

    # Recursively process nested structures
    if "properties" in result and isinstance(result["properties"], dict):
        result["properties"] = {
            k: normalize_schema_type(v)
            for k, v in result["properties"].items()
        }

    if "items" in result and isinstance(result["items"], dict):
        result["items"] = normalize_schema_type(result["items"])

    if "anyOf" in result and isinstance(result["anyOf"], list):
        result["anyOf"] = [normalize_schema_type(item) if isinstance(item, dict) else item
                          for item in result["anyOf"]]

    return result


def patch_transform_mcp_tool_schema() -> None:
    """
    Monkey patch transform_mcp_tool_schema to handle list-type schemas.
    
    Patches the transform function in mcp_agent.workflows.llm.augmented_llm_google
    to handle schema type fields that are lists (e.g., type: ['boolean', 'null'])
    instead of the standard JSON Schema format.
    
    Note:
        If patching fails, the function logs a warning and continues without
        the patch to avoid breaking the application.
    """
    try:
        from mcp_agent.workflows.llm import augmented_llm_google

        original_transform = augmented_llm_google.transform_mcp_tool_schema

        def patched_transform(schema: Dict[str, Any]) -> Dict[str, Any]:
            # Normalize schema before passing to original function
            normalized = normalize_schema_type(schema)
            return original_transform(normalized)

        # Apply monkey patch
        augmented_llm_google.transform_mcp_tool_schema = patched_transform
        logger.info("Successfully patched transform_mcp_tool_schema to handle list-type schemas")

    except Exception as e:
        logger.warning(f"Failed to patch transform_mcp_tool_schema: {e}")
        # Don't raise - allow code to continue without patch
