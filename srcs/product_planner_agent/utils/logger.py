import logging
from mcp_agent.logging.logger import get_logger as get_base_logger

def get_product_planner_logger(sub_name: str) -> logging.Logger:
    """
    Returns a logger instance with the 'product_planner' prefix.
    This ensures all loggers within the Product Planner project have a unified,
    hierarchical naming convention for consistent log handling.
    
    Args:
        sub_name: The specific name for the sub-module or agent logger.
                  For example, 'prd_writer' or 'reporting_coordinator'.
                  If __name__ is passed, it will be structured automatically.
                  
    Returns:
        A configured logger instance.
    """
    # If a standard module path is passed, extract the relevant part
    if '.' in sub_name:
        parts = sub_name.split('.')
        if 'product_planner_agent' in parts:
            # e.g., srcs.product_planner_agent.agents.prd_writer_agent -> agents.prd_writer_agent
            try:
                index = parts.index('product_planner_agent')
                sub_name = '.'.join(parts[index+1:])
            except (ValueError, IndexError):
                pass # Fallback to using the full name if parsing fails

    return get_base_logger(f"product_planner.{sub_name}") 