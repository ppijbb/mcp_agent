"""
Base Product Planner Agent

Common base class for product planner agents to reduce code duplication.
"""

from typing import Dict, Any
from srcs.core.agent.base import BaseAgent
from srcs.core.errors import APIError, WorkflowError


class BaseProductPlannerAgent(BaseAgent):
    """Base class for product planner agents with common functionality."""
    
    def __init__(self, agent_name: str):
        """Initialize the agent with a name."""
        super().__init__(agent_name)
    
    async def run_workflow(self, context: Any) -> Dict[str, Any]:
        """
        Execute the agent's workflow.
        
        Args:
            context: Workflow context and parameters
            
        Returns:
            Dict containing workflow results
            
        Raises:
            WorkflowError: If workflow execution fails
            APIError: If external API calls fail
        """
        try:
            return await self._execute_workflow(context)
        except Exception as e:
            if isinstance(e, (WorkflowError, APIError)):
                raise
            raise WorkflowError(f"Workflow execution failed in {self.__class__.__name__}: {str(e)}")
    
    async def _execute_workflow(self, context: Any) -> Dict[str, Any]:
        """
        Override this method in subclasses to implement specific workflow logic.
        
        Args:
            context: Workflow context and parameters
            
        Returns:
            Dict containing workflow results
        """
        raise NotImplementedError("Subclasses must implement _execute_workflow method")
    
    def _validate_context(self, context: Any, required_fields: list = None) -> None:
        """
        Validate workflow context parameters.
        
        Args:
            context: Context to validate
            required_fields: List of required field names
            
        Raises:
            WorkflowError: If validation fails
        """
        if required_fields and isinstance(context, dict):
            missing_fields = [field for field in required_fields if field not in context]
            if missing_fields:
                raise WorkflowError(f"Missing required fields in context: {missing_fields}")