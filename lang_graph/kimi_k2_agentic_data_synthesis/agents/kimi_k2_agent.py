"""
Kimi-K2 Conversable Agent for LangGraph integration.

This agent extends autogen.ConversableAgent to be compatible with the Kimi-K2 system's
agent configurations and tool registry.
"""

from autogen import ConversableAgent
from typing import Dict, Any, List, Optional
from models.agent import AgentConfig, BehaviorPattern
from models.tool import Tool
import logging

logger = logging.getLogger(__name__)

class KimiK2ConversableAgent(ConversableAgent):
    """
    A ConversableAgent wrapper for Kimi-K2 agents, integrating with AgentConfig
    and the ToolRegistry.
    """
    def __init__(
        self,
        agent_config: AgentConfig,
        llm_config: Optional[Dict[str, Any]] = None,
        tool_registry: Optional[Any] = None, # Will be ToolRegistry instance
        **kwargs
    ):
        self.agent_config = agent_config
        self.tool_registry = tool_registry
        
        system_message = self._build_system_message(agent_config)
        
        super().__init__(
            name=agent_config.agent_id,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs,
        )
        
        self._register_tools()
        logger.info(f"KimiK2ConversableAgent '{self.agent_config.name}' initialized.")

    def _build_system_message(self, config: AgentConfig) -> str:
        """
        Builds a comprehensive system message for the agent based on its configuration.
        """
        system_message = f"""
You are {config.name}, a {config.description}. Your role is to assist in complex tasks.

Expertise Domains: {', '.join(config.expertise_domains)}
Tool Preferences: {', '.join(config.tool_preferences)}
Communication Style: {config.communication_style}
Problem-Solving Approach: {config.problem_solving_approach}
Collaboration Style: {config.collaboration_style}

Your goal is to contribute effectively to the given task, utilizing your skills and preferred tools.
"""
        return system_message

    def _register_tools(self) -> None:
        """
        Registers tools from the tool registry that the agent prefers.
        """
        if not self.tool_registry:
            logger.warning(f"No tool registry provided for agent {self.agent_config.name}. Cannot register tools.")
            return

        for tool_id in self.agent_config.tool_preferences:
            tool_instance = self.tool_registry.get_tool(tool_id)
            if tool_instance:
                # Create a proper tool callable that connects to the tool registry
                def create_tool_callable(tool_name: str, tool_registry: Any):
                    def tool_callable(*args, **kwargs):
                        logger.info(f"Agent {self.agent_config.name} calling tool: {tool_name} with args: {args}, kwargs: {kwargs}")
                        
                        # Convert args and kwargs to parameters dict
                        parameters = {}
                        if args:
                            # If args provided, assume they match the first parameter
                            param_names = list(tool_instance.parameters.keys()) if tool_instance.parameters else []
                            if param_names:
                                parameters[param_names[0]] = args[0]
                        
                        # Add kwargs to parameters
                        parameters.update(kwargs)
                        
                        # Execute the tool through the registry
                        result = tool_registry.execute_tool(tool_name, parameters)
                        
                        if "error" in result:
                            logger.error(f"Tool {tool_name} execution failed: {result['error']}")
                            return f"Error: {result['error']}"
                        else:
                            logger.info(f"Tool {tool_name} executed successfully: {result.get('result', 'No result')}")
                            return result.get('result', 'Tool executed successfully')
                    
                    return tool_callable
                
                # Register the tool with the conversable agent
                tool_callable = create_tool_callable(tool_instance.name, self.tool_registry)
                # Temporarily disable tool registration to avoid API issues
                # self.register_for_llm(tool_callable, description=tool_instance.description)
                logger.info(f"Tool '{tool_instance.name}' would be registered for agent '{self.agent_config.name}' (temporarily disabled).")
            else:
                logger.warning(f"Tool '{tool_id}' not found in registry for agent {self.agent_config.name}.")

    async def a_run_task(self, task_message: str, **kwargs) -> Dict[str, Any]:
        """
        Asynchronously runs a task for this agent.
        This can be called as a node in LangGraph.
        """
        logger.info(f"Agent '{self.agent_config.name}' receiving task: {task_message}")
        # Simulate agent processing and response generation
        # In a real AutoGen setup, this would be handled by the group chat manager or direct message
        
        # For now, a simplified direct response
        response_content = f"Agent {self.agent_config.name} processed task: '{task_message}'."
        
        # If the agent needs to call tools based on the task, that logic would be here.
        # For demonstration, we'll just log and return.
        
        return {"content": response_content, "agent_id": self.agent_config.agent_id} 