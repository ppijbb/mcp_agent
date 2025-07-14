"""
Kimi-K2 Conversable Agent for LangGraph integration.

This agent extends autogen.ConversableAgent to be compatible with the Kimi-K2 system's
agent configurations and tool registry.
"""

from autogen import ConversableAgent
from typing import Dict, Any, List, Optional
from ..models.agent import AgentConfig, BehaviorPattern
from ..models.tool import Tool
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
                # In a real scenario, you'd integrate the actual tool callable here.
                # For now, we'll use a placeholder function.
                def generic_tool_callable(*args, **kwargs):
                    logger.info(f"Agent {self.agent_config.name} calling placeholder tool: {tool_instance.name} with args: {args}, kwargs: {kwargs}")
                    # Simulate tool execution result
                    return {"status": "success", "message": f"Simulated execution of {tool_instance.name}"}
                
                self.register_for_llm(generic_tool_callable, description=tool_instance.description)
                logger.info(f"Registered tool '{tool_instance.name}' for agent '{self.agent_config.name}'.")
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