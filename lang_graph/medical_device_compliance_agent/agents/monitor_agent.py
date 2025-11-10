"""
Monitor Agent

지속적인 규제 변경 모니터링 및 알림
"""

import logging
from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper
from ..tools.regulatory_tools import RegulatoryTools

logger = logging.getLogger(__name__)


class MonitorAgent:
    """규제 변경 모니터링 Agent"""
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None
    ):
        self.model_manager = model_manager
        self.fallback_handler = fallback_handler
        self.preferred_provider = preferred_provider
        
        self.mcp_tools = MCPToolsWrapper()
        self.regulatory_tools = RegulatoryTools()
        self.tools = self.mcp_tools.get_tools() + self.regulatory_tools.get_tools()
        
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        self._initialize_agent()
    
    def _initialize_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a regulatory monitoring agent for medical devices.

Monitor regulatory changes and updates from:
- FDA website
- EU regulatory bodies
- ISO standards updates
- Industry guidance documents

Identify changes that may affect device compliance and provide alerts."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
    
    def monitor(self, frameworks: List[str]) -> List[Dict[str, Any]]:
        """규제 변경 모니터링"""
        try:
            input_text = f"Monitor regulatory changes for: {', '.join(frameworks)}"
            result = self.agent_executor.invoke({"input": input_text})
            return [{"framework": f, "changes": []} for f in frameworks]
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            return []

