"""
LangGraph Agent A2A Adapter.

A2A (Agent-to-Agent) wrapper for LangGraph-based agents in the lang_graph/
directory. Provides message passing, capability registration, and state
serialization for LangGraph applications.

Classes:
    LangGraphAgentA2AWrapper: A2A adapter for LangGraph agents

Example:
    wrapper = LangGraphAgentA2AWrapper(
        agent_id="analysis_agent",
        agent_metadata={"name": "Analysis Agent"},
        graph_app=graph_app
    )
    await wrapper.start_listener()
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

# 상위 디렉토리의 공통 모듈 import
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.a2a_integration import (
    A2AAdapter,
    A2AMessage,
    MessagePriority,
    get_global_broker,
    get_global_registry,
)
from srcs.common.agent_interface import AgentType

logger = logging.getLogger(__name__)


class LangGraphAgentA2AWrapper(A2AAdapter):
    """
    A2A adapter for LangGraph-based agents.
    
    Extends A2AAdapter to provide LangGraph-specific functionality including
    graph execution, state management, and message processing.
    
    Attributes:
        agent_id: Unique identifier for the agent
        agent_metadata: Metadata dictionary containing agent information
        graph_app: LangGraph application instance
        state_manager: Optional callable for managing agent state
        is_listening: Whether the message listener is active
        _message_processor_task: Background task for processing messages
        _current_state: Current state of the agent
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_metadata: Dict[str, Any],
        graph_app: Any = None,
        state_manager: Optional[Callable] = None
    ):
        """
        Initialize the LangGraph A2A wrapper.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_metadata: Dictionary containing agent metadata
            graph_app: LangGraph application instance (optional)
            state_manager: Optional callable for managing agent state
        """
        super().__init__(agent_id, agent_metadata)
        self.graph_app = graph_app
        self.state_manager = state_manager
        self._message_processor_task: Optional[asyncio.Task] = None
        self._current_state: Optional[Dict[str, Any]] = None
    
    async def send_message(
        self,
        target_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: int = MessagePriority.MEDIUM.value,
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        Send a message to another agent.
        
        Args:
            target_agent: Target agent ID (empty string for broadcast)
            message_type: Type of message being sent
            payload: Message payload data
            priority: Message priority (default: MEDIUM)
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            True if message was routed successfully
        """
        message = A2AMessage(
            source_agent=self.agent_id,
            target_agent=target_agent,
            message_type=message_type,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
        )
        
        broker = get_global_broker()
        return await broker.route_message(message)
    
    async def start_listener(self) -> None:
        """
        Start the message listener for this agent.
        
        Creates and starts a background task that processes incoming messages
        from the message queue.
        """
        if self.is_listening:
            logger.warning(f"Listener already started for agent {self.agent_id}")
            return
        
        self.is_listening = True
        self._message_processor_task = asyncio.create_task(self._process_messages())
        logger.info(f"Message listener started for LangGraph agent {self.agent_id}")
    
    async def stop_listener(self) -> None:
        """
        Stop the message listener for this agent.
        
        Cancels the background message processing task and marks the
        listener as inactive.
        """
        if not self.is_listening:
            return
        
        self.is_listening = False
        if self._message_processor_task:
            self._message_processor_task.cancel()
            try:
                await self._message_processor_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Message listener stopped for LangGraph agent {self.agent_id}")
    
    async def _process_messages(self) -> None:
        """
        Message processing loop.
        
        Continuously listens for messages from the queue and processes them
        using registered message handlers. Runs until is_listening is False.
        """
        while self.is_listening:
            try:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self.handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message in LangGraph agent: {e}")
    
    async def register_capabilities(self, capabilities: List[str]) -> None:
        """
        Register agent capabilities with the global registry.
        
        Args:
            capabilities: List of capability strings this agent supports
        """
        self.agent_metadata["capabilities"] = capabilities
        registry = get_global_registry()
        await registry.register_agent(
            agent_id=self.agent_id,
            agent_type=AgentType.LANGGRAPH_AGENT.value,
            metadata=self.agent_metadata,
            a2a_adapter=self,
        )
        logger.info(f"Capabilities registered for LangGraph agent {self.agent_id}: {capabilities}")
    
    async def execute_graph(
        self,
        input_data: Dict[str, Any],
        stream: bool = False
    ) -> Any:
        """
        Execute the LangGraph with given input data.
        
        Args:
            input_data: Input data (can be LangGraph state format or plain dict)
            stream: Whether to use streaming mode
            
        Returns:
            Execution result or stream iterator
        """
        if not self.graph_app:
            raise ValueError(f"Graph app not initialized for agent {self.agent_id}")
        
        if stream:
            return self.graph_app.astream(input_data)
        else:
            result = await self.graph_app.ainvoke(input_data)
            if hasattr(result, "model_dump"):
                return result.model_dump()
            elif hasattr(result, "dict"):
                return result.dict()
            elif isinstance(result, dict):
                return result
            else:
                return {"result": result, "state": str(result)}
    
    def serialize_state(self) -> Dict[str, Any]:
        """
        Serialize the current agent state for persistence.
        
        Returns:
            Dictionary containing serialized state information
        """
        state = {
            "agent_id": self.agent_id,
            "agent_metadata": self.agent_metadata,
            "is_listening": self.is_listening,
            "message_queue_size": self._message_queue.qsize(),
        }
        
        if self._current_state:
            state["current_state"] = self._current_state
        
        if self.state_manager:
            try:
                state["managed_state"] = self.state_manager()
            except Exception as e:
                logger.warning(f"Failed to serialize managed state: {e}")
        
        return state


__all__ = [
    "LangGraphAgentA2AWrapper",
]

