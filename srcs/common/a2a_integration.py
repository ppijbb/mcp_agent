"""
A2A (Agent-to-Agent) Integration System.

Common interface and message broker for all agent-to-agent communication.
Provides message routing, agent registration, and capability management.

Classes:
    MessagePriority: Enum for message priority levels
    A2AMessage: Dataclass representing an A2A protocol message
    A2AAdapter: Abstract base class for A2A adapters
    AgentRegistry: Central agent registration system
    A2AMessageBroker: Message routing and distribution broker

Functions:
    get_global_registry: Get the global agent registry instance
    get_global_broker: Get the global message broker instance

Example:
    registry = get_global_registry()
    broker = get_global_broker()
    await registry.register_agent(agent_id="agent1", agent_type="mcp", metadata={})
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
import uuid
import asyncio
import logging

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """
    Message priority levels for A2A communication.
    
    Attributes:
        HIGH: High priority messages (value: 1)
        MEDIUM: Medium priority messages (value: 2)
        LOW: Low priority messages (value: 3)
    """
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class A2AMessage:
    """
    A2A protocol message.
    
    Represents a message in the agent-to-agent communication protocol,
    including routing information, payload, and metadata.
    
    Attributes:
        message_id: Unique identifier for the message
        timestamp: ISO format timestamp when message was created
        source_agent: ID of the agent that sent the message
        target_agent: ID of the target agent (empty for broadcast)
        message_type: Type of message (e.g., "task_request", "help_request")
        payload: Message payload data dictionary
        correlation_id: Optional ID for correlating related messages
        priority: Message priority (default: MEDIUM)
        ttl: Time to live in seconds (default: 300)
    """
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    source_agent: str = ""
    target_agent: str = ""
    message_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    priority: int = MessagePriority.MEDIUM.value
    ttl: int = 300

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert message to dictionary.
        
        Returns:
            Dictionary representation of the message
        """
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "message_type": self.message_type,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        """
        Create message from dictionary.
        
        Args:
            data: Dictionary containing message data
            
        Returns:
            A2AMessage instance created from the dictionary
        """
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat() + 'Z'),
            source_agent=data.get("source_agent", ""),
            target_agent=data.get("target_agent", ""),
            message_type=data.get("message_type", ""),
            payload=data.get("payload", {}),
            correlation_id=data.get("correlation_id"),
            priority=data.get("priority", MessagePriority.MEDIUM.value),
            ttl=data.get("ttl", 300),
        )


class A2AAdapter(ABC):
    """
    Abstract base class for A2A protocol adapters.
    
    Defines the interface that all agent adapters must implement to
    participate in agent-to-agent communication.
    
    Attributes:
        agent_id: Unique identifier for the agent
        agent_metadata: Metadata dictionary containing agent information
        message_handlers: Dictionary of message type to handler mappings
        is_listening: Whether the message listener is active
        _message_queue: Asyncio queue for incoming messages
    """

    def __init__(self, agent_id: str, agent_metadata: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_metadata = agent_metadata
        self.message_handlers: Dict[str, Callable] = {}
        self.is_listening = False
        self._message_queue: Optional[asyncio.Queue] = None

    def _ensure_queue(self) -> asyncio.Queue:
        """
        Get or create the message queue for this adapter.
        
        Creates a new queue if needed, handling event loop binding
        and recreation when necessary (e.g., page transitions).
        
        Returns:
            asyncio.Queue instance for this adapter
        """
        try:
            current_loop = asyncio.get_running_loop()
            if self._message_queue is None:
                logger.debug(f"Creating new queue for agent {self.agent_id} in loop {id(current_loop)}")
                self._message_queue = asyncio.Queue()
            else:
                try:
                    if hasattr(self._message_queue, '_loop'):
                        queue_loop = self._message_queue._loop
                        if queue_loop is not None:
                            if queue_loop != current_loop:
                                logger.warning(f"Queue for agent {self.agent_id} is bound to different loop, recreating")
                                self._message_queue = asyncio.Queue()
                            elif queue_loop.is_closed():
                                logger.warning(f"Queue for agent {self.agent_id} is bound to closed loop, recreating")
                                self._message_queue = asyncio.Queue()
                except (AttributeError, RuntimeError) as e:
                    logger.warning(f"Error checking queue loop for agent {self.agent_id}: {e}, recreating")
                    self._message_queue = asyncio.Queue()
        except RuntimeError:
            logger.debug(f"No event loop for agent {self.agent_id}, creating queue without loop")
            self._message_queue = asyncio.Queue()

        return self._message_queue

    @abstractmethod
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

    @abstractmethod
    async def start_listener(self) -> None:
        """Start the message listener for this agent."""

    @abstractmethod
    async def stop_listener(self) -> None:
        """Stop the message listener for this agent."""

    @abstractmethod
    async def register_capabilities(self, capabilities: List[str]) -> None:
        """Register agent capabilities with the global registry."""

    def register_handler(self, message_type: str, handler: Callable) -> None:
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Callable to handle the message (sync or async)
        """
        self.message_handlers[message_type] = handler

    async def handle_message(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """
        Handle an incoming message.
        
        Args:
            message: The A2AMessage to handle
            
        Returns:
            Result from the handler if one exists, None otherwise
        """
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                logger.info(f"Calling handler for message type {message.message_type} in agent {self.agent_id}")
                result = await handler(message) if asyncio.iscoroutinefunction(handler) else handler(message)
                logger.debug(f"Handler for {message.message_type} returned result: {result is not None}")
                return result
            except Exception as e:
                logger.error(f"Error handling message {message.message_id} in agent {self.agent_id}: {e}", exc_info=True)
                return None
        else:
            logger.warning(f"No handler for message type: {message.message_type} in agent {self.agent_id}")
            return None


class AgentRegistry:
    """
    Central agent registration system.
    
    Maintains a registry of all active agents and their metadata,
    enabling discovery and routing of messages between agents.
    
    Attributes:
        agents: Dictionary of agent_id to agent info mappings
        _lock: Async lock for thread-safe operations
    """

    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        metadata: Dict[str, Any],
        a2a_adapter: Optional[A2AAdapter] = None
    ) -> None:
        """
        Register an agent with the registry.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (e.g., "mcp", "langgraph")
            metadata: Agent metadata dictionary
            a2a_adapter: Optional A2A adapter instance
        """
        async with self._lock:
            self.agents[agent_id] = {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "metadata": metadata,
                "a2a_adapter": a2a_adapter,
                "registered_at": datetime.utcnow().isoformat(),
                "status": "active",
            }
            logger.info(f"Agent registered: {agent_id}")

    async def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the registry.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        async with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                logger.info(f"Agent unregistered: {agent_id}")

    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get agent information by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            Agent info dictionary or None if not found
        """
        async with self._lock:
            return self.agents.get(agent_id)

    async def list_agents(self, agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered agents.
        
        Args:
            agent_type: Optional filter for agent type
            
        Returns:
            List of agent info dictionaries
        """
        async with self._lock:
            agents = list(self.agents.values())
            if agent_type:
                agents = [a for a in agents if a.get("agent_type") == agent_type]
            return agents

    async def find_agents_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """
        Find agents that have a specific capability.
        
        Args:
            capability: Capability string to search for
            
        Returns:
            List of agent info dictionaries with matching capability
        """
        async with self._lock:
            matching_agents = []
            for agent in self.agents.values():
                capabilities = agent.get("metadata", {}).get("capabilities", [])
                if capability in capabilities:
                    matching_agents.append(agent)
            return matching_agents


class A2AMessageBroker:
    """
    Message routing and distribution broker.
    
    Handles routing of A2A messages between agents, including
    unicast routing to specific agents and broadcast to all agents.
    
    Attributes:
        registry: AgentRegistry instance for agent lookup
        _message_history: List of recent messages
        _max_history: Maximum number of messages to retain
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._message_history: deque[A2AMessage] = deque(maxlen=1000)
        self._max_history = 1000

    async def route_message(self, message: A2AMessage) -> bool:
        """
        Route a message to its destination.
        
        Args:
            message: A2AMessage to route
            
        Returns:
            True if message was routed successfully
        """
        self._message_history.append(message)

        if message.ttl > 0:
            message_time = datetime.fromisoformat(message.timestamp.replace('Z', '+00:00'))
            age = (datetime.utcnow() - message_time.replace(tzinfo=None)).total_seconds()
            if age > message.ttl:
                logger.warning(f"Message {message.message_id} expired (age: {age}s, ttl: {message.ttl}s)")
                return False

        if not message.target_agent:
            return await self._broadcast_message(message)

        target_agent_info = await self.registry.get_agent(message.target_agent)
        if not target_agent_info:
            logger.warning(f"Target agent not found: {message.target_agent}")
            return False

        a2a_adapter = target_agent_info.get("a2a_adapter")
        if not a2a_adapter:
            logger.warning(f"Target agent has no A2A adapter: {message.target_agent}")
            return False

        logger.info(f"Routing message {message.message_id} ({message.message_type}) from {message.source_agent} to {message.target_agent}")
        try:
            if not a2a_adapter.is_listening:
                logger.warning(f"Agent {message.target_agent} listener is not running, starting it...")
                await a2a_adapter.start_listener()

            queue = a2a_adapter._ensure_queue()
            queue_size_before = queue.qsize()
            logger.info(f"Queue for agent {message.target_agent}: {queue}, size before: {queue_size_before}, listener running: {a2a_adapter.is_listening}")

            await queue.put(message)
            queue_size_after = queue.qsize()
            logger.info(f"Message {message.message_id} added to queue for agent {message.target_agent}, size after: {queue_size_after}")

            if hasattr(a2a_adapter, "_message_processor_task") and a2a_adapter._message_processor_task:
                if a2a_adapter._message_processor_task.done():
                    logger.warning(f"Listener task for agent {message.target_agent} is done, restarting...")
                    await a2a_adapter.start_listener()
                else:
                    logger.debug(f"Listener task for agent {message.target_agent} is running")

            return True
        except (RuntimeError, AttributeError) as e:
            if "Event loop is closed" in str(e) or "bound to a different event loop" in str(e):
                logger.warning(f"Cannot route message to {message.target_agent}: event loop issue")
                return False
            raise

    async def _broadcast_message(self, message: A2AMessage) -> bool:
        """
        Broadcast a message to all registered agents.
        
        Args:
            message: A2AMessage to broadcast
            
        Returns:
            True if message was sent to at least one agent
        """
        agents = await self.registry.list_agents()
        success_count = 0

        for agent_info in agents:
            agent_id = agent_info.get("agent_id")
            if agent_id == message.source_agent:
                continue

            a2a_adapter = agent_info.get("a2a_adapter")
            if a2a_adapter:
                try:
                    queue = a2a_adapter._ensure_queue()
                    await queue.put(message)
                    success_count += 1
                except (RuntimeError, AttributeError) as e:
                    if "Event loop is closed" in str(e) or "bound to a different event loop" in str(e):
                        logger.debug(f"Cannot send message to {agent_id}: event loop issue")
                    else:
                        logger.error(f"Failed to send message to {agent_id}: {e}")
                except Exception as e:
                    logger.error(f"Failed to send message to {agent_id}: {e}")

        logger.info(f"Broadcast message {message.message_id} sent to {success_count} agents")
        return success_count > 0

    def get_message_history(self, limit: int = 100) -> List[A2AMessage]:
        """
        Get recent message history.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of recent A2AMessage objects
        """
        return self._message_history[-limit:]


_global_registry: Optional[AgentRegistry] = None
_global_broker: Optional[A2AMessageBroker] = None


def get_global_registry() -> AgentRegistry:
    """
    Get the global AgentRegistry instance.
    
    Returns:
        Singleton AgentRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def get_global_broker() -> A2AMessageBroker:
    """
    Get the global A2AMessageBroker instance.
    
    Returns:
        Singleton A2AMessageBroker instance
    """
    global _global_broker
    if _global_broker is None:
        _global_broker = A2AMessageBroker(get_global_registry())
    return _global_broker
