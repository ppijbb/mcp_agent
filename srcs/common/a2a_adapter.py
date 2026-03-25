"""
A2A Adapter for MCP and Base Agents.

A2A (Agent-to-Agent) wrapper for MCP-based agents in the srcs/ directory.
Provides message passing, logging integration, and state serialization for
MCP and standard Python agents.

Classes:
    A2ALogHandler: Logging handler that converts log messages to A2A notifications
    CommonAgentA2AWrapper: A2A adapter for MCP/base agents

Example:
    wrapper = CommonAgentA2AWrapper(
        agent_id="analysis_agent",
        agent_metadata={"name": "Analysis Agent"},
        agent_instance=my_agent
    )
    await wrapper.start_listener()
"""

import asyncio
import logging
import contextvars
from typing import Dict, Any, List, Optional, Callable

from srcs.common.a2a_integration import (
    A2AAdapter,
    A2AMessage,
    MessagePriority,
    get_global_broker,
    get_global_registry,
)
from srcs.common.agent_interface import AgentType

logger = logging.getLogger(__name__)

# 현재 실행 중인 작업의 correlation_id를 추적하기 위한 ContextVar
current_correlation_id = contextvars.ContextVar("current_correlation_id", default=None)


class A2ALogHandler(logging.Handler):
    """
    Logging handler that converts log messages to A2A notification messages.
    
    Intercepts logging events and broadcasts them as A2A notifications,
    enabling centralized logging across distributed agents.
    
    Attributes:
        adapter: A2AAdapter instance for sending messages
        correlation_id: Optional correlation ID for multi-tenancy support
        ignored_loggers: List of logger names to exclude from forwarding
    """

    def __init__(self, adapter: "A2AAdapter", correlation_id: Optional[str] = None):
        """
        Initialize the A2A log handler.
        
        Args:
            adapter: A2AAdapter instance for sending log notifications
            correlation_id: Optional correlation ID for multi-tenancy support
        """
        super().__init__()
        self.adapter = adapter
        self.correlation_id = correlation_id
        self.ignored_loggers = [__name__, "srcs.common.a2a_integration", "srcs.common.a2a_adapter"]

    def emit(self, record):
        """
        Emit a log record as an A2A notification.
        
        Converts the log record to an A2A notification message and sends
        it asynchronously if running within an event loop.
        
        Args:
            record: Log record to emit
        """
        if record.name in self.ignored_loggers:
            return

        ctx_id = current_correlation_id.get()

        if self.correlation_id and ctx_id and self.correlation_id != ctx_id:
            return

        try:
            msg = self.format(record)
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    asyncio.create_task(self.adapter.send_message(
                        target_agent="",
                        message_type="notification",
                        payload={
                            "type": "log",
                            "level": record.levelname,
                            "logger": record.name,
                            "message": msg
                        },
                        correlation_id=self.correlation_id
                    ))
            except RuntimeError:
                pass
        except Exception:
            self.handleError(record)


class CommonAgentA2AWrapper(A2AAdapter):
    """
    A2A adapter for MCP and base Python agents.
    
    Extends A2AAdapter to provide agent-to-agent messaging capabilities
    for MCP-based agents and standard Python agents with execute/run methods.
    
    Attributes:
        agent_id: Unique identifier for the agent
        agent_metadata: Metadata dictionary containing agent information
        agent_instance: Optional agent instance with execute or run method
        execute_function: Optional standalone function for agent execution
        is_listening: Whether the message listener is active
        _message_processor_task: Background task for processing messages
    """

    def __init__(
        self,
        agent_id: str,
        agent_metadata: Dict[str, Any],
        agent_instance: Any = None,
        execute_function: Optional[Callable] = None
    ):
        """
        Initialize the Common Agent A2A wrapper.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_metadata: Dictionary containing agent metadata
            agent_instance: Optional agent instance with execute or run method
            execute_function: Optional standalone function (used if no agent_instance)
        """
        super().__init__(agent_id, agent_metadata)
        self.agent_instance = agent_instance
        self.execute_function = execute_function
        self._message_processor_task: Optional[asyncio.Task] = None

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
        from the message queue. Safely handles restart if already running.
        """
        if self.is_listening:
            logger.warning(f"Listener already started for agent {self.agent_id}")
            if self._message_processor_task and not self._message_processor_task.done():
                logger.debug(f"Listener task for agent {self.agent_id} is already running")
                return
            else:
                logger.info(f"Restarting listener for agent {self.agent_id}")
                self.is_listening = False

        self._ensure_queue()

        self.is_listening = True
        self._message_processor_task = asyncio.create_task(self._process_messages())
        logger.info(f"Message listener started for agent {self.agent_id}, task: {self._message_processor_task}")

    async def stop_listener(self) -> None:
        """
        Stop the message listener for this agent.
        
        Cancels the background message processing task and marks the
        listener as inactive. Safely handles event loop edge cases.
        """
        if not self.is_listening:
            return

        self.is_listening = False

        try:
            queue = self._ensure_queue()
            try:
                queue.put_nowait(None)
            except Exception:
                pass
        except Exception:
            pass

        if self._message_processor_task:
            try:
                if not self._message_processor_task.done():
                    self._message_processor_task.cancel()
                    try:
                        await asyncio.wait_for(self._message_processor_task, timeout=0.5)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    except RuntimeError as e:
                        if "Event loop is closed" not in str(e) and "cannot be called from a running event loop" not in str(e):
                            logger.debug(f"RuntimeError while stopping listener: {e}")
                    except Exception as e:
                        logger.debug(f"Exception while stopping listener task: {e}")
            except (RuntimeError, AttributeError) as e:
                if "Event loop is closed" not in str(e) and "cannot be called from a running event loop" not in str(e):
                    logger.debug(f"Error stopping listener task for agent {self.agent_id}: {e}")
            except Exception as e:
                logger.debug(f"Unexpected error stopping listener task: {e}")
            finally:
                try:
                    if self._message_processor_task and not self._message_processor_task.done():
                        self._message_processor_task.cancel()
                except Exception:
                    pass
                self._message_processor_task = None

        logger.debug(f"Message listener stopped for agent {self.agent_id}")

    async def _process_messages(self) -> None:
        """
        Message processing loop.
        
        Continuously listens for messages from the queue and processes them
        using registered message handlers. Runs until is_listening is False.
        Safely handles event loop closure and other edge cases.
        """
        try:
            while self.is_listening:
                try:
                    try:
                        loop = asyncio.get_running_loop()
                        if loop.is_closed():
                            logger.debug(f"Event loop is closed, stopping listener for {self.agent_id}")
                            break
                    except RuntimeError:
                        logger.debug(f"No event loop, stopping listener for {self.agent_id}")
                        break

                    queue = self._ensure_queue()
                    queue_size = queue.qsize()
                    if queue_size > 0:
                        logger.info(f"Agent {self.agent_id} has {queue_size} messages in queue")
                    logger.debug(f"Agent {self.agent_id} waiting for message, queue size: {queue_size}")
                    try:
                        message = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Error getting message from queue for agent {self.agent_id}: {e}", exc_info=True)
                        await asyncio.sleep(0.1)
                        continue

                    if message is None:
                        logger.debug(f"Received stop signal for agent {self.agent_id}")
                        break

                    logger.info(f"Agent {self.agent_id} received message: {message.message_type} (id: {message.message_id})")
                    try:
                        result = await self.handle_message(message)
                        logger.debug(f"Agent {self.agent_id} handled message {message.message_id}, result: {result is not None}")
                    except Exception as e:
                        logger.error(f"Error handling message {message.message_id} in agent {self.agent_id}: {e}", exc_info=True)
                        continue
                except asyncio.TimeoutError:
                    continue
                except (RuntimeError, AttributeError) as e:
                    if "Event loop is closed" in str(e) or "bound to a different event loop" in str(e):
                        logger.debug(f"Event loop issue detected, stopping listener for {self.agent_id}: {e}")
                        break
                    else:
                        logger.error(f"Error processing message in agent {self.agent_id}: {e}", exc_info=True)
                except asyncio.CancelledError:
                    logger.debug(f"Message processor cancelled for {self.agent_id}")
                    break
                except Exception as e:
                    logger.error(f"Error processing message in agent {self.agent_id}: {e}", exc_info=True)
        except asyncio.CancelledError:
            logger.debug(f"Message processor task cancelled for {self.agent_id}")
        except Exception as e:
            logger.error(f"Fatal error in message processor for {self.agent_id}: {e}", exc_info=True)
        finally:
            self.is_listening = False
            logger.debug(f"Message processor stopped for {self.agent_id}")

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
            agent_type=AgentType.MCP_AGENT.value,
            metadata=self.agent_metadata,
            a2a_adapter=self,
        )
        logger.info(f"Capabilities registered for agent {self.agent_id}: {capabilities}")

    async def execute_with_a2a(
        self,
        input_data: Dict[str, Any],
        request_help: bool = False
    ) -> Dict[str, Any]:
        """
        Execute agent task with optional A2A help requests.
        
        Args:
            input_data: Input data for the agent
            request_help: Whether to request help from other agents
            
        Returns:
            Execution result dictionary
        """
        if request_help:
            help_message = {
                "task": input_data.get("task", ""),
                "context": input_data.get("context", {}),
            }
            await self.send_message(
                target_agent="",
                message_type="help_request",
                payload=help_message,
                priority=MessagePriority.HIGH.value,
            )

        if self.agent_instance:
            if hasattr(self.agent_instance, "execute"):
                if asyncio.iscoroutinefunction(self.agent_instance.execute):
                    result = await self.agent_instance.execute(input_data)
                else:
                    result = self.agent_instance.execute(input_data)
            elif hasattr(self.agent_instance, "run"):
                if asyncio.iscoroutinefunction(self.agent_instance.run):
                    result = await self.agent_instance.run(input_data)
                else:
                    result = self.agent_instance.run(input_data)
            else:
                raise ValueError(f"Agent instance {self.agent_id} has no execute or run method")
        elif self.execute_function:
            if asyncio.iscoroutinefunction(self.execute_function):
                result = await self.execute_function(input_data)
            else:
                result = self.execute_function(input_data)
        else:
            raise ValueError(f"No execution method available for agent {self.agent_id}")

        return result

    def serialize_state(self) -> Dict[str, Any]:
        """
        Serialize the current agent state for persistence.
        
        Returns:
            Dictionary containing serialized state information
        """
        return {
            "agent_id": self.agent_id,
            "agent_metadata": self.agent_metadata,
            "is_listening": self.is_listening,
            "message_queue_size": self._message_queue.qsize(),
        }


__all__ = [
    "A2ALogHandler",
    "CommonAgentA2AWrapper",
]
