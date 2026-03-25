"""
Cron Agent A2A Adapter.

A2A (Agent-to-Agent) wrapper for cron-based agents in the cron_agents/
directory. Provides message passing, scheduled task execution, and capability
registration for cron jobs.

Classes:
    CronAgentA2AWrapper: A2A adapter for cron-based agents

Example:
    wrapper = CronAgentA2AWrapper(
        agent_id="scheduled_task_agent",
        agent_metadata={"name": "Scheduled Task Agent"},
        cron_schedule="*/5 * * * *",
        execute_function=my_task
    )
    await wrapper.start_listener()
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
import schedule
import threading

# 상위 디렉토리의 공통 모듈 import
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.a2a_integration import (  # noqa: E402
    A2AAdapter,
    A2AMessage,
    MessagePriority,
    get_global_broker,
    get_global_registry,
)
from srcs.common.agent_interface import AgentType  # noqa: E402

logger = logging.getLogger(__name__)


class CronAgentA2AWrapper(A2AAdapter):
    """
    A2A adapter for cron-based agents.
    
    Extends A2AAdapter to provide cron scheduling functionality alongside
    agent-to-agent messaging. Supports both async and sync execution functions.
    
    Attributes:
        agent_id: Unique identifier for the agent
        agent_metadata: Metadata dictionary containing agent information
        cron_schedule: Cron schedule string (e.g., "*/5 * * * *")
        execute_function: Function to execute on schedule
        cron_job: Optional existing cron job instance
        is_listening: Whether the message listener is active
        _scheduler_thread: Background thread running the scheduler
        _scheduler_running: Whether the scheduler thread is active
    """

    def __init__(
        self,
        agent_id: str,
        agent_metadata: Dict[str, Any],
        cron_schedule: str = "",
        execute_function: Optional[Callable] = None,
        cron_job: Optional[Any] = None
    ):
        """
        Initialize the Cron A2A wrapper.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_metadata: Dictionary containing agent metadata
            cron_schedule: Cron schedule string (e.g., "*/5 * * * *")
            execute_function: Function to execute on schedule
            cron_job: Optional existing cron job instance
        """
        super().__init__(agent_id, agent_metadata)
        self.cron_schedule = cron_schedule
        self.execute_function = execute_function
        self.cron_job = cron_job
        self._message_processor_task: Optional[asyncio.Task] = None
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_running = False

    async def register_capabilities(self, capabilities: List[str]) -> None:
        """
        Register agent capabilities with the global registry.
        
        Args:
            capabilities: List of capability strings this agent supports
        """
        self.agent_metadata["capabilities"] = capabilities
        self.agent_metadata["cron_schedule"] = self.cron_schedule
        registry = get_global_registry()
        await registry.register_agent(
            agent_id=self.agent_id,
            agent_type=AgentType.CRON_AGENT.value,
            metadata=self.agent_metadata,
            a2a_adapter=self,
        )
        logger.info(f"Capabilities registered for Cron agent {self.agent_id}: {capabilities}")

    def serialize_state(self) -> Dict[str, Any]:
        """
        Serialize the current agent state for persistence.
        
        Returns:
            Dictionary containing serialized state information including
            agent metadata, listening status, message queue size, cron
            schedule, and scheduler status.
        """
        return {
            "agent_id": self.agent_id,
            "agent_metadata": self.agent_metadata,
            "is_listening": self.is_listening,
            "message_queue_size": self._message_queue.qsize(),
            "cron_schedule": self.cron_schedule,
            "scheduler_running": self._scheduler_running,
        }
