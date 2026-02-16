"""
Cron Agent용 A2A Adapter

cron_agents/ 폴더의 Cron 기반 agent들을 위한 A2A wrapper
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
    """Cron Agent용 A2A Wrapper"""

    def __init__(
        self,
        agent_id: str,
        agent_metadata: Dict[str, Any],
        cron_schedule: str = "",
        execute_function: Optional[Callable] = None,
        cron_job: Optional[Any] = None
    ):
        """
        초기화

        Args:
            agent_id: Agent ID
            agent_metadata: Agent 메타데이터
            cron_schedule: Cron 스케줄 (예: "*/5 * * * *")
            execute_function: 실행 함수
            cron_job: 기존 cron job 인스턴스 (선택)
        """
        super().__init__(agent_id, agent_metadata)
        self.cron_schedule = cron_schedule
        self.execute_function = execute_function
        self.cron_job = cron_job
        self._message_processor_task: Optional[asyncio.Task] = None
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_running = False

    async def send_message(
        self,
        target_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: int = MessagePriority.MEDIUM.value,
        correlation_id: Optional[str] = None
    ) -> bool:
        """메시지 전송"""
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
        """메시지 리스너 시작"""
        if self.is_listening:
            logger.warning(f"Listener already started for agent {self.agent_id}")
            return

        self.is_listening = True
        self._message_processor_task = asyncio.create_task(self._process_messages())

        # Cron 스케줄러 시작
        if self.cron_schedule:
            self._start_scheduler()

        logger.info(f"Message listener started for Cron agent {self.agent_id}")

    async def stop_listener(self) -> None:
        """메시지 리스너 중지"""
        if not self.is_listening:
            return

        self.is_listening = False

        # Cron 스케줄러 중지
        self._stop_scheduler()

        if self._message_processor_task:
            self._message_processor_task.cancel()
            try:
                await self._message_processor_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Message listener stopped for Cron agent {self.agent_id}")

    def _start_scheduler(self) -> None:
        """Cron 스케줄러 시작"""
        if self._scheduler_running:
            return

        self._scheduler_running = True

        def run_scheduler():
            while self._scheduler_running:
                schedule.run_pending()
                time.sleep(1)

        self._scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self._scheduler_thread.start()

        # 스케줄 등록
        if self.cron_schedule and self.execute_function:
            # 간단한 cron 파싱 (예: "*/5 * * * *" -> 5분마다)
            # 실제로는 croniter 라이브러리 사용 권장
            schedule.every(5).minutes.do(self._run_scheduled_task)
            logger.info(f"Scheduled task registered for agent {self.agent_id}: {self.cron_schedule}")

    def _stop_scheduler(self) -> None:
        """Cron 스케줄러 중지"""
        self._scheduler_running = False
        schedule.clear()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)

    def _run_scheduled_task(self) -> None:
        """스케줄된 작업 실행"""
        if self.execute_function:
            try:
                if asyncio.iscoroutinefunction(self.execute_function):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.execute_function({}))
                    loop.close()
                else:
                    self.execute_function({})
                logger.info(f"Scheduled task executed for agent {self.agent_id}")
            except Exception as e:
                logger.error(f"Error executing scheduled task for agent {self.agent_id}: {e}")

    async def _process_messages(self) -> None:
        """메시지 처리 루프"""
        while self.is_listening:
            try:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self.handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message in Cron agent: {e}")

    async def register_capabilities(self, capabilities: List[str]) -> None:
        """Agent 능력 등록"""
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
        """상태 직렬화"""
        return {
            "agent_id": self.agent_id,
            "agent_metadata": self.agent_metadata,
            "is_listening": self.is_listening,
            "message_queue_size": self._message_queue.qsize(),
            "cron_schedule": self.cron_schedule,
            "scheduler_running": self._scheduler_running,
        }
