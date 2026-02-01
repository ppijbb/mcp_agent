"""
MCP/기본 Agent용 A2A Adapter

srcs/ 폴더의 MCP 기반 agent들을 위한 A2A wrapper
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
    """로그 메시지를 A2A 알림 메시지로 변환하는 핸들러"""

    def __init__(self, adapter: "A2AAdapter", correlation_id: Optional[str] = None):
        super().__init__()
        self.adapter = adapter
        self.correlation_id = correlation_id
        # 무한 루프 방지를 위해 자기 자신의 로거는 제외
        self.ignored_loggers = [__name__, "srcs.common.a2a_integration", "srcs.common.a2a_adapter"]

    def emit(self, record):
        if record.name in self.ignored_loggers:
            return

        # ContextVar에서 현재 correlation_id 확인
        ctx_id = current_correlation_id.get()

        # 만약 핸들러에 지정된 correlation_id가 있고, 현재 컨텍스트와 일치하지 않으면 무시
        # (멀티테넌시/세션 분리 지원)
        if self.correlation_id and ctx_id and self.correlation_id != ctx_id:
            return

        try:
            msg = self.format(record)
            # 비동기 루프 내에서 실행되는지 확인하고 메시지 전송
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # 비동기적으로 메시지 전송
                    asyncio.create_task(self.adapter.send_message(
                        target_agent="",  # 브로드캐스트
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
                # Event loop가 없는 경우 무시 (동기 컨텍스트에서는 전송 불가)
                pass
        except Exception:
            self.handleError(record)


class CommonAgentA2AWrapper(A2AAdapter):
    """MCP/기본 Agent용 A2A Wrapper"""

    def __init__(
        self,
        agent_id: str,
        agent_metadata: Dict[str, Any],
        agent_instance: Any = None,
        execute_function: Optional[Callable] = None
    ):
        """
        초기화

        Args:
            agent_id: Agent ID
            agent_metadata: Agent 메타데이터
            agent_instance: Agent 인스턴스 (선택)
            execute_function: 실행 함수 (선택, agent_instance가 없을 때 사용)
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
            # 기존 task가 실행 중인지 확인
            if self._message_processor_task and not self._message_processor_task.done():
                logger.debug(f"Listener task for agent {self.agent_id} is already running")
                return
            else:
                # Task가 완료되었거나 없으면 재시작
                logger.info(f"Restarting listener for agent {self.agent_id}")
                self.is_listening = False

        # Queue를 먼저 생성하여 현재 event loop에 바인딩
        self._ensure_queue()

        self.is_listening = True
        self._message_processor_task = asyncio.create_task(self._process_messages())
        logger.info(f"Message listener started for agent {self.agent_id}, task: {self._message_processor_task}")

    async def stop_listener(self) -> None:
        """메시지 리스너 중지 - 안전하게 처리"""
        if not self.is_listening:
            return

        self.is_listening = False

        # Queue에 종료 신호를 보내서 루프를 빠르게 종료
        try:
            queue = self._ensure_queue()
            # None을 보내서 루프 종료 신호
            try:
                queue.put_nowait(None)  # type: ignore
            except Exception:
                pass  # Queue가 이미 닫혀있을 수 있음
        except Exception:
            pass  # Queue 생성 실패 무시

        if self._message_processor_task:
            try:
                # Task가 완료되지 않았고 취소되지 않았을 때만 취소
                if not self._message_processor_task.done():
                    self._message_processor_task.cancel()
                    try:
                        # 타임아웃을 조금 늘려서 완료 대기
                        await asyncio.wait_for(self._message_processor_task, timeout=0.5)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        # 정상적인 취소 또는 타임아웃
                        pass
                    except RuntimeError as e:
                        # Event loop가 닫혀있는 경우 무시
                        if "Event loop is closed" not in str(e) and "cannot be called from a running event loop" not in str(e):
                            logger.debug(f"RuntimeError while stopping listener: {e}")
                    except Exception as e:
                        # 기타 예외는 로그만 남기고 무시
                        logger.debug(f"Exception while stopping listener task: {e}")
            except (RuntimeError, AttributeError) as e:
                # Event loop가 이미 닫혀있는 경우 무시
                if "Event loop is closed" not in str(e) and "cannot be called from a running event loop" not in str(e):
                    logger.debug(f"Error stopping listener task for agent {self.agent_id}: {e}")
            except Exception as e:
                # 기타 예외는 로그만 남기고 무시
                logger.debug(f"Unexpected error stopping listener task: {e}")
            finally:
                # Task 참조 제거 (메모리 누수 방지)
                try:
                    if self._message_processor_task and not self._message_processor_task.done():
                        # Task가 아직 실행 중이면 취소만 하고 완료 대기는 하지 않음
                        self._message_processor_task.cancel()
                except Exception:
                    pass
                self._message_processor_task = None

        logger.debug(f"Message listener stopped for agent {self.agent_id}")

    async def _process_messages(self) -> None:
        """메시지 처리 루프 - event loop가 닫혀있을 때 안전하게 처리"""
        try:
            while self.is_listening:
                try:
                    # Event loop가 닫혀있는지 확인
                    try:
                        loop = asyncio.get_running_loop()
                        if loop.is_closed():
                            logger.debug(f"Event loop is closed, stopping listener for {self.agent_id}")
                            break
                    except RuntimeError:
                        # Event loop가 없는 경우
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
                        # 타임아웃은 정상 (메시지가 없을 때)
                        continue
                    except Exception as e:
                        logger.error(f"Error getting message from queue for agent {self.agent_id}: {e}", exc_info=True)
                        await asyncio.sleep(0.1)  # 에러 발생 시 잠시 대기
                        continue

                    # None은 종료 신호
                    if message is None:
                        logger.debug(f"Received stop signal for agent {self.agent_id}")
                        break

                    logger.info(f"✅ Agent {self.agent_id} received message: {message.message_type} (id: {message.message_id})")
                    try:
                        result = await self.handle_message(message)
                        logger.debug(f"Agent {self.agent_id} handled message {message.message_id}, result: {result is not None}")
                    except Exception as e:
                        logger.error(f"Error handling message {message.message_id} in agent {self.agent_id}: {e}", exc_info=True)
                        # 에러가 발생해도 계속 진행
                        continue
                except asyncio.TimeoutError:
                    continue
                except (RuntimeError, AttributeError) as e:
                    # Event loop가 닫혀있거나 queue가 다른 loop에 바인딩된 경우
                    if "Event loop is closed" in str(e) or "bound to a different event loop" in str(e):
                        logger.debug(f"Event loop issue detected, stopping listener for {self.agent_id}: {e}")
                        break
                    else:
                        logger.error(f"Error processing message in agent {self.agent_id}: {e}", exc_info=True)
                except asyncio.CancelledError:
                    # Task가 취소된 경우 정상 종료
                    logger.debug(f"Message processor cancelled for {self.agent_id}")
                    break
                except Exception as e:
                    logger.error(f"Error processing message in agent {self.agent_id}: {e}", exc_info=True)
        except asyncio.CancelledError:
            # 최상위 레벨에서도 CancelledError 처리
            logger.debug(f"Message processor task cancelled for {self.agent_id}")
        except Exception as e:
            logger.error(f"Fatal error in message processor for {self.agent_id}: {e}", exc_info=True)
        finally:
            # 정리 작업
            self.is_listening = False
            logger.debug(f"Message processor stopped for {self.agent_id}")

    async def register_capabilities(self, capabilities: List[str]) -> None:
        """Agent 능력 등록"""
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
        A2A 지원 실행

        Args:
            input_data: 입력 데이터
            request_help: 다른 agent의 도움이 필요한지 여부

        Returns:
            실행 결과
        """
        # 도움이 필요한 경우 다른 agent에게 요청
        if request_help:
            help_message = {
                "task": input_data.get("task", ""),
                "context": input_data.get("context", {}),
            }
            await self.send_message(
                target_agent="",  # 브로드캐스트
                message_type="help_request",
                payload=help_message,
                priority=MessagePriority.HIGH.value,
            )

        # Agent 실행
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
        """상태 직렬화"""
        return {
            "agent_id": self.agent_id,
            "agent_metadata": self.agent_metadata,
            "is_listening": self.is_listening,
            "message_queue_size": self._message_queue.qsize(),
        }
