"""
MCP/기본 Agent용 A2A Adapter

srcs/ 폴더의 MCP 기반 agent들을 위한 A2A wrapper
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from srcs.common.a2a_integration import (
    A2AAdapter,
    A2AMessage,
    MessagePriority,
    get_global_broker,
    get_global_registry,
)
from srcs.common.agent_interface import AgentMetadata, AgentType

logger = logging.getLogger(__name__)


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
            return
        
        self.is_listening = True
        self._message_processor_task = asyncio.create_task(self._process_messages())
        logger.info(f"Message listener started for agent {self.agent_id}")
    
    async def stop_listener(self) -> None:
        """메시지 리스너 중지 - 안전하게 처리"""
        if not self.is_listening:
            return
        
        self.is_listening = False
        if self._message_processor_task:
            try:
                # Task가 완료되지 않았고 취소되지 않았을 때만 취소
                if not self._message_processor_task.done():
                    self._message_processor_task.cancel()
                    try:
                        # 짧은 타임아웃으로 완료 대기 (event loop가 닫혀있을 수 있음)
                        await asyncio.wait_for(self._message_processor_task, timeout=0.5)
                    except (asyncio.CancelledError, asyncio.TimeoutError, RuntimeError):
                        # Event loop가 닫혀있거나 타임아웃된 경우 무시
                        pass
            except (RuntimeError, AttributeError) as e:
                # Event loop가 이미 닫혀있는 경우 무시
                logger.debug(f"Listener stop warning (expected on page switch): {e}")
            except Exception as e:
                logger.warning(f"Unexpected error stopping listener: {e}")
        
        logger.info(f"Message listener stopped for agent {self.agent_id}")
    
    async def _process_messages(self) -> None:
        """메시지 처리 루프 - event loop가 닫혀있을 때 안전하게 처리"""
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
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
                logger.info(f"Agent {self.agent_id} received message: {message.message_type} (id: {message.message_id})")
                result = await self.handle_message(message)
                logger.debug(f"Agent {self.agent_id} handled message {message.message_id}, result: {result is not None}")
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

