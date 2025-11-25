"""
LangGraph Agent용 A2A Adapter

lang_graph/ 폴더의 LangGraph 기반 agent들을 위한 A2A wrapper
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
    """LangGraph Agent용 A2A Wrapper"""
    
    def __init__(
        self,
        agent_id: str,
        agent_metadata: Dict[str, Any],
        graph_app: Any = None,
        state_manager: Optional[Callable] = None
    ):
        """
        초기화
        
        Args:
            agent_id: Agent ID
            agent_metadata: Agent 메타데이터
            graph_app: LangGraph app 인스턴스
            state_manager: 상태 관리 함수 (선택)
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
        logger.info(f"Message listener started for LangGraph agent {self.agent_id}")
    
    async def stop_listener(self) -> None:
        """메시지 리스너 중지"""
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
        """메시지 처리 루프"""
        while self.is_listening:
            try:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self.handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message in LangGraph agent: {e}")
    
    async def register_capabilities(self, capabilities: List[str]) -> None:
        """Agent 능력 등록"""
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
        LangGraph 실행
        
        Args:
            input_data: 입력 데이터 (LangGraph state 형식)
            stream: 스트리밍 모드 여부
            
        Returns:
            실행 결과 또는 스트림
        """
        if not self.graph_app:
            raise ValueError(f"Graph app not initialized for agent {self.agent_id}")
        
        if stream:
            return self.graph_app.astream(input_data)
        else:
            return await self.graph_app.ainvoke(input_data)
    
    def serialize_state(self) -> Dict[str, Any]:
        """상태 직렬화"""
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

