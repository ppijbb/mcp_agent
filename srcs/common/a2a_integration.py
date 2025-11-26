"""
A2A (Agent-to-Agent) 통합 시스템

모든 agent 간 통신을 위한 공통 인터페이스 및 메시지 브로커
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
import uuid
import asyncio
import logging

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """메시지 우선순위"""
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class A2AMessage:
    """A2A 프로토콜 메시지"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    source_agent: str = ""
    target_agent: str = ""  # 빈 문자열이면 브로드캐스트
    message_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    priority: int = MessagePriority.MEDIUM.value
    ttl: int = 300  # Time to live in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
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
        """딕셔너리에서 생성"""
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
    """A2A 프로토콜 어댑터 추상 클래스"""
    
    def __init__(self, agent_id: str, agent_metadata: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_metadata = agent_metadata
        self.message_handlers: Dict[str, Callable] = {}
        self.is_listening = False
        self._message_queue: asyncio.Queue = asyncio.Queue()
    
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
        메시지 전송
        
        Args:
            target_agent: 대상 agent ID (빈 문자열이면 브로드캐스트)
            message_type: 메시지 타입
            payload: 메시지 페이로드
            priority: 우선순위
            correlation_id: 상관관계 ID
            
        Returns:
            bool: 전송 성공 여부
        """
        pass
    
    @abstractmethod
    async def start_listener(self) -> None:
        """메시지 리스너 시작"""
        pass
    
    @abstractmethod
    async def stop_listener(self) -> None:
        """메시지 리스너 중지"""
        pass
    
    @abstractmethod
    async def register_capabilities(self, capabilities: List[str]) -> None:
        """Agent 능력 등록"""
        pass
    
    def register_handler(self, message_type: str, handler: Callable) -> None:
        """메시지 핸들러 등록"""
        self.message_handlers[message_type] = handler
    
    async def handle_message(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """메시지 처리"""
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
    """Agent 레지스트리 - 중앙 등록 시스템"""
    
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
        """Agent 등록"""
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
        """Agent 등록 해제"""
        async with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                logger.info(f"Agent unregistered: {agent_id}")
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Agent 정보 조회"""
        async with self._lock:
            return self.agents.get(agent_id)
    
    async def list_agents(self, agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Agent 목록 조회"""
        async with self._lock:
            agents = list(self.agents.values())
            if agent_type:
                agents = [a for a in agents if a.get("agent_type") == agent_type]
            return agents
    
    async def find_agents_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """능력으로 Agent 검색"""
        async with self._lock:
            matching_agents = []
            for agent in self.agents.values():
                capabilities = agent.get("metadata", {}).get("capabilities", [])
                if capability in capabilities:
                    matching_agents.append(agent)
            return matching_agents


class A2AMessageBroker:
    """A2A 메시지 브로커 - 메시지 라우팅 및 배포"""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._message_history: List[A2AMessage] = []
        self._max_history = 1000
    
    async def route_message(self, message: A2AMessage) -> bool:
        """
        메시지 라우팅
        
        Args:
            message: A2A 메시지
            
        Returns:
            bool: 라우팅 성공 여부
        """
        # 메시지 히스토리에 추가
        self._message_history.append(message)
        if len(self._message_history) > self._max_history:
            self._message_history.pop(0)
        
        # TTL 확인
        if message.ttl > 0:
            message_time = datetime.fromisoformat(message.timestamp.replace('Z', '+00:00'))
            age = (datetime.utcnow() - message_time.replace(tzinfo=None)).total_seconds()
            if age > message.ttl:
                logger.warning(f"Message {message.message_id} expired (age: {age}s, ttl: {message.ttl}s)")
                return False
        
        # 브로드캐스트인 경우
        if not message.target_agent:
            return await self._broadcast_message(message)
        
        # 특정 agent로 전송
        target_agent_info = await self.registry.get_agent(message.target_agent)
        if not target_agent_info:
            logger.warning(f"Target agent not found: {message.target_agent}")
            return False
        
        a2a_adapter = target_agent_info.get("a2a_adapter")
        if not a2a_adapter:
            logger.warning(f"Target agent has no A2A adapter: {message.target_agent}")
            return False
        
        # 메시지 큐에 추가 (비동기 처리)
        logger.info(f"Routing message {message.message_id} ({message.message_type}) from {message.source_agent} to {message.target_agent}")
        await a2a_adapter._message_queue.put(message)
        logger.debug(f"Message {message.message_id} added to queue for agent {message.target_agent}")
        return True
    
    async def _broadcast_message(self, message: A2AMessage) -> bool:
        """브로드캐스트 메시지 전송"""
        agents = await self.registry.list_agents()
        success_count = 0
        
        for agent_info in agents:
            agent_id = agent_info.get("agent_id")
            if agent_id == message.source_agent:
                continue  # 자기 자신에게는 전송하지 않음
            
            a2a_adapter = agent_info.get("a2a_adapter")
            if a2a_adapter:
                try:
                    await a2a_adapter._message_queue.put(message)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to send message to {agent_id}: {e}")
        
        logger.info(f"Broadcast message {message.message_id} sent to {success_count} agents")
        return success_count > 0
    
    def get_message_history(self, limit: int = 100) -> List[A2AMessage]:
        """메시지 히스토리 조회"""
        return self._message_history[-limit:]


# 전역 레지스트리 및 브로커 인스턴스
_global_registry: Optional[AgentRegistry] = None
_global_broker: Optional[A2AMessageBroker] = None


def get_global_registry() -> AgentRegistry:
    """전역 레지스트리 인스턴스 반환"""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def get_global_broker() -> A2AMessageBroker:
    """전역 브로커 인스턴스 반환"""
    global _global_broker
    if _global_broker is None:
        _global_broker = A2AMessageBroker(get_global_registry())
    return _global_broker

