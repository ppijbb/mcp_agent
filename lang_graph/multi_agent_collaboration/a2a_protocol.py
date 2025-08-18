"""
A2A (Agent-to-Agent) 프로토콜 구현
에이전트 간의 표준화된 통신과 협업을 위한 프로토콜을 정의합니다.
"""

import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """A2A 메시지 타입"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

class MessagePriority(Enum):
    """메시지 우선순위"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class AgentCapability:
    """에이전트 능력 정의"""
    name: str
    description: str
    version: str
    supported_operations: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_metrics: Optional[Dict[str, Any]] = None

@dataclass
class AgentCard:
    """에이전트 카드 - 에이전트의 기능과 API를 설명하는 메타데이터"""
    agent_id: str
    name: str
    description: str
    version: str
    capabilities: List[AgentCapability]
    contact_info: Dict[str, str]
    security_requirements: List[str]
    api_endpoints: List[str]
    authentication_methods: List[str]
    created_at: datetime
    last_updated: datetime
    status: str = "active"
    
    def to_dict(self) -> Dict[str, Any]:
        """에이전트 카드를 딕셔너리로 변환"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        return data

@dataclass
class A2AMessage:
    """A2A 메시지 구조"""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    security_context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """메시지를 딕셔너리로 변환"""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data

class A2AMessageBroker:
    """A2A 메시지 브로커"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[callable]] = {}
        self._message_queue: List[A2AMessage] = []
        self._delivered_messages: Dict[str, bool] = {}
        self._agent_cards: Dict[str, AgentCard] = {}
        self._running = False
    
    async def start(self):
        """메시지 브로커 시작"""
        self._running = True
        logger.info("A2A Message Broker started")
    
    async def stop(self):
        """메시지 브로커 중지"""
        self._running = False
        logger.info("A2A Message Broker stopped")
    
    def register_agent(self, agent_card: AgentCard) -> bool:
        """에이전트 등록"""
        try:
            self._agent_cards[agent_card.agent_id] = agent_card
            self._subscribers[agent_card.agent_id] = []
            logger.info(f"Agent {agent_card.name} ({agent_card.agent_id}) registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register agent {agent_card.agent_id}: {str(e)}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """에이전트 등록 해제"""
        try:
            if agent_id in self._agent_cards:
                del self._agent_cards[agent_id]
                if agent_id in self._subscribers:
                    del self._subscribers[agent_id]
                logger.info(f"Agent {agent_id} unregistered successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {str(e)}")
            return False
    
    def subscribe(self, agent_id: str, callback: callable) -> bool:
        """에이전트 메시지 구독"""
        try:
            if agent_id not in self._subscribers:
                self._subscribers[agent_id] = []
            self._subscribers[agent_id].append(callback)
            logger.info(f"Agent {agent_id} subscribed to messages")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe agent {agent_id}: {str(e)}")
            return False
    
    def unsubscribe(self, agent_id: str, callback: callable) -> bool:
        """에이전트 메시지 구독 해제"""
        try:
            if agent_id in self._subscribers and callback in self._subscribers[agent_id]:
                self._subscribers[agent_id].remove(callback)
                logger.info(f"Agent {agent_id} unsubscribed from messages")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unsubscribe agent {agent_id}: {str(e)}")
            return False
    
    async def publish_message(self, message: A2AMessage) -> bool:
        """메시지 발행"""
        try:
            # 메시지 유효성 검사
            if not self._validate_message(message):
                logger.warning(f"Invalid message received: {message.message_id}")
                return False
            
            # 메시지 큐에 추가
            self._message_queue.append(message)
            self._delivered_messages[message.message_id] = False
            
            # 수신자에게 즉시 전달
            await self._deliver_message(message)
            
            logger.info(f"Message {message.message_id} published successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message {message.message_id}: {str(e)}")
            return False
    
    async def _deliver_message(self, message: A2AMessage):
        """메시지 전달"""
        try:
            if message.receiver_id in self._subscribers:
                for callback in self._subscribers[message.receiver_id]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        logger.error(f"Error in message callback for agent {message.receiver_id}: {str(e)}")
            
            # 브로드캐스트 메시지인 경우 모든 구독자에게 전달
            if message.receiver_id == "*":
                for agent_id, callbacks in self._subscribers.items():
                    if agent_id != message.sender_id:  # 발신자 제외
                        for callback in callbacks:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(message)
                                else:
                                    callback(message)
                            except Exception as e:
                                logger.error(f"Error in broadcast callback for agent {agent_id}: {str(e)}")
                                
        except Exception as e:
            logger.error(f"Failed to deliver message {message.message_id}: {str(e)}")
    
    def _validate_message(self, message: A2AMessage) -> bool:
        """메시지 유효성 검사"""
        if not message.message_id or not message.sender_id or not message.receiver_id:
            return False
        
        if message.sender_id not in self._agent_cards:
            logger.warning(f"Unknown sender agent: {message.sender_id}")
            return False
        
        if message.receiver_id != "*" and message.receiver_id not in self._agent_cards:
            logger.warning(f"Unknown receiver agent: {message.receiver_id}")
            return False
        
        return True
    
    def get_agent_card(self, agent_id: str) -> Optional[AgentCard]:
        """에이전트 카드 조회"""
        return self._agent_cards.get(agent_id)
    
    def list_agents(self) -> List[AgentCard]:
        """등록된 에이전트 목록 조회"""
        return list(self._agent_cards.values())
    
    def search_agents(self, capability: str = None, status: str = None) -> List[AgentCard]:
        """에이전트 검색"""
        agents = self.list_agents()
        
        if capability:
            agents = [
                agent for agent in agents
                if any(cap.name == capability for cap in agent.capabilities)
            ]
        
        if status:
            agents = [agent for agent in agents if agent.status == status]
        
        return agents

class A2AAgent:
    """A2A 프로토콜을 지원하는 에이전트 기본 클래스"""
    
    def __init__(self, agent_card: AgentCard, message_broker: A2AMessageBroker):
        self.agent_card = agent_card
        self.message_broker = message_broker
        self.message_handlers: Dict[MessageType, callable] = {}
        self._running = False
        
        # 기본 메시지 핸들러 등록
        self._register_default_handlers()
        
        # 메시지 브로커에 에이전트 등록
        self.message_broker.register_agent(agent_card)
        self.message_broker.subscribe(agent_card.agent_id, self._handle_message)
    
    def _register_default_handlers(self):
        """기본 메시지 핸들러 등록"""
        self.register_message_handler(MessageType.HEARTBEAT, self._handle_heartbeat)
        self.register_message_handler(MessageType.REQUEST, self._handle_request)
        self.register_message_handler(MessageType.NOTIFICATION, self._handle_notification)
    
    def register_message_handler(self, message_type: MessageType, handler: callable):
        """메시지 핸들러 등록"""
        self.message_handlers[message_type] = handler
    
    async def _handle_message(self, message: A2AMessage):
        """메시지 처리"""
        try:
            if message.message_type in self.message_handlers:
                handler = self.message_handlers[message.message_type]
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            else:
                logger.warning(f"No handler registered for message type: {message.message_type}")
                
        except Exception as e:
            logger.error(f"Error handling message {message.message_id}: {str(e)}")
            # 에러 응답 전송
            await self._send_error_response(message, str(e))
    
    async def _handle_heartbeat(self, message: A2AMessage):
        """하트비트 메시지 처리"""
        # 하트비트 응답 전송
        response = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_card.agent_id,
            receiver_id=message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=MessagePriority.LOW,
            content={"status": "alive", "timestamp": datetime.now().isoformat()},
            timestamp=datetime.now(),
            correlation_id=message.message_id
        )
        await self.send_message(response)
    
    async def _handle_request(self, message: A2AMessage):
        """요청 메시지 처리"""
        # 기본 구현 - 하위 클래스에서 오버라이드
        logger.info(f"Received request from {message.sender_id}: {message.content}")
    
    async def _handle_notification(self, message: A2AMessage):
        """알림 메시지 처리"""
        # 기본 구현 - 하위 클래스에서 오버라이드
        logger.info(f"Received notification from {message.sender_id}: {message.content}")
    
    async def _send_error_response(self, original_message: A2AMessage, error: str):
        """에러 응답 전송"""
        error_response = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_card.agent_id,
            receiver_id=original_message.sender_id,
            message_type=MessageType.ERROR,
            priority=MessagePriority.HIGH,
            content={"error": error, "original_message_id": original_message.message_id},
            timestamp=datetime.now(),
            correlation_id=original_message.message_id
        )
        await self.send_message(error_response)
    
    async def send_message(self, message: A2AMessage) -> bool:
        """메시지 전송"""
        return await self.message_broker.publish_message(message)
    
    async def send_request(self, receiver_id: str, content: Dict[str, Any], 
                          priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """요청 메시지 전송"""
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_card.agent_id,
            receiver_id=receiver_id,
            message_type=MessageType.REQUEST,
            priority=priority,
            content=content,
            timestamp=datetime.now()
        )
        
        if await self.send_message(message):
            return message.message_id
        return None
    
    async def send_response(self, receiver_id: str, content: Dict[str, Any], 
                           correlation_id: str, priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """응답 메시지 전송"""
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_card.agent_id,
            receiver_id=receiver_id,
            message_type=MessageType.RESPONSE,
            priority=priority,
            content=content,
            timestamp=datetime.now(),
            correlation_id=correlation_id
        )
        
        if await self.send_message(message):
            return message.message_id
        return None
    
    async def start(self):
        """에이전트 시작"""
        self._running = True
        logger.info(f"Agent {self.agent_card.name} started")
    
    async def stop(self):
        """에이전트 중지"""
        self._running = False
        logger.info(f"Agent {self.agent_card.name} stopped")
    
    def is_running(self) -> bool:
        """에이전트 실행 상태 확인"""
        return self._running

# 전역 A2A 메시지 브로커 인스턴스
a2a_message_broker = A2AMessageBroker()
