"""
메시지 허브

다중 에이전트 시스템의 메시지 통신을 관리하는 중앙 허브
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MessageType(Enum):
    """메시지 타입"""
    # 시스템 메시지
    SYSTEM_INFO = "system_info"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_ERROR = "system_error"
    
    # 게임 메시지
    GAME_START = "game_start"
    GAME_END = "game_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    PHASE_CHANGE = "phase_change"
    
    # 에이전트 메시지
    AGENT_REQUEST = "agent_request"
    AGENT_RESPONSE = "agent_response"
    AGENT_NOTIFICATION = "agent_notification"
    
    # 플레이어 메시지
    PLAYER_ACTION = "player_action"
    PLAYER_MESSAGE = "player_message"
    PLAYER_STATUS = "player_status"
    
    # 게임 액션 메시지
    ACTION_SUBMITTED = "action_submitted"
    ACTION_EXECUTED = "action_executed"
    ACTION_FAILED = "action_failed"
    
    # 분석 메시지
    ANALYSIS_REQUEST = "analysis_request"
    ANALYSIS_RESULT = "analysis_result"
    
    # 규칙 메시지
    RULE_QUERY = "rule_query"
    RULE_RESPONSE = "rule_response"
    
    # 페르소나 메시지
    PERSONA_REQUEST = "persona_request"
    PERSONA_GENERATED = "persona_generated"
    
    # 심판 메시지
    REFEREE_REQUEST = "referee_request"
    REFEREE_DECISION = "referee_decision"
    
    # 점수 메시지
    SCORE_UPDATE = "score_update"
    SCORE_CALCULATION = "score_calculation"


class MessagePriority(Enum):
    """메시지 우선순위"""
    CRITICAL = 0      # 즉시 처리 필요
    HIGH = 1          # 높은 우선순위
    NORMAL = 2        # 일반 우선순위
    LOW = 3           # 낮은 우선순위
    BACKGROUND = 4    # 백그라운드 처리


@dataclass
class Message:
    """메시지 정의"""
    # 기본 정보
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.SYSTEM_INFO
    priority: MessagePriority = MessagePriority.NORMAL
    
    # 발신자/수신자
    sender_id: str = ""
    recipient_ids: List[str] = field(default_factory=list)
    broadcast: bool = False  # 모든 에이전트에게 브로드캐스트
    
    # 메시지 내용
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 시간 정보
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    # 상태
    is_processed: bool = False
    is_delivered: bool = False
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
    
    def is_expired(self) -> bool:
        """메시지 만료 여부 확인"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "sender_id": self.sender_id,
            "recipient_ids": self.recipient_ids,
            "broadcast": self.broadcast,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "is_processed": self.is_processed,
            "is_delivered": self.is_delivered,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """딕셔너리에서 생성"""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            message_type=MessageType(data.get("message_type", "system_info")),
            priority=MessagePriority(data.get("priority", 2)),
            sender_id=data.get("sender_id", ""),
            recipient_ids=data.get("recipient_ids", []),
            broadcast=data.get("broadcast", False),
            content=data.get("content", {}),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            processed_at=datetime.fromisoformat(data["processed_at"]) if data.get("processed_at") else None,
            is_processed=data.get("is_processed", False),
            is_delivered=data.get("is_delivered", False),
            error_message=data.get("error_message")
        )


class MessageHub:
    """
    메시지 허브
    
    다중 에이전트 시스템의 메시지 라우팅, 필터링, 배달을 관리
    """
    
    def __init__(self):
        self.agents: Dict[str, Callable] = {}  # 에이전트 ID -> 메시지 핸들러
        self.message_queue: List[Message] = []
        self.message_history: List[Message] = []
        self.subscribers: Dict[str, List[Callable]] = {}  # 메시지 타입별 구독자
        self.routing_rules: Dict[str, List[str]] = {}  # 라우팅 규칙
        
        # 통계
        self.stats = {
            "total_messages": 0,
            "delivered_messages": 0,
            "failed_messages": 0,
            "average_processing_time": 0.0
        }
        
        # 백그라운드 작업
        self._cleanup_task = None
        self._is_running = False
        
        logger.info("MessageHub 초기화 완료")
    
    async def start(self):
        """메시지 허브 시작"""
        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_messages())
        logger.info("MessageHub 시작됨")
    
    async def stop(self):
        """메시지 허브 중지"""
        self._is_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("MessageHub 중지됨")
    
    def register_agent(self, agent_id: str, message_handler: Callable):
        """에이전트 등록"""
        self.agents[agent_id] = message_handler
        logger.info(f"에이전트 등록: {agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """에이전트 등록 해제"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"에이전트 등록 해제: {agent_id}")
    
    def subscribe(self, message_type: MessageType, handler: Callable):
        """메시지 타입별 구독"""
        if message_type.value not in self.subscribers:
            self.subscribers[message_type.value] = []
        self.subscribers[message_type.value].append(handler)
        logger.debug(f"메시지 구독 등록: {message_type.value}")
    
    def unsubscribe(self, message_type: MessageType, handler: Callable):
        """메시지 구독 해제"""
        if message_type.value in self.subscribers:
            if handler in self.subscribers[message_type.value]:
                self.subscribers[message_type.value].remove(handler)
                logger.debug(f"메시지 구독 해제: {message_type.value}")
    
    async def send_message(self, message: Message) -> str:
        """메시지 전송"""
        self.stats["total_messages"] += 1
        
        # 만료 시간 설정 (기본값: 1시간)
        if message.expires_at is None:
            message.expires_at = datetime.now() + timedelta(hours=1)
        
        # 큐에 추가 (우선순위 정렬)
        self.message_queue.append(message)
        self.message_queue.sort(key=lambda x: x.priority.value)
        
        # 즉시 처리 시도
        await self._process_message_queue()
        
        logger.info(f"메시지 전송: {message.message_type.value} (ID: {message.message_id})")
        return message.message_id
    
    async def send_to_agent(self, agent_id: str, message_type: MessageType, content: Dict[str, Any], **kwargs) -> str:
        """특정 에이전트에게 메시지 전송"""
        message = Message(
            message_type=message_type,
            sender_id="system",
            recipient_ids=[agent_id],
            content=content,
            **kwargs
        )
        return await self.send_message(message)
    
    async def broadcast(self, message_type: MessageType, content: Dict[str, Any], **kwargs) -> str:
        """모든 에이전트에게 브로드캐스트"""
        message = Message(
            message_type=message_type,
            sender_id="system",
            broadcast=True,
            content=content,
            **kwargs
        )
        return await self.send_message(message)
    
    async def _process_message_queue(self):
        """메시지 큐 처리"""
        while self.message_queue:
            message = self.message_queue.pop(0)
            
            try:
                # 만료된 메시지 제거
                if message.is_expired():
                    self.stats["failed_messages"] += 1
                    logger.warning(f"만료된 메시지 무시: {message.message_id}")
                    continue
                
                # 메시지 배달
                await self._deliver_message(message)
                
            except Exception as e:
                message.error_message = str(e)
                self.stats["failed_messages"] += 1
                logger.error(f"메시지 처리 실패: {message.message_id} - {e}")
            
            finally:
                # 히스토리에 추가
                self.message_history.append(message)
    
    async def _deliver_message(self, message: Message):
        """메시지 배달"""
        start_time = datetime.now()
        
        try:
            # 구독자들에게 알림
            if message.message_type.value in self.subscribers:
                for handler in self.subscribers[message.message_type.value]:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"구독자 핸들러 오류: {e}")
            
            # 특정 수신자들에게 배달
            if message.broadcast:
                # 모든 에이전트에게 배달
                for agent_id, handler in self.agents.items():
                    if agent_id != message.sender_id:  # 발신자 제외
                        await self._deliver_to_agent(agent_id, handler, message)
            else:
                # 지정된 수신자들에게만 배달
                for recipient_id in message.recipient_ids:
                    if recipient_id in self.agents:
                        handler = self.agents[recipient_id]
                        await self._deliver_to_agent(recipient_id, handler, message)
                    else:
                        logger.warning(f"수신자 없음: {recipient_id}")
            
            # 처리 완료
            message.is_delivered = True
            message.processed_at = datetime.now()
            self.stats["delivered_messages"] += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"메시지 배달 완료: {message.message_id} ({processing_time:.3f}s)")
            
        except Exception as e:
            message.error_message = str(e)
            self.stats["failed_messages"] += 1
            logger.error(f"메시지 배달 실패: {message.message_id} - {e}")
            raise
    
    async def _deliver_to_agent(self, agent_id: str, handler: Callable, message: Message):
        """개별 에이전트에게 메시지 배달"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(message)
            else:
                handler(message)
        except Exception as e:
            logger.error(f"에이전트 {agent_id} 메시지 처리 실패: {e}")
            raise
    
    async def _cleanup_expired_messages(self):
        """만료된 메시지 정리"""
        while self._is_running:
            try:
                # 만료된 메시지 제거
                expired_count = 0
                for message in self.message_history[:]:
                    if message.is_expired():
                        self.message_history.remove(message)
                        expired_count += 1
                
                if expired_count > 0:
                    logger.debug(f"만료된 메시지 {expired_count}개 정리됨")
                
                # 히스토리 크기 제한 (최대 1000개)
                if len(self.message_history) > 1000:
                    excess = len(self.message_history) - 1000
                    self.message_history = self.message_history[excess:]
                    logger.debug(f"메시지 히스토리 {excess}개 정리됨")
                
                await asyncio.sleep(60)  # 1분마다 정리
                
            except Exception as e:
                logger.error(f"메시지 정리 중 오류: {e}")
                await asyncio.sleep(60)
    
    def get_message(self, message_id: str) -> Optional[Message]:
        """메시지 조회"""
        for message in self.message_history:
            if message.message_id == message_id:
                return message
        return None
    
    def get_messages_for_agent(self, agent_id: str, limit: int = 50) -> List[Message]:
        """에이전트별 메시지 조회"""
        messages = []
        for message in reversed(self.message_history):
            if (agent_id in message.recipient_ids or message.broadcast) and message.sender_id != agent_id:
                messages.append(message)
                if len(messages) >= limit:
                    break
        return messages
    
    def get_messages_by_type(self, message_type: MessageType, limit: int = 50) -> List[Message]:
        """메시지 타입별 조회"""
        messages = []
        for message in reversed(self.message_history):
            if message.message_type == message_type:
                messages.append(message)
                if len(messages) >= limit:
                    break
        return messages
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        total = self.stats["delivered_messages"] + self.stats["failed_messages"]
        if total > 0:
            avg_time = sum(
                (m.processed_at - m.created_at).total_seconds() 
                for m in self.message_history 
                if m.processed_at
            ) / total
        else:
            avg_time = 0.0
        
        return {
            **self.stats,
            "average_processing_time": avg_time,
            "queue_size": len(self.message_queue),
            "history_size": len(self.message_history),
            "registered_agents": len(self.agents),
            "active_subscribers": sum(len(subscribers) for subscribers in self.subscribers.values())
        }


# 편의 함수들
def create_system_message(content: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL) -> Message:
    """시스템 메시지 생성"""
    return Message(
        message_type=MessageType.SYSTEM_INFO,
        sender_id="system",
        broadcast=True,
        content=content,
        priority=priority
    )


def create_agent_request(sender_id: str, recipient_id: str, request_type: str, data: Dict[str, Any]) -> Message:
    """에이전트 요청 메시지 생성"""
    return Message(
        message_type=MessageType.AGENT_REQUEST,
        sender_id=sender_id,
        recipient_ids=[recipient_id],
        content={
            "request_type": request_type,
            "data": data
        }
    )


def create_game_action_message(player_id: str, action_type: str, action_data: Dict[str, Any]) -> Message:
    """게임 액션 메시지 생성"""
    return Message(
        message_type=MessageType.PLAYER_ACTION,
        sender_id=player_id,
        broadcast=True,
        content={
            "action_type": action_type,
            "action_data": action_data
        }
    )


# 싱글톤 인스턴스
_message_hub = None

def get_message_hub() -> MessageHub:
    """메시지 허브 싱글톤 인스턴스 반환"""
    global _message_hub
    if _message_hub is None:
        _message_hub = MessageHub()
    return _message_hub 