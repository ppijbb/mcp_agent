"""
실시간 WebSocket 서비스

배틀 업데이트, 리더보드 업데이트를 실시간으로 브로드캐스트
"""

import asyncio
import json
import logging
from typing import Dict, Set, Any
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class RealtimeService:
    """
    실시간 WebSocket 서비스

    배틀 업데이트, 리더보드 업데이트를 실시간으로 브로드캐스트
    """

    def __init__(self):
        """
        RealtimeService 초기화
        """
        # 연결 관리: battle_id -> WebSocket 연결 집합
        self.battle_connections: Dict[str, Set[Any]] = defaultdict(set)

        # 글로벌 연결 (리더보드 등)
        self.global_connections: Set[Any] = set()

        # 업데이트 큐
        self.update_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)

        # 락
        self.connection_lock = asyncio.Lock()

        logger.info("RealtimeService initialized")

    async def register_battle_connection(self, battle_id: str, websocket: Any):
        """
        배틀 연결 등록

        Args:
            battle_id: 배틀 ID
            websocket: WebSocket 연결
        """
        async with self.connection_lock:
            self.battle_connections[battle_id].add(websocket)
            logger.info(f"Battle connection registered: battle_id={battle_id}, total={len(self.battle_connections[battle_id])}")

    async def unregister_battle_connection(self, battle_id: str, websocket: Any):
        """
        배틀 연결 해제

        Args:
            battle_id: 배틀 ID
            websocket: WebSocket 연결
        """
        async with self.connection_lock:
            if battle_id in self.battle_connections:
                self.battle_connections[battle_id].discard(websocket)
                if not self.battle_connections[battle_id]:
                    del self.battle_connections[battle_id]
            logger.info(f"Battle connection unregistered: battle_id={battle_id}")

    async def register_global_connection(self, websocket: Any):
        """
        글로벌 연결 등록 (리더보드 등)

        Args:
            websocket: WebSocket 연결
        """
        async with self.connection_lock:
            self.global_connections.add(websocket)
            logger.info(f"Global connection registered: total={len(self.global_connections)}")

    async def unregister_global_connection(self, websocket: Any):
        """
        글로벌 연결 해제

        Args:
            websocket: WebSocket 연결
        """
        async with self.connection_lock:
            self.global_connections.discard(websocket)
            logger.info(f"Global connection unregistered: total={len(self.global_connections)}")

    async def broadcast_battle_update(self, battle_id: str, update: Dict[str, Any]):
        """
        배틀 업데이트 브로드캐스트

        Args:
            battle_id: 배틀 ID
            update: 업데이트 데이터
        """
        if battle_id not in self.battle_connections:
            return

        message = json.dumps({
            "type": "battle_update",
            "battle_id": battle_id,
            "data": update,
            "timestamp": datetime.now().isoformat()
        })

        connections = list(self.battle_connections[battle_id])
        if not connections:
            return

        # 모든 연결에 메시지 전송
        disconnected = []
        for ws in connections:
            try:
                await ws.send(message)
            except Exception as e:
                logger.warning(f"Failed to send message to connection: {e}")
                disconnected.append(ws)

        # 연결 해제된 것들 정리
        if disconnected:
            async with self.connection_lock:
                for ws in disconnected:
                    self.battle_connections[battle_id].discard(ws)

        logger.info(f"Broadcasted battle update: battle_id={battle_id}, connections={len(connections)}")

    async def broadcast_leaderboard_update(self, update: Dict[str, Any]):
        """
        리더보드 업데이트 브로드캐스트

        Args:
            update: 업데이트 데이터
        """
        message = json.dumps({
            "type": "leaderboard_update",
            "data": update,
            "timestamp": datetime.now().isoformat()
        })

        connections = list(self.global_connections)
        if not connections:
            return

        # 모든 연결에 메시지 전송
        disconnected = []
        for ws in connections:
            try:
                await ws.send(message)
            except Exception as e:
                logger.warning(f"Failed to send leaderboard update: {e}")
                disconnected.append(ws)

        # 연결 해제된 것들 정리
        if disconnected:
            async with self.connection_lock:
                for ws in disconnected:
                    self.global_connections.discard(ws)

        logger.info(f"Broadcasted leaderboard update: connections={len(connections)}")

    async def send_user_notification(self, user_id: str, notification: Dict[str, Any]):
        """
        사용자에게 알림 전송

        Args:
            user_id: 사용자 ID
            notification: 알림 데이터
        """
        # 사용자별 연결 관리가 필요한 경우 여기에 구현
        # 현재는 간단한 구조로 구현
        message = json.dumps({
            "type": "notification",
            "user_id": user_id,
            "data": notification,
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"Sent notification to user: {user_id}")

    def get_battle_connection_count(self, battle_id: str) -> int:
        """
        배틀 연결 수 조회

        Args:
            battle_id: 배틀 ID
        Returns:
            연결 수
        """
        return len(self.battle_connections.get(battle_id, set()))

    def get_global_connection_count(self) -> int:
        """
        글로벌 연결 수 조회

        Returns:
            연결 수
        """
        return len(self.global_connections)
