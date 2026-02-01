"""
Redis 서비스

리더보드, 배틀 상태, 사용자 데이터를 Redis에 저장
"""

import logging
import json
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Redis가 없는 경우를 대비한 메모리 기반 구현
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory storage")


class RedisService:
    """
    Redis 서비스

    리더보드, 배틀 상태, 사용자 데이터 관리
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        RedisService 초기화

        Args:
            host: Redis 호스트
            port: Redis 포트
            db: Redis 데이터베이스 번호
        """
        self.host = host
        self.port = port
        self.db = db
        self.redis_client: Optional[redis.Redis] = None

        # Redis가 없는 경우 메모리 기반 저장소
        self.memory_store: Dict[str, Any] = {}

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    decode_responses=True
                )
                logger.info(f"RedisService initialized: {host}:{port}/{db}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}, using in-memory storage")
                self.redis_client = None
        else:
            logger.info("RedisService using in-memory storage")

    async def connect(self):
        """Redis 연결"""
        if self.redis_client and REDIS_AVAILABLE:
            try:
                await self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None

    async def close(self):
        """Redis 연결 종료"""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None

    async def update_leaderboard(self, user_id: str, score: float, category: str = "global"):
        """
        리더보드 업데이트

        Args:
            user_id: 사용자 ID
            score: 점수
            category: 카테고리 (global/weekly/monthly)
        """
        key = f"leaderboard:{category}"

        if self.redis_client:
            try:
                await self.redis_client.zadd(key, {user_id: score})
                logger.info(f"Updated leaderboard: {category}, user={user_id}, score={score}")
            except Exception as e:
                logger.error(f"Failed to update leaderboard in Redis: {e}")
        else:
            # 메모리 기반
            if key not in self.memory_store:
                self.memory_store[key] = {}
            self.memory_store[key][user_id] = score

    async def get_leaderboard(self, category: str = "global", limit: int = 100) -> List[Dict[str, Any]]:
        """
        리더보드 조회

        Args:
            category: 카테고리
            limit: 조회할 상위 순위 수
        Returns:
            리더보드 데이터
        """
        key = f"leaderboard:{category}"

        if self.redis_client:
            try:
                rankings = await self.redis_client.zrevrange(
                    key,
                    0,
                    limit - 1,
                    withscores=True
                )
                return [
                    {"user_id": uid, "score": float(score), "rank": idx + 1}
                    for idx, (uid, score) in enumerate(rankings)
                ]
            except Exception as e:
                logger.error(f"Failed to get leaderboard from Redis: {e}")
                return []
        else:
            # 메모리 기반
            if key not in self.memory_store:
                return []

            rankings = sorted(
                self.memory_store[key].items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]

            return [
                {"user_id": uid, "score": score, "rank": idx + 1}
                for idx, (uid, score) in enumerate(rankings)
            ]

    async def get_user_rank(self, user_id: str, category: str = "global") -> Optional[int]:
        """
        사용자 순위 조회

        Args:
            user_id: 사용자 ID
            category: 카테고리
        Returns:
            순위 (None if not found)
        """
        key = f"leaderboard:{category}"

        if self.redis_client:
            try:
                rank = await self.redis_client.zrevrank(key, user_id)
                return rank + 1 if rank is not None else None
            except Exception as e:
                logger.error(f"Failed to get user rank from Redis: {e}")
                return None
        else:
            # 메모리 기반
            if key not in self.memory_store:
                return None

            rankings = sorted(
                self.memory_store[key].items(),
                key=lambda x: x[1],
                reverse=True
            )

            for idx, (uid, _) in enumerate(rankings):
                if uid == user_id:
                    return idx + 1

            return None

    async def set_battle_state(self, battle_id: str, state: Dict[str, Any], ttl: Optional[int] = None):
        """
        배틀 상태 저장

        Args:
            battle_id: 배틀 ID
            state: 상태 데이터
            ttl: TTL (초)
        """
        key = f"battle:{battle_id}"
        value = json.dumps(state)

        if self.redis_client:
            try:
                if ttl:
                    await self.redis_client.setex(key, ttl, value)
                else:
                    await self.redis_client.set(key, value)
            except Exception as e:
                logger.error(f"Failed to set battle state in Redis: {e}")
        else:
            # 메모리 기반
            self.memory_store[key] = state

    async def get_battle_state(self, battle_id: str) -> Optional[Dict[str, Any]]:
        """
        배틀 상태 조회

        Args:
            battle_id: 배틀 ID
        Returns:
            상태 데이터
        """
        key = f"battle:{battle_id}"

        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                return json.loads(value) if value else None
            except Exception as e:
                logger.error(f"Failed to get battle state from Redis: {e}")
                return None
        else:
            # 메모리 기반
            return self.memory_store.get(key)

    async def set_user_data(self, user_id: str, data: Dict[str, Any]):
        """
        사용자 데이터 저장

        Args:
            user_id: 사용자 ID
            data: 데이터
        """
        key = f"user:{user_id}"
        value = json.dumps(data)

        if self.redis_client:
            try:
                await self.redis_client.set(key, value)
            except Exception as e:
                logger.error(f"Failed to set user data in Redis: {e}")
        else:
            # 메모리 기반
            self.memory_store[key] = data

    async def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        사용자 데이터 조회

        Args:
            user_id: 사용자 ID
        Returns:
            데이터
        """
        key = f"user:{user_id}"

        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                return json.loads(value) if value else None
            except Exception as e:
                logger.error(f"Failed to get user data from Redis: {e}")
                return None
        else:
            # 메모리 기반
            return self.memory_store.get(key)
