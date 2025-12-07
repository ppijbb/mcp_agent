"""
배틀 설정
"""

from dataclasses import dataclass
from typing import Dict
import os


@dataclass
class BattleConfig:
    """배틀 설정"""
    
    # 배틀 지속 시간 (초)
    QUICK_DURATION: int = 300  # 5분
    STANDARD_DURATION: int = 900  # 15분
    EXTENDED_DURATION: int = 1800  # 30분
    
    # 참가자 제한
    MIN_PARTICIPANTS: int = 2
    MAX_PARTICIPANTS: int = 100
    
    # 베팅 설정
    MIN_BET_AMOUNT: float = 10.0
    MAX_BET_AMOUNT: float = 10000.0
    DEFAULT_MULTIPLIER: float = 1.0
    
    # 보상 설정
    JACKPOT_PROBABILITY: float = 0.1  # 10%
    JACKPOT_MULTIPLIER: float = 100.0  # 100x
    
    # Redis 설정
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # WebSocket 설정
    WEBSOCKET_PORT: int = int(os.getenv("WEBSOCKET_PORT", "8765"))
    
    @classmethod
    def from_env(cls) -> "BattleConfig":
        """환경 변수에서 설정 로드"""
        return cls(
            QUICK_DURATION=int(os.getenv("BATTLE_DURATION_QUICK", "300")),
            STANDARD_DURATION=int(os.getenv("BATTLE_DURATION_STANDARD", "900")),
            EXTENDED_DURATION=int(os.getenv("BATTLE_DURATION_EXTENDED", "1800")),
            MIN_PARTICIPANTS=int(os.getenv("MIN_PARTICIPANTS", "2")),
            MAX_PARTICIPANTS=int(os.getenv("MAX_PARTICIPANTS", "100")),
            MIN_BET_AMOUNT=float(os.getenv("MIN_BET_AMOUNT", "10.0")),
            MAX_BET_AMOUNT=float(os.getenv("MAX_BET_AMOUNT", "10000.0")),
            REDIS_HOST=os.getenv("REDIS_HOST", "localhost"),
            REDIS_PORT=int(os.getenv("REDIS_PORT", "6379")),
            REDIS_DB=int(os.getenv("REDIS_DB", "0")),
            WEBSOCKET_PORT=int(os.getenv("WEBSOCKET_PORT", "8765"))
        )


# 전역 설정 인스턴스
battle_config = BattleConfig.from_env()

