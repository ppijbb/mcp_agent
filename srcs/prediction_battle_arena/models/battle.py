"""
배틀 데이터 모델
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Set
from uuid import uuid4


class BattleStatus(Enum):
    """배틀 상태"""
    WAITING = "waiting"  # 대기 중 (참가자 모집)
    STARTED = "started"  # 시작됨 (예측 수집 중)
    PREDICTING = "predicting"  # 예측 단계
    BETTING = "betting"  # 베팅 단계
    CALCULATING = "calculating"  # 결과 계산 중
    FINISHED = "finished"  # 종료됨
    CANCELLED = "cancelled"  # 취소됨


class BattleType(Enum):
    """배틀 유형"""
    QUICK = "quick"  # 5분 배틀
    STANDARD = "standard"  # 15분 배틀
    EXTENDED = "extended"  # 30분 배틀


@dataclass
class Battle:
    """배틀 데이터 모델"""

    battle_id: str = field(default_factory=lambda: str(uuid4()))
    battle_type: BattleType = BattleType.QUICK
    status: BattleStatus = BattleStatus.WAITING
    topic: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # 참가자
    participants: Set[str] = field(default_factory=set)  # user_id 집합
    min_participants: int = 2
    max_participants: int = 100

    # 예측 및 베팅
    predictions: Dict[str, str] = field(default_factory=dict)  # user_id -> prediction_id
    bets: Dict[str, float] = field(default_factory=dict)  # user_id -> bet_amount

    # 결과
    winner_id: Optional[str] = None
    results: Dict[str, Dict] = field(default_factory=dict)  # user_id -> 결과 데이터

    # 메타데이터
    metadata: Dict = field(default_factory=dict)

    def get_duration_seconds(self) -> int:
        """배틀 지속 시간 (초)"""
        duration_map = {
            BattleType.QUICK: 300,  # 5분
            BattleType.STANDARD: 900,  # 15분
            BattleType.EXTENDED: 1800,  # 30분
        }
        return duration_map.get(self.battle_type, 300)

    def get_remaining_seconds(self) -> Optional[int]:
        """남은 시간 (초)"""
        if not self.started_at:
            return None

        elapsed = (datetime.now() - self.started_at).total_seconds()
        remaining = self.get_duration_seconds() - elapsed
        return max(0, int(remaining))

    def is_full(self) -> bool:
        """참가자 가득 찼는지"""
        return len(self.participants) >= self.max_participants

    def can_join(self) -> bool:
        """참가 가능한지"""
        return (
            self.status == BattleStatus.WAITING and
            not self.is_full()
        )

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "battle_id": self.battle_id,
            "battle_type": self.battle_type.value,
            "status": self.status.value,
            "topic": self.topic,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "participants": list(self.participants),
            "participant_count": len(self.participants),
            "predictions": self.predictions,
            "bets": self.bets,
            "winner_id": self.winner_id,
            "results": self.results,
            "remaining_seconds": self.get_remaining_seconds(),
            "metadata": self.metadata,
        }
