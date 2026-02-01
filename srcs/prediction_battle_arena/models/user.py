"""
사용자 데이터 모델
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class UserStats:
    """사용자 통계"""

    total_battles: int = 0
    wins: int = 0
    losses: int = 0
    win_streak: int = 0
    best_win_streak: int = 0

    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy_rate: float = 0.0

    total_bet_amount: float = 0.0
    total_winnings: float = 0.0
    net_profit: float = 0.0

    level: int = 1
    experience_points: int = 0

    badges: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)

    def calculate_win_rate(self) -> float:
        """승률 계산"""
        if self.total_battles == 0:
            return 0.0
        return self.wins / self.total_battles

    def calculate_accuracy(self) -> float:
        """정확도 계산"""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "total_battles": self.total_battles,
            "wins": self.wins,
            "losses": self.losses,
            "win_streak": self.win_streak,
            "best_win_streak": self.best_win_streak,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "accuracy_rate": self.calculate_accuracy(),
            "win_rate": self.calculate_win_rate(),
            "total_bet_amount": self.total_bet_amount,
            "total_winnings": self.total_winnings,
            "net_profit": self.net_profit,
            "level": self.level,
            "experience_points": self.experience_points,
            "badges": self.badges,
            "achievements": self.achievements,
        }


@dataclass
class User:
    """사용자 데이터 모델"""

    user_id: str
    username: str = ""
    email: Optional[str] = None

    # 가상 화폐
    coins: float = 1000.0  # 기본 시작 코인

    # 통계
    stats: UserStats = field(default_factory=UserStats)

    # 메타데이터
    created_at: datetime = field(default_factory=datetime.now)
    last_active_at: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def add_coins(self, amount: float) -> bool:
        """코인 추가"""
        if amount < 0:
            return False
        self.coins += amount
        return True

    def spend_coins(self, amount: float) -> bool:
        """코인 사용"""
        if amount < 0 or self.coins < amount:
            return False
        self.coins -= amount
        return True

    def has_enough_coins(self, amount: float) -> bool:
        """충분한 코인 보유 여부"""
        return self.coins >= amount

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "coins": self.coins,
            "stats": self.stats.to_dict(),
            "created_at": self.created_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat(),
            "metadata": self.metadata,
        }
