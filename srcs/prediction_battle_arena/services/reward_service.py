"""
보상 서비스

랜덤 보상, 연승 보너스, 특별 이벤트 보상 관리
"""

import logging
import random
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RewardService:
    """
    보상 서비스
    
    랜덤 보상, 연승 보너스, 특별 이벤트 보상 관리
    """
    
    def __init__(self):
        """
        RewardService 초기화
        """
        # 보상 설정
        self.jackpot_probability = 0.1  # 10% 확률
        self.jackpot_multiplier = 100.0  # 100x
        self.normal_bonus_range = (1.5, 5.0)  # 1.5x ~ 5.0x
        
        # 연승 보너스 설정
        self.streak_bonuses = {
            3: 0.2,   # 3연승: 20% 보너스
            5: 0.5,   # 5연승: 50% 보너스
            10: 1.0,  # 10연승: 100% 보너스
            20: 2.0,  # 20연승: 200% 보너스
        }
        
        logger.info("RewardService initialized")
    
    def calculate_random_bonus(self, base_amount: float = 10.0) -> Dict[str, Any]:
        """
        랜덤 보상 계산
        
        Args:
            base_amount: 기본 보상 금액
        Returns:
            보상 정보
        """
        # 잭팟 확률 체크
        if random.random() < self.jackpot_probability:
            multiplier = self.jackpot_multiplier
            bonus_type = "jackpot"
            message = "🎉 잭팟! 100x 보너스!"
        else:
            multiplier = random.uniform(*self.normal_bonus_range)
            bonus_type = "normal"
            message = f"🎁 {multiplier:.1f}x 보너스!"
        
        amount = base_amount * multiplier
        
        return {
            "bonus_type": bonus_type,
            "multiplier": multiplier,
            "amount": amount,
            "message": message
        }
    
    def calculate_streak_bonus(self, base_reward: float, win_streak: int) -> float:
        """
        연승 보너스 계산
        
        Args:
            base_reward: 기본 보상
            win_streak: 연승 횟수
        Returns:
            보너스 금액
        """
        if win_streak < 3:
            return 0.0
        
        # 가장 높은 연승 보너스 적용
        bonus_rate = 0.0
        for streak, rate in sorted(self.streak_bonuses.items(), reverse=True):
            if win_streak >= streak:
                bonus_rate = rate
                break
        
        return base_reward * bonus_rate
    
    def calculate_total_reward(
        self,
        base_reward: float,
        accuracy_score: float,
        bet_amount: float,
        multiplier: float,
        win_streak: int = 0
    ) -> Dict[str, Any]:
        """
        총 보상 계산
        
        Args:
            base_reward: 기본 보상
            accuracy_score: 정확도 점수 (0.0 ~ 1.0)
            bet_amount: 베팅 금액
            multiplier: 베팅 배율
            win_streak: 연승 횟수
        Returns:
            보상 상세 정보
        """
        # 기본 보상 (정확도 기반)
        accuracy_reward = bet_amount * multiplier * accuracy_score
        
        # 연승 보너스
        streak_bonus = self.calculate_streak_bonus(accuracy_reward, win_streak)
        
        # 총 보상
        total_reward = accuracy_reward + streak_bonus
        
        # 랜덤 보너스 (10% 확률)
        random_bonus_info = None
        if random.random() < 0.1:
            random_bonus_info = self.calculate_random_bonus()
            total_reward += random_bonus_info["amount"]
        
        return {
            "base_reward": base_reward,
            "accuracy_reward": accuracy_reward,
            "streak_bonus": streak_bonus,
            "random_bonus": random_bonus_info,
            "total_reward": total_reward,
            "accuracy_score": accuracy_score,
            "win_streak": win_streak,
            "breakdown": {
                "accuracy_portion": accuracy_reward,
                "streak_portion": streak_bonus,
                "random_portion": random_bonus_info["amount"] if random_bonus_info else 0.0
            }
        }
    
    def get_streak_message(self, win_streak: int) -> Optional[str]:
        """
        연승 메시지 생성
        
        Args:
            win_streak: 연승 횟수
        Returns:
            메시지
        """
        if win_streak >= 20:
            return f"🔥🔥🔥 {win_streak}연승! 전설의 예언자!"
        elif win_streak >= 10:
            return f"🔥🔥 {win_streak}연승! 예언의 신!"
        elif win_streak >= 5:
            return f"🔥 {win_streak}연승! 대단해요!"
        elif win_streak >= 3:
            return f"✨ {win_streak}연승! 좋은 흐름이에요!"
        return None

