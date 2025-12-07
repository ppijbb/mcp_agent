"""
보상 관련 MCP 도구
"""

import logging
import json
import random
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CalculateRewardInput(BaseModel):
    """보상 계산 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    battle_id: str = Field(description="배틀 ID")
    accuracy_score: float = Field(description="정확도 점수 (0.0 ~ 1.0)")
    bet_amount: float = Field(description="베팅 금액")
    multiplier: float = Field(description="베팅 배율")
    win_streak: Optional[int] = Field(default=0, description="연승 횟수")


class RewardTools:
    """
    보상 관련 도구 모음
    
    보상 계산, 랜덤 보상, 연승 보너스 기능 제공
    """
    
    def __init__(self, data_dir: str = "prediction_battle_data"):
        """
        RewardTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rewards_file = self.data_dir / "rewards.json"
        self.users_file = self.data_dir / "users.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if self.rewards_file.exists():
            with open(self.rewards_file, 'r', encoding='utf-8') as f:
                self.rewards = json.load(f)
        else:
            self.rewards = {}
        
        if self.users_file.exists():
            with open(self.users_file, 'r', encoding='utf-8') as f:
                self.users = json.load(f)
        else:
            self.users = {}
    
    def _save_data(self):
        """데이터 저장"""
        with open(self.rewards_file, 'w', encoding='utf-8') as f:
            json.dump(self.rewards, f, indent=2, ensure_ascii=False)
        
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """보상 도구 초기화"""
        self.tools.append(self._calculate_reward_tool())
        self.tools.append(self._random_bonus_tool())
        logger.info(f"Initialized {len(self.tools)} reward tools")
    
    def _calculate_reward_tool(self) -> BaseTool:
        @tool("reward_calculate", args_schema=CalculateRewardInput)
        def calculate_reward(
            user_id: str,
            battle_id: str,
            accuracy_score: float,
            bet_amount: float,
            multiplier: float,
            win_streak: Optional[int] = 0
        ) -> str:
            """
            보상을 계산합니다.
            
            Args:
                user_id: 사용자 ID
                battle_id: 배틀 ID
                accuracy_score: 정확도 점수 (0.0 ~ 1.0)
                bet_amount: 베팅 금액
                multiplier: 베팅 배율
                win_streak: 연승 횟수
            Returns:
                보상 계산 결과 (JSON 문자열)
            """
            logger.info(f"Calculating reward for user {user_id}, accuracy: {accuracy_score}")
            
            # 기본 보상 계산 (정확도 기반)
            base_reward = bet_amount * multiplier * accuracy_score
            
            # 연승 보너스
            streak_bonus = 0.0
            if win_streak >= 3:
                streak_bonus = base_reward * 0.2  # 3연승: 20% 보너스
            if win_streak >= 5:
                streak_bonus = base_reward * 0.5  # 5연승: 50% 보너스
            if win_streak >= 10:
                streak_bonus = base_reward * 1.0  # 10연승: 100% 보너스
            
            # 총 보상
            total_reward = base_reward + streak_bonus
            
            # 사용자 업데이트
            if user_id not in self.users:
                self.users[user_id] = {
                    "user_id": user_id,
                    "coins": 1000.0,
                    "win_streak": 0,
                    "total_winnings": 0.0
                }
            
            user = self.users[user_id]
            user["coins"] = user.get("coins", 0) + total_reward
            user["total_winnings"] = user.get("total_winnings", 0) + total_reward
            
            if accuracy_score >= 0.7:  # 승리
                user["win_streak"] = user.get("win_streak", 0) + 1
            else:  # 패배
                user["win_streak"] = 0
            
            # 보상 기록
            reward_id = f"reward_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            reward_data = {
                "reward_id": reward_id,
                "user_id": user_id,
                "battle_id": battle_id,
                "base_reward": base_reward,
                "streak_bonus": streak_bonus,
                "total_reward": total_reward,
                "accuracy_score": accuracy_score,
                "win_streak": win_streak,
                "created_at": datetime.now().isoformat()
            }
            
            self.rewards[reward_id] = reward_data
            self._save_data()
            
            result = {
                "reward_id": reward_id,
                "user_id": user_id,
                "base_reward": base_reward,
                "streak_bonus": streak_bonus,
                "total_reward": total_reward,
                "new_coins": user["coins"],
                "new_win_streak": user["win_streak"]
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return calculate_reward
    
    def _random_bonus_tool(self) -> BaseTool:
        @tool("reward_random_bonus")
        def random_bonus(user_id: str) -> str:
            """
            랜덤 보너스를 지급합니다.
            10% 확률로 100x 보너스, 그 외에는 일반 보너스
            
            Args:
                user_id: 사용자 ID
            Returns:
                랜덤 보너스 결과 (JSON 문자열)
            """
            logger.info(f"Random bonus for user {user_id}")
            
            # 10% 확률로 100x 보너스
            if random.random() < 0.1:
                bonus_multiplier = 100.0
                bonus_type = "jackpot"
            else:
                bonus_multiplier = random.uniform(1.5, 5.0)
                bonus_type = "normal"
            
            base_amount = 10.0  # 기본 보너스 금액
            bonus_amount = base_amount * bonus_multiplier
            
            # 사용자 업데이트
            if user_id not in self.users:
                self.users[user_id] = {
                    "user_id": user_id,
                    "coins": 1000.0
                }
            
            user = self.users[user_id]
            user["coins"] = user.get("coins", 0) + bonus_amount
            
            self._save_data()
            
            result = {
                "user_id": user_id,
                "bonus_type": bonus_type,
                "bonus_multiplier": bonus_multiplier,
                "bonus_amount": bonus_amount,
                "new_coins": user["coins"],
                "message": "🎉 잭팟!" if bonus_type == "jackpot" else "🎁 보너스 획득!"
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return random_bonus
    
    def get_tools(self) -> List[BaseTool]:
        """모든 보상 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 보상 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

