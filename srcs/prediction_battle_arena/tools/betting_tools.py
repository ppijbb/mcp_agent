"""
베팅 관련 MCP 도구
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PlaceBetInput(BaseModel):
    """베팅 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    battle_id: str = Field(description="배틀 ID")
    prediction_id: str = Field(description="예측 ID")
    amount: float = Field(description="베팅 금액 (코인)")
    multiplier: Optional[float] = Field(default=1.0, description="배율")


class BettingTools:
    """
    베팅 관련 도구 모음
    
    베팅 처리, 베팅 조회, 베팅 취소 기능 제공
    """
    
    def __init__(self, data_dir: str = "prediction_battle_data"):
        """
        BettingTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.bets_file = self.data_dir / "bets.json"
        self.users_file = self.data_dir / "users.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if self.bets_file.exists():
            with open(self.bets_file, 'r', encoding='utf-8') as f:
                self.bets = json.load(f)
        else:
            self.bets = {}
        
        if self.users_file.exists():
            with open(self.users_file, 'r', encoding='utf-8') as f:
                self.users = json.load(f)
        else:
            self.users = {}
    
    def _save_data(self):
        """데이터 저장"""
        with open(self.bets_file, 'w', encoding='utf-8') as f:
            json.dump(self.bets, f, indent=2, ensure_ascii=False)
        
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """베팅 도구 초기화"""
        self.tools.append(self._place_bet_tool())
        self.tools.append(self._get_bet_tool())
        logger.info(f"Initialized {len(self.tools)} betting tools")
    
    def _place_bet_tool(self) -> BaseTool:
        @tool("betting_place", args_schema=PlaceBetInput)
        def place_bet(
            user_id: str,
            battle_id: str,
            prediction_id: str,
            amount: float,
            multiplier: Optional[float] = 1.0
        ) -> str:
            """
            베팅을 처리합니다.
            
            Args:
                user_id: 사용자 ID
                battle_id: 배틀 ID
                prediction_id: 예측 ID
                amount: 베팅 금액 (코인)
                multiplier: 배율
            Returns:
                베팅 결과 (JSON 문자열)
            """
            logger.info(f"Placing bet: user {user_id}, amount {amount}, multiplier {multiplier}")
            
            # 사용자 확인
            if user_id not in self.users:
                self.users[user_id] = {
                    "user_id": user_id,
                    "coins": 1000.0,  # 기본 코인
                    "created_at": datetime.now().isoformat()
                }
            
            user = self.users[user_id]
            
            # 코인 확인
            if user.get("coins", 0) < amount:
                return json.dumps({
                    "error": "코인이 부족합니다.",
                    "required": amount,
                    "available": user.get("coins", 0)
                }, ensure_ascii=False)
            
            # 베팅 생성
            bet_id = f"bet_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            bet_data = {
                "bet_id": bet_id,
                "user_id": user_id,
                "battle_id": battle_id,
                "prediction_id": prediction_id,
                "amount": amount,
                "multiplier": multiplier,
                "potential_winnings": amount * multiplier,
                "status": "active",
                "created_at": datetime.now().isoformat()
            }
            
            # 코인 차감
            user["coins"] -= amount
            user["total_bet_amount"] = user.get("total_bet_amount", 0) + amount
            
            # 베팅 저장
            self.bets[bet_id] = bet_data
            self._save_data()
            
            result = {
                "bet_id": bet_id,
                "user_id": user_id,
                "amount": amount,
                "multiplier": multiplier,
                "potential_winnings": amount * multiplier,
                "remaining_coins": user["coins"],
                "status": "success"
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return place_bet
    
    def _get_bet_tool(self) -> BaseTool:
        @tool("betting_get")
        def get_bet(bet_id: str) -> str:
            """
            베팅 정보를 조회합니다.
            
            Args:
                bet_id: 베팅 ID
            Returns:
                베팅 정보 (JSON 문자열)
            """
            logger.info(f"Getting bet {bet_id}")
            
            if bet_id not in self.bets:
                return json.dumps({"error": "베팅을 찾을 수 없습니다."}, ensure_ascii=False)
            
            return json.dumps(self.bets[bet_id], ensure_ascii=False, indent=2)
        return get_bet
    
    def get_tools(self) -> List[BaseTool]:
        """모든 베팅 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 베팅 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

