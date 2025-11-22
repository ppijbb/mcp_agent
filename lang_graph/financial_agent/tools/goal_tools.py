"""
재무 목표 달성 도구

목표 설정, 진행률 추적, 목표별 투자 전략 제안
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SetFinancialGoalInput(BaseModel):
    """재무 목표 설정 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    goal_name: str = Field(description="목표 이름 (예: 집 구매, 은퇴, 교육비)")
    target_amount: float = Field(description="목표 금액")
    deadline: str = Field(description="목표 기한 (ISO format)")
    priority: Optional[str] = Field(default="medium", description="우선순위 (high/medium/low)")


class TrackGoalProgressInput(BaseModel):
    """목표 진행률 추적 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    goal_name: Optional[str] = Field(default=None, description="목표 이름 (없으면 전체)")


class GoalTools:
    """
    재무 목표 달성 도구 모음
    
    목표 설정, 진행률 추적, 목표별 투자 전략 제안
    """
    
    def __init__(self, data_dir: str = "financial_data"):
        """
        GoalTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.goals_file = self.data_dir / "financial_goals.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if self.goals_file.exists():
            with open(self.goals_file, 'r', encoding='utf-8') as f:
                self.goals = json.load(f)
        else:
            self.goals = {}
    
    def _save_data(self):
        """데이터 저장"""
        with open(self.goals_file, 'w', encoding='utf-8') as f:
            json.dump(self.goals, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """Goal 도구 초기화"""
        self.tools.append(self._create_set_financial_goal_tool())
        self.tools.append(self._create_track_goal_progress_tool())
        self.tools.append(self._create_recommend_investment_strategy_tool())
        logger.info(f"Initialized {len(self.tools)} goal tools")
    
    def _create_set_financial_goal_tool(self) -> BaseTool:
        @tool("goal_set_financial_goal", args_schema=SetFinancialGoalInput)
        def set_financial_goal(
            user_id: str,
            goal_name: str,
            target_amount: float,
            deadline: str,
            priority: Optional[str] = "medium"
        ) -> str:
            """
            재무 목표를 설정합니다.
            Args:
                user_id: 사용자 ID
                goal_name: 목표 이름
                target_amount: 목표 금액
                deadline: 목표 기한
                priority: 우선순위
            Returns:
                설정된 목표 정보 (JSON 문자열)
            """
            logger.info(f"Setting financial goal for user {user_id}, goal: {goal_name}")
            
            if user_id not in self.goals:
                self.goals[user_id] = []
            
            deadline_date = datetime.fromisoformat(deadline)
            days_remaining = (deadline_date - datetime.now()).days
            
            if days_remaining <= 0:
                return json.dumps({"error": "목표 기한이 과거입니다"}, ensure_ascii=False)
            
            months_remaining = days_remaining / 30.0
            monthly_savings_needed = target_amount / months_remaining if months_remaining > 0 else target_amount
            
            goal = {
                "user_id": user_id,
                "goal_name": goal_name,
                "target_amount": target_amount,
                "current_amount": 0.0,
                "deadline": deadline,
                "days_remaining": days_remaining,
                "months_remaining": months_remaining,
                "monthly_savings_needed": monthly_savings_needed,
                "priority": priority,
                "progress_percentage": 0.0,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            self.goals[user_id].append(goal)
            self._save_data()
            
            return json.dumps(goal, ensure_ascii=False, indent=2)
        return set_financial_goal
    
    def _create_track_goal_progress_tool(self) -> BaseTool:
        @tool("goal_track_progress", args_schema=TrackGoalProgressInput)
        def track_goal_progress(user_id: str, goal_name: Optional[str] = None) -> str:
            """
            목표 진행률을 추적합니다.
            Args:
                user_id: 사용자 ID
                goal_name: 목표 이름 (없으면 전체)
            Returns:
                목표 진행률 추적 결과 (JSON 문자열)
            """
            logger.info(f"Tracking goal progress for user {user_id}, goal: {goal_name or 'all'}")
            
            if user_id not in self.goals:
                return json.dumps({"error": "No goals found for this user"}, ensure_ascii=False)
            
            user_goals = self.goals[user_id]
            
            if goal_name:
                # 특정 목표만 추적
                goals_to_track = [g for g in user_goals if g["goal_name"] == goal_name]
            else:
                # 전체 목표 추적
                goals_to_track = user_goals
            
            if not goals_to_track:
                return json.dumps({"error": f"No goals found matching '{goal_name}'"}, ensure_ascii=False)
            
            # 진행률 업데이트
            for goal in goals_to_track:
                progress = (goal.get("current_amount", 0.0) / goal.get("target_amount", 1.0)) * 100
                goal["progress_percentage"] = progress
                goal["updated_at"] = datetime.now().isoformat()
                
                # 기한 재계산
                deadline_date = datetime.fromisoformat(goal["deadline"])
                days_remaining = (deadline_date - datetime.now()).days
                goal["days_remaining"] = days_remaining
                goal["months_remaining"] = days_remaining / 30.0
                
                # 필요한 월 저축액 재계산
                remaining_amount = goal["target_amount"] - goal.get("current_amount", 0.0)
                if goal["months_remaining"] > 0:
                    goal["monthly_savings_needed"] = remaining_amount / goal["months_remaining"]
            
            self._save_data()
            
            result = {
                "user_id": user_id,
                "goals": goals_to_track,
                "total_progress": sum(g.get("progress_percentage", 0.0) for g in goals_to_track) / len(goals_to_track) if goals_to_track else 0.0
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return track_goal_progress
    
    def _create_recommend_investment_strategy_tool(self) -> BaseTool:
        @tool("goal_recommend_investment_strategy")
        def recommend_investment_strategy(user_id: str, goal_name: str) -> str:
            """
            목표별 투자 전략을 제안합니다.
            Args:
                user_id: 사용자 ID
                goal_name: 목표 이름
            Returns:
                투자 전략 제안 (JSON 문자열)
            """
            logger.info(f"Recommending investment strategy for user {user_id}, goal: {goal_name}")
            
            if user_id not in self.goals:
                return json.dumps({"error": "No goals found for this user"}, ensure_ascii=False)
            
            goal = None
            for g in self.goals[user_id]:
                if g["goal_name"] == goal_name:
                    goal = g
                    break
            
            if not goal:
                return json.dumps({"error": f"Goal '{goal_name}' not found"}, ensure_ascii=False)
            
            months_remaining = goal.get("months_remaining", 0)
            target_amount = goal.get("target_amount", 0.0)
            current_amount = goal.get("current_amount", 0.0)
            remaining_amount = target_amount - current_amount
            
            # 목표 기간에 따른 투자 전략
            if months_remaining <= 12:
                # 단기 목표: 안정적 투자
                strategy = {
                    "risk_level": "conservative",
                    "recommended_assets": ["예금", "단기 채권", "MMF"],
                    "expected_return": 0.03,
                    "description": "단기 목표이므로 안정적인 자산에 투자하는 것을 권장합니다."
                }
            elif months_remaining <= 60:
                # 중기 목표: 균형 투자
                strategy = {
                    "risk_level": "moderate",
                    "recommended_assets": ["채권", "배당주", "ETF"],
                    "expected_return": 0.06,
                    "description": "중기 목표이므로 균형 잡힌 포트폴리오를 권장합니다."
                }
            else:
                # 장기 목표: 성장 투자
                strategy = {
                    "risk_level": "aggressive",
                    "recommended_assets": ["주식", "ETF", "부동산"],
                    "expected_return": 0.10,
                    "description": "장기 목표이므로 성장 자산에 투자하는 것을 권장합니다."
                }
            
            # 목표 달성을 위한 월 투자액 계산
            if strategy["expected_return"] > 0:
                monthly_investment = remaining_amount / (
                    ((1 + strategy["expected_return"] / 12) ** months_remaining - 1) / (strategy["expected_return"] / 12)
                ) if months_remaining > 0 else remaining_amount / months_remaining
            else:
                monthly_investment = remaining_amount / months_remaining if months_remaining > 0 else 0.0
            
            result = {
                "user_id": user_id,
                "goal_name": goal_name,
                "target_amount": target_amount,
                "current_amount": current_amount,
                "remaining_amount": remaining_amount,
                "months_remaining": months_remaining,
                "investment_strategy": strategy,
                "recommended_monthly_investment": monthly_investment
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return recommend_investment_strategy
    
    def get_tools(self) -> List[BaseTool]:
        """모든 Goal 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 Goal 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

