"""
재무 분석 도구

소비 패턴 분석, 예산 관리, 저축 목표 추적
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AnalyzeSpendingPatternInput(BaseModel):
    """소비 패턴 분석 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    period_days: Optional[int] = Field(default=30, description="분석 기간 (일)")


class CreateBudgetInput(BaseModel):
    """예산 생성 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    monthly_income: float = Field(description="월 소득")
    budget_categories: Dict[str, float] = Field(description="카테고리별 예산 (카테고리명: 금액)")


class TrackSavingsGoalInput(BaseModel):
    """저축 목표 추적 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    goal_name: str = Field(description="목표 이름")
    target_amount: float = Field(description="목표 금액")
    current_amount: float = Field(description="현재 금액")
    deadline: Optional[str] = Field(default=None, description="목표 기한 (ISO format)")


class FinanceTools:
    """
    재무 분석 도구 모음
    
    소비 패턴 분석, 예산 관리, 저축 목표 추적
    """
    
    def __init__(self, data_dir: str = "financial_data"):
        """
        FinanceTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.spending_file = self.data_dir / "spending_data.json"
        self.budget_file = self.data_dir / "budget_data.json"
        self.savings_file = self.data_dir / "savings_goals.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if self.spending_file.exists():
            with open(self.spending_file, 'r', encoding='utf-8') as f:
                self.spending_data = json.load(f)
        else:
            self.spending_data = {}
        
        if self.budget_file.exists():
            with open(self.budget_file, 'r', encoding='utf-8') as f:
                self.budget_data = json.load(f)
        else:
            self.budget_data = {}
        
        if self.savings_file.exists():
            with open(self.savings_file, 'r', encoding='utf-8') as f:
                self.savings_goals = json.load(f)
        else:
            self.savings_goals = {}
    
    def _save_spending(self):
        """소비 데이터 저장"""
        with open(self.spending_file, 'w', encoding='utf-8') as f:
            json.dump(self.spending_data, f, indent=2, ensure_ascii=False)
    
    def _save_budget(self):
        """예산 데이터 저장"""
        with open(self.budget_file, 'w', encoding='utf-8') as f:
            json.dump(self.budget_data, f, indent=2, ensure_ascii=False)
    
    def _save_savings(self):
        """저축 목표 데이터 저장"""
        with open(self.savings_file, 'w', encoding='utf-8') as f:
            json.dump(self.savings_goals, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """Finance 도구 초기화"""
        self.tools.append(self._create_analyze_spending_pattern_tool())
        self.tools.append(self._create_create_budget_tool())
        self.tools.append(self._create_track_savings_goal_tool())
        self.tools.append(self._create_calculate_financial_health_tool())
        logger.info(f"Initialized {len(self.tools)} finance tools")
    
    def _create_analyze_spending_pattern_tool(self) -> BaseTool:
        @tool("finance_analyze_spending", args_schema=AnalyzeSpendingPatternInput)
        def analyze_spending_pattern(user_id: str, period_days: Optional[int] = 30) -> str:
            """
            소비 패턴을 분석합니다.
            Args:
                user_id: 사용자 ID
                period_days: 분석 기간 (일)
            Returns:
                소비 패턴 분석 결과 (JSON 문자열)
            """
            logger.info(f"Analyzing spending pattern for user {user_id}, period: {period_days} days")
            
            if user_id not in self.spending_data:
                return json.dumps({"error": "No spending data found for this user"}, ensure_ascii=False)
            
            cutoff_date = datetime.now() - timedelta(days=period_days)
            user_spending = self.spending_data[user_id]
            
            # 기간 내 소비 데이터 필터링
            recent_spending = [
                record for record in user_spending
                if datetime.fromisoformat(record["date"]) >= cutoff_date
            ]
            
            if not recent_spending:
                return json.dumps({"error": f"No spending data found in the last {period_days} days"}, ensure_ascii=False)
            
            # 카테고리별 소비 분석
            category_totals = {}
            total_spending = 0.0
            
            for record in recent_spending:
                category = record.get("category", "기타")
                amount = record.get("amount", 0.0)
                category_totals[category] = category_totals.get(category, 0.0) + amount
                total_spending += amount
            
            # 평균 일일 소비
            avg_daily_spending = total_spending / period_days
            
            result = {
                "user_id": user_id,
                "period_days": period_days,
                "total_spending": total_spending,
                "avg_daily_spending": avg_daily_spending,
                "category_breakdown": category_totals,
                "top_categories": sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:5],
                "transaction_count": len(recent_spending)
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return analyze_spending_pattern
    
    def _create_create_budget_tool(self) -> BaseTool:
        @tool("finance_create_budget", args_schema=CreateBudgetInput)
        def create_budget(
            user_id: str,
            monthly_income: float,
            budget_categories: Dict[str, float]
        ) -> str:
            """
            예산을 생성합니다.
            Args:
                user_id: 사용자 ID
                monthly_income: 월 소득
                budget_categories: 카테고리별 예산
            Returns:
                생성된 예산 정보 (JSON 문자열)
            """
            logger.info(f"Creating budget for user {user_id}")
            
            total_budget = sum(budget_categories.values())
            
            if total_budget > monthly_income:
                return json.dumps({"error": f"총 예산 ({total_budget})이 월 소득 ({monthly_income})을 초과합니다"}, ensure_ascii=False)
            
            budget = {
                "user_id": user_id,
                "monthly_income": monthly_income,
                "budget_categories": budget_categories,
                "total_budget": total_budget,
                "remaining_income": monthly_income - total_budget,
                "created_at": datetime.now().isoformat()
            }
            
            self.budget_data[user_id] = budget
            self._save_budget()
            
            return json.dumps(budget, ensure_ascii=False, indent=2)
        return create_budget
    
    def _create_track_savings_goal_tool(self) -> BaseTool:
        @tool("finance_track_savings_goal", args_schema=TrackSavingsGoalInput)
        def track_savings_goal(
            user_id: str,
            goal_name: str,
            target_amount: float,
            current_amount: float,
            deadline: Optional[str] = None
        ) -> str:
            """
            저축 목표를 추적합니다.
            Args:
                user_id: 사용자 ID
                goal_name: 목표 이름
                target_amount: 목표 금액
                current_amount: 현재 금액
                deadline: 목표 기한
            Returns:
                저축 목표 추적 결과 (JSON 문자열)
            """
            logger.info(f"Tracking savings goal for user {user_id}, goal: {goal_name}")
            
            if user_id not in self.savings_goals:
                self.savings_goals[user_id] = []
            
            # 기존 목표 찾기 또는 새로 생성
            existing_goal = None
            for goal in self.savings_goals[user_id]:
                if goal["goal_name"] == goal_name:
                    existing_goal = goal
                    break
            
            progress_percentage = (current_amount / target_amount * 100) if target_amount > 0 else 0.0
            remaining_amount = target_amount - current_amount
            
            # 기한이 있으면 남은 일수 계산
            days_remaining = None
            if deadline:
                deadline_date = datetime.fromisoformat(deadline)
                days_remaining = (deadline_date - datetime.now()).days
            
            # 필요한 월 저축액 계산
            monthly_savings_needed = None
            if days_remaining and days_remaining > 0:
                months_remaining = days_remaining / 30.0
                monthly_savings_needed = remaining_amount / months_remaining if months_remaining > 0 else None
            
            goal_data = {
                "user_id": user_id,
                "goal_name": goal_name,
                "target_amount": target_amount,
                "current_amount": current_amount,
                "remaining_amount": remaining_amount,
                "progress_percentage": progress_percentage,
                "deadline": deadline,
                "days_remaining": days_remaining,
                "monthly_savings_needed": monthly_savings_needed,
                "updated_at": datetime.now().isoformat()
            }
            
            if existing_goal:
                # 기존 목표 업데이트
                idx = self.savings_goals[user_id].index(existing_goal)
                self.savings_goals[user_id][idx] = goal_data
            else:
                # 새 목표 추가
                goal_data["created_at"] = datetime.now().isoformat()
                self.savings_goals[user_id].append(goal_data)
            
            self._save_savings()
            
            return json.dumps(goal_data, ensure_ascii=False, indent=2)
        return track_savings_goal
    
    def _create_calculate_financial_health_tool(self) -> BaseTool:
        @tool("finance_calculate_health")
        def calculate_financial_health(user_id: str) -> str:
            """
            재무 건강 점수를 계산합니다.
            Args:
                user_id: 사용자 ID
            Returns:
                재무 건강 점수 및 분석 (JSON 문자열)
            """
            logger.info(f"Calculating financial health for user {user_id}")
            
            # 예산 데이터 확인
            budget = self.budget_data.get(user_id, {})
            monthly_income = budget.get("monthly_income", 0.0)
            
            # 소비 패턴 분석
            spending_analysis = {}
            if user_id in self.spending_data:
                recent_spending = [
                    record for record in self.spending_data[user_id]
                    if datetime.fromisoformat(record["date"]) >= datetime.now() - timedelta(days=30)
                ]
                total_spending = sum(record.get("amount", 0.0) for record in recent_spending)
                spending_analysis = {
                    "total_spending": total_spending,
                    "spending_rate": (total_spending / monthly_income * 100) if monthly_income > 0 else 0.0
                }
            
            # 저축 목표 진행률
            savings_progress = 0.0
            if user_id in self.savings_goals:
                goals = self.savings_goals[user_id]
                if goals:
                    avg_progress = sum(goal.get("progress_percentage", 0.0) for goal in goals) / len(goals)
                    savings_progress = avg_progress
            
            # 재무 건강 점수 계산 (0-100)
            health_score = 0.0
            
            # 소비율 점수 (40점 만점)
            spending_rate = spending_analysis.get("spending_rate", 0.0)
            if spending_rate <= 50:
                health_score += 40
            elif spending_rate <= 70:
                health_score += 30
            elif spending_rate <= 90:
                health_score += 20
            else:
                health_score += 10
            
            # 저축 목표 진행률 점수 (30점 만점)
            if savings_progress >= 80:
                health_score += 30
            elif savings_progress >= 50:
                health_score += 20
            elif savings_progress >= 30:
                health_score += 10
            
            # 예산 계획 점수 (30점 만점)
            if budget:
                health_score += 30
            
            result = {
                "user_id": user_id,
                "health_score": health_score,
                "spending_analysis": spending_analysis,
                "savings_progress": savings_progress,
                "has_budget": bool(budget),
                "recommendations": []
            }
            
            # 권장사항 생성
            if spending_rate > 90:
                result["recommendations"].append("소비율이 높습니다. 지출을 줄이는 것을 고려하세요.")
            if savings_progress < 30:
                result["recommendations"].append("저축 목표 진행이 느립니다. 월 저축액을 늘리는 것을 고려하세요.")
            if not budget:
                result["recommendations"].append("예산 계획을 수립하는 것을 권장합니다.")
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return calculate_financial_health
    
    def get_tools(self) -> List[BaseTool]:
        """모든 Finance 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 Finance 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

