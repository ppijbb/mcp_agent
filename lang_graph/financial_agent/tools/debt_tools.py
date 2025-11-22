"""
부채 관리 도구

대출 상환 전략, 이자 최소화 계획, 부채 구조 분석
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AnalyzeDebtInput(BaseModel):
    """부채 분석 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    loans: List[Dict[str, Any]] = Field(description="대출 목록 (각 대출은 principal, interest_rate, remaining_months 포함)")


class CreateRepaymentStrategyInput(BaseModel):
    """상환 전략 생성 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    monthly_payment_capacity: float = Field(description="월 상환 가능 금액")
    strategy_type: Optional[str] = Field(default="snowball", description="전략 유형 (snowball/avalanche)")


class DebtTools:
    """
    부채 관리 도구 모음
    
    대출 상환 전략, 이자 최소화 계획, 부채 구조 분석
    """
    
    def __init__(self, data_dir: str = "financial_data"):
        """
        DebtTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.debt_data_file = self.data_dir / "debt_data.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if self.debt_data_file.exists():
            with open(self.debt_data_file, 'r', encoding='utf-8') as f:
                self.debt_data = json.load(f)
        else:
            self.debt_data = {}
    
    def _save_data(self):
        """데이터 저장"""
        with open(self.debt_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.debt_data, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """Debt 도구 초기화"""
        self.tools.append(self._create_analyze_debt_tool())
        self.tools.append(self._create_create_repayment_strategy_tool())
        self.tools.append(self._create_minimize_interest_tool())
        logger.info(f"Initialized {len(self.tools)} debt tools")
    
    def _create_analyze_debt_tool(self) -> BaseTool:
        @tool("debt_analyze", args_schema=AnalyzeDebtInput)
        def analyze_debt(user_id: str, loans: List[Dict[str, Any]]) -> str:
            """
            부채 구조를 분석합니다.
            Args:
                user_id: 사용자 ID
                loans: 대출 목록
            Returns:
                부채 분석 결과 (JSON 문자열)
            """
            logger.info(f"Analyzing debt for user {user_id}")
            
            total_principal = sum(loan.get("principal", 0.0) for loan in loans)
            total_monthly_payment = sum(loan.get("monthly_payment", 0.0) for loan in loans)
            
            # 가중 평균 이자율 계산
            weighted_interest = 0.0
            for loan in loans:
                principal = loan.get("principal", 0.0)
                interest_rate = loan.get("interest_rate", 0.0)
                if total_principal > 0:
                    weighted_interest += (principal / total_principal) * interest_rate
            
            # 총 이자 계산 (간단한 추정)
            total_interest = 0.0
            for loan in loans:
                principal = loan.get("principal", 0.0)
                interest_rate = loan.get("interest_rate", 0.0) / 100.0
                remaining_months = loan.get("remaining_months", 0)
                # 단순 이자 계산 (복리 고려 안 함)
                total_interest += principal * interest_rate * (remaining_months / 12.0)
            
            result = {
                "user_id": user_id,
                "total_principal": total_principal,
                "total_monthly_payment": total_monthly_payment,
                "weighted_interest_rate": weighted_interest,
                "total_interest": total_interest,
                "total_debt": total_principal + total_interest,
                "loan_count": len(loans),
                "loans": loans,
                "debt_to_income_ratio": None  # 소득 정보가 없으면 None
            }
            
            # 데이터 저장
            self.debt_data[user_id] = result
            self._save_data()
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return analyze_debt
    
    def _create_create_repayment_strategy_tool(self) -> BaseTool:
        @tool("debt_create_repayment_strategy", args_schema=CreateRepaymentStrategyInput)
        def create_repayment_strategy(
            user_id: str,
            monthly_payment_capacity: float,
            strategy_type: Optional[str] = "snowball"
        ) -> str:
            """
            대출 상환 전략을 생성합니다.
            Args:
                user_id: 사용자 ID
                monthly_payment_capacity: 월 상환 가능 금액
                strategy_type: 전략 유형 (snowball: 작은 금액부터, avalanche: 높은 이자율부터)
            Returns:
                상환 전략 (JSON 문자열)
            """
            logger.info(f"Creating repayment strategy for user {user_id}, type: {strategy_type}")
            
            if user_id not in self.debt_data:
                return json.dumps({"error": "No debt data found for this user"}, ensure_ascii=False)
            
            loans = self.debt_data[user_id].get("loans", [])
            if not loans:
                return json.dumps({"error": "No loans found"}, ensure_ascii=False)
            
            # 전략에 따라 대출 정렬
            if strategy_type == "snowball":
                # 작은 잔액부터
                sorted_loans = sorted(loans, key=lambda x: x.get("principal", 0.0))
            elif strategy_type == "avalanche":
                # 높은 이자율부터
                sorted_loans = sorted(loans, key=lambda x: x.get("interest_rate", 0.0), reverse=True)
            else:
                sorted_loans = loans
            
            repayment_plan = []
            remaining_capacity = monthly_payment_capacity
            
            for loan in sorted_loans:
                principal = loan.get("principal", 0.0)
                interest_rate = loan.get("interest_rate", 0.0) / 100.0
                current_monthly = loan.get("monthly_payment", 0.0)
                
                # 추가 상환 가능 금액 계산
                extra_payment = max(0, remaining_capacity - current_monthly)
                
                # 상환 기간 계산 (간단한 추정)
                if extra_payment > 0:
                    # 추가 상환 시 기간 단축
                    months_to_payoff = principal / (current_monthly + extra_payment)
                else:
                    months_to_payoff = loan.get("remaining_months", 0)
                
                repayment_plan.append({
                    "loan_id": loan.get("loan_id", f"loan_{len(repayment_plan)}"),
                    "principal": principal,
                    "interest_rate": interest_rate * 100,
                    "current_monthly_payment": current_monthly,
                    "recommended_extra_payment": extra_payment,
                    "new_monthly_payment": current_monthly + extra_payment,
                    "months_to_payoff": months_to_payoff,
                    "total_interest_saved": extra_payment * months_to_payoff * interest_rate / 12.0
                })
                
                remaining_capacity -= (current_monthly + extra_payment)
                if remaining_capacity <= 0:
                    break
            
            total_interest_saved = sum(plan.get("total_interest_saved", 0.0) for plan in repayment_plan)
            total_months_saved = sum(
                loan.get("remaining_months", 0) - plan.get("months_to_payoff", 0)
                for loan, plan in zip(loans, repayment_plan)
            )
            
            result = {
                "user_id": user_id,
                "strategy_type": strategy_type,
                "monthly_payment_capacity": monthly_payment_capacity,
                "repayment_plan": repayment_plan,
                "total_interest_saved": total_interest_saved,
                "total_months_saved": total_months_saved
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return create_repayment_strategy
    
    def _create_minimize_interest_tool(self) -> BaseTool:
        @tool("debt_minimize_interest")
        def minimize_interest(user_id: str) -> str:
            """
            이자를 최소화하는 계획을 제안합니다.
            Args:
                user_id: 사용자 ID
            Returns:
                이자 최소화 계획 (JSON 문자열)
            """
            logger.info(f"Minimizing interest for user {user_id}")
            
            if user_id not in self.debt_data:
                return json.dumps({"error": "No debt data found for this user"}, ensure_ascii=False)
            
            loans = self.debt_data[user_id].get("loans", [])
            if not loans:
                return json.dumps({"error": "No loans found"}, ensure_ascii=False)
            
            # 높은 이자율 대출부터 상환하는 것이 이자 최소화
            sorted_by_interest = sorted(loans, key=lambda x: x.get("interest_rate", 0.0), reverse=True)
            
            recommendations = []
            for loan in sorted_by_interest:
                principal = loan.get("principal", 0.0)
                interest_rate = loan.get("interest_rate", 0.0)
                
                recommendations.append({
                    "loan_id": loan.get("loan_id", "unknown"),
                    "principal": principal,
                    "interest_rate": interest_rate,
                    "priority": "high" if interest_rate > 10 else "medium" if interest_rate > 5 else "low",
                    "recommendation": f"이자율 {interest_rate}%로 높은 편입니다. 우선 상환을 권장합니다."
                })
            
            result = {
                "user_id": user_id,
                "minimization_strategy": "avalanche",
                "recommendations": recommendations,
                "expected_interest_savings": "높은 이자율 대출부터 상환 시 총 이자 최소화"
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return minimize_interest
    
    def get_tools(self) -> List[BaseTool]:
        """모든 Debt 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 Debt 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

