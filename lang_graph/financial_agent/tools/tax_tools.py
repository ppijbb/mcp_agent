"""
세금 최적화 도구

공제 항목 발견, 세금 계산, 세금 신고 자동화 준비
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FindDeductionsInput(BaseModel):
    """공제 항목 발견 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    income: float = Field(description="연 소득")
    expenses: Optional[Dict[str, float]] = Field(default=None, description="지출 내역 (카테고리: 금액)")


class CalculateTaxInput(BaseModel):
    """세금 계산 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    income: float = Field(description="연 소득")
    deductions: Optional[Dict[str, float]] = Field(default=None, description="공제 항목 (항목명: 금액)")
    tax_year: Optional[int] = Field(default=None, description="세금 연도 (기본: 현재 연도)")


class TaxTools:
    """
    세금 최적화 도구 모음
    
    공제 항목 발견, 세금 계산, 세금 신고 자동화 준비
    """
    
    def __init__(self, data_dir: str = "financial_data"):
        """
        TaxTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tax_data_file = self.data_dir / "tax_data.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if self.tax_data_file.exists():
            with open(self.tax_data_file, 'r', encoding='utf-8') as f:
                self.tax_data = json.load(f)
        else:
            self.tax_data = {}
    
    def _save_data(self):
        """데이터 저장"""
        with open(self.tax_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.tax_data, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """Tax 도구 초기화"""
        self.tools.append(self._create_find_deductions_tool())
        self.tools.append(self._create_calculate_tax_tool())
        self.tools.append(self._create_optimize_tax_strategy_tool())
        logger.info(f"Initialized {len(self.tools)} tax tools")
    
    def _create_find_deductions_tool(self) -> BaseTool:
        @tool("tax_find_deductions", args_schema=FindDeductionsInput)
        def find_deductions(
            user_id: str,
            income: float,
            expenses: Optional[Dict[str, float]] = None
        ) -> str:
            """
            공제 항목을 발견합니다.
            Args:
                user_id: 사용자 ID
                income: 연 소득
                expenses: 지출 내역
            Returns:
                발견된 공제 항목 목록 (JSON 문자열)
            """
            logger.info(f"Finding tax deductions for user {user_id}")
            
            # 일반적인 공제 항목 (한국 세법 기준)
            common_deductions = {
                "의료비": 0.0,
                "교육비": 0.0,
                "기부금": 0.0,
                "주택자금": 0.0,
                "연금보험료": 0.0,
                "보험료": 0.0,
                "신용카드": 0.0,
                "현금영수증": 0.0,
            }
            
            # 지출 내역에서 공제 가능 항목 추출
            if expenses:
                for category, amount in expenses.items():
                    category_lower = category.lower()
                    if "의료" in category_lower or "병원" in category_lower:
                        common_deductions["의료비"] += amount
                    elif "교육" in category_lower or "학원" in category_lower:
                        common_deductions["교육비"] += amount
                    elif "기부" in category_lower:
                        common_deductions["기부금"] += amount
                    elif "주택" in category_lower or "부동산" in category_lower:
                        common_deductions["주택자금"] += amount
                    elif "연금" in category_lower:
                        common_deductions["연금보험료"] += amount
                    elif "보험" in category_lower:
                        common_deductions["보험료"] += amount
                    elif "신용카드" in category_lower or "카드" in category_lower:
                        common_deductions["신용카드"] += amount
                    elif "현금영수증" in category_lower:
                        common_deductions["현금영수증"] += amount
            
            # 0인 항목 제거
            available_deductions = {k: v for k, v in common_deductions.items() if v > 0}
            total_deductions = sum(available_deductions.values())
            
            result = {
                "user_id": user_id,
                "income": income,
                "available_deductions": available_deductions,
                "total_deductions": total_deductions,
                "taxable_income": income - total_deductions,
                "estimated_tax_savings": total_deductions * 0.15  # 대략적인 세액 공제 (15% 가정)
            }
            
            # 데이터 저장
            if user_id not in self.tax_data:
                self.tax_data[user_id] = {}
            self.tax_data[user_id]["deductions"] = result
            self._save_data()
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return find_deductions
    
    def _create_calculate_tax_tool(self) -> BaseTool:
        @tool("tax_calculate", args_schema=CalculateTaxInput)
        def calculate_tax(
            user_id: str,
            income: float,
            deductions: Optional[Dict[str, float]] = None,
            tax_year: Optional[int] = None
        ) -> str:
            """
            세금을 계산합니다.
            Args:
                user_id: 사용자 ID
                income: 연 소득
                deductions: 공제 항목
                tax_year: 세금 연도
            Returns:
                세금 계산 결과 (JSON 문자열)
            """
            logger.info(f"Calculating tax for user {user_id}")
            
            if tax_year is None:
                tax_year = datetime.now().year
            
            # 공제 금액 계산
            total_deductions = sum(deductions.values()) if deductions else 0.0
            
            # 과세 표준 계산
            taxable_income = max(0, income - total_deductions)
            
            # 간이 세액표 기반 세금 계산 (한국 소득세 구간별)
            tax_amount = 0.0
            if taxable_income <= 12000000:
                tax_amount = taxable_income * 0.06
            elif taxable_income <= 46000000:
                tax_amount = 720000 + (taxable_income - 12000000) * 0.15
            elif taxable_income <= 88000000:
                tax_amount = 5820000 + (taxable_income - 46000000) * 0.24
            elif taxable_income <= 150000000:
                tax_amount = 15900000 + (taxable_income - 88000000) * 0.35
            else:
                tax_amount = 37600000 + (taxable_income - 150000000) * 0.38
            
            # 지방소득세 (소득세의 10%)
            local_tax = tax_amount * 0.1
            total_tax = tax_amount + local_tax
            
            result = {
                "user_id": user_id,
                "tax_year": tax_year,
                "income": income,
                "total_deductions": total_deductions,
                "taxable_income": taxable_income,
                "income_tax": tax_amount,
                "local_tax": local_tax,
                "total_tax": total_tax,
                "effective_tax_rate": (total_tax / income * 100) if income > 0 else 0.0,
                "after_tax_income": income - total_tax
            }
            
            # 데이터 저장
            if user_id not in self.tax_data:
                self.tax_data[user_id] = {}
            self.tax_data[user_id]["tax_calculation"] = result
            self._save_data()
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return calculate_tax
    
    def _create_optimize_tax_strategy_tool(self) -> BaseTool:
        @tool("tax_optimize_strategy")
        def optimize_tax_strategy(user_id: str) -> str:
            """
            세금 최적화 전략을 제안합니다.
            Args:
                user_id: 사용자 ID
            Returns:
                세금 최적화 전략 (JSON 문자열)
            """
            logger.info(f"Optimizing tax strategy for user {user_id}")
            
            if user_id not in self.tax_data:
                return json.dumps({"error": "No tax data found for this user"}, ensure_ascii=False)
            
            user_tax_data = self.tax_data[user_id]
            deductions = user_tax_data.get("deductions", {})
            tax_calc = user_tax_data.get("tax_calculation", {})
            
            strategies = []
            
            # 공제 항목 활용 전략
            if deductions.get("total_deductions", 0) < deductions.get("income", 0) * 0.1:
                strategies.append({
                    "strategy": "공제 항목 확대",
                    "description": "현재 공제 항목이 적습니다. 의료비, 교육비, 기부금 등을 늘려 세액을 절감할 수 있습니다.",
                    "potential_savings": deductions.get("income", 0) * 0.05
                })
            
            # 연금보험료 공제
            if deductions.get("available_deductions", {}).get("연금보험료", 0) == 0:
                strategies.append({
                    "strategy": "연금보험료 납입",
                    "description": "연금보험료 납입 시 소득공제를 받을 수 있습니다.",
                    "potential_savings": deductions.get("income", 0) * 0.02
                })
            
            # 신용카드 사용 확대
            if deductions.get("available_deductions", {}).get("신용카드", 0) < deductions.get("income", 0) * 0.25:
                strategies.append({
                    "strategy": "신용카드 사용 확대",
                    "description": "신용카드 사용액의 일정 비율을 소득공제 받을 수 있습니다.",
                    "potential_savings": deductions.get("income", 0) * 0.01
                })
            
            result = {
                "user_id": user_id,
                "current_tax": tax_calc.get("total_tax", 0),
                "optimization_strategies": strategies,
                "total_potential_savings": sum(s.get("potential_savings", 0) for s in strategies)
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return optimize_tax_strategy
    
    def get_tools(self) -> List[BaseTool]:
        """모든 Tax 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 Tax 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

