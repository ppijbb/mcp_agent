"""
투자 수익률 분석 도구

ROI 계산, 캐시플로우 분석, 자금 회수 기간, 세금 최적화, 대출 활용 분석
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CalculateROIInput(BaseModel):
    """ROI 계산 입력 스키마"""
    purchase_price: float = Field(description="매입 가격 (만원)")
    monthly_rent: Optional[float] = Field(default=None, description="월 임대료 (만원)")
    expected_sale_price: Optional[float] = Field(default=None, description="예상 매도 가격 (만원)")
    holding_period_years: Optional[int] = Field(default=5, description="보유 기간 (년)")
    maintenance_cost: Optional[float] = Field(default=0.0, description="월 유지보수비 (만원)")
    property_tax: Optional[float] = Field(default=0.0, description="연 재산세 (만원)")


class CalculateCashFlowInput(BaseModel):
    """캐시플로우 계산 입력 스키마"""
    monthly_rent: float = Field(description="월 임대료 (만원)")
    monthly_expenses: float = Field(description="월 지출 (만원)")
    loan_amount: Optional[float] = Field(default=None, description="대출 금액 (만원)")
    loan_interest_rate: Optional[float] = Field(default=None, description="대출 이자율 (%)")
    loan_term_years: Optional[int] = Field(default=20, description="대출 기간 (년)")


class ROITools:
    """
    투자 수익률 분석 도구 모음
    
    ROI 계산, 캐시플로우 분석, 자금 회수 기간, 세금 최적화, 대출 활용 분석
    """
    
    def __init__(self, data_dir: str = "real_estate_data"):
        """
        ROITools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.roi_data_file = self.data_dir / "roi_data.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if self.roi_data_file.exists():
            with open(self.roi_data_file, 'r', encoding='utf-8') as f:
                self.roi_data = json.load(f)
        else:
            self.roi_data = {}
    
    def _save_data(self):
        """데이터 저장"""
        with open(self.roi_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.roi_data, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """ROI 도구 초기화"""
        self.tools.append(self._create_calculate_roi_tool())
        self.tools.append(self._create_calculate_cash_flow_tool())
        self.tools.append(self._create_calculate_payback_period_tool())
        self.tools.append(self._create_optimize_tax_tool())
        self.tools.append(self._create_analyze_leverage_tool())
        logger.info(f"Initialized {len(self.tools)} ROI tools")
    
    def _create_calculate_roi_tool(self) -> BaseTool:
        @tool("roi_calculate", args_schema=CalculateROIInput)
        def calculate_roi(
            purchase_price: float,
            monthly_rent: Optional[float] = None,
            expected_sale_price: Optional[float] = None,
            holding_period_years: Optional[int] = 5,
            maintenance_cost: Optional[float] = 0.0,
            property_tax: Optional[float] = 0.0
        ) -> str:
            """
            투자 수익률(ROI)을 계산합니다.
            Args:
                purchase_price: 매입 가격 (만원)
                monthly_rent: 월 임대료 (만원)
                expected_sale_price: 예상 매도 가격 (만원)
                holding_period_years: 보유 기간 (년)
                maintenance_cost: 월 유지보수비 (만원)
                property_tax: 연 재산세 (만원)
            Returns:
                ROI 계산 결과 (JSON 문자열)
            """
            logger.info(f"Calculating ROI for purchase price: {purchase_price}만원")
            
            # 임대 수익 계산
            rental_income = 0.0
            if monthly_rent:
                annual_rent = monthly_rent * 12
                annual_expenses = (maintenance_cost * 12) + property_tax
                net_annual_rent = annual_rent - annual_expenses
                rental_income = net_annual_rent * holding_period_years
            
            # 자본이득 계산
            capital_gain = 0.0
            if expected_sale_price:
                capital_gain = expected_sale_price - purchase_price
            
            # 총 수익
            total_return = rental_income + capital_gain
            
            # ROI 계산
            roi = (total_return / purchase_price) * 100 if purchase_price > 0 else 0.0
            annualized_roi = roi / holding_period_years if holding_period_years > 0 else 0.0
            
            # 임대 수익률 (Yield)
            yield_rate = (monthly_rent * 12 / purchase_price * 100) if monthly_rent and purchase_price > 0 else 0.0
            
            result = {
                "purchase_price": purchase_price,
                "holding_period_years": holding_period_years,
                "rental_income": rental_income,
                "capital_gain": capital_gain,
                "total_return": total_return,
                "roi_percentage": roi,
                "annualized_roi": annualized_roi,
                "yield_rate": yield_rate,
                "break_even_years": purchase_price / (monthly_rent * 12) if monthly_rent and monthly_rent > 0 else None
            }
            
            # 데이터 저장
            calculation_id = f"roi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.roi_data[calculation_id] = result
            self._save_data()
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return calculate_roi
    
    def _create_calculate_cash_flow_tool(self) -> BaseTool:
        @tool("roi_calculate_cash_flow", args_schema=CalculateCashFlowInput)
        def calculate_cash_flow(
            monthly_rent: float,
            monthly_expenses: float,
            loan_amount: Optional[float] = None,
            loan_interest_rate: Optional[float] = None,
            loan_term_years: Optional[int] = 20
        ) -> str:
            """
            캐시플로우를 계산합니다.
            Args:
                monthly_rent: 월 임대료 (만원)
                monthly_expenses: 월 지출 (만원)
                loan_amount: 대출 금액 (만원)
                loan_interest_rate: 대출 이자율 (%)
                loan_term_years: 대출 기간 (년)
            Returns:
                캐시플로우 계산 결과 (JSON 문자열)
            """
            logger.info(f"Calculating cash flow for monthly rent: {monthly_rent}만원")
            
            # 월 대출 상환액 계산 (원리금 균등분할)
            monthly_loan_payment = 0.0
            if loan_amount and loan_interest_rate:
                monthly_rate = (loan_interest_rate / 100.0) / 12.0
                num_payments = loan_term_years * 12
                if monthly_rate > 0:
                    monthly_loan_payment = loan_amount * (
                        monthly_rate * (1 + monthly_rate) ** num_payments
                    ) / ((1 + monthly_rate) ** num_payments - 1)
                else:
                    monthly_loan_payment = loan_amount / num_payments
            
            # 월 순 현금 흐름
            monthly_cash_flow = monthly_rent - monthly_expenses - monthly_loan_payment
            
            # 연간 현금 흐름
            annual_cash_flow = monthly_cash_flow * 12
            
            # 손익분기점 (월 임대료)
            break_even_rent = monthly_expenses + monthly_loan_payment
            
            result = {
                "monthly_rent": monthly_rent,
                "monthly_expenses": monthly_expenses,
                "monthly_loan_payment": monthly_loan_payment,
                "monthly_cash_flow": monthly_cash_flow,
                "annual_cash_flow": annual_cash_flow,
                "break_even_rent": break_even_rent,
                "cash_flow_positive": monthly_cash_flow > 0,
                "cash_flow_margin": (monthly_cash_flow / monthly_rent * 100) if monthly_rent > 0 else 0.0
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return calculate_cash_flow
    
    def _create_calculate_payback_period_tool(self) -> BaseTool:
        @tool("roi_calculate_payback_period")
        def calculate_payback_period(
            purchase_price: float,
            annual_cash_flow: float
        ) -> str:
            """
            자금 회수 기간을 계산합니다.
            Args:
                purchase_price: 매입 가격 (만원)
                annual_cash_flow: 연간 현금 흐름 (만원)
            Returns:
                자금 회수 기간 계산 결과 (JSON 문자열)
            """
            logger.info(f"Calculating payback period for purchase price: {purchase_price}만원")
            
            if annual_cash_flow <= 0:
                return json.dumps({
                    "error": "연간 현금 흐름이 0 이하입니다. 자금 회수가 불가능합니다.",
                    "purchase_price": purchase_price,
                    "annual_cash_flow": annual_cash_flow
                }, ensure_ascii=False)
            
            payback_period_years = purchase_price / annual_cash_flow
            payback_period_months = payback_period_years * 12
            
            result = {
                "purchase_price": purchase_price,
                "annual_cash_flow": annual_cash_flow,
                "payback_period_years": payback_period_years,
                "payback_period_months": payback_period_months,
                "payback_efficiency": "excellent" if payback_period_years < 10 else "good" if payback_period_years < 20 else "poor"
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return calculate_payback_period
    
    def _create_optimize_tax_tool(self) -> BaseTool:
        @tool("roi_optimize_tax")
        def optimize_tax(
            purchase_price: float,
            expected_sale_price: float,
            holding_period_years: int,
            is_first_home: bool = False
        ) -> str:
            """
            부동산 세금을 최적화합니다.
            Args:
                purchase_price: 매입 가격 (만원)
                expected_sale_price: 예상 매도 가격 (만원)
                holding_period_years: 보유 기간 (년)
                is_first_home: 자가주택 여부
            Returns:
                세금 최적화 결과 (JSON 문자열)
            """
            logger.info(f"Optimizing tax for purchase: {purchase_price}만원, sale: {expected_sale_price}만원")
            
            capital_gain = expected_sale_price - purchase_price
            
            # 양도소득세 계산 (간단한 버전)
            # 실제로는 복잡한 세법 적용 필요
            if is_first_home and holding_period_years >= 2:
                # 자가주택 2년 이상 보유 시 비과세
                transfer_tax = 0.0
            elif holding_period_years >= 2:
                # 장기 보유 시 세율 감면
                transfer_tax = capital_gain * 0.06  # 6% (간소화)
            else:
                # 단기 보유
                transfer_tax = capital_gain * 0.11  # 11% (간소화)
            
            # 종부세 (간소화)
            comprehensive_real_estate_tax = purchase_price * 0.001  # 0.1% (간소화)
            
            total_tax = transfer_tax + comprehensive_real_estate_tax
            after_tax_profit = capital_gain - total_tax
            
            result = {
                "purchase_price": purchase_price,
                "expected_sale_price": expected_sale_price,
                "capital_gain": capital_gain,
                "transfer_tax": transfer_tax,
                "comprehensive_real_estate_tax": comprehensive_real_estate_tax,
                "total_tax": total_tax,
                "after_tax_profit": after_tax_profit,
                "tax_rate": (total_tax / capital_gain * 100) if capital_gain > 0 else 0.0,
                "optimization_strategies": self._generate_tax_strategies(holding_period_years, is_first_home)
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return optimize_tax
    
    def _create_analyze_leverage_tool(self) -> BaseTool:
        @tool("roi_analyze_leverage")
        def analyze_leverage(
            purchase_price: float,
            down_payment: float,
            loan_interest_rate: float,
            expected_annual_return: float
        ) -> str:
            """
            대출 활용(레버리지) 효과를 분석합니다.
            Args:
                purchase_price: 매입 가격 (만원)
                down_payment: 계약금 (만원)
                loan_interest_rate: 대출 이자율 (%)
                expected_annual_return: 예상 연 수익률 (%)
            Returns:
                레버리지 분석 결과 (JSON 문자열)
            """
            logger.info(f"Analyzing leverage for purchase: {purchase_price}만원, down: {down_payment}만원")
            
            loan_amount = purchase_price - down_payment
            leverage_ratio = purchase_price / down_payment if down_payment > 0 else 1.0
            
            # 레버리지 없는 경우 수익률
            unleveraged_return = expected_annual_return
            
            # 레버리지 있는 경우 수익률
            # (총 수익 - 대출 이자) / 자본금
            annual_interest = loan_amount * (loan_interest_rate / 100.0)
            total_return = purchase_price * (expected_annual_return / 100.0)
            net_return = total_return - annual_interest
            leveraged_return = (net_return / down_payment * 100) if down_payment > 0 else 0.0
            
            # 레버리지 효과
            leverage_effect = leveraged_return - unleveraged_return
            
            result = {
                "purchase_price": purchase_price,
                "down_payment": down_payment,
                "loan_amount": loan_amount,
                "leverage_ratio": leverage_ratio,
                "loan_interest_rate": loan_interest_rate,
                "expected_annual_return": expected_annual_return,
                "unleveraged_return": unleveraged_return,
                "leveraged_return": leveraged_return,
                "leverage_effect": leverage_effect,
                "is_leverage_beneficial": leverage_effect > 0,
                "recommendations": self._generate_leverage_recommendations(leverage_effect, loan_interest_rate, expected_annual_return)
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return analyze_leverage
    
    def _generate_tax_strategies(self, holding_period: int, is_first_home: bool) -> List[str]:
        """세금 최적화 전략 생성"""
        strategies = []
        
        if holding_period < 2:
            strategies.append("2년 이상 보유 시 양도소득세 감면 혜택을 받을 수 있습니다.")
        
        if is_first_home:
            strategies.append("자가주택으로 2년 이상 보유 시 양도소득세 비과세 혜택이 있습니다.")
        else:
            strategies.append("장기 보유를 통해 세금 부담을 줄일 수 있습니다.")
        
        return strategies
    
    def _generate_leverage_recommendations(self, leverage_effect: float, interest_rate: float, expected_return: float) -> List[str]:
        """레버리지 권장사항 생성"""
        recommendations = []
        
        if leverage_effect > 0:
            recommendations.append("레버리지 활용이 유리합니다. 대출을 활용하는 것을 권장합니다.")
        else:
            recommendations.append("레버리지 활용이 불리합니다. 현금 매입을 고려하세요.")
        
        if interest_rate > expected_return:
            recommendations.append("대출 이자율이 예상 수익률보다 높습니다. 신중한 검토가 필요합니다.")
        
        return recommendations
    
    def get_tools(self) -> List[BaseTool]:
        """모든 ROI 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 ROI 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

