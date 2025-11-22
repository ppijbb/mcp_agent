"""
수수료 계산 도구

거래 수수료 계산, 제휴 수수료 계산, 수익 추적
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CalculateCommissionInput(BaseModel):
    """수수료 계산 입력 스키마"""
    transaction_amount: float = Field(description="거래 금액")
    commission_rate: Optional[float] = Field(default=None, description="수수료율 (기본값 사용 시 None)")


class CalculateAffiliateCommissionInput(BaseModel):
    """제휴 수수료 계산 입력 스키마"""
    product_type: str = Field(description="상품 유형 (credit_card/loan/insurance/real_estate)")
    transaction_amount: float = Field(description="거래 금액")
    affiliate_rate: Optional[float] = Field(default=None, description="제휴 수수료율 (기본값 사용 시 None)")


class CommissionTools:
    """
    수수료 계산 도구 모음
    
    거래 수수료 계산, 제휴 수수료 계산, 수익 추적
    """
    
    def __init__(self, data_dir: str = "financial_data", default_commission_rate: float = 0.005, default_affiliate_rate: float = 0.03):
        """
        CommissionTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
            default_commission_rate: 기본 거래 수수료율 (0.5%)
            default_affiliate_rate: 기본 제휴 수수료율 (3%)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.commission_file = self.data_dir / "commission_data.json"
        self.default_commission_rate = default_commission_rate
        self.default_affiliate_rate = default_affiliate_rate
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if self.commission_file.exists():
            with open(self.commission_file, 'r', encoding='utf-8') as f:
                self.commission_data = json.load(f)
        else:
            self.commission_data = {
                "transactions": [],
                "total_commission": 0.0,
                "total_affiliate_commission": 0.0
            }
    
    def _save_data(self):
        """데이터 저장"""
        with open(self.commission_file, 'w', encoding='utf-8') as f:
            json.dump(self.commission_data, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """Commission 도구 초기화"""
        self.tools.append(self._create_calculate_commission_tool())
        self.tools.append(self._create_calculate_affiliate_commission_tool())
        self.tools.append(self._create_track_revenue_tool())
        logger.info(f"Initialized {len(self.tools)} commission tools")
    
    def _create_calculate_commission_tool(self) -> BaseTool:
        @tool("commission_calculate", args_schema=CalculateCommissionInput)
        def calculate_commission(
            transaction_amount: float,
            commission_rate: Optional[float] = None
        ) -> str:
            """
            거래 수수료를 계산합니다.
            Args:
                transaction_amount: 거래 금액
                commission_rate: 수수료율 (기본값 사용 시 None)
            Returns:
                수수료 계산 결과 (JSON 문자열)
            """
            logger.info(f"Calculating commission for transaction amount: {transaction_amount}")
            
            rate = commission_rate if commission_rate is not None else self.default_commission_rate
            commission = transaction_amount * rate
            
            result = {
                "transaction_amount": transaction_amount,
                "commission_rate": rate,
                "commission": commission,
                "net_amount": transaction_amount - commission,
                "platform_revenue": commission
            }
            
            # 거래 기록
            transaction = {
                "transaction_id": f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "transaction_amount": transaction_amount,
                "commission": commission,
                "commission_rate": rate,
                "type": "trade",
                "timestamp": datetime.now().isoformat()
            }
            
            self.commission_data["transactions"].append(transaction)
            self.commission_data["total_commission"] += commission
            self._save_data()
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return calculate_commission
    
    def _create_calculate_affiliate_commission_tool(self) -> BaseTool:
        @tool("commission_calculate_affiliate", args_schema=CalculateAffiliateCommissionInput)
        def calculate_affiliate_commission(
            product_type: str,
            transaction_amount: float,
            affiliate_rate: Optional[float] = None
        ) -> str:
            """
            제휴 수수료를 계산합니다.
            Args:
                product_type: 상품 유형
                transaction_amount: 거래 금액
                affiliate_rate: 제휴 수수료율 (기본값 사용 시 None)
            Returns:
                제휴 수수료 계산 결과 (JSON 문자열)
            """
            logger.info(f"Calculating affiliate commission for {product_type}, amount: {transaction_amount}")
            
            # 상품 유형별 기본 수수료율
            product_rates = {
                "credit_card": 0.03,
                "loan": 0.05,
                "insurance": 0.04,
                "real_estate": 0.01,
            }
            
            rate = affiliate_rate if affiliate_rate is not None else product_rates.get(product_type, self.default_affiliate_rate)
            commission = transaction_amount * rate
            
            result = {
                "product_type": product_type,
                "transaction_amount": transaction_amount,
                "affiliate_rate": rate,
                "affiliate_commission": commission,
                "platform_revenue": commission
            }
            
            # 거래 기록
            transaction = {
                "transaction_id": f"aff_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "product_type": product_type,
                "transaction_amount": transaction_amount,
                "affiliate_commission": commission,
                "affiliate_rate": rate,
                "type": "affiliate",
                "timestamp": datetime.now().isoformat()
            }
            
            self.commission_data["transactions"].append(transaction)
            self.commission_data["total_affiliate_commission"] += commission
            self._save_data()
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return calculate_affiliate_commission
    
    def _create_track_revenue_tool(self) -> BaseTool:
        @tool("commission_track_revenue")
        def track_revenue(period_days: int = 30) -> str:
            """
            수익을 추적합니다.
            Args:
                period_days: 추적 기간 (일)
            Returns:
                수익 추적 결과 (JSON 문자열)
            """
            logger.info(f"Tracking revenue for period: {period_days} days")
            
            cutoff_date = datetime.now() - timedelta(days=period_days)
            
            recent_transactions = [
                txn for txn in self.commission_data["transactions"]
                if datetime.fromisoformat(txn["timestamp"]) >= cutoff_date
            ]
            
            trade_commission = sum(
                txn.get("commission", 0.0) for txn in recent_transactions
                if txn.get("type") == "trade"
            )
            
            affiliate_commission = sum(
                txn.get("affiliate_commission", 0.0) for txn in recent_transactions
                if txn.get("type") == "affiliate"
            )
            
            total_revenue = trade_commission + affiliate_commission
            
            result = {
                "period_days": period_days,
                "total_revenue": total_revenue,
                "trade_commission": trade_commission,
                "affiliate_commission": affiliate_commission,
                "transaction_count": len(recent_transactions),
                "breakdown_by_product": {}
            }
            
            # 상품 유형별 분석
            for txn in recent_transactions:
                if txn.get("type") == "affiliate":
                    product_type = txn.get("product_type", "unknown")
                    if product_type not in result["breakdown_by_product"]:
                        result["breakdown_by_product"][product_type] = 0.0
                    result["breakdown_by_product"][product_type] += txn.get("affiliate_commission", 0.0)
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return track_revenue
    
    def get_tools(self) -> List[BaseTool]:
        """모든 Commission 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 Commission 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

