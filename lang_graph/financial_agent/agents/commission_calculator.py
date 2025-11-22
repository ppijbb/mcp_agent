"""
수수료 계산 Agent 노드

거래 수수료 계산, 제휴 수수료 계산, 수익 추적
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List
from ..state import AgentState
from ..llm.model_manager import ModelManager, ModelProvider
from ..llm_client import call_llm, model_manager
from ..tools.commission_tools import CommissionTools
from ..config import get_trading_config, get_multi_model_llm_config

logger = logging.getLogger(__name__)

# 전역 인스턴스 (초기화 시 생성)
_commission_calculator_agent = None


def _get_commission_calculator_agent():
    """Commission Calculator Agent 싱글톤 인스턴스 반환"""
    global _commission_calculator_agent
    if _commission_calculator_agent is None:
        trading_config = get_trading_config()
        multi_model_config = get_multi_model_llm_config()
        _commission_calculator_agent = CommissionCalculatorAgent(
            model_manager=model_manager,
            preferred_provider=multi_model_config.preferred_provider,
            default_commission_rate=trading_config.commission_rate,
            default_affiliate_rate=trading_config.affiliate_commission_rate
        )
    return _commission_calculator_agent


class CommissionCalculatorAgent:
    """
    수수료 계산 Agent
    
    거래 수수료 계산, 제휴 수수료 계산, 수익 추적을 수행합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "financial_data",
        default_commission_rate: float = 0.005,
        default_affiliate_rate: float = 0.03
    ):
        """
        CommissionCalculatorAgent 초기화
        
        Args:
            model_manager: ModelManager 인스턴스
            preferred_provider: 선호하는 Provider
            data_dir: 데이터 저장 디렉토리
            default_commission_rate: 기본 거래 수수료율
            default_affiliate_rate: 기본 제휴 수수료율
        """
        self.model_manager = model_manager
        self.preferred_provider = preferred_provider
        
        # 도구 초기화
        self.commission_tools = CommissionTools(
            data_dir=data_dir,
            default_commission_rate=default_commission_rate,
            default_affiliate_rate=default_affiliate_rate
        )
        self.tools = self.commission_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
    
    async def calculate_commissions(
        self,
        trade_results: Optional[List[Dict[str, Any]]] = None,
        affiliate_transactions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        수수료를 계산합니다.
        
        Args:
            trade_results: 거래 결과 목록
            affiliate_transactions: 제휴 거래 목록
        
        Returns:
            수수료 계산 결과
        """
        input_message = f"""
        다음 거래들에 대한 수수료를 계산해주세요.
        
        거래 결과:
        {json.dumps(trade_results, ensure_ascii=False, indent=2) if trade_results else "없음"}
        
        제휴 거래:
        {json.dumps(affiliate_transactions, ensure_ascii=False, indent=2) if affiliate_transactions else "없음"}
        
        다음 항목을 수행해주세요:
        1. 각 거래별 수수료 계산
        2. 제휴 거래별 수수료 계산
        3. 총 수수료 집계
        
        결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "total_commission": 0.0,
            "total_affiliate_commission": 0.0,
            "total_revenue": 0.0,
            "commission_breakdown": {{
                "trade_commissions": [],
                "affiliate_commissions": []
            }}
        }}
        """
        
        try:
            response = call_llm(input_message)
            logger.info("Commission calculation completed")
            return json.loads(response) if isinstance(response, str) else {"total_commission": 0.0, "total_revenue": 0.0}
        except Exception as e:
            logger.error(f"Failed to calculate commissions: {e}")
            raise


def commission_calculator_node(state: AgentState) -> Dict:
    """
    수수료 계산 노드: 거래 및 제휴 거래에 대한 수수료를 계산합니다.
    """
    print("--- AGENT: Commission Calculator ---")
    log_message = "수수료 계산을 시작합니다."
    state["log"].append(log_message)
    print(log_message)
    
    trade_results = state.get("trade_results", [])
    
    try:
        agent = _get_commission_calculator_agent()
        result = asyncio.run(agent.calculate_commissions(trade_results=trade_results))
        
        total_commission = result.get("total_commission", 0.0)
        total_affiliate_commission = result.get("total_affiliate_commission", 0.0)
        total_revenue = result.get("total_revenue", 0.0)
        
        # 거래 설정에서 수수료율 가져오기
        trading_config = get_trading_config()
        
        log_message = f"수수료 계산 완료. 총 수익: ${total_revenue:.2f}"
        state["log"].append(log_message)
        print(log_message)
        
        return {
            "commission_rate": trading_config.commission_rate,
            "total_commission": total_commission,
            "affiliate_commission": total_affiliate_commission,
            "total_revenue": total_revenue
        }
    except Exception as e:
        error_message = f"수수료 계산 중 오류 발생: {e}"
        print(error_message)
        state["log"].append(error_message)
        return {"error_message": error_message}

