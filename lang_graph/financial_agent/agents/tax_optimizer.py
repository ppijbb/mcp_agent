"""
세금 최적화 Agent 노드

공제 항목 발견, 세금 계산, 세금 최적화 전략 제안
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional
from ..state import AgentState
from ..llm.model_manager import ModelManager, ModelProvider
from ..llm_client import call_llm, model_manager
from ..tools.tax_tools import TaxTools
from ..config import get_multi_model_llm_config

logger = logging.getLogger(__name__)

# 전역 인스턴스 (초기화 시 생성)
_tax_optimizer_agent = None


def _get_tax_optimizer_agent():
    """Tax Optimizer Agent 싱글톤 인스턴스 반환"""
    global _tax_optimizer_agent
    if _tax_optimizer_agent is None:
        multi_model_config = get_multi_model_llm_config()
        _tax_optimizer_agent = TaxOptimizerAgent(
            model_manager=model_manager,
            preferred_provider=multi_model_config.preferred_provider
        )
    return _tax_optimizer_agent


class TaxOptimizerAgent:
    """
    세금 최적화 Agent
    
    공제 항목 발견, 세금 계산, 세금 최적화 전략 제안을 수행합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "financial_data"
    ):
        """
        TaxOptimizerAgent 초기화
        
        Args:
            model_manager: ModelManager 인스턴스
            preferred_provider: 선호하는 Provider
            data_dir: 데이터 저장 디렉토리
        """
        self.model_manager = model_manager
        self.preferred_provider = preferred_provider
        self.data_dir = data_dir
        
        # 도구 초기화
        self.tax_tools = TaxTools(data_dir=data_dir)
        self.tools = self.tax_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
    
    async def optimize_taxes(self, user_id: str, user_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        세금을 최적화합니다.
        
        Args:
            user_id: 사용자 ID
            user_data: 사용자 재무 데이터 (income, expenses 등)
        
        Returns:
            세금 최적화 결과
        """
        input_message = f"""
        사용자 '{user_id}'의 세금을 최적화해주세요.
        
        사용자 데이터:
        {json.dumps(user_data, ensure_ascii=False, indent=2) if user_data else "없음 (도구를 사용하여 조회)"}
        
        다음 항목을 수행해주세요:
        1. 공제 항목 발견
        2. 세금 계산
        3. 세금 최적화 전략 제안
        
        결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "user_id": "{user_id}",
            "tax_optimization": {{
                "available_deductions": {{
                    "항목명": 금액
                }},
                "total_deductions": 0.0,
                "taxable_income": 0.0,
                "current_tax": 0.0,
                "optimized_tax": 0.0,
                "tax_savings": 0.0,
                "optimization_strategies": [
                    {{
                        "strategy": "전략명",
                        "description": "설명",
                        "potential_savings": 0.0
                    }}
                ]
            }}
        }}
        """
        
        try:
            response = call_llm(input_message)
            logger.info(f"Tax optimization completed for user {user_id}")
            return json.loads(response) if isinstance(response, str) else {"user_id": user_id, "tax_optimization": {}}
        except Exception as e:
            logger.error(f"Failed to optimize taxes for user {user_id}: {e}")
            raise


def tax_optimizer_node(state: AgentState) -> Dict:
    """
    세금 최적화 노드: 사용자의 세금을 최적화합니다.
    """
    print("--- AGENT: Tax Optimizer ---")
    log_message = "세금 최적화를 시작합니다."
    state["log"].append(log_message)
    print(log_message)
    
    user_id = state.get("user_id", "default_user")
    
    try:
        agent = _get_tax_optimizer_agent()
        result = asyncio.run(agent.optimize_taxes(user_id))
        
        tax_optimization = result.get("tax_optimization", {})
        
        log_message = f"세금 최적화 완료. 예상 절감액: ${tax_optimization.get('tax_savings', 0.0):.2f}"
        state["log"].append(log_message)
        print(log_message)
        
        return {"tax_optimization": tax_optimization}
    except Exception as e:
        error_message = f"세금 최적화 중 오류 발생: {e}"
        print(error_message)
        state["log"].append(error_message)
        return {"error_message": error_message}

