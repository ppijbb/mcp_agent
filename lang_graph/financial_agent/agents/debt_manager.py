"""
부채 관리 Agent 노드

대출 상환 전략, 이자 최소화 계획, 부채 구조 분석
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List
from ..state import AgentState
from ..llm.model_manager import ModelManager, ModelProvider
from ..llm_client import call_llm, model_manager
from ..tools.debt_tools import DebtTools
from ..config import get_multi_model_llm_config

logger = logging.getLogger(__name__)

# 전역 인스턴스 (초기화 시 생성)
_debt_manager_agent = None


def _get_debt_manager_agent():
    """Debt Manager Agent 싱글톤 인스턴스 반환"""
    global _debt_manager_agent
    if _debt_manager_agent is None:
        multi_model_config = get_multi_model_llm_config()
        _debt_manager_agent = DebtManagerAgent(
            model_manager=model_manager,
            preferred_provider=multi_model_config.preferred_provider
        )
    return _debt_manager_agent


class DebtManagerAgent:
    """
    부채 관리 Agent
    
    대출 상환 전략, 이자 최소화 계획, 부채 구조 분석을 수행합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "financial_data"
    ):
        """
        DebtManagerAgent 초기화
        
        Args:
            model_manager: ModelManager 인스턴스
            preferred_provider: 선호하는 Provider
            data_dir: 데이터 저장 디렉토리
        """
        self.model_manager = model_manager
        self.preferred_provider = preferred_provider
        self.data_dir = data_dir
        
        # 도구 초기화
        self.debt_tools = DebtTools(data_dir=data_dir)
        self.tools = self.debt_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
    
    async def manage_debt(
        self,
        user_id: str,
        loans: Optional[List[Dict[str, Any]]] = None,
        monthly_payment_capacity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        부채를 관리합니다.
        
        Args:
            user_id: 사용자 ID
            loans: 대출 목록 (선택 사항)
            monthly_payment_capacity: 월 상환 가능 금액 (선택 사항)
        
        Returns:
            부채 관리 결과
        """
        input_message = f"""
        사용자 '{user_id}'의 부채를 관리해주세요.
        
        대출 정보:
        {json.dumps(loans, ensure_ascii=False, indent=2) if loans else "없음 (도구를 사용하여 조회)"}
        
        월 상환 가능 금액: {monthly_payment_capacity if monthly_payment_capacity else "미지정"}
        
        다음 항목을 수행해주세요:
        1. 부채 구조 분석
        2. 상환 전략 생성 (snowball 또는 avalanche)
        3. 이자 최소화 계획
        
        결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "user_id": "{user_id}",
            "debt_management": {{
                "total_debt": 0.0,
                "total_monthly_payment": 0.0,
                "weighted_interest_rate": 0.0,
                "repayment_strategy": {{
                    "strategy_type": "snowball/avalanche",
                    "repayment_plan": [],
                    "total_interest_saved": 0.0,
                    "total_months_saved": 0
                }},
                "minimization_plan": {{
                    "recommendations": [],
                    "expected_savings": 0.0
                }}
            }}
        }}
        """
        
        try:
            response = call_llm(input_message)
            logger.info(f"Debt management completed for user {user_id}")
            return json.loads(response) if isinstance(response, str) else {"user_id": user_id, "debt_management": {}}
        except Exception as e:
            logger.error(f"Failed to manage debt for user {user_id}: {e}")
            raise


def debt_manager_node(state: AgentState) -> Dict:
    """
    부채 관리 노드: 사용자의 부채를 관리합니다.
    """
    print("--- AGENT: Debt Manager ---")
    log_message = "부채 관리를 시작합니다."
    state["log"].append(log_message)
    print(log_message)
    
    user_id = state.get("user_id", "default_user")
    
    try:
        agent = _get_debt_manager_agent()
        result = asyncio.run(agent.manage_debt(user_id))
        
        debt_management = result.get("debt_management", {})
        
        log_message = f"부채 관리 완료. 총 부채: ${debt_management.get('total_debt', 0.0):.2f}"
        state["log"].append(log_message)
        print(log_message)
        
        return {"debt_management": debt_management}
    except Exception as e:
        error_message = f"부채 관리 중 오류 발생: {e}"
        print(error_message)
        state["log"].append(error_message)
        return {"error_message": error_message}

