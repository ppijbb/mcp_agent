"""
재무 분석 Agent 노드

소비 패턴 분석, 예산 관리, 저축 목표 추적, 재무 건강 점수 계산
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional
from ..state import AgentState
from ..llm.model_manager import ModelManager, ModelProvider
from ..llm_client import call_llm, model_manager
from ..tools.finance_tools import FinanceTools
from ..config import get_multi_model_llm_config

logger = logging.getLogger(__name__)

# 전역 인스턴스 (초기화 시 생성)
_financial_analyzer_agent = None


def _get_financial_analyzer_agent():
    """Financial Analyzer Agent 싱글톤 인스턴스 반환"""
    global _financial_analyzer_agent
    if _financial_analyzer_agent is None:
        multi_model_config = get_multi_model_llm_config()
        _financial_analyzer_agent = FinancialAnalyzerAgent(
            model_manager=model_manager,
            preferred_provider=multi_model_config.preferred_provider
        )
    return _financial_analyzer_agent


class FinancialAnalyzerAgent:
    """
    재무 분석 Agent
    
    소비 패턴 분석, 예산 관리, 저축 목표 추적, 재무 건강 점수 계산을 수행합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "financial_data"
    ):
        """
        FinancialAnalyzerAgent 초기화
        
        Args:
            model_manager: ModelManager 인스턴스
            preferred_provider: 선호하는 Provider
            data_dir: 데이터 저장 디렉토리
        """
        self.model_manager = model_manager
        self.preferred_provider = preferred_provider
        self.data_dir = data_dir
        
        # 도구 초기화
        self.finance_tools = FinanceTools(data_dir=data_dir)
        self.tools = self.finance_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
    
    async def analyze_financial_status(self, user_id: str, user_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        재무 상태를 종합 분석합니다.
        
        Args:
            user_id: 사용자 ID
            user_data: 사용자 재무 데이터 (선택 사항)
        
        Returns:
            재무 분석 결과
        """
        input_message = f"""
        사용자 '{user_id}'의 재무 상태를 종합 분석해주세요.
        
        사용자 데이터:
        {json.dumps(user_data, ensure_ascii=False, indent=2) if user_data else "없음 (도구를 사용하여 조회)"}
        
        다음 항목을 분석해주세요:
        1. 소비 패턴 분석 (최근 30일)
        2. 예산 관리 상태
        3. 저축 목표 진행률
        4. 재무 건강 점수 계산
        
        결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "user_id": "{user_id}",
            "financial_analysis": {{
                "spending_pattern": {{
                    "total_spending": 0.0,
                    "avg_daily_spending": 0.0,
                    "top_categories": []
                }},
                "budget_status": {{
                    "monthly_income": 0.0,
                    "total_budget": 0.0,
                    "remaining_income": 0.0
                }},
                "savings_progress": {{
                    "goals": [],
                    "total_progress": 0.0
                }},
                "health_score": 0.0,
                "recommendations": []
            }}
        }}
        """
        
        try:
            response = call_llm(input_message)
            logger.info(f"Financial analysis completed for user {user_id}")
            return json.loads(response) if isinstance(response, str) else {"user_id": user_id, "financial_analysis": {}}
        except Exception as e:
            logger.error(f"Failed to analyze financial status for user {user_id}: {e}")
            raise


def financial_analyzer_node(state: AgentState) -> Dict:
    """
    재무 분석 노드: 사용자의 재무 상태를 종합 분석합니다.
    """
    print("--- AGENT: Financial Analyzer ---")
    log_message = "재무 분석을 시작합니다."
    state["log"].append(log_message)
    print(log_message)
    
    user_id = state.get("user_id")
    if not user_id:
        # user_id가 없으면 기본값 사용
        user_id = "default_user"
        state["user_id"] = user_id
    
    try:
        agent = _get_financial_analyzer_agent()
        result = asyncio.run(agent.analyze_financial_status(user_id))
        
        financial_analysis = result.get("financial_analysis", {})
        budget_status = financial_analysis.get("budget_status", {})
        savings_progress = financial_analysis.get("savings_progress", {})
        
        log_message = f"재무 분석 완료. 건강 점수: {financial_analysis.get('health_score', 0.0)}"
        state["log"].append(log_message)
        print(log_message)
        
        return {
            "financial_analysis": financial_analysis,
            "budget_status": budget_status,
            "savings_progress": savings_progress
        }
    except Exception as e:
        error_message = f"재무 분석 중 오류 발생: {e}"
        print(error_message)
        state["log"].append(error_message)
        return {"error_message": error_message}

