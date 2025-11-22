"""
재무 목표 달성 Agent 노드

장기 재무 목표 관리, 목표 달성 진행률 추적, 목표별 투자 전략 제안
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List
from ..state import AgentState
from ..llm.model_manager import ModelManager, ModelProvider
from ..llm_client import call_llm, model_manager
from ..tools.goal_tools import GoalTools
from ..config import get_multi_model_llm_config

logger = logging.getLogger(__name__)

# 전역 인스턴스 (초기화 시 생성)
_goal_tracker_agent = None


def _get_goal_tracker_agent():
    """Goal Tracker Agent 싱글톤 인스턴스 반환"""
    global _goal_tracker_agent
    if _goal_tracker_agent is None:
        multi_model_config = get_multi_model_llm_config()
        _goal_tracker_agent = GoalTrackerAgent(
            model_manager=model_manager,
            preferred_provider=multi_model_config.preferred_provider
        )
    return _goal_tracker_agent


class GoalTrackerAgent:
    """
    재무 목표 달성 Agent
    
    장기 재무 목표 관리, 목표 달성 진행률 추적, 목표별 투자 전략 제안을 수행합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "financial_data"
    ):
        """
        GoalTrackerAgent 초기화
        
        Args:
            model_manager: ModelManager 인스턴스
            preferred_provider: 선호하는 Provider
            data_dir: 데이터 저장 디렉토리
        """
        self.model_manager = model_manager
        self.preferred_provider = preferred_provider
        self.data_dir = data_dir
        
        # 도구 초기화
        self.goal_tools = GoalTools(data_dir=data_dir)
        self.tools = self.goal_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
    
    async def track_goals(
        self,
        user_id: str,
        goals: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        재무 목표를 추적합니다.
        
        Args:
            user_id: 사용자 ID
            goals: 목표 목록 (선택 사항)
        
        Returns:
            목표 추적 결과
        """
        input_message = f"""
        사용자 '{user_id}'의 재무 목표를 추적해주세요.
        
        목표 목록:
        {json.dumps(goals, ensure_ascii=False, indent=2) if goals else "없음 (도구를 사용하여 조회)"}
        
        다음 항목을 수행해주세요:
        1. 목표 진행률 추적
        2. 목표별 투자 전략 제안
        3. 목표 달성 가능성 분석
        
        결과는 다음 JSON 형식으로 반환해주세요:
        {{
            "user_id": "{user_id}",
            "financial_goals": [
                {{
                    "goal_name": "목표명",
                    "target_amount": 0.0,
                    "current_amount": 0.0,
                    "progress_percentage": 0.0,
                    "months_remaining": 0,
                    "monthly_savings_needed": 0.0,
                    "investment_strategy": {{
                        "risk_level": "conservative/moderate/aggressive",
                        "recommended_assets": [],
                        "expected_return": 0.0
                    }}
                }}
            ],
            "goal_progress": {{
                "total_progress": 0.0,
                "on_track_goals": [],
                "at_risk_goals": []
            }}
        }}
        """
        
        try:
            response = call_llm(input_message)
            logger.info(f"Goal tracking completed for user {user_id}")
            return json.loads(response) if isinstance(response, str) else {"user_id": user_id, "financial_goals": []}
        except Exception as e:
            logger.error(f"Failed to track goals for user {user_id}: {e}")
            raise


def goal_tracker_node(state: AgentState) -> Dict:
    """
    재무 목표 추적 노드: 사용자의 재무 목표를 추적합니다.
    """
    print("--- AGENT: Goal Tracker ---")
    log_message = "재무 목표 추적을 시작합니다."
    state["log"].append(log_message)
    print(log_message)
    
    user_id = state.get("user_id", "default_user")
    
    try:
        agent = _get_goal_tracker_agent()
        result = asyncio.run(agent.track_goals(user_id))
        
        financial_goals = result.get("financial_goals", [])
        goal_progress = result.get("goal_progress", {})
        
        log_message = f"재무 목표 추적 완료. 총 목표 수: {len(financial_goals)}"
        state["log"].append(log_message)
        print(log_message)
        
        return {
            "financial_goals": financial_goals,
            "goal_progress": goal_progress
        }
    except Exception as e:
        error_message = f"재무 목표 추적 중 오류 발생: {e}"
        print(error_message)
        state["log"].append(error_message)
        return {"error_message": error_message}

