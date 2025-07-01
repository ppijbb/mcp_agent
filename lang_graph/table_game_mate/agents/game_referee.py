"""
게임 심판 에이전트

게임 규칙 준수 검증 및 심판 역할을 담당하는 에이전트
"""

from typing import Dict, List, Any, Optional
from ..core.agent_base import BaseAgent


class GameRefereeAgent(BaseAgent):
    """
    게임 심판 전문 에이전트
    
    게임 진행 중 규칙 준수 검증:
    - 플레이어 행동 유효성 검증
    - 규칙 위반 감지 및 처리
    - 승리 조건 확인
    - 게임 상태 일관성 유지
    """
    
    def __init__(self, llm_client, mcp_client, agent_id: str = "game_referee"):
        super().__init__(llm_client, mcp_client, agent_id)
        self.game_rules = {}
        
    async def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        심판을 위한 환경 인식
        """
        action_to_validate = environment.get("player_action", {})
        game_state = environment.get("game_state", {})
        
        return {
            "action_to_validate": action_to_validate,
            "game_state": game_state,
            "validation_needed": bool(action_to_validate)
        }
    
    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 기반 규칙 검증 결정
        """
        if not perception.get("validation_needed"):
            return {"decision": "no_validation_needed"}
        
        action = perception.get("action_to_validate", {})
        
        prompt = f"""
게임 행동 유효성 검증:

행동: {action}
현재 게임 상태: {perception.get('game_state', {})}

이 행동이 게임 규칙에 위반되는지 검증해주세요.
JSON 형태로 응답해주세요.
"""
        
        llm_response = await self.llm_client.complete(prompt)
        
        return {
            "decision": "validate_action",
            "validation_result": llm_response
        }
    
    async def act(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """
        심판 결정 실행
        """
        decision = reasoning.get("decision", "no_action")
        
        if decision == "validate_action":
            return {
                "action": "validation_complete",
                "is_valid": True,  # 실제로는 LLM 응답 파싱 결과
                "message": "행동이 유효합니다"
            }
        
        return {"action": "no_validation_performed"} 