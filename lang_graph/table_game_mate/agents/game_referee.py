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
        # 게임 규칙 업데이트
        if "parsed_rules" in environment:
            self.game_rules = environment["parsed_rules"]

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
        game_rules = self.game_rules or "No specific rules loaded. Use general game knowledge."
        
        prompt = f"""
당신은 엄격한 게임 심판입니다. 주어진 게임 규칙과 현재 상태를 바탕으로 플레이어의 행동이 유효한지 판단해주세요.

**게임 규칙:**
{game_rules}

**현재 게임 상태:** 
{perception.get('game_state', {})}

**검증할 플레이어 행동:**
{action}

**요청:**
이 행동이 게임 규칙에 따라 유효한지(valid) 아닌지(invalid) 판단하고, 그 이유를 설명해주세요.

**응답 형식 (반드시 JSON만 반환):**
{{
  "is_valid": true/false,
  "reason": "판단에 대한 구체적인 이유"
}}
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
            try:
                import json
                validation_data = json.loads(reasoning.get("validation_result", "{}"))
                is_valid = validation_data.get("is_valid", False)
                reason = validation_data.get("reason", "Could not parse LLM response.")
                
                return {
                    "action": "validation_complete",
                    "is_valid": is_valid,
                    "message": reason
                }
            except (json.JSONDecodeError, AttributeError):
                 return {
                    "action": "validation_failed",
                    "is_valid": False, # 파싱 실패 시 안전하게 무효 처리
                    "message": "LLM의 응답을 파싱하는 데 실패했습니다."
                }

        
        return {"action": "no_validation_performed"} 