"""
점수 계산 에이전트

게임 점수 계산 및 승부 판정을 담당하는 에이전트
"""

from typing import Dict, List, Any, Optional
from ..core.agent_base import BaseAgent


class ScoreCalculatorAgent(BaseAgent):
    """
    점수 계산 및 승부 판정 전문 에이전트
    
    게임 결과 계산:
    - 실시간 점수 계산
    - 승리 조건 확인
    - 최종 순위 결정
    - 게임 통계 생성
    """
    
    def __init__(self, llm_client, mcp_client, agent_id: str = "score_calculator"):
        super().__init__(llm_client, mcp_client, agent_id)
        
    async def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        점수 계산을 위한 환경 인식
        """
        game_state = environment.get("game_state", {})
        players = environment.get("players", [])
        parsed_rules = environment.get("parsed_rules", {})
        
        return {
            "game_state": game_state,
            "players": players,
            "parsed_rules": parsed_rules,
            "calculation_needed": True
        }
    
    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 기반 점수 계산 로직
        """
        prompt = f"""
당신은 게임 점수 계산 전문가입니다. 주어진 게임의 규칙, 최종 상태, 플레이어 정보를 바탕으로 최종 점수를 계산하고 승자를 결정해주세요.

**게임 규칙 (승리 조건):**
{perception.get("parsed_rules", {}).get("objectives", "점수가 가장 높은 플레이어가 승리합니다.")}

**최종 게임 상태:**
{perception.get("game_state", {})}

**플레이어 정보:** 
{perception.get("players", [])}

**요청:**
모든 정보를 종합하여 각 플레이어의 최종 점수를 계산하고, 순위를 매긴 후, 승자를 결정해주세요.

**응답 형식 (반드시 JSON만 반환):**
{{
  "final_scores": {{
    "플레이어ID_1": 점수,
    "플레이어ID_2": 점수
  }},
  "rankings": ["플레이어ID_1", "플레이어ID_2", ...],
  "winners": ["승리한_플레이어ID_1", "필요시_공동승자_ID_2"],
  "reason": "점수 계산 및 승자 결정에 대한 상세한 설명"
}}
"""
        
        llm_response = await self.llm_client.complete(prompt)
        
        return {
            "decision": "calculate_scores",
            "calculation_result": llm_response
        }
    
    async def act(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """
        점수 계산 실행
        """
        try:
            import json
            result_data = json.loads(reasoning.get("calculation_result", "{}"))
            
            return {
                "action": "scores_calculated",
                "final_scores": result_data.get("final_scores", {}),
                "winners": result_data.get("winners", []),
                "rankings": result_data.get("rankings", []),
                "reason": result_data.get("reason", "No reason provided.")
            }
        except (json.JSONDecodeError, AttributeError):
            return {
                "action": "calculation_failed",
                "final_scores": {},
                "winners": [],
                "rankings": [],
                "reason": "Failed to parse LLM response for score calculation."
            } 