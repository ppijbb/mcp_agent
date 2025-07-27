"""
플레이어 관리 에이전트

AI 플레이어들의 생성, 관리, 상태 추적을 담당하는 에이전트
"""

from typing import Dict, List, Any, Optional
from ..core.agent_base import BaseAgent
from ..models.persona import PersonaProfile
from ..models.game_state import PlayerState


class PlayerManagerAgent(BaseAgent):
    """
    AI 플레이어 관리 전문 에이전트
    
    AI 플레이어들의 전체 생명주기 관리:
    - 페르소나 기반 플레이어 생성
    - 플레이어 상태 추적 및 업데이트
    - 턴 순서 관리
    - 플레이어 간 상호작용 조정
    """
    
    def __init__(self, llm_client, mcp_client, agent_id: str = "player_manager"):
        super().__init__(llm_client, mcp_client, agent_id)
        self.active_players: Dict[str, PlayerState] = {}
        
    async def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        플레이어 관리를 위한 환경 인식
        """
        personas = environment.get("persona_profiles", [])
        game_config = environment.get("game_config", {})
        action_type = environment.get("action", "create_players")
        
        return {
            "action_type": action_type,
            "personas_available": len(personas),
            "persona_profiles": personas,
            "game_config": game_config,
            "current_players": list(self.active_players.keys())
        }
    
    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 기반 플레이어 관리 결정
        """
        action_type = perception.get("action_type", "create_players")
        
        if action_type == "create_players":
            return await self._reason_player_creation(perception)
        elif action_type == "update_players":
            return await self._reason_player_update(perception)
        else:
            return {"decision": "no_action_needed"}
    
    async def act(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """
        플레이어 관리 액션 실행
        """
        decision = reasoning.get("decision", "no_action")
        
        if decision == "create_players":
            return await self._create_players(reasoning)
        elif decision == "update_players":
            return await self._update_players(reasoning)
        else:
            return {"action": "no_action_taken"}
    
    async def _reason_player_creation(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """플레이어 생성 결정"""
        personas = perception.get("persona_profiles", [])
        game_config = perception.get("game_config", {})
        
        prompt = f"""
AI 플레이어 생성 계획:

사용 가능한 페르소나: {len(personas)}개
게임 설정: {game_config}

각 페르소나를 바탕으로 AI 플레이어를 생성하고 초기 상태를 설정해주세요.
JSON 형태로 응답해주세요.
"""
        
        llm_response = await self.llm_client.complete(prompt)
        
        return {
            "decision": "create_players",
            "personas": personas,
            "llm_guidance": llm_response
        }
    
    async def _create_players(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """AI 플레이어들 생성"""
        personas = reasoning.get("personas", [])
        created_players = []
        
        for i, persona in enumerate(personas):
            player_state = PlayerState(
                player_id=f"ai_player_{i+1}",
                name=persona["name"],
                persona=persona,
                score=0,
                is_ai=True,
                turn_order=i
            )
            
            self.active_players[f"ai_player_{i+1}"] = player_state
            created_players.append(player_state)
        
        return {
            "action": "players_created",
            "players": created_players,
            "created_players": created_players,
            "total_players": len(created_players)
        }
    
    async def _reason_player_update(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """플레이어 상태 업데이트 결정"""
        return {"decision": "update_players"}
    
    async def _update_players(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """플레이어 상태 업데이트"""
        return {"action": "players_updated"} 