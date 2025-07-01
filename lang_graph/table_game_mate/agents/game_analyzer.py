"""
게임 분석 에이전트

BGG API를 활용하여 게임 정보를 수집하고 분석하는 에이전트
"""

from typing import Dict, List, Any, Optional
import json
from ..core.agent_base import BaseAgent


class GameAnalyzerAgent(BaseAgent):
    """
    게임 분석 전문 에이전트
    
    BGG API와 연동하여:
    - 게임 기본 정보 수집
    - 복잡도 및 플레이어 수 분석
    - 메커니즘 및 카테고리 분석
    - 리뷰 및 평점 데이터 수집
    """
    
    def __init__(self, llm_client, mcp_client, agent_id: str = "game_analyzer"):
        super().__init__(llm_client, mcp_client, agent_id)
        self.supported_sources = ["bgg", "tabletopia", "manual"]
        
    async def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        게임 분석을 위한 환경 인식
        
        - 게임 이름/ID 확인
        - BGG에서 게임 데이터 수집
        - 현재 플레이어 수 및 설정 확인
        """
        game_query = environment.get("game_name") or environment.get("game_id")
        player_count = environment.get("player_count", 4)
        
        if not game_query:
            return {"error": "게임 이름 또는 ID가 필요합니다"}
        
        # BGG에서 게임 정보 수집
        try:
            # BGG 검색
            search_result = await self.mcp_client.call(
                "bgg_server",
                "search_games", 
                {"query": game_query, "limit": 5}
            )
            
            if not search_result.get("games"):
                return {"error": f"'{game_query}' 게임을 찾을 수 없습니다"}
            
            # 가장 적합한 게임 선택 (첫 번째 결과)
            best_match = search_result["games"][0]
            game_id = best_match["id"]
            
            # 상세 정보 수집
            game_details = await self.mcp_client.call(
                "bgg_server",
                "get_game_details",
                {"game_id": game_id}
            )
            
            return {
                "game_found": True,
                "game_id": game_id,
                "search_results": search_result["games"],
                "game_details": game_details,
                "target_player_count": player_count,
                "source": "bgg"
            }
            
        except Exception as e:
            return {
                "error": f"BGG API 호출 실패: {str(e)}",
                "game_query": game_query,
                "fallback_needed": True
            }
    
    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 기반 게임 분석 및 추론
        
        수집된 게임 데이터를 바탕으로:
        - 게임 적합성 평가
        - 필요한 AI 플레이어 수 및 성격 추천
        - 게임 복잡도 분석
        - 예상 플레이 시간 계산
        """
        if perception.get("error"):
            return {"analysis_failed": True, "error": perception["error"]}
        
        game_details = perception.get("game_details", {})
        target_players = perception.get("target_player_count", 4)
        
        # LLM에게 게임 분석 요청
        analysis_prompt = f"""
게임 분석 요청:

게임 정보:
- 이름: {game_details.get('name', 'Unknown')}
- 최소/최대 플레이어: {game_details.get('min_players')}-{game_details.get('max_players')}명
- 플레이 시간: {game_details.get('playing_time', 0)}분
- 복잡도: {game_details.get('complexity', 0)}/5
- 메커니즘: {', '.join(game_details.get('mechanics', []))}
- 카테고리: {', '.join(game_details.get('categories', []))}
- 평점: {game_details.get('average_rating', 0)}/10

목표 플레이어 수: {target_players}명

다음 항목들을 JSON 형태로 분석해주세요:
1. player_feasibility: 목표 플레이어 수로 플레이 가능한지 (true/false)
2. complexity_level: 복잡도 수준 ("simple", "moderate", "complex", "expert")
3. recommended_ai_personas: AI 플레이어에게 적합한 성격 유형들 (리스트)
4. estimated_play_time: 예상 플레이 시간 (분)
5. key_mechanics: 핵심 메커니즘들 (리스트)
6. difficulty_factors: 게임의 어려운 요소들 (리스트)
7. strategic_depth: 전략적 깊이 평가 (1-10)
8. social_interaction: 플레이어 간 상호작용 수준 (1-10)

응답은 반드시 유효한 JSON 형태로 해주세요.
"""
        
        try:
            llm_response = await self.llm_client.complete(analysis_prompt)
            
            # JSON 파싱 시도
            try:
                analysis = json.loads(llm_response)
            except json.JSONDecodeError:
                # JSON 파싱 실패시 기본값 사용
                analysis = {
                    "player_feasibility": target_players >= game_details.get('min_players', 2) and target_players <= game_details.get('max_players', 8),
                    "complexity_level": "moderate",
                    "recommended_ai_personas": ["strategic", "casual", "aggressive"],
                    "estimated_play_time": game_details.get('playing_time', 60),
                    "key_mechanics": game_details.get('mechanics', []),
                    "difficulty_factors": ["rules_complexity", "strategic_depth"],
                    "strategic_depth": min(int(game_details.get('complexity', 2) * 2), 10),
                    "social_interaction": 5
                }
            
            return {
                "analysis_complete": True,
                "game_analysis": analysis,
                "raw_llm_response": llm_response,
                "confidence": "high" if perception.get("source") == "bgg" else "medium"
            }
            
        except Exception as e:
            return {
                "analysis_failed": True,
                "error": f"LLM 분석 실패: {str(e)}",
                "fallback_analysis": {
                    "complexity_level": "moderate",
                    "recommended_ai_personas": ["balanced"],
                    "estimated_play_time": 60
                }
            }
    
    async def act(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """
        게임 분석 결과를 정리하고 다음 단계 준비
        
        - 분석 결과 검증
        - 게임 설정 정보 생성
        - 에이전트 초기화 매개변수 준비
        """
        if reasoning.get("analysis_failed"):
            return {
                "action": "analysis_failed",
                "error": reasoning.get("error"),
                "recommendations": ["다른 게임 시도", "수동 설정 진행"]
            }
        
        analysis = reasoning.get("game_analysis", {})
        
        # 게임 설정 생성
        game_config = {
            "is_playable": analysis.get("player_feasibility", False),
            "complexity": analysis.get("complexity_level", "moderate"),
            "estimated_duration": analysis.get("estimated_play_time", 60),
            "ai_persona_suggestions": analysis.get("recommended_ai_personas", ["balanced"]),
            "strategic_depth": analysis.get("strategic_depth", 5),
            "social_interaction": analysis.get("social_interaction", 5),
            "key_mechanics": analysis.get("key_mechanics", []),
            "difficulty_factors": analysis.get("difficulty_factors", [])
        }
        
        # 다음 단계 에이전트들을 위한 매개변수
        next_steps = []
        
        if game_config["is_playable"]:
            next_steps.extend([
                {
                    "agent": "persona_generator",
                    "params": {
                        "personas_needed": len(analysis.get("recommended_ai_personas", [])),
                        "complexity": game_config["complexity"],
                        "suggested_types": analysis.get("recommended_ai_personas", [])
                    }
                },
                {
                    "agent": "rule_parser", 
                    "params": {
                        "complexity": game_config["complexity"],
                        "key_mechanics": game_config["key_mechanics"]
                    }
                }
            ])
        
        return {
            "action": "game_analysis_complete",
            "game_config": game_config,
            "next_steps": next_steps,
            "confidence": reasoning.get("confidence", "medium"),
            "timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()

    async def analyze_specific_game(self, game_name: str, player_count: int = 4) -> Dict[str, Any]:
        """
        특정 게임 분석을 위한 편의 메서드
        
        Args:
            game_name: 분석할 게임 이름
            player_count: 목표 플레이어 수
            
        Returns:
            게임 분석 결과
        """
        environment = {
            "game_name": game_name,
            "player_count": player_count
        }
        
        return await self.run_cycle(environment) 