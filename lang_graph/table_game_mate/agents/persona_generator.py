"""
페르소나 생성 에이전트

게임별 특성에 맞는 AI 플레이어 페르소나를 동적으로 생성하는 에이전트
"""

from typing import Dict, List, Any, Optional
import json
import random
import re
from ..core.agent_base import BaseAgent
from ..models.persona import PersonaArchetype, PersonaProfile, CommunicationStyle, PersonaTraits


class PersonaGeneratorAgent(BaseAgent):
    """
    AI 플레이어 페르소나 생성 전문 에이전트
    
    게임별 특성에 맞는 다양한 AI 성격 생성:
    - 전략적 깊이에 따른 성격 조정
    - 게임 메커니즘별 특화 성격
    - 플레이어 간 밸런스 고려
    - 게임 재미 극대화를 위한 다양성
    """
    
    def __init__(self, llm_client, mcp_client, agent_id: str = "persona_generator"):
        super().__init__(llm_client, mcp_client, agent_id)
        self.base_personas = self._load_base_personas()
        
    def _load_base_personas(self) -> Dict[str, Dict]:
        """기본 페르소나 템플릿 로딩"""
        return {
            "strategic": {
                "name": "전략가",
                "traits": ["analytical", "patient", "calculating"],
                "play_style": "long_term_planning",
                "risk_tolerance": "low",
                "interaction_style": "minimal"
            },
            "aggressive": {
                "name": "공격적 플레이어", 
                "traits": ["bold", "competitive", "impatient"],
                "play_style": "high_risk_high_reward",
                "risk_tolerance": "high",
                "interaction_style": "confrontational"
            },
            "casual": {
                "name": "여유로운 플레이어",
                "traits": ["friendly", "relaxed", "social"],
                "play_style": "fun_focused",
                "risk_tolerance": "medium",
                "interaction_style": "social"
            },
            "adaptive": {
                "name": "적응형 플레이어",
                "traits": ["flexible", "observant", "reactive"],
                "play_style": "situational",
                "risk_tolerance": "variable",
                "interaction_style": "responsive"
            },
            "chaotic": {
                "name": "무작위 플레이어",
                "traits": ["unpredictable", "spontaneous", "creative"],
                "play_style": "random",
                "risk_tolerance": "extreme",
                "interaction_style": "chaotic"
            }
        }
    
    async def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        페르소나 생성을 위한 환경 인식
        
        - 게임 복잡도 및 메커니즘 분석
        - 필요한 플레이어 수 확인
        - 게임 특성별 요구사항 파악
        """
        game_analysis = environment.get("game_analysis", {})
        personas_needed = environment.get("personas_needed", 3)
        complexity = environment.get("complexity", "moderate")
        suggested_types = environment.get("suggested_types", ["strategic", "casual", "aggressive"])
        
        # 게임 특성 분석
        strategic_depth = game_analysis.get("strategic_depth", 5)
        social_interaction = game_analysis.get("social_interaction", 5)
        key_mechanics = game_analysis.get("key_mechanics", [])
        
        return {
            "personas_required": personas_needed,
            "game_complexity": complexity,
            "strategic_depth": strategic_depth,
            "social_interaction": social_interaction,
            "key_mechanics": key_mechanics,
            "suggested_persona_types": suggested_types,
            "base_templates": self.base_personas
        }
    
    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 기반 페르소나 설계 및 최적화
        
        게임 특성을 바탕으로:
        - 각 페르소나의 성격 특성 설계
        - 게임별 전략 성향 할당
        - 플레이어 간 밸런스 조정
        - 상호작용 패턴 설계
        """
        personas_needed = perception.get("personas_required", 3)
        complexity = perception.get("game_complexity", "moderate")
        strategic_depth = perception.get("strategic_depth", 5)
        social_interaction = perception.get("social_interaction", 5)
        mechanics = perception.get("key_mechanics", [])
        suggested_types = perception.get("suggested_persona_types", [])
        
        # LLM에게 페르소나 생성 요청
        persona_prompt = f"""
게임별 AI 플레이어 페르소나 생성 요청:

게임 특성:
- 복잡도: {complexity}
- 전략적 깊이: {strategic_depth}/10
- 사회적 상호작용: {social_interaction}/10
- 핵심 메커니즘: {', '.join(mechanics)}

요구사항:
- 생성할 페르소나 수: {personas_needed}개
- 추천 유형: {', '.join(suggested_types)}

각 페르소나별로 다음 정보를 JSON 배열로 생성해주세요:
1. name: 페르소나 이름 (한국어)
2. type: 기본 유형 ({', '.join(self.base_personas.keys())})
3. personality_traits: 성격 특성 리스트
4. decision_style: 의사결정 스타일 설명
5. risk_preference: 위험 선호도 (1-10)
6. cooperation_tendency: 협력 성향 (1-10)
7. aggression_level: 공격성 수준 (1-10)
8. adaptability: 적응력 (1-10)
9. game_focus: 게임 집중 영역 (예: "resource_management", "player_elimination", "area_control")
10. catchphrase: 특징적인 말버릇 또는 철학

조건:
- 각 페르소나는 명확히 구별되어야 함
- 게임 밸런스를 위해 다양한 성향 조합
- 현재 게임 메커니즘에 적합한 특성 부여

응답은 반드시 유효한 JSON 배열 형태로 해주세요.
"""
        
        try:
            # LLM 클라이언트 사용 (엄격 모드, 폴백 없음)
            llm_response_str = await self.llm_client.complete(persona_prompt)
            
            # JSON 파싱 (실패 시 에러 반환)
            try:
                llm_response = llm_response_str
                personas_data = json.loads(llm_response)
                if not isinstance(personas_data, list):
                    raise ValueError("응답이 배열 형태가 아닙니다")
            except (json.JSONDecodeError, ValueError) as parse_err:
                return {"personas_generated": False, "error": f"LLM 응답 파싱 실패: {parse_err}"}
            
            # 페르소나 유효성 검증 및 보완
            validated_personas = []
            for i, persona_data in enumerate(personas_data[:personas_needed]):
                validated_persona = self._validate_and_complete_persona(persona_data, i)
                validated_personas.append(validated_persona)
            
            return {
                "personas_generated": True,
                "personas": validated_personas,
                "generation_method": "llm",
                "raw_llm_response": llm_response
            }
            
        except Exception as e:
            return {"personas_generated": False, "error": f"LLM 생성 실패: {str(e)}"}
    
    async def act(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """
        생성된 페르소나들을 PersonaProfile 객체로 변환하고 최종 검증
        
        - PersonaProfile 객체 생성
        - 페르소나 간 밸런스 검증
        - 게임 적합성 최종 확인
        """
        if not reasoning.get("personas_generated"):
            return {
                "action": "persona_generation_failed",
                "error": reasoning.get("error", "알 수 없는 오류")
            }
        
        personas_data = reasoning.get("personas", [])
        
        # PersonaProfile 객체들 생성 (실패 시 폴백 없이 에러 반환)
        persona_profiles = []
        for i, persona_data in enumerate(personas_data):
            try:
                archetype_str = persona_data.get("type", "social")
                profile = {
                    "name": persona_data.get("name", f"플레이어 {i+1}"),
                    "archetype": PersonaArchetype(archetype_str),
                    "communication_style": CommunicationStyle.FRIENDLY,
                    "traits": PersonaTraits(),
                    "background_story": persona_data.get("background_story", "게임을 즐기는 AI 플레이어입니다."),
                    "catchphrases": [persona_data.get("catchphrase", "좋은 게임이네요!")],
                    "game_behaviors": {},
                    "created_for_game": "test_game",
                    "difficulty_level": "medium"
                }
                persona_profiles.append(profile)
            except Exception as e:
                return {"action": "persona_generation_failed", "error": f"페르소나 변환 실패(index={i}): {e}"}
        
        # 페르소나 밸런스 분석
        balance_analysis = self._analyze_persona_balance(persona_profiles)
        
        return {
            "action": "personas_created",
            "persona_profiles": persona_profiles,
            "balance_analysis": balance_analysis,
            "generation_method": reasoning.get("generation_method", "unknown"),
            "personas_count": len(persona_profiles),
            "timestamp": self._get_timestamp()
        }
    
    # 폴백 페르소나 생성 로직 제거 (프로덕션 모드)
    
    def _validate_and_complete_persona(self, persona_data: Dict, index: int) -> Dict:
        """페르소나 데이터 유효성 검증 및 보완"""
        validated = {
            "name": persona_data.get("name", f"AI 플레이어 {index+1}"),
            "type": persona_data.get("type", "casual"),
            "personality_traits": persona_data.get("personality_traits", ["balanced"]),
            "decision_style": persona_data.get("decision_style", "균형잡힌 의사결정"),
            "risk_preference": max(1, min(10, persona_data.get("risk_preference", 5))),
            "cooperation_tendency": max(1, min(10, persona_data.get("cooperation_tendency", 5))),
            "aggression_level": max(1, min(10, persona_data.get("aggression_level", 5))),
            "adaptability": max(1, min(10, persona_data.get("adaptability", 5))),
            "game_focus": persona_data.get("game_focus", "general_strategy"),
            "catchphrase": persona_data.get("catchphrase", "재미있는 게임이네요!"),
            "communication_style": persona_data.get("communication_style"),
            "background_story": persona_data.get("background_story")
        }
        
        # 타입이 유효하지 않으면 에러 반환
        if validated["type"] not in self.base_personas:
            raise ValueError(f"유효하지 않은 페르소나 타입: {validated['type']}")
        
        return validated
    
    # 기본 페르소나 생성 로직 제거 (프로덕션 모드)
    
    def _analyze_persona_balance(self, personas: List[PersonaProfile]) -> Dict[str, Any]:
        """페르소나 간 밸런스 분석"""
        if not personas:
            return {"balanced": False, "reason": "페르소나가 없음"}
        
        # 각 특성별 평균 계산 (TypedDict 접근)
        avg_risk = sum(p["traits"].risk_tolerance for p in personas) / len(personas)
        avg_coop = sum(p["traits"].cooperation for p in personas) / len(personas)
        avg_aggr = sum(p["traits"].aggression for p in personas) / len(personas)
        avg_adapt = sum(p["traits"].adaptability for p in personas) / len(personas)
        
        # 다양성 측정 (표준편차)
        import statistics
        risk_diversity = statistics.stdev([p["traits"].risk_tolerance for p in personas]) if len(personas) > 1 else 0
        
        return {
            "balanced": risk_diversity > 1.5,  # 충분한 다양성
            "averages": {
                "risk_preference": avg_risk,
                "cooperation_tendency": avg_coop,
                "aggression_level": avg_aggr,
                "adaptability": avg_adapt
            },
            "diversity_score": risk_diversity,
            "persona_types": [p["archetype"].value for p in personas],
            "recommendations": self._get_balance_recommendations(avg_risk, avg_coop, avg_aggr, risk_diversity)
        }
    
    def _get_balance_recommendations(self, avg_risk: float, avg_coop: float, avg_aggr: float, diversity: float) -> List[str]:
        """밸런스 개선 권장사항"""
        recommendations = []
        
        if diversity < 1.5:
            recommendations.append("페르소나 간 다양성 부족 - 더 극단적인 성격 추가 권장")
        
        if avg_risk < 3:
            recommendations.append("전체적으로 너무 보수적 - 모험적인 플레이어 추가 권장")
        elif avg_risk > 7:
            recommendations.append("전체적으로 너무 위험선호 - 신중한 플레이어 추가 권장")
        
        if avg_aggr < 3:
            recommendations.append("게임 긴장감 부족 - 더 공격적인 플레이어 추가 권장")
        elif avg_aggr > 7:
            recommendations.append("과도한 갈등 예상 - 평화로운 플레이어 추가 권장")
        
        if not recommendations:
            recommendations.append("밸런스가 양호함")
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()

    async def generate_personas_for_game(self, game_analysis: Dict[str, Any], player_count: int = 3) -> Dict[str, Any]:
        """
        특정 게임을 위한 페르소나 생성 편의 메서드
        
        Args:
            game_analysis: 게임 분석 결과
            player_count: 생성할 AI 플레이어 수
            
        Returns:
            생성된 페르소나들과 분석 결과
        """
        environment = {
            "game_analysis": game_analysis,
            "personas_needed": player_count,
            "complexity": game_analysis.get("complexity_level", "moderate"),
            "suggested_types": game_analysis.get("recommended_ai_personas", ["strategic", "casual", "aggressive"])
        }
        
        return await self.run_cycle(environment) 