"""
플레이어 에이전트

개별 AI 플레이어의 의사결정을 담당하는 에이전트
각 플레이어는 고유한 페르소나와 전략을 가지고 게임에 참여합니다.
"""

from typing import Dict, List, Any, Optional
import json
import random
from datetime import datetime

from ..core.agent_base import BaseAgent
from ..core.action_executor import ActionExecutor
from ..models.game_state import GameState, GamePhase
from ..models.persona import PersonaProfile, PersonaArchetype
from ..models.action import Action, ActionType, ActionContext
from ..models.llm import LLMResponse, ParsedLLMResponse
from ..models.game_state import GameAction, PlayerInfo


class PlayerAgent(BaseAgent):
    """
    개별 플레이어 에이전트
    
    각 AI 플레이어의 개별적인 의사결정을 담당:
    - 페르소나 기반 행동 결정
    - 게임 상황 분석
    - 전략적 선택
    - 다른 플레이어와의 상호작용
    """
    
    def __init__(self, llm_client, mcp_client, player_info: PlayerInfo, persona: PersonaProfile, agent_id: str = None):
        """
        Args:
            llm_client: LLM 클라이언트
            mcp_client: MCP 클라이언트
            player_info: 플레이어 정보
            persona: 플레이어 페르소나
            agent_id: 에이전트 ID (선택사항)
        """
        if agent_id is None:
            agent_id = f"player_{player_info.id}"
        
        super().__init__(llm_client, mcp_client, agent_id)
        
        self.player_info = player_info
        self.persona = persona
        self.game_memory = {
            "decisions_made": [],
            "interactions": [],
            "observations": [],
            "strategy_adjustments": []
        }
        self.current_strategy = self._initialize_strategy()
    
    def _initialize_strategy(self) -> Dict[str, Any]:
        """페르소나 기반 초기 전략 설정"""
        # persona가 dict인 경우 처리
        if isinstance(self.persona, dict):
            traits_obj = self.persona.get("traits")
            base_strategy = {
                "risk_tolerance": getattr(traits_obj, "risk_tolerance", 0.5),
                "cooperation_level": getattr(traits_obj, "cooperation", 0.5),
                "aggression_level": getattr(traits_obj, "aggression", 0.3),
                "adaptability": getattr(traits_obj, "adaptability", 0.5),
                "focus_areas": ["general"],
                "decision_style": "balanced",
            }
            
            # 페르소나 타입별 특화 전략
            arch = self.persona.get("archetype")
            persona_type = arch.value if hasattr(arch, "value") else (arch or "strategic")
            if persona_type == "aggressive":
                base_strategy.update({
                    "preferred_actions": ["attack", "challenge", "compete"],
                    "avoid_actions": ["cooperate", "defend", "wait"],
                    "interaction_style": "confrontational"
                })
            elif persona_type == "strategic":
                base_strategy.update({
                    "preferred_actions": ["analyze", "plan", "optimize"],
                    "avoid_actions": ["random", "impulsive"],
                    "interaction_style": "calculated"
                })
            elif persona_type == "social":
                base_strategy.update({
                    "preferred_actions": ["cooperate", "negotiate", "alliance"],
                    "avoid_actions": ["isolate", "betray"],
                    "interaction_style": "collaborative"
                })
            else:  # 기본값
                base_strategy.update({
                    "preferred_actions": ["observe", "calculate", "precise"],
                    "avoid_actions": ["hasty", "emotional"],
                    "interaction_style": "methodical"
                })
        else:
            # 기존 객체 기반 로직 (호환성 유지)
            base_strategy = {
                "risk_tolerance": getattr(self.persona, 'risk_preference', 5) / 10.0,
                "cooperation_level": getattr(self.persona, 'cooperation_tendency', 5) / 10.0,
                "aggression_level": getattr(self.persona, 'aggression_level', 3) / 10.0,
                "adaptability": getattr(self.persona, 'adaptability', 5) / 10.0,
                "focus_areas": [getattr(self.persona, 'game_focus', 'general')],
                "decision_style": getattr(self.persona, 'decision_style', 'balanced')
            }
            
            # 페르소나 타입별 특화 전략
            archetype = getattr(self.persona, 'archetype', PersonaArchetype.STRATEGIC)
            if archetype == PersonaArchetype.AGGRESSIVE:
                base_strategy.update({
                    "preferred_actions": ["attack", "challenge", "compete"],
                    "avoid_actions": ["cooperate", "defend", "wait"],
                    "interaction_style": "confrontational"
                })
            elif archetype == PersonaArchetype.STRATEGIC:
                base_strategy.update({
                    "preferred_actions": ["analyze", "plan", "optimize"],
                    "avoid_actions": ["random", "impulsive"],
                    "interaction_style": "calculated"
                })
            elif archetype == PersonaArchetype.SOCIAL:
                base_strategy.update({
                    "preferred_actions": ["cooperate", "negotiate", "alliance"],
                    "avoid_actions": ["isolate", "betray"],
                    "interaction_style": "collaborative"
                })
            else:  # 기본값
                base_strategy.update({
                    "preferred_actions": ["observe", "calculate", "precise"],
                    "avoid_actions": ["hasty", "emotional"],
                    "interaction_style": "methodical"
                })
        
        return base_strategy
    
    async def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        게임 환경 인식 - 플레이어 관점에서
        
        - 현재 게임 상태 파악
        - 다른 플레이어 행동 관찰
        - 사용 가능한 행동 옵션 확인
        - 승리 조건 대비 현재 위치 평가
        """
        game_state = environment.get("game_state", {})
        parsed_rules = environment.get("parsed_rules", {})
        other_players = environment.get("other_players", [])
        
        # 현재 상황 분석
        current_situation = {
            "my_turn": environment.get("is_my_turn", False),
            "game_phase": game_state.get("phase", "unknown"),
            "turn_count": game_state.get("turn_count", 0),
            "my_score": self.player_info.score,
            "my_position": self._analyze_my_position(game_state),
            "available_actions": self._identify_available_actions(parsed_rules, game_state),
            "other_players_status": self._analyze_other_players(other_players),
            "winning_probability": self._estimate_winning_chance(game_state),
            "immediate_threats": self._identify_threats(game_state, other_players),
            "opportunities": self._identify_opportunities(game_state, parsed_rules)
        }
        
        # 관찰 기록
        observation = {
            "timestamp": datetime.now().isoformat(),
            "situation": current_situation,
            "emotional_state": self._assess_emotional_state(current_situation)
        }
        
        self.game_memory["observations"].append(observation)
        
        return {
            "situation_understood": True,
            "current_situation": current_situation,
            "perception_confidence": self._calculate_perception_confidence(current_situation),
            "needs_more_info": self._check_if_need_more_info(current_situation)
        }
    
    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        페르소나 기반 의사결정 추론
        
        - 현재 상황에 대한 페르소나별 해석
        - 가능한 행동들의 평가
        - 최적 행동 선택
        - 장기 전략과의 일치성 검토
        """
        if not perception.get("situation_understood"):
            return {"decision": "wait_for_more_info", "confidence": 0.1}
        
        current_situation = perception["current_situation"]
        
        # 페르소나 기반 상황 해석
        situation_interpretation = await self._interpret_situation_with_persona(current_situation)
        
        # 가능한 행동들 평가 및 최적 행동 선택 (엄격 모드)
        available_actions = current_situation.get("available_actions", [])
        try:
            action_evaluations = await self._evaluate_actions_with_llm(
                available_actions,
                current_situation,
                situation_interpretation,
            )
            best_action = self._select_best_action(action_evaluations)
        except Exception as e:
            return {
                "decision": "no_valid_action",
                "error": f"행동 평가/선택 실패: {e}",
                "confidence": 0.0
            }
        
        # 전략 조정 필요성 검토
        strategy_adjustment = self._check_strategy_adjustment(current_situation, best_action)
        
        reasoning_result = {
            "chosen_action": best_action,
            "action_reasoning": action_evaluations.get(best_action, {}),
            "situation_interpretation": situation_interpretation,
            "strategy_adjustment": strategy_adjustment,
            "confidence": self._calculate_decision_confidence(best_action, action_evaluations),
            "alternative_actions": list(action_evaluations.keys())[:3]  # 상위 3개 대안
        }
        
        return reasoning_result
    
    async def act(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """
        결정된 행동 실행
        
        - 선택된 행동을 게임 액션으로 변환
        - 행동 실행 전 최종 검증
        - 행동 기록 및 학습
        """
        chosen_action = reasoning.get("chosen_action")
        
        if not chosen_action:
            return {
                "action": "no_action_decided",
                "error": "의사결정 실패"
            }
        
        # 게임 액션으로 변환
        game_action = self._convert_to_game_action(chosen_action, reasoning)
        
        # 최종 검증
        validation_result = self._validate_action_before_execution(game_action)
        
        if not validation_result["is_valid"]:
            # 대안 행동 시도 (실패 시 폴백 없음)
            alternative_actions = reasoning.get("alternative_actions", [])
            for alt_action in alternative_actions:
                alt_game_action = self._convert_to_game_action(alt_action, reasoning)
                alt_validation = self._validate_action_before_execution(alt_game_action)
                if alt_validation["is_valid"]:
                    game_action = alt_game_action
                    chosen_action = alt_action
                    break
            else:
                return {
                    "action": "no_valid_action",
                    "error": "검증 가능한 대안 행동이 없습니다",
                    "timestamp": datetime.now().isoformat()
                }
        
        # 행동 기록
        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "chosen_action": chosen_action,
            "reasoning": reasoning.get("action_reasoning", {}),
            "confidence": reasoning.get("confidence", 0.5),
            "game_action": game_action
        }
        
        self.game_memory["decisions_made"].append(decision_record)
        
        # 전략 조정 적용
        if reasoning.get("strategy_adjustment"):
            self._apply_strategy_adjustment(reasoning["strategy_adjustment"])
        
        return {
            "action": "turn_completed",
            "action_type": game_action["action_type"],
            "action_data": game_action["action_data"],
            "action_description": self._generate_action_description(chosen_action, game_action),
            "confidence": reasoning.get("confidence", 0.5),
            "persona_influence": self._describe_persona_influence(reasoning),
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_my_position(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """내 현재 위치/상태 분석"""
        players = game_state.get("players", [])
        my_data = next((p for p in players if p.get("id") == self.player_info.id), {})
        
        if not players:
            return {"rank": "unknown", "relative_score": 0}
        
        scores = [p.get("score", 0) for p in players]
        my_score = my_data.get("score", 0)
        
        # 순위 계산
        sorted_scores = sorted(scores, reverse=True)
        my_rank = sorted_scores.index(my_score) + 1 if my_score in sorted_scores else len(players)
        
        return {
            "rank": my_rank,
            "total_players": len(players),
            "score": my_score,
            "score_gap_to_leader": max(scores) - my_score if scores else 0,
            "relative_performance": "leading" if my_rank == 1 else "trailing" if my_rank == len(players) else "middle"
        }
    
    def _identify_available_actions(self, parsed_rules: Dict[str, Any], game_state: Dict[str, Any]) -> List[str]:
        """사용 가능한 행동 식별"""
        base_actions = parsed_rules.get("actions", ["basic_move", "pass"])
        
        # 게임 상태에 따른 추가 행동
        phase = game_state.get("phase", "")
        if "setup" in phase:
            base_actions.extend(["choose_position", "select_resource"])
        elif "player_turn" in phase:
            base_actions.extend(["main_action", "bonus_action"])
        
        return base_actions
    
    def _analyze_other_players(self, other_players: List[Dict[str, Any]]) -> Dict[str, Any]:
        """다른 플레이어들 분석"""
        if not other_players:
            return {"threat_level": "unknown", "alliance_potential": []}
        
        analysis = {
            "total_opponents": len(other_players),
            "threat_levels": {},
            "alliance_candidates": [],
            "leader": None,
            "weakest": None
        }
        
        scores = [(p.get("id"), p.get("score", 0)) for p in other_players]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if scores:
            analysis["leader"] = scores[0][0]
            analysis["weakest"] = scores[-1][0]
        
        # 위협 수준 평가
        my_score = self.player_info.score
        for player in other_players:
            player_id = player.get("id")
            player_score = player.get("score", 0)
            
            if player_score > my_score * 1.2:
                analysis["threat_levels"][player_id] = "high"
            elif player_score > my_score:
                analysis["threat_levels"][player_id] = "medium"
            else:
                analysis["threat_levels"][player_id] = "low"
                analysis["alliance_candidates"].append(player_id)
        
        return analysis
    
    def _estimate_winning_chance(self, game_state: Dict[str, Any]) -> float:
        """승리 확률 추정"""
        players = game_state.get("players", [])
        if not players:
            return 0.5
        
        my_score = self.player_info.score
        all_scores = [p.get("score", 0) for p in players]
        
        if not all_scores:
            return 1.0 / len(players)  # 균등 확률
        
        max_score = max(all_scores)
        min_score = min(all_scores)
        
        if max_score == min_score:
            return 1.0 / len(players)
        
        # 정규화된 점수로 확률 계산
        normalized_score = (my_score - min_score) / (max_score - min_score)
        
        # 게임 진행도에 따른 조정
        turn_count = game_state.get("turn_count", 1)
        game_progress = min(turn_count / 10.0, 1.0)  # 10턴을 기준으로 진행도 계산
        
        # 초기에는 확률이 더 균등하고, 후반으로 갈수록 점수가 더 중요해짐
        base_probability = 1.0 / len(players)
        score_influence = normalized_score * game_progress
        
        return base_probability + (score_influence - base_probability) * 0.7
    
    def _identify_threats(self, game_state: Dict[str, Any], other_players: List[Dict[str, Any]]) -> List[str]:
        """즉각적인 위협 식별"""
        threats = []
        
        # 점수 기반 위협
        my_score = self.player_info.score
        for player in other_players:
            if player.get("score", 0) > my_score * 1.3:
                threats.append(f"score_threat_{player.get('id')}")
        
        return threats
    
    def _identify_opportunities(self, game_state: Dict[str, Any], parsed_rules: Dict[str, Any]) -> List[str]:
        """기회 요소 식별"""
        opportunities = []
        
        # 점수 격차가 작으면 역전 기회
        players = game_state.get("players", [])
        if players:
            scores = [p.get("score", 0) for p in players]
            score_range = max(scores) - min(scores)
            if score_range < 5:  # 작은 격차
                opportunities.append("close_game_opportunity")
        
        # 특수 규칙 기반 기회
        special_mechanics = parsed_rules.get("special_mechanics", [])
        for mechanic in special_mechanics:
            if "bonus" in mechanic.lower() or "extra" in mechanic.lower():
                opportunities.append(f"mechanic_opportunity_{mechanic}")
        
        return opportunities
    
    def _assess_emotional_state(self, situation: Dict[str, Any]) -> str:
        """현재 감정 상태 평가"""
        winning_prob = situation.get("winning_probability", 0.5)
        threats = situation.get("immediate_threats", [])
        position = situation.get("my_position", {})
        
        if winning_prob > 0.7:
            return "confident"
        elif winning_prob < 0.3:
            return "concerned"
        elif len(threats) > 2:
            return "defensive"
        elif position.get("rank", 1) == 1:
            return "satisfied"
        else:
            return "focused"
    
    def _calculate_perception_confidence(self, situation: Dict[str, Any]) -> float:
        """인식 신뢰도 계산"""
        base_confidence = 0.7
        
        # 정보 완성도에 따른 조정
        if situation.get("available_actions"):
            base_confidence += 0.1
        if situation.get("other_players_status"):
            base_confidence += 0.1
        if situation.get("winning_probability", 0) > 0:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _check_if_need_more_info(self, situation: Dict[str, Any]) -> bool:
        """추가 정보 필요성 검사"""
        # 기본 정보가 부족하면 더 필요
        if not situation.get("available_actions"):
            return True
        if situation.get("winning_probability", 0) == 0:
            return True
        
        return False
    
    async def _interpret_situation_with_persona(self, situation: Dict[str, Any]) -> Dict[str, Any]:
        """페르소나 기반 상황 해석"""
        interpretation = {
            "situation_assessment": "",
            "emotional_response": "",
            "strategic_priority": "",
            "risk_assessment": ""
        }
        
        winning_prob = situation.get("winning_probability", 0.5)
        my_position = situation.get("my_position", {})
        threats = situation.get("immediate_threats", [])
        
        # 페르소나별 해석
        persona_type = self.persona.get("persona_type", "strategic") if isinstance(self.persona, dict) else getattr(self.persona, 'archetype', PersonaArchetype.STRATEGIC)
        
        if persona_type == "aggressive" or persona_type == PersonaArchetype.AGGRESSIVE:
            if winning_prob < 0.4:
                interpretation["situation_assessment"] = "도전적인 상황, 공격적 행동 필요"
                interpretation["strategic_priority"] = "적극적 공격"
            else:
                interpretation["situation_assessment"] = "우위 유지, 압박 지속"
                interpretation["strategic_priority"] = "주도권 유지"
        
        elif persona_type == "strategic" or persona_type == PersonaArchetype.STRATEGIC:
            interpretation["situation_assessment"] = f"현재 {my_position.get('rank', '?')}위, 분석적 접근 필요"
            interpretation["strategic_priority"] = "최적화된 선택"
            interpretation["risk_assessment"] = "계산된 위험만 감수"
        
        elif persona_type == "social" or persona_type == PersonaArchetype.SOCIAL:
            if len(threats) > 1:
                interpretation["situation_assessment"] = "협력이 필요한 상황"
                interpretation["strategic_priority"] = "동맹 구축"
            else:
                interpretation["situation_assessment"] = "안정적 관계 유지"
                interpretation["strategic_priority"] = "상호 이익"
        
        else:  # 기본값
            interpretation["situation_assessment"] = "데이터 기반 현황 분석 완료"
            interpretation["strategic_priority"] = "논리적 최선책"
            interpretation["risk_assessment"] = "확률적 접근"
        
        return interpretation
    
    async def _evaluate_actions_with_llm(
        self,
        available_actions: List[str],
        situation: Dict[str, Any],
        interpretation: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """LLM을 사용한 행동 평가 (폴백 없이 엄격 모드)"""

        if not available_actions:
            raise ValueError("사용 가능한 행동이 없습니다")

        if not self.llm_client:
            raise RuntimeError("LLM 클라이언트가 설정되지 않았습니다")

        persona_type = (
            self.persona.get("persona_type", "strategic")
            if isinstance(self.persona, dict)
            else getattr(self.persona, "archetype", PersonaArchetype.STRATEGIC)
        )

        prompt = (
            "당신은 보드게임 플레이어 AI입니다. 주어진 상황과 페르소나에 따라 가능한 행동을 평가하세요.\n"
            "반드시 다음 JSON 형식만 반환하세요 (키는 행동명):\n"
            "{\n"
            "  \"<action_name>\": {\n"
            "    \"score\": 0.0-1.0, \"persona_alignment\": 0.0-1.0, \"risk_level\": \"low|medium|high\", \"reasoning\": \"...\"\n"
            "  }\n"
            "}\n"
            "available_actions: "
            + json.dumps(available_actions, ensure_ascii=False)
            + "\n"
            + "situation: "
            + json.dumps(situation, ensure_ascii=False)
            + "\n"
            + f"persona: {str(persona_type)}\n"
        )

        llm_response = await self.llm_client.complete(prompt)
        try:
            evaluations = json.loads(llm_response)
        except json.JSONDecodeError as je:
            raise ValueError(f"LLM 응답 JSON 파싱 실패: {je}")

        # 유효성 검사 및 최종 점수 계산
        for act, eval_data in list(evaluations.items()):
            if act not in available_actions or not isinstance(eval_data, dict):
                evaluations.pop(act, None)
                continue
            base_score = float(eval_data.get("score", 0.0))
            persona_alignment = float(eval_data.get("persona_alignment", 0.5))
            risk_level = str(eval_data.get("risk_level", "medium"))
            eval_data["final_score"] = self._adjust_score_for_persona(
                base_score, persona_alignment, risk_level
            )

        if not evaluations:
            raise ValueError("유효한 행동 평가가 없습니다")

        return evaluations
    
    def _adjust_score_for_persona(self, base_score: float, persona_alignment: float, risk_level: str) -> float:
        """페르소나 특성을 반영한 점수 조정"""
        adjusted_score = base_score
        
        # 페르소나 일치도 반영
        adjusted_score = adjusted_score * 0.7 + persona_alignment * 0.3
        
        # 위험 성향 반영
        risk_multiplier = 1.0
        if risk_level == "high":
            risk_multiplier = self.current_strategy["risk_tolerance"]
        elif risk_level == "low":
            risk_multiplier = 1.0 + (1.0 - self.current_strategy["risk_tolerance"]) * 0.2
        
        adjusted_score *= risk_multiplier
        
        return max(0.0, min(1.0, adjusted_score))
    
    # 폴백 평가 로직 제거 (프로덕션 모드)
    
    def _select_best_action(self, action_evaluations: Dict[str, Dict[str, Any]]) -> str:
        """최적 행동 선택"""
        if not action_evaluations:
            raise ValueError("행동 평가가 비어 있습니다")
        
        # final_score 기준으로 정렬
        sorted_actions = sorted(
            action_evaluations.items(),
            key=lambda x: x[1].get("final_score", x[1].get("score", 0)),
            reverse=True
        )
        
        # 약간의 랜덤성 추가 (페르소나의 adaptability 반영)
        traits_obj = (
            self.persona.get("traits") if isinstance(self.persona, dict) else getattr(self.persona, "traits", None)
        )
        adaptability_value = getattr(traits_obj, "adaptability", 0.0) if traits_obj else 0.0
        adaptability = adaptability_value  # already 0.0-1.0 scale
        if random.random() < adaptability * 0.3:  # 최대 30% 확률로 두 번째 선택
            if len(sorted_actions) > 1:
                return sorted_actions[1][0]
        
        return sorted_actions[0][0]
    
    def _check_strategy_adjustment(self, situation: Dict[str, Any], chosen_action: str) -> Optional[Dict[str, Any]]:
        """전략 조정 필요성 검토"""
        winning_prob = situation.get("winning_probability", 0.5)
        my_rank = situation.get("my_position", {}).get("rank", 1)
        
        adjustments = {}
        
        # 성과가 좋지 않으면 더 공격적으로
        if winning_prob < 0.3 and my_rank > 2:
            adjustments["risk_tolerance"] = min(1.0, self.current_strategy["risk_tolerance"] + 0.2)
            adjustments["aggression_level"] = min(1.0, self.current_strategy["aggression_level"] + 0.1)
        
        # 선두라면 안정적으로
        elif my_rank == 1 and winning_prob > 0.6:
            adjustments["risk_tolerance"] = max(0.0, self.current_strategy["risk_tolerance"] - 0.1)
            adjustments["cooperation_level"] = min(1.0, self.current_strategy["cooperation_level"] + 0.1)
        
        return adjustments if adjustments else None
    
    def _calculate_decision_confidence(self, chosen_action: str, evaluations: Dict[str, Dict[str, Any]]) -> float:
        """의사결정 신뢰도 계산"""
        if not evaluations or chosen_action not in evaluations:
            return 0.5
        
        chosen_score = evaluations[chosen_action].get("final_score", 0.5)
        all_scores = [eval_data.get("final_score", 0.5) for eval_data in evaluations.values()]
        
        if len(all_scores) <= 1:
            return chosen_score
        
        # 선택된 행동과 다른 행동들 간의 점수 차이
        other_scores = [score for score in all_scores if score != chosen_score]
        max_other_score = max(other_scores) if other_scores else 0
        
        score_gap = chosen_score - max_other_score
        confidence = 0.5 + score_gap * 0.5  # 점수 차이를 신뢰도로 변환
        
        return max(0.1, min(1.0, confidence))
    
    def _convert_to_game_action(self, chosen_action: str, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """선택된 행동을 게임 액션으로 변환"""
        if isinstance(self.persona, dict):
            arch = self.persona.get("archetype")
            persona_archetype_value = arch.value if hasattr(arch, "value") else str(arch)
        else:
            persona_archetype_value = getattr(self.persona, "archetype", PersonaArchetype.STRATEGIC).value
        return {
            "action_type": chosen_action,
            "action_data": {
                "player_id": self.player_info.id,
                "player_name": self.player_info.name,
                "persona_type": persona_archetype_value,
                "confidence": reasoning.get("confidence", 0.5),
                "reasoning_summary": reasoning.get("action_reasoning", {}).get("reasoning", ""),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _validate_action_before_execution(self, game_action: Dict[str, Any]) -> Dict[str, Any]:
        """행동 실행 전 검증"""
        # 기본 검증
        if not game_action.get("action_type"):
            return {"is_valid": False, "reason": "행동 타입 없음"}
        
        if not game_action.get("action_data", {}).get("player_id"):
            return {"is_valid": False, "reason": "플레이어 ID 없음"}
        
        return {"is_valid": True}
    
    def _create_fallback_action(self) -> Dict[str, Any]:
        """폴백 행동 생성"""
        return {
            "action_type": "pass",
            "action_data": {
                "player_id": self.player_info.id,
                "player_name": self.player_info.name,
                "reason": "fallback_action",
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _apply_strategy_adjustment(self, adjustments: Dict[str, Any]):
        """전략 조정 적용"""
        for key, value in adjustments.items():
            if key in self.current_strategy:
                old_value = self.current_strategy[key]
                self.current_strategy[key] = value
                
                # 조정 기록
                self.game_memory["strategy_adjustments"].append({
                    "timestamp": datetime.now().isoformat(),
                    "adjustment": {key: {"from": old_value, "to": value}},
                    "reason": "performance_based_adjustment"
                })
    
    def _generate_action_description(self, chosen_action: str, game_action: Dict[str, Any]) -> str:
        """행동 설명 생성"""
        base_description = f"{self.player_info.name}이(가) {chosen_action}을(를) 선택했습니다"
        
        # 페르소나 특성 반영
        if isinstance(self.persona, dict):
            catchphrases = self.persona.get("catchphrases", [])
        else:
            catchphrases = getattr(self.persona, "catchphrases", [])
        if catchphrases and random.random() < 0.3:
            base_description += f". \"{catchphrases[0]}\""
        
        return base_description
    
    def _describe_persona_influence(self, reasoning: Dict[str, Any]) -> str:
        """페르소나 영향 설명"""
        interpretation = reasoning.get("situation_interpretation", {})
        strategic_priority = interpretation.get("strategic_priority", "")
        if isinstance(self.persona, dict):
            arch = self.persona.get("archetype")
            persona_arch_value = arch.value if hasattr(arch, "value") else str(arch)
        else:
            persona_arch_value = getattr(self.persona, "archetype", PersonaArchetype.STRATEGIC).value
        return f"{persona_arch_value} 성향으로 {strategic_priority} 중심의 판단"
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """게임 메모리 요약"""
        return {
            "total_decisions": len(self.game_memory["decisions_made"]),
            "total_observations": len(self.game_memory["observations"]),
            "strategy_adjustments": len(self.game_memory["strategy_adjustments"]),
            "current_strategy": self.current_strategy,
            "recent_decisions": self.game_memory["decisions_made"][-3:],  # 최근 3개
            "persona_summary": {
                "type": self.persona.archetype.value,
                "traits": self.persona.personality_traits,
                "decision_style": self.persona.decision_style
            }
        }
    
    async def handle_interaction(self, interaction_type: str, other_player_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """다른 플레이어와의 상호작용 처리"""
        interaction_record = {
            "timestamp": datetime.now().isoformat(),
            "type": interaction_type,
            "other_player": other_player_id,
            "data": interaction_data
        }
        
        self.game_memory["interactions"].append(interaction_record)
        
        # 페르소나 기반 상호작용 응답
        if self.persona.archetype == PersonaArchetype.SOCIAL:
            response_style = "cooperative"
        elif self.persona.archetype == PersonaArchetype.AGGRESSIVE:
            response_style = "competitive"
        elif self.persona.archetype == PersonaArchetype.DECEPTIVE:
            response_style = "strategic"
        else:
            response_style = "neutral"
        
        return {
            "interaction_response": f"{response_style} 방식으로 응답",
            "response_data": {
                "style": response_style,
                "persona_influence": self.persona.archetype.value
            }
        } 