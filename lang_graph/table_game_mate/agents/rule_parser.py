"""
규칙 파서 에이전트

게임 규칙을 수집, 파싱, 해석하여 AI가 이해할 수 있는 형태로 변환하는 에이전트
"""

from typing import Dict, List, Any, Optional
import json
import re
from ..core.agent_base import BaseAgent


class RuleParserAgent(BaseAgent):
    """
    게임 규칙 파싱 및 해석 전문 에이전트
    
    다양한 소스에서 게임 규칙을 수집하고 구조화:
    - BGG 규칙 요약 파싱
    - PDF 규칙서 텍스트 추출
    - 온라인 규칙 데이터베이스 검색
    - AI가 이해할 수 있는 구조화된 규칙 생성
    """
    
    def __init__(self, llm_client, mcp_client, agent_id: str = "rule_parser"):
        super().__init__(llm_client, mcp_client, agent_id)
        self.supported_formats = ["bgg_summary", "pdf_text", "online_wiki", "manual_input"]
        
    async def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        규칙 파싱을 위한 환경 인식
        
        - 게임 ID/이름 확인
        - BGG에서 규칙 요약 수집
        - 추가 규칙 소스 탐색
        - 게임 복잡도 및 메커니즘 정보 수집
        """
        game_id = environment.get("game_id")
        game_name = environment.get("game_name")
        complexity = environment.get("complexity", "moderate")
        key_mechanics = environment.get("key_mechanics", [])
        
        if not game_id and not game_name:
            return {"error": "게임 ID 또는 이름이 필요합니다"}
        
        rules_data = {"sources": [], "content": {}}
        
        # BGG에서 규칙 요약 수집
        try:
            if game_id:
                game_details = await self.mcp_client.call(
                    "bgg_server",
                    "get_game_details",
                    {"game_id": game_id}
                )
                
                rules_summary = game_details.get("rules_summary", "")
                if rules_summary:
                    rules_data["sources"].append("bgg_summary")
                    rules_data["content"]["bgg_summary"] = rules_summary
                
                # 추가 메타데이터
                rules_data["content"]["mechanics"] = game_details.get("mechanics", [])
                rules_data["content"]["categories"] = game_details.get("categories", [])
                rules_data["content"]["player_count"] = {
                    "min": game_details.get("min_players", 2),
                    "max": game_details.get("max_players", 4)
                }
                rules_data["content"]["playing_time"] = game_details.get("playing_time", 60)
                
        except Exception as e:
            return {"error": f"BGG 규칙 수집 실패: {str(e)}"}
        
        # TODO: 추가 규칙 소스 연동
        # - PDF 규칙서 링크 확인
        # - BoardGameGeek Files 섹션 탐색
        # - 온라인 위키 검색
        
        return {
            "rules_found": len(rules_data["sources"]) > 0,
            "rules_data": rules_data,
            "game_complexity": complexity,
            "key_mechanics": key_mechanics,
            "parsing_needed": True
        }
    
    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 기반 규칙 해석 및 구조화
        
        수집된 규칙 텍스트를 바탕으로:
        - 핵심 규칙 추출 및 정리
        - 게임 플로우 파악
        - 승리 조건 명확화
        - AI 행동 가능한 액션 정의
        - 규칙 예외 상황 처리
        """
        if not perception.get("rules_found"):
            return {"parsing_failed": True, "error": "파싱할 규칙이 없습니다"}
        
        rules_data = perception.get("rules_data", {})
        complexity = perception.get("game_complexity", "moderate")
        mechanics = perception.get("key_mechanics", [])
        
        # 모든 규칙 소스 통합
        all_rules_text = ""
        for source, content in rules_data.get("content", {}).items():
            if isinstance(content, str) and content.strip():
                all_rules_text += f"\n=== {source.upper()} ===\n{content}\n"
        
        # LLM에게 규칙 구조화 요청
        parsing_prompt = f"""
게임 규칙 구조화 분석 요청:

게임 정보:
- 복잡도: {complexity}
- 핵심 메커니즘: {', '.join(mechanics)}
- 플레이어 수: {rules_data.get('content', {}).get('player_count', {})}
- 플레이 시간: {rules_data.get('content', {}).get('playing_time', 60)}분

규칙 텍스트:
{all_rules_text}

다음 구조로 규칙을 분석하고 JSON 형태로 정리해주세요:

1. setup: 게임 준비
   - components: 필요한 구성품
   - initial_setup: 초기 설정 방법

2. game_flow: 게임 진행
   - turn_structure: 턴 구조
   - phases: 각 페이즈별 행동
   - player_actions: 플레이어가 할 수 있는 행동들

3. win_conditions: 승리 조건
   - primary: 주요 승리 조건
   - alternative: 대안 승리 조건 (있다면)

4. special_rules: 특별 규칙
   - exceptions: 예외 상황들
   - clarifications: 명확화가 필요한 부분들

5. ai_guidance: AI 플레이어를 위한 가이드
   - decision_points: 주요 의사결정 포인트들
   - strategy_hints: 전략 힌트들
   - common_mistakes: 주의해야 할 실수들

응답은 반드시 유효한 JSON 형태로 해주세요.
"""
        
        try:
            llm_response = await self.llm_client.complete(parsing_prompt)
            
            # JSON 파싱 시도
            try:
                parsed_rules = json.loads(llm_response)
            except json.JSONDecodeError:
                # 파싱 실패시 기본 구조 생성
                parsed_rules = self._generate_fallback_rules(rules_data, mechanics)
            
            # 규칙 완성도 평가
            completeness_score = self._evaluate_rule_completeness(parsed_rules)
            
            return {
                "parsing_complete": True,
                "parsed_rules": parsed_rules,
                "completeness_score": completeness_score,
                "parsing_method": "llm" if isinstance(json.loads(llm_response), dict) else "fallback",
                "raw_llm_response": llm_response
            }
            
        except Exception as e:
            # 완전 실패시 기본 규칙 구조 사용
            fallback_rules = self._generate_fallback_rules(rules_data, mechanics)
            return {
                "parsing_complete": True,
                "parsed_rules": fallback_rules,
                "completeness_score": 0.3,
                "parsing_method": "fallback",
                "error": f"LLM 파싱 실패: {str(e)}"
            }
    
    async def act(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """
        파싱된 규칙을 검증하고 AI 플레이어용 액션 가이드 생성
        
        - 규칙 일관성 검증
        - AI 액션 매핑 생성
        - 게임 상태 추적 구조 설계
        - 규칙 예외 처리 로직 구현
        """
        if not reasoning.get("parsing_complete"):
            return {
                "action": "rule_parsing_failed",
                "error": reasoning.get("error", "알 수 없는 파싱 오류")
            }
        
        parsed_rules = reasoning.get("parsed_rules", {})
        completeness = reasoning.get("completeness_score", 0)
        
        # AI 액션 매핑 생성
        ai_action_map = self._generate_ai_action_map(parsed_rules)
        
        # 게임 상태 추적 구조 설계
        state_tracking = self._design_state_tracking(parsed_rules)
        
        # 규칙 검증 결과
        validation_results = self._validate_rules(parsed_rules)
        
        return {
            "action": "rules_processed",
            "structured_rules": parsed_rules,
            "ai_action_map": ai_action_map,
            "state_tracking": state_tracking,
            "validation_results": validation_results,
            "completeness_score": completeness,
            "parsing_method": reasoning.get("parsing_method", "unknown"),
            "ready_for_gameplay": completeness > 0.6,
            "timestamp": self._get_timestamp()
        }
    
    def _generate_fallback_rules(self, rules_data: Dict, mechanics: List[str]) -> Dict[str, Any]:
        """기본 규칙 구조 생성 (LLM 실패시)"""
        player_count = rules_data.get("content", {}).get("player_count", {"min": 2, "max": 4})
        
        return {
            "setup": {
                "components": ["게임 보드", "플레이어 말", "카드", "주사위"],
                "initial_setup": "각 플레이어는 시작 위치에 말을 놓고 카드를 받습니다."
            },
            "game_flow": {
                "turn_structure": "시계방향으로 순서대로 턴을 진행합니다.",
                "phases": ["행동 선택", "행동 실행", "결과 처리"],
                "player_actions": ["카드 플레이", "말 이동", "자원 수집"]
            },
            "win_conditions": {
                "primary": "목표 점수에 먼저 도달하는 플레이어가 승리",
                "alternative": []
            },
            "special_rules": {
                "exceptions": ["동점시 추가 라운드 진행"],
                "clarifications": []
            },
            "ai_guidance": {
                "decision_points": ["어떤 카드를 플레이할지", "어디로 이동할지"],
                "strategy_hints": ["균형잡힌 플레이가 중요"],
                "common_mistakes": ["성급한 행동", "장기적 계획 부족"]
            }
        }
    
    def _evaluate_rule_completeness(self, rules: Dict[str, Any]) -> float:
        """규칙 완성도 평가 (0.0 ~ 1.0)"""
        required_sections = ["setup", "game_flow", "win_conditions"]
        optional_sections = ["special_rules", "ai_guidance"]
        
        score = 0.0
        
        # 필수 섹션 체크 (70% 가중치)
        for section in required_sections:
            if section in rules and rules[section]:
                score += 0.7 / len(required_sections)
                
                # 세부 완성도 체크
                if section == "setup" and isinstance(rules[section], dict):
                    if "components" in rules[section] and "initial_setup" in rules[section]:
                        score += 0.1
                elif section == "game_flow" and isinstance(rules[section], dict):
                    if "turn_structure" in rules[section] and "player_actions" in rules[section]:
                        score += 0.1
                elif section == "win_conditions" and isinstance(rules[section], dict):
                    if "primary" in rules[section]:
                        score += 0.1
        
        # 선택 섹션 체크 (30% 가중치)
        for section in optional_sections:
            if section in rules and rules[section]:
                score += 0.3 / len(optional_sections)
        
        return min(1.0, score)
    
    def _generate_ai_action_map(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        """AI 플레이어용 액션 매핑 생성"""
        game_flow = rules.get("game_flow", {})
        player_actions = game_flow.get("player_actions", [])
        
        action_map = {
            "available_actions": [],
            "action_prerequisites": {},
            "action_effects": {},
            "decision_tree": {}
        }
        
        # 기본 액션들 매핑
        for action in player_actions:
            if isinstance(action, str):
                action_key = action.lower().replace(" ", "_")
                action_map["available_actions"].append(action_key)
                action_map["action_prerequisites"][action_key] = []
                action_map["action_effects"][action_key] = "게임 상태 변경"
        
        # 의사결정 트리 기본 구조
        ai_guidance = rules.get("ai_guidance", {})
        decision_points = ai_guidance.get("decision_points", [])
        
        for i, decision in enumerate(decision_points):
            action_map["decision_tree"][f"decision_{i+1}"] = {
                "description": decision,
                "options": action_map["available_actions"],
                "evaluation_criteria": ["점수 향상", "리스크 최소화", "장기 전략"]
            }
        
        return action_map
    
    def _design_state_tracking(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        """게임 상태 추적 구조 설계"""
        setup = rules.get("setup", {})
        game_flow = rules.get("game_flow", {})
        
        return {
            "game_state": {
                "current_player": "player_id",
                "turn_number": 0,
                "phase": "setup",
                "game_board": {},
                "shared_resources": {}
            },
            "player_state": {
                "player_id": "string",
                "score": 0,
                "resources": {},
                "cards": [],
                "position": {},
                "status_effects": []
            },
            "tracked_events": [
                "turn_start",
                "action_performed", 
                "resource_change",
                "score_change",
                "game_end"
            ],
            "validation_rules": [
                "turn_order_enforcement",
                "action_legality_check",
                "resource_constraints",
                "win_condition_check"
            ]
        }
    
    def _validate_rules(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        """규칙 일관성 및 완성도 검증"""
        issues = []
        warnings = []
        
        # 필수 섹션 체크
        required_sections = ["setup", "game_flow", "win_conditions"]
        for section in required_sections:
            if section not in rules or not rules[section]:
                issues.append(f"필수 섹션 누락: {section}")
        
        # 승리 조건 검증
        win_conditions = rules.get("win_conditions", {})
        if not win_conditions.get("primary"):
            issues.append("주요 승리 조건이 명시되지 않음")
        
        # 플레이어 액션 검증
        game_flow = rules.get("game_flow", {})
        player_actions = game_flow.get("player_actions", [])
        if not player_actions:
            warnings.append("플레이어 액션이 명시되지 않음")
        
        # AI 가이드 검증
        ai_guidance = rules.get("ai_guidance", {})
        if not ai_guidance.get("decision_points"):
            warnings.append("AI 의사결정 가이드가 부족함")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "completeness": "high" if len(issues) == 0 and len(warnings) < 2 else "medium" if len(issues) == 0 else "low"
        }
    
    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()

    async def parse_game_rules(self, game_id: str = None, game_name: str = None, complexity: str = "moderate") -> Dict[str, Any]:
        """
        특정 게임의 규칙 파싱을 위한 편의 메서드
        
        Args:
            game_id: BGG 게임 ID
            game_name: 게임 이름
            complexity: 게임 복잡도
            
        Returns:
            파싱된 규칙과 분석 결과
        """
        environment = {
            "game_id": game_id,
            "game_name": game_name,
            "complexity": complexity
        }
        
        return await self.run_cycle(environment) 