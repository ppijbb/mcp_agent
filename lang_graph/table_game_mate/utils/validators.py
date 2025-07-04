"""
검증 도구 - Agent 및 게임 품질 보장

이 모듈은 Agent의 행동과 게임 진행의 품질을 검증하고 보장합니다.
Agent가 올바르게 동작하고 게임이 규칙에 맞게 진행되는지 확인합니다.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import asyncio
import time
from datetime import datetime, timedelta

from ..models.game_state import GameState, GamePhase, PlayerInfo, GameAction
from ..models.persona import PersonaProfile, PersonaArchetype
from ..models.action import Action, ActionType, ActionContext
from ..models.llm import LLMResponse, ParsedLLMResponse


class ValidationLevel(Enum):
    """검증 레벨"""
    BASIC = "basic"      # 기본 검증
    STANDARD = "standard"  # 표준 검증
    STRICT = "strict"    # 엄격한 검증
    DEBUG = "debug"      # 디버그 모드


@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    level: ValidationLevel
    issues: List[str]
    warnings: List[str]
    score: float  # 0-100
    timestamp: datetime
    validation_time: float  # 초
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AgentValidator:
    """
    Agent 검증기 - Agent의 행동과 품질을 검증
    
    Agent가 올바르게 동작하는지, 예상된 행동을 하는지,
    성능이 적절한지 등을 검증합니다.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.validation_history: List[ValidationResult] = []
        self.performance_metrics = {
            "response_times": [],
            "success_rates": {},
            "error_counts": {},
            "quality_scores": []
        }
    
    async def validate_agent_response(
        self, 
        agent_id: str,
        input_data: Dict[str, Any],
        response: Dict[str, Any],
        expected_format: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0
    ) -> ValidationResult:
        """
        Agent 응답 검증
        
        Args:
            agent_id: Agent ID
            input_data: Agent에게 제공된 입력
            response: Agent의 응답
            expected_format: 예상 응답 형식
            timeout_seconds: 타임아웃 (초)
            
        Returns:
            검증 결과
        """
        start_time = time.time()
        issues = []
        warnings = []
        score = 100.0
        
        try:
            # 1. 기본 응답 구조 검증
            if not isinstance(response, dict):
                issues.append("응답이 딕셔너리 형태가 아님")
                score -= 30
            
            # 2. 필수 필드 검증
            required_fields = ["action", "timestamp"]
            for field in required_fields:
                if field not in response:
                    issues.append(f"필수 필드 누락: {field}")
                    score -= 20
            
            # 3. 응답 시간 검증
            response_time = time.time() - start_time
            if response_time > timeout_seconds:
                issues.append(f"응답 시간 초과: {response_time:.2f}초 > {timeout_seconds}초")
                score -= 25
            elif response_time > timeout_seconds * 0.8:
                warnings.append(f"응답 시간이 긴 편: {response_time:.2f}초")
                score -= 5
            
            # 4. 예상 형식과 비교 (제공된 경우)
            if expected_format:
                format_score = self._validate_format(response, expected_format)
                score = min(score, format_score)
            
            # 5. Agent별 특화 검증
            agent_score = await self._validate_agent_specific(agent_id, input_data, response)
            score = min(score, agent_score)
            
            # 6. 성능 메트릭 업데이트
            self._update_performance_metrics(agent_id, response_time, len(issues) == 0, score)
            
        except Exception as e:
            issues.append(f"검증 중 오류 발생: {str(e)}")
            score = 0
        
        validation_time = time.time() - start_time
        
        result = ValidationResult(
            is_valid=len(issues) == 0,
            level=self.validation_level,
            issues=issues,
            warnings=warnings,
            score=max(0, score),
            timestamp=datetime.now(),
            validation_time=validation_time
        )
        
        self.validation_history.append(result)
        return result
    
    def _validate_format(self, response: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """응답 형식 검증"""
        score = 100.0
        
        for key, expected_type in expected.items():
            if key not in response:
                score -= 20
                continue
            
            if not isinstance(response[key], expected_type):
                score -= 15
        
        return max(0, score)
    
    async def _validate_agent_specific(
        self, 
        agent_id: str, 
        input_data: Dict[str, Any], 
        response: Dict[str, Any]
    ) -> float:
        """Agent별 특화 검증"""
        score = 100.0
        
        # GameAnalyzerAgent 검증
        if "game_analyzer" in agent_id.lower():
            if "game_config" not in response:
                score -= 30
            if response.get("action") != "game_analysis_complete":
                score -= 20
        
        # RuleParserAgent 검증
        elif "rule_parser" in agent_id.lower():
            if "parsed_rules" not in response:
                score -= 30
            rules = response.get("parsed_rules", {})
            if not isinstance(rules, dict) or not rules.get("setup"):
                score -= 20
        
        # PersonaGeneratorAgent 검증
        elif "persona_generator" in agent_id.lower():
            if "personas" not in response:
                score -= 30
            personas = response.get("personas", [])
            if not isinstance(personas, list) or len(personas) == 0:
                score -= 20
        
        # PlayerManagerAgent 검증
        elif "player_manager" in agent_id.lower():
            if response.get("action") == "turn_completed":
                if "action_type" not in response or "action_data" not in response:
                    score -= 25
        
        # GameRefereeAgent 검증
        elif "game_referee" in agent_id.lower():
            if response.get("action") == "validation_complete":
                if "is_valid" not in response:
                    score -= 30
        
        # ScoreCalculatorAgent 검증
        elif "score_calculator" in agent_id.lower():
            if "final_scores" not in response:
                score -= 30
            scores = response.get("final_scores", {})
            if not isinstance(scores, dict):
                score -= 20
        
        return max(0, score)
    
    def _update_performance_metrics(
        self, 
        agent_id: str, 
        response_time: float, 
        success: bool, 
        quality_score: float
    ):
        """성능 메트릭 업데이트"""
        self.performance_metrics["response_times"].append(response_time)
        self.performance_metrics["quality_scores"].append(quality_score)
        
        if agent_id not in self.performance_metrics["success_rates"]:
            self.performance_metrics["success_rates"][agent_id] = {"total": 0, "success": 0}
        
        self.performance_metrics["success_rates"][agent_id]["total"] += 1
        if success:
            self.performance_metrics["success_rates"][agent_id]["success"] += 1
        
        if not success:
            if agent_id not in self.performance_metrics["error_counts"]:
                self.performance_metrics["error_counts"][agent_id] = 0
            self.performance_metrics["error_counts"][agent_id] += 1
    
    def get_agent_performance_report(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Agent 성능 리포트 생성"""
        if agent_id:
            # 특정 Agent 리포트
            success_data = self.performance_metrics["success_rates"].get(agent_id, {"total": 0, "success": 0})
            success_rate = success_data["success"] / success_data["total"] if success_data["total"] > 0 else 0
            error_count = self.performance_metrics["error_counts"].get(agent_id, 0)
            
            return {
                "agent_id": agent_id,
                "success_rate": success_rate,
                "total_calls": success_data["total"],
                "error_count": error_count,
                "average_quality": sum(self.performance_metrics["quality_scores"]) / len(self.performance_metrics["quality_scores"]) if self.performance_metrics["quality_scores"] else 0
            }
        else:
            # 전체 리포트
            avg_response_time = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"]) if self.performance_metrics["response_times"] else 0
            avg_quality = sum(self.performance_metrics["quality_scores"]) / len(self.performance_metrics["quality_scores"]) if self.performance_metrics["quality_scores"] else 0
            
            return {
                "total_validations": len(self.validation_history),
                "average_response_time": avg_response_time,
                "average_quality_score": avg_quality,
                "success_rates": self.performance_metrics["success_rates"],
                "error_counts": self.performance_metrics["error_counts"]
            }


class GameValidator:
    """
    게임 검증기 - 게임 진행과 규칙 준수를 검증
    
    게임이 올바른 규칙에 따라 진행되는지,
    플레이어 행동이 유효한지 등을 검증합니다.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.game_validations: Dict[str, List[ValidationResult]] = {}
    
    async def validate_game_state(
        self, 
        game_state: Dict[str, Any],
        previous_state: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        게임 상태 검증
        
        Args:
            game_state: 현재 게임 상태
            previous_state: 이전 게임 상태 (선택사항)
            
        Returns:
            검증 결과
        """
        start_time = time.time()
        issues = []
        warnings = []
        score = 100.0
        
        try:
            # 1. 기본 게임 상태 구조 검증
            required_fields = ["game_id", "phase", "players", "turn_count"]
            for field in required_fields:
                if field not in game_state:
                    issues.append(f"게임 상태 필수 필드 누락: {field}")
                    score -= 20
            
            # 2. 플레이어 검증
            players = game_state.get("players", [])
            if not isinstance(players, list) or len(players) == 0:
                issues.append("플레이어 정보가 없거나 잘못됨")
                score -= 30
            else:
                player_score = self._validate_players(players)
                score = min(score, player_score)
            
            # 3. 게임 진행 상태 검증
            phase = game_state.get("phase")
            if phase not in [p.value for p in GamePhase]:
                issues.append(f"잘못된 게임 단계: {phase}")
                score -= 25
            
            # 4. 턴 순서 검증
            current_player_index = game_state.get("current_player_index", 0)
            if current_player_index < 0 or current_player_index >= len(players):
                issues.append(f"잘못된 현재 플레이어 인덱스: {current_player_index}")
                score -= 20
            
            # 5. 이전 상태와 비교 (제공된 경우)
            if previous_state:
                transition_score = self._validate_state_transition(previous_state, game_state)
                score = min(score, transition_score)
            
            # 6. 게임별 특화 검증
            game_specific_score = await self._validate_game_specific_rules(game_state)
            score = min(score, game_specific_score)
            
        except Exception as e:
            issues.append(f"게임 상태 검증 중 오류: {str(e)}")
            score = 0
        
        validation_time = time.time() - start_time
        
        result = ValidationResult(
            is_valid=len(issues) == 0,
            level=self.validation_level,
            issues=issues,
            warnings=warnings,
            score=max(0, score),
            timestamp=datetime.now(),
            validation_time=validation_time
        )
        
        # 게임별 검증 기록 저장
        game_id = game_state.get("game_id", "unknown")
        if game_id not in self.game_validations:
            self.game_validations[game_id] = []
        self.game_validations[game_id].append(result)
        
        return result
    
    def _validate_players(self, players: List[Dict[str, Any]]) -> float:
        """플레이어 정보 검증"""
        score = 100.0
        
        for i, player in enumerate(players):
            if not isinstance(player, dict):
                score -= 20
                continue
            
            # 필수 플레이어 필드
            required_fields = ["id", "name", "is_ai"]
            for field in required_fields:
                if field not in player:
                    score -= 10
            
            # 플레이어 ID 중복 체크
            player_ids = [p.get("id") for p in players if isinstance(p, dict)]
            if len(set(player_ids)) != len(player_ids):
                score -= 25
                break
        
        return max(0, score)
    
    def _validate_state_transition(
        self, 
        previous_state: Dict[str, Any], 
        current_state: Dict[str, Any]
    ) -> float:
        """상태 전환 검증"""
        score = 100.0
        
        # 턴 카운트 검증
        prev_turn = previous_state.get("turn_count", 0)
        curr_turn = current_state.get("turn_count", 0)
        
        if curr_turn < prev_turn:
            score -= 30  # 턴이 뒤로 갈 수 없음
        elif curr_turn > prev_turn + 1:
            score -= 15  # 턴이 너무 많이 증가
        
        # 플레이어 수 변경 검증 (게임 중에는 변경되면 안됨)
        prev_players = len(previous_state.get("players", []))
        curr_players = len(current_state.get("players", []))
        
        if prev_players != curr_players:
            phase = current_state.get("phase")
            if phase not in [GamePhase.SETUP.value, GamePhase.PLAYER_GENERATION.value]:
                score -= 25
        
        return max(0, score)
    
    async def _validate_game_specific_rules(self, game_state: Dict[str, Any]) -> float:
        """게임별 특화 규칙 검증"""
        score = 100.0
        
        # 게임 메타데이터 기반 검증
        game_metadata = game_state.get("game_metadata", {})
        if isinstance(game_metadata, dict):
            min_players = game_metadata.get("min_players", 2)
            max_players = game_metadata.get("max_players", 8)
            current_players = len(game_state.get("players", []))
            
            if current_players < min_players:
                score -= 20
            elif current_players > max_players:
                score -= 20
        
        # 파싱된 규칙 기반 검증
        parsed_rules = game_state.get("parsed_rules", {})
        if isinstance(parsed_rules, dict):
            # 승리 조건 확인
            win_conditions = parsed_rules.get("win_conditions", {})
            if not win_conditions:
                score -= 10
        
        return max(0, score)
    
    def get_game_validation_report(self, game_id: str) -> Dict[str, Any]:
        """게임 검증 리포트 생성"""
        validations = self.game_validations.get(game_id, [])
        
        if not validations:
            return {"game_id": game_id, "message": "검증 기록 없음"}
        
        total_validations = len(validations)
        successful_validations = sum(1 for v in validations if v.is_valid)
        average_score = sum(v.score for v in validations) / total_validations
        
        all_issues = []
        all_warnings = []
        for v in validations:
            all_issues.extend(v.issues)
            all_warnings.extend(v.warnings)
        
        return {
            "game_id": game_id,
            "total_validations": total_validations,
            "success_rate": successful_validations / total_validations,
            "average_score": average_score,
            "total_issues": len(all_issues),
            "total_warnings": len(all_warnings),
            "common_issues": self._get_common_items(all_issues),
            "common_warnings": self._get_common_items(all_warnings),
            "validation_trend": [v.score for v in validations[-10:]]  # 최근 10개
        }
    
    def _get_common_items(self, items: List[str]) -> List[Tuple[str, int]]:
        """공통 항목과 빈도 반환"""
        from collections import Counter
        counter = Counter(items)
        return counter.most_common(5)


# 전역 검증기 인스턴스들
_global_agent_validator: Optional[AgentValidator] = None
_global_game_validator: Optional[GameValidator] = None


def get_agent_validator(level: ValidationLevel = ValidationLevel.STANDARD) -> AgentValidator:
    """전역 Agent 검증기 반환"""
    global _global_agent_validator
    
    if _global_agent_validator is None:
        _global_agent_validator = AgentValidator(level)
    
    return _global_agent_validator


def get_game_validator(level: ValidationLevel = ValidationLevel.STANDARD) -> GameValidator:
    """전역 게임 검증기 반환"""
    global _global_game_validator
    
    if _global_game_validator is None:
        _global_game_validator = GameValidator(level)
    
    return _global_game_validator 