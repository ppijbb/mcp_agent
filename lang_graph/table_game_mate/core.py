"""
Table Game Mate 통합 코어 시스템

모든 핵심 시스템을 하나의 파일로 통합:
- 상태 관리 (GameState, Player, 등)
- 에러 처리 (ErrorHandler)
- 게임 엔진 (GameEngine)
- BGG 규칙 파서 (BGGRuleParser)
- 게임 상태 관리자 (GameStateManager)
- 실시간 게임 테이블 (DynamicGameTable)
"""

from typing import Dict, List, Any, Optional, Union, Callable, Set
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
import traceback
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
import re
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from uuid import uuid4


# ============================================================================
# 상태 관리 시스템
# ============================================================================

class GamePhase(Enum):
    """게임 단계"""
    INITIALIZING = "initializing"
    PLAYERS_SETUP = "players_setup"
    PLAYING = "playing"
    COMPLETED = "completed"
    ERROR = "error"


class GameStatus(Enum):
    """게임 상태"""
    PENDING = "pending"
    READY = "ready"
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"


class Player(BaseModel):
    """플레이어 모델"""
    id: str
    name: str
    type: str = "human"  # human, ai
    score: int = 0
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GameConfig(BaseModel):
    """게임 설정"""
    name: str
    type: str
    min_players: int = 2
    max_players: int = 4
    estimated_duration: int = 60  # minutes
    rules: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GameState(BaseModel):
    """게임 상태"""
    # 기본 정보
    game_id: str = Field(default_factory=lambda: f"game_{int(datetime.now().timestamp())}")
    game_config: Optional[GameConfig] = None
    players: List[Player] = Field(default_factory=list)
    
    # 게임 진행 상태
    current_phase: GamePhase = GamePhase.INITIALIZING
    game_status: GameStatus = GameStatus.PENDING
    current_round: int = 1
    max_rounds: int = 10
    current_player: Optional[Player] = None
    current_action: Optional[Dict[str, Any]] = None
    
    # 게임 데이터
    game_data: Dict[str, Any] = Field(default_factory=dict)
    moves: List[Dict[str, Any]] = Field(default_factory=list)
    scores: Dict[str, int] = Field(default_factory=dict)
    
    # 결과
    winner: Optional[Player] = None
    final_scores: Dict[str, int] = Field(default_factory=dict)
    
    # 시스템 정보
    messages: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class AnalysisState(BaseModel):
    """분석 상태"""
    # 분석 대상
    game_data: Dict[str, Any] = Field(default_factory=dict)
    
    # 분석 진행 상태
    status: str = "pending"  # pending, processing, completed, error
    current_step: str = "initializing"
    
    # 분석 데이터
    processed_data: Optional[Dict[str, Any]] = None
    patterns: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    insights: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None
    
    # 시스템 정보
    messages: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class MonitoringState(BaseModel):
    """모니터링 상태"""
    # 모니터링 상태
    status: str = "pending"  # pending, monitoring, completed, error
    current_step: str = "initializing"
    
    # 메트릭 데이터
    current_metrics: Optional[Dict[str, Any]] = None
    metrics_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # 분석 결과
    performance_analysis: Optional[Dict[str, Any]] = None
    threshold_violations: List[Dict[str, Any]] = Field(default_factory=list)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    dashboard_data: Optional[Dict[str, Any]] = None
    
    # 시스템 정보
    messages: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class AgentState(BaseModel):
    """에이전트 상태"""
    agent_id: str
    status: str = "idle"  # idle, busy, error
    current_task: Optional[str] = None
    last_activity: datetime = Field(default_factory=datetime.now)
    error_count: int = 0
    success_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SystemState(BaseModel):
    """시스템 전체 상태"""
    # 에이전트 상태
    agents: Dict[str, AgentState] = Field(default_factory=dict)
    
    # 활성 게임
    active_games: List[str] = Field(default_factory=list)
    
    # 시스템 메트릭
    system_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # 상태 정보
    status: str = "initializing"  # initializing, running, maintenance, error
    last_health_check: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# 에러 처리 시스템
# ============================================================================

class ErrorSeverity(Enum):
    """에러 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """에러 카테고리"""
    SYSTEM_ERROR = "system_error"
    AGENT_ERROR = "agent_error"
    GAME_ERROR = "game_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"


class ErrorRecord:
    """에러 기록"""
    
    def __init__(self, error_id: str, error: Exception, severity: ErrorSeverity, 
                 category: ErrorCategory, agent_id: str, context: Dict[str, Any] = None):
        self.error_id = error_id
        self.timestamp = datetime.now()
        self.error_type = type(error).__name__
        self.error_message = str(error)
        self.severity = severity
        self.category = category
        self.agent_id = agent_id
        self.context = context or {}
        self.stack_trace = traceback.format_exc()
        self.resolved = False


class ErrorHandler:
    """에러 핸들러"""
    
    def __init__(self):
        self.logger = logging.getLogger("error_handler")
        self.error_records: List[ErrorRecord] = []
        self.error_counts: Dict[str, int] = {}
        
        # 로깅 설정
        self.logger.setLevel(logging.ERROR)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    async def handle_error(self, error: Exception, agent_id: str, 
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                          category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
                          context: Dict[str, Any] = None) -> str:
        """에러 처리"""
        try:
            # 에러 ID 생성
            error_id = f"err_{int(datetime.now().timestamp())}_{len(self.error_records)}"
            
            # 에러 기록 생성
            error_record = ErrorRecord(
                error_id=error_id,
                error=error,
                severity=severity,
                category=category,
                agent_id=agent_id,
                context=context
            )
            
            # 에러 기록 저장
            self.error_records.append(error_record)
            
            # 에러 카운트 업데이트
            error_key = f"{agent_id}_{category.value}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            # 로깅
            self._log_error(error_record)
            
            # 심각한 에러의 경우 추가 처리
            if severity == ErrorSeverity.CRITICAL:
                await self._handle_critical_error(error_record)
            
            return error_id
            
        except Exception as e:
            # 에러 핸들러 자체에서 에러 발생
            self.logger.critical(f"Error handler failed: {str(e)}")
            return "error_handler_failed"
    
    def _log_error(self, error_record: ErrorRecord) -> None:
        """에러 로깅"""
        log_message = (
            f"Error {error_record.error_id}: {error_record.error_type} - "
            f"{error_record.error_message} (Agent: {error_record.agent_id}, "
            f"Severity: {error_record.severity.value}, Category: {error_record.category.value})"
        )
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    async def _handle_critical_error(self, error_record: ErrorRecord) -> None:
        """심각한 에러 처리"""
        # 심각한 에러 발생 시 시스템 상태 확인
        # 필요시 시스템 종료 또는 복구 시도
        pass
    
    def get_error_summary(self) -> Dict[str, Any]:
        """에러 요약"""
        total_errors = len(self.error_records)
        unresolved_errors = len([r for r in self.error_records if not r.resolved])
        
        # 심각도별 분류
        severity_counts = {}
        for record in self.error_records:
            severity = record.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # 카테고리별 분류
        category_counts = {}
        for record in self.error_records:
            category = record.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # 에이전트별 분류
        agent_counts = {}
        for record in self.error_records:
            agent = record.agent_id
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        return {
            "total_errors": total_errors,
            "unresolved_errors": unresolved_errors,
            "severity_distribution": severity_counts,
            "category_distribution": category_counts,
            "agent_distribution": agent_counts,
            "error_counts": self.error_counts
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 에러 목록"""
        recent_errors = sorted(
            self.error_records, 
            key=lambda x: x.timestamp, 
            reverse=True
        )[:limit]
        
        return [
            {
                "error_id": record.error_id,
                "timestamp": record.timestamp.isoformat(),
                "error_type": record.error_type,
                "error_message": record.error_message,
                "severity": record.severity.value,
                "category": record.category.value,
                "agent_id": record.agent_id,
                "resolved": record.resolved
            }
            for record in recent_errors
        ]
    
    def resolve_error(self, error_id: str) -> bool:
        """에러 해결 처리"""
        for record in self.error_records:
            if record.error_id == error_id:
                record.resolved = True
                return True
        return False
    
    def clear_old_errors(self, days: int = 7) -> int:
        """오래된 에러 정리"""
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
        
        old_errors = [r for r in self.error_records if r.timestamp < cutoff_date]
        self.error_records = [r for r in self.error_records if r.timestamp >= cutoff_date]
        
        return len(old_errors)


# ============================================================================
# 게임 엔진
# ============================================================================

class GameEngine:
    """게임 실행 엔진"""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.active_games: Dict[str, GameState] = {}
        self.game_types = {}
        self._load_game_types()
    
    async def initialize(self, game_config: GameConfig) -> bool:
        """게임 엔진 초기화"""
        try:
            if not game_config:
                raise ValueError("게임 설정이 필요합니다")
            
            # 게임 타입 검증
            if game_config.type not in self.game_types:
                raise ValueError(f"지원하지 않는 게임 타입: {game_config.type}")
            
            # 게임별 초기화 함수 실행
            init_func = self.game_types[game_config.type]
            await init_func(game_config)
            
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(e, "game_engine")
            return False
    
    async def add_player(self, player: Player) -> bool:
        """플레이어 추가"""
        try:
            if not player:
                raise ValueError("플레이어 정보가 필요합니다")
            
            if not player.id or not player.name:
                raise ValueError("플레이어 ID와 이름이 필요합니다")
            
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(e, "game_engine")
            return False
    
    async def start_game(self) -> bool:
        """게임 시작"""
        try:
            # 게임 시작 로직
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(e, "game_engine")
            return False
    
    async def execute_player_action(self, player: Player, action: Dict[str, Any]) -> Dict[str, Any]:
        """플레이어 액션 실행"""
        try:
            if not player:
                raise ValueError("플레이어가 필요합니다")
            
            if not action:
                raise ValueError("액션이 필요합니다")
            
            # 액션 실행 로직
            result = {
                "success": True,
                "player_id": player.id,
                "action_type": action.get("type", "unknown"),
                "game_data": {},
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            await self.error_handler.handle_error(e, "game_engine")
            return {
                "success": False,
                "error": str(e),
                "player_id": player.id if player else "unknown"
            }
    
    async def get_next_player(self) -> Optional[Player]:
        """다음 플레이어 반환"""
        try:
            # 다음 플레이어 로직
            return None
            
        except Exception as e:
            await self.error_handler.handle_error(e, "game_engine")
            return None
    
    async def is_game_over(self) -> bool:
        """게임 종료 여부 확인"""
        try:
            # 게임 종료 조건 확인
            return False
            
        except Exception as e:
            await self.error_handler.handle_error(e, "game_engine")
            return True
    
    async def get_winner(self) -> Optional[Player]:
        """승자 반환"""
        try:
            # 승자 결정 로직
            return None
            
        except Exception as e:
            await self.error_handler.handle_error(e, "game_engine")
            return None
    
    async def calculate_final_scores(self) -> Dict[str, int]:
        """최종 점수 계산"""
        try:
            # 점수 계산 로직
            return {}
            
        except Exception as e:
            await self.error_handler.handle_error(e, "game_engine")
            return {}
    
    async def cleanup(self) -> bool:
        """게임 정리"""
        try:
            # 정리 로직
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(e, "game_engine")
            return False
    
    def _load_game_types(self) -> None:
        """게임 타입 동적 로드"""
        try:
            import json
            import os
            
            # game_data.json에서 게임 타입 로드
            data_file = os.path.join(os.path.dirname(__file__), 'data', 'game_data.json')
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    games = data.get('games', {})
                    
                    for game_id, game_info in games.items():
                        self.game_types[game_id] = self._create_game_initializer(game_id, game_info)
            else:
                # 기본 게임 타입들
                self.game_types = {
                    "chess": self._create_game_initializer("chess", {"name": "Chess", "type": "strategy"}),
                    "checkers": self._create_game_initializer("checkers", {"name": "Checkers", "type": "strategy"}),
                    "go": self._create_game_initializer("go", {"name": "Go", "type": "strategy"}),
                    "poker": self._create_game_initializer("poker", {"name": "Poker", "type": "card"})
                }
        except Exception as e:
            # 기본 게임 타입들로 폴백
            self.game_types = {
                "chess": self._create_game_initializer("chess", {"name": "Chess", "type": "strategy"}),
                "checkers": self._create_game_initializer("checkers", {"name": "Checkers", "type": "strategy"}),
                "go": self._create_game_initializer("go", {"name": "Go", "type": "strategy"}),
                "poker": self._create_game_initializer("poker", {"name": "Poker", "type": "card"})
            }
    
    def _create_game_initializer(self, game_id: str, game_info: dict):
        """게임 초기화 함수 생성"""
        async def init_func(config: GameConfig) -> None:
            # 게임별 초기화 로직
            pass
        return init_func


# ============================================================================
# BGG 규칙 파서 (BoardGameGeek API 통합)
# ============================================================================

class GameCategory(Enum):
    """게임 카테고리"""
    STRATEGY = "strategy"
    CARD = "card"
    FAMILY = "family"
    PARTY = "party"
    ABSTRACT = "abstract"
    WARGAME = "wargame"


@dataclass
class GameSetup:
    """게임 설정 정보"""
    board_config: str = ""
    pieces_distribution: Dict[str, int] = None
    initial_resources: Dict[str, Any] = None
    player_setup: str = ""
    special_setup: List[str] = None
    
    def __post_init__(self):
        if self.pieces_distribution is None:
            self.pieces_distribution = {}
        if self.initial_resources is None:
            self.initial_resources = {}
        if self.special_setup is None:
            self.special_setup = []


@dataclass
class TurnStructure:
    """턴 구조 정보"""
    phases: List[str] = None
    actions_per_turn: List[str] = None
    turn_order: str = ""
    duration_hint: str = ""
    
    def __post_init__(self):
        if self.phases is None:
            self.phases = []
        if self.actions_per_turn is None:
            self.actions_per_turn = []


@dataclass
class GameAction:
    """게임 액션 정보"""
    action_type: str
    description: str
    requirements: List[str] = None
    consequences: List[str] = None
    examples: List[str] = None
    
    def __post_init__(self):
        if self.requirements is None:
            self.requirements = []
        if self.consequences is None:
            self.consequences = []
        if self.examples is None:
            self.examples = []


@dataclass
class WinCondition:
    """승리 조건"""
    condition_type: str
    description: str
    triggers: List[str] = None
    tie_breaker: str = ""
    
    def __post_init__(self):
        if self.triggers is None:
            self.triggers = []


@dataclass
class SpecialRule:
    """특별 규칙"""
    rule_name: str
    description: str
    trigger_condition: str
    effects: List[str] = None
    
    def __post_init__(self):
        if self.effects is None:
            self.effects = []


class GameRules(BaseModel):
    """구조화된 게임 규칙"""
    bgg_id: int = 0
    name: str = ""
    description: str = ""
    setup: GameSetup = None
    turn_structure: TurnStructure = None
    actions: List[GameAction] = None
    win_conditions: List[WinCondition] = None
    special_rules: List[SpecialRule] = None
    categories: List[str] = None
    mechanics: List[str] = None
    complexity: float = 0.0
    playing_time: int = 0
    player_count: Dict[str, int] = None
    parsed_at: str = ""
    raw_description: str = ""
    
    def __init__(self, **data):
        # None으로 전달된 컬렉션들을 빈 리스트/dict로 초기화
        if 'setup' not in data or data['setup'] is None:
            data['setup'] = GameSetup()
        if 'turn_structure' not in data or data['turn_structure'] is None:
            data['turn_structure'] = TurnStructure()
        if 'actions' not in data or data['actions'] is None:
            data['actions'] = []
        if 'win_conditions' not in data or data['win_conditions'] is None:
            data['win_conditions'] = []
        if 'special_rules' not in data or data['special_rules'] is None:
            data['special_rules'] = []
        if 'categories' not in data or data['categories'] is None:
            data['categories'] = []
        if 'mechanics' not in data or data['mechanics'] is None:
            data['mechanics'] = []
        if 'player_count' not in data or data['player_count'] is None:
            data['player_count'] = {}
        super().__init__(**data)
    
    def to_llm_prompt(self) -> str:
        """LLM용 프롬프트 문자열 생성"""
        prompt = f"""
# 게임: {self.name}

## 기본 정보
- 플레이어 수: {self.player_count.get('min', '?')}-{self.player_count.get('max', '?')}명
- 예상 게임 시간: {self.playing_time}분
- 복잡도: {self.complexity:.2f}/5
- 카테고리: {', '.join(self.categories)}
- 메커닉: {', '.join(self.mechanics)}

## 게임 설명
{self.description}

## 게임 설정
{self.setup.board_config}

플레이어별 초기 자산:
{json.dumps(self.setup.initial_resources, ensure_ascii=False, indent=2) if self.setup else '{}'}

## 턴 구조
{self.turn_structure.turn_order if self.turn_structure else ''}

phases:
{chr(10).join(['- ' + p for p in self.turn_structure.phases]) if self.turn_structure else ''}

각 턴에서 가능한 행동:
{chr(10).join(['- ' + a.action_type + ': ' + a.description for a in self.actions])}

## 승리 조건
"""
        for win in self.win_conditions:
            prompt += f"- {win.condition_type}: {win.description}\n"
        
        if self.special_rules:
            prompt += "\n## 특별 규칙\n"
            for rule in self.special_rules:
                prompt += f"- {rule.rule_name}: {rule.description}\n"
        
        prompt += "\n## 현재 게임 상태\n"
        prompt += "현재 게임의 상태를 나타내는 JSON을 제공할 것입니다. \
당신은 이 규칙에 따라 합법적인 움직임만 해야 합니다.\n"
        
        return prompt


class BGGRuleParser:
    """BGG 규칙 파서"""
    
    BGG_API_BASE = "https://boardgamegeek.com/xmlapi2"
    REQUEST_DELAY = 5.0
    
    def __init__(self):
        self.last_request_time = 0
        self.rules_cache: Dict[int, GameRules] = {}
    
    async def _rate_limited_request(self, url: str) -> str:
        """BGG API rate limit을 준수하는 HTTP 요청"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.REQUEST_DELAY:
            await asyncio.sleep(self.REQUEST_DELAY - time_since_last)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                self.last_request_time = asyncio.get_event_loop().time()
                if response.status == 200:
                    return await response.text()
                else:
                    raise Exception(f"BGG API error: {response.status}")
    
    def _clean_html(self, html_text: str) -> str:
        """HTML 태그 제거"""
        clean_text = re.sub(r'<[^>]+>', '', html_text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        return clean_text.strip()
    
    async def fetch_game_rules(self, bgg_id: int) -> Optional[GameRules]:
        """게임 규칙 가져오기"""
        if bgg_id in self.rules_cache:
            return self.rules_cache[bgg_id]
        
        try:
            url = f"{self.BGG_API_BASE}/thing?id={bgg_id}&stats=1"
            xml_content = await self._rate_limited_request(url)
            
            root = ET.fromstring(xml_content)
            item = root.find('.//item')
            
            if item is None:
                return None
            
            game_info = {
                "id": int(item.get("id")),
                "name": item.find(".//name[@type='primary']").get("value") if item.find(".//name[@type='primary']") is not None else "Unknown",
                "description": item.find("description").text if item.find("description") is not None else "",
            }
            
            categories = []
            for link in item.findall(".//link[@type='boardgamecategory']"):
                categories.append(link.get("value"))
            
            mechanics = []
            for link in item.findall(".//link[@type='boardgamemechanic']"):
                mechanics.append(link.get("value"))
            
            stats = item.find(".//statistics/ratings")
            complexity = 0.0
            if stats is not None:
                complexity = float(stats.find("averageweight").get("value", 0))
            
            min_players = int(item.find("minplayers").get("value", 2)) if item.find("minplayers") is not None else 2
            max_players = int(item.find("maxplayers").get("value", 4)) if item.find("maxplayers") is not None else 4
            playing_time = int(item.find("playingtime").get("value", 60)) if item.find("playingtime") is not None else 60
            
            rules = GameRules(
                bgg_id=bgg_id,
                name=game_info["name"],
                description=self._clean_html(game_info["description"]),
                setup=self._parse_setup(game_info["name"]),
                turn_structure=self._parse_turn_structure(),
                actions=self._parse_actions(),
                win_conditions=self._parse_win_conditions(),
                special_rules=[],
                categories=categories,
                mechanics=mechanics,
                complexity=complexity,
                playing_time=playing_time,
                player_count={"min": min_players, "max": max_players},
                parsed_at=datetime.now().isoformat(),
                raw_description=game_info["description"]
            )
            
            self.rules_cache[bgg_id] = rules
            return rules
            
        except Exception as e:
            logging.error(f"Failed to fetch rules for game {bgg_id}: {e}")
            return None
    
    def _parse_setup(self, game_name: str) -> GameSetup:
        """게임 설정 파싱"""
        setup = GameSetup()
        setup.board_config = f"{game_name} 보드 설정"
        setup.player_setup = "각 플레이어는 자신의 차례에 행동을 수행합니다."
        
        game_name_lower = game_name.lower()
        
        if "chess" in game_name_lower:
            setup.board_config = "8x8 체스보드, 각 플레이어: King 1, Queen 1, Rook 2, Bishop 2, Knight 2, Pawn 8"
            setup.initial_resources = {"pieces": {"king": 1, "queen": 1, "rook": 2, "bishop": 2, "knight": 2, "pawn": 8}}
        elif "poker" in game_name_lower:
            setup.board_config = "52장 카드 덱"
            setup.initial_resources = {"cards": 5, "chips": 1000}
        elif "go" in game_name_lower:
            setup.board_config = "19x19 바둑판"
            setup.initial_resources = {"stones": {"black": 181, "white": 180}}
        
        return setup
    
    def _parse_turn_structure(self) -> TurnStructure:
        """턴 구조 파싱"""
        return TurnStructure(
            phases=["플레이어 턴 시작", "행동 선택", "행동 실행", "턴 종료"],
            actions_per_turn=["기본 행동 1회", "조건부 특수 행동"],
            turn_order="시계 방향 순서",
            duration_hint="각 플레이어의 턴은 상황에 따라 다름"
        )
    
    def _parse_actions(self) -> List[GameAction]:
        """게임 액션 파싱"""
        actions = []
        
        actions.append(GameAction(
            action_type="PASS",
            description="행동 없이 턴 넘기기",
            requirements=[],
            consequences=["다음 플레이어로 순서 이동"],
            examples=["아무것도 하지 않고 턴을 넘긴다"]
        ))
        
        actions.append(GameAction(
            action_type="MOVE",
            description="게임 피스 이동",
            requirements=["이동할 피스가 있어야 함", "이동 가능한 경로가 있어야 함"],
            consequences=["피스 위치 변경", "상대 피스.capture 가능"],
            examples=["폰 전진", "나이트 L자형 이동"]
        ))
        
        return actions
    
    def _parse_win_conditions(self) -> List[WinCondition]:
        """승리 조건 파싱"""
        win_conditions = []
        
        win_conditions.append(WinCondition(
            condition_type="HIGHEST_SCORE",
            description="게임 종료 시 가장 많은 점수를 가진 플레이어 승리",
            triggers=["게임 종료 조건 달성"],
            tie_breaker="동점 시 사전 정의된 규칙 적용"
        ))
        
        return win_conditions


# ============================================================================
# 실시간 게임 상태 관리
# ============================================================================

class PlayerStatus(Enum):
    """플레이어 상태"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    READY = "ready"
    PLAYING = "playing"
    WAITING = "waiting"
    ELIMINATED = "eliminated"


class TablePlayer(BaseModel):
    """테이블 플레이어"""
    player_id: str
    name: str
    status: PlayerStatus = PlayerStatus.CONNECTED
    score: int = 0
    is_human: bool = False
    llm_model: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    joined_at: str = ""
    
    def __init__(self, **data):
        if 'joined_at' not in data or data['joined_at'] is None:
            data['joined_at'] = datetime.now().isoformat()
        super().__init__(**data)


class GameTable(BaseModel):
    """게임 테이블"""
    table_id: str
    game_type: str
    bgg_id: Optional[int] = None
    max_players: int = 4
    min_players: int = 2
    status: GameStatus = GameStatus.PENDING
    current_turn: int = 0
    current_player_id: Optional[str] = None
    players: Dict[str, TablePlayer] = Field(default_factory=dict)
    player_order: List[str] = Field(default_factory=list)
    board_state: Dict[str, Any] = Field(default_factory=dict)
    rules: Dict[str, Any] = Field(default_factory=dict)
    legal_moves: List[str] = Field(default_factory=list)
    move_history: List[Dict] = Field(default_factory=list)
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    winner_id: Optional[str] = None
    
    def __init__(self, **data):
        if 'created_at' not in data or data['created_at'] is None:
            data['created_at'] = datetime.now().isoformat()
        super().__init__(**data)
    
    def add_player(self, player: TablePlayer) -> bool:
        """플레이어 추가"""
        if len(self.players) >= self.max_players:
            return False
        self.players[player.player_id] = player
        self.player_order.append(player.player_id)
        return True
    
    def remove_player(self, player_id: str) -> bool:
        """플레이어 제거"""
        if player_id not in self.players:
            return False
        del self.players[player_id]
        if player_id in self.player_order:
            self.player_order.remove(player_id)
        return True
    
    def next_turn(self) -> Optional[str]:
        """다음 턴으로"""
        if not self.player_order:
            return None
        self.current_turn += 1
        player_idx = (self.current_turn - 1) % len(self.player_order)
        self.current_player_id = self.player_order[player_idx]
        return self.current_player_id
    
    def is_full(self) -> bool:
        """테이블이 가득 찼는지"""
        return len(self.players) >= self.max_players
    
    def can_start(self) -> bool:
        """게임 시작 가능한지"""
        return (
            len(self.players) >= self.min_players and
            self.status == GameStatus.PENDING and
            all(p.status == PlayerStatus.READY for p in self.players.values())
        )


class GameStateManager:
    """게임 상태 관리자"""
    
    def __init__(self):
        self.tables: Dict[str, GameTable] = {}
        self.player_sessions: Dict[str, str] = {}
        self.subscribers: Dict[str, Set[Callable]] = defaultdict(set)
        self._lock = asyncio.Lock()
    
    async def create_table(
        self,
        game_type: str,
        bgg_id: Optional[int] = None,
        max_players: int = 4,
        min_players: int = 2
    ) -> GameTable:
        """게임 테이블 생성"""
        async with self._lock:
            table_id = f"table_{uuid4().hex[:8]}"
            table = GameTable(
                table_id=table_id,
                game_type=game_type,
                bgg_id=bgg_id,
                max_players=max_players,
                min_players=min_players
            )
            self.tables[table_id] = table
            return table
    
    async def join_table(
        self,
        table_id: str,
        player_id: str,
        player_name: str,
        is_human: bool = False,
        llm_model: str = ""
    ) -> Optional[TablePlayer]:
        """테이블 참여"""
        async with self._lock:
            table = self.tables.get(table_id)
            if not table or table.is_full():
                return None
            
            player = TablePlayer(
                player_id=player_id,
                name=player_name,
                is_human=is_human,
                llm_model=llm_model,
                status=PlayerStatus.READY
            )
            
            table.add_player(player)
            self.player_sessions[player_id] = table_id
            return player
    
    async def start_game(self, table_id: str) -> bool:
        """게임 시작"""
        table = self.tables.get(table_id)
        if not table or not table.can_start():
            return False
        
        table.status = GameStatus.ACTIVE
        table.started_at = datetime.now().isoformat()
        table.current_turn = 0
        table.next_turn()
        return True
    
    async def update_board_state(
        self,
        table_id: str,
        board_state: Dict[str, Any],
        legal_moves: List[str]
    ) -> bool:
        """보드 상태 업데이트"""
        table = self.tables.get(table_id)
        if not table:
            return False
        table.board_state = board_state
        table.legal_moves = legal_moves
        return True
    
    async def record_move(
        self,
        table_id: str,
        player_id: str,
        move_type: str,
        move_data: Dict[str, Any],
        reasoning: str = ""
    ) -> bool:
        """움직임 기록"""
        table = self.tables.get(table_id)
        if not table:
            return False
        
        move_record = {
            "player_id": player_id,
            "move_type": move_type,
            "move_data": move_data,
            "reasoning": reasoning,
            "turn": table.current_turn,
            "timestamp": datetime.now().isoformat()
        }
        
        table.move_history.append(move_record)
        return True
    
    async def end_game(self, table_id: str, winner_id: Optional[str] = None) -> bool:
        """게임 종료"""
        table = self.tables.get(table_id)
        if not table:
            return False
        
        table.status = GameStatus.COMPLETED
        table.completed_at = datetime.now().isoformat()
        table.winner_id = winner_id
        return True
    
    def get_table(self, table_id: str) -> Optional[GameTable]:
        """테이블 조회"""
        return self.tables.get(table_id)


# ============================================================================
# LLM 게임 에이전트 (agents.py에 통합될 내용)
# ============================================================================

class PlayerType(Enum):
    """플레이어 타입"""
    HUMAN = "human"
    LLM = "llm"
    AI = "ai"


class MoveResult(Enum):
    """움직임 결과"""
    VALID = "valid"
    INVALID = "invalid"
    ILLEGAL = "illegal"
    WIN = "win"
    LOSE = "lose"
    DRAW = "draw"


class GameMove(BaseModel):
    """게임 움직임"""
    player_id: str
    move_type: str
    move_data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = ""
    reasoning: str = ""
    is_llm_generated: bool = False
    llm_model: str = ""
    
    def __init__(self, **data):
        if 'timestamp' not in data or data['timestamp'] is None:
            data['timestamp'] = datetime.now().isoformat()
        super().__init__(**data)


class GameStateSnapshot(BaseModel):
    """게임 상태 스냅샷"""
    game_id: str
    turn_number: int
    current_player: str
    board_state: Dict[str, Any]
    player_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    move_history: List[GameMove] = Field(default_factory=list)
    legal_moves: List[str] = Field(default_factory=list)
    game_status: str
    timestamp: str = ""
    
    def __init__(self, **data):
        if 'timestamp' not in data or data['timestamp'] is None:
            data['timestamp'] = datetime.now().isoformat()
        super().__init__(**data)


class LLMGameAgent:
    """LLM 게임 에이전트"""
    
    def __init__(
        self,
        agent_id: str,
        player_type: PlayerType = PlayerType.LLM,
        llm_model: str = "gemini-2.5-flash-lite"
    ):
        self.agent_id = agent_id
        self.player_type = player_type
        self.llm_model = llm_model
        self.is_active = True
        self.score = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.move_history: List[GameMove] = []
    
    async def think_and_move(
        self,
        game_state: GameStateSnapshot,
        rules: str,
        available_moves: List[str],
        timeout: float = 30.0
    ) -> GameMove:
        """게임 상태를 분석하고 움직임 결정"""
        # 시뮬레이션 - 실제 구현에서는 LLM API 호출
        if available_moves:
            move = GameMove(
                player_id=self.agent_id,
                move_type=available_moves[0],
                move_data={},
                reasoning="LLM이 최적의 움직임을 결정했습니다.",
                is_llm_generated=(self.player_type == PlayerType.LLM),
                llm_model=self.llm_model
            )
        else:
            move = GameMove(
                player_id=self.agent_id,
                move_type="PASS",
                move_data={},
                reasoning="가능한 움직임이 없습니다.",
                is_llm_generated=(self.player_type == PlayerType.LLM),
                llm_model=self.llm_model
            )
        
        self.move_history.append(move)
        return move
    
    @classmethod
    def create_llm_agent(
        cls,
        agent_id: str,
        provider: str = "google",
        model: str = "gemini-2.5-flash-lite"
    ) -> 'LLMGameAgent':
        """LLM 에이전트 생성 팩토리"""
        return cls(
            agent_id=agent_id,
            player_type=PlayerType.LLM,
            llm_model=model
        )


# ============================================================================
# 다이나믹 게임 테이블
# ============================================================================

class GameEngine(ABC):
    """게임 엔진 추상 클래스"""
    
    @abstractmethod
    async def initialize(self, rules: GameRules, players: List[TablePlayer]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def get_legal_moves(self, player_id: str, board_state: Dict) -> List[str]:
        pass
    
    @abstractmethod
    async def apply_move(self, player_id: str, move_type: str, move_data: Dict) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def check_game_over(self, board_state: Dict) -> tuple:
        pass
    
    @abstractmethod
    def get_board_state(self) -> Dict[str, Any]:
        pass


class ChessGameEngine(GameEngine):
    """체스 게임 엔진"""
    
    def __init__(self):
        self.board_state: Dict[str, Any] = {}
        self.move_count = 0
    
    async def initialize(self, rules: GameRules, players: List[TablePlayer]) -> Dict[str, Any]:
        self.board_state = {
            "board": self._create_initial_board(),
            "current_player": "white",
            "move_number": 1
        }
        return self.board_state
    
    def _create_initial_board(self) -> Dict[str, Any]:
        board = {}
        for col in range(8):
            board[f"a{col + 1}"] = {"piece": "pawn", "color": "white"}
            board[f"a{col + 8}"] = {"piece": "pawn", "color": "black"}
        pieces = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]
        for i, piece in enumerate(pieces):
            board[f"{chr(97 + i)}1"] = {"piece": piece, "color": "white"}
            board[f"{chr(97 + i)}8"] = {"piece": piece, "color": "black"}
        return board
    
    async def get_legal_moves(self, player_id: str, board_state: Dict) -> List[str]:
        return ["MOVE_PIECE", "CASTLE", "CAPTURE", "PROMOTE"]
    
    async def apply_move(self, player_id: str, move_type: str, move_data: Dict) -> Dict[str, Any]:
        from_pos = move_data.get("from")
        to_pos = move_data.get("to")
        
        if from_pos and from_pos in self.board_state["board"]:
            piece = self.board_state["board"][from_pos]
            self.board_state["board"][to_pos] = piece
            del self.board_state["board"][from_pos]
        
        self.board_state["current_player"] = (
            "black" if self.board_state["current_player"] == "white" else "white"
        )
        self.move_count += 1
        
        return {"success": True, "move_type": move_type}
    
    async def check_game_over(self, board_state: Dict) -> tuple:
        if self.move_count >= 100:
            return (True, "draw", "50-move rule")
        return (False, None, None)
    
    def get_board_state(self) -> Dict[str, Any]:
        return self.board_state


class DynamicGameTable:
    """동적 게임 테이블"""
    
    def __init__(
        self,
        table_id: str,
        game_type: str,
        bgg_id: Optional[int] = None,
        state_manager: Optional[GameStateManager] = None
    ):
        self.table_id = table_id
        self.game_type = game_type
        self.bgg_id = bgg_id
        self.state_manager = state_manager or GameStateManager()
        self.game_engine: Optional[GameEngine] = None
        self.rules: Optional[GameRules] = None
        self.agents: Dict[str, LLMGameAgent] = {}
        self._running = False
    
    async def initialize(self, rules: Optional[GameRules] = None) -> bool:
        try:
            if rules:
                self.rules = rules
            elif self.bgg_id:
                parser = BGGRuleParser()
                self.rules = await parser.fetch_game_rules(self.bgg_id)
            
            self.game_engine = self._create_game_engine()
            
            await self.state_manager.create_table(
                game_type=self.game_type,
                bgg_id=self.bgg_id,
                max_players=4,
                min_players=2
            )
            
            return True
        except Exception as e:
            logging.error(f"Failed to initialize table: {e}")
            return False
    
    def _create_game_engine(self) -> GameEngine:
        if self.game_type.lower() in ["chess", "체스"]:
            return ChessGameEngine()
        return ChessGameEngine()
    
    async def add_player(
        self,
        player_id: str,
        player_name: str,
        is_human: bool = False,
        llm_model: str = ""
    ) -> bool:
        player = await self.state_manager.join_table(
            table_id=self.table_id,
            player_id=player_id,
            player_name=player_name,
            is_human=is_human,
            llm_model=llm_model
        )
        
        if not player:
            return False
        
        if not is_human:
            agent = LLMGameAgent.create_llm_agent(
                agent_id=player_id,
                model=llm_model or "gemini-2.5-flash-lite"
            )
            self.agents[agent.agent_id] = agent
        
        return True
    
    async def start_game(self) -> bool:
        table = self.state_manager.get_table(self.table_id)
        if not table or not table.can_start():
            return False
        
        success = await self.state_manager.start_game(self.table_id)
        if success:
            self._running = True
            asyncio.create_task(self._game_loop())
        
        return success
    
    async def _game_loop(self):
        """게임 루프"""
        try:
            while self._running:
                table = self.state_manager.get_table(self.table_id)
                
                if not table or table.status != GameStatus.ACTIVE:
                    break
                
                current_player_id = table.current_player_id
                if not current_player_id:
                    break
                
                player = table.players.get(current_player_id)
                if not player:
                    break
                
                legal_moves = await self.game_engine.get_legal_moves(
                    current_player_id,
                    table.board_state
                )
                
                if not player.is_human:
                    agent = self.agents.get(current_player_id)
                    if agent:
                        snapshot = GameStateSnapshot(
                            game_id=self.table_id,
                            turn_number=table.current_turn,
                            current_player=current_player_id,
                            board_state=table.board_state,
                            legal_moves=legal_moves,
                            game_status=table.status.value
                        )
                        
                        rules_text = self.rules.to_llm_prompt() if self.rules else ""
                        move = await agent.think_and_move(
                            game_state=snapshot,
                            rules=rules_text,
                            available_moves=legal_moves
                        )
                        
                        if move.move_type != "WAITING_FOR_INPUT":
                            await self.game_engine.apply_move(
                                current_player_id,
                                move.move_type,
                                move.move_data
                            )
                            
                            await self.state_manager.record_move(
                                self.table_id,
                                current_player_id,
                                move.move_type,
                                move.move_data,
                                move.reasoning
                            )
                
                is_over, result_type, _ = await self.game_engine.check_game_over(table.board_state)
                
                if is_over:
                    winner_id = current_player_id if result_type == "win" else None
                    await self.state_manager.end_game(self.table_id, winner_id)
                    self._running = False
                    break
                
                self.state_manager.get_table(self.table_id).next_turn()
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logging.error(f"Game loop error: {e}")
            self._running = False
    
    def get_table_status(self) -> Dict[str, Any]:
        table = self.state_manager.get_table(self.table_id)
        if not table:
            return {}
        
        return {
            "table_id": self.table_id,
            "game_type": self.game_type,
            "status": table.status.value,
            "players": [
                {"id": pid, "name": p.name, "is_human": p.is_human, "score": p.score}
                for pid, p in table.players.items()
            ],
            "current_turn": table.current_turn,
            "current_player": table.current_player_id,
            "total_moves": len(table.move_history)
        }
