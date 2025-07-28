"""
게임 상태 관리 모델
LangGraph State와 호환되는 게임 상태 정의
"""

from typing import Dict, List, Any, Optional, Union, TypedDict, Annotated
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid

class GamePhase(Enum):
    """게임 진행 단계"""
    SETUP = "setup"
    PLAYER_GENERATION = "player_generation"
    RULE_PARSING = "rule_parsing"
    PERSONA_GENERATION = "persona_generation"
    GAME_START = "game_start"
    PLAYER_TURN = "player_turn"
    GAME_END = "game_end"
    SCORE_CALCULATION = "score_calculation"

class GameType(Enum):
    """게임 유형 분류"""
    STRATEGY = "strategy"          # 전략 게임 (카탄, 스플렌더)
    SOCIAL_DEDUCTION = "social"    # 사회적 추론 (마피아, 뱅)
    NEGOTIATION = "negotiation"    # 협상 게임 (디플로마시)
    CARD_GAME = "card"            # 카드 게임 (UNO, 포커)
    BOARD_GAME = "board"          # 보드 게임 (체커, 오델로)
    PARTY_GAME = "party"          # 파티 게임
    DECK_BUILDING = "deck_building" # 덱빌딩

@dataclass
class PlayerInfo:
    """플레이어 정보"""
    id: str
    name: str
    is_ai: bool
    persona_type: Optional[str] = None
    score: int = 0
    is_active: bool = True
    turn_order: int = 0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


# 별칭 추가
Player = PlayerInfo


@dataclass
class GameInfo:
    """게임 정보"""
    name: str
    description: str
    min_players: int
    max_players: int
    estimated_duration: int  # 분
    complexity: str  # simple, moderate, complex
    game_type: str
    rules_url: Optional[str] = None
    bgg_id: Optional[int] = None

@dataclass 
class GameMetadata:
    """게임 메타데이터"""
    name: str
    bgg_id: Optional[int] = None
    min_players: int = 2
    max_players: int = 4
    estimated_duration: int = 30  # 분
    complexity: float = 2.5  # 1-5 스케일
    game_type: GameType = GameType.STRATEGY
    description: str = ""
    rules_url: Optional[str] = None

@dataclass
class GameAction:
    """게임 액션"""
    player_id: str
    action_type: str
    action_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now())
    is_valid: Optional[bool] = None
    validation_message: Optional[str] = None
    reason: Optional[str] = None

# LangGraph State 호환을 위한 TypedDict
class GameState(TypedDict):
    """LangGraph 상태 관리용 게임 상태"""
    # 게임 기본 정보
    game_id: str
    game_metadata: Optional[GameMetadata]
    phase: GamePhase
    
    # 플레이어 정보
    players: List[PlayerInfo]
    current_player_index: int
    
    # 게임 진행 상태
    turn_count: int
    game_board: Dict[str, Any]  # 게임별 보드 상태
    game_history: List[GameAction]
    
    # 규칙 및 설정
    parsed_rules: Optional[Dict[str, Any]]
    game_config: Dict[str, Any]
    
    # 에이전트 간 통신
    last_action: Optional[GameAction]
    pending_actions: List[GameAction]
    error_messages: List[str]
    
    # 게임 결과
    winner_ids: List[str]
    final_scores: Dict[str, int]
    game_ended: bool
    
    # 메타 정보
    created_at: datetime
    updated_at: datetime

class PlayerState(TypedDict):
    """개별 플레이어 상태 (숨겨진 정보 포함)"""
    player_id: str
    public_info: PlayerInfo
    private_data: Dict[str, Any]  # 손패, 비밀 정보 등
    ai_memory: Dict[str, Any]     # AI 플레이어의 기억/전략
    persona_context: Dict[str, Any]  # 페르소나 컨텍스트

class GameConfig(TypedDict):
    """게임 설정"""
    # 기본 설정
    target_game_name: str
    desired_player_count: int
    difficulty_level: str  # easy, medium, hard
    
    # AI 설정
    ai_creativity: float  # 0.0-1.0
    ai_aggression: float  # 0.0-1.0
    enable_persona_chat: bool
    
    # 게임 진행 설정
    auto_progress: bool
    turn_timeout_seconds: int
    enable_hints: bool
    
    # 디버그 설정
    verbose_logging: bool
    save_game_history: bool 