"""
Table Game Mate 통합 코어 시스템

모든 핵심 시스템을 하나의 파일로 통합
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from enum import Enum
import logging
import traceback


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
