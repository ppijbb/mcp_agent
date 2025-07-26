"""
게임 액션 모델 정의

게임 내 모든 액션을 정의하고 관리하는 모델
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class ActionType(Enum):
    """게임 액션 타입"""
    # 기본 게임 액션
    MOVE = "move"                    # 이동
    DRAW_CARD = "draw_card"          # 카드 뽑기
    PLAY_CARD = "play_card"          # 카드 사용
    ROLL_DICE = "roll_dice"          # 주사위 굴리기
    COLLECT_RESOURCE = "collect_resource"  # 자원 수집
    BUILD = "build"                  # 건설
    TRADE = "trade"                  # 거래
    ATTACK = "attack"                # 공격
    DEFEND = "defend"                # 방어
    VOTE = "vote"                    # 투표
    ELIMINATE = "eliminate"          # 제거
    
    # 특수 액션
    USE_SPECIAL_ABILITY = "use_special_ability"  # 특수 능력 사용
    PASS_TURN = "pass_turn"          # 턴 패스
    SURRENDER = "surrender"          # 항복
    
    # 시스템 액션
    GAME_START = "game_start"        # 게임 시작
    GAME_END = "game_end"            # 게임 종료
    TURN_START = "turn_start"        # 턴 시작
    TURN_END = "turn_end"            # 턴 종료
    PHASE_CHANGE = "phase_change"    # 페이즈 변경
    
    # 에러/예외 액션
    INVALID_ACTION = "invalid_action"  # 무효한 액션
    TIMEOUT = "timeout"              # 시간 초과
    ERROR = "error"                  # 에러


class ActionStatus(Enum):
    """액션 상태"""
    PENDING = "pending"              # 대기 중
    VALIDATING = "validating"        # 검증 중
    EXECUTING = "executing"          # 실행 중
    COMPLETED = "completed"          # 완료
    FAILED = "failed"                # 실패
    CANCELLED = "cancelled"          # 취소됨
    INVALID = "invalid"              # 무효함


@dataclass
class ActionContext:
    """액션 실행 컨텍스트"""
    game_id: str
    session_id: str
    turn_number: int
    phase: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 게임 상태 스냅샷
    game_state_snapshot: Optional[Dict[str, Any]] = None
    
    # 실행 환경 정보
    execution_environment: Dict[str, Any] = field(default_factory=dict)
    
    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """게임 액션 정의"""
    # 기본 정보
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: ActionType = ActionType.INVALID_ACTION
    player_id: str = ""
    
    # 액션 데이터
    action_data: Dict[str, Any] = field(default_factory=dict)
    target_data: Optional[Dict[str, Any]] = None
    
    # 컨텍스트
    context: Optional[ActionContext] = None
    
    # 상태 및 결과
    status: ActionStatus = ActionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 검증 및 에러
    is_valid: Optional[bool] = None
    validation_message: Optional[str] = None
    error_message: Optional[str] = None
    
    # 결과
    result: Optional[Dict[str, Any]] = None
    side_effects: List[Dict[str, Any]] = field(default_factory=list)
    
    # 메타데이터
    priority: int = 0
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if not self.action_id:
            self.action_id = str(uuid.uuid4())
    
    def validate(self) -> bool:
        """액션 유효성 검증"""
        if not self.player_id:
            self.validation_message = "플레이어 ID가 필요합니다"
            return False
        
        if self.action_type == ActionType.INVALID_ACTION:
            self.validation_message = "유효하지 않은 액션 타입입니다"
            return False
        
        return True
    
    def execute(self) -> Dict[str, Any]:
        """액션 실행 (기본 구현)"""
        self.status = ActionStatus.EXECUTING
        self.executed_at = datetime.now()
        
        try:
            # 기본 검증
            if not self.validate():
                self.status = ActionStatus.INVALID
                return {"success": False, "error": self.validation_message}
            
            # 액션 타입별 실행 로직
            result = self._execute_action_type()
            
            self.status = ActionStatus.COMPLETED
            self.completed_at = datetime.now()
            self.result = result
            
            return result
            
        except Exception as e:
            self.status = ActionStatus.FAILED
            self.error_message = str(e)
            return {"success": False, "error": str(e)}
    
    def _execute_action_type(self) -> Dict[str, Any]:
        """액션 타입별 실행 로직"""
        # 기본 구현 - 실제로는 각 액션 타입별로 오버라이드
        return {
            "success": True,
            "action_type": self.action_type.value,
            "message": f"{self.action_type.value} 액션이 실행되었습니다"
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "player_id": self.player_id,
            "action_data": self.action_data,
            "target_data": self.target_data,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "is_valid": self.is_valid,
            "validation_message": self.validation_message,
            "error_message": self.error_message,
            "result": self.result,
            "priority": self.priority,
            "retry_count": self.retry_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        """딕셔너리에서 생성"""
        return cls(
            action_id=data.get("action_id", str(uuid.uuid4())),
            action_type=ActionType(data.get("action_type", "invalid_action")),
            player_id=data.get("player_id", ""),
            action_data=data.get("action_data", {}),
            target_data=data.get("target_data"),
            status=ActionStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            executed_at=datetime.fromisoformat(data["executed_at"]) if data.get("executed_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            is_valid=data.get("is_valid"),
            validation_message=data.get("validation_message"),
            error_message=data.get("error_message"),
            result=data.get("result"),
            priority=data.get("priority", 0),
            retry_count=data.get("retry_count", 0)
        )


# 특화된 액션 클래스들
@dataclass
class MoveAction(Action):
    """이동 액션"""
    def __post_init__(self):
        super().__post_init__()
        self.action_type = ActionType.MOVE
    
    def _execute_action_type(self) -> Dict[str, Any]:
        from_position = self.action_data.get("from_position")
        to_position = self.action_data.get("to_position")
        
        return {
            "success": True,
            "action_type": "move",
            "from_position": from_position,
            "to_position": to_position,
            "message": f"플레이어가 {from_position}에서 {to_position}로 이동했습니다"
        }


@dataclass
class CardAction(Action):
    """카드 액션"""
    def __post_init__(self):
        super().__post_init__()
        if self.action_type not in [ActionType.DRAW_CARD, ActionType.PLAY_CARD]:
            self.action_type = ActionType.PLAY_CARD
    
    def _execute_action_type(self) -> Dict[str, Any]:
        card_id = self.action_data.get("card_id")
        card_name = self.action_data.get("card_name", "알 수 없는 카드")
        
        if self.action_type == ActionType.DRAW_CARD:
            return {
                "success": True,
                "action_type": "draw_card",
                "card_id": card_id,
                "card_name": card_name,
                "message": f"{card_name}을 뽑았습니다"
            }
        else:  # PLAY_CARD
            return {
                "success": True,
                "action_type": "play_card",
                "card_id": card_id,
                "card_name": card_name,
                "message": f"{card_name}을 사용했습니다"
            }


@dataclass
class VoteAction(Action):
    """투표 액션"""
    def __post_init__(self):
        super().__post_init__()
        self.action_type = ActionType.VOTE
    
    def _execute_action_type(self) -> Dict[str, Any]:
        target_player = self.action_data.get("target_player")
        vote_type = self.action_data.get("vote_type", "elimination")
        
        return {
            "success": True,
            "action_type": "vote",
            "target_player": target_player,
            "vote_type": vote_type,
            "message": f"{target_player}에 대한 {vote_type} 투표를 진행합니다"
        }


# 액션 팩토리
class ActionFactory:
    """액션 생성 팩토리"""
    
    @staticmethod
    def create_action(
        action_type: ActionType,
        player_id: str,
        action_data: Dict[str, Any],
        **kwargs
    ) -> Action:
        """액션 타입에 따른 적절한 액션 객체 생성"""
        
        action_classes = {
            ActionType.MOVE: MoveAction,
            ActionType.DRAW_CARD: CardAction,
            ActionType.PLAY_CARD: CardAction,
            ActionType.VOTE: VoteAction,
        }
        
        action_class = action_classes.get(action_type, Action)
        return action_class(
            action_type=action_type,
            player_id=player_id,
            action_data=action_data,
            **kwargs
        )
    
    @staticmethod
    def create_move_action(
        player_id: str,
        from_position: str,
        to_position: str,
        **kwargs
    ) -> MoveAction:
        """이동 액션 생성"""
        return MoveAction(
            player_id=player_id,
            action_data={
                "from_position": from_position,
                "to_position": to_position
            },
            **kwargs
        )
    
    @staticmethod
    def create_card_action(
        action_type: ActionType,
        player_id: str,
        card_id: str,
        card_name: str,
        **kwargs
    ) -> CardAction:
        """카드 액션 생성"""
        return CardAction(
            action_type=action_type,
            player_id=player_id,
            action_data={
                "card_id": card_id,
                "card_name": card_name
            },
            **kwargs
        )
    
    @staticmethod
    def create_vote_action(
        player_id: str,
        target_player: str,
        vote_type: str = "elimination",
        **kwargs
    ) -> VoteAction:
        """투표 액션 생성"""
        return VoteAction(
            player_id=player_id,
            action_data={
                "target_player": target_player,
                "vote_type": vote_type
            },
            **kwargs
        ) 