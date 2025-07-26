"""
액션 실행 엔진

게임 액션의 실행, 검증, 관리를 담당하는 핵심 시스템
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import traceback

from ..models.action import Action, ActionType, ActionStatus, ActionContext, ActionFactory
from ..models.game_state import GameState, GamePhase
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionPriority(Enum):
    """실행 우선순위"""
    CRITICAL = 0      # 즉시 실행 (시스템 액션)
    HIGH = 1          # 높은 우선순위
    NORMAL = 2        # 일반 우선순위
    LOW = 3           # 낮은 우선순위
    BACKGROUND = 4    # 백그라운드 실행


@dataclass
class ExecutionResult:
    """실행 결과"""
    success: bool
    action_id: str
    result: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None
    side_effects: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ActionExecutor:
    """
    액션 실행 엔진
    
    게임 액션의 실행, 검증, 재시도, 우선순위 관리를 담당
    """
    
    def __init__(self, max_concurrent_actions: int = 10):
        self.max_concurrent_actions = max_concurrent_actions
        self.execution_queue: List[Action] = []
        self.running_actions: Dict[str, Action] = {}
        self.completed_actions: List[ExecutionResult] = []
        self.action_handlers: Dict[ActionType, Callable] = {}
        self.execution_stats = {
            "total_executed": 0,
            "successful": 0,
            "failed": 0,
            "average_execution_time": 0.0
        }
        
        # 기본 액션 핸들러 등록
        self._register_default_handlers()
        
        logger.info("ActionExecutor 초기화 완료")
    
    def _register_default_handlers(self):
        """기본 액션 핸들러 등록"""
        self.register_handler(ActionType.MOVE, self._handle_move_action)
        self.register_handler(ActionType.DRAW_CARD, self._handle_card_action)
        self.register_handler(ActionType.PLAY_CARD, self._handle_card_action)
        self.register_handler(ActionType.VOTE, self._handle_vote_action)
        self.register_handler(ActionType.ROLL_DICE, self._handle_dice_action)
        self.register_handler(ActionType.COLLECT_RESOURCE, self._handle_resource_action)
        self.register_handler(ActionType.BUILD, self._handle_build_action)
        self.register_handler(ActionType.TRADE, self._handle_trade_action)
        self.register_handler(ActionType.ATTACK, self._handle_attack_action)
        self.register_handler(ActionType.DEFEND, self._handle_defend_action)
        self.register_handler(ActionType.ELIMINATE, self._handle_eliminate_action)
        self.register_handler(ActionType.PASS_TURN, self._handle_pass_turn_action)
        self.register_handler(ActionType.SURRENDER, self._handle_surrender_action)
        
        # 시스템 액션 핸들러
        self.register_handler(ActionType.GAME_START, self._handle_game_start_action)
        self.register_handler(ActionType.GAME_END, self._handle_game_end_action)
        self.register_handler(ActionType.TURN_START, self._handle_turn_start_action)
        self.register_handler(ActionType.TURN_END, self._handle_turn_end_action)
        self.register_handler(ActionType.PHASE_CHANGE, self._handle_phase_change_action)
    
    def register_handler(self, action_type: ActionType, handler: Callable):
        """액션 타입별 핸들러 등록"""
        self.action_handlers[action_type] = handler
        logger.debug(f"액션 핸들러 등록: {action_type.value}")
    
    async def submit_action(self, action: Action, priority: ExecutionPriority = ExecutionPriority.NORMAL) -> str:
        """액션 제출"""
        action.priority = priority.value
        
        # 기본 검증
        if not action.validate():
            logger.warning(f"액션 검증 실패: {action.validation_message}")
            return action.action_id
        
        # 큐에 추가
        self.execution_queue.append(action)
        self.execution_queue.sort(key=lambda x: x.priority)
        
        logger.info(f"액션 제출: {action.action_type.value} (ID: {action.action_id})")
        return action.action_id
    
    async def execute_action(self, action: Action) -> ExecutionResult:
        """단일 액션 실행"""
        start_time = datetime.now()
        
        try:
            # 실행 중 상태로 변경
            action.status = ActionStatus.EXECUTING
            self.running_actions[action.action_id] = action
            
            # 핸들러 찾기
            handler = self.action_handlers.get(action.action_type)
            if not handler:
                # 기본 실행
                result = action.execute()
            else:
                # 커스텀 핸들러 실행
                result = await handler(action)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 성공 처리
            action.status = ActionStatus.COMPLETED
            action.completed_at = datetime.now()
            action.result = result
            
            execution_result = ExecutionResult(
                success=True,
                action_id=action.action_id,
                result=result,
                execution_time=execution_time
            )
            
            self.execution_stats["successful"] += 1
            logger.info(f"액션 실행 성공: {action.action_type.value} ({execution_time:.3f}s)")
            
            return execution_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_message = str(e)
            
            # 실패 처리
            action.status = ActionStatus.FAILED
            action.error_message = error_message
            
            execution_result = ExecutionResult(
                success=False,
                action_id=action.action_id,
                result={"success": False, "error": error_message},
                execution_time=execution_time,
                error_message=error_message
            )
            
            self.execution_stats["failed"] += 1
            logger.error(f"액션 실행 실패: {action.action_type.value} - {error_message}")
            logger.debug(traceback.format_exc())
            
            return execution_result
        
        finally:
            # 실행 중 목록에서 제거
            self.running_actions.pop(action.action_id, None)
    
    async def process_queue(self) -> List[ExecutionResult]:
        """큐의 액션들을 처리"""
        results = []
        
        while self.execution_queue and len(self.running_actions) < self.max_concurrent_actions:
            action = self.execution_queue.pop(0)
            result = await self.execute_action(action)
            results.append(result)
            self.completed_actions.append(result)
        
        return results
    
    async def execute_all_pending(self) -> List[ExecutionResult]:
        """모든 대기 중인 액션 실행"""
        results = []
        
        while self.execution_queue:
            batch_results = await self.process_queue()
            results.extend(batch_results)
            
            # 동시 실행 제한으로 인한 대기
            if self.execution_queue and len(self.running_actions) >= self.max_concurrent_actions:
                await asyncio.sleep(0.1)
        
        return results
    
    def get_action_status(self, action_id: str) -> Optional[ActionStatus]:
        """액션 상태 조회"""
        # 실행 중인 액션
        if action_id in self.running_actions:
            return self.running_actions[action_id].status
        
        # 완료된 액션
        for result in self.completed_actions:
            if result.action_id == action_id:
                return ActionStatus.COMPLETED if result.success else ActionStatus.FAILED
        
        # 대기 중인 액션
        for action in self.execution_queue:
            if action.action_id == action_id:
                return action.status
        
        return None
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """실행 통계 반환"""
        total = self.execution_stats["successful"] + self.execution_stats["failed"]
        if total > 0:
            avg_time = sum(r.execution_time for r in self.completed_actions) / total
        else:
            avg_time = 0.0
        
        return {
            **self.execution_stats,
            "average_execution_time": avg_time,
            "queue_size": len(self.execution_queue),
            "running_count": len(self.running_actions),
            "completed_count": len(self.completed_actions)
        }
    
    # 기본 액션 핸들러들
    async def _handle_move_action(self, action: Action) -> Dict[str, Any]:
        """이동 액션 처리"""
        from_pos = action.action_data.get("from_position")
        to_pos = action.action_data.get("to_position")
        
        # 게임 상태 업데이트 로직 (실제 구현에서는 게임 상태를 받아서 처리)
        return {
            "success": True,
            "action_type": "move",
            "from_position": from_pos,
            "to_position": to_pos,
            "message": f"플레이어가 {from_pos}에서 {to_pos}로 이동했습니다"
        }
    
    async def _handle_card_action(self, action: Action) -> Dict[str, Any]:
        """카드 액션 처리"""
        card_id = action.action_data.get("card_id")
        card_name = action.action_data.get("card_name", "알 수 없는 카드")
        
        if action.action_type == ActionType.DRAW_CARD:
            return {
                "success": True,
                "action_type": "draw_card",
                "card_id": card_id,
                "card_name": card_name,
                "message": f"{card_name}을 뽑았습니다"
            }
        else:
            return {
                "success": True,
                "action_type": "play_card",
                "card_id": card_id,
                "card_name": card_name,
                "message": f"{card_name}을 사용했습니다"
            }
    
    async def _handle_vote_action(self, action: Action) -> Dict[str, Any]:
        """투표 액션 처리"""
        target_player = action.action_data.get("target_player")
        vote_type = action.action_data.get("vote_type", "elimination")
        
        return {
            "success": True,
            "action_type": "vote",
            "target_player": target_player,
            "vote_type": vote_type,
            "message": f"{target_player}에 대한 {vote_type} 투표를 진행합니다"
        }
    
    async def _handle_dice_action(self, action: Action) -> Dict[str, Any]:
        """주사위 액션 처리"""
        import random
        dice_count = action.action_data.get("dice_count", 1)
        dice_sides = action.action_data.get("dice_sides", 6)
        
        results = [random.randint(1, dice_sides) for _ in range(dice_count)]
        total = sum(results)
        
        return {
            "success": True,
            "action_type": "roll_dice",
            "dice_count": dice_count,
            "dice_sides": dice_sides,
            "results": results,
            "total": total,
            "message": f"주사위 결과: {results} (합계: {total})"
        }
    
    async def _handle_resource_action(self, action: Action) -> Dict[str, Any]:
        """자원 수집 액션 처리"""
        resource_type = action.action_data.get("resource_type", "unknown")
        amount = action.action_data.get("amount", 1)
        
        return {
            "success": True,
            "action_type": "collect_resource",
            "resource_type": resource_type,
            "amount": amount,
            "message": f"{resource_type} {amount}개를 수집했습니다"
        }
    
    async def _handle_build_action(self, action: Action) -> Dict[str, Any]:
        """건설 액션 처리"""
        building_type = action.action_data.get("building_type", "unknown")
        location = action.action_data.get("location", "unknown")
        
        return {
            "success": True,
            "action_type": "build",
            "building_type": building_type,
            "location": location,
            "message": f"{location}에 {building_type}을 건설했습니다"
        }
    
    async def _handle_trade_action(self, action: Action) -> Dict[str, Any]:
        """거래 액션 처리"""
        offer = action.action_data.get("offer", {})
        request = action.action_data.get("request", {})
        
        return {
            "success": True,
            "action_type": "trade",
            "offer": offer,
            "request": request,
            "message": f"거래 제안: {offer} ↔ {request}"
        }
    
    async def _handle_attack_action(self, action: Action) -> Dict[str, Any]:
        """공격 액션 처리"""
        target = action.action_data.get("target", "unknown")
        weapon = action.action_data.get("weapon", "unknown")
        
        return {
            "success": True,
            "action_type": "attack",
            "target": target,
            "weapon": weapon,
            "message": f"{weapon}으로 {target}을 공격합니다"
        }
    
    async def _handle_defend_action(self, action: Action) -> Dict[str, Any]:
        """방어 액션 처리"""
        defense_type = action.action_data.get("defense_type", "unknown")
        
        return {
            "success": True,
            "action_type": "defend",
            "defense_type": defense_type,
            "message": f"{defense_type} 방어를 시도합니다"
        }
    
    async def _handle_eliminate_action(self, action: Action) -> Dict[str, Any]:
        """제거 액션 처리"""
        target = action.action_data.get("target", "unknown")
        reason = action.action_data.get("reason", "unknown")
        
        return {
            "success": True,
            "action_type": "eliminate",
            "target": target,
            "reason": reason,
            "message": f"{target}을 {reason}으로 제거합니다"
        }
    
    async def _handle_pass_turn_action(self, action: Action) -> Dict[str, Any]:
        """턴 패스 액션 처리"""
        return {
            "success": True,
            "action_type": "pass_turn",
            "message": "턴을 패스합니다"
        }
    
    async def _handle_surrender_action(self, action: Action) -> Dict[str, Any]:
        """항복 액션 처리"""
        return {
            "success": True,
            "action_type": "surrender",
            "message": "게임에서 항복합니다"
        }
    
    # 시스템 액션 핸들러들
    async def _handle_game_start_action(self, action: Action) -> Dict[str, Any]:
        """게임 시작 액션 처리"""
        return {
            "success": True,
            "action_type": "game_start",
            "message": "게임이 시작되었습니다"
        }
    
    async def _handle_game_end_action(self, action: Action) -> Dict[str, Any]:
        """게임 종료 액션 처리"""
        winner = action.action_data.get("winner", "unknown")
        return {
            "success": True,
            "action_type": "game_end",
            "winner": winner,
            "message": f"게임이 종료되었습니다. 승자: {winner}"
        }
    
    async def _handle_turn_start_action(self, action: Action) -> Dict[str, Any]:
        """턴 시작 액션 처리"""
        player_id = action.action_data.get("player_id", "unknown")
        return {
            "success": True,
            "action_type": "turn_start",
            "player_id": player_id,
            "message": f"{player_id}의 턴이 시작되었습니다"
        }
    
    async def _handle_turn_end_action(self, action: Action) -> Dict[str, Any]:
        """턴 종료 액션 처리"""
        player_id = action.action_data.get("player_id", "unknown")
        return {
            "success": True,
            "action_type": "turn_end",
            "player_id": player_id,
            "message": f"{player_id}의 턴이 종료되었습니다"
        }
    
    async def _handle_phase_change_action(self, action: Action) -> Dict[str, Any]:
        """페이즈 변경 액션 처리"""
        from_phase = action.action_data.get("from_phase", "unknown")
        to_phase = action.action_data.get("to_phase", "unknown")
        
        return {
            "success": True,
            "action_type": "phase_change",
            "from_phase": from_phase,
            "to_phase": to_phase,
            "message": f"페이즈가 {from_phase}에서 {to_phase}로 변경되었습니다"
        }


# 싱글톤 인스턴스
_action_executor = None

def get_action_executor() -> ActionExecutor:
    """액션 실행기 싱글톤 인스턴스 반환"""
    global _action_executor
    if _action_executor is None:
        _action_executor = ActionExecutor()
    return _action_executor 