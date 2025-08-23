"""
에러 핸들링 및 모니터링 시스템
Table Game Mate의 안정성을 위한 통합 에러 관리
"""

import asyncio
import traceback
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

from ..utils.logger import get_logger


class ErrorSeverity(Enum):
    """에러 심각도 레벨"""
    LOW = "low"           # 경고, 성능 저하
    MEDIUM = "medium"     # 기능 제한, 사용자 경험 저하
    HIGH = "high"         # 주요 기능 실패
    CRITICAL = "critical" # 시스템 중단, 데이터 손실


class ErrorCategory(Enum):
    """에러 카테고리"""
    LLM_ERROR = "llm_error"           # LLM 관련 에러
    MCP_ERROR = "mcp_error"           # MCP 서버 에러
    AGENT_ERROR = "agent_error"       # 에이전트 에러
    GAME_LOGIC_ERROR = "game_logic_error"  # 게임 로직 에러
    UI_ERROR = "ui_error"             # UI 관련 에러
    SYSTEM_ERROR = "system_error"     # 시스템 에러
    NETWORK_ERROR = "network_error"   # 네트워크 에러


@dataclass
class ErrorRecord:
    """에러 기록"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    agent_id: Optional[str]
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "agent_id": self.agent_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "context": self.context,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }


class ErrorHandler:
    """통합 에러 핸들러"""
    
    def __init__(self):
        self.logger = get_logger("error_handler")
        self.error_records: List[ErrorRecord] = []
        self.error_callbacks: Dict[ErrorCategory, List[Callable]] = defaultdict(list)
        self.retry_strategies: Dict[ErrorCategory, Dict[str, Any]] = {}
        self.alert_thresholds: Dict[ErrorSeverity, int] = {
            ErrorSeverity.LOW: 10,
            ErrorSeverity.MEDIUM: 5,
            ErrorSeverity.HIGH: 3,
            ErrorSeverity.CRITICAL: 1
        }
        
        # 에러 통계
        self.error_stats = {
            "total_errors": 0,
            "errors_by_severity": defaultdict(int),
            "errors_by_category": defaultdict(int),
            "errors_by_agent": defaultdict(int),
            "resolution_time_avg": 0.0
        }
        
        # 자동 해결 전략
        self._setup_auto_resolution_strategies()
    
    def _setup_auto_resolution_strategies(self):
        """자동 해결 전략 설정"""
        self.retry_strategies = {
            ErrorCategory.LLM_ERROR: {
                "max_retries": 3,
                "retry_delay": 1.0,
                "backoff_multiplier": 2.0,
                "auto_resolve": True
            },
            ErrorCategory.MCP_ERROR: {
                "max_retries": 5,
                "retry_delay": 2.0,
                "backoff_multiplier": 1.5,
                "auto_resolve": True
            },
            ErrorCategory.AGENT_ERROR: {
                "max_retries": 2,
                "retry_delay": 0.5,
                "backoff_multiplier": 1.0,
                "auto_resolve": False
            },
            ErrorCategory.NETWORK_ERROR: {
                "max_retries": 3,
                "retry_delay": 3.0,
                "backoff_multiplier": 2.0,
                "auto_resolve": True
            }
        }
    
    async def handle_error(
        self,
        error: Exception,
        severity: ErrorSeverity,
        category: ErrorCategory,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        auto_retry: bool = True
    ) -> str:
        """에러 처리 및 기록"""
        error_id = f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.error_records)}"
        
        # 에러 기록 생성
        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            agent_id=agent_id,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
        
        # 에러 기록 저장
        self.error_records.append(error_record)
        
        # 통계 업데이트
        self._update_error_stats(error_record)
        
        # 로깅
        self.logger.error(
            f"Error occurred: {error_id}",
            error_type=error_record.error_type,
            severity=severity.value,
            category=category.value,
            agent_id=agent_id,
            message=str(error)
        )
        
        # 콜백 실행
        await self._execute_error_callbacks(error_record)
        
        # 자동 재시도
        if auto_retry and self._should_auto_retry(error_record):
            await self._schedule_retry(error_record)
        
        # 알림 임계값 확인
        self._check_alert_thresholds(severity)
        
        return error_id
    
    def _update_error_stats(self, error_record: ErrorRecord):
        """에러 통계 업데이트"""
        self.error_stats["total_errors"] += 1
        self.error_stats["errors_by_severity"][error_record.severity.value] += 1
        self.error_stats["errors_by_category"][error_record.category.value] += 1
        
        if error_record.agent_id:
            self.error_stats["errors_by_agent"][error_record.agent_id] += 1
    
    async def _execute_error_callbacks(self, error_record: ErrorRecord):
        """에러 콜백 실행"""
        callbacks = self.error_callbacks.get(error_record.category, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error_record)
                else:
                    callback(error_record)
            except Exception as e:
                self.logger.error(f"Error callback execution failed: {e}")
    
    def _should_auto_retry(self, error_record: ErrorRecord) -> bool:
        """자동 재시도 여부 확인"""
        strategy = self.retry_strategies.get(error_record.category, {})
        return (
            strategy.get("auto_resolve", False) and
            error_record.retry_count < strategy.get("max_retries", 0)
        )
    
    async def _schedule_retry(self, error_record: ErrorRecord):
        """재시도 스케줄링"""
        strategy = self.retry_strategies.get(error_record.category, {})
        delay = strategy.get("retry_delay", 1.0) * (strategy.get("backoff_multiplier", 1.0) ** error_record.retry_count)
        
        asyncio.create_task(self._retry_operation(error_record, delay))
    
    async def _retry_operation(self, error_record: ErrorRecord, delay: float):
        """재시도 작업 실행"""
        await asyncio.sleep(delay)
        
        try:
            # 재시도 로직 구현 (에러 타입별로 다르게 처리)
            if error_record.category == ErrorCategory.LLM_ERROR:
                await self._retry_llm_operation(error_record)
            elif error_record.category == ErrorCategory.MCP_ERROR:
                await self._retry_mcp_operation(error_record)
            
            # 성공 시 해결된 것으로 표시
            error_record.resolved = True
            error_record.resolution_time = datetime.now()
            
        except Exception as e:
            error_record.retry_count += 1
            if error_record.retry_count >= error_record.max_retries:
                self.logger.error(f"Max retries exceeded for error {error_record.error_id}")
            else:
                # 재시도 실패 시 다시 스케줄링
                await self._schedule_retry(error_record)
    
    async def _retry_llm_operation(self, error_record: ErrorRecord):
        """LLM 작업 재시도"""
        # LLM 재연결 또는 프롬프트 재시도 로직
        pass
    
    async def _retry_mcp_operation(self, error_record: ErrorRecord):
        """MCP 작업 재시도"""
        # MCP 서버 재연결 로직
        pass
    
    def _check_alert_thresholds(self, severity: ErrorSeverity):
        """알림 임계값 확인"""
        threshold = self.alert_thresholds.get(severity, 0)
        current_count = self.error_stats["errors_by_severity"][severity.value]
        
        if current_count >= threshold:
            self._send_alert(severity, current_count)
    
    def _send_alert(self, severity: ErrorSeverity, count: int):
        """알림 전송"""
        alert_message = f"ALERT: {severity.value.upper()} errors exceeded threshold. Current count: {count}"
        self.logger.critical(alert_message)
        
        # 여기에 실제 알림 시스템 연동 (Slack, Email, SMS 등)
        # await self.notification_service.send_alert(alert_message)
    
    def register_error_callback(self, category: ErrorCategory, callback: Callable):
        """에러 콜백 등록"""
        self.error_callbacks[category].append(callback)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """에러 요약 정보"""
        recent_errors = [
            record for record in self.error_records 
            if record.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "total_errors": self.error_stats["total_errors"],
            "recent_errors_24h": len(recent_errors),
            "errors_by_severity": dict(self.error_stats["errors_by_severity"]),
            "errors_by_category": dict(self.error_stats["errors_by_category"]),
            "errors_by_agent": dict(self.error_stats["errors_by_agent"]),
            "unresolved_errors": len([r for r in self.error_records if not r.resolved]),
            "resolution_rate": self._calculate_resolution_rate()
        }
    
    def _calculate_resolution_rate(self) -> float:
        """에러 해결률 계산"""
        if not self.error_records:
            return 0.0
        
        resolved_count = len([r for r in self.error_records if r.resolved])
        return resolved_count / len(self.error_records) * 100
    
    def get_errors_by_agent(self, agent_id: str, limit: int = 10) -> List[ErrorRecord]:
        """특정 에이전트의 에러 기록 조회"""
        agent_errors = [r for r in self.error_records if r.agent_id == agent_id]
        return sorted(agent_errors, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_errors_by_severity(self, severity: ErrorSeverity, limit: int = 10) -> List[ErrorRecord]:
        """특정 심각도의 에러 기록 조회"""
        severity_errors = [r for r in self.error_records if r.severity == severity]
        return sorted(severity_errors, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def mark_error_resolved(self, error_id: str, resolution_notes: str = ""):
        """에러를 해결된 것으로 표시"""
        for record in self.error_records:
            if record.error_id == error_id:
                record.resolved = True
                record.resolution_time = datetime.now()
                record.context["resolution_notes"] = resolution_notes
                break
    
    def clear_old_errors(self, days: int = 30):
        """오래된 에러 기록 정리"""
        cutoff_date = datetime.now() - timedelta(days=days)
        self.error_records = [
            record for record in self.error_records 
            if record.timestamp > cutoff_date
        ]


# 전역 에러 핸들러 인스턴스
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """전역 에러 핸들러 인스턴스 반환"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


# 데코레이터: 에러 자동 처리
def handle_errors(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
    auto_retry: bool = True
):
    """에러 자동 처리 데코레이터"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                error_id = await error_handler.handle_error(
                    error=e,
                    severity=severity,
                    category=category,
                    auto_retry=auto_retry
                )
                raise RuntimeError(f"Error handled: {error_id}. Original error: {str(e)}")
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                # 동기 함수의 경우 비동기 에러 핸들러를 동기적으로 호출
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 이미 실행 중인 루프가 있으면 새 태스크 생성
                    asyncio.create_task(
                        error_handler.handle_error(
                            error=e,
                            severity=severity,
                            category=category,
                            auto_retry=auto_retry
                        )
                    )
                else:
                    # 루프가 없으면 새로 생성하여 실행
                    loop.run_until_complete(
                        error_handler.handle_error(
                            error=e,
                            severity=severity,
                            category=category,
                            auto_retry=auto_retry
                        )
                    )
                raise RuntimeError(f"Error handled. Original error: {str(e)}")
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
