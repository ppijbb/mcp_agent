"""
통합 로깅 시스템

프로젝트 전체의 로깅을 관리하는 중앙화된 시스템
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
from functools import wraps
import traceback
import asyncio


class ColoredFormatter(logging.Formatter):
    """컬러 로그 포매터"""
    
    # ANSI 색상 코드
    COLORS = {
        'DEBUG': '\033[36m',      # 청록색
        'INFO': '\033[32m',       # 녹색
        'WARNING': '\033[33m',    # 노란색
        'ERROR': '\033[31m',      # 빨간색
        'CRITICAL': '\033[35m',   # 자홍색
        'RESET': '\033[0m'        # 리셋
    }
    
    def format(self, record):
        # 원본 메시지 가져오기 (포맷팅 전)
        if record.args:
            # 포맷팅이 필요한 경우
            original_msg = record.msg % record.args
        else:
            # 포맷팅이 필요 없는 경우
            original_msg = record.getMessage()
        
        # 색상 적용
        if record.levelname in self.COLORS:
            colored_msg = f"{self.COLORS[record.levelname]}{original_msg}{self.COLORS['RESET']}"
            # record.msg를 수정하지 말고 임시로 저장
            record.msg = colored_msg
            record.args = ()  # args를 비워서 재포맷팅 방지
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON 형태 로그 포매터"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 예외 정보 추가
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 추가 필드들
        if hasattr(record, 'agent_id'):
            log_entry['agent_id'] = record.agent_id
        if hasattr(record, 'game_id'):
            log_entry['game_id'] = record.game_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        
        return json.dumps(log_entry, ensure_ascii=False)


class GameLogger:
    """게임 전용 로거"""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 로거 생성
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 중복 핸들러 방지
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """핸들러 설정"""
        # 콘솔 핸들러 (컬러) - stderr로 리다이렉트 (MCP JSONRPC 파서와 충돌 방지)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 파일 핸들러 (일반)
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # JSON 파일 핸들러
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_json.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        json_handler.setLevel(logging.DEBUG)
        json_formatter = JSONFormatter()
        json_handler.setFormatter(json_formatter)
        self.logger.addHandler(json_handler)
    
    def log_with_context(self, level: str, message: str, **context):
        """컨텍스트와 함께 로그 기록"""
        extra = {
            'agent_id': context.get('agent_id'),
            'game_id': context.get('game_id'),
            'session_id': context.get('session_id')
        }
        
        # None 값 제거
        extra = {k: v for k, v in extra.items() if v is not None}
        
        log_method = getattr(self.logger, level.lower())
        log_method(message, extra=extra)
    
    def debug(self, message: str, **context):
        self.log_with_context('DEBUG', message, **context)
    
    def info(self, message: str, **context):
        self.log_with_context('INFO', message, **context)
    
    def warning(self, message: str, **context):
        self.log_with_context('WARNING', message, **context)
    
    def error(self, message: str, **context):
        self.log_with_context('ERROR', message, **context)
    
    def critical(self, message: str, **context):
        self.log_with_context('CRITICAL', message, **context)
    
    def exception(self, message: str, **context):
        """예외 정보와 함께 로그 기록"""
        context['exception'] = traceback.format_exc()
        self.log_with_context('ERROR', message, **context)


class PerformanceLogger:
    """성능 측정 로거"""
    
    def __init__(self, name: str = "performance"):
        self.logger = GameLogger(name)
        self.timers: Dict[str, datetime] = {}
    
    def start_timer(self, timer_name: str):
        """타이머 시작"""
        self.timers[timer_name] = datetime.now()
        self.logger.debug(f"타이머 시작: {timer_name}")
    
    def end_timer(self, timer_name: str, **context):
        """타이머 종료 및 로그"""
        if timer_name in self.timers:
            start_time = self.timers[timer_name]
            duration = (datetime.now() - start_time).total_seconds()
            del self.timers[timer_name]
            
            self.logger.info(
                f"타이머 완료: {timer_name} ({duration:.3f}초)",
                duration=duration,
                **context
            )
            return duration
        else:
            self.logger.warning(f"타이머를 찾을 수 없음: {timer_name}")
            return None
    
    def log_performance(self, operation: str, duration: float, **context):
        """성능 로그 기록"""
        self.logger.info(
            f"성능 측정: {operation} ({duration:.3f}초)",
            operation=operation,
            duration=duration,
            **context
        )
    
    def timer(self, timer_name: str):
        """타이머 컨텍스트 매니저"""
        return TimerContext(self, timer_name)


class TimerContext:
    """타이머 컨텍스트 매니저"""
    
    def __init__(self, logger: PerformanceLogger, timer_name: str):
        self.logger = logger
        self.timer_name = timer_name
    
    def __enter__(self):
        self.logger.start_timer(self.timer_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.end_timer(self.timer_name)


class AgentLogger:
    """에이전트 전용 로거"""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.logger = GameLogger(f"agent.{agent_type}.{agent_id}")
        self.performance_logger = PerformanceLogger(f"agent_performance.{agent_id}")
    
    def log_agent_action(self, action: str, details: Dict[str, Any] = None):
        """에이전트 액션 로그"""
        message = f"에이전트 액션: {action}"
        if details:
            message += f" - {json.dumps(details, ensure_ascii=False)}"
        
        self.logger.info(message, agent_id=self.agent_id, action=action)
    
    def log_agent_decision(self, decision: str, reasoning: str = None):
        """에이전트 의사결정 로그"""
        message = f"에이전트 의사결정: {decision}"
        if reasoning:
            message += f" - 이유: {reasoning}"
        
        self.logger.info(message, agent_id=self.agent_id, decision=decision)
    
    def log_agent_error(self, error: str, context: Dict[str, Any] = None):
        """에이전트 에러 로그"""
        self.logger.error(
            f"에이전트 에러: {error}",
            agent_id=self.agent_id,
            error_context=context
        )
    
    def start_operation_timer(self, operation: str):
        """작업 타이머 시작"""
        self.performance_logger.start_timer(f"{self.agent_id}_{operation}")
    
    def end_operation_timer(self, operation: str):
        """작업 타이머 종료"""
        return self.performance_logger.end_timer(f"{self.agent_id}_{operation}")


class GameSessionLogger:
    """게임 세션 전용 로거"""
    
    def __init__(self, session_id: str, game_name: str):
        self.session_id = session_id
        self.game_name = game_name
        self.logger = GameLogger(f"game.{game_name}.{session_id}")
        self.performance_logger = PerformanceLogger(f"game_performance.{session_id}")
    
    def log_game_event(self, event: str, details: Dict[str, Any] = None):
        """게임 이벤트 로그"""
        message = f"게임 이벤트: {event}"
        if details:
            message += f" - {json.dumps(details, ensure_ascii=False)}"
        
        self.logger.info(message, session_id=self.session_id, event=event)
    
    def log_player_action(self, player_id: str, action: str, details: Dict[str, Any] = None):
        """플레이어 액션 로그"""
        message = f"플레이어 액션: {player_id} - {action}"
        if details:
            message += f" - {json.dumps(details, ensure_ascii=False)}"
        
        self.logger.info(message, session_id=self.session_id, player_id=player_id, action=action)
    
    def log_game_state_change(self, from_state: str, to_state: str, reason: str = None):
        """게임 상태 변경 로그"""
        message = f"게임 상태 변경: {from_state} → {to_state}"
        if reason:
            message += f" - 이유: {reason}"
        
        self.logger.info(message, session_id=self.session_id, state_change=f"{from_state}→{to_state}")
    
    def log_game_error(self, error: str, context: Dict[str, Any] = None):
        """게임 에러 로그"""
        self.logger.error(
            f"게임 에러: {error}",
            session_id=self.session_id,
            error_context=context
        )


# 전역 로거 인스턴스들
_loggers: Dict[str, GameLogger] = {}
_performance_logger = PerformanceLogger()
_agent_loggers: Dict[str, AgentLogger] = {}
_session_loggers: Dict[str, GameSessionLogger] = {}


def get_logger(name: str) -> GameLogger:
    """로거 인스턴스 반환"""
    if name not in _loggers:
        _loggers[name] = GameLogger(name)
    return _loggers[name]


def get_performance_logger() -> PerformanceLogger:
    """성능 로거 반환"""
    return _performance_logger


def get_agent_logger(agent_id: str, agent_type: str) -> AgentLogger:
    """에이전트 로거 반환"""
    key = f"{agent_type}.{agent_id}"
    if key not in _agent_loggers:
        _agent_loggers[key] = AgentLogger(agent_id, agent_type)
    return _agent_loggers[key]


def get_session_logger(session_id: str, game_name: str) -> GameSessionLogger:
    """세션 로거 반환"""
    if session_id not in _session_loggers:
        _session_loggers[session_id] = GameSessionLogger(session_id, game_name)
    return _session_loggers[session_id]


def log_function_call(func_name: str = None):
    """함수 호출 로깅 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = func_name or func.__name__
            logger = get_logger("function_calls")
            logger.info(f"함수 호출 시작: {name}")
            
            start_time = datetime.now()
            try:
                result = await func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"함수 호출 완료: {name} ({duration:.3f}초)")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"함수 호출 실패: {name} ({duration:.3f}초) - {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = func_name or func.__name__
            logger = get_logger("function_calls")
            logger.info(f"함수 호출 시작: {name}")
            
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"함수 호출 완료: {name} ({duration:.3f}초)")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"함수 호출 실패: {name} ({duration:.3f}초) - {e}")
                raise
        
        # 비동기 함수인지 확인
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_exceptions(logger_name: str = "exceptions"):
    """예외 로깅 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"예외 발생: {func.__name__} - {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"예외 발생: {func.__name__} - {e}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# 로그 레벨 설정 함수
def set_log_level(level: str):
    """전역 로그 레벨 설정"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"유효하지 않은 로그 레벨: {level}")
    
    for logger_instance in _loggers.values():
        logger_instance.logger.setLevel(numeric_level)
    
    # 루트 로거도 설정
    logging.getLogger().setLevel(numeric_level)


# 로그 초기화
def initialize_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """로깅 시스템 초기화"""
    # 로그 디렉토리 생성
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # 로그 레벨 설정
    set_log_level(log_level)
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 기본 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 기본 핸들러 추가 - stderr로 리다이렉트 (MCP JSONRPC 파서와 충돌 방지)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 기본 로거 생성
    get_logger("system")
    get_logger("game_master")
    get_logger("agents")
    get_logger("performance")
    
    logging.info("로깅 시스템 초기화 완료")


# 자동 초기화
if not _loggers:
    initialize_logging() 