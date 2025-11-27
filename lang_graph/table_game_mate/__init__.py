"""
테이블게임 메이트 에이전트 시스템
LangGraph 기반 다중 에이전트 보드게임 플랫폼
"""

from .core import GameState, GameConfig, Player, GamePhase, ErrorHandler
# agents 모듈은 아직 구현되지 않았으므로 import 제거
# from .agents import GameAgent, AnalysisAgent, MonitoringAgent
from .utils.game_factory import GameFactory
from .utils.logger import get_logger

# 임시로 빈 클래스 정의 (나중에 구현 예정)
class GameAgent:
    pass

class AnalysisAgent:
    pass

class MonitoringAgent:
    pass

__version__ = "0.1.0"
__all__ = [
    "GameState",
    "GameConfig", 
    "Player",
    "GamePhase",
    "ErrorHandler",
    "GameAgent",
    "AnalysisAgent",
    "MonitoringAgent",
    "GameFactory",
    "get_logger",
] 