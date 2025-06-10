"""
🧳 Travel Scout Agent Package

시크릿 모드를 활용한 여행 검색 AI 에이전트
가격 조작 없는 진짜 최저가 여행 정보 제공
"""

from .travel_scout_agent import TravelScoutAgent
from .travel_utils import TravelSearchUtils
from .travel_browser_utils import (
    IncognitoBrowserManager,
    HotelSearchAutomator,
    FlightSearchAutomator,
    TravelPriceAnalyzer
)

__version__ = "1.0.0"
__all__ = [
    "TravelScoutAgent", 
    "TravelSearchUtils", 
    "IncognitoBrowserManager",
    "HotelSearchAutomator", 
    "FlightSearchAutomator",
    "TravelPriceAnalyzer"
] 