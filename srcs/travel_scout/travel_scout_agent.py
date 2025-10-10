#!/usr/bin/env python3
"""
Travel Scout Agent - streamlined MCP integration

Provides a thin orchestration layer over `MCPBrowserClient` and concrete
scrapers so the CLI runner can call a consistent API.
"""
import logging
from typing import Dict, Any, Optional
from .mcp_browser_client import MCPBrowserClient
from .scrapers import BookingComScraper, GoogleFlightsScraper
from .ai_analyzer import TravelAIAnalyzer
from .utils import TravelSearchUtils

# Re-export commonly used helper functions so that UI code can import directly
from .utils import load_destination_options, load_origin_options  # noqa: F401

__all__ = [
    "TravelScoutAgent",
    "load_destination_options",
    "load_origin_options",
]


class TravelScoutAgent:
    """Orchestrates travel searches using MCP-controlled browser with AI analysis."""

    def __init__(self, browser_client: MCPBrowserClient):
        self.browser_client = browser_client
        self.ai_analyzer = TravelAIAnalyzer()
        self.logger = logging.getLogger(__name__)

    async def search_hotels(
        self,
        destination: str,
        check_in: str,
        check_out: str,
        guests: int = 2,
    ) -> Dict[str, Any]:
        """Run a hotel search via Booking.com scraper with AI analysis."""
        # 파라미터 검증
        is_valid, error_msg = TravelSearchUtils.validate_search_params(
            destination, check_in, check_out, check_in, check_out
        )
        if not is_valid:
            raise ValueError(f"검색 파라미터 오류: {error_msg}")
        
        scraper = BookingComScraper(self.browser_client)
        search_params: Dict[str, Any] = {
            "destination": destination,
            "check_in": check_in,
            "check_out": check_out,
            "guests": guests,
        }
        
        # 스크레이핑 실행
        raw_results = await scraper.search(search_params)
        hotels_data = raw_results.get("data", [])
        
        # AI 분석 실행
        ai_analysis = await self.ai_analyzer.analyze_hotel_data(hotels_data, search_params)
        
        # 결과 통합
        return {
            "type": "hotel",
            "data": hotels_data,
            "ai_analysis": ai_analysis,
            "search_params": search_params,
            "total_found": len(hotels_data)
        }

    async def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a flight search via Google Flights scraper with AI analysis."""
        if not return_date:
            raise ValueError("항공편 검색에는 왕복 날짜가 필요합니다")
        
        # 파라미터 검증
        is_valid, error_msg = TravelSearchUtils.validate_search_params(
            destination, departure_date, return_date, departure_date, return_date
        )
        if not is_valid:
            raise ValueError(f"검색 파라미터 오류: {error_msg}")
        
        scraper = GoogleFlightsScraper(self.browser_client)
        search_params: Dict[str, Any] = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date,
        }
        
        # 스크레이핑 실행
        raw_results = await scraper.search(search_params)
        flights_data = raw_results.get("data", [])
        
        # AI 분석 실행
        ai_analysis = await self.ai_analyzer.analyze_flight_data(flights_data, search_params)
        
        # 결과 통합
        return {
            "type": "flight",
            "data": flights_data,
            "ai_analysis": ai_analysis,
            "search_params": search_params,
            "total_found": len(flights_data)
        }

    async def search_complete_travel(
        self,
        origin: str,
        destination: str,
        check_in: str,
        check_out: str,
        guests: int = 2,
    ) -> Dict[str, Any]:
        """Complete travel search with both hotels and flights, plus AI recommendations."""
        try:
            # 호텔 검색
            hotel_results = await self.search_hotels(destination, check_in, check_out, guests)
            
            # 항공편 검색
            flight_results = await self.search_flights(origin, destination, check_in, check_out)
            
            # 통합 AI 추천 생성
            combined_recommendations = await self.ai_analyzer.generate_travel_recommendations(
                hotel_results.get("ai_analysis", {}),
                flight_results.get("ai_analysis", {}),
                {
                    "origin": origin,
                    "destination": destination,
                    "check_in": check_in,
                    "check_out": check_out,
                    "guests": guests
                }
            )
            
            return {
                "success": True,
                "hotels": hotel_results,
                "flights": flight_results,
                "recommendations": combined_recommendations,
                "search_summary": {
                    "origin": origin,
                    "destination": destination,
                    "dates": f"{check_in} to {check_out}",
                    "guests": guests,
                    "total_hotels": hotel_results.get("total_found", 0),
                    "total_flights": flight_results.get("total_found", 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"통합 여행 검색 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "hotels": {"data": [], "ai_analysis": {}},
                "flights": {"data": [], "ai_analysis": {}},
                "recommendations": {}
            }

    async def cleanup(self) -> None:
        """Release browser resources held by the underlying MCP session."""
        await self.browser_client.cleanup()
