#!/usr/bin/env python3
"""
Travel Scout Agent - MCP-Agent Implementation

A travel search agent using the mcp_agent framework for consistent
integration with the MCP ecosystem.
"""

import asyncio
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from mcp_agent.app import MCPApp
from mcp_agent.config import get_settings
from .mcp_browser_client import MCPBrowserClient
from .scrapers import BookingComScraper, GoogleFlightsScraper
from .travel_utils import TravelSearchUtils

logger = logging.getLogger(__name__)

# ✅ P1-5: Travel Scout 메서드 구현 (5개 함수)

def load_destination_options() -> List[str]:
    """목적지 옵션 로드"""
    return [
        # 아시아 주요 도시
        "Seoul (서울)",
        "Tokyo (도쿄)",
        "Osaka (오사카)",
        "Bangkok (방콕)",
        "Singapore (싱가포르)",
        "Hong Kong (홍콩)",
        "Shanghai (상하이)",
        "Beijing (베이징)",
        "Taipei (타이베이)",
        "Kuala Lumpur (쿠알라룸푸르)",
        "Manila (마닐라)",
        "Ho Chi Minh City (호치민)",
        "Jakarta (자카르타)",
        
        # 유럽 주요 도시
        "London (런던)",
        "Paris (파리)",
        "Rome (로마)",
        "Barcelona (바르셀로나)",
        "Amsterdam (암스테르담)",
        "Berlin (베를린)",
        "Vienna (비엔나)",
        "Prague (프라하)",
        "Zurich (취리히)",
        "Stockholm (스톡홀름)",
        "Copenhagen (코펜하겐)",
        "Oslo (오슬로)",
        
        # 북미 주요 도시
        "New York (뉴욕)",
        "Los Angeles (로스앤젤레스)",
        "San Francisco (샌프란시스코)",
        "Las Vegas (라스베이거스)",
        "Chicago (시카고)",
        "Miami (마이애미)",
        "Toronto (토론토)",
        "Vancouver (밴쿠버)",
        
        # 오세아니아
        "Sydney (시드니)",
        "Melbourne (멜버른)",
        "Auckland (오클랜드)",
        
        # 중동/아프리카
        "Dubai (두바이)",
        "Istanbul (이스탄불)",
        "Cairo (카이로)",
        "Cape Town (케이프타운)"
    ]

def load_origin_options() -> List[str]:
    """출발지 옵션 로드"""
    return [
        # 한국 주요 도시
        "Seoul (서울)",
        "Busan (부산)",
        "Incheon (인천)",
        "Daegu (대구)",
        "Gwangju (광주)",
        "Daejeon (대전)",
        "Ulsan (울산)",
        "Jeju (제주)",
        
        # 아시아 주요 출발지
        "Tokyo (도쿄)",
        "Osaka (오사카)",
        "Bangkok (방콕)",
        "Singapore (싱가포르)",
        "Hong Kong (홍콩)",
        "Shanghai (상하이)",
        "Beijing (베이징)",
        "Taipei (타이베이)",
        
        # 유럽 주요 출발지
        "London (런던)",
        "Paris (파리)",
        "Frankfurt (프랑크푸르트)",
        "Amsterdam (암스테르담)",
        "Rome (로마)",
        "Barcelona (바르셀로나)",
        
        # 북미 주요 출발지
        "New York (뉴욕)",
        "Los Angeles (로스앤젤레스)",
        "San Francisco (샌프란시스코)",
        "Toronto (토론토)",
        "Vancouver (밴쿠버)",
        
        # 오세아니아
        "Sydney (시드니)",
        "Melbourne (멜버른)",
        
        # 중동
        "Dubai (두바이)",
        "Doha (도하)"
    ]

def get_user_location() -> Dict[str, str]:
    """사용자 위치 기반 기본값 설정"""
    try:
        # 실제 환경에서는 IP 기반 위치 감지 또는 사용자 설정을 사용할 수 있음
        # 현재는 한국을 기본값으로 설정
        default_location = {
            "origin": "Seoul (서울)",
            "country": "South Korea",
            "timezone": "Asia/Seoul",
            "currency": "KRW",
            "language": "ko",
            "detected_method": "default_korean_user"
        }
        
        # 환경 변수나 설정 파일에서 사용자 기본 위치 읽기 시도
        user_origin = os.environ.get('TRAVEL_DEFAULT_ORIGIN', 'Seoul (서울)')
        user_country = os.environ.get('TRAVEL_DEFAULT_COUNTRY', 'South Korea')
        
        return {
            "origin": user_origin,
            "country": user_country,
            "timezone": os.environ.get('TRAVEL_DEFAULT_TIMEZONE', 'Asia/Seoul'),
            "currency": os.environ.get('TRAVEL_DEFAULT_CURRENCY', 'KRW'),
            "language": os.environ.get('TRAVEL_DEFAULT_LANGUAGE', 'ko'),
            "detected_method": "environment_variable" if user_origin != 'Seoul (서울)' else "default_korean_user",
            "available_origins": load_origin_options(),
            "available_destinations": load_destination_options()
        }
        
    except Exception as e:
        # 에러 발생 시 안전한 기본값 반환
        return {
            "origin": "Seoul (서울)",
            "country": "South Korea", 
            "timezone": "Asia/Seoul",
            "currency": "KRW",
            "language": "ko",
            "detected_method": "fallback_default",
            "error": str(e)
        }

def save_travel_report(content: str, filename: str) -> str:
    """여행 검색 보고서를 파일로 저장"""
    try:
        # 설정에서 보고서 경로 가져오기
        try:
            from configs.settings import get_reports_path
            reports_dir = get_reports_path('travel_scout')
        except ImportError:
            reports_dir = "travel_scout_reports"
        
        # 디렉토리 생성
        os.makedirs(reports_dir, exist_ok=True)
        
        # 파일명에 타임스탬프 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not filename.endswith('.md'):
            filename = f"{filename}_{timestamp}.md"
        
        file_path = os.path.join(reports_dir, filename)
        
        # 보고서 헤더 생성
        report_header = f"""# 🧳 Travel Scout Search Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Agent Type**: Travel Scout MCP Agent  
**Report ID**: travel_search_{timestamp}  
**Data Source**: MCP Browser + Real-time Travel Sites

---

"""
        
        # 메타데이터 생성
        metadata = {
            "report_id": f"travel_search_{timestamp}",
            "generated_at": datetime.now().isoformat(),
            "agent_type": "Travel Scout MCP Agent",
            "data_source": "MCP Browser + Real-time Travel Sites",
            "content_length": len(content),
            "file_path": file_path,
            "user_location": get_user_location(),
            "destination_options": load_destination_options(),
            "origin_options": load_origin_options(),
            "report_sections": [
                "Search Summary",
                "Hotel Results",
                "Flight Results", 
                "Price Analysis",
                "Recommendations",
                "Booking Strategy",
                "Total Cost Estimate"
            ]
        }
        
        # Markdown 보고서 저장
        full_content = report_header + content
        
        # 보고서 메타데이터 추가
        full_content += f"\n\n---\n\n### Report Metadata\n\n```json\n{json.dumps(metadata, indent=2, ensure_ascii=False)}\n```"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        # 메타데이터 JSON 저장
        metadata_file = file_path.replace('.md', '_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return file_path
        
    except Exception as e:
        raise Exception(f"여행 보고서 저장 실패: {str(e)}")

def generate_travel_report_content(results: dict, search_params: dict) -> str:
    """여행 검색 보고서 내용 생성"""
    try:
        # 기본 정보 추출
        hotels = results.get('hotels', [])
        flights = results.get('flights', [])
        recommendations = results.get('recommendations', {})
        analysis = results.get('analysis', {})
        performance = results.get('performance', {})
        
        # 검색 매개변수 추출
        destination = search_params.get('destination', 'Unknown')
        origin = search_params.get('origin', 'Unknown')
        check_in = search_params.get('check_in', 'N/A')
        check_out = search_params.get('check_out', 'N/A')
        
        # 보고서 내용 생성
        report_content = f"""
## 🎯 Search Summary

**Destination**: {destination}  
**Origin**: {origin}  
**Check-in**: {check_in}  
**Check-out**: {check_out}  
**Search Duration**: {performance.get('total_duration', 0):.1f} seconds  
**Data Source**: MCP Browser (Real-time)

---

## 📊 Search Results Overview

| Category | Found | Status |
|----------|-------|--------|
| 🏨 Hotels | {len(hotels)} | {'✅ Found' if hotels else '❌ None'} |
| ✈️ Flights | {len(flights)} | {'✅ Found' if flights else '❌ None'} |
| 💡 Recommendations | {'✅ Available' if recommendations else '❌ None'} | {'Generated' if recommendations else 'Not Available'} |

---

## 🏨 Hotel Search Results

"""
        
        # 호텔 결과 추가
        if hotels:
            report_content += f"Found **{len(hotels)} hotels** matching your criteria:\n\n"
            
            for i, hotel in enumerate(hotels[:10], 1):  # 상위 10개만 표시
                price = hotel.get('price', 'N/A')
                rating = hotel.get('rating', 'N/A')
                location = hotel.get('location', 'N/A')
                platform = hotel.get('platform', 'N/A')
                
                report_content += f"{i}. **{hotel.get('name', 'Unknown Hotel')}**\n"
                report_content += f"   - 💰 Price: {price}\n"
                report_content += f"   - ⭐ Rating: {rating}\n"
                report_content += f"   - 📍 Location: {location}\n"
                report_content += f"   - 🌐 Platform: {platform}\n\n"
                
            # 호텔 가격 분석
            if 'hotel_analysis' in analysis:
                hotel_analysis = analysis['hotel_analysis']
                report_content += f"""
### 📈 Hotel Price Analysis

- **Average Rating**: {hotel_analysis.get('average_rating', 0):.1f}/5.0
- **Average Price**: ${hotel_analysis.get('price_range', {}).get('average', 0):.0f}/night
- **Quality Hotels**: {hotel_analysis.get('quality_hotels_count', 0)} hotels meet criteria
- **Price Range**: ${hotel_analysis.get('price_range', {}).get('min', 0)} - ${hotel_analysis.get('price_range', {}).get('max', 0)}

"""
        else:
            report_content += "❌ No hotels found matching your search criteria.\n\n"
        
        # 항공편 결과 추가
        report_content += "## ✈️ Flight Search Results\n\n"
        
        if flights:
            report_content += f"Found **{len(flights)} flights** for your travel dates:\n\n"
            
            for i, flight in enumerate(flights[:10], 1):  # 상위 10개만 표시
                airline = flight.get('airline', 'Unknown Airline')
                price = flight.get('price', 'N/A')
                duration = flight.get('duration', 'N/A')
                departure_time = flight.get('departure_time', 'N/A')
                platform = flight.get('platform', 'N/A')
                
                report_content += f"{i}. **{airline}**\n"
                report_content += f"   - 💰 Price: {price}\n"
                report_content += f"   - ⏱️ Duration: {duration}\n"
                report_content += f"   - 🛫 Departure: {departure_time}\n"
                report_content += f"   - 🌐 Platform: {platform}\n\n"
                
            # 항공편 가격 분석
            if 'flight_analysis' in analysis:
                flight_analysis = analysis['flight_analysis']
                report_content += f"""
### 📈 Flight Price Analysis

- **Average Price**: ${flight_analysis.get('price_range', {}).get('average', 0):.0f}
- **Airlines Found**: {len(flight_analysis.get('airlines_found', []))} airlines
- **Quality Flights**: {flight_analysis.get('quality_flights_count', 0)} flights meet criteria
- **Price Range**: ${flight_analysis.get('price_range', {}).get('min', 0)} - ${flight_analysis.get('price_range', {}).get('max', 0)}

"""
        else:
            report_content += "❌ No flights found for your travel dates.\n\n"
        
        # 추천 사항 추가
        report_content += "## 💡 Recommendations\n\n"
        
        if recommendations:
            # 최고 호텔 추천
            if 'best_hotel' in recommendations:
                hotel = recommendations['best_hotel']
                report_content += f"""
### 🏨 Recommended Hotel

**{hotel.get('name', 'N/A')}**
- 💰 Price: {hotel.get('price', 'N/A')}
- ⭐ Rating: {hotel.get('rating', 'N/A')}
- 📍 Location: {hotel.get('location', 'N/A')}

"""
            
            # 최고 항공편 추천
            if 'best_flight' in recommendations:
                flight = recommendations['best_flight']
                report_content += f"""
### ✈️ Recommended Flight

**{flight.get('airline', 'N/A')}**
- 💰 Price: {flight.get('price', 'N/A')}
- ⏱️ Duration: {flight.get('duration', 'N/A')}
- 🛫 Departure: {flight.get('departure_time', 'N/A')}

"""
            
            # 예약 전략
            if 'booking_strategy' in recommendations:
                report_content += "### 📋 Booking Strategy\n\n"
                for strategy in recommendations['booking_strategy']:
                    report_content += f"• {strategy}\n"
                report_content += "\n"
            
            # 총 비용 추정
            if 'total_trip_cost_estimate' in recommendations:
                cost = recommendations['total_trip_cost_estimate']
                report_content += f"""
### 💰 Total Trip Cost Estimate

| Item | Cost | Details |
|------|------|---------|
| 🏨 Hotel | ${cost.get('hotel_total', 0)} | ${cost.get('hotel_per_night', 0)}/night × {cost.get('nights', 0)} nights |
| ✈️ Flight | ${cost.get('flight_total', 0)} | Round-trip airfare |
| **💳 Total** | **${cost.get('grand_total', 0)}** | **Complete trip cost** |

"""
        else:
            report_content += "ℹ️ No specific recommendations available.\n\n"
        
        # 검색 성능 및 데이터 소스
        report_content += f"""
---

## 🔍 Search Performance & Data Sources

### Performance Metrics
- **Search Duration**: {performance.get('total_duration', 0):.1f} seconds
- **Hotels Found**: {performance.get('hotels_found', 0)} results
- **Flights Found**: {performance.get('flights_found', 0)} results

### Data Sources
- **MCP Browser**: Real-time data collection ✅
- **Travel Platforms**: Live search results ✅
- **Price Comparison**: Cross-platform analysis ✅

---

## 📞 Next Steps

1. **Book Early**: Prices may change - consider booking soon
2. **Compare Platforms**: Check original websites for final prices
3. **Read Reviews**: Check recent reviews before booking
4. **Travel Insurance**: Consider purchasing travel insurance
5. **Check Requirements**: Verify visa/passport requirements

---

*This report was generated by Travel Scout MCP Agent using real-time data from travel platforms.*
*Search performed on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        return report_content
        
    except Exception as e:
        # 에러 발생 시 기본 보고서 생성
        return f"""
# 🧳 Travel Scout Search Report

**Analysis Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Search Parameters**: {search_params}

## ⚠️ Report Generation Error

An error occurred while generating the detailed report: {str(e)}

## Basic Search Results

- **Hotels Found**: {len(results.get('hotels', []))} results
- **Flights Found**: {len(results.get('flights', []))} results
- **Search Status**: {results.get('status', 'Unknown')}

## Raw Data

```json
{json.dumps(results, indent=2, default=str, ensure_ascii=False)}
```

---

*Please check the Travel Scout MCP Agent configuration and try again.*
"""

class TravelScoutAgent:
    """Travel Scout Agent - Refactored for clarity and stability."""
    
    def __init__(self, config: Optional[Dict] = None, browser_client: Optional[MCPBrowserClient] = None):
        """에이전트 초기화. 연결 확인 로직 제거."""
        self.mcp_client = browser_client or MCPBrowserClient()
        self.booking_scraper = BookingComScraper(self.mcp_client)
        self.flights_scraper = GoogleFlightsScraper(self.mcp_client)
        self.config = config
        logger.info("TravelScoutAgent initialized without immediate connection checks.")

    async def _ensure_mcp_connection(self):
        """MCP 서버 연결을 확인하고, 필요시 연결."""
        if not self.mcp_client.is_connected():
            logger.info("MCP client not connected. Attempting to connect...")
            connected = await self.mcp_client.connect_to_mcp_server()
            if not connected:
                raise ConnectionError("Failed to connect to MCP server.")
            logger.info("Successfully connected to MCP server.")

    async def search_travel_options(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """주요 여행 옵션 검색 (호텔 및 항공편)"""
        start_time = time.time()
        search_type = search_params.get("search_type", "all")
        destination = search_params.get("destination", "Unknown")
        logger.info(f"🔍 Starting MCP travel search for {destination} (type: {search_type})")

        try:
            await self._ensure_mcp_connection()
            self.mcp_client.clear_screenshots()

            tasks = []
            if search_type in ["hotel", "all"]:
                tasks.append(self.booking_scraper.search(search_params))
            
            if search_type in ["flight", "all"]:
                tasks.append(self.flights_scraper.search(search_params))

            if not tasks:
                return {"error": "Invalid search type provided."}

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 처리
            hotels = []
            flights = []
            
            for res in results:
                if isinstance(res, Exception):
                    logger.error(f"Search task failed: {res}")
                    continue
                if res.get("type") == "hotel":
                    hotels = res.get("data", [])
                elif res.get("type") == "flight":
                    flights = res.get("data", [])

            performance = {
                "total_duration_seconds": round(time.time() - start_time, 2),
                "hotel_search_successful": any(res.get("type") == "hotel" and not isinstance(res, Exception) for res in results),
                "flight_search_successful": any(res.get("type") == "flight" and not isinstance(res, Exception) for res in results),
            }

            return {
                "hotels": hotels,
                "flights": flights,
                "performance": performance,
                "search_params": search_params
            }

        except Exception as e:
            logger.error(f"❌ MCP travel search failed: {e}", exc_info=True)
            return {"error": str(e), "hotels": [], "flights": []}

    async def search_hotels(self, destination: str, check_in: str, check_out: str, guests: int = 2) -> Dict[str, Any]:
        """호텔 검색 실행"""
        search_params = {
            "destination": destination,
            "check_in": check_in,
            "check_out": check_out,
            "guests": guests,
            "search_type": "hotel"
        }
        return await self.search_travel_options(search_params)

    async def search_flights(self, origin: str, destination: str, departure_date: str, return_date: str = None) -> Dict[str, Any]:
        """항공편 검색 실행"""
        search_params = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date,
            "search_type": "flight"
        }
        return await self.search_travel_options(search_params)

    async def cleanup(self):
        """에이전트 리소스 정리"""
        logger.info("Cleaning up TravelScoutAgent resources...")
        if self.mcp_client:
            await self.mcp_client.cleanup()


# Convenience function for direct usage
async def search_travel(destination: str, check_in: str, check_out: str) -> Dict[str, Any]:
    """Search for travel options using TravelScoutAgent"""
    agent = TravelScoutAgent()
    return await agent.search_travel_options({'destination': destination, 'check_in': check_in, 'check_out': check_out})


if __name__ == "__main__":
    # Command line usage
    DESTINATION = "Tokyo" if len(sys.argv) <= 1 else sys.argv[1]
    CHECK_IN = "2025-08-01" if len(sys.argv) <= 2 else sys.argv[2]
    CHECK_OUT = "2025-08-05" if len(sys.argv) <= 3 else sys.argv[3]
    
    print(f"🧳 Travel Scout Agent - MCP Browser Mode")
    print(f"📍 Destination: {DESTINATION}")
    print(f"📅 Check-in: {CHECK_IN}")
    print(f"📅 Check-out: {CHECK_OUT}")
    print("-" * 50)
    
    async def run_search():
        print("🔌 Initializing MCP connection...")
        agent = TravelScoutAgent()
        try:
            start_time = datetime.now()
            result = await agent.search_travel_options({'destination': DESTINATION, 'check_in': CHECK_IN, 'check_out': CHECK_OUT})
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            print(f"\n⏱️  Total execution time: {duration:.2f} seconds")
            
            if result.get("status") == "completed":
                print("✅ Travel search completed successfully!")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print("❌ Travel search failed!")
                if "error" in result:
                    print(f"Error: {result['error']}")
        finally:
            await agent.cleanup()
    
    asyncio.run(run_search())