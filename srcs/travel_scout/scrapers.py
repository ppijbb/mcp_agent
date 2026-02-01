#!/usr/bin/env python3
"""
Web Scrapers for Travel Scout
"""

import asyncio
import json
import logging
from typing import Dict, Any

from .mcp_browser_client import MCPBrowserClient
from .utils import TravelSearchUtils
from .config_loader import config

logger = logging.getLogger(__name__)


class BookingComScraper:
    """booking.com 스크레이퍼"""
    def __init__(self, mcp_client: MCPBrowserClient):
        self.mcp_client = mcp_client

    async def search(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """Booking.com에서 실제 호텔 데이터 스크레이핑 - 설정 기반"""
        # 설정에서 기본값 로드
        booking_config = config.get_booking_com_config()
        base_url = booking_config.get('base_url', 'https://www.booking.com/searchresults.html')
        selectors = booking_config.get('selectors', {})
        wait_time = booking_config.get('wait_time', 3)
        max_results = booking_config.get('max_results', 10)

        # 검색 파라미터 검증
        destination = search_params.get("destination")
        check_in = search_params.get("check_in")
        check_out = search_params.get("check_out")
        guests = search_params.get("guests", 2)

        if not all([destination, check_in, check_out]):
            raise ValueError("필수 검색 파라미터가 누락되었습니다")

        # URL 생성
        url_params = {
            'ss': destination,
            'checkin': check_in,
            'checkout': check_out,
            'group_adults': guests
        }
        url = TravelSearchUtils.format_search_url(base_url, url_params)

        logger.info(f"Navigating to Booking.com: {url}")
        nav_result = await self.mcp_client.navigate_and_capture(url)
        if not nav_result.get("success"):
            raise Exception(f"Booking.com 탐색 실패: {nav_result.get('error')}")

        await asyncio.sleep(wait_time)

        # 설정 기반 셀렉터 사용
        extract_script = f"""
        () => {{
            const hotels = [];
            document.querySelectorAll('{selectors.get('property_card', '[data-testid="property-card"]')}').forEach(element => {{
                const nameEl = element.querySelector('{selectors.get('title', '[data-testid="title"]')}');
                const priceEl = element.querySelector('{selectors.get('price', '[data-testid="price-and-discounted-price"]')}');
                const ratingEl = element.querySelector('{selectors.get('rating', '[data-testid="review-score"] .ac78a73c96')}');
                if (nameEl) {{
                    hotels.push({{
                        name: nameEl.textContent.trim(),
                        price: priceEl?.textContent?.trim() || 'N/A',
                        rating: ratingEl?.textContent?.trim() || 'N/A',
                    }});
                }}
            }});
            return hotels.slice(0, {max_results});
        }}
        """
        eval_result = await self.mcp_client.session.call_tool("puppeteer_evaluate", {"script": extract_script})

        hotels_data = []
        if eval_result and not eval_result.isError and eval_result.content:
            raw_data = json.loads(eval_result.content[0].text)
            for item in raw_data:
                try:
                    hotels_data.append({
                        'name': item.get('name'),
                        'price': item.get('price', 'N/A'),
                        'price_numeric': TravelSearchUtils.extract_price_from_text(item.get('price', '0')),
                        'rating': item.get('rating', 'N/A'),
                        'rating_numeric': TravelSearchUtils.extract_rating_from_text(item.get('rating', '0')),
                        'platform': 'booking.com'
                    })
                except Exception as e:
                    logger.warning(f"호텔 데이터 처리 오류: {e}")
                    continue

        logger.info(f"Scraped {len(hotels_data)} hotels from Booking.com")
        return {"type": "hotel", "data": hotels_data}


class GoogleFlightsScraper:
    """Google Flights 스크레이퍼"""
    def __init__(self, mcp_client: MCPBrowserClient):
        self.mcp_client = mcp_client

    async def search(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """Google Flights에서 실제 항공편 데이터 스크레이핑 - 설정 기반"""
        # 설정에서 기본값 로드
        flights_config = config.get_google_flights_config()
        base_url = flights_config.get('base_url', 'https://www.google.com/travel/flights')
        selectors = flights_config.get('selectors', {})
        wait_time = flights_config.get('wait_time', 5)
        max_results = flights_config.get('max_results', 10)

        # 검색 파라미터 검증
        origin = search_params.get("origin")
        destination = search_params.get("destination")
        departure_date = search_params.get("departure_date")
        return_date = search_params.get("return_date")

        if not all([origin, destination, departure_date, return_date]):
            raise ValueError("필수 검색 파라미터가 누락되었습니다")

        # 도시 코드 추출
        origin_code = origin.split('(')[-1].replace(')', '') if '(' in origin else origin
        dest_code = destination.split('(')[-1].replace(')', '') if '(' in destination else destination

        # URL 생성
        url = f"{base_url}?q=Flights%20to%20{dest_code}%20from%20{origin_code}%20on%20{departure_date}%20through%20{return_date}"

        logger.info(f"Navigating to Google Flights: {url}")
        nav_result = await self.mcp_client.navigate_and_capture(url)
        if not nav_result.get("success"):
            raise Exception(f"Google Flights 탐색 실패: {nav_result.get('error')}")

        await asyncio.sleep(wait_time)

        # 설정 기반 셀렉터 사용
        extract_script = f"""
        () => {{
            const flights = [];
            document.querySelectorAll('{selectors.get('result_item', 'div.gws-flights-results__result-item-inner')}').forEach(element => {{
                const airlineEl = element.querySelector('{selectors.get('airline_name', '.gws-flights-results__airline-name')}');
                const priceEl = element.querySelector('{selectors.get('price', '.gws-flights-results__price')}');
                const durationEl = element.querySelector('{selectors.get('duration', '.gws-flights-results__duration')}');
                if (airlineEl && priceEl) {{
                    flights.push({{
                        airline: airlineEl.textContent.trim(),
                        price: priceEl?.textContent?.trim() || 'N/A',
                        duration: durationEl?.textContent?.trim() || 'N/A',
                    }});
                }}
            }});
            return flights.slice(0, {max_results});
        }}
        """
        eval_result = await self.mcp_client.session.call_tool("puppeteer_evaluate", {"script": extract_script})

        flights_data = []
        if eval_result and not eval_result.isError and eval_result.content:
            raw_data = json.loads(eval_result.content[0].text)
            for item in raw_data:
                try:
                    flights_data.append({
                        'airline': item.get('airline'),
                        'price': item.get('price', 'N/A'),
                        'price_numeric': TravelSearchUtils.extract_price_from_text(item.get('price', '0')),
                        'duration': item.get('duration', 'N/A'),
                        'platform': 'google_flights'
                    })
                except Exception as e:
                    logger.warning(f"항공편 데이터 처리 오류: {e}")
                    continue

        logger.info(f"Scraped {len(flights_data)} flights from Google Flights")
        return {"type": "flight", "data": flights_data}
