#!/usr/bin/env python3
"""
Web Scrapers for Travel Scout
"""

import asyncio
import json
import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta

from .mcp_browser_client import MCPBrowserClient
from .travel_utils import TravelSearchUtils

logger = logging.getLogger(__name__)

class BookingComScraper:
    """booking.com 스크레이퍼"""
    def __init__(self, mcp_client: MCPBrowserClient):
        self.mcp_client = mcp_client

    async def search(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """Booking.com에서 실제 호텔 데이터 스크레이핑"""
        destination = search_params.get("destination", "Seoul")
        check_in = search_params.get("check_in", (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'))
        check_out = search_params.get("check_out", (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'))
        guests = search_params.get("guests", 2)
        
        url = f"https://www.booking.com/searchresults.html?ss={destination}&checkin={check_in}&checkout={check_out}&group_adults={guests}"
        
        logger.info(f"Navigating to Booking.com: {url}")
        nav_result = await self.mcp_client.navigate_and_capture(url)
        if not nav_result.get("success"):
            logger.error(f"Failed to navigate to booking.com: {nav_result.get('error')}")
            return {"type": "hotel", "data": []}

        await asyncio.sleep(3)

        extract_script = """
        () => {
            const hotels = [];
            document.querySelectorAll('[data-testid="property-card"]').forEach(element => {
                const nameEl = element.querySelector('[data-testid="title"]');
                const priceEl = element.querySelector('[data-testid="price-and-discounted-price"]');
                const ratingEl = element.querySelector('[data-testid="review-score"] .ac78a73c96');
                if (nameEl) {
                    hotels.push({
                        name: nameEl.textContent.trim(),
                        price: priceEl?.textContent?.trim() || 'N/A',
                        rating: ratingEl?.textContent?.trim() || 'N/A',
                    });
                }
            });
            return hotels.slice(0, 10);
        }
        """
        eval_result = await self.mcp_client.session.call_tool("puppeteer_evaluate", {"script": extract_script})
        
        hotels_data = []
        if eval_result and not eval_result.isError and eval_result.content:
            raw_data = json.loads(eval_result.content[0].text)
            for item in raw_data:
                hotels_data.append({
                    'name': item.get('name'),
                    'price': item.get('price', 'N/A'),
                    'price_numeric': TravelSearchUtils.extract_price_from_text(item.get('price', '0')),
                    'rating': item.get('rating', 'N/A'),
                    'rating_numeric': TravelSearchUtils.extract_rating_from_text(item.get('rating', '0')),
                    'platform': 'booking.com'
                })
        
        logger.info(f"Scraped {len(hotels_data)} hotels from Booking.com")
        return {"type": "hotel", "data": hotels_data}

class GoogleFlightsScraper:
    """Google Flights 스크레이퍼"""
    def __init__(self, mcp_client: MCPBrowserClient):
        self.mcp_client = mcp_client

    async def search(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """Google Flights에서 실제 항공편 데이터 스크레이핑"""
        origin = search_params.get("origin", "ICN")
        destination = search_params.get("destination", "NRT")
        departure_date = search_params.get("departure_date", (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'))
        return_date = search_params.get("return_date", (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'))

        origin_code = origin.split('(')[-1].replace(')', '') if '(' in origin else origin
        dest_code = destination.split('(')[-1].replace(')', '') if '(' in destination else destination

        url = f"https://www.google.com/travel/flights?q=Flights%20to%20{dest_code}%20from%20{origin_code}%20on%20{departure_date}%20through%20{return_date}"
        
        logger.info(f"Navigating to Google Flights: {url}")
        nav_result = await self.mcp_client.navigate_and_capture(url)
        if not nav_result.get("success"):
            logger.error(f"Failed to navigate to Google Flights: {nav_result.get('error')}")
            return {"type": "flight", "data": []}

        await asyncio.sleep(5)

        extract_script = """
        () => {
            const flights = [];
            document.querySelectorAll('div.gws-flights-results__result-item-inner').forEach(element => {
                const airlineEl = element.querySelector('.gws-flights-results__airline-name');
                const priceEl = element.querySelector('.gws-flights-results__price');
                const durationEl = element.querySelector('.gws-flights-results__duration');
                if (airlineEl && priceEl) {
                    flights.push({
                        airline: airlineEl.textContent.trim(),
                        price: priceEl?.textContent?.trim() || 'N/A',
                        duration: durationEl?.textContent?.trim() || 'N/A',
                    });
                }
            });
            return flights.slice(0, 10);
        }
        """
        eval_result = await self.mcp_client.session.call_tool("puppeteer_evaluate", {"script": extract_script})
        
        flights_data = []
        if eval_result and not eval_result.isError and eval_result.content:
            raw_data = json.loads(eval_result.content[0].text)
            for item in raw_data:
                flights_data.append({
                    'airline': item.get('airline'),
                    'price': item.get('price', 'N/A'),
                    'price_numeric': TravelSearchUtils.extract_price_from_text(item.get('price', '0')),
                    'duration': item.get('duration', 'N/A'),
                    'platform': 'google_flights'
                })

        logger.info(f"Scraped {len(flights_data)} flights from Google Flights")
        return {"type": "flight", "data": flights_data} 