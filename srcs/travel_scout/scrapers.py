#!/usr/bin/env python3
"""
Web Scrapers for Travel Scout
"""

import asyncio
import json
import logging
from typing import Dict, List

from .mcp_browser_client import MCPBrowserClient
from .travel_utils import TravelSearchUtils

logger = logging.getLogger(__name__)

class BookingComScraper:
    """Scraper for Booking.com"""

    def __init__(self, client: MCPBrowserClient):
        self.client = client
        if not self.client.session:
            raise ConnectionError("MCPBrowserClient session not initialized.")
        self.session = self.client.session

    async def search(self, destination: str, check_in: str, check_out: str) -> List[Dict]:
        """Search for hotels on Booking.com"""
        logger.info(f"🏨 Starting Booking.com search for {destination}")
        
        try:
            await self.session.call_tool("puppeteer_navigate", arguments={"url": "https://www.booking.com"})
            await self.client._debug_screenshot("booking_home")

            await self.session.call_tool("puppeteer_fill", arguments={"selector": "input[name='ss']", "value": destination})
            await self.session.call_tool("puppeteer_fill", arguments={"selector": "input[name='checkin']", "value": check_in})
            await self.session.call_tool("puppeteer_fill", arguments={"selector": "input[name='checkout']", "value": check_out})
            await self.session.call_tool("puppeteer_click", arguments={"selector": "button[type='submit']"})
            await self.client._debug_screenshot("booking_search_results")

            await asyncio.sleep(3)

            extract_script = """
            () => {
                const hotels = [];
                document.querySelectorAll('[data-testid="property-card"]').forEach(element => {
                    const nameEl = element.querySelector('[data-testid="title"]');
                    const priceEl = element.querySelector('[data-testid="price-and-discounted-price"]');
                    const ratingEl = element.querySelector('[data-testid="review-score"] .ac78a73c96');
                    const locationEl = element.querySelector('[data-testid="address"]');
                    if (nameEl) {
                        hotels.push({
                            name: nameEl.textContent.trim(),
                            price: priceEl?.textContent?.trim() || 'N/A',
                            rating: ratingEl?.textContent?.trim() || 'N/A',
                            location: locationEl?.textContent?.trim() || ''
                        });
                    }
                });
                return hotels.slice(0, 10);
            }
            """
            extract_result = await self.session.call_tool("puppeteer_evaluate", arguments={"script": extract_script})
            
            hotels = []
            if hasattr(extract_result, 'content') and extract_result.content and extract_result.content[0].text:
                extracted_data = json.loads(extract_result.content[0].text)
                for item in extracted_data:
                    if item.get('name'):
                        hotel = {
                            'name': item.get('name'),
                            'price': item.get('price', 'N/A'),
                            'price_numeric': TravelSearchUtils.extract_price_from_text(item.get('price', '0')),
                            'rating': item.get('rating', 'N/A'),
                            'rating_numeric': TravelSearchUtils.extract_rating_from_text(item.get('rating', '0')),
                            'location': item.get('location', destination),
                            'platform': 'booking.com',
                            'source': 'Real Booking.com Scraping',
                            'quality_score': TravelSearchUtils.calculate_hotel_quality_score(item),
                            'meets_quality_criteria': TravelSearchUtils.extract_rating_from_text(item.get('rating', '0')) >= 4.0
                        }
                        hotels.append(hotel)
            return hotels
        except Exception as e:
            logger.error(f"Booking.com search failed: {e}")
            return []

class GoogleFlightsScraper:
    """Scraper for Google Flights"""

    def __init__(self, client: MCPBrowserClient):
        self.client = client
        if not self.client.session:
            raise ConnectionError("MCPBrowserClient session not initialized.")
        self.session = self.client.session

    async def search(self, origin: str, destination: str, departure_date: str, return_date: str) -> List[Dict]:
        """Search for flights on Google Flights"""
        logger.info(f"✈️ Starting Google Flights search for {origin} -> {destination}")

        try:
            await self.session.call_tool("puppeteer_navigate", arguments={"url": "https://www.google.com/travel/flights"})
            await self.client._debug_screenshot("flights_home")

            await self.session.call_tool("puppeteer_fill", arguments={"selector": "input[placeholder*='Where from']", "value": origin})
            await self.session.call_tool("puppeteer_fill", arguments={"selector": "input[placeholder*='Where to']", "value": destination})
            await self.session.call_tool("puppeteer_fill", arguments={"selector": "input[placeholder*='Departure']", "value": departure_date})
            await self.session.call_tool("puppeteer_fill", arguments={"selector": "input[placeholder*='Return']", "value": return_date})
            await self.session.call_tool("puppeteer_click", arguments={"selector": "button[aria-label*='Search']"})
            await self.client._debug_screenshot("flights_search_results")
            
            await asyncio.sleep(5)

            extract_script = """
            () => {
                const flights = [];
                document.querySelectorAll('.gws-flights-results__result-item').forEach(element => {
                    const airlineEl = element.querySelector('.gws-flights-results__airline-name');
                    const priceEl = element.querySelector('.gws-flights-results__price');
                    const durationEl = element.querySelector('.gws-flights-results__duration');
                    const timeEl = element.querySelector('.gws-flights-results__leg-departure');
                    if (airlineEl) {
                        flights.push({
                            airline: airlineEl.textContent.trim(),
                            price: priceEl?.textContent?.trim() || 'N/A',
                            duration: durationEl?.textContent?.trim() || 'N/A',
                            departure_time: timeEl?.textContent?.trim() || 'N/A'
                        });
                    }
                });
                return flights.slice(0, 10);
            }
            """
            extract_result = await self.session.call_tool("puppeteer_evaluate", arguments={"script": extract_script})

            flights = []
            if hasattr(extract_result, 'content') and extract_result.content and extract_result.content[0].text:
                extracted_data = json.loads(extract_result.content[0].text)
                for item in extracted_data:
                    if item.get('airline'):
                        flight = {
                            'airline': item.get('airline'),
                            'price': item.get('price', 'N/A'),
                            'price_numeric': TravelSearchUtils.extract_price_from_text(item.get('price', '0')),
                            'duration': item.get('duration', 'N/A'),
                            'departure_time': item.get('departure_time', 'N/A'),
                            'platform': 'google_flights',
                            'source': 'Real Google Flights Scraping',
                            'route': f'{origin} → {destination}',
                            'quality_score': TravelSearchUtils.calculate_flight_quality_score(item),
                            'meets_quality_criteria': item.get('airline', '') in ['Korean Air', 'Asiana Airlines', 'Delta', 'Emirates']
                        }
                        flights.append(flight)
            return flights
        except Exception as e:
            logger.error(f"Google Flights search failed: {e}")
            return [] 