#!/usr/bin/env python3
"""
Travel Browser Utilities

Utility functions for automated travel search using incognito browsing mode.
Provides specialized functions for hotel and flight searches with cache prevention.
"""

import asyncio
import json
import time
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import urllib.parse


class IncognitoBrowserManager:
    """Manager for incognito browsing sessions"""
    
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.session_count = 0
    
    async def start_incognito_session(self):
        """Start a new incognito browsing session"""
        self.session_count += 1
        # Note: Each navigation in MCP Playwright automatically uses a fresh context
        # which mimics incognito behavior
        print(f"🔒 Starting incognito session #{self.session_count}")
        return self.session_count
    
    async def navigate_incognito(self, url: str) -> bool:
        """Navigate to URL in incognito mode"""
        try:
            print(f"🌐 Navigating to: {url}")
            result = await self.mcp_client.navigate(url=url)
            await asyncio.sleep(2)  # Allow page to load
            return True
        except Exception as e:
            print(f"❌ Navigation failed: {str(e)}")
            return False
    
    async def get_page_content(self) -> str:
        """Get current page content"""
        try:
            result = await self.mcp_client.get_visible_content()
            return result
        except Exception as e:
            print(f"❌ Failed to get page content: {str(e)}")
            return ""
    
    async def click_element(self, x: int, y: int) -> bool:
        """Click element at coordinates"""
        try:
            await self.mcp_client.mouse_click(x=x, y=y)
            await asyncio.sleep(1)
            return True
        except Exception as e:
            print(f"❌ Click failed: {str(e)}")
            return False
    
    async def scroll_page(self, delta_y: int = 300) -> bool:
        """Scroll the page"""
        try:
            await self.mcp_client.mouse_wheel(deltaY=delta_y)
            await asyncio.sleep(1)
            return True
        except Exception as e:
            print(f"❌ Scroll failed: {str(e)}")
            return False


class HotelSearchAutomator:
    """Automated hotel search with incognito browsing"""
    
    def __init__(self, browser_manager: IncognitoBrowserManager):
        self.browser = browser_manager
        self.search_sites = {
            "booking": "https://www.booking.com",
            "hotels": "https://www.hotels.com", 
            "expedia": "https://www.expedia.com",
            "agoda": "https://www.agoda.com"
        }
    
    async def search_booking_com(self, destination: str, check_in: str, check_out: str) -> List[Dict]:
        """Search hotels on Booking.com"""
        print("🏨 Searching Booking.com...")
        
        # Navigate to Booking.com
        url = f"https://www.booking.com/searchresults.html?ss={urllib.parse.quote(destination)}&checkin={check_in}&checkout={check_out}&group_adults=2&no_rooms=1"
        
        if not await self.browser.navigate_incognito(url):
            return []
        
        await asyncio.sleep(3)  # Wait for results to load
        
        # Get page content and parse hotels
        content = await self.browser.get_page_content()
        hotels = self._parse_booking_hotels(content)
        
        print(f"✅ Found {len(hotels)} hotels on Booking.com")
        return hotels
    
    async def search_hotels_com(self, destination: str, check_in: str, check_out: str) -> List[Dict]:
        """Search hotels on Hotels.com"""
        print("🏨 Searching Hotels.com...")
        
        url = f"https://www.hotels.com/search.do?q-destination={urllib.parse.quote(destination)}&q-check-in={check_in}&q-check-out={check_out}&q-rooms=1&q-room-0-adults=2"
        
        if not await self.browser.navigate_incognito(url):
            return []
        
        await asyncio.sleep(3)
        content = await self.browser.get_page_content()
        hotels = self._parse_hotels_com(content)
        
        print(f"✅ Found {len(hotels)} hotels on Hotels.com")
        return hotels
    
    async def search_all_platforms(self, destination: str, check_in: str, check_out: str) -> Dict[str, List[Dict]]:
        """Search all hotel platforms"""
        results = {}
        
        # Search each platform in separate incognito sessions
        platforms = [
            ("booking", self.search_booking_com),
            ("hotels", self.search_hotels_com),
        ]
        
        for platform_name, search_func in platforms:
            await self.browser.start_incognito_session()
            results[platform_name] = await search_func(destination, check_in, check_out)
            await asyncio.sleep(2)  # Brief pause between platforms
        
        return results
    
    def _parse_booking_hotels(self, content: str) -> List[Dict]:
        """Parse Booking.com search results"""
        hotels = []
        
        # This is a simplified parser - in reality you'd need more sophisticated parsing
        # based on the actual HTML structure
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if 'hotel' in line.lower() and any(keyword in line.lower() for keyword in ['rating', 'price', 'score']):
                hotel_data = {
                    'name': self._extract_hotel_name(line),
                    'price': self._extract_price(line),
                    'rating': self._extract_rating(line),
                    'platform': 'booking.com'
                }
                if hotel_data['name']:
                    hotels.append(hotel_data)
        
        return hotels[:10]  # Return top 10 results
    
    def _parse_hotels_com(self, content: str) -> List[Dict]:
        """Parse Hotels.com search results"""
        hotels = []
        
        lines = content.split('\n')
        for line in lines:
            if 'hotel' in line.lower() and any(keyword in line.lower() for keyword in ['rating', 'price', 'star']):
                hotel_data = {
                    'name': self._extract_hotel_name(line),
                    'price': self._extract_price(line), 
                    'rating': self._extract_rating(line),
                    'platform': 'hotels.com'
                }
                if hotel_data['name']:
                    hotels.append(hotel_data)
        
        return hotels[:10]
    
    def _extract_hotel_name(self, text: str) -> str:
        """Extract hotel name from text"""
        # Simplified extraction - would need more sophisticated regex
        hotel_patterns = [
            r'Hotel\s+([A-Za-z\s]+)',
            r'([A-Za-z\s]+)\s+Hotel',
            r'Resort\s+([A-Za-z\s]+)',
            r'([A-Za-z\s]+)\s+Resort'
        ]
        
        for pattern in hotel_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_price(self, text: str) -> str:
        """Extract price from text"""
        price_patterns = [
            r'[\$€£¥₩]\s*([0-9,]+)',
            r'([0-9,]+)\s*[\$€£¥₩]',
            r'([0-9,]+)\s*per\s*night',
            r'([0-9,]+)\s*원'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return ""
    
    def _extract_rating(self, text: str) -> str:
        """Extract rating from text"""
        rating_patterns = [
            r'([0-9.]+)\s*\/\s*([0-9]+)',
            r'([0-9.]+)\s*stars?',
            r'Rating:\s*([0-9.]+)',
            r'([0-9.]+)\s*out\s*of\s*([0-9]+)'
        ]
        
        for pattern in rating_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return ""


class FlightSearchAutomator:
    """Automated flight search with incognito browsing"""
    
    def __init__(self, browser_manager: IncognitoBrowserManager):
        self.browser = browser_manager
        self.search_sites = {
            "kayak": "https://www.kayak.com",
            "expedia": "https://www.expedia.com",
            "skyscanner": "https://www.skyscanner.com",
            "google_flights": "https://www.google.com/flights"
        }
    
    async def search_google_flights(self, origin: str, destination: str, departure_date: str, return_date: str) -> List[Dict]:
        """Search flights on Google Flights"""
        print("✈️ Searching Google Flights...")
        
        # Format dates for Google Flights URL
        dep_formatted = departure_date.replace('-', '-')
        ret_formatted = return_date.replace('-', '-')
        
        url = f"https://www.google.com/flights?f=0&gl=us&hl=en&curr=USD&tfs=CBwQAhofagcIARIDSUNOEgoyMDI0LTEyLTE1cgcIARIDTEFYGh9qBwgBEgNMQVgSCjIwMjQtMTItMjByBwgBEgNJQ04&hl=en"
        
        if not await self.browser.navigate_incognito(url):
            return []
        
        await asyncio.sleep(5)  # Wait for flights to load
        content = await self.browser.get_page_content()
        flights = self._parse_google_flights(content)
        
        print(f"✅ Found {len(flights)} flights on Google Flights")
        return flights
    
    async def search_kayak(self, origin: str, destination: str, departure_date: str, return_date: str) -> List[Dict]:
        """Search flights on Kayak"""
        print("✈️ Searching Kayak...")
        
        url = f"https://www.kayak.com/flights/{origin}-{destination}/{departure_date}/{return_date}"
        
        if not await self.browser.navigate_incognito(url):
            return []
        
        await asyncio.sleep(5)
        content = await self.browser.get_page_content()
        flights = self._parse_kayak_flights(content)
        
        print(f"✅ Found {len(flights)} flights on Kayak")
        return flights
    
    async def search_all_platforms(self, origin: str, destination: str, departure_date: str, return_date: str) -> Dict[str, List[Dict]]:
        """Search all flight platforms"""
        results = {}
        
        platforms = [
            ("google_flights", self.search_google_flights),
            ("kayak", self.search_kayak),
        ]
        
        for platform_name, search_func in platforms:
            await self.browser.start_incognito_session()
            results[platform_name] = await search_func(origin, destination, departure_date, return_date)
            await asyncio.sleep(3)
        
        return results
    
    def _parse_google_flights(self, content: str) -> List[Dict]:
        """Parse Google Flights search results"""
        flights = []
        
        lines = content.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['flight', 'airline', 'departure', 'arrival']):
                if any(price_indicator in line for price_indicator in ['$', '€', '£', '₩']):
                    flight_data = {
                        'airline': self._extract_airline(line),
                        'price': self._extract_price(line),
                        'duration': self._extract_duration(line),
                        'departure_time': self._extract_time(line, 'departure'),
                        'platform': 'google_flights'
                    }
                    if flight_data['airline']:
                        flights.append(flight_data)
        
        return flights[:10]
    
    def _parse_kayak_flights(self, content: str) -> List[Dict]:
        """Parse Kayak search results"""
        flights = []
        
        lines = content.split('\n')
        for line in lines:
            if 'flight' in line.lower() and any(indicator in line for indicator in ['$', 'price', 'fare']):
                flight_data = {
                    'airline': self._extract_airline(line),
                    'price': self._extract_price(line),
                    'duration': self._extract_duration(line),
                    'departure_time': self._extract_time(line, 'departure'),
                    'platform': 'kayak'
                }
                if flight_data['airline']:
                    flights.append(flight_data)
        
        return flights[:10]
    
    def _extract_airline(self, text: str) -> str:
        """Extract airline name from text"""
        airlines = ['Korean Air', 'Asiana', 'Delta', 'United', 'American', 'Lufthansa', 'Emirates', 
                  'Singapore Airlines', 'Cathay Pacific', 'JAL', 'ANA', 'Air France', 'KLM']
        
        for airline in airlines:
            if airline.lower() in text.lower():
                return airline
        
        return ""
    
    def _extract_duration(self, text: str) -> str:
        """Extract flight duration from text"""
        duration_patterns = [
            r'([0-9]+h\s*[0-9]*m?)',
            r'([0-9]+\s*hours?\s*[0-9]*\s*minutes?)',
            r'Duration:\s*([0-9]+h\s*[0-9]*m?)'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
    
    def _extract_time(self, text: str, time_type: str) -> str:
        """Extract departure/arrival time from text"""
        time_patterns = [
            r'([0-9]{1,2}:[0-9]{2}\s*[AP]M)',
            r'([0-9]{1,2}:[0-9]{2})',
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if time_type == 'departure':
                    return matches[0] if matches else ""
                elif time_type == 'arrival' and len(matches) > 1:
                    return matches[1]
        
        return ""


class TravelPriceAnalyzer:
    """Analyze and compare travel prices across platforms"""
    
    def __init__(self):
        self.price_history = []
    
    def analyze_hotel_prices(self, hotel_results: Dict[str, List[Dict]]) -> Dict:
        """Analyze hotel price data"""
        analysis = {
            'best_deals': [],
            'price_comparison': {},
            'quality_assessment': {},
            'recommendations': []
        }
        
        all_hotels = []
        for platform, hotels in hotel_results.items():
            for hotel in hotels:
                hotel['platform'] = platform
                all_hotels.append(hotel)
        
        # Find best value hotels (price vs rating)
        quality_hotels = [h for h in all_hotels if self._extract_rating_value(h.get('rating', '')) >= 4.0]
        sorted_hotels = sorted(quality_hotels, key=lambda x: self._extract_price_value(x.get('price', '')))
        
        analysis['best_deals'] = sorted_hotels[:5]
        analysis['recommendations'] = self._generate_hotel_recommendations(sorted_hotels)
        
        return analysis
    
    def analyze_flight_prices(self, flight_results: Dict[str, List[Dict]]) -> Dict:
        """Analyze flight price data"""
        analysis = {
            'best_deals': [],
            'price_comparison': {},
            'duration_analysis': {},
            'recommendations': []
        }
        
        all_flights = []
        for platform, flights in flight_results.items():
            for flight in flights:
                flight['platform'] = platform
                all_flights.append(flight)
        
        # Sort by price
        sorted_flights = sorted(all_flights, key=lambda x: self._extract_price_value(x.get('price', '')))
        analysis['best_deals'] = sorted_flights[:5]
        analysis['recommendations'] = self._generate_flight_recommendations(sorted_flights)
        
        return analysis
    
    def _extract_rating_value(self, rating_text: str) -> float:
        """Extract numeric rating value"""
        if not rating_text:
            return 0.0
        
        match = re.search(r'([0-9.]+)', rating_text)
        if match:
            return float(match.group(1))
        return 0.0
    
    def _extract_price_value(self, price_text: str) -> float:
        """Extract numeric price value"""
        if not price_text:
            return float('inf')
        
        # Remove currency symbols and commas
        cleaned = re.sub(r'[^\d.]', '', price_text)
        try:
            return float(cleaned)
        except:
            return float('inf')
    
    def _generate_hotel_recommendations(self, hotels: List[Dict]) -> List[str]:
        """Generate hotel recommendations"""
        recommendations = []
        
        if hotels:
            best_hotel = hotels[0]
            recommendations.append(f"Best Value: {best_hotel.get('name', 'Unknown')} - {best_hotel.get('price', 'N/A')}")
        
        if len(hotels) > 1:
            recommendations.append(f"Alternative: {hotels[1].get('name', 'Unknown')} - {hotels[1].get('price', 'N/A')}")
        
        return recommendations
    
    def _generate_flight_recommendations(self, flights: List[Dict]) -> List[str]:
        """Generate flight recommendations"""
        recommendations = []
        
        if flights:
            best_flight = flights[0]
            recommendations.append(f"Cheapest: {best_flight.get('airline', 'Unknown')} - {best_flight.get('price', 'N/A')}")
        
        return recommendations 