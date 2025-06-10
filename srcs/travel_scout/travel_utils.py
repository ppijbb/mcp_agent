#!/usr/bin/env python3
"""
Travel Search Utilities

Utility functions for automated travel search using incognito browsing mode.
"""

import asyncio
import re
from typing import Dict, List, Optional


class TravelSearchUtils:
    """Utility functions for travel search operations"""
    
    @staticmethod
    def validate_search_params(destination: str, check_in: str, check_out: str, 
                              departure_date: str, return_date: str) -> bool:
        """Validate travel search parameters"""
        required_params = [destination, check_in, check_out, departure_date, return_date]
        return all(param and param.strip() for param in required_params)
    
    @staticmethod
    def extract_price_from_text(text: str) -> float:
        """Extract numeric price value from text"""
        if not text:
            return float('inf')
        
        # Remove currency symbols and commas
        cleaned = re.sub(r'[^\d.]', '', text)
        try:
            return float(cleaned)
        except:
            return float('inf')
    
    @staticmethod
    def extract_rating_from_text(text: str) -> float:
        """Extract numeric rating value from text"""
        if not text:
            return 0.0
        
        match = re.search(r'([0-9.]+)', text)
        if match:
            try:
                return float(match.group(1))
            except:
                return 0.0
        return 0.0
    
    @staticmethod
    def format_search_url(base_url: str, params: Dict[str, str]) -> str:
        """Format search URL with parameters"""
        import urllib.parse
        
        query_params = []
        for key, value in params.items():
            if value:
                encoded_value = urllib.parse.quote(str(value))
                query_params.append(f"{key}={encoded_value}")
        
        if query_params:
            return f"{base_url}?{'&'.join(query_params)}"
        return base_url
    
    @staticmethod
    def parse_hotel_data(content: str, platform: str) -> List[Dict]:
        """Parse hotel data from page content"""
        hotels = []
        lines = content.split('\n')
        
        for line in lines:
            if 'hotel' in line.lower():
                if any(keyword in line.lower() for keyword in ['rating', 'price', 'score', 'star']):
                    hotel_data = {
                        'name': TravelSearchUtils._extract_hotel_name(line),
                        'price': TravelSearchUtils._extract_price_pattern(line),
                        'rating': TravelSearchUtils._extract_rating_pattern(line),
                        'platform': platform,
                        'raw_text': line.strip()
                    }
                    
                    if hotel_data['name'] or hotel_data['price']:
                        hotels.append(hotel_data)
        
        return hotels[:15]  # Return top 15 results
    
    @staticmethod
    def parse_flight_data(content: str, platform: str) -> List[Dict]:
        """Parse flight data from page content"""
        flights = []
        lines = content.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['flight', 'airline', 'departure', 'arrival']):
                if any(indicator in line for indicator in ['$', '€', '£', '₩', 'price', 'fare']):
                    flight_data = {
                        'airline': TravelSearchUtils._extract_airline_name(line),
                        'price': TravelSearchUtils._extract_price_pattern(line),
                        'duration': TravelSearchUtils._extract_duration(line),
                        'departure_time': TravelSearchUtils._extract_time_pattern(line),
                        'platform': platform,
                        'raw_text': line.strip()
                    }
                    
                    if flight_data['airline'] or flight_data['price']:
                        flights.append(flight_data)
        
        return flights[:15]
    
    @staticmethod
    def _extract_hotel_name(text: str) -> str:
        """Extract hotel name from text"""
        hotel_patterns = [
            r'Hotel\s+([A-Za-z\s\-&]+?)(?:\s|$|,|\||\.)',
            r'([A-Za-z\s\-&]+?)\s+Hotel(?:\s|$|,|\||\.)',
            r'Resort\s+([A-Za-z\s\-&]+?)(?:\s|$|,|\||\.)',
            r'([A-Za-z\s\-&]+?)\s+Resort(?:\s|$|,|\||\.)',
            r'Inn\s+([A-Za-z\s\-&]+?)(?:\s|$|,|\||\.)',
            r'([A-Za-z\s\-&]+?)\s+Inn(?:\s|$|,|\||\.)'
        ]
        
        for pattern in hotel_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 3 and name.replace(' ', '').isalpha():
                    return name
        
        return ""
    
    @staticmethod
    def _extract_price_pattern(text: str) -> str:
        """Extract price from text with various patterns"""
        price_patterns = [
            r'[\$€£¥₩]\s*([0-9,]+\.?[0-9]*)',
            r'([0-9,]+\.?[0-9]*)\s*[\$€£¥₩]',
            r'([0-9,]+\.?[0-9]*)\s*per\s*night',
            r'([0-9,]+\.?[0-9]*)\s*원',
            r'Price:\s*[\$€£¥₩]?\s*([0-9,]+\.?[0-9]*)',
            r'from\s*[\$€£¥₩]\s*([0-9,]+\.?[0-9]*)'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return ""
    
    @staticmethod
    def _extract_rating_pattern(text: str) -> str:
        """Extract rating from text with various patterns"""
        rating_patterns = [
            r'([0-9.]+)\s*\/\s*([0-9]+)',
            r'([0-9.]+)\s*stars?',
            r'Rating:\s*([0-9.]+)',
            r'([0-9.]+)\s*out\s*of\s*([0-9]+)',
            r'Score:\s*([0-9.]+)',
            r'([0-9.]+)\s*\/\s*10'
        ]
        
        for pattern in rating_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return ""
    
    @staticmethod
    def _extract_airline_name(text: str) -> str:
        """Extract airline name from text"""
        airlines = [
            'Korean Air', 'Asiana', 'Delta', 'United', 'American', 'Lufthansa', 
            'Emirates', 'Singapore Airlines', 'Cathay Pacific', 'JAL', 'ANA', 
            'Air France', 'KLM', 'British Airways', 'Qatar Airways', 'Turkish Airlines',
            'Southwest', 'JetBlue', 'Alaska Airlines', 'Spirit', 'Frontier'
        ]
        
        for airline in airlines:
            if airline.lower() in text.lower():
                return airline
        
        # Try to extract airline-like patterns
        airline_patterns = [
            r'([A-Z][a-z]+\s+Air(?:lines?)?)',
            r'([A-Z][a-z]+\s+Airways?)',
            r'(Air\s+[A-Z][a-z]+)'
        ]
        
        for pattern in airline_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return ""
    
    @staticmethod
    def _extract_duration(text: str) -> str:
        """Extract flight duration from text"""
        duration_patterns = [
            r'([0-9]+h\s*[0-9]*m?)',
            r'([0-9]+\s*hours?\s*[0-9]*\s*minutes?)',
            r'Duration:\s*([0-9]+h\s*[0-9]*m?)',
            r'([0-9]+:[0-9]+)\s*(?:hours?|hrs?)'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
    
    @staticmethod
    def _extract_time_pattern(text: str) -> str:
        """Extract time from text"""
        time_patterns = [
            r'([0-9]{1,2}:[0-9]{2}\s*[AP]M)',
            r'([0-9]{1,2}:[0-9]{2})',
            r'Departure:\s*([0-9]{1,2}:[0-9]{2})',
            r'Depart:\s*([0-9]{1,2}:[0-9]{2})'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "" 