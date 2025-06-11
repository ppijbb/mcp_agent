#!/usr/bin/env python3
"""
Travel Search Utilities

Utility functions for automated travel search using incognito browsing mode.
"""

import asyncio
import re
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TravelSearchError(Exception):
    """Custom exception for travel search errors"""
    pass


class TravelSearchUtils:
    """Utility functions for travel search operations"""
    
    @staticmethod
    def validate_search_params(destination: str, check_in: str, check_out: str, 
                              departure_date: str, return_date: str) -> Tuple[bool, str]:
        """Validate travel search parameters with detailed error messages"""
        try:
            # Check required parameters
            required_params = {
                'destination': destination,
                'check_in': check_in, 
                'check_out': check_out,
                'departure_date': departure_date,
                'return_date': return_date
            }
            
            for param_name, param_value in required_params.items():
                if not param_value or not param_value.strip():
                    return False, f"Missing required parameter: {param_name}"
            
            # Validate date formats
            date_params = {
                'check_in': check_in,
                'check_out': check_out,
                'departure_date': departure_date,
                'return_date': return_date
            }
            
            parsed_dates = {}
            for date_name, date_str in date_params.items():
                try:
                    parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
                    parsed_dates[date_name] = parsed_date
                except ValueError:
                    return False, f"Invalid date format for {date_name}: {date_str}. Expected format: YYYY-MM-DD"
            
            # Validate date logic
            today = datetime.now().date()
            
            # Check if dates are in the future
            for date_name, parsed_date in parsed_dates.items():
                if parsed_date.date() < today:
                    return False, f"{date_name} cannot be in the past: {date_str}"
            
            # Check date relationships
            if parsed_dates['check_out'] <= parsed_dates['check_in']:
                return False, "Check-out date must be after check-in date"
            
            if parsed_dates['return_date'] <= parsed_dates['departure_date']:
                return False, "Return date must be after departure date"
            
            # Check reasonable date ranges (not too far in future)
            max_future_date = today + timedelta(days=365)
            for date_name, parsed_date in parsed_dates.items():
                if parsed_date.date() > max_future_date:
                    return False, f"{date_name} is too far in the future (max 1 year ahead)"
            
            # Validate destination format
            if len(destination.strip()) < 3:
                return False, "Destination must be at least 3 characters long"
            
            return True, "All parameters are valid"
            
        except Exception as e:
            logger.error(f"Parameter validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    async def retry_operation(operation, max_retries: int = 3, delay: float = 1.0, 
                            backoff: float = 2.0) -> Optional[any]:
        """Retry operation with exponential backoff"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation()
                else:
                    result = operation()
                return result
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed. Last error: {e}")
        
        raise TravelSearchError(f"Operation failed after {max_retries} attempts: {last_exception}")
    
    @staticmethod
    def extract_price_from_text(text: str) -> float:
        """Extract numeric price value from text with enhanced patterns"""
        if not text:
            return float('inf')
        
        try:
            # Enhanced price patterns
            price_patterns = [
                r'[\$€£¥₩]\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                r'([0-9,]+(?:\.[0-9]{1,2})?)\s*[\$€£¥₩]',
                r'([0-9,]+)\s*원',
                r'USD\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                r'EUR\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                r'([0-9,]+(?:\.[0-9]{1,2})?)\s*dollars?',
                r'([0-9,]+(?:\.[0-9]{1,2})?)\s*euros?'
            ]
            
            for pattern in price_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Extract the numeric part
                    price_str = match.group(1) if match.group(1) else match.group(0)
                    # Remove currency symbols and commas
                    cleaned = re.sub(r'[^\d.]', '', price_str)
                    if cleaned:
                        return float(cleaned)
            
            return float('inf')
            
        except (ValueError, AttributeError) as e:
            logger.warning(f"Price extraction error for text '{text}': {e}")
            return float('inf')
    
    @staticmethod
    def extract_rating_from_text(text: str) -> float:
        """Extract numeric rating value from text with validation"""
        if not text:
            return 0.0
        
        try:
            rating_patterns = [
                r'([0-9]\.[0-9])\s*(?:/|\s*out\s+of)\s*([0-9])',  # 4.5/5 or 4.5 out of 5
                r'([0-9]\.[0-9])',  # Simple decimal rating
                r'([0-9])\s*stars?',  # 4 stars
                r'Rating:\s*([0-9]\.[0-9])',  # Rating: 4.5
                r'Score:\s*([0-9]\.[0-9])'  # Score: 4.5
            ]
            
            for pattern in rating_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    rating_value = float(match.group(1))
                    # Validate rating range (0-5 scale)
                    if 0 <= rating_value <= 5:
                        return rating_value
                    # Convert if it's on a 10-point scale
                    elif 0 <= rating_value <= 10:
                        return rating_value / 2
                        
            return 0.0
            
        except (ValueError, AttributeError) as e:
            logger.warning(f"Rating extraction error for text '{text}': {e}")
            return 0.0
    
    @staticmethod
    def format_search_url(base_url: str, params: Dict[str, str]) -> str:
        """Format search URL with parameters and validation"""
        try:
            import urllib.parse
            
            if not base_url:
                raise ValueError("Base URL cannot be empty")
            
            query_params = []
            for key, value in params.items():
                if value is not None and str(value).strip():
                    encoded_value = urllib.parse.quote(str(value))
                    query_params.append(f"{key}={encoded_value}")
            
            if query_params:
                formatted_url = f"{base_url}?{'&'.join(query_params)}"
            else:
                formatted_url = base_url
            
            # Validate final URL
            parsed = urllib.parse.urlparse(formatted_url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL format: {formatted_url}")
            
            return formatted_url
            
        except Exception as e:
            logger.error(f"URL formatting error: {e}")
            raise TravelSearchError(f"Failed to format URL: {e}")
    
    @staticmethod
    def parse_hotel_data(content: str, platform: str) -> List[Dict]:
        """Parse hotel data from page content with enhanced error handling"""
        try:
            if not content or not content.strip():
                logger.warning(f"Empty content for platform {platform}")
                return []
            
            hotels = []
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines):
                try:
                    if 'hotel' in line.lower() or any(keyword in line.lower() for keyword in ['resort', 'inn', 'lodge']):
                        if any(keyword in line.lower() for keyword in ['rating', 'price', 'score', 'star']):
                            hotel_data = {
                                'name': TravelSearchUtils._extract_hotel_name_safe(line),
                                'price': TravelSearchUtils._extract_price_pattern_safe(line),
                                'rating': TravelSearchUtils._extract_rating_pattern_safe(line),
                                'platform': platform,
                                'raw_text': line.strip(),
                                'line_number': line_num + 1
                            }
                            
                            # Only add if we have meaningful data
                            if hotel_data['name'] or hotel_data['price']:
                                hotels.append(hotel_data)
                                
                except Exception as line_error:
                    logger.warning(f"Error parsing line {line_num + 1} for {platform}: {line_error}")
                    continue
            
            # Sort by quality score and limit results
            hotels = TravelSearchUtils._rank_hotels(hotels)
            
            logger.info(f"Successfully parsed {len(hotels)} hotels from {platform}")
            return hotels[:15]  # Return top 15 results
            
        except Exception as e:
            logger.error(f"Hotel data parsing error for {platform}: {e}")
            return []
    
    @staticmethod
    def parse_flight_data(content: str, platform: str) -> List[Dict]:
        """Parse flight data from page content with enhanced error handling"""
        try:
            if not content or not content.strip():
                logger.warning(f"Empty content for platform {platform}")
                return []
            
            flights = []
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines):
                try:
                    if any(keyword in line.lower() for keyword in ['flight', 'airline', 'departure', 'arrival']):
                        if any(indicator in line for indicator in ['$', '€', '£', '₩', 'price', 'fare', 'USD']):
                            flight_data = {
                                'airline': TravelSearchUtils._extract_airline_name_safe(line),
                                'price': TravelSearchUtils._extract_price_pattern_safe(line),
                                'duration': TravelSearchUtils._extract_duration_safe(line),
                                'departure_time': TravelSearchUtils._extract_time_pattern_safe(line),
                                'platform': platform,
                                'raw_text': line.strip(),
                                'line_number': line_num + 1
                            }
                            
                            if flight_data['airline'] or flight_data['price']:
                                flights.append(flight_data)
                                
                except Exception as line_error:
                    logger.warning(f"Error parsing flight line {line_num + 1} for {platform}: {line_error}")
                    continue
            
            flights = TravelSearchUtils._rank_flights(flights)
            
            logger.info(f"Successfully parsed {len(flights)} flights from {platform}")
            return flights[:15]
            
        except Exception as e:
            logger.error(f"Flight data parsing error for {platform}: {e}")
            return []
    
    @staticmethod
    def _extract_hotel_name_safe(text: str) -> str:
        """Safe hotel name extraction with error handling"""
        try:
            return TravelSearchUtils._extract_hotel_name(text)
        except Exception as e:
            logger.debug(f"Hotel name extraction error: {e}")
            return ""
    
    @staticmethod
    def _extract_price_pattern_safe(text: str) -> str:
        """Safe price extraction with error handling"""
        try:
            return TravelSearchUtils._extract_price_pattern(text)
        except Exception as e:
            logger.debug(f"Price extraction error: {e}")
            return ""
    
    @staticmethod
    def _extract_rating_pattern_safe(text: str) -> str:
        """Safe rating extraction with error handling"""
        try:
            return TravelSearchUtils._extract_rating_pattern(text)
        except Exception as e:
            logger.debug(f"Rating extraction error: {e}")
            return ""
    
    @staticmethod
    def _extract_airline_name_safe(text: str) -> str:
        """Safe airline name extraction with error handling"""
        try:
            return TravelSearchUtils._extract_airline_name(text)
        except Exception as e:
            logger.debug(f"Airline name extraction error: {e}")
            return ""
    
    @staticmethod
    def _extract_duration_safe(text: str) -> str:
        """Safe duration extraction with error handling"""
        try:
            return TravelSearchUtils._extract_duration(text)
        except Exception as e:
            logger.debug(f"Duration extraction error: {e}")
            return ""
    
    @staticmethod
    def _extract_time_pattern_safe(text: str) -> str:
        """Safe time extraction with error handling"""
        try:
            return TravelSearchUtils._extract_time_pattern(text)
        except Exception as e:
            logger.debug(f"Time extraction error: {e}")
            return ""
    
    @staticmethod
    def _rank_hotels(hotels: List[Dict]) -> List[Dict]:
        """Rank hotels by quality and price"""
        try:
            for hotel in hotels:
                quality_score = 0
                
                # Rating score
                rating_text = hotel.get('rating', '')
                if rating_text:
                    rating_value = TravelSearchUtils.extract_rating_from_text(rating_text)
                    if rating_value >= 4.0:
                        quality_score += 3
                    elif rating_value >= 3.5:
                        quality_score += 2
                    elif rating_value >= 3.0:
                        quality_score += 1
                
                # Price score (lower is better)
                price_text = hotel.get('price', '')
                if price_text:
                    price_value = TravelSearchUtils.extract_price_from_text(price_text)
                    if price_value != float('inf'):
                        quality_score += 1
                
                # Name quality (longer names usually indicate more detail)
                name = hotel.get('name', '')
                if len(name) > 10:
                    quality_score += 1
                
                hotel['quality_score'] = quality_score
            
            # Sort by quality score (descending)
            hotels.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            return hotels
            
        except Exception as e:
            logger.warning(f"Hotel ranking error: {e}")
            return hotels
    
    @staticmethod
    def _rank_flights(flights: List[Dict]) -> List[Dict]:
        """Rank flights by quality and price"""
        try:
            for flight in flights:
                quality_score = 0
                
                # Airline quality
                airline = flight.get('airline', '')
                major_airlines = ['Korean Air', 'Asiana', 'Delta', 'United', 'American', 
                                'Lufthansa', 'Emirates', 'Singapore Airlines']
                if any(major in airline for major in major_airlines):
                    quality_score += 2
                
                # Price availability
                if flight.get('price'):
                    quality_score += 1
                
                # Duration info
                if flight.get('duration'):
                    quality_score += 1
                
                # Time info
                if flight.get('departure_time'):
                    quality_score += 1
                
                flight['quality_score'] = quality_score
            
            flights.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            return flights
            
        except Exception as e:
            logger.warning(f"Flight ranking error: {e}")
            return flights
    
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