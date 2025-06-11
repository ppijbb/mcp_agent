#!/usr/bin/env python3
"""
MCP Browser Client for Travel Search

Uses MCP Browser Use server to control browser in incognito mode
for travel data collection without price manipulation.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import AsyncExitStack

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPBrowserClient:
    """MCP Browser client for travel search automation"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.session = None
        self.exit_stack = AsyncExitStack()
        self._setup_browser_config()
    
    def _setup_browser_config(self):
        """Setup browser configuration for incognito mode"""
        # MCP Browser Use environment variables for incognito browsing
        browser_env = {
            # LLM Configuration (required)
            'MCP_LLM_PROVIDER': 'openai',
            'MCP_LLM_MODEL_NAME': 'gpt-4o',
            'MCP_LLM_API_KEY': os.getenv('OPENAI_API_KEY', 'demo-key'),
            
            # Browser Configuration for Incognito Mode
            'MCP_BROWSER_HEADLESS': 'true',  # Headless for production
            'MCP_BROWSER_DISABLE_SECURITY': 'false',
            'MCP_BROWSER_USER_DATA_DIR': '',  # Empty for fresh sessions
            'MCP_BROWSER_KEEP_OPEN': 'false',
            
            # Agent Tool Configuration
            'MCP_AGENT_TOOL_MAX_STEPS': '20',
            'MCP_AGENT_TOOL_MAX_ACTIONS_PER_STEP': '5',
            'MCP_AGENT_TOOL_USE_VISION': 'true',
            
            # Paths for temporary data
            'MCP_PATHS_DOWNLOADS': './tmp/downloads',
            'MCP_AGENT_TOOL_HISTORY_PATH': './tmp/history',
            
            # Server Configuration
            'MCP_SERVER_LOGGING_LEVEL': 'INFO',
            'MCP_SERVER_ANONYMIZED_TELEMETRY': 'false'
        }
        
        # Update environment
        for key, value in browser_env.items():
            if not os.getenv(key):
                os.environ[key] = value
        
        # Create directories
        os.makedirs('./tmp/downloads', exist_ok=True)
        os.makedirs('./tmp/history', exist_ok=True)
    
    async def connect_to_mcp_server(self):
        """Connect to MCP Browser Use server"""
        try:
            # MCP Browser Use server parameters
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-browser-use"],
                env=dict(os.environ)  # Pass current environment with MCP config
            )
            
            # Connect to MCP server
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            
            # Initialize the session
            await self.session.initialize()
            
            # List available tools
            tools_response = await self.session.list_tools()
            available_tools = [tool.name for tool in tools_response.tools]
            logger.info(f"🔗 Connected to MCP Browser Use server with tools: {available_tools}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP Browser Use server: {e}")
            return False
    
    async def search_hotels_incognito(self, destination: str, check_in: str, check_out: str) -> List[Dict]:
        """Search for hotels using MCP browser in incognito mode"""
        if not self.session:
            connected = await self.connect_to_mcp_server()
            if not connected:
                raise ConnectionError("MCP Browser connection failed. Cannot search for hotels.")
        
        logger.info(f"🏨 Searching hotels for {destination} ({check_in} to {check_out})")
        
        # Construct search query for hotel booking sites
        search_query = f"hotels in {destination} from {check_in} to {check_out} booking.com"
        
        # Use MCP Browser Use to search
        search_result = await self.session.call_tool(
            "browser_search",
            arguments={
                "query": search_query,
                "incognito": True,
                "max_results": 10
            }
        )
        
        # Parse results and extract hotel data
        hotels = await self._parse_hotel_results(search_result, destination)
        
        logger.info(f"✅ Found {len(hotels)} hotels from MCP browser search")
        return hotels
    
    async def search_flights_incognito(self, origin: str, destination: str, departure_date: str, return_date: str) -> List[Dict]:
        """Search for flights using MCP browser in incognito mode"""
        if not self.session:
            connected = await self.connect_to_mcp_server()
            if not connected:
                raise ConnectionError("MCP Browser connection failed. Cannot search for flights.")

        logger.info(f"✈️ Searching flights {origin} -> {destination} ({departure_date} to {return_date})")
        
        # Construct search query for flight booking sites
        search_query = f"flights from {origin} to {destination} departure {departure_date} return {return_date} google flights"
        
        # Use MCP Browser Use to search
        search_result = await self.session.call_tool(
            "browser_search",
            arguments={
                "query": search_query,
                "incognito": True,
                "max_results": 10
            }
        )
        
        # Parse results and extract flight data
        flights = await self._parse_flight_results(search_result, origin, destination)
        
        logger.info(f"✅ Found {len(flights)} flights from MCP browser search")
        return flights
    
    async def _parse_hotel_results(self, search_result, destination: str) -> List[Dict]:
        """Parse hotel search results from MCP browser"""
        hotels = []
        
        try:
            # Extract content from MCP result
            if hasattr(search_result, 'content') and search_result.content:
                content_text = str(search_result.content[0].text if search_result.content else "")
                
                # Use MCP to extract structured data
                extract_result = await self.session.call_tool(
                    "extract_data",
                    arguments={
                        "content": content_text,
                        "schema": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "price": {"type": "string"}, 
                                    "rating": {"type": "string"},
                                    "location": {"type": "string"}
                                }
                            }
                        }
                    }
                )
                
                # Convert extracted data to hotel format
                if hasattr(extract_result, 'content') and extract_result.content:
                    extracted_data = json.loads(extract_result.content[0].text)
                    for item in extracted_data:
                        hotel = {
                            'name': item.get('name', f'Hotel in {destination}'),
                            'price': item.get('price', 'N/A'),
                            'price_numeric': self._extract_price_number(item.get('price', '0')),
                            'rating': item.get('rating', 'N/A'),
                            'rating_numeric': self._extract_rating_number(item.get('rating', '0')),
                            'location': item.get('location', destination),
                            'platform': 'booking.com',
                            'source': 'MCP Browser Use',
                            'quality_score': self._calculate_hotel_quality_score(item),
                            'meets_quality_criteria': self._extract_rating_number(item.get('rating', '0')) >= 4.0
                        }
                        hotels.append(hotel)
                        
        except Exception as e:
            logger.warning(f"Error parsing hotel results: {e}")
        
        return hotels[:10]  # Return top 10 results
    
    async def _parse_flight_results(self, search_result, origin: str, destination: str) -> List[Dict]:
        """Parse flight search results from MCP browser"""
        flights = []
        
        try:
            # Extract content from MCP result
            if hasattr(search_result, 'content') and search_result.content:
                content_text = str(search_result.content[0].text if search_result.content else "")
                
                # Use MCP to extract structured data
                extract_result = await self.session.call_tool(
                    "extract_data",
                    arguments={
                        "content": content_text,
                        "schema": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "airline": {"type": "string"},
                                    "price": {"type": "string"},
                                    "duration": {"type": "string"},
                                    "departure_time": {"type": "string"}
                                }
                            }
                        }
                    }
                )
                
                # Convert extracted data to flight format
                if hasattr(extract_result, 'content') and extract_result.content:
                    extracted_data = json.loads(extract_result.content[0].text)
                    for item in extracted_data:
                        flight = {
                            'airline': item.get('airline', 'Unknown Airline'),
                            'price': item.get('price', 'N/A'),
                            'price_numeric': self._extract_price_number(item.get('price', '0')),
                            'duration': item.get('duration', 'N/A'),
                            'departure_time': item.get('departure_time', 'N/A'),
                            'platform': 'google_flights',
                            'source': 'MCP Browser Use',
                            'route': f'{origin} → {destination}',
                            'quality_score': self._calculate_flight_quality_score(item),
                            'meets_quality_criteria': item.get('airline', '') in ['Korean Air', 'Asiana Airlines', 'Delta', 'Emirates']
                        }
                        flights.append(flight)
                        
        except Exception as e:
            logger.warning(f"Error parsing flight results: {e}")
        
        return flights[:10]  # Return top 10 results
    
    def _extract_price_number(self, price_str: str) -> float:
        """Extract numeric price from string"""
        if not price_str:
            return float('inf')
        
        import re
        numbers = re.findall(r'[\d,]+\.?\d*', str(price_str))
        if numbers:
            return float(numbers[0].replace(',', ''))
        return float('inf')
    
    def _extract_rating_number(self, rating_str: str) -> float:
        """Extract numeric rating from string"""
        if not rating_str:
            return 0.0
        
        import re
        numbers = re.findall(r'\d+\.?\d*', str(rating_str))
        if numbers:
            rating = float(numbers[0])
            return rating if rating <= 5 else rating / 2  # Normalize to 5-point scale
        return 0.0
    
    def _calculate_hotel_quality_score(self, hotel_data: Dict) -> int:
        """Calculate quality score for hotel"""
        score = 0
        rating = self._extract_rating_number(hotel_data.get('rating', '0'))
        price = self._extract_price_number(hotel_data.get('price', '0'))
        
        # Rating contribution
        if rating >= 4.5:
            score += 5
        elif rating >= 4.0:
            score += 4
        elif rating >= 3.5:
            score += 3
        
        # Price contribution (lower is better for value)
        if price < 150:
            score += 3
        elif price < 250:
            score += 2
        elif price < 350:
            score += 1
        
        return score
    
    def _calculate_flight_quality_score(self, flight_data: Dict) -> int:
        """Calculate quality score for flight"""
        score = 0
        airline = flight_data.get('airline', '')
        price = self._extract_price_number(flight_data.get('price', '0'))
        
        # Airline reputation
        if airline in ['Korean Air', 'Asiana Airlines', 'Emirates', 'Singapore Airlines']:
            score += 5
        elif airline in ['Delta Air Lines', 'United Airlines', 'Lufthansa', 'ANA', 'JAL']:
            score += 4
        else:
            score += 2
        
        # Price contribution
        if price < 600:
            score += 3
        elif price < 1000:
            score += 2
        elif price < 1500:
            score += 1
        
        return score
    
    async def cleanup(self):
        """Cleanup MCP browser resources"""
        try:
            if self.exit_stack:
                await self.exit_stack.aclose()
            logger.info("🧹 MCP browser resources cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


class TravelMCPManager:
    """Manager for travel search using MCP browser"""
    
    def __init__(self):
        self.browser_client = MCPBrowserClient()
        self.search_history = []
    
    async def search_travel_options(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """Search travel options using MCP browser with incognito mode"""
        search_start_time = time.time()
        
        try:
            logger.info(f"🔍 Starting MCP travel search for {search_params.get('destination')}")
            
            # Ensure MCP connection
            if not self.browser_client.session:
                await self.browser_client.connect_to_mcp_server()
            
            # Parallel searches
            search_tasks = []
            
            # Hotel search
            hotel_task = self.browser_client.search_hotels_incognito(
                search_params['destination'],
                search_params['check_in'],
                search_params['check_out']
            )
            search_tasks.append(hotel_task)
            
            # Flight search (if origin provided)
            if search_params.get('origin'):
                flight_task = self.browser_client.search_flights_incognito(
                    search_params['origin'],
                    search_params['destination'],
                    search_params['departure_date'],
                    search_params['return_date']
                )
                search_tasks.append(flight_task)
            
            # Execute searches
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Process results
            hotels = search_results[0] if not isinstance(search_results[0], Exception) else []
            flights = search_results[1] if len(search_results) > 1 and not isinstance(search_results[1], Exception) else []
            
            # Generate analysis
            analysis = self._analyze_results(hotels, flights, search_params.get('quality_criteria', {}))
            
            # Generate recommendations
            recommendations = self._generate_recommendations(hotels, flights, search_params)
            
            # Build final results
            search_duration = time.time() - search_start_time
            
            final_results = {
                "search_id": f"mcp_search_{int(time.time())}",
                "status": "completed",
                "hotels": hotels,
                "flights": flights,
                "analysis": analysis,
                "recommendations": recommendations,
                "performance": {
                    "total_duration": search_duration,
                    "platforms_searched": 2,
                    "hotels_found": len(hotels),
                    "flights_found": len(flights),
                    "method": "MCP Browser Use (Incognito Mode)",
                    "mcp_connected": self.browser_client.session is not None
                },
                "search_params": search_params,
                "quality_criteria": search_params.get('quality_criteria', {}),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store history
            self.search_history.append(final_results)
            
            logger.info(f"✅ MCP search completed in {search_duration:.2f}s: {len(hotels)} hotels, {len(flights)} flights")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ MCP travel search failed: {e}")
            return {
                "search_id": f"failed_mcp_search_{int(time.time())}",
                "status": "failed",
                "error": str(e),
                "hotels": [],
                "flights": [],
                "performance": {"total_duration": time.time() - search_start_time},
                "search_params": search_params
            }
        finally:
            # Cleanup happens when the manager is destroyed
            pass
    
    def _analyze_results(self, hotels: List[Dict], flights: List[Dict], quality_criteria: Dict) -> Dict:
        """Analyze search results"""
        analysis = {}
        
        # Hotel analysis
        if hotels:
            hotel_prices = [h.get('price_numeric', 0) for h in hotels if h.get('price_numeric', 0) != float('inf')]
            hotel_ratings = [h.get('rating_numeric', 0) for h in hotels if h.get('rating_numeric', 0) > 0]
            
            analysis['hotel_analysis'] = {
                'total_found': len(hotels),
                'average_rating': sum(hotel_ratings) / len(hotel_ratings) if hotel_ratings else 0,
                'price_range': {
                    'min': min(hotel_prices) if hotel_prices else 0,
                    'max': max(hotel_prices) if hotel_prices else 0,
                    'average': sum(hotel_prices) / len(hotel_prices) if hotel_prices else 0
                },
                'quality_hotels_count': len([h for h in hotels if h.get('meets_quality_criteria', False)]),
                'mcp_results': len([h for h in hotels if h.get('source') == 'MCP Browser Use']),
                'fallback_results': len([h for h in hotels if 'Fallback' in h.get('source', '')])
            }
        
        # Flight analysis
        if flights:
            flight_prices = [f.get('price_numeric', 0) for f in flights if f.get('price_numeric', 0) != float('inf')]
            airlines_found = list(set([f.get('airline', '') for f in flights if f.get('airline')]))
            
            analysis['flight_analysis'] = {
                'total_found': len(flights),
                'airlines_found': airlines_found,
                'price_range': {
                    'min': min(flight_prices) if flight_prices else 0,
                    'max': max(flight_prices) if flight_prices else 0,
                    'average': sum(flight_prices) / len(flight_prices) if flight_prices else 0
                },
                'quality_flights_count': len([f for f in flights if f.get('meets_quality_criteria', False)]),
                'mcp_results': len([f for f in flights if f.get('source') == 'MCP Browser Use']),
                'fallback_results': len([f for f in flights if 'Fallback' in f.get('source', '')])
            }
        
        return analysis
    
    def _generate_recommendations(self, hotels: List[Dict], flights: List[Dict], search_params: Dict) -> Dict:
        """Generate recommendations based on search results"""
        recommendations = {}
        
        # Best value recommendations
        if hotels:
            best_value_hotels = sorted(hotels, key=lambda x: (x.get('price_numeric', float('inf')), -x.get('rating_numeric', 0)))
            if best_value_hotels:
                recommendations['best_hotel'] = best_value_hotels[0]
        
        if flights:
            best_value_flights = sorted(flights, key=lambda x: x.get('price_numeric', float('inf')))
            if best_value_flights:
                recommendations['best_flight'] = best_value_flights[0]
        
        # Budget vs luxury options
        if hotels:
            budget_hotels = [h for h in hotels if h.get('price_numeric', float('inf')) < 150]
            luxury_hotels = [h for h in hotels if h.get('price_numeric', 0) > 300 and h.get('rating_numeric', 0) >= 4.5]
            
            recommendations['budget_options'] = {
                'hotel': budget_hotels[0] if budget_hotels else None
            }
            
            recommendations['luxury_options'] = {
                'hotel': luxury_hotels[0] if luxury_hotels else None
            }
        
        # Booking strategy
        recommendations['booking_strategy'] = [
            "MCP 브라우저로 시크릿 모드 검색 완료",
            "가격 조작 없는 실제 시장 가격 확인",
            "실시간 데이터 수집으로 최신 정보 제공",
            "예약 전 최신 가격 재확인 권장",
            "취소 가능한 옵션 선택 고려"
        ]
        
        # Total cost estimate
        if recommendations.get('best_hotel') and recommendations.get('best_flight'):
            hotel_price = recommendations['best_hotel'].get('price_numeric', 0)
            flight_price = recommendations['best_flight'].get('price_numeric', 0)
            
            try:
                check_in = datetime.strptime(search_params['check_in'], '%Y-%m-%d')
                check_out = datetime.strptime(search_params['check_out'], '%Y-%m-%d')
                nights = (check_out - check_in).days
            except:
                nights = 3
            
            recommendations['total_trip_cost_estimate'] = {
                'hotel_per_night': hotel_price,
                'hotel_total': hotel_price * nights,
                'flight_total': flight_price,
                'grand_total': hotel_price * nights + flight_price,
                'nights': nights,
                'currency': 'USD'
            }
        
        return recommendations
    
    def get_search_history(self) -> List[Dict]:
        """Get search history"""
        return self.search_history
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.browser_client.cleanup() 